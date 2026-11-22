#!/usr/bin/env python3
"""Self-test of verify/decode_rate.py against a toy lane whose answers are known.

    python3 verify/decode_rate_selftest.py

The toy streams a fixed number of tokens per speculative round at a fixed time per round,
after a fixed delay to the first token, and counts its rounds in the same Prometheus
counters vLLM exports. The script has to read those three numbers back.

A second toy is a two-stage pipeline with vLLM's cadence under pipeline parallelism: a
stream decodes on every second engine step, two steps are in flight at once, and each
stage works on one step at a time. Streams that share their steps then take stage 0's
time plus stage 1's per round, and streams on alternating steps take twice the slower
stage's. The script has to say which streams share their steps and what each way costs.
"""

from __future__ import annotations

import collections
import json
import os
import queue
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "decode_rate.py"


class Pipeline:
    """Two stages, one engine loop. `policy` says which steps a stream that has finished
    its prefill decodes on: "pack" puts every stream on the first one's steps, "split"
    puts every second stream on the other steps, "flip" packs in the toy's odd-numbered
    runs and splits in the even ones, and "wander" is "split" with one stream that moves
    to the other steps halfway through."""

    def __init__(self, toy: "Toy", stage0: float, stage1: float, policy: str) -> None:
        self.toy, self.stage0, self.stage1, self.policy = toy, stage0, stage1, policy
        self.lock = threading.Lock()
        self.pending: list[dict] = []
        self.streams: list[dict] = []
        self.runs = 0
        self.joined = 0
        self.wake = threading.Event()
        threading.Thread(target=self.loop, daemon=True).start()

    def join(self, left: int) -> "queue.Queue":
        """A stream whose prefill is done; `left` tokens to decode. The queue gets one
        token count per round and None at the end."""
        box: queue.Queue = queue.Queue()
        with self.lock:
            if not self.streams and not self.pending:
                self.runs += 1
                self.joined = 0
            self.pending.append({"box": box, "left": left, "total": left, "order": self.joined, "next": 0})
            self.joined += 1
        self.wake.set()
        return box

    def place(self, entry: dict, step: int) -> None:
        """The first step `entry` decodes on, under this toy's policy."""
        policy = self.policy
        if policy == "flip":
            policy = "pack" if self.runs % 2 else "split"
        first = self.streams[0]["next"] if self.streams else step
        other = policy in ("split", "wander") and entry["order"] % 2 == 1
        entry["next"] = step if (step - first) % 2 == (1 if other else 0) else step + 1

    def loop(self) -> None:
        step = 0
        free0 = free1 = 0.0
        flying: collections.deque = collections.deque()
        while True:
            with self.lock:
                idle = not self.streams and not self.pending
            if idle:
                flying.clear()  # only steps that served nobody can be left
                self.wake.wait()
                self.wake.clear()
                continue
            step += 1
            with self.lock:
                for entry in self.pending:
                    self.place(entry, step)
                    self.streams.append(entry)
                self.pending = []
                batch = [e for e in self.streams if e["next"] <= step]
                for entry in batch:
                    entry["next"] = step + 2
                    if self.policy == "wander" and entry["order"] == 0 and entry["left"] * 2 <= entry["total"] \
                            and not entry.get("moved"):
                        entry["next"], entry["moved"] = step + 3, True  # from now on, the other steps
            now = time.monotonic()
            if batch:
                end0 = max(now, free0) + self.stage0
                end1 = max(end0, free1) + self.stage1
                free0, free1 = end0, end1
                flying.append((end1, batch))
            else:
                flying.append((now, []))
            if len(flying) < 2:
                continue  # room for another step in flight: schedule it before waiting
            ready, served = flying.popleft()
            wait = ready - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            if not served:
                continue
            with self.lock, self.toy.lock:
                self.toy.steps += 1
                for entry in served:
                    count = min(self.toy.per_round, entry["left"])
                    entry["left"] -= count
                    self.toy.drafts += 1
                    self.toy.accepted += count - 1
                    for k in range(count - 1):
                        self.toy.by_position[k] += 1
                    self.toy.generated += count
                    entry["box"].put(count)
                    if entry["left"] <= 0:
                        entry["box"].put(None)
                        self.streams.remove(entry)


class Toy:
    def __init__(self, ttft: float, round_s: float, per_round: int, counters: bool = True, stranger: int = 0,
                 broken: str = "", cache: str = "usage", pipeline: tuple[float, float] | None = None,
                 policy: str = "pack", halves: bool = False, uneven: tuple[float, ...] = (),
                 requests: str = "", reader: int = 0, hiccup: tuple[int, float] | None = None,
                 alternate: tuple[int, ...] = (), by_prompt: dict[str, int] | None = None, key: str = ""):
        self.ttft, self.round_s, self.per_round, self.has_counters, self.stranger = ttft, round_s, per_round, counters, stranger
        # uneven: the share of max_tokens each stream stops at, by the order they arrive in,
        # as sampled answers of different lengths do. requests: "counted" exports the
        # prompt-token and finished-request counters. reader: prompt tokens that a client
        # that is not this run prefills during each request, generating nothing.
        self.uneven, self.requests, self.reader = uneven, requests, reader
        self.key = key  # a lane started with an API key: 401 on /v1 without "Authorization: Bearer <key>"
        self.joined = self.prompt_total = self.finished = 0
        self.prompts: list[str] = []
        # hiccup: (every, seconds): each stream's every Nth round takes that much longer.
        self.hiccup = hiccup
        # alternate: tokens per round, round after round, in place of per_round.
        # by_prompt: tokens per round for a prompt that begins with the key.
        self.alternate, self.by_prompt = alternate, by_prompt or {}
        self.by_position = [0] * 7
        # pipeline: (stage 0 seconds, stage 1 seconds) of the two-stage toy; round_s is
        # then unused. halves: every round's tokens arrive in two chunks.
        self.halves = halves
        self.steps = 0
        self.arrived: list[float] = []
        self.pipeline = Pipeline(self, *pipeline, policy) if pipeline else None
        self.broken = broken  # "refuses": HTTP 400; "stream-error": an error object as the first event
        # How this lane reports prompt tokens served from its prefix cache: in each
        # response's usage, in a counter only, or not at all. "ignores-salt" is a lane
        # whose cache serves a prompt it has seen whatever cache_salt says.
        self.cache = cache
        self.seen: set[str] = set()
        self.cached_total = 0
        self.last_body: dict = {}
        self.lock = threading.Lock()
        self.drafts = self.accepted = self.generated = 0

    def metrics(self) -> str:
        if not self.has_counters:
            return "# nothing\n"
        with self.lock:
            return (f'vllm:spec_decode_num_drafts_total{{engine="0"}} {float(self.drafts)}\n'
                    f'vllm:spec_decode_num_drafts_created{{engine="0"}} 1.7e9\n'
                    f'vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {float(5 * self.drafts)}\n'
                    f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {float(self.accepted)}\n'
                    + "".join(f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="{k}"}} '
                              f"{float(v)}\n" for k, v in enumerate(self.by_position))
                    + f'vllm:spec_decode_num_accepted_tokens_per_pos_created{{engine="0",position="0"}} 1.7e9\n'
                    f'vllm:generation_tokens_total{{engine="0"}} {float(self.generated)}\n'
                    + ("" if self.pipeline is None else
                       f'vllm:iteration_tokens_total_bucket{{engine="0",le="+Inf"}} {float(self.steps)}\n'
                       f'vllm:iteration_tokens_total_count{{engine="0"}} {float(self.steps)}\n'
                       f'vllm:iteration_tokens_total_sum{{engine="0"}} {float(self.generated)}\n')
                    + ("" if self.cache == "none" else
                       f'vllm:prompt_tokens_cached_total{{engine="0"}} {float(self.cached_total)}\n'
                       f'vllm:prompt_tokens_cached_by_source_total{{engine="0",source="x"}} 1e9\n')
                    + ("" if not self.requests else
                       f'vllm:prompt_tokens_total{{engine="0"}} {float(self.prompt_total)}\n'
                       f'vllm:prompt_tokens_by_source_total{{engine="0",source="x"}} 1e9\n'
                       f'vllm:request_success_total{{engine="0",finished_reason="length"}} {float(self.finished)}\n'
                       f'vllm:request_success_total{{engine="0",finished_reason="stop"}} 0.0\n'))


def serve(toy: Toy):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args) -> None:
            pass

        def setup(self) -> None:
            super().setup()
            # A chunk goes out when it is written, as an asyncio server's does.
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def refused(self) -> bool:
            if not toy.key or not self.path.startswith("/v1") or self.headers.get("Authorization") == "Bearer " + toy.key:
                return False
            payload = json.dumps({"error": "Unauthorized"}).encode()
            self.send_response(401)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return True

        def do_GET(self) -> None:  # noqa: N802
            if self.refused():
                return
            payload = (json.dumps({"data": [{"id": "toy"}]}) if self.path == "/v1/models" else toy.metrics()).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.refused():
                return
            total = asked = int(body["max_tokens"])
            with toy.lock:
                toy.arrived.append(time.monotonic())
                toy.prompts.append(body["messages"][0]["content"])
                if toy.uneven:
                    total = max(toy.per_round, int(asked * toy.uneven[toy.joined % len(toy.uneven)]))
                toy.joined += 1
            if toy.broken == "refuses":
                payload = json.dumps({"error": {"message": "steering_scale must be in [0, 1]", "code": 400}}).encode()
                self.send_response(400)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def event(payload) -> None:
                self.wfile.write(b"data: " + (payload if isinstance(payload, bytes) else json.dumps(payload).encode()) + b"\n\n")
                self.wfile.flush()

            time.sleep(toy.ttft)
            if toy.broken == "stream-error":
                event({"error": {"message": "the engine died mid-request", "code": 500}})
                event(b"[DONE]")
                return
            sent = 0
            if toy.pipeline is not None:
                # The prefill's own token, then one chunk per engine step this stream is on.
                with toy.lock:
                    toy.generated += 1
                    toy.steps += 1
                sent = 1
                event({"choices": [{"delta": {"reasoning_content": "x "}, "finish_reason": None}]})
                box = toy.pipeline.join(total - 1)
                while (count := box.get()) is not None:
                    sent += count
                    if toy.halves and count > 1:
                        event({"choices": [{"delta": {"reasoning_content": "x "}, "finish_reason": None}]})
                        time.sleep(0.001)
                        count -= 1
                    event({"choices": [{"delta": {"reasoning_content": "x " * count}, "finish_reason": None}]})
            rounds = 0
            content = body["messages"][0]["content"]
            own = next((v for key, v in toy.by_prompt.items() if content.startswith(key)), toy.per_round)
            while sent < total:
                count = min(toy.alternate[rounds % len(toy.alternate)] if toy.alternate else own, total - sent)
                if sent:
                    rounds += 1
                    time.sleep(toy.round_s + (toy.hiccup[1] if toy.hiccup and rounds % toy.hiccup[0] == 0 else 0.0))
                with toy.lock:
                    toy.drafts += 1
                    toy.accepted += count - 1
                    for k in range(count - 1):
                        toy.by_position[k] += 1
                    toy.generated += count + toy.stranger
                sent += count
                event({"choices": [{"delta": {"reasoning_content": "x " * count}, "finish_reason": None}]})
            event({"choices": [{"delta": {}, "finish_reason": "length" if total == asked else "stop"}]})
            words = len(body["messages"][0]["content"].split())
            usage = {"completion_tokens": sent, "prompt_tokens": words}
            with toy.lock:
                toy.last_body = body
                toy.prompt_total += words + toy.reader
                toy.finished += 1
                key = body["messages"][0]["content"] + ("" if toy.cache == "ignores-salt" else body.get("cache_salt", ""))
                cached = words if key in toy.seen else 0
                toy.seen.add(key)
                toy.cached_total += cached
            if toy.cache in ("usage", "ignores-salt"):
                usage["prompt_tokens_details"] = {"cached_tokens": cached}
            event({"choices": [], "usage": usage})
            event(b"[DONE]")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def run(toy: Toy, *extra: str) -> tuple[int, dict, str]:
    server, url = serve(toy)
    try:
        done = subprocess.run([sys.executable, "-B", str(SCRIPT), "--url", url, "--json", "-",
                               *(() if "--repeat" in extra else ("--repeat", "3")),
                               *(() if "--tokens" in extra else ("--tokens", "240")), *extra],
                              capture_output=True, text=True, timeout=300,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    finally:
        server.shutdown()
    start = done.stdout.find('{\n  "url"')
    data = json.loads(done.stdout[start:]) if start >= 0 else {}
    return done.returncode, data, done.stdout + done.stderr


def main() -> int:
    failures = []

    def check(name: str, good: bool, detail: str) -> None:
        print(f"{'ok  ' if good else 'FAIL'}  {name:<46} {detail}")
        if not good:
            failures.append(name)

    def middle(data: dict, index: int) -> dict:
        runs = data["runs"][index]
        return sorted(runs, key=lambda r: r["tok_s_aggregate"])[len(runs) // 2]

    # 4 tokens per 10 ms round after 0.20 s: 400 tok/s per stream, whatever the first token cost.
    status, data, text = run(Toy(0.20, 0.010, 4), "--streams", "1,4")
    one, four = (middle(data, 0), middle(data, 1)) if data else ({}, {})
    check("one stream: tokens per second after the first", status == 0 and 330 <= one.get("tok_s_per_stream", 0) <= 410,
          f"{one.get('tok_s_per_stream', 0):.0f} tok/s (the toy makes 400; 240 tokens in 0.79 s would read 304)")
    check("one stream: time to first token", 0.19 <= one.get("ttft_s", 0) <= 0.30, f"{one.get('ttft_s', 0):.3f} s (0.20)")
    check("one stream: tokens per round", abs(one.get("tokens_per_round", 0) - 4.0) < 0.01,
          f"{one.get('tokens_per_round', 0):.2f} (4)")
    check("one stream: milliseconds per round", 9.5 <= one.get("ms_per_round", 0) <= 12.5,
          f"{one.get('ms_per_round', 0):.2f} ms (10, over 59 of 60 rounds)")
    check("four streams: aggregate and per-stream", 1250 <= four.get("tok_s_aggregate", 0) <= 1640
          and 330 <= four.get("tok_s_per_stream", 0) <= 410,
          f"{four.get('tok_s_aggregate', 0):.0f} aggregate, {four.get('tok_s_per_stream', 0):.0f} per stream (1600, 400)")
    check("four streams: per round is per stream", abs(four.get("tokens_per_round", 0) - 4.0) < 0.01
          and 9.5 <= four.get("ms_per_round", 0) <= 13.0,
          f"{four.get('tokens_per_round', 0):.2f} tokens, {four.get('ms_per_round', 0):.2f} ms")
    # Nothing accepted: the same round time yields a quarter of the speed.
    status, data, text = run(Toy(0.05, 0.010, 1), "--streams", "1")
    one = middle(data, 0) if data else {}
    check("no draft accepted: 1 token per round", abs(one.get("tokens_per_round", 0) - 1.0) < 0.01
          and 80 <= one.get("tok_s_per_stream", 0) <= 102,
          f"{one.get('tokens_per_round', 0):.2f} tokens per round, {one.get('tok_s_per_stream', 0):.0f} tok/s (1, 100)")
    status, data, text = run(Toy(0.05, 0.002, 4, counters=False), "--streams", "1")
    check("a lane without the counters", status == 0 and "per-round numbers unmeasured" in text, "says so, still reports tok/s")
    status, data, text = run(Toy(0.05, 0.002, 4, stranger=3), "--streams", "1")
    check("other traffic on the lane", status == 0 and "other traffic in 3 of 3 runs" in text
          and "more generated tokens than this run received" in text, "says so")
    # A client that prefills a long prompt and says little moves no generated tokens.
    status, data, text = run(Toy(0.05, 0.002, 4, requests="counted", reader=900), "--streams", "2")
    check("other traffic that only prefills", status == 0 and "900 more prompt tokens than this run sent" not in text
          and "1800 more prompt tokens than this run sent" in text and "more generated tokens" not in text,
          "said from the prompt-token counter")
    status, data, text = run(Toy(0.05, 0.002, 4, requests="counted"), "--streams", "2")
    check("a quiet lane with those counters", status == 0 and "other traffic" not in text, "nothing is said")
    # Streams that end at different times, as sampled answers do: 4 tokens per 10 ms round
    # each, stopping at a quarter, a half, three quarters and all of 480 tokens. These two
    # checks compare two timings of one run: a machine that stalls for a moment gets up to
    # three tries.
    for attempt in range(3):
        status, data, text = run(Toy(0.05, 0.010, 4, uneven=(0.25, 0.5, 0.75, 1.0)), "--streams", "4", "--tokens", "480")
        row = middle(data, 0) if data else {}
        line = re.search(r"where all streams decode: ([0-9.]+) ms/round, ([0-9.]+) tok/s per stream, ([0-9.]+) aggregate", text)
        found = [float(v) for v in line.groups()] if line else [0.0, 0.0, 0.0]
        good = (status == 0 and 14.5 <= row.get("ms_per_round", 0) <= 19.0
                and 9.5 <= found[0] <= 12.5 and 330 <= found[1] <= 410 and 1250 <= found[2] <= 1640
                and "fewer streams were decoding" in text and "3 stream(s) stopped before max_tokens" in text)
        if good:
            break
    check("streams that end at different times", good,
          f"row {row.get('ms_per_round', 0):.2f} ms/round over the whole run (16); where all four decode "
          f"{found[0]:.2f} ms, {found[1]:.0f} per stream, {found[2]:.0f} aggregate (10, 400, 1600)")
    for attempt in range(3):
        status, data, text = run(Toy(0.05, 0.010, 4), "--streams", "4", "--tokens", "480")
        line = re.search(r"where all streams decode: ([0-9.]+) ms/round", text)
        good = (status == 0 and line is not None and 9.5 <= float(line.group(1)) <= 12.5
                and "fewer streams were decoding" not in text)
        if good:
            break
    check("...and streams that end together", good, "the same line, and no remark on the row")
    # A receipt read back prints the rows it was written with, and sends nothing.
    with tempfile.TemporaryDirectory() as work:
        saved = str(Path(work) / "receipt.json")
        toy = Toy(0.05, 0.010, 4, uneven=(0.25, 0.5, 0.75, 1.0))
        status, _, live = run(toy, "--streams", "1,4", "--tokens", "480", "--json", saved)
        requests = len(toy.arrived)
        again = subprocess.run([sys.executable, "-B", str(SCRIPT), "--read", saved], capture_output=True, text=True, timeout=60)
        # Two rows, and under them: the rate where all four streams decode, and for each
        # row the drafts accepted by position and the chunk gaps.
        rows = [line for line in live.splitlines() if line[:8].strip().isdigit() or line.startswith(" " * 8)]
        check("a receipt read back", status == 0 and again.returncode == 0 and len(rows) == 7
              and all(row in again.stdout for row in rows) and again.stdout.startswith("read back from")
              and len(toy.arrived) == requests, "the same rows and lines, and no request")
        # ...one written before the prompt-token and finished-request counters were read.
        old = json.loads(Path(saved).read_text())
        for runs in old["runs"]:
            for one_run in runs:
                one_run.pop("other_traffic_by", None)
                one_run["other_traffic"] = True
        Path(saved).write_text(json.dumps(old))
        again = subprocess.run([sys.executable, "-B", str(SCRIPT), "--read", saved], capture_output=True, text=True, timeout=60)
        check("...and a receipt of an earlier revision", again.returncode == 0
              and "more generated tokens than this run received" in again.stdout, "its other-traffic mark is kept")
        Path(saved).write_text("[]")
        again = subprocess.run([sys.executable, "-B", str(SCRIPT), "--read", saved], capture_output=True, text=True, timeout=60)
        check("...and a file that is no receipt", again.returncode == 2 and "is not a receipt" in again.stderr, "exit 2")
    again = subprocess.run([sys.executable, "-B", str(SCRIPT)], capture_output=True, text=True, timeout=60)
    check("no lane and no receipt named", again.returncode == 2 and "--url is needed" in again.stderr, "exit 2")
    # --vary: every stream of a run its own prompt, the first one the prompt as it is.
    toy = Toy(0.05, 0.002, 4)
    status, data, text = run(toy, "--streams", "4", "--train-prompt", "--tokens", "64", "--vary", "--repeat", "2")
    sent = toy.prompts[1:]  # after the request that settles the lane
    groups = [set(sent[0:4]), set(sent[4:8])]
    base = toy.prompts[0]
    check("--vary: a prompt per stream", status == 0 and len(sent) == 8 and all(len(g) == 4 and base in g for g in groups)
          and groups[0] == groups[1] and "at 60 mph" in base and any("at 62 mph" in q and "at 93 mph" in q for q in groups[0])
          and "each stream has its own prompt" in text and data.get("vary") is True,
          "four prompts a run, the README's among them, the same four in each run")
    toy = Toy(0.05, 0.002, 4)
    status, data, text = run(toy, "--streams", "3", "--tokens", "64", "--vary", "--repeat", "1")
    check("...of any other prompt too", status == 0 and len(set(toy.prompts[1:])) == 3 and toy.prompts[0] in toy.prompts[1:]
          and any(q.startswith("(Request 2.) ") for q in toy.prompts[1:]), "a numbered line in front")
    toy = Toy(0.05, 0.002, 4)
    status, data, text = run(toy, "--streams", "4", "--tokens", "64", "--repeat", "1")
    check("...and without it one prompt for all", status == 0 and len(set(toy.prompts)) == 1
          and "each stream has its own prompt" not in text and data.get("vary") is False, "as before")
    # --prompts-file: stream k sends line k.
    shipped = SCRIPT.parent / "diverse_prompts.txt"
    listed = [line.strip() for line in shipped.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    check("the diverse prompts that ship with it", len(listed) == 32 and len(set(listed)) == 32
          and listed[0].startswith("A train leaves at 60 mph") and min(len(q) for q in listed) > 50,
          f"{len(listed)} prompts, all different, the README's first")
    toy = Toy(0.05, 0.002, 4)
    status, data, text = run(toy, "--streams", "4", "--tokens", "64", "--repeat", "2", "--prompts-file", str(shipped))
    sent = toy.prompts[1:]
    check("--prompts-file: stream k sends line k", status == 0 and toy.prompts[0] == listed[0] and len(sent) == 8
          and set(sent[0:4]) == set(listed[:4]) == set(sent[4:8]) and "32 prompts of" in text and "one per stream" in text
          and data.get("prompts_file") == str(shipped), "the first four lines in each run")
    with tempfile.TemporaryDirectory() as work:
        few = Path(work) / "few.txt"
        few.write_text("# a comment\nfirst prompt\n\nsecond prompt\nthird prompt\n")
        toy = Toy(0.05, 0.002, 4)
        status, data, text = run(toy, "--streams", "5", "--tokens", "64", "--repeat", "1", "--prompts-file", str(few))
        counts = collections.Counter(toy.prompts[1:])
        check("...more streams than prompts", status == 0 and sorted(counts.values()) == [1, 2, 2]
              and "streams past the first 3 repeat them" in text, "they repeat, and it is said")
        few.write_text("only one\n")
        status, data, text = run(Toy(0.05, 0.002, 4), "--streams", "2", "--prompts-file", str(few))
        check("...a file with one prompt is refused", status == 2 and "needs at least two prompts" in text, "exit 2")
    status, data, text = run(Toy(0.05, 0.002, 4), "--streams", "2", "--vary", "--prompts-file", str(shipped))
    check("...and not together with --vary", status == 2 and "a set of prompts of its own" in text, "exit 2")
    # Chunk gaps: a lane on which every eighth round takes 15 ms longer keeps its median
    # and shows the long rounds in the upper percentiles.
    gap_line = re.compile(r"chunk gaps(?: where all streams decode)?, the median run: mean ([0-9.]+) ms, half under "
                          r"([0-9.]+), 90% under ([0-9.]+), 99% under ([0-9.]+), longest ([0-9.]+); ([0-9.]+)% longer")
    for attempt in range(3):
        status, data, text = run(Toy(0.05, 0.010, 4, hiccup=(8, 0.015)), "--streams", "1,2", "--tokens", "480")
        found = [[float(v) for v in m.groups()] for m in gap_line.finditer(text)]
        good = (status == 0 and len(found) == 2 and "chunk gaps where all streams decode" in text
                and all(11.2 <= f[0] <= 14.0 and 9.5 <= f[1] <= 12.5 and f[3] >= 23.0 and 9.0 <= f[5] <= 17.0 for f in found))
        if good:
            break
    check("chunk gaps: some rounds much longer", good,
          "; ".join(f"mean {f[0]:.1f} ms, half under {f[1]:.1f}, 99% under {f[3]:.1f}, {f[5]:.1f}% long" for f in found)
          + " (11.9, 10, 25, 12.5)")
    for attempt in range(3):
        status, data, text = run(Toy(0.05, 0.010, 4), "--streams", "1", "--tokens", "480")
        found = [[float(v) for v in m.groups()] for m in gap_line.finditer(text)]
        good = (status == 0 and len(found) == 1 and 9.5 <= found[0][1] <= 12.5 and found[0][5] <= 3.0
                and abs(found[0][0] - found[0][1]) <= 0.5)
        if good:
            break
    check("...and a lane whose rounds are alike", good,
          "; ".join(f"mean {f[0]:.1f} ms, half under {f[1]:.1f}, {f[5]:.1f}% long" for f in found) + " (10, 10, 0)")
    # Drafts accepted by position: rounds of 4 and 2 tokens in turn accept the first draft
    # always and the second and third every other round.
    status, data, text = run(Toy(0.05, 0.002, 4, alternate=(4, 2)), "--streams", "1", "--tokens", "360")
    one = middle(data, 0) if data else {}
    shares = one.get("accepted_by_position") or []
    check("drafts accepted by position", status == 0 and len(shares) == 7
          and all(abs(a - b) < 0.02 for a, b in zip(shares, (1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0)))
          and abs(one.get("tokens_per_round", 0) - 3.0) < 0.03
          and re.search(r"tokens a round at a draft of 1: 2\.0\d, 2: 2\.5\d, 3: 3\.0\d, 4: 3\.0\d", text) is not None,
          " ".join(f"{v:.2f}" for v in shares) + "; a draft of 1 would yield 2, of 2 2.5, of 3 the lane's 3")
    status, data, text = run(Toy(0.05, 0.002, 4, counters=False), "--streams", "1")
    check("...a lane without those counters", status == 0 and "drafts accepted by position" not in text, "the line is left out")
    # --each: every prompt by itself, and what its text yields a round.
    with tempfile.TemporaryDirectory() as work:
        two = Path(work) / "two.txt"
        two.write_text("alpha: a text that drafts well\nbeta: a text that drafts badly\ngamma: never sent\n")
        toy = Toy(0.05, 0.002, 4, by_prompt={"alpha": 4, "beta": 2})
        saved = str(Path(work) / "each.json")
        status, _, text = run(toy, "--prompts-file", str(two), "--each", "2", "--tokens", "240", "--repeat", "2", "--json", saved)
        rows = [line.split() for line in text.splitlines() if line[:8].strip().isdigit()]
        sent = toy.prompts[1:]
        check("--each: a row per prompt, one stream each", status == 0 and len(rows) == 2 and len(sent) == 4
              and sent[0] == sent[1] != sent[2] == sent[3] and not any(q.startswith("gamma") for q in toy.prompts)
              and abs(float(rows[0][2]) - 4.0) < 0.05 and abs(float(rows[1][2]) - 2.0) < 0.05
              and abs(float(rows[0][4]) - 4.0) < 0.05 and abs(float(rows[1][4]) - 2.0) < 0.05
              and "each by itself at one stream" in text,
              f"tokens a round {rows[0][2] if rows else '-'} and {rows[1][2] if len(rows) > 1 else '-'} (4, 2)")
        again = subprocess.run([sys.executable, "-B", str(SCRIPT), "--read", saved], capture_output=True, text=True, timeout=60)
        check("...and its receipt reads back", again.returncode == 0 and "each by itself" in again.stdout
              and len([line for line in again.stdout.splitlines() if line[:8].strip().isdigit()]) == 2, "two rows")
    status, data, text = run(Toy(0.05, 0.002, 4), "--streams", "1", "--each", "2")
    check("...and not without a file of prompts", status == 2 and "--each N needs --prompts-file" in text, "exit 2")
    status, data, text = run(Toy(0.05, 0.002, 4), "--streams", "1", "--context-words", "3000")
    words = middle(data, 0).get("prompt_tokens") or 0 if data else 0
    check("a long context in front of the prompt", status == 0 and 3000 <= words <= 3200, f"prompt of {words} words reached the lane")
    check("...and its prefill is reported cold", "prefill cold in every run" in text, "each request has its own cache_salt")
    # Tokens per round depend on the text, so two lanes are compared on one prompt.
    toy = Toy(0.05, 0.002, 4)
    status, data, text = run(toy, "--streams", "1", "--train-prompt", "--tokens", "64")
    body = toy.last_body
    check("the README's train prompt, as the README sent it",
          status == 0 and body.get("messages", [{}])[0].get("content", "").startswith("A train leaves at 60 mph.")
          and body.get("top_p") == 0.95 and body.get("max_tokens") == 64 and body.get("temperature") == 0
          and "train prompt" in text, f"top_p {body.get('top_p')}, max_tokens {body.get('max_tokens')}")
    # What real clients send is not temperature 0: the tool can send theirs, or nothing.
    toy = Toy(0.05, 0.002, 4)
    status, _, text = run(toy, "--streams", "1", "--train-prompt", "--tokens", "64", "--temperature", "0.7")
    body = toy.last_body
    check("a temperature of the caller's choosing", status == 0 and body.get("temperature") == 0.7
          and body.get("top_p") == 0.95 and "temperature 0.7," in text and "sampled requests" in text,
          f"sent temperature {body.get('temperature')}, top_p {body.get('top_p')}")
    toy = Toy(0.05, 0.002, 4)
    status, sampled, text = run(toy, "--streams", "1", "--train-prompt", "--tokens", "64", "--temperature", "server")
    body = toy.last_body
    check("the lane's own sampling defaults", status == 0 and "temperature" not in body and "top_p" not in body
          and "temperature not sent (the lane decides)" in text and sampled.get("temperature") == "server",
          "neither temperature nor top_p in the request")
    status, _, text = run(Toy(0.05, 0.002, 4), "--streams", "1", "--temperature", "warm")
    check("a temperature that is no number is refused", status == 2 and "a number or the word server" in text, "exit 2")
    one = middle(data, 0) if data else {}
    check("tokens over wall time is reported beside it", 0 < one.get("tok_s_with_ttft", 0) < one.get("tok_s_per_stream", 0),
          f"{one.get('tok_s_with_ttft', 0):.0f} with the first token's latency, {one.get('tok_s_per_stream', 0):.0f} after it")
    status, data, text = run(Toy(0.05, 0.002, 4, cache="ignores-salt"), "--streams", "1")
    check("a prefill that was not cold is said so", status == 0 and "prefill NOT cold" in text, "from the responses' usage")
    status, data, text = run(Toy(0.05, 0.002, 4, cache="counter"), "--streams", "1")
    check("cold prefill, lane with the counter only", status == 0 and "prefill cold in every run" in text,
          "from vllm:prompt_tokens_cached")
    status, data, text = run(Toy(0.05, 0.002, 4, cache="none"), "--streams", "1")
    check("a lane that reports no cached tokens", status == 0 and "does not report cached prompt tokens" in text, "says so")
    status, data, text = run(Toy(0.01, 0.002, 4, broken="refuses"), "--streams", "1")
    check("a lane that refuses the request", status == 2 and "HTTP 400" in text and "must be in [0, 1]" in text,
          "exit 2, with the lane's own message")
    status, data, text = run(Toy(0.01, 0.002, 4, broken="stream-error"), "--streams", "1")
    check("an error inside the stream", status == 2 and "the engine died mid-request" in text and "Traceback" not in text,
          "exit 2, with the lane's own message")

    # The two-stage toy: stage 0 takes 8 ms and stage 1 takes 14. Streams that share their
    # steps take 22 ms a round; streams on alternating steps take 28, twice the slower stage.
    stages = (0.008, 0.014)

    def splits(data: dict, index: int) -> list:
        return [r.get("split") for r in data["runs"][index]] if data else []

    def mostly(found: list, want: str) -> bool:
        """Every run that gives a split gives this one, and at most one run gives none (a
        busy machine can smear one run's chunk times)."""
        return found.count(want) >= len(found) - 1 and all(f in (want, None) for f in found)

    status, data, text = run(Toy(0.03, 0, 4, pipeline=stages, policy="pack"), "--streams", "1,4")
    one, four = (middle(data, 0), middle(data, 1)) if data else ({}, {})
    check("pipeline, one stream", status == 0 and mostly(splits(data, 0), "1")
          and 20.5 <= one.get("window", {}).get("period_ms", 0) <= 25.5,
          f"split {one.get('split')}, {one.get('window', {}).get('period_ms', 0):.1f} ms a round (22: both stages, one after the other)")
    check("pipeline, four streams on the same steps", mostly(splits(data, 1), "4+0")
          and 3.4 <= four.get("streams_per_step", 0) <= 4.1 and 20.5 <= four.get("window", {}).get("period_ms", 0) <= 25.5,
          f"split {four.get('split')}, {four.get('streams_per_step', 0):.2f} streams a step, "
          f"{four.get('window', {}).get('period_ms', 0):.1f} ms a round (4+0, 4, 22)")
    status, data, text = run(Toy(0.03, 0, 4, pipeline=stages, policy="split"), "--streams", "4,3")
    four, three = (middle(data, 0), middle(data, 1)) if data else ({}, {})
    check("pipeline, four streams on alternating steps", status == 0 and mostly(splits(data, 0), "2+2")
          and 1.7 <= four.get("streams_per_step", 0) <= 2.2 and 26.5 <= four.get("window", {}).get("period_ms", 0) <= 31.5,
          f"split {four.get('split')}, {four.get('streams_per_step', 0):.2f} streams a step, "
          f"{four.get('window', {}).get('period_ms', 0):.1f} ms a round (2+2, 2, 28: twice the slower stage)")
    check("pipeline, three streams, two and one", mostly(splits(data, 1), "2+1") and "do not agree" not in text,
          f"split {three.get('split')}, {three.get('streams_per_step', 0):.2f} streams a step (2+1, 1.5)")
    # Runs that land differently are reported each by itself: that is the comparison.
    status, data, text = run(Toy(0.03, 0, 4, pipeline=stages, policy="flip"), "--streams", "4", "--repeat", "4")
    lines = {m.group(1): (int(m.group(2)), float(m.group(3)), float(m.group(4)))
             for m in re.finditer(r"split (\S+): (\d) of 4 runs; ([0-9.]+) ms/round, ([0-9.]+) tok/s per stream", text)}
    packed, alternating = lines.get("4+0", (0, 0.0, 0.0)), lines.get("2+2", (0, 0.0, 0.0))
    check("two splits among the runs: one line each", status == 0 and packed[0] == 2 and alternating[0] == 2
          and 20.5 <= packed[1] <= 25.5 and 26.5 <= alternating[1] <= 31.5 and packed[2] > 1.15 * alternating[2],
          f"4+0: {packed[1]:.1f} ms, {packed[2]:.0f} tok/s; 2+2: {alternating[1]:.1f} ms, {alternating[2]:.0f} tok/s "
          "(22 and 28 ms; 182 and 143 tok/s)")
    toy = Toy(0.03, 0, 4, pipeline=stages, policy="pack")
    status, data, text = run(toy, "--streams", "3", "--repeat", "1", "--stagger-ms", "150")
    gaps = [b - a for a, b in zip(toy.arrived[-3:], toy.arrived[-2:])]
    check("--stagger-ms starts the streams apart", status == 0 and len(gaps) == 2 and all(0.110 <= g <= 0.230 for g in gaps)
          and data.get("stagger_ms") == 150 and "starts 150 ms after the one before" in text
          and ("split 3+0: 1 of 1 runs" in text or "split not told in 1 of 1 runs" in text),
          "requests reached the lane " + ", ".join(f"{1000 * g:.0f}" for g in gaps) + " ms apart (150)")
    status, data, text = run(Toy(0.03, 0, 4, pipeline=stages, policy="split", halves=True), "--streams", "2")
    check("a lane that sends two chunks a round", status == 0 and "2.00 chunks arrived per engine round" in text
          and splits(data, 0) == [None] * 3, "no split is given, and the reason is")
    status, data, text = run(Toy(0.03, 0, 4, pipeline=stages, policy="wander"), "--streams", "2")
    check("a stream that changes steps during the run", status == 0 and splits(data, 0) == [None] * 3
          and "changed steps during the run" in text, "no split is given, and the reason is")
    # A lane started with an API key, and fields every request has to carry.
    status, data, text = run(Toy(0.03, 0.02, 4, key="s3cret"), "--streams", "1", "--repeat", "1")
    check("a lane that asks for an API key, none given", status == 1 and "HTTP 401" in text and "--api-key-file" in text,
          "it stops and names the option")
    keyfile = Path(tempfile.mkdtemp(prefix="decode-rate-key-")) / "api-key"
    keyfile.write_text("s3cret\n")
    toy = Toy(0.03, 0.02, 4, key="s3cret")
    status, data, text = run(toy, "--streams", "1", "--repeat", "1", "--api-key-file", str(keyfile),
                             "--extra", '{"chat_template_kwargs": {"thinking": false}, "vllm_xargs": {"a": 1}}',
                             "--scale", "0.5")
    body = toy.last_body
    check("...the key from a file, and --extra in every request",
          status == 0 and body.get("chat_template_kwargs") == {"thinking": False}
          and body.get("vllm_xargs") == {"a": 1, "steering_scale": 0.5} and data.get("api_key") is True
          and data.get("extra", {}).get("chat_template_kwargs") == {"thinking": False} and "s3cret" not in text
          and "every request also carries" in text and "sent with an API key" in text,
          "the body carries both, the receipt says so, the key is not printed")
    status, data, text = run(Toy(0.03, 0.02, 4, key="s3cret"), "--streams", "1", "--repeat", "1", "--api-key", "wrong")
    check("a key the lane refuses", status == 1 and "HTTP 401" in text and "the one given was refused" in text,
          "it says the key was refused")
    status, data, text = run(Toy(0.03, 0.02, 4), "--streams", "1", "--repeat", "1", "--extra", '{"max_tokens": 5}')
    check("--extra may not replace what the script sets", status == 2 and "--extra may not set max_tokens" in text,
          "refused before anything is sent")
    print(f"\nDECODE-RATE-SELFTEST {'PASS' if not failures else 'FAIL'}: {len(failures)} check(s) failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
