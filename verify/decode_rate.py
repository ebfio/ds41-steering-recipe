#!/usr/bin/env python3
"""Decode speed of a lane, taken apart: time to first token, tokens per second after it,
and what one speculative round yields and costs.

Standard library only; nothing is restarted and nothing but requests is sent:

    python3 verify/decode_rate.py --url http://<rank0>:8001 --streams 1,4 --tokens 1024

For each concurrency it streams that many chat completions of the same prompt (each with
its own cache_salt, so none is served from the prefix cache) and reports

    ttft        seconds to the first streamed token (median over streams)
    tok/s       per stream: (tokens - 1) / (time from the first token to the last);
                aggregate: all tokens after the first / the time they took together.
                The first chunk can carry a whole round's tokens, so both read high by
                at most one round in --tokens (under 1 % at 1024).
    per round   from the engine's own counters over the run: tokens per speculative
                round (1 = nothing accepted beyond the verified token) and milliseconds
                per round. A round is one draft-and-verify step of one request.

tok/s falls when a round gets slower or when it yields fewer tokens; this tells which.
"128 tokens in 0.70 s" does not: it has the first token's latency in it and no rounds.
The round counter is vllm:spec_decode_num_drafts, which counts a request's steps that
carried drafts. Where the engine skips drafting on some steps, their tokens still count,
so both per-round numbers then read high; vllm:spec_decode_num_draft_tokens / drafts in
the JSON says how many drafts a round carried.

Tokens per round depend on the text: a lane drafts a worked arithmetic answer far better
than an essay. Compare two lanes, or a lane with a published number, on the same prompt
only. --train-prompt sends the prompt this repository's README numbers are on (the one
in verify/texts_prod.py, with its top_p 0.95; use --tokens 512 as the README did), and
--prompt-file FILE sends the text of FILE. Without either, the prompt is the essay below.

--temperature T sends that temperature instead of 0, and --temperature server sends
neither temperature nor top_p, so the lane's own generation defaults apply: what a client
that sets nothing gets. Every published number of this repository is at temperature 0. A
sampled run takes another path through the drafter and the verifier, yields another
number of tokens per round, and its text differs from run to run: read the median and
the spread over --repeat, and compare two lanes at the same setting only.

--scale S sends steering_scale S with every request (a lane booted with steering).
--context-words N puts a ledger of about N words in front of the prompt, so that ttft
measures a cold prefill of that length (about 1.5 tokens a word; the script prints the
prompt's token count from the lane's own usage). Whether a run's prefill was cold is
reported, not assumed: from the usage of each response where the lane gives it
(prompt_tokens_details.cached_tokens), else from vllm:prompt_tokens_cached.
Run it on an otherwise idle lane: the counters are the engine's, so other traffic
during the run lands in the "per round" numbers. The script says so when the lane counted
more generated tokens than it received, more prompt tokens than it sent, or more finished
requests than it made (vllm:generation_tokens, vllm:prompt_tokens, vllm:request_success).

Streams that do not decode side by side for the whole run. The row's aggregate and its
ms/round are taken over the whole run: from the first stream's first token to the last
stream's last. When streams end at different times, as sampled answers of different
lengths do, the tail of the run has fewer streams in it, and both numbers then read
slower than the lane is at that concurrency. So below each row of two or more streams
the script prints the rate "where all streams decode": the round time, tokens per second
per stream and their sum, from the chunks that arrived while every stream was decoding.
Compare runs whose streams end at different times on that line and on the per-stream
column, not on the row's aggregate or ms/round.

--read FILE prints the table of a receipt that --json wrote, with these lines, and sends
nothing: a table from an earlier revision can be read again without loading the lane.

--vary gives each stream its own prompt: the train prompt with other speeds for every
stream but the first, any other prompt with a line that numbers the request. Without it
every stream of a run sends the same prompt, and at temperature 0 streams on the same
engine steps then write the same text, accept the same drafts in the same rounds and
end together. No client traffic is like that. Whether a lane is as fast when its streams
differ is a measurement: the same command with and without --vary.

--prompts-file FILE goes further: one prompt per line, stream k sends line k. With
verify/diverse_prompts.txt the streams write code, prose in several languages, tables
and arithmetic side by side, which is nearer to mixed client traffic than one problem
with other numbers in it. Round time is the number to compare across prompt sets, not
tokens per second: tokens per round depend on the text.

Drafts accepted by position. The engine counts, for each draft position, the rounds in
which that draft was accepted (vllm:spec_decode_num_accepted_tokens_per_pos, which a
round moves for every position up to its last accepted draft:
vllm/v1/spec_decode/metrics.py:41-49 in v0.34.0). Under each row the script prints
those shares and what they add up to at each draft length: 1 + the shares of the first
w positions is the tokens a round would yield if the lane drafted w tokens and its
first w drafts were what they are now. That last part is an assumption, so the figure
for a shorter draft is an estimate; the one for the lane's own length is the measured
tokens per round. Drafts cost rows whether they are accepted or not: this line is what
the last ones buy.

--each N, with --prompts-file: every one of the first N prompts by itself, one stream
at a time, a row per prompt. Tokens per round depend on the text, and this is the table
of it: what one stream gets on code, on prose, on a table, with the estimate for a
draft of 3 and of 5 beside the lane's own.

Chunk gaps. Under the rows the script prints how the times between a stream's chunks
are spread while every stream decodes: the mean, the median, the 90th and 99th
percentile, the longest, and the share that is longer than one and a half times the
median. A lane whose every round got slower moves the median with the mean. A lane on
which some rounds are much slower than the rest moves the mean, the upper percentiles
and that share, and leaves the median: the "ms/round" of the lines above it is that
median, so on such a lane read the mean.

Which streams share an engine step ("split"). Under pipeline parallelism with async
scheduling a stream decodes on every second engine step (vllm/v1/core/sched/
async_scheduler.py:14, 45 and scheduler.py:832-836 in v0.34.0), so two streams either
ride the same steps or alternate, and that is settled by the step each first decodes
on. Read from the engine's step loop (vllm/v1/engine/core.py:758-828), not measured:
the two cases should cost differently. Streams on the same steps run one batch through
stage 0 and then stage 1, the other stage idle meanwhile. Streams on alternating steps
keep both stages busy, and then the slower stage sets the pace for both. This script
is how that gets measured. It reads the split two ways:

    split         from the client side: every stream's chunks carry a time, and streams
                  whose chunks arrive together are on the same steps. "4+0": all four on
                  the same steps. "3+1", "2+2": on alternating steps. Taken over the
                  time in which all streams decode; "-" when it cannot be told (the
                  reason is in the notes)
    streams/step  from the engine: rounds over engine steps that returned output
                  (vllm:spec_decode_num_drafts over the count of
                  vllm:iteration_tokens_total). N when all N share their steps, N/2
                  when they alternate, whatever the split

When the runs of one concurrency do not all have the same split, each split gets a line
of its own under the table row: the milliseconds per round and tokens per second of
the runs that had it, measured from the chunks where all streams decode. That is the
comparison of "same steps" against "alternating steps" on one lane, with nothing
changed on it. --stagger-ms M starts each stream M milliseconds after the one before,
which varies the step each first decodes on from run to run (try 7 with --repeat 10).
With a stagger the streams do not start together, so the row's own numbers include the
ramp: read the split lines.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid

# verify/texts_prod.py: the prompt the README's single-stream tok/s and step time are on.
TRAIN_PROMPT = (
    "A train leaves at 60 mph. Two hours later a second train leaves the same station at 90 mph on a "
    "parallel track. At what distance does the second catch the first? Reason step by step before answering."
)
PROMPT = (
    "Write a detailed, numbered walk through the history of timekeeping, from sundials and "
    "water clocks to pendulum clocks, marine chronometers, quartz and atomic clocks. Give "
    "each device a paragraph: how it works, what limited its accuracy, what it made possible. "
    "Do not stop before you have covered at least twelve devices."
)


def context(words: int) -> str:
    """Numbered ledger lines, no two alike, about `words` words."""
    rng = random.Random(1017)
    things = ["harbor", "lantern", "granite", "willow", "copper", "meadow", "signal", "ember", "orchard", "ledger"]
    lines, count = [], 0
    while count < words:
        a, b = rng.sample(things, 2)
        line = (f"Entry {len(lines) + 1}: the {a} was logged at {rng.randrange(24):02d}:{rng.randrange(60):02d} "
                f"with {rng.randrange(2, 900)} units and stored beside the {b}.")
        lines.append(line)
        count += len(line.split())
    return "Here is a ledger you may ignore.\n" + "\n".join(lines) + "\n\n"


COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "accepted": "vllm:spec_decode_num_accepted_tokens",
    "generated": "vllm:generation_tokens",
    "cached": "vllm:prompt_tokens_cached",
    "prompt": "vllm:prompt_tokens",
    "finished": "vllm:request_success",
    "steps": "vllm:iteration_tokens_total_count",  # engine steps that returned output
}


def variant(prompt: str, k: int) -> str:
    """The prompt of stream k under --vary: stream 0 keeps the prompt as it is."""
    if k == 0:
        return prompt
    if "at 60 mph" in prompt and "at 90 mph" in prompt:
        return prompt.replace("at 60 mph", f"at {60 + 2 * k} mph").replace("at 90 mph", f"at {90 + 3 * k} mph")
    return f"(Request {k + 1}.) " + prompt


BY_POSITION = "vllm:spec_decode_num_accepted_tokens_per_pos"  # label position="k", one series per draft position


AUTH: dict[str, str] = {}  # the Authorization header, when the lane asks for an API key
EXTRA: dict = {}  # --extra: fields every request body also carries
OWN_FIELDS = ("model", "messages", "max_tokens", "stream", "stream_options", "cache_salt", "temperature", "top_p")


def counters(url: str, timeout: float) -> dict:
    """Each counter summed over its label sets; None when the lane does not export it.
    Under "by_position": {draft position: rounds that accepted it}, empty if not exported."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url + "/metrics", headers=AUTH), timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError):
        return {**dict.fromkeys(COUNTERS), "by_position": {}}
    found: dict[str, list[float]] = {key: [] for key in COUNTERS}
    by_position: dict[int, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        sample = line.split("{", 1)[0].split(" ", 1)[0]
        for key, name in COUNTERS.items():
            if sample in (name, name + "_total"):
                found[key].append(float(line.rsplit(" ", 1)[1]))
        if sample in (BY_POSITION, BY_POSITION + "_total") and 'position="' in line:
            position = line.split('position="', 1)[1].split('"', 1)[0]
            if position.isdigit():
                by_position[int(position)] = by_position.get(int(position), 0.0) + float(line.rsplit(" ", 1)[1])
    return {**{key: (sum(values) if values else None) for key, values in found.items()}, "by_position": by_position}


def stream_one(url: str, model: str, tokens: int, scale: float | None, timeout: float, prompt: str,
               top_p: float | None = None, temperature: float | None = 0, delay: float = 0.0) -> dict:
    """One streamed chat completion: {first, last, tokens, finish, arrivals} with wall-clock
    times; arrivals is when each chunk that carried text came in. temperature None sends no
    temperature: the lane's own default applies. delay: seconds to wait before sending."""
    if delay:
        time.sleep(delay)
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": uuid.uuid4().hex,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if scale is not None:
        body["vllm_xargs"] = {"steering_scale": scale}
    for key, value in EXTRA.items():
        if key == "vllm_xargs" and isinstance(value, dict):
            body["vllm_xargs"] = {**value, **body.get("vllm_xargs", {})}  # --scale wins over --extra's
        else:
            body[key] = value
    request = urllib.request.Request(
        url + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **AUTH},
    )
    sent = time.perf_counter()
    first = last = None
    arrivals: list[float] = []
    completion_tokens = prompt_tokens = cached_tokens = None
    finish = None
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        key = " (this lane asks for an API key: --api-key-file or --api-key)" if exc.code == 401 and not AUTH else ""
        raise RuntimeError(f"HTTP {exc.code} from /v1/chat/completions{key}: {detail}") from None
    with response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise RuntimeError(f"the lane answered with an error in the stream: {str(chunk['error'])[:300]}")
            if chunk.get("usage"):
                completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)
                prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)
                details = chunk["usage"].get("prompt_tokens_details") or {}
                if details.get("cached_tokens") is not None:
                    cached_tokens = int(details["cached_tokens"])
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if any(delta.get(key) for key in ("content", "reasoning_content", "reasoning")):
                    last = time.perf_counter()
                    arrivals.append(last)
                    if first is None:
                        first = last
                finish = choice.get("finish_reason") or finish
    if first is None or completion_tokens is None:
        raise RuntimeError("the lane streamed no tokens or no usage (stream_options.include_usage)")
    return {"sent": sent, "first": first, "last": last, "tokens": int(completion_tokens), "finish": finish,
            "prompt_tokens": prompt_tokens, "cached_tokens": cached_tokens, "arrivals": arrivals}


TOGETHER, APART = 0.6, 0.4  # share of one stream's chunks that have one of the other's beside them
BESIDE = 0.25  # "beside": within this part of a round


def split_of(runs: list[dict]) -> dict:
    """Which streams' rounds end together, from when their chunks arrived.

    Looks only at the time in which every stream is decoding. Streams on the same engine
    steps get their chunks together; streams on alternating steps get them about half a
    round apart. For each pair, the share of one stream's chunks that have a chunk of the
    other within a quarter of a round is taken in the first and in the second half of
    that time: at least 60 % in both halves is "same steps", at most 40 % in both is
    "other steps". Anything else (a stream that changed steps on the way, a lane that
    does not send one chunk a round) gives no split, with the reason.
    Returns {"split", "why", "period_ms", "tok_s_per_stream", "tok_s_aggregate", "rounds",
    "seconds", "gaps_ms"}; split is a string like "3+1", or None. gaps_ms is how the times
    between a stream's chunks in that window are spread, all streams together."""
    start = max(r["arrivals"][0] for r in runs)
    end = min(r["arrivals"][-1] for r in runs)
    series = [[t for t in r["arrivals"] if start <= t <= end] for r in runs]
    out: dict = {"split": None, "why": None, "period_ms": None, "tok_s_per_stream": None, "tok_s_aggregate": None,
                 "rounds": min(len(s) for s in series), "seconds": max(end - start, 0.0)}
    if out["rounds"] < 16:
        out["why"] = f"the streams decode side by side for {out['rounds']} chunks only"
        return out
    periods = [statistics.median(b - a for a, b in zip(s, s[1:])) for s in series]
    period = statistics.median(periods)
    if period <= 0:
        out["why"] = "the chunks of a stream arrive at once"
        return out
    out["period_ms"] = 1000.0 * period
    gaps = sorted(1000.0 * (b - a) for s in series for a, b in zip(s, s[1:]))
    cut = lambda share: gaps[min(len(gaps) - 1, int(share * len(gaps)))]  # noqa: E731
    out["gaps_ms"] = {"count": len(gaps), "mean": sum(gaps) / len(gaps), "p50": cut(0.5), "p90": cut(0.9),
                      "p99": cut(0.99), "max": gaps[-1], "over": sum(g > 1.5 * cut(0.5) for g in gaps) / len(gaps)}
    # Tokens a chunk carries, from each stream's totals; over the round time in the window.
    rates = [(r["tokens"] / len(r["arrivals"])) / gap for r, gap in zip(runs, periods) if gap > 0]
    out["tok_s_per_stream"] = statistics.median(rates) if rates else None
    out["tok_s_aggregate"] = sum(rates) if rates else None
    count = len(series)
    if count == 1:
        out["split"] = "1"
        return out
    tolerance = BESIDE * period

    def beside(a: list[float], b: list[float]) -> float:
        hits, k = 0, 0
        for t in a:
            while k + 1 < len(b) and abs(b[k + 1] - t) <= abs(b[k] - t):
                k += 1
            hits += abs(b[k] - t) <= tolerance
        return hits / len(a)

    def relation(a: list[float], b: list[float]) -> str:
        """'same', 'other' or 'unsettled', the two halves of the window agreeing."""
        seen = set()
        for first, second in ((a[: len(a) // 2], b), (a[len(a) // 2:], b), (b[: len(b) // 2], a), (b[len(b) // 2:], a)):
            share = beside(first, second)
            seen.add("same" if share >= TOGETHER else "other" if share <= APART else "unsettled")
        return seen.pop() if len(seen) == 1 else "unsettled"

    group = list(range(count))

    def find(i: int) -> int:
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i

    unsettled = 0
    for i in range(count):
        for j in range(i + 1, count):
            kind = relation(series[i], series[j])
            if kind == "same":
                group[find(i)] = find(j)
            elif kind == "unsettled":
                unsettled += 1
    if unsettled:
        out["why"] = (f"{unsettled} pair(s) of streams had their chunks arrive together part of the time: a stream "
                      "changed steps during the run, or the lane does not send one chunk a round")
        return out
    sizes = sorted((sum(find(i) == root for i in range(count)) for root in {find(i) for i in range(count)}), reverse=True)
    out["split"] = "+".join(str(size) for size in sizes) if len(sizes) > 1 else f"{sizes[0]}+0"
    return out


def measure(url: str, model: str, streams: int, tokens: int, scale: float | None, timeout: float,
            prompt: str, top_p: float | None = None, temperature: float | None = 0, stagger: float = 0.0,
            vary: bool = False, prompts: list[str] | None = None) -> dict:
    """prompts: a prompt per stream (stream k sends prompts[k % len(prompts)]) in place of prompt."""
    def of(k: int) -> str:
        if prompts:
            return prompts[k % len(prompts)]
        return variant(prompt, k) if vary else prompt

    before = counters(url, timeout)
    with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as pool:
        futures = [pool.submit(stream_one, url, model, tokens, scale, timeout, of(k), top_p, temperature, k * stagger)
                   for k in range(streams)]
        runs = [future.result() for future in futures]
    # The engine hands a step's output to the streams before it records that step's
    # counters (vllm/v1/engine/async_llm.py:919, then :944): let the last step land.
    time.sleep(0.3)
    after = counters(url, timeout)
    per_stream = [(r["tokens"] - 1) / (r["last"] - r["first"]) for r in runs if r["last"] > r["first"]]
    decode_start, decode_end = min(r["first"] for r in runs), max(r["last"] for r in runs)
    received = sum(r["tokens"] for r in runs)
    if decode_end <= decode_start:
        raise RuntimeError("every token arrived in one chunk: there is no decode interval to time "
                           "(is the lane streaming token by token?)")
    out = {
        "streams": streams,
        "prompt_tokens": runs[0]["prompt_tokens"],
        "tokens": received,
        "ttft_s": sorted(r["first"] - r["sent"] for r in runs)[len(runs) // 2],
        "tok_s_per_stream": sorted(per_stream)[len(per_stream) // 2] if per_stream else None,
        "tok_s_aggregate": (received - streams) / (decode_end - decode_start),
        # All tokens over all the time, the first token's latency included: what a
        # "tokens / wall seconds" number measures.
        "tok_s_with_ttft": sorted(r["tokens"] / (r["last"] - r["sent"]) for r in runs)[len(runs) // 2],
        "short": sum(r["tokens"] < tokens for r in runs),
    }
    delta = {key: (after[key] - before[key]) if None not in (after[key], before[key]) else None for key in COUNTERS}
    out["counters"] = delta
    # Prompt tokens that came from the prefix cache: per response where the lane says so,
    # else the counter's movement over the run (which other traffic moves too).
    if all(r["cached_tokens"] is not None for r in runs):
        out["cached_prompt_tokens"], out["cached_source"] = sum(r["cached_tokens"] for r in runs), "usage"
    elif delta["cached"] is not None:
        out["cached_prompt_tokens"], out["cached_source"] = int(round(delta["cached"])), "vllm:prompt_tokens_cached"
    else:
        out["cached_prompt_tokens"], out["cached_source"] = None, None
    if delta["drafts"]:
        generated = delta["generated"] if delta["generated"] is not None else received
        out["tokens_per_round"] = generated / delta["drafts"]
        out["accepted_per_round"] = (delta["accepted"] or 0.0) / delta["drafts"]
        # The share of rounds that accepted the draft at each position, first position first.
        positions = sorted(set(before["by_position"]) & set(after["by_position"]))
        if positions and positions == list(range(len(positions))):
            out["accepted_by_position"] = [(after["by_position"][k] - before["by_position"][k]) / delta["drafts"]
                                           for k in positions]
        # Each stream runs its own rounds side by side: rounds per stream over the decode time.
        out["ms_per_round"] = 1000.0 * (decode_end - decode_start) / (delta["drafts"] / streams)
        # Rounds over engine steps that returned output: how many streams a step serves.
        out["streams_per_step"] = delta["drafts"] / delta["steps"] if delta["steps"] else None
        # One chunk a round is what the client-side split rests on (a stream's first chunk
        # is its prefill's, which is no round).
        out["chunks_per_round"] = (sum(len(r["arrivals"]) for r in runs) - streams) / delta["drafts"]
    # Other traffic, three ways: tokens the lane generated that this run did not receive,
    # prompt tokens it counted that this run did not send (a client that prefills a long
    # prompt and says little moves only this one), requests that finished and were not ours.
    strangers = []
    if delta["generated"] is not None and delta["generated"] > 1.05 * received + 8:
        strangers.append(f"{delta['generated'] - received:.0f} more generated tokens than this run received")
    sent_prompt = sum(r["prompt_tokens"] or 0 for r in runs)
    if delta["prompt"] is not None and all(r["prompt_tokens"] is not None for r in runs) \
            and delta["prompt"] > sent_prompt + 8:
        strangers.append(f"{delta['prompt'] - sent_prompt:.0f} more prompt tokens than this run sent")
    if delta["finished"] is not None and delta["finished"] > streams:
        strangers.append(f"{delta['finished'] - streams:.0f} more finished request(s) than this run made")
    out["other_traffic"], out["other_traffic_by"] = bool(strangers), strangers
    out["window"] = split_of(runs)
    if out.get("chunks_per_round") is not None and not 0.9 <= out["chunks_per_round"] <= 1.1:
        # Chunk times say which steps a stream is on only when a chunk is a round.
        out["window"]["from_chunks"], out["window"]["split"] = out["window"]["split"], None
        out["window"]["why"] = (f"{out['chunks_per_round']:.2f} chunks arrived per engine round, and the split is "
                                "read from chunk times, one chunk a round")
    out["split"] = out["window"]["split"]
    return out


TABLE_HEAD = (f"{'streams':>7} {'ttft s':>7} {'tok/s per stream':>17} {'tok/s aggregate':>16} {'with ttft':>10} "
              f"{'tokens/round':>13} {'ms/round':>9} {'split':>6} {'streams/step':>13}  notes")


def show(streams: int, runs: list[dict], stagger_ms: float) -> None:
    """One concurrency's row, and the lines under it, from its runs (measured now or read back)."""
    mid = sorted(runs, key=lambda r: r["tok_s_aggregate"])[len(runs) // 2]
    spread = (max(r["tok_s_aggregate"] for r in runs) - min(r["tok_s_aggregate"] for r in runs)) / mid["tok_s_aggregate"]
    notes = [f"runs within {100 * spread:.1f}%"]
    cached = [r["cached_prompt_tokens"] for r in runs]
    if None in cached:
        notes.append("prefill: the lane does not report cached prompt tokens")
    elif not any(cached):
        notes.append("prefill cold in every run")
    else:
        notes.append(f"prefill NOT cold: {max(cached)} prompt tokens from the prefix cache in one run "
                     f"({runs[0]['cached_source']})")
    if mid["short"]:
        notes.append(f"{mid['short']} stream(s) stopped before max_tokens")
    if "tokens_per_round" not in mid:
        notes.append("no speculative-decoding counters on this lane: per-round numbers unmeasured")
    busy = [r for r in runs if r.get("other_traffic")]
    if busy:
        # A receipt written before revision 6.5 says only that the generated tokens did not add up.
        said = [r.get("other_traffic_by") or ["more generated tokens than this run received"] for r in busy]
        notes.append(f"other traffic in {len(busy)} of {len(runs)} runs: the lane counted "
                     + ", ".join(max(said, key=len)))
    if mid["split"] is None:
        notes.append("split not told: " + mid["window"]["why"])
    elif streams > 1 and mid.get("streams_per_step") is not None:
        # The engine's own count must say the same: N streams a step when they share
        # their steps, N/2 when they alternate.
        alternating = "+0" not in mid["split"]
        expected = streams / 2 if alternating else streams
        if abs(mid["streams_per_step"] - expected) > 0.25 * expected:
            notes.append(f"the chunks say {mid['split']} and the engine served {mid['streams_per_step']:.2f} "
                         f"streams a step where that split makes {expected:g}: the two do not agree")
    show = lambda key, form: format(mid[key], form) if mid.get(key) is not None else "-"  # noqa: E731
    print(f"{streams:>7} {mid['ttft_s']:>7.3f} {show('tok_s_per_stream', '17.1f'):>17} {mid['tok_s_aggregate']:>16.1f} "
          f"{mid['tok_s_with_ttft']:>10.1f} {show('tokens_per_round', '13.2f'):>13} {show('ms_per_round', '9.2f'):>9} "
          f"{mid['split'] or '-':>6} {show('streams_per_step', '13.2f'):>13}  {'; '.join(notes)}")
    splits = sorted({r["split"] for r in runs if r["split"]})
    by_split = len(splits) > 1 or bool(stagger_ms and splits)
    side = [r["window"] for r in runs if r["window"].get("tok_s_per_stream")]
    if streams > 1 and side and not by_split:
        # The rate while every stream decodes: what the row's whole-run numbers are
        # when the streams start and end together, and what they hide when they do not.
        period = statistics.median(w["period_ms"] for w in side)
        late = mid.get("ms_per_round") is not None and mid["ms_per_round"] > 1.05 * period
        print(f"{'':>7} where all streams decode: {period:.2f} ms/round, "
              f"{statistics.median(w['tok_s_per_stream'] for w in side):.1f} tok/s per stream, "
              f"{statistics.median(w['tok_s_aggregate'] for w in side):.1f} aggregate ({len(side)} of {len(runs)} runs)"
              + ("; the row's ms/round and aggregate are over the whole run, in part of which fewer "
                 "streams were decoding" if late else ""))
    if by_split:
        # The runs did not all put their streams on the same steps: each way by itself,
        # from the chunks that arrived while every stream was decoding.
        for split in splits:
            had = [r["window"] for r in runs if r["split"] == split]
            print(f"{'':>7} split {split}: {len(had)} of {len(runs)} runs; "
                  f"{statistics.median(w['period_ms'] for w in had):.2f} ms/round, "
                  f"{statistics.median(w['tok_s_per_stream'] for w in had):.1f} tok/s per stream, "
                  f"{statistics.median(w['tok_s_aggregate'] for w in had):.1f} aggregate, where all streams decode")
        untold = sum(r["split"] is None for r in runs)
        if untold:
            print(f"{'':>7} split not told in {untold} of {len(runs)} runs")
    shares = mid.get("accepted_by_position")
    if shares:
        yields, total = [], 1.0
        for share in shares:
            total += share
            yields.append(total)
        print(f"{'':>7} drafts accepted by position, the median run: " + " ".join(f"{share:.2f}" for share in shares)
              + "; tokens a round at a draft of " + ", ".join(f"{k + 1}: {y:.2f}" for k, y in enumerate(yields))
              + (" (other traffic is in these counts)" if mid.get("other_traffic") else ""))
    gaps = mid["window"].get("gaps_ms")
    if gaps:
        print(f"{'':>7} chunk gaps{' where all streams decode' if streams > 1 else ''}, the median run: mean "
              f"{gaps['mean']:.2f} ms, half under {gaps['p50']:.2f}, 90% under {gaps['p90']:.2f}, 99% under "
              f"{gaps['p99']:.2f}, longest {gaps['max']:.2f}; {100 * gaps['over']:.1f}% longer than one and a half "
              f"times the median ({gaps['count']} gaps)")


EACH_HEAD = (f"{'prompt':>7} {'tok/s':>8} {'tokens/round':>13} {'ms/round':>9} {'at 3 drafts':>12} {'at 5 drafts':>12}  "
             "begins (median run of each; the two estimates are tokens a round)")


def show_each(index: int, text: str, runs: list[dict]) -> None:
    """One prompt by itself at one stream: its row."""
    mid = sorted(runs, key=lambda r: r["tok_s_aggregate"])[len(runs) // 2]
    shares = mid.get("accepted_by_position") or []
    at = lambda width: f"{1.0 + sum(shares[:width]):.2f}" if len(shares) >= width else "-"  # noqa: E731
    gaps = mid["window"].get("gaps_ms")
    round_ms = gaps["mean"] if gaps else mid.get("ms_per_round")
    flags = []
    if any(r.get("other_traffic") for r in runs):
        flags.append("OTHER TRAFFIC")
    if mid["short"]:
        flags.append("stopped before max_tokens")
    print(f"{index + 1:>7} {mid['tok_s_per_stream']:>8.1f} "
          f"{format(mid['tokens_per_round'], '13.2f') if 'tokens_per_round' in mid else '-':>13} "
          f"{format(round_ms, '9.2f') if round_ms is not None else '-':>9} {at(3):>12} {at(5):>12}  "
          f"{text[:44]}{' [' + '; '.join(flags) + ']' if flags else ''}")


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--url", help="API base, e.g. http://127.0.0.1:8001")
    parser.add_argument("--read", metavar="FILE",
                        help="print the table of a receipt written with --json, the lines under each row included; "
                             "no request is sent")
    parser.add_argument("--model", help="served model name (default: first of /v1/models)")
    parser.add_argument("--streams", default="1,4", help="concurrencies to measure, in order")
    parser.add_argument("--tokens", type=int, default=1024, help="max_tokens of each stream")
    parser.add_argument("--repeat", type=int, default=3, help="runs per concurrency; the median is reported")
    parser.add_argument("--scale", type=float, help="steering_scale sent with every request")
    parser.add_argument("--context-words", type=int, default=0, help="words of ledger in front of the prompt")
    parser.add_argument("--train-prompt", action="store_true",
                        help="the README's prompt (verify/texts_prod.py) with its top_p 0.95; use --tokens 512")
    parser.add_argument("--prompt-file", help="send the text of this file as the prompt")
    parser.add_argument("--top-p", type=float, help="top_p sent with every request (default: not sent)")
    parser.add_argument("--temperature", default="0",
                        help="temperature sent with every request (default 0), or 'server': send neither "
                             "temperature nor top_p, so the lane's own generation defaults apply")
    parser.add_argument("--stagger-ms", type=float, default=0.0,
                        help="start each stream this many milliseconds after the one before (default 0: together)")
    parser.add_argument("--prompts-file", metavar="FILE",
                        help="a prompt per line: stream k sends line k (verify/diverse_prompts.txt is one such file)")
    parser.add_argument("--each", type=int, metavar="N",
                        help="with --prompts-file: each of its first N prompts by itself at one stream, a row per prompt")
    parser.add_argument("--vary", action="store_true",
                        help="give each stream its own prompt (other speeds in the train prompt, a numbered line "
                             "before any other), so that streams do not write the same text")
    parser.add_argument("--api-key-file", metavar="FILE",
                        help="send the key in this file as 'Authorization: Bearer ...' (a lane started with an API key)")
    parser.add_argument("--api-key", help="the same, with the key on the command line (or DECODE_RATE_API_KEY)")
    parser.add_argument("--extra", metavar="JSON",
                        help="a JSON object of fields every request also carries, e.g. "
                             "'{\"chat_template_kwargs\": {\"thinking\": false}}' on a lane that thinks by default")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--json", help="write every run as JSON to this file ('-' = stdout)")
    args = parser.parse_args()
    if args.read:
        with open(args.read, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and data.get("each"):
            print(f"read back from {args.read}: lane {data.get('model')} at {data.get('url')}; {data.get('prompt')}, "
                  f"each by itself, max_tokens {data.get('tokens')}, temperature {data.get('temperature')}")
            print(EACH_HEAD)
            for index, one in enumerate(data["each"]):
                show_each(index, one["prompt"], one["runs"])
            return 0
        if not isinstance(data, dict) or not data.get("runs"):
            parser.error(f"{args.read} is not a receipt of this script (no runs in it)")
        sent = data.get("temperature")
        print(f"read back from {args.read}: lane {data.get('model')} at {data.get('url')}; {data.get('prompt')}, "
              f"max_tokens {data.get('tokens')}, "
              f"temperature {'not sent (the lane decides)' if sent == 'server' else sent}, "
              f"top_p {'not sent' if data.get('top_p') is None else data['top_p']}, "
              f"steering_scale {'not sent' if data.get('scale') is None else data['scale']}; "
              f"median of {len(data['runs'][0])} runs"
              + (f"; each stream starts {data['stagger_ms']:g} ms after the one before" if data.get("stagger_ms") else "")
              + ("; each stream has its own prompt" if data.get("vary") or data.get("prompts_file") else ""))
        print(TABLE_HEAD)
        for runs in data["runs"]:
            show(runs[0]["streams"], runs, data.get("stagger_ms") or 0.0)
        return 0
    if not args.url:
        parser.error("--url is needed (or --read FILE, to print a saved receipt)")
    url = args.url.rstrip("/")
    key = args.api_key or os.environ.get("DECODE_RATE_API_KEY", "")
    if args.api_key_file:
        with open(args.api_key_file, encoding="utf-8") as handle:
            key = handle.read().strip()
    if key:
        AUTH["Authorization"] = "Bearer " + key
    if args.extra:
        try:
            extra = json.loads(args.extra)
        except ValueError as exc:
            parser.error(f"--extra is not JSON: {exc}")
        if not isinstance(extra, dict):
            parser.error("--extra must be a JSON object")
        own = sorted(set(extra) & set(OWN_FIELDS))
        if own:
            parser.error(f"--extra may not set {', '.join(own)}: this script sets those itself, and has options for them")
        EXTRA.update(extra)
    model = args.model
    if model is None:
        try:
            with urllib.request.urlopen(urllib.request.Request(url + "/v1/models", headers=AUTH),
                                        timeout=args.timeout) as response:
                model = json.loads(response.read())["data"][0]["id"]
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                print(f"DECODE-RATE FAIL: HTTP 401 from /v1/models: this lane asks for an API key "
                      f"({'the one given was refused' if AUTH else '--api-key-file or --api-key'})")
                return 1
            raise
    if args.train_prompt and args.prompt_file:
        parser.error("--train-prompt and --prompt-file are two different prompts")
    server_sampling = args.temperature.strip().lower() == "server"
    try:
        temperature = None if server_sampling else float(args.temperature)
    except ValueError:
        parser.error("--temperature takes a number or the word server")
    if temperature is not None and temperature < 0:
        parser.error("--temperature must not be negative")
    if args.stagger_ms < 0:
        parser.error("--stagger-ms must not be negative")
    top_p, which = args.top_p, "the essay prompt"
    text = PROMPT
    if args.train_prompt:
        text, which = TRAIN_PROMPT, "the README's train prompt"
        if not server_sampling:  # with the lane's own defaults nothing is added to the request
            top_p = 0.95 if top_p is None else top_p
    elif args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as handle:
            text, which = handle.read().strip(), f"the prompt in {args.prompt_file}"
    front = context(args.context_words) if args.context_words else ""
    prompts = None
    if args.prompts_file:
        if args.train_prompt or args.prompt_file or args.vary:
            parser.error("--prompts-file is a set of prompts of its own: leave out --train-prompt, --prompt-file and --vary")
        with open(args.prompts_file, encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
        if len(lines) < 2:
            parser.error(f"--prompts-file needs at least two prompts, one per line; {args.prompts_file} has {len(lines)}")
        most = max(int(v) for v in args.streams.split(","))
        which = (f"{len(lines)} prompts of {args.prompts_file}, one per stream"
                 + (f" (streams past the first {len(lines)} repeat them)" if most > len(lines) else ""))
        text, prompts = lines[0], [front + line for line in lines]
    if args.each is not None:
        if not prompts or args.each < 1:
            parser.error("--each N needs --prompts-file and an N of 1 or more")
        which = f"the first {min(args.each, len(prompts))} prompts of {args.prompts_file}, each by itself at one stream"
    prompt = front + text
    settle = stream_one(url, model, 16, args.scale, args.timeout, prompt, top_p, temperature)  # settles the lane
    print(f"lane {model} at {url}; {which}, {settle['prompt_tokens']} prompt tokens, max_tokens {args.tokens}, "
          f"temperature {'not sent (the lane decides)' if temperature is None else f'{temperature:g}'}, "
          f"top_p {'not sent' if top_p is None else top_p}, "
          f"steering_scale {'not sent' if args.scale is None else args.scale}; median of {args.repeat} runs"
          + (f"; each stream starts {args.stagger_ms:g} ms after the one before" if args.stagger_ms else "")
          + ("; each stream has its own prompt" if args.vary else ""))
    if EXTRA or AUTH:
        print("every request also carries " + (json.dumps(EXTRA) if EXTRA else "nothing more")
              + ("; sent with an API key" if AUTH else ""))
    if temperature != 0:
        print("sampled requests: the text differs from run to run and tokens per round with it. Read the "
              "median and the spread, and compare with another lane at this same setting only")
    if args.each is not None:
        assert prompts is not None
        print(EACH_HEAD)
        alone = []
        for index, own in enumerate(prompts[: args.each]):
            runs = [measure(url, model, 1, args.tokens, args.scale, args.timeout, own, top_p, temperature)
                    for _ in range(args.repeat)]
            alone.append({"prompt": own, "runs": runs})
            show_each(index, own[len(front):], runs)
        if args.json:
            payload = json.dumps({"url": url, "model": model, "tokens": args.tokens, "scale": args.scale, "prompt": which,
                                  "top_p": top_p, "temperature": "server" if temperature is None else temperature,
                                  "context_words": args.context_words, "prompts_file": args.prompts_file,
                                  "extra": EXTRA, "api_key": bool(AUTH),
                                  "each": alone}, indent=2)
            if args.json == "-":
                print(payload)
            else:
                with open(args.json, "w", encoding="utf-8") as handle:
                    handle.write(payload)
        return 0
    print(TABLE_HEAD)
    everything = []
    for streams in [int(v) for v in args.streams.split(",")]:
        runs = [measure(url, model, streams, args.tokens, args.scale, args.timeout, prompt, top_p, temperature,
                        args.stagger_ms / 1000.0, args.vary, prompts)
                for _ in range(args.repeat)]
        everything.append(runs)
        show(streams, runs, args.stagger_ms)
    if args.json:
        payload = json.dumps({"url": url, "model": model, "tokens": args.tokens, "scale": args.scale, "prompt": which,
                              "top_p": top_p, "temperature": "server" if temperature is None else temperature,
                              "context_words": args.context_words, "stagger_ms": args.stagger_ms, "vary": args.vary,
                              "prompts_file": args.prompts_file, "extra": EXTRA, "api_key": bool(AUTH),
                              "runs": everything}, indent=2)
        if args.json == "-":
            print(payload)
        else:
            with open(args.json, "w", encoding="utf-8") as handle:
                handle.write(payload)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, urllib.error.URLError) as exc:
        print(f"DECODE-RATE ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
