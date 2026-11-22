#!/usr/bin/env python3
"""What did this boot's steering module say it is running, on each rank?

Standard library only; it reads log text and sends nothing. The steering module prints
one line per rank when it has installed its hooks:

    [pp2_steering] rank 0: steering layers 4-19 of this stage's 0-19; 32 hooks, 32 extra
    kernels per forward pass by op count; kernel fused; decoder replay off; sidecar sha256
    68551f9b8deca315; default scale 0.0; debug off

That line is the module's own account of the environment it got. A knob that was set in
a config file and never reached the container changes nothing in it: on 2026-11-21 a
boot configured for the fused kernel ran the torch path, and the only sign was one word
in this line. This script reads the line of every rank and compares it with what the
boot was meant to be, so that the comparison is a command and an exit status:

    docker logs <rank-0 container> > rank0.log 2>&1      # on each Station
    python3 verify/boot_receipt.py --kernel fused --debug on --sidecar 68551f9b8deca315 \\
        rank0.log rank1.log --out results/boot-receipt.json

    python3 verify/boot_receipt.py --no-module rank0.log rank1.log   # a boot without steering

--ranks N (default 2) is how many ranks must have spoken. Each of --kernel, --debug,
--replay, --sidecar and --default-scale is checked only when given. A line that says an
output fell back to the torch path, or a FATAL line of the module, fails the receipt
whatever else matches. --no-module passes only when no rank printed anything of the
module's. Exit status 0 when everything asked for holds, 1 when not.

--selftest runs the script on log text written here and exits.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

LINE = re.compile(
    r"\[pp2_steering\] rank (?P<rank>\d+): steering layers (?P<layers>\S+) of this stage's (?P<stage>\d+-\d+); "
    r"(?P<hooks>\d+) hooks, (?P<kernels>\d+) extra kernels per forward pass by op count; kernel (?P<kernel>\w+); "
    r"decoder replay (?P<replay>[^;]+); sidecar sha256 (?P<sidecar>[0-9a-f]+); default scale (?P<default_scale>[0-9.]+); "
    r"debug (?P<debug>on|off)"
)
FELL_BACK = re.compile(r"\[pp2_steering\] rank (\d+): the fused kernel does not take (.*?); torch path there")
FATAL = re.compile(r"\[pp2_steering\] FATAL: (.*)")
ANY = re.compile(r"\[pp2_steering\] ")
# Module revision 6 and later: what was registered, and that all of it ran.
COUNTED = re.compile(r"\[pp2_steering\] rank (\d+): sites (\S+); (\d+) operators, (\d+) of them a shared expert "
                     r"edited in the MoE kernel's own slot")
RAN = re.compile(r"\[pp2_steering\] rank (\d+): all (\d+) operators ran in the first forward pass "
                 r"\((\d+) attention, (\d+) shared-expert\)")
NEVER_RAN = re.compile(r"steering: (\d+) of (\d+) operators did not run in the first forward pass")


def read(texts: list[str]) -> dict:
    """{"ranks": {rank: fields of its last install line}, "fell_back": [...], "fatal": [...],
    "lines": count of the module's lines}."""
    ranks: dict[int, dict] = {}
    fell_back, fatal, lines = [], [], 0
    counted: dict[int, dict] = {}
    ran: dict[int, dict] = {}
    never_ran: list[str] = []
    for text in texts:
        for line in text.splitlines():
            found = NEVER_RAN.search(line)  # an exception's text, not one of the module's log lines
            if found:
                never_ran.append(f"{found.group(1)} of {found.group(2)}")
            if not ANY.search(line):
                continue
            lines += 1
            found = COUNTED.search(line)
            if found:
                counted[int(found.group(1))] = {"sites": found.group(2), "operators": int(found.group(3)),
                                                "slot": int(found.group(4))}
                continue
            found = RAN.search(line)
            if found:
                ran[int(found.group(1))] = {"operators": int(found.group(2)), "attn": int(found.group(3)),
                                            "shared": int(found.group(4))}
                continue
            found = LINE.search(line)
            if found:
                fields = found.groupdict()
                ranks[int(fields.pop("rank"))] = {**fields, "hooks": int(fields["hooks"]), "kernels": int(fields["kernels"])}
                continue
            found = FELL_BACK.search(line)
            if found:
                fell_back.append({"rank": int(found.group(1)), "why": found.group(2)})
                continue
            found = FATAL.search(line)
            if found:
                fatal.append(found.group(1)[:300])
    return {"ranks": ranks, "fell_back": fell_back, "fatal": fatal, "lines": lines, "counted": counted, "ran": ran,
            "never_ran": never_ran}


def judge(seen: dict, want: dict, ranks: int, no_module: bool) -> tuple[bool, list[str]]:
    """(holds, one line per finding)."""
    out, good = [], True
    if no_module:
        if seen["lines"]:
            good = False
            out.append(f"FAIL  the steering module printed {seen['lines']} line(s): this boot has it loaded")
        else:
            out.append("ok    no line of the steering module in any log: this boot runs without it")
        return good, out
    for rank in range(ranks):
        fields = seen["ranks"].get(rank)
        if fields is None:
            good = False
            out.append(f"FAIL  rank {rank}: no install line of the steering module (is this the right log, from the start "
                       "of the boot, with stderr in it?)")
            continue
        said = (f"layers {fields['layers']} of {fields['stage']}, {fields['hooks']} hooks, {fields['kernels']} extra kernels, "
                f"kernel {fields['kernel']}, replay {fields['replay']}, sidecar {fields['sidecar']}, default scale "
                f"{fields['default_scale']}, debug {fields['debug']}")
        wrong = []
        registered = seen.get("counted", {}).get(rank)
        for key, value in want.items():
            if key in ("sites", "slot"):
                if registered is None:
                    wrong.append(f"{key} was asked for and this module revision does not say it")
                elif str(registered[key]) != str(value):
                    wrong.append(f"{key} is {registered[key]}, wanted {value}")
                continue
            have = fields[key]
            if key == "sidecar":
                matches = have.startswith(value[:len(have)]) and value.startswith(have[:len(value)])
            elif key == "default_scale":
                matches = float(have) == float(value)
            elif key == "replay":
                matches = have == value or have.startswith(value + " ")
            else:
                matches = have == value
            if not matches:
                wrong.append(f"{key.replace('_', ' ')} is {have}, wanted {value}")
        if wrong:
            good = False
            out.append(f"FAIL  rank {rank}: " + "; ".join(wrong) + f"  [{said}]")
        else:
            out.append(f"ok    rank {rank}: {said}")
        if fields["kernel"] == "fused" and fields["kernels"] != fields["hooks"]:
            good = False
            out.append(f"FAIL  rank {rank}: kernel fused with {fields['kernels']} extra kernels for {fields['hooks']} hooks: "
                       "one each is what fused means")
        # A registered hook proves nothing: the module that carries it may never be called.
        if registered is None:
            out.append(f"INFO  rank {rank}: this module revision does not count the operators that ran (revision 5 or "
                       "older). Its hooks are registered; whether they are called is not shown here")
            continue
        done = seen.get("ran", {}).get(rank)
        if done is None:
            good = False
            out.append(f"FAIL  rank {rank}: {registered['operators']} operators registered (sites {registered['sites']}) "
                       "and no line saying they ran. That line comes with the second forward pass: the log ends before "
                       "it, or the engine never got there")
        elif done["operators"] != registered["operators"] or registered["operators"] != fields["hooks"]:
            good = False
            out.append(f"FAIL  rank {rank}: {fields['hooks']} hooks, {registered['operators']} operators registered, "
                       f"{done['operators']} said to have run: three numbers that must be one")
        else:
            out.append(f"ok    rank {rank}: all {done['operators']} operators ran in the first forward pass "
                       f"({done['attn']} attention, {done['shared']} shared-expert, {registered['slot']} of those in the MoE "
                       f"kernel's own slot); sites {registered['sites']}")
    extra = sorted(set(seen["ranks"]) - set(range(ranks)))
    if extra:
        good = False
        out.append(f"FAIL  rank(s) {extra} spoke and --ranks is {ranks}")
    for item in seen["fell_back"]:
        good = False
        out.append(f"FAIL  rank {item['rank']}: an output fell back to the torch path: the fused kernel does not take {item['why']}")
    for message in seen["fatal"]:
        good = False
        out.append(f"FAIL  the module stopped a process: {message}")
    for message in seen.get("never_ran", []):
        good = False
        out.append(f"FAIL  {message} operators never ran: the module refused to serve part of the sidecar")
    return good, out


def selftest() -> int:
    def line(rank: int, kernel: str = "fused", debug: str = "off", hooks: int = 32, kernels: int | None = None,
             replay: str = "off") -> str:
        span, stage = ("4-19", "0-19") if rank == 0 else ("20-36", "20-39")
        return (f"(EngineCore pid=7) [pp2_steering] rank {rank}: steering layers {span} of this stage's {stage}; {hooks} hooks, "
                f"{hooks if kernels is None else kernels} extra kernels per forward pass by op count; kernel {kernel}; "
                f"decoder replay {replay}; sidecar sha256 68551f9b8deca315; default scale 0.0; debug {debug}\n")

    noise = "INFO 11-21 09:00:01 [core.py:120] Initializing a V1 LLM engine\n"
    fused = dict(kernel="fused", debug="off", sidecar="68551f9b8deca315")
    cases = [
        ("both ranks as wanted", [noise + line(0), line(1, hooks=34)], fused, False, True),
        ("the knob never reached the container: torch", [line(0, "torch", kernels=224), line(1, "torch", hooks=34, kernels=238)],
         fused, False, False),
        ("one rank only is fused", [line(0), line(1, "torch", hooks=34, kernels=238)], fused, False, False),
        ("rank 1's log is missing", [line(0)], fused, False, False),
        ("debug was wanted on", [line(0), line(1, hooks=34)], {**fused, "debug": "on"}, False, False),
        ("...and is on", [line(0, debug="on"), line(1, debug="on", hooks=34)], {**fused, "debug": "on"}, False, True),
        ("another sidecar", [line(0), line(1, hooks=34)], {**fused, "sidecar": "0000000000000000"}, False, False),
        ("the sidecar's full sha256 is given", [line(0), line(1, hooks=34)],
         {"sidecar": "68551f9b8deca315" + "a" * 48}, False, True),
        ("an output fell back to the torch path",
         [line(0) + "[pp2_steering] rank 0: the fused kernel does not take a 3-D output; torch path there\n", line(1, hooks=34)],
         fused, False, False),
        ("the module stopped the boot", [line(0), "[pp2_steering] FATAL: sidecar sha256 mismatch\n"], fused, False, False),
        ("replay kept is wanted and said", [line(0, replay="kept (this stage has no replay layers)"),
                                             line(1, hooks=34, replay="kept (17 hooks follow the replay batch)")],
         {"replay": "kept"}, False, True),
        ("a boot without the module", [noise, noise], {}, True, True),
        ("a boot that was meant to be without it", [noise + line(0), line(1, hooks=34)], {}, True, False),
        ("nothing asked: the lines are only read", [line(0, "torch", kernels=224), line(1, "torch", hooks=34, kernels=238)],
         {}, False, True),
        ("a restart inside one log: the last line counts", [line(0, "torch", kernels=224) + line(0), line(1, hooks=34)],
         fused, False, True),
    ]
    failures = 0
    for name, texts, want, no_module, expect in cases:
        good, _ = judge(read(texts), want, 2, no_module)
        failures += good != expect
        print(f"{'ok  ' if good == expect else 'FAIL'}  {name:<50} receipt {'holds' if good else 'does not hold'}")

    # Module revision 6: what was registered, and that all of it ran. One rank, as on a one-Station lane.
    def counted(operators: int = 66, slot: int = 33, sites: str = "attn,shared") -> str:
        return (f"[pp2_steering] rank 0: sites {sites}; {operators} operators, {slot} of them a shared expert edited in "
                "the MoE kernel's own slot\n")

    def ran(operators: int = 66, attn: int = 33, shared: int = 33) -> str:
        return (f"[pp2_steering] rank 0: all {operators} operators ran in the first forward pass ({attn} attention, "
                f"{shared} shared-expert)\n")

    one = ("(EngineCore pid=7) [pp2_steering] rank 0: steering layers 4-36 of this stage's 0-39; 66 hooks, 66 extra kernels "
           "per forward pass by op count; kernel fused; decoder replay kept (the replay layers compact the scale buffer to "
           "their rows); sidecar sha256 68551f9b8deca315; default scale 0.0; debug off\n")
    half = "RuntimeError: steering: 33 of 66 operators did not run in the first forward pass (layer 4 shared, ...)\n"
    for name, text, want, expect in (
        ("one rank, every operator registered and ran", one + counted() + ran(), {**fused, "replay": "kept"}, True),
        ("...with the sites and the slot count asked for", one + counted() + ran(),
         {**fused, "sites": "attn,shared", "slot": "33"}, True),
        ("registered, and no line saying they ran", one + counted(), fused, False),
        ("fewer ran than were registered", one + counted() + ran(operators=33, shared=0), fused, False),
        ("the module refused half a sidecar", one + counted() + half, fused, False),
        ("attention only was wanted, both are on", one + counted() + ran(), {**fused, "sites": "attn"}, False),
        ("the slot was wanted, the modules were hooked", one + counted(slot=0) + ran(), {**fused, "slot": "33"}, False),
        ("an older module is asked for its sites", one, {**fused, "sites": "attn,shared"}, False),
        ("an older module, nothing more asked", one, fused, True),
    ):
        good, _ = judge(read([text]), want, 1, False)
        failures += good != expect
        print(f"{'ok  ' if good == expect else 'FAIL'}  {name:<50} receipt {'holds' if good else 'does not hold'}")
    # ...and once through the command line, files and exit status included.
    work = Path(tempfile.mkdtemp(prefix="boot-receipt-selftest-"))
    (work / "rank0.log").write_text(noise + line(0, "torch", kernels=224))
    (work / "rank1.log").write_text(line(1, "torch", hooks=34, kernels=238))
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        status = main(["--kernel", "fused", str(work / "rank0.log"), str(work / "rank1.log"), "--out", str(work / "r.json")])
    receipt = json.loads((work / "r.json").read_text())
    good = (status == 1 and receipt["holds"] is False and receipt["ranks"]["0"]["kernel"] == "torch"
            and "BOOT-RECEIPT FAIL: asked for kernel fused; 2 log(s) read" in said.getvalue())
    failures += not good
    print(f"{'ok  ' if good else 'FAIL'}  {'from files, with a receipt and an exit status':<50} exit {status}")
    print(f"\nBOOT-RECEIPT-SELFTEST {'PASS' if not failures else 'FAIL'}: {failures} check(s) failed")
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("logs", nargs="*", help="log files, one or more per rank ('-' = standard input)")
    parser.add_argument("--ranks", type=int, default=2, help="ranks that must have printed their line (default 2)")
    parser.add_argument("--kernel", choices=("torch", "fused"))
    parser.add_argument("--debug", choices=("on", "off"))
    parser.add_argument("--replay", choices=("off", "kept"))
    parser.add_argument("--sidecar", help="the sidecar's sha256, or its first 16 hex digits")
    parser.add_argument("--default-scale", type=float)
    parser.add_argument("--sites", help="the sites the module must say it edits, e.g. attn,shared (module revision 6)")
    parser.add_argument("--slot", type=int, help="how many shared-expert operators must go through the MoE kernel's slot")
    parser.add_argument("--no-module", action="store_true", help="the boot is meant to run without the steering module")
    parser.add_argument("--out", help="write the receipt as JSON")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.logs:
        parser.error("give the log files of the ranks")
    want = {key: value for key, value in (("kernel", args.kernel), ("debug", args.debug), ("replay", args.replay),
                                          ("sidecar", args.sidecar), ("default_scale", args.default_scale),
                                          ("sites", args.sites), ("slot", args.slot))
            if value is not None}
    if args.no_module and want:
        parser.error("--no-module goes with no other expectation")
    texts = [sys.stdin.read() if name == "-" else Path(name).read_text(encoding="utf-8", errors="replace")
             for name in args.logs]
    seen = read(texts)
    good, lines = judge(seen, {key: str(value) for key, value in want.items()}, args.ranks, args.no_module)
    for line in lines:
        print(line)
    asked = "no steering module" if args.no_module else (", ".join(f"{k.replace('_', ' ')} {v}" for k, v in want.items())
                                                         or "nothing in particular")
    print(f"BOOT-RECEIPT {'PASS' if good else 'FAIL'}: asked for {asked}; {len(args.logs)} log(s) read")
    if args.out:
        receipt = {"holds": good, "asked": {"no_module": args.no_module, **want}, "ranks": seen["ranks"],
                   "fell_back": seen["fell_back"], "fatal": seen["fatal"], "module_lines": seen["lines"], "logs": args.logs,
                   "registered": seen["counted"], "ran": seen["ran"], "never_ran": seen["never_ran"]}
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2)
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
