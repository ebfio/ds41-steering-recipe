#!/usr/bin/env python3
"""Add the steering component to a checkout of garnermccloud/dgx-station-recipes.

    apply_to_recipe.py <their checkout> [--check]

Work on a copy or a branch of their repository: this edits files in it.

It copies three files into deepseek-v4.1-flash/fastpath/

    pp2_steering.py, pp2_steering_kernel.py    from patches/pp2-steering/ of this repository
    fastpath_steer.py                          from this directory

and makes five edits in three of their files. Each edit replaces text exactly as it was
read at their commit 8043a68; when one of them is not found exactly once, nothing at all
is written.

    fastpath/components.py   import fastpath_steer, and name it in REGISTERED
    config/recipe.env        "steer" first in DS41_FASTPATH_COMPONENTS (their last three
                             components wrap the graph capture in a fixed order, the
                             memory guard last; first keeps out of that)
    serve/serve.sh           when "steer" is in the list: the sidecar is mounted and the
                             PP2_STEERING_* environment is set from DS41_STEER_SIDECAR,
                             DS41_STEER_SHA256, DS41_STEER_SITES and DS41_STEER_DEBUG;
                             a missing sidecar or sha256 stops the start

--check changes nothing and says which of the three states the checkout is in: theirs,
with steering, or neither. Exit status 0 for the first two.

To boot their recipe unmodified, use their checkout as it is. To take steering out of a
patched one, `git checkout` the three files and delete the three copies.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEERING = HERE.parent.parent / "patches" / "pp2-steering"
THEIR_COMMIT = "8043a6811a5feda70fdfb6e827eebb2ec25c65d1"
RECIPE = "deepseek-v4.1-flash"
COPIES = (
    (STEERING / "pp2_steering.py", "fastpath/pp2_steering.py"),
    (STEERING / "pp2_steering_kernel.py", "fastpath/pp2_steering_kernel.py"),
    (HERE / "fastpath_steer.py", "fastpath/fastpath_steer.py"),
)

SERVE_BLOCK_OLD = "  local fastpath=$DS41_FASTPATH_COMPONENTS\n"
SERVE_BLOCK_NEW = SERVE_BLOCK_OLD + r"""  # steer (not part of the upstream recipe): per-request activation steering, fastpath_steer.py.
  local steer_mounts=() steer_envs=()
  case ",$fastpath," in *,steer,*)
    [ -n "${DS41_STEER_SIDECAR:-}" ] || die "the steer component is in the list and DS41_STEER_SIDECAR is not set (the steering sidecar on this host)"
    [ -f "$DS41_STEER_SIDECAR" ] || die "steering sidecar not found at $DS41_STEER_SIDECAR"
    [ -n "${DS41_STEER_SHA256:-}" ] || die "DS41_STEER_SHA256 is not set (the sha256 of $DS41_STEER_SIDECAR)"
    steer_mounts=(-v "$DS41_STEER_SIDECAR:/steering/sidecar.safetensors:ro")
    steer_envs=(-e PP2_STEERING_SIDECAR=/steering/sidecar.safetensors -e "PP2_STEERING_SHA256=$DS41_STEER_SHA256"
                -e PP2_STEERING_KERNEL=fused -e PP2_STEERING_REPLAY=keep
                -e "PP2_STEERING_SITES=${DS41_STEER_SITES:-attn,shared}" -e "PP2_STEERING_DEBUG=${DS41_STEER_DEBUG:-0}")
    log "steering: sidecar $DS41_STEER_SIDECAR, sites ${DS41_STEER_SITES:-attn,shared}, debug ${DS41_STEER_DEBUG:-0}"
  ;; esac
"""

EDITS = (
    (
        "fastpath/components.py",
        "import fastpath_a12_memory_guard\n",
        "import fastpath_a12_memory_guard\nimport fastpath_steer  # not upstream: ebfio/deepseek-v41-flash-pp2-pair, port/ds41\n",
    ),
    (
        "fastpath/components.py",
        "    fastpath_a12_memory_guard,\n)\n",
        "    fastpath_a12_memory_guard,\n    fastpath_steer,\n)\n",
    ),
    (
        "config/recipe.env",
        "DS41_FASTPATH_COMPONENTS='tail_markov_fused,",
        "DS41_FASTPATH_COMPONENTS='steer,tail_markov_fused,",
    ),
    ("serve/serve.sh", SERVE_BLOCK_OLD, SERVE_BLOCK_NEW),
    (
        "serve/serve.sh",
        '--env-file "$envf" "${mounts[@]}" "${envs[@]}" --entrypoint',
        '--env-file "$envf" "${mounts[@]}" "${steer_mounts[@]}" "${envs[@]}" "${steer_envs[@]}" --entrypoint',
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state(recipe: Path) -> tuple[str, list[str]]:
    """("theirs" | "steered" | "neither", what was found)."""
    notes, old, new = [], 0, 0
    texts = {rel: (recipe / rel).read_text(encoding="utf-8") for rel in {rel for rel, _, _ in EDITS}}
    for rel, before, after in EDITS:
        text = texts[rel]
        if text.count(after) == 1:
            new += 1
        elif text.count(before) == 1:
            old += 1
        else:
            notes.append(f"{rel}: neither the upstream text nor the edited one is there exactly once: {before.strip()[:60]!r}")
    copies = [rel for _, rel in COPIES if (recipe / rel).is_file()]
    stale = [rel for source, rel in COPIES if (recipe / rel).is_file() and sha256(recipe / rel) != sha256(source)]
    if stale:
        notes.append("copied files that differ from this repository's: " + ", ".join(stale))
    if old == len(EDITS) and not copies and not notes:
        return "theirs", notes
    if new == len(EDITS) and len(copies) == len(COPIES) and not notes:
        return "steered", notes
    if not notes:
        notes.append(f"{old} of {len(EDITS)} edits are still upstream text, {new} are applied, "
                     f"{len(copies)} of {len(COPIES)} files are copied")
    return "neither", notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkout", type=Path, help="a checkout of garnermccloud/dgx-station-recipes")
    parser.add_argument("--check", action="store_true", help="say what state the checkout is in; change nothing")
    args = parser.parse_args()
    recipe = args.checkout / RECIPE
    for rel in sorted({rel for rel, _, _ in EDITS}):
        if not (recipe / rel).is_file():
            print(f"FAIL  {recipe / rel} is missing: is {args.checkout} their repository?")
            return 1
    for source, _ in COPIES:
        if not source.is_file():
            print(f"FAIL  {source} is missing from this repository")
            return 1
    head = subprocess.run(["git", "-C", str(args.checkout), "rev-parse", "HEAD"], capture_output=True, text=True)
    commit = head.stdout.strip() if head.returncode == 0 else ""
    if commit == THEIR_COMMIT:
        print(f"ok    their repository is at {commit[:7]}, the commit the edits were read on")
    else:
        print(f"WARN  their repository is at {commit[:12] or 'no git commit'}, not {THEIR_COMMIT[:7]}: "
              "the edits apply only where the text is unchanged, and the code around them was not re-read")
    found, notes = state(recipe)
    for note in notes:
        print(f"      {note}")
    if args.check or found != "theirs":
        said = {"theirs": "the upstream recipe, no steering", "steered": "steering applied, files current",
                "neither": "NEITHER: do not boot this"}[found]
        print(f"{'ok  ' if found != 'neither' else 'FAIL'}  the checkout is: {said}")
        if not args.check and found == "steered":
            print("      nothing to do")
        return 0 if found != "neither" else 1
    texts = {rel: (recipe / rel).read_text(encoding="utf-8") for rel in {rel for rel, _, _ in EDITS}}
    for rel, before, after in EDITS:
        texts[rel] = texts[rel].replace(before, after)
    for rel, text in texts.items():
        (recipe / rel).write_text(text, encoding="utf-8")
        print(f"ok    edited {RECIPE}/{rel}")
    for source, rel in COPIES:
        shutil.copyfile(source, recipe / rel)
        print(f"ok    copied {RECIPE}/{rel}  sha256 {sha256(recipe / rel)}")
    found, notes = state(recipe)
    print(f"{'ok  ' if found == 'steered' else 'FAIL'}  the checkout is now: {found}")
    return 0 if found == "steered" else 1


if __name__ == "__main__":
    raise SystemExit(main())
