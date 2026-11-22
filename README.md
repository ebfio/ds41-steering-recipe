# Per-request abliteration on one DGX Station

One engine, one copy of the weights, two names: a stock lane and an abliterated lane, chosen
per request. Built on [`garnermccloud/dgx-station-recipes`](https://github.com/garnermccloud/dgx-station-recipes)
at commit `8043a68` — their recipe, unmodified, plus a steering component.

This repository is the port and the receipts. It does **not** vendor their tree: clone theirs,
apply the port, run the checks.

```
deepseek-v4.1-flash       no body field                → stock weights
deepseek-v4.1-unlimited   vllm_xargs.steering_scale 1  → abliterated behaviour
```

Same prompt, same engine, temperature 0:

```
scale 0.0 → "I can't provide a script for brute-forcing a phone lock screen…"
scale 1.0 → "# Brute-Force Script for a 4-Digit Phone Lock Screen  Here's a Python script…"
```

## What the port is

| Piece | What it does |
|---|---|
| `patches/pp2-steering/pp2_steering.py` | the steering module: a sidecar of per-layer directions applied at two sites (the attention output and the shared-expert output), keyed by a per-token scale |
| `patches/pp2-steering/pp2_steering_kernel.py` | the fused Triton path for the same edit |
| `port/ds41/fastpath_steer.py` | their 30th fastpath component: registers the module, counts the operators that actually ran, and refuses to serve if a registered operator never fired |
| `port/ds41/apply_to_recipe.py` | the five edits it makes to their tree, each matched against their text at `8043a68` and refused if the text has moved |
| `patches/ds41-upstream-fixes/` | two one-line fixes for their tree, found by running it |

`apply_to_recipe.py <their checkout> --check` says which of three states a tree is in, and
`--steering-dir` points at the module. It accepts either their repository root or the recipe
directory itself, because config-management tools often stage the recipe flat.

## Why it is safe on the stock name

Scale 0 has to be *their* model, not something close to it. Measured against a second Station
running their recipe with no module at all:

```
T1b   largest distance 0 (noise 0), same text on 8/8, and 0 at the first generated token
T2    scale 1 moves 8/8 prompts, including 4/4 refusal prompts
T4    no cross-scale leak in mixed batches: 46/341 vs 51/339 at scale 0, 59/340 vs 73/361 at
      scale 1, 67 sites, smallest p = 0.755, 0 of 48 whole answers crossed
T7    validation: 1.5 / -0.1 / 'abc' / top-level 2 → 400; 0 and 1 → 200
```

And the module is provably loaded, not merely present: `vllm_xargs.steering_scale: 7` returns
`400 steering_scale must be a number in [0, 1], got 7`. A stock lane accepts 7 and ignores it.
That single request is the cheapest post-start check there is.

## What it costs

Steering is not measurable at these shapes. `decode_rate.py`, one Station, thinking off,
`--train-prompt`, 512 tokens, three runs, median:

| | scale 0 | scale 1 |
|---|---|---|
| C1 tok/s | 608.1 | 646.3 |
| C8 aggregate | 2,020.1 | 2,000.2 |
| C32 aggregate | 4,243.5 | 4,160.9 |
| C1 ms/round | 7.41 | 7.50 |

The scale-1 C1 figure is *higher* because the steered text accepts more drafts per round
(4.86 against 4.52), not because the engine got faster. KV cache costs about 1,700 tokens:
11,853,879 steered against 11,855,600 unsteered.

## Their recipe, measured here

One Station, their defaults, 1,048,576 context, driver 615.71.09:

```
coding prompt, greedy, thinking off   780.5 tok/s   (their published 724.7 = 108%)
GSM8K, 200 questions, thinking on     193/200 = 96.5%, 0 runaways
decode.py  C1 499 (107%)  C4 964 (97%)  C8 1,363 (100%)  C16 1,978 (100%)
needle     3/3 at 607,431 prompt tokens and 3/3 at 997,546
memory guard PASS, 10.4 GiB headroom, KV cache 11.85M tokens
```

Against a two-Station pipeline pair on the same model, with the settings side by side (same
prompt, 512 tokens, temperature 0, top_p 0.95):

| | pair, two Stations | this recipe, one Station |
|---|---|---|
| 1 stream | 447.0 | 625.9 |
| 8 streams, unlike prompts | ~1,140 | 1,003.4 |
| 32 streams, unlike prompts | ~2,550 | 1,620.6 |

One Station matches the pair at one stream and is ahead of it; the pair still wins at 32. The
comparison is not like-for-like on drafting: the pair drafted 7 tokens a round at every size,
their recipe drafts 5 up to four streams and 3 above.

## The traps, in the order they cost us boots

1. **The first start dies; the second fits.** With no tuning decisions on disk the RTX sidecar
   runs out of memory moving the last layer to the card (layer 38 of 40, 726 MiB short). It
   leaves its tuning decisions behind and the next start serves. The cold start allocates
   1.0 GB more from layer 0 onward — `call()` grows scratch to the largest candidate the tuning
   tries and never shrinks it. One boot, do not debug it.
2. **Never rewrite the tree under a running lane.** The containers mount it, and Docker
   restarts them by itself. A play that starts the engine before patching the recipe serves
   **stock under the steered name** and nothing fails: both ids answer, both refuse, and the
   module's absence is invisible. Patch, assert, then start.
3. **`preflight.sh boot` fails a healthy serving lane.** It greps the live log for
   `No tuned config covers`, which appears the first time a 257–512-token prompt arrives (their
   pinned tune file is missing two `(K, N)` rows). A fresh boot passes; a busy lane does not.
4. **`ds41 install` preflights before it renders its config**, so `DS41_MODEL_DIR` must be in
   the environment or it reports `need 500 GiB` on a box whose weights are complete.
5. **The API key must be owned by the lane's user.** Created by root, their installer's `chmod`
   fails `Operation not permitted` and install exits non-zero.
6. **`DS41_ALIASES` must be quoted.** The config file is sourced, so an unquoted alias list
   runs `DeepSeek-V4.1-Flash: command not found`.
7. **Per-file staging wedges.** Ansible `copy: src=recipe/` over their 791-file tree wrote
   nothing in ten minutes at 0.6 % CPU; one tarball moves the same 8 MB in 2 s.
8. **A sidecar crash is survivable but not graceful.** `kill -9` on the sidecar: Docker brings
   it back in 55 s (`unless-stopped`), the engine then computes exactly what it did before
   (4/4 identical), but a request issued during the outage hangs — it does not return a
   degraded answer, it does not return. The engine serialises behind it.

## Reproducing

```sh
git clone https://github.com/garnermccloud/dgx-station-recipes.git theirs
git -C theirs checkout 8043a68
cp -r patches/pp2-steering port theirs/deepseek-v4.1-flash/
python3 theirs/deepseek-v4.1-flash/port/ds41/apply_to_recipe.py theirs/deepseek-v4.1-flash --check
python3 theirs/deepseek-v4.1-flash/port/ds41/apply_to_recipe.py theirs/deepseek-v4.1-flash
# then their ./ds41 install --no-model, ./ds41 up, and:
python3 verify/boot_receipt.py --ranks 1 --kernel fused --debug off --replay kept \
    --sites attn,shared --slot 33 --sidecar <sha256> boot.log
python3 verify/steering_probe.py --url http://127.0.0.1:8001 --record-baseline base.json   # on a second, unsteered Station
python3 verify/steering_probe.py --url http://127.0.0.1:8001 --baseline base.json --refusal-prompts refusal4.txt
python3 verify/decode_rate.py --url http://127.0.0.1:8001 --train-prompt --tokens 512 --streams 1,8,32 --scale 1
```

Every number above is in `results/` as the receipt that produced it.

## Credits

Built on **[`garnermccloud/dgx-station-recipes`](https://github.com/garnermccloud/dgx-station-recipes)**
by **Garner McCloud** (MIT) — the MegaMoE boundary, the W1/W3/W6 kernels, the `peer3` RTX expert
tier, the engram offload, the launcher and the checks are all his. The steering sidecar format is
from **`NVIDIA/hyperflare-inference` PR #404** (`jcartu`), with the published vectors from
**`jcartu/WARMIND-1017-Steering-Sidecar`**. See `ACKNOWLEDGEMENTS.md`.

## Licence

The port (`port/`, `patches/`) is MIT. Their recipe and kernels keep their own licence — clone
theirs, do not copy it from here.
