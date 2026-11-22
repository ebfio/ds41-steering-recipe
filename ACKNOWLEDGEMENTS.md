# Acknowledgements

## The recipe this is built on

**[`garnermccloud/dgx-station-recipes`](https://github.com/garnermccloud/dgx-station-recipes)** —
copyright 2026 **Garner McCloud**, MIT.

Everything that makes DeepSeek-V4.1-Flash-1017 run on one DGX Station GB300 is his: the MegaMoE
boundary, the W1 attention kernel, the W3 pre-attention path, the W6 combine, the `peer3` expert
tier that puts the cold experts on the RTX PRO 6000, the engram offload, the pinned FlashInfer
tune file, the launcher, the preflight and the benchmark suite. This repository adds one
component and the receipts; it does not vendor his tree, because his tree is his and is better
read at the source.

The measurements in `README.md` that describe *his* recipe (coding, GSM8K, decode, needle) are
his numbers reproduced on different hardware — a second pair of Stations with driver 615.71.09 —
not a claim about his reference box.

## The steering format

- **`NVIDIA/hyperflare-inference` PR #404** (`jcartu`), which defines the published (v1) sidecar:
  per layer one unit direction, one positive FP32 scalar, and one attention row gain, applied at
  the attention output and the shared-expert output.
- **`jcartu/WARMIND-1017-Steering-Sidecar`**, the published sidecar those tensors come from.

The module in `patches/pp2-steering/` reads that format and adds a second, derived one: per site
`read` / `write` vectors plus a row gain, fitted so that applying them at scale 1 reproduces an
abliterated checkpoint's own bytes on its quantisation grid.

## Upstream

NVIDIA DGX Station GB300 · vLLM (`vllm/vllm-openai:v0.31.0`, the tag this recipe pins) ·
FlashInfer · DeepGEMM · Triton · PyTorch.

## Who wrote the port

**Iris, working with Kestrel.** The steering module, the fused kernel, the `fastpath_steer`
component, `apply_to_recipe.py`, the probe / decode_rate / boot_receipt checks and their
selftests, the two upstream fixes, the port assessment and every measurement in `results/` came
out of that pairing, run against two DGX Station GB300s.

## What is ours

The steering module and its fused kernel, the `fastpath_steer` component, `apply_to_recipe.py`,
the probe / decode_rate / boot_receipt checks and their selftests, the two upstream fixes, and
every number in `results/`.
