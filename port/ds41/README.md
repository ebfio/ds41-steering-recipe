# Steering on the one-Station recipe (garnermccloud/dgx-station-recipes)

Status: written and unit-tested on CPU. **It has never run on a GPU or against their lane.**
Nothing here is a measurement.

## What is in this directory

| File | What it is |
|---|---|
| `fastpath_steer.py` | Their 30th component, `steer`. Installs `pp2_steering` when the `vllm` package is first imported, in every process of the container. Prints its ACTIVE line only when the engine has seen every operator run in a forward pass. |
| `apply_to_recipe.py` | Copies three files into a checkout of their repository and makes five exact edits in three of their files. `--check` says which state a checkout is in and changes nothing. |

The steering itself is `patches/pp2-steering/pp2_steering.py` at revision 6 and
`pp2_steering_kernel.py`, the same files the pair would run.

## What revision 6 of the module does there

- **Attention site:** unchanged. A forward hook on `layer.attn`.
- **Shared-expert site:** their lane computes the shared expert inside the MoE kernel, so
  no module is called for it. The kernel writes each expert's result of a row to a slot
  before it adds them up, and the shared expert's is the last slot. A forward hook on the
  MoE module (`layer.ffn`) edits that slot and adds what the edit added to the MoE's
  output, both in place. It runs after everything the MoE launched and before the next
  layer's boundary reads the result. Where the slots are is read from their MegaMoE fork
  (`fastpath_megamoe.c11_layout`), once per buffer, outside graph capture.
- **Nothing of theirs is wrapped or rebuilt.** Their own note warns that wrapping
  `deep_gemm.fp8_fp4_mega_moe` once cost them their MoE path
  (`fastpath/fastpath_engram_side.py:12-14`). This does not wrap it.
- **Every operator is counted.** The second forward pass judges the first; an operator that
  is registered and did not run stops the boot.
- **Decoder replay stays on** (`PP2_STEERING_REPLAY=keep`): the scale buffer joins the list
  of per-row buffers their replay object compacts to the rows it keeps.
- **`PP2_STEERING_SITES`** = `attn,shared` (default), `attn` or `shared`. A boot-time
  setting.

## Using it

```
git clone https://github.com/garnermccloud/dgx-station-recipes theirs      # or a copy of yours
python3 port/ds41/apply_to_recipe.py theirs --check                         # "the upstream recipe"
python3 port/ds41/apply_to_recipe.py theirs                                 # "steered"
```

Then, in `~/.ds41/config.env` or the environment of their launcher:

```
DS41_STEER_SIDECAR=/path/on/the/host/to/the/sidecar.safetensors
DS41_STEER_SHA256=<its sha256>
DS41_STEER_SITES=attn,shared        # optional
DS41_STEER_DEBUG=0                  # optional
```

`serve.sh` refuses to start when `steer` is in the component list and the first two are not
set. To boot their recipe unmodified, use an untouched checkout.

## Receipts

| What | How |
|---|---|
| The module's logic, 33 tests | `python3 patches/pp2-steering/test_pp2_steering.py` |
| Their tree has the seams, and a fused shared expert cannot be missed | `python3 verify/steering_seams.py --vllm-root <v0.31.0 + their overlays> --steering-dir patches/pp2-steering` |
| A boot engaged every operator | `python3 verify/boot_receipt.py --ranks 1 --kernel fused --replay kept --sites attn,shared --slot 33 --sidecar <sha256> <log>` |
| Their own gate counts the component | their `ds41 verify`: `DS41_FASTPATH ACTIVE steer` |

## What a boot has to show, because reading cannot

1. The slot layout their kernel publishes is inside its buffer and has rows of
   2 x 5120 bytes. If not, the first forward stops with "unexpected MoE slot layout".
2. Their capture checks and their log gate hold with two more kernels per steered layer.
3. Scale 0 gives their stock output; scale 1 gives the steered behaviour; rows of
   different scales in one step do not mix.
4. The cost per step.
