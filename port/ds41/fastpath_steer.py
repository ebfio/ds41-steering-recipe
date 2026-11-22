"""ds41 fastpath component "steer": per-request activation steering on this lane.

Not part of garnermccloud/dgx-station-recipes. Added by port/ds41/apply_to_recipe.py of
ebfio/deepseek-v41-flash-pp2-pair, next to pp2_steering.py and pp2_steering_kernel.py.

One set of stock weights serves stock and steered behaviour: a request carries
`"vllm_xargs": {"steering_scale": 0.0..1.0}` (or `steering_scale` at the top level), and
two outputs of every steered layer are edited by that much: the attention module's and
the shared expert's. On this lane the shared expert is computed inside the MegaMoE
kernel, so its edit is made in the kernel's own slot, by a hook on the MoE module that
runs after everything the MoE launched and before the next boundary reads the result.
Rows at scale 0 are not written anywhere. pp2_steering.py says the rest.

What this file does: it runs pp2_steering.install() as soon as the vllm package has been
imported, in every process of the container, and prints this component's ACTIVE line
only when the engine has seen every one of its operators run in a forward pass. A
sidecar that is missing, has another sha256, or half of whose operators never run stops
the boot.

Container environment, set by serve.sh from DS41_STEER_* when "steer" is in the list:
  PP2_STEERING_SIDECAR, PP2_STEERING_SHA256   the sidecar and its sha256 (required)
  PP2_STEERING_KERNEL=fused                   one kernel per edited output
  PP2_STEERING_REPLAY=keep                    the #58132 replay stays on
  PP2_STEERING_SITES                          attn,shared | attn | shared
  PP2_STEERING_DEBUG                          0 | 1
"""
from __future__ import annotations

import sys

from fastpath_hook import activate, register

NAME = "steer"


def _log(message: str) -> None:
    sys.stderr.write(f"DS41_FASTPATH {NAME}: {message}\n")
    sys.stderr.flush()


@register(NAME, "vllm", defer_active=True)
def _install(module) -> None:
    import pp2_steering  # next to this file, in the mounted fastpath directory

    if pp2_steering.SLOT_LAYOUT is not pp2_steering.recipe_slot_layout:
        raise RuntimeError("steer: pp2_steering.SLOT_LAYOUT was replaced before the component installed")
    config = pp2_steering.install()
    if config.replay != "keep" or config.kernel != "fused":
        raise RuntimeError(
            f"steer: this lane needs PP2_STEERING_REPLAY=keep and PP2_STEERING_KERNEL=fused, got "
            f"{config.replay} and {config.kernel}"
        )
    pp2_steering.ON_VERIFIED.append(lambda: activate(NAME))
    _log(
        f"installed: sites {','.join(config.sites)}, kernel {config.kernel}, replay {config.replay}, "
        f"sidecar sha256 {config.sha256[:16]}, debug {'on' if config.debug else 'off'}. "
        "ACTIVE follows when every operator has run in a forward pass"
    )
