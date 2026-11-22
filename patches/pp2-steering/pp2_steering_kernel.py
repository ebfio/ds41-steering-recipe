"""One fused kernel per steered output, for PP2_STEERING_KERNEL=fused.

The torch path in pp2_steering.steer() costs 6 kernels per output without a row gain
and 8 with one, on every forward pass, whether any row is steered or not. This kernel
does the same edit in one launch, in place:

    x' = (x + s * write * (read . x)) * (1 + s * gain_delta)        per row, s its scale

One program per row. A row whose scale is 0 returns before it reads the row, so stock
rows are not written at all: the same guarantee as the torch path's select, at none of
its cost. The dot product and the edit are FP32, the store rounds to the tensor's dtype
(round to nearest even, what torch's cast does). The sum is taken in a different order
than torch's matmul, so a steered value can differ from the torch path's by one unit in
the last place of the output dtype; stock rows cannot differ.

Compiled here for sm_90, sm_100, sm_103 and sm_120 and run in Triton's interpreter on a
CPU (test_pp2_steering.py). It has never run on a GPU: verify/steering_kernel_check.py
is that test, and the module only uses this file when PP2_STEERING_KERNEL=fused.

steer_slot_rows is the same edit for a shared expert that is computed inside the MoE
kernel (a MegaMoE backend). There the expert's output stands alone only in the kernel's
own buffer, one slot among those the kernel sums. The kernel edits that slot in place
and adds what the edit added to the MoE's output as well, so the result is the same
whether the MoE kernel has already done the sum or a later reader of the slots does it.
It has run in the interpreter only.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def steer_rows(
    x_ptr,  # [rows, H], BF16 or FP32, edited in place
    read_ptr,  # FP32[H]
    write_ptr,  # FP32[H]
    gain_ptr,  # FP32[H], row gain minus 1; unread when HAS_GAIN is false
    scale_ptr,  # FP32[rows]
    row_stride,
    H,
    HAS_GAIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    s = tl.load(scale_ptr + row)
    if s == 0.0:
        return
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    read = tl.load(read_ptr + cols, mask=mask, other=0.0)
    write = tl.load(write_ptr + cols, mask=mask, other=0.0)
    along = s * tl.sum(x * read, axis=0)
    y = x + along * write
    if HAS_GAIN:
        gain = tl.load(gain_ptr + cols, mask=mask, other=0.0)
        y = y + (s * y) * gain
    tl.store(x_ptr + row * row_stride + cols, y.to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def steer_slot_rows(
    slot_ptr,  # [rows, H], the shared expert's output in the MoE kernel's buffer, edited in place
    total_ptr,  # [rows, H], the MoE's output, which gets what the edit added
    read_ptr,  # FP32[H]
    write_ptr,  # FP32[H]
    gain_ptr,  # FP32[H], row gain minus 1; unread when HAS_GAIN is false
    scale_ptr,  # FP32[rows]
    slot_stride,
    total_stride,
    H,
    HAS_GAIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    s = tl.load(scale_ptr + row)
    if s == 0.0:
        return
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    x = tl.load(slot_ptr + row * slot_stride + cols, mask=mask, other=0.0).to(tl.float32)
    read = tl.load(read_ptr + cols, mask=mask, other=0.0)
    write = tl.load(write_ptr + cols, mask=mask, other=0.0)
    along = s * tl.sum(x * read, axis=0)
    y = x + along * write
    if HAS_GAIN:
        gain = tl.load(gain_ptr + cols, mask=mask, other=0.0)
        y = y + (s * y) * gain
    total = tl.load(total_ptr + row * total_stride + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(slot_ptr + row * slot_stride + cols, y.to(slot_ptr.dtype.element_ty), mask=mask)
    tl.store(total_ptr + row * total_stride + cols, (total + (y - x)).to(total_ptr.dtype.element_ty), mask=mask)


def block_size(hidden: int) -> int:
    return triton.next_power_of_2(hidden)


def launch(output, read, write, gain_delta, token_scale) -> None:
    """Edit `output` [rows, H] in place. `token_scale` is FP32 with one value per row."""
    hidden = output.shape[1]
    steer_rows[(output.shape[0],)](
        output,
        read,
        write,
        read if gain_delta is None else gain_delta,
        token_scale,
        output.stride(0),
        hidden,
        HAS_GAIN=gain_delta is not None,
        BLOCK=block_size(hidden),
    )


def warm(output, read, write, gain_delta, token_scale) -> None:
    """Compile the variant these arguments select without launching it, so the first
    launch does not compile inside a CUDA graph capture."""
    hidden = output.shape[1]
    steer_rows.warmup(
        output,
        read,
        write,
        read if gain_delta is None else gain_delta,
        token_scale,
        output.stride(0),
        hidden,
        HAS_GAIN=gain_delta is not None,
        BLOCK=block_size(hidden),
        grid=(1,),
    )


def _slot_args(slot, total, read, write, gain_delta, token_scale):
    hidden = slot.shape[1]
    return (
        slot,
        total,
        read,
        write,
        read if gain_delta is None else gain_delta,
        token_scale,
        slot.stride(0),
        total.stride(0),
        hidden,
    ), dict(HAS_GAIN=gain_delta is not None, BLOCK=block_size(hidden))


def launch_slot(slot, total, read, write, gain_delta, token_scale) -> None:
    """Edit `slot` [rows, H] in place and add what the edit added to `total` [rows, H]."""
    args, constants = _slot_args(slot, total, read, write, gain_delta, token_scale)
    steer_slot_rows[(slot.shape[0],)](*args, **constants)


def warm_slot(slot, total, read, write, gain_delta, token_scale) -> None:
    """Compile the slot variant these arguments select without launching it."""
    args, constants = _slot_args(slot, total, read, write, gain_delta, token_scale)
    steer_slot_rows.warmup(*args, **constants, grid=(1,))
