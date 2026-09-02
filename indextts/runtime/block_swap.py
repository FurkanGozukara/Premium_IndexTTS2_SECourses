"""Hook-driven H2D-only transformer block streaming.

The selected weights have immutable CPU masters and are copied into a small GPU
ring before each block runs. Biases, normalizations, adapters, and all unselected
tensors remain resident. Training is supported only for frozen streamed weights
and requires gradient checkpointing so backward recomputation reads the correct
ring contents.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
from torch import nn


SwapTensorSelector = Callable[[nn.Module], list[tuple[nn.Module, str]]]


@dataclass
class BlockSwapConfig:
    device: str | torch.device
    supports_backward: bool = False
    use_pinned_memory: bool = True
    ring_size: int = 2
    debug: bool = False
    gradient_checkpointing: bool | None = None

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.supports_backward = bool(self.supports_backward)
        self.use_pinned_memory = bool(self.use_pinned_memory)
        self.ring_size = min(4, max(1, int(self.ring_size)))
        self.debug = bool(self.debug or os.getenv("INDEXTTS_BLOCK_SWAP_DEBUG", "0") == "1")
        if self.supports_backward and self.gradient_checkpointing is False:
            raise ValueError("Block-swap training requires gradient checkpointing")


def compute_h2d_stream_indices(num_blocks: int, blocks_to_swap: int) -> set[int]:
    """Choose evenly-spaced midpoint blocks, matching the Musubi offloader."""

    count = min(max(0, int(blocks_to_swap)), max(0, int(num_blocks)))
    if count == 0:
        return set()
    return {((2 * index + 1) * num_blocks) // (2 * count) for index in range(count)}


def default_swap_tensor_selector(block: nn.Module) -> list[tuple[nn.Module, str]]:
    """Select only matrix weights, including the two mixed-dtype INT8 buffers."""

    result: list[tuple[nn.Module, str]] = []
    seen: set[tuple[int, str]] = set()
    # LoRA/DoRA branches are small trainable resident weights. Their wrapped base
    # matrix is still selected through ``adapter.base``.
    resident_adapter_linears: set[int] = set()
    for candidate in block.modules():
        if candidate.__class__.__name__ == "LoRAAdapter":
            for name in ("lora_A", "lora_B"):
                child = getattr(candidate, name, None)
                if isinstance(child, nn.Module):
                    resident_adapter_linears.add(id(child))
    for module in block.modules():
        if id(module) in resident_adapter_linears:
            continue
        int8_weight = getattr(module, "weight_int8", None)
        int8_scale = getattr(module, "weight_scale", None)
        if isinstance(int8_weight, torch.Tensor) and isinstance(int8_scale, torch.Tensor):
            for name in ("weight_int8", "weight_scale"):
                key = (id(module), name)
                if key not in seen:
                    result.append((module, name))
                    seen.add(key)
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and (
            isinstance(module, nn.Linear) or module.__class__.__name__ == "Conv1D"
        ):
            key = (id(module), "weight")
            if key not in seen:
                result.append((module, "weight"))
                seen.add(key)
    return result


def resolve_blocks_to_swap(
    requested: int,
    num_blocks: int,
    block_bytes: int,
    budget_bytes: int,
    ring_size: int,
) -> int:
    """Resolve an explicit count or the smallest automatic count that fits.

    ``budget_bytes`` is the VRAM available to block matrix weights after the caller
    has subtracted non-block weights, activations, and its reserve. Streamed blocks
    still need up to ``ring_size`` GPU slots.
    """

    blocks = max(0, int(num_blocks))
    if blocks == 0:
        return 0
    try:
        value = int(requested)
    except (TypeError, ValueError, OverflowError):
        value = 0
    if value != -1:
        return min(blocks, max(0, value))
    size = max(0, int(block_bytes))
    if size == 0:
        return 0
    fit_blocks = max(0, int(budget_bytes) // size)
    if fit_blocks >= blocks:
        return 0
    slots = min(blocks, max(1, int(ring_size)))
    # With S swapped blocks, device weight slots are N-S+min(S,R).
    for swapped in range(1, blocks + 1):
        device_blocks = blocks - swapped + min(swapped, slots)
        if device_blocks <= fit_blocks:
            return swapped
    return blocks


def _layout(tensors: Iterable[torch.Tensor]) -> tuple[list[int], int]:
    offsets: list[int] = []
    total = 0
    alignment = 256
    for tensor in tensors:
        total = (total + alignment - 1) // alignment * alignment
        offsets.append(total)
        total += tensor.numel() * tensor.element_size()
    return offsets, total


def _flat_views(
    flat: torch.Tensor,
    templates: list[torch.Tensor],
    layout: tuple[list[int], int],
) -> list[torch.Tensor]:
    offsets, _ = layout
    result = []
    for offset, template in zip(offsets, templates):
        byte_count = template.numel() * template.element_size()
        result.append(flat[offset : offset + byte_count].view(template.dtype).view(template.shape))
    return result


def _is_parameter(module: nn.Module, name: str) -> bool:
    return name in module._parameters and module._parameters[name] is not None


def _set_tensor(module: nn.Module, name: str, tensor: torch.Tensor, is_parameter: bool, requires_grad: bool = False) -> None:
    if is_parameter:
        module._parameters[name] = tensor if isinstance(tensor, nn.Parameter) else nn.Parameter(tensor, requires_grad=requires_grad)
    else:
        if name not in module._buffers:
            raise ValueError(f"{module.__class__.__name__}.{name} is not a registered buffer")
        module._buffers[name] = tensor


class BlockSwapController:
    def __init__(
        self,
        blocks: list[nn.Module],
        blocks_to_swap: int,
        config: BlockSwapConfig,
        swap_tensor_selector: SwapTensorSelector | None = None,
    ):
        self.blocks = list(blocks)
        self.config = config
        self.device = torch.device(config.device)
        self.num_blocks = len(self.blocks)
        self.blocks_to_swap = min(self.num_blocks, max(0, int(blocks_to_swap)))
        self.selector = swap_tensor_selector or default_swap_tensor_selector
        self.stream_indices = sorted(compute_h2d_stream_indices(self.num_blocks, self.blocks_to_swap))
        self._rank = {block_index: rank for rank, block_index in enumerate(self.stream_indices)}
        self.ring_size = min(config.ring_size, max(1, len(self.stream_indices))) if self.stream_indices else 0
        self.forward_only = not config.supports_backward
        self._prepared = False
        self._removed = False
        self._handles: list[Any] = []
        self._jobs: dict[int, list[tuple[nn.Module, str, bool, bool]]] = {}
        self._masters: dict[int, list[torch.Tensor]] = {}
        self._master_flat: dict[int, torch.Tensor] = {}
        self._layouts: dict[int, tuple[list[int], int]] = {}
        self._ring_flat: list[torch.Tensor] = []
        self._ring_views: dict[tuple[int, int], list[torch.Tensor]] = {}
        self._in_slot: list[int | None] = [None] * self.ring_size
        self._free_events: list[torch.cuda.Event | None] = [None] * self.ring_size
        self._ready_events: dict[int, torch.cuda.Event] = {}
        self._copy_stream: torch.cuda.Stream | None = None
        self._pin_method = "disabled"
        self._stats = {"loads": 0, "bytes_h2d": 0, "stalls": 0}
        self._checkpoint_observed = bool(config.gradient_checkpointing)

    def _selected_jobs(self, block_index: int) -> list[tuple[nn.Module, str, bool, bool]]:
        cached = self._jobs.get(block_index)
        if cached is not None:
            return cached
        result: list[tuple[nn.Module, str, bool, bool]] = []
        for module, name in self.selector(self.blocks[block_index]):
            if name not in module._parameters and name not in module._buffers:
                raise ValueError(
                    f"Swap selector returned {module.__class__.__name__}.{name}, which is not a parameter or buffer"
                )
            tensor = getattr(module, name)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Swap tensor {module.__class__.__name__}.{name} is not a tensor")
            parameter = _is_parameter(module, name)
            result.append((module, name, parameter, bool(tensor.requires_grad)))
        self._jobs[block_index] = result
        return result

    def _make_cpu_flat(self, size: int) -> torch.Tensor:
        if self.config.use_pinned_memory and self.device.type == "cuda":
            try:
                result = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
                self._pin_method = "pinned_master"
                return result
            except RuntimeError:
                self._pin_method = "pageable_fallback"
        elif self._pin_method == "disabled":
            self._pin_method = "pageable"
        return torch.empty(size, dtype=torch.uint8, device="cpu")

    def _prepare_streamed_block(self, block_index: int) -> None:
        block = self.blocks[block_index]
        jobs = self._selected_jobs(block_index)
        if not jobs:
            block.to(self.device)
            return
        originals = [getattr(module, name).detach() for module, name, _, _ in jobs]
        if self.config.supports_backward:
            trainable = [name for (_, name, _, requires_grad) in jobs if requires_grad]
            if trainable:
                raise ValueError(
                    "Block swap training requires frozen streamed base weights; found trainable tensors: "
                    + ", ".join(trainable)
                    + ". Freeze the base and train adapters under gradient checkpointing."
                )
        layout = _layout(originals)
        flat = self._make_cpu_flat(layout[1])
        views = _flat_views(flat, originals, layout)
        for view, original in zip(views, originals):
            view.copy_(original, non_blocking=False)

        # Empty placeholders let a CPU-loaded block move its biases and norms without
        # temporarily allocating the large matrix weights on the GPU.
        for module, name, parameter, requires_grad in jobs:
            source = getattr(module, name)
            placeholder = torch.empty(0, dtype=source.dtype, device=self.device)
            _set_tensor(module, name, placeholder, parameter, requires_grad)
        block.to(self.device)

        masters: list[torch.Tensor] = []
        for (module, name, parameter, requires_grad), view in zip(jobs, views):
            master: torch.Tensor
            if parameter:
                master = nn.Parameter(view, requires_grad=requires_grad)
            else:
                master = view
            _set_tensor(module, name, master, parameter, requires_grad)
            masters.append(master)
        self._layouts[block_index] = layout
        self._master_flat[block_index] = flat
        self._masters[block_index] = masters

    def _bind_master(self, block_index: int) -> None:
        for (module, name, parameter, requires_grad), tensor in zip(
            self._selected_jobs(block_index), self._masters[block_index]
        ):
            _set_tensor(module, name, tensor, parameter, requires_grad)

    def _slot_views(self, slot: int, block_index: int) -> list[torch.Tensor]:
        key = (slot, block_index)
        cached = self._ring_views.get(key)
        if cached is not None:
            return cached
        templates = self._masters[block_index]
        raw_views = _flat_views(self._ring_flat[slot], templates, self._layouts[block_index])
        result: list[torch.Tensor] = []
        for (_, _, parameter, _), view in zip(self._selected_jobs(block_index), raw_views):
            result.append(nn.Parameter(view, requires_grad=False) if parameter else view)
        self._ring_views[key] = result
        return result

    def _bind_slot(self, block_index: int, slot: int) -> None:
        for (module, name, parameter, _), tensor in zip(
            self._selected_jobs(block_index), self._slot_views(slot, block_index)
        ):
            _set_tensor(module, name, tensor, parameter, False)

    def _load(self, rank: int, slot: int) -> None:
        block_index = self.stream_indices[rank]
        if self._in_slot[slot] == block_index:
            self._bind_slot(block_index, slot)
            return
        previous = self._in_slot[slot]
        if previous is not None:
            self._bind_master(previous)
        assert self._copy_stream is not None
        with torch.cuda.stream(self._copy_stream):
            free_event = self._free_events[slot]
            if free_event is not None:
                self._copy_stream.wait_event(free_event)
            source = self._master_flat[block_index]
            target = self._ring_flat[slot][: source.numel()]
            target.copy_(source, non_blocking=source.is_pinned())
            ready = torch.cuda.Event()
            ready.record(self._copy_stream)
        self._ready_events[block_index] = ready
        self._bind_slot(block_index, slot)
        self._in_slot[slot] = block_index
        self._stats["loads"] += 1
        self._stats["bytes_h2d"] += source.numel()
        if self.config.debug:
            print(f">> block swap load block={block_index} slot={slot} bytes={source.numel()}")

    def prepare_before_forward(self) -> None:
        if self._removed:
            raise RuntimeError("This block-swap controller has been removed")
        if self._prepared:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            for block_index in self.stream_indices:
                self._bind_master(block_index)
            if self.device.type == "cuda":
                self._ready_events.clear()
                self._free_events = [None] * self.ring_size
                for rank in range(self.ring_size):
                    self._load(rank, rank)
                torch.cuda.synchronize(self.device)
            return

        if not self.stream_indices or self.device.type != "cuda":
            for block in self.blocks:
                block.to(self.device)
            self._prepared = True
            return
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA block swap requested for {self.device}, but CUDA is unavailable")

        self._copy_stream = torch.cuda.Stream(device=self.device)
        for block_index, block in enumerate(self.blocks):
            if block_index in self._rank:
                self._prepare_streamed_block(block_index)
            else:
                block.to(self.device)
        active_indices = [index for index in self.stream_indices if index in self._layouts]
        if len(active_indices) != len(self.stream_indices):
            self.stream_indices = active_indices
            self._rank = {block_index: rank for rank, block_index in enumerate(active_indices)}
            self.blocks_to_swap = len(active_indices)
            self.ring_size = min(self.config.ring_size, len(active_indices)) if active_indices else 0
            self._in_slot = [None] * self.ring_size
            self._free_events = [None] * self.ring_size
        if not active_indices:
            self._prepared = True
            return
        largest = max(self._layouts[index][1] for index in active_indices)
        self._ring_flat = [
            torch.empty(largest, dtype=torch.uint8, device=self.device) for _ in range(self.ring_size)
        ]
        for rank in range(self.ring_size):
            self._load(rank, rank)
        torch.cuda.synchronize(self.device)
        self._prepared = True
        gc.collect()
        torch.cuda.empty_cache()

    # Compatibility with Musubi-integrated model code.
    def prepare_block_devices_before_forward(self, blocks: list[nn.Module] | None = None) -> None:
        del blocks
        self.prepare_before_forward()

    def _wait_for_block(self, block_index: int) -> None:
        rank = self._rank.get(block_index)
        if rank is None or self.device.type != "cuda":
            return
        if self.config.supports_backward and rank == 0:
            if not torch.is_grad_enabled():
                self._checkpoint_observed = True
            elif not self._checkpoint_observed:
                raise RuntimeError(
                    "Block-swap training requires reentrant gradient checkpointing around the managed blocks. "
                    "Without checkpoint recomputation, ring weights would be overwritten before backward."
                )
        slot = rank % self.ring_size
        if self._in_slot[slot] != block_index:
            self._load(rank, slot)
        event = self._ready_events.get(block_index)
        if event is not None:
            if not event.query():
                self._stats["stalls"] += 1
            torch.cuda.current_stream(self.device).wait_event(event)

    def _submit_forward(self, block_index: int) -> None:
        rank = self._rank.get(block_index)
        if rank is None or self.device.type != "cuda":
            return
        slot = rank % self.ring_size
        self._free_events[slot] = torch.cuda.current_stream(self.device).record_event()
        # Reentrant gradient checkpoint recomputation runs with grad enabled and
        # immediately backpropagates through this block. Do not overwrite its
        # saved ring view until the full backward hook marks it consumed.
        if self.config.supports_backward and torch.is_grad_enabled():
            return
        if rank + self.ring_size < len(self.stream_indices):
            next_rank = rank + self.ring_size
            self._load(next_rank, next_rank % self.ring_size)
        elif self.forward_only and rank == len(self.stream_indices) - 1 and len(self.stream_indices) > self.ring_size:
            for first_rank in range(self.ring_size):
                self._load(first_rank, first_rank)

    def _backward_hook(self, block_index: int):
        prefetch = block_index in self._rank
        previous = block_index - 1
        wait_previous = previous in self._rank
        if not prefetch and not wait_previous:
            return None

        def hook(module, grad_input, grad_output):
            del module, grad_input, grad_output
            if prefetch and self.device.type == "cuda":
                rank = self._rank[block_index]
                slot = rank % self.ring_size
                self._free_events[slot] = torch.cuda.current_stream(self.device).record_event()
                if rank - self.ring_size >= 0:
                    previous_rank = rank - self.ring_size
                    self._load(previous_rank, previous_rank % self.ring_size)
            if wait_previous:
                self._wait_for_block(previous)
            return None

        return hook

    def attach_hooks(self) -> None:
        if self._handles or not self.stream_indices or self.device.type != "cuda":
            return
        if not self._prepared:
            self.prepare_before_forward()
        for index, block in enumerate(self.blocks):
            self._handles.append(block.register_forward_pre_hook(
                lambda module, args, block_index=index: self._wait_for_block(block_index)
            ))
            self._handles.append(block.register_forward_hook(
                lambda module, args, output, block_index=index: self._submit_forward(block_index)
            ))
            if self.config.supports_backward:
                backward = self._backward_hook(index)
                if backward is not None:
                    self._handles.append(block.register_full_backward_hook(backward))

    def set_forward_only(self, flag: bool) -> None:
        if self.device.type == "cuda" and self._copy_stream is not None:
            self._copy_stream.synchronize()
        if not flag and not self.config.supports_backward:
            raise ValueError("This block-swap controller was created without backward support")
        self.forward_only = bool(flag)

    def remove(self, to_cpu: bool = False) -> None:
        if self._removed:
            return
        if self.device.type == "cuda" and self._prepared:
            torch.cuda.synchronize(self.device)
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        restore_device = torch.device("cpu") if to_cpu else self.device
        if to_cpu and self._prepared:
            for block in self.blocks:
                block.to("cpu")
        if self._prepared and self.stream_indices and self.device.type == "cuda":
            for block_index in self.stream_indices:
                jobs = self._selected_jobs(block_index)
                for (module, name, parameter, requires_grad), master in zip(jobs, self._masters[block_index]):
                    restored = master.detach().to(restore_device)
                    _set_tensor(module, name, restored, parameter, requires_grad)
        self._ring_views.clear()
        self._ring_flat.clear()
        self._ready_events.clear()
        self._master_flat.clear()
        self._masters.clear()
        self._removed = True
        if self.device.type == "cuda" and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    def summary(self) -> dict[str, Any]:
        swapped_bytes = sum(tensor.numel() for tensor in self._master_flat.values())
        ring_bytes = sum(tensor.numel() for tensor in self._ring_flat)
        swapped = len(self.stream_indices) if self.device.type == "cuda" else 0
        return {
            "total_blocks": self.num_blocks,
            "resident_blocks": self.num_blocks - swapped,
            "swapped_blocks": swapped,
            "resident_count": self.num_blocks - swapped,
            "swapped_count": swapped,
            "stream_indices": list(self.stream_indices),
            "swapped_bytes": swapped_bytes,
            "ring_bytes": ring_bytes,
            "pin_method": self._pin_method,
            "ring_size": self.ring_size,
            "supports_backward": self.config.supports_backward,
        }

    def stats(self) -> dict[str, int]:
        return dict(self._stats)


def enable_block_swap(
    blocks: list[nn.Module],
    blocks_to_swap: int,
    config: BlockSwapConfig,
    swap_tensor_selector: SwapTensorSelector | None = None,
) -> BlockSwapController:
    block_list = list(blocks)
    requested = int(blocks_to_swap)
    if requested == -1:
        selector = swap_tensor_selector or default_swap_tensor_selector
        block_bytes = 0
        if block_list:
            block_bytes = sum(
                tensor.numel() * tensor.element_size()
                for module, name in selector(block_list[0])
                if isinstance((tensor := getattr(module, name, None)), torch.Tensor)
            )
        budget = 0
        device = torch.device(config.device)
        if device.type == "cuda" and torch.cuda.is_available():
            budget = torch.cuda.mem_get_info(device)[0]
        requested = resolve_blocks_to_swap(-1, len(block_list), block_bytes, budget, config.ring_size)
    controller = BlockSwapController(block_list, requested, config, swap_tensor_selector)
    controller.prepare_before_forward()
    controller.attach_hooks()
    return controller


__all__ = [
    "BlockSwapConfig",
    "BlockSwapController",
    "compute_h2d_stream_indices",
    "default_swap_tensor_selector",
    "enable_block_swap",
    "resolve_blocks_to_swap",
]
