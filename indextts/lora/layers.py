"""LoRA and DoRA adapter layers for the IndexTTS GPT projections."""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover - transformers is an IndexTTS dependency.
    Conv1D = None  # type: ignore[assignment,misc]


def _is_int8_base(module: nn.Module) -> bool:
    """Use the quantized layer's public protocol without importing it eagerly."""

    return (
        callable(getattr(module, "dequantize_weight", None))
        and hasattr(module, "in_features")
        and hasattr(module, "out_features")
    )


def _invalidate_adapter_cache_after_load(
    module: nn.Module, incompatible_keys: object
) -> None:
    del incompatible_keys
    if isinstance(module, LoRAAdapter):
        module.invalidate_cache()


_TRITON_FUSED_LORA: Callable[..., torch.Tensor] | bool | None = None
_TRITON_FUSED_BASE_LORA: Callable[..., torch.Tensor] | bool | None = None


def _optional_fused_lora() -> Callable[..., torch.Tensor] | None:
    """Resolve the optional Triton kernel once, retaining the eager fallback."""

    global _TRITON_FUSED_LORA
    if _TRITON_FUSED_LORA is False:
        return None
    if _TRITON_FUSED_LORA is None:
        if os.getenv("INDEXTTS_LORA_TRITON", "1") == "0":
            _TRITON_FUSED_LORA = False
            return None
        try:
            from ._triton_fused import fused_lora
        except (ImportError, OSError):
            _TRITON_FUSED_LORA = False
            return None
        _TRITON_FUSED_LORA = fused_lora
    return _TRITON_FUSED_LORA


def _optional_fused_base_lora() -> Callable[..., torch.Tensor] | None:
    """Resolve the optional fused base-plus-adapter kernel once."""

    global _TRITON_FUSED_BASE_LORA
    if _TRITON_FUSED_BASE_LORA is False:
        return None
    if _TRITON_FUSED_BASE_LORA is None:
        if os.getenv("INDEXTTS_LORA_TRITON", "1") == "0":
            _TRITON_FUSED_BASE_LORA = False
            return None
        try:
            from ._triton_fused import fused_base_lora
        except (ImportError, OSError):
            _TRITON_FUSED_BASE_LORA = False
            return None
        _TRITON_FUSED_BASE_LORA = fused_base_lora
    return _TRITON_FUSED_BASE_LORA


class LoRAAdapter(nn.Module):
    """Wrap a linear-like base module and add a LoRA or DoRA branch.

    Adapter parameters default to fp32. When a containing model is converted with
    ``model.bfloat16()`` or ``model.to(dtype=...)``, this wrapper lets the base
    follow that conversion while retaining the adapter parameter dtype. Calling a
    dtype conversion directly on the adapter is treated as an explicit request and
    converts both the base and adapter parameters.

    DoRA strength uses ``m_s = 1 + strength * (m - 1)``, where
    ``m = magnitude / ||W + delta||``. Its effective weight is
    ``m_s * (W + strength * delta)``. Thus strength zero is exactly the base,
    while strength one is the PEFT DoRA formula.
    """

    def __init__(
        self,
        base: nn.Module,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_dora: bool = False,
        init: str = "kaiming",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Module):
            raise TypeError("base must be a torch.nn.Module")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not math.isfinite(float(alpha)):
            raise ValueError(f"alpha must be finite, got {alpha}")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if not dtype.is_floating_point:
            raise TypeError(f"adapter dtype must be floating point, got {dtype}")

        self.base = base
        self.in_features, self.out_features, self._base_kind = self._base_shape(base)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.use_dora = bool(use_dora)
        self._strength = 1.0
        self._enabled = True
        self._merged = False
        self._pre_merge_enabled: bool | None = None
        self._original_base_weight: torch.Tensor | None = None
        self._fused_base_cache_ready = False
        self._fused_adapter_cache_ready = False
        self._fused_weight_stride_n = 0
        self._fused_weight_stride_k = 0
        self._allow_adapter_dtype_cast = False
        self._base_requires_grad = {
            name: parameter.requires_grad for name, parameter in base.named_parameters()
        }
        self._base_training = base.training
        self._base_training_ste = getattr(base, "training_ste", None)

        device = self._base_device(base)
        self.lora_A = nn.Linear(
            self.in_features,
            self.rank,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.lora_B = nn.Linear(
            self.rank,
            self.out_features,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.lora_dropout = nn.Dropout(float(dropout))

        # These derived values are runtime-only.  In particular, keeping them out
        # of the state dict preserves the existing adapter file format.
        self.register_buffer("_effective_scale_cache", None, persistent=False)
        self.register_buffer("_inference_lora_A_cache", None, persistent=False)
        self.register_buffer("_inference_lora_B_cache", None, persistent=False)
        self.register_buffer("_inference_scale_cache", None, persistent=False)
        self.register_buffer("_inference_bias_correction_cache", None, persistent=False)

        if init == "kaiming":
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        elif init == "gaussian":
            nn.init.normal_(self.lora_A.weight, std=1.0 / self.rank)
        elif init == "zeros":
            nn.init.zeros_(self.lora_A.weight)
        else:
            raise ValueError(
                f"unsupported LoRA initialization {init!r}; expected kaiming, gaussian, or zeros"
            )
        nn.init.zeros_(self.lora_B.weight)

        if self.use_dora:
            with torch.no_grad():
                magnitude = torch.linalg.vector_norm(
                    self.base_weight_linear().to(device=device, dtype=torch.float32), dim=1
                )
            self.lora_magnitude = nn.Parameter(magnitude.to(dtype=dtype))
        else:
            self.register_parameter("lora_magnitude", None)

        # The optimizer must never receive the wrapped base parameters.
        self.base.requires_grad_(False)
        self.register_load_state_dict_post_hook(_invalidate_adapter_cache_after_load)

    @property
    def strength(self) -> float:
        return self._strength

    @strength.setter
    def strength(self, value: float) -> None:
        resolved = float(value)
        changed = resolved != getattr(self, "_strength", resolved)
        self._strength = resolved
        if changed:
            self.invalidate_cache()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        resolved = bool(value)
        changed = resolved != getattr(self, "_enabled", resolved)
        self._enabled = resolved
        if changed:
            self.invalidate_cache()

    def invalidate_cache(self) -> None:
        """Discard all values derived from the base or adapter weights.

        The cache is value-based rather than storage-identity-based. Block swap
        may rebind base and adapter tensors to value-identical storage without
        invalidating it. Training/eval transitions, strength/enabled changes,
        state-dict loads, and adapter hot swaps invalidate it explicitly.
        Call this method after any other in-place weight-value modification.
        """

        buffers = getattr(self, "_buffers", {})
        self._fused_base_cache_ready = False
        self._fused_adapter_cache_ready = False
        for name in (
            "_effective_scale_cache",
            "_inference_lora_A_cache",
            "_inference_lora_B_cache",
            "_inference_scale_cache",
            "_inference_bias_correction_cache",
        ):
            if name in buffers:
                buffers[name] = None

    def train(self, mode: bool = True):
        previous = self.training
        result = super().train(mode)
        if previous != self.training:
            self.invalidate_cache()
        return result

    @staticmethod
    def _base_shape(base: nn.Module) -> tuple[int, int, str]:
        if isinstance(base, nn.Linear):
            return int(base.in_features), int(base.out_features), "linear"
        if Conv1D is not None and isinstance(base, Conv1D):
            return int(base.nx), int(base.nf), "conv1d"
        if _is_int8_base(base):
            return int(base.in_features), int(base.out_features), "int8"
        raise TypeError(
            "LoRAAdapter supports nn.Linear, transformers Conv1D, and "
            "ConvRotInt8Linear-compatible modules"
        )

    @staticmethod
    def _base_device(base: nn.Module) -> torch.device:
        for tensor in base.parameters(recurse=True):
            return tensor.device
        for tensor in base.buffers(recurse=True):
            return tensor.device
        return torch.device("cpu")

    def base_weight_linear(self) -> torch.Tensor:
        """Return the base weight in standard ``[out_features, in_features]`` layout."""

        if self._base_kind == "linear":
            weight = self.base.weight
        elif self._base_kind == "conv1d":
            weight = self.base.weight.transpose(0, 1)
        else:
            weight = self.base.dequantize_weight()
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise RuntimeError(
                "base weight has shape "
                f"{tuple(weight.shape)}, expected {(self.out_features, self.in_features)}"
            )
        return weight

    def delta_weight(self) -> torch.Tensor:
        """Return ``scaling * B @ A`` in standard linear weight layout."""

        return self.scaling * torch.matmul(self.lora_B.weight, self.lora_A.weight)

    def _adapter_forward_training(self, x: torch.Tensor) -> torch.Tensor:
        adapter_input = self.lora_dropout(x)
        adapter_input = adapter_input.to(
            device=self.lora_A.weight.device, dtype=self.lora_A.weight.dtype
        )
        return self.scaling * self.lora_B(self.lora_A(adapter_input))

    def _base_projection(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        projection_input = x.to(device=weight.device, dtype=weight.dtype)
        return F.linear(projection_input, weight)

    def _base_bias(self) -> torch.Tensor | None:
        bias = getattr(self.base, "bias", None)
        return bias if isinstance(bias, torch.Tensor) else None

    def _compute_effective_scale(self, strength: float) -> torch.Tensor:
        delta = self.delta_weight()
        weight = self.base_weight_linear().to(device=delta.device, dtype=delta.dtype)
        # PEFT treats the weight norm as a fixed normalization term. In
        # particular, no A/B gradient flows through this norm calculation.
        weight_norm = torch.linalg.vector_norm(weight + delta.detach(), dim=1).detach()
        magnitude_scale = self.lora_magnitude / weight_norm
        return 1.0 + strength * (magnitude_scale - 1.0)

    @torch.no_grad()
    def _cached_effective_scale(self, strength: float) -> torch.Tensor:
        cached = self._effective_scale_cache
        if cached is not None:
            return cached

        device = self.lora_A.weight.device
        if cached is None:
            # Norms are intentionally calculated and retained in fp32 even if an
            # adapter was explicitly cast to a lower precision.
            delta = self.scaling * torch.matmul(
                self.lora_B.weight.to(dtype=torch.float32),
                self.lora_A.weight.to(dtype=torch.float32),
            )
            weight = self.base_weight_linear().to(device=device, dtype=torch.float32)
            weight_norm = torch.linalg.vector_norm(weight + delta, dim=1)
            magnitude_scale = self.lora_magnitude.to(dtype=torch.float32) / weight_norm
            cached = (1.0 + strength * (magnitude_scale - 1.0)).detach()
            self._effective_scale_cache = cached
        return cached

    @staticmethod
    def _cache_matches(
        tensor: torch.Tensor | None, *, device: torch.device, dtype: torch.dtype
    ) -> bool:
        return tensor is not None and tensor.device == device and tensor.dtype == dtype

    @torch.no_grad()
    def _prepare_inference_cache(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        effective_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # Device/dtype transforms go through _apply(), which invalidates these
        # buffers. Block swap only rebinds value-identical storage, so this is the
        # deliberately tiny steady-state check used by every decoded token.
        if self._cache_matches(
            self._inference_lora_A_cache, device=device, dtype=dtype
        ) and self._cache_matches(
            self._inference_lora_B_cache, device=device, dtype=dtype
        ):
            return (
                self._inference_lora_A_cache,
                self._inference_lora_B_cache,
                self._inference_scale_cache,
                self._inference_bias_correction_cache,
            )

        if not dtype.is_floating_point:
            raise TypeError(f"base output must be floating point, got {dtype}")

        a_cached = self.lora_A.weight.detach().to(
            device=device, dtype=dtype, copy=True
        )
        row_multiplier: torch.Tensor | float = self.scaling * float(self.strength)
        if effective_scale is not None:
            row_multiplier = effective_scale.to(device=device, dtype=torch.float32) * float(
                row_multiplier
            )
            scale_cached = effective_scale.to(device=device, dtype=dtype)
        else:
            scale_cached = None
        bias_correction = None

        b_fp32 = self.lora_B.weight.detach().to(device=device, dtype=torch.float32)
        if isinstance(row_multiplier, torch.Tensor):
            b_fp32 = b_fp32 * row_multiplier.unsqueeze(1)
        else:
            b_fp32 = b_fp32 * row_multiplier
        b_cached = b_fp32.to(dtype=dtype)

        bias = self._base_bias()
        if effective_scale is not None and bias is not None:
            bias_value = bias.detach().to(device=device, dtype=dtype)
            bias_correction = bias_value * (1.0 - scale_cached)
        else:
            bias_correction = None

        self._inference_lora_A_cache = a_cached
        self._inference_lora_B_cache = b_cached
        self._inference_scale_cache = scale_cached
        self._inference_bias_correction_cache = bias_correction
        return a_cached, b_cached, scale_cached, bias_correction

    @staticmethod
    def _autocast_context(device: torch.device, dtype: torch.dtype):
        enabled = (device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)) or (
            device.type == "cpu" and dtype == torch.bfloat16
        )
        if enabled:
            return torch.autocast(device_type=device.type, dtype=dtype)
        return nullcontext()

    def _forward_fused_base_inference(
        self, x: torch.Tensor, strength: float
    ) -> torch.Tensor | None:
        """Fuse a token-sized floating base projection with its rank-32 branch."""

        if (
            torch.is_grad_enabled()
            or self._base_kind not in ("linear", "conv1d")
            or x.device.type != "cuda"
            or x.dtype not in (torch.float16, torch.bfloat16)
            or not x.is_contiguous()
            or x.shape[-1] != self.in_features
            or self.rank != 32
            or (self.lora_dropout.training and self.lora_dropout.p != 0.0)
        ):
            return None

        rows = x.numel() // self.in_features
        if rows < 1 or rows > 3:
            return None
        weight = getattr(self.base, "weight", None)
        if not isinstance(weight, torch.Tensor):
            return None
        bias = self._base_bias()
        if (
            weight.device != x.device
            or weight.dtype != x.dtype
            or (bias is not None and (bias.device != x.device or bias.dtype != x.dtype))
        ):
            return None

        fused_base_lora = _optional_fused_base_lora()
        if fused_base_lora is None:
            return None
        effective_scale = (
            self._cached_effective_scale(strength) if self.use_dora else None
        )
        a_weight, b_weight, scale, _ = self._prepare_inference_cache(
            device=x.device,
            dtype=x.dtype,
            effective_scale=effective_scale,
        )
        if self._base_kind == "linear":
            weight_stride_n = weight.stride(0)
            weight_stride_k = weight.stride(1)
        else:
            weight_stride_n = weight.stride(1)
            weight_stride_k = weight.stride(0)
        try:
            result = fused_base_lora(
                x,
                weight,
                bias,
                a_weight,
                b_weight,
                scale,
                weight_stride_n=weight_stride_n,
                weight_stride_k=weight_stride_k,
            )
            self._fused_weight_stride_n = weight_stride_n
            self._fused_weight_stride_k = weight_stride_k
            self._fused_base_cache_ready = True
            return result
        except (RuntimeError, ValueError):
            # A portable eager implementation remains available on CUDA stacks
            # where Triton imports successfully but cannot compile this kernel.
            global _TRITON_FUSED_BASE_LORA
            _TRITON_FUSED_BASE_LORA = False
            return None

    def _forward_inference(
        self,
        x: torch.Tensor,
        base_result: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        effective_scale = (
            self._cached_effective_scale(strength) if self.use_dora else None
        )
        a_weight, b_weight, scale, bias_correction = self._prepare_inference_cache(
            device=base_result.device,
            dtype=base_result.dtype,
            effective_scale=effective_scale,
        )

        adapter_input = self.lora_dropout(x)
        rows = base_result.numel() // self.out_features
        use_fused_kernel = (
            not torch.is_grad_enabled()
            and base_result.device.type == "cuda"
            and base_result.dtype in (torch.float16, torch.bfloat16)
            and adapter_input.dtype == base_result.dtype
            and adapter_input.is_contiguous()
            and base_result.is_contiguous()
            and self.rank == 32
            and rows <= 3
        )
        if use_fused_kernel:
            fused_lora = _optional_fused_lora()
            if fused_lora is not None:
                try:
                    result = fused_lora(
                        adapter_input,
                        a_weight,
                        b_weight,
                        base_result,
                        scale,
                        bias_correction,
                    )
                    self._fused_adapter_cache_ready = (
                        not self.lora_dropout.training or self.lora_dropout.p == 0.0
                    )
                    return result
                except (RuntimeError, ValueError):
                    # Triton is optional (and not available on every supported
                    # platform/toolchain); an eager path remains fully portable.
                    global _TRITON_FUSED_LORA
                    _TRITON_FUSED_LORA = False

        with self._autocast_context(base_result.device, base_result.dtype):
            rank_result = F.linear(adapter_input, a_weight)
            if scale is None:
                scaled_base = base_result
            else:
                bias = self._base_bias()
                if bias is None:
                    # base_projection is base_result when there is no bias.
                    scaled_base = base_result * scale
                else:
                    # Recovering base_projection = base_result - bias avoids a
                    # second x @ W.T. This rearrangement performs the scale and
                    # bias restoration in one pointwise kernel:
                    # scale * base_projection + bias
                    #   == scale * base_result + bias * (1 - scale).
                    scaled_base = torch.addcmul(bias_correction, base_result, scale)

            result = torch.addmm(
                scaled_base.reshape(rows, self.out_features),
                rank_result.reshape(rows, self.rank),
                b_weight.transpose(0, 1),
            )
        return result.reshape(base_result.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._fused_base_cache_ready
            and not torch.is_grad_enabled()
            and x.is_contiguous()
            and x.device == self._inference_lora_A_cache.device
            and x.dtype == self._inference_lora_A_cache.dtype
            and x.shape[-1] == self.in_features
            and 0 < x.numel() <= 3 * self.in_features
        ):
            # Block swapping may replace the value-identical base weight object,
            # so the pointer is read anew while all derived values stay cached.
            return _TRITON_FUSED_BASE_LORA(  # type: ignore[operator]
                x,
                self.base.weight,
                self.base.bias,
                self._inference_lora_A_cache,
                self._inference_lora_B_cache,
                self._inference_scale_cache,
                weight_stride_n=self._fused_weight_stride_n,
                weight_stride_k=self._fused_weight_stride_k,
            )
        if (
            self._fused_adapter_cache_ready
            and not torch.is_grad_enabled()
            and x.is_contiguous()
            and x.device == self._inference_lora_A_cache.device
            and x.dtype == self._inference_lora_A_cache.dtype
            and x.shape[-1] == self.in_features
            and 0 < x.numel() <= 3 * self.in_features
        ):
            base_result = self.base(x)
            if (
                base_result.is_contiguous()
                and base_result.device == self._inference_lora_B_cache.device
                and base_result.dtype == self._inference_lora_B_cache.dtype
            ):
                return _TRITON_FUSED_LORA(  # type: ignore[operator]
                    x,
                    self._inference_lora_A_cache,
                    self._inference_lora_B_cache,
                    base_result,
                    self._inference_scale_cache,
                    self._inference_bias_correction_cache,
                )
            self._fused_adapter_cache_ready = False

        strength = float(self.strength)
        active = self.enabled and strength != 0.0
        inference = not torch.is_grad_enabled() or not self.training
        if active and inference:
            fused_result = self._forward_fused_base_inference(x, strength)
            if fused_result is not None:
                return fused_result

        base_result = self.base(x)
        if not active:
            return base_result
        if inference:
            return self._forward_inference(x, base_result, strength)

        lora_result = self._adapter_forward_training(x)
        if not self.use_dora:
            return base_result + (strength * lora_result).to(
                device=base_result.device, dtype=base_result.dtype
            )

        effective_scale = self._compute_effective_scale(strength)
        weight = self.base_weight_linear().to(
            device=effective_scale.device, dtype=effective_scale.dtype
        )

        base_projection = self._base_projection(x, weight)
        correction = (effective_scale - 1.0) * base_projection
        correction = correction + effective_scale * (strength * lora_result)
        return base_result + correction.to(device=base_result.device, dtype=base_result.dtype)

    @torch.no_grad()
    def merge_into_base(self) -> None:
        """Fold the active adapter into an fp linear/Conv1D base and disable it."""

        if self._base_kind == "int8":
            raise TypeError("LoRA/DoRA cannot be merged directly into an int8 base")
        if self._merged:
            return

        delta = self.delta_weight()
        weight = self.base_weight_linear().to(device=delta.device, dtype=delta.dtype)
        strength = float(self.strength) if self.enabled else 0.0
        if self.use_dora:
            weight_norm = torch.linalg.vector_norm(weight + delta.detach(), dim=1)
            magnitude_scale = self.lora_magnitude / weight_norm
            effective_scale = 1.0 + strength * (magnitude_scale - 1.0)
            merged = effective_scale.unsqueeze(1) * (weight + strength * delta)
        else:
            merged = weight + strength * delta

        if self._base_kind == "linear":
            destination = self.base.weight
            source = merged
        else:
            destination = self.base.weight
            source = merged.transpose(0, 1)
        self._original_base_weight = destination.detach().to(
            device="cpu", copy=True
        )
        self._pre_merge_enabled = self.enabled
        destination.copy_(source.to(device=destination.device, dtype=destination.dtype))
        self.enabled = False
        self._merged = True

    @torch.no_grad()
    def unmerge_from_base(self) -> None:
        """Restore the exact pre-merge floating base weight and re-enable the adapter."""

        if not self._merged:
            return
        if self._original_base_weight is None:
            raise RuntimeError("cannot unmerge because the original base weight is unavailable")
        destination = self.base.weight
        destination.copy_(
            self._original_base_weight.to(
                device=destination.device, dtype=destination.dtype
            )
        )
        enabled = bool(self._pre_merge_enabled)
        self._merged = False
        self._pre_merge_enabled = None
        self._original_base_weight = None
        self._enabled = enabled
        self.invalidate_cache()

    def state_dict_keys(self) -> list[str]:
        keys = ["lora_A.weight", "lora_B.weight"]
        if self.use_dora:
            keys.append("lora_magnitude")
        return keys

    def restore_base_requires_grad(self) -> None:
        """Restore base flags captured before the module was wrapped."""

        for name, parameter in self.base.named_parameters():
            if name in self._base_requires_grad:
                parameter.requires_grad_(self._base_requires_grad[name])
        self.base.train(self._base_training)
        if self._base_training_ste is not None and hasattr(self.base, "training_ste"):
            self.base.training_ste = self._base_training_ste

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True):
        protected_dtypes: dict[str, torch.dtype] = {}
        if not self._allow_adapter_dtype_cast:
            for name, parameter in self._named_adapter_parameters():
                if parameter.is_floating_point():
                    protected_dtypes[name] = parameter.dtype

        result = super()._apply(fn, recurse=recurse)
        if protected_dtypes:
            current = dict(self._named_adapter_parameters())
            with torch.no_grad():
                for name, original_dtype in protected_dtypes.items():
                    parameter = current[name]
                    if parameter.dtype != original_dtype:
                        parameter.data = parameter.data.to(dtype=original_dtype)
                        if parameter.grad is not None:
                            parameter.grad.data = parameter.grad.data.to(dtype=original_dtype)
        # A dtype/device transform may round either the base or an adapter value.
        # Rebuild all derived values lazily on the next inference call.
        self.invalidate_cache()
        return result

    def _named_adapter_parameters(self):
        yield "lora_A.weight", self.lora_A.weight
        yield "lora_B.weight", self.lora_B.weight
        if self.lora_magnitude is not None:
            yield "lora_magnitude", self.lora_magnitude

    def _explicit_dtype_apply(self, operation: Callable[[], LoRAAdapter]):
        previous = self._allow_adapter_dtype_cast
        self._allow_adapter_dtype_cast = True
        try:
            return operation()
        finally:
            self._allow_adapter_dtype_cast = previous

    def to(self, *args, **kwargs):
        _, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is None:
            return super().to(*args, **kwargs)
        return self._explicit_dtype_apply(lambda: super(LoRAAdapter, self).to(*args, **kwargs))

    def bfloat16(self):
        return self._explicit_dtype_apply(lambda: super(LoRAAdapter, self).bfloat16())

    def half(self):
        return self._explicit_dtype_apply(lambda: super(LoRAAdapter, self).half())

    def float(self):
        return self._explicit_dtype_apply(lambda: super(LoRAAdapter, self).float())

    def double(self):
        return self._explicit_dtype_apply(lambda: super(LoRAAdapter, self).double())


__all__ = ["LoRAAdapter"]
