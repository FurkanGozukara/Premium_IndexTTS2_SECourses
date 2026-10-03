"""PyTorch Whisper runtime for INT8 ConvRot checkpoints.

The same runtime also runs plain FP16 OpenAI weights (``Linear`` without a
scale), which is how the engine is validated against CTranslate2 before any
quantization is involved.

Layout conventions (OpenAI names): encoder ``conv1``/``conv2``/``blocks.i``/
``ln_post``; decoder ``token_embedding``/``positional_embedding``/``blocks.i``/
``ln``. Quantized linears follow ComfyUI's native INT8 ConvRot format:
``<p>.weight`` int8 (rotated), ``<p>.weight_scale`` fp32 per output channel and
``<p>.comfy_quant`` JSON.

Ported from Whisper-WebUI Premium (SECourses), modules/whisper/convrot at revision 613beba (September 2026).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F

from . import decode_attention as DA
from . import kernels as K
from .cuda_graph import CudaGraph

try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
except Exception:  # flash-attn is optional in this app: PyTorch SDPA runs the same attention
    flash_attn_func = None
    flash_attn_with_kvcache = None


def _raw_flash_attn_forward():
    """The body of flash-attn's ``_flash_attn_forward`` op, or None when this flash-attn has another layout.

    flash_attn_func reaches the kernel through the op's dispatcher wrapper, whose first call imports
    torch._dynamo and registers the distributed tensor ops: about 8 seconds added to the first transcription of
    every process. Calling the body runs the same kernel with the same arguments, so the results are the same.
    """
    expected = ["q", "k", "v", "dropout_p", "softmax_scale", "causal", "window_size_left", "window_size_right",
                "softcap", "alibi_slopes", "return_softmax"]
    try:
        import inspect

        from flash_attn.flash_attn_interface import _flash_attn_forward

        forward = getattr(_flash_attn_forward, "_init_fn", None)
        if forward is not None and list(inspect.signature(forward).parameters) == expected:
            return forward
    except Exception:
        pass
    return None


_FLASH_ATTN_FORWARD = _raw_flash_attn_forward() if flash_attn_func is not None else None


@dataclass
class WhisperDims:
    n_mels: int
    n_vocab: int
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 32
    n_text_ctx: int = 448
    n_text_state: int = 1280
    n_text_head: int = 20
    n_text_layer: int = 32

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _move(t: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    """Move a tensor. CPU copies use ordinary (pageable) RAM: pinned copies took about 530 MB more for
    large-v3 (the pinned allocator rounds sizes up) and stayed locked in PyTorch's pinned cache after the model
    went back to the GPU, for about 70 ms faster reloads."""
    if t is None:
        return None
    if device.type == "cpu":
        return t.to("cpu") if t.is_cuda else t
    return t.to(device, non_blocking=True)


class LayerNormParams:
    __slots__ = ("weight", "bias", "eps")

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
        self.weight = weight.float().contiguous()
        self.bias = bias.float().contiguous()
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.float(), (x.shape[-1],), self.weight, self.bias, self.eps).to(x.dtype)

    def to(self, device: torch.device):
        self.weight = _move(self.weight, device)
        self.bias = _move(self.bias, device)


class Linear:
    """Linear layer that is either INT8 ConvRot (``scale`` set) or plain FP16."""

    __slots__ = ("weight", "scale", "bias", "quantized", "out_features", "in_features", "group_scales",
                 "weight_only")

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None, scale: torch.Tensor | None = None):
        self.weight = weight.contiguous()
        self.bias = bias.contiguous() if bias is not None else None
        self.scale = scale.reshape(-1).float().contiguous() if scale is not None else None
        self.quantized = scale is not None
        self.out_features, self.in_features = weight.shape
        # Dynamic activation scales: one per row (False) or one per row and 256-group (True).
        self.group_scales = False
        # Weight-only INT8 (rotated fp16 activations, no activation quantization).
        self.weight_only = False

    def to(self, device: torch.device):
        self.weight = _move(self.weight, device)
        self.bias = _move(self.bias, device)
        self.scale = _move(self.scale, device)

    def __call__(self, x: torch.Tensor, ln: LayerNormParams | None = None, pre_gelu: bool = False,
                 gelu: bool = False, residual: torch.Tensor | None = None, out: torch.Tensor | None = None):
        if self.quantized:
            if self.weight_only:
                if ln is not None:
                    xr = K.rotate_act(x, K.PRE_LAYERNORM, ln.weight, ln.bias, ln.eps)
                else:
                    xr = K.rotate_act(x, K.PRE_GELU if pre_gelu else K.PRE_NONE)
                return K.w8a16_linear(xr, self.weight, self.scale, bias=self.bias, gelu=gelu,
                                      residual=residual, out=out)
            if ln is not None:
                xq, xs = K.rotate_quantize(x, K.PRE_LAYERNORM, ln.weight, ln.bias, ln.eps,
                                           group_scales=self.group_scales)
            else:
                xq, xs = K.rotate_quantize(x, K.PRE_GELU if pre_gelu else K.PRE_NONE,
                                           group_scales=self.group_scales)
            return K.int8_linear_q(xq, xs, self.weight, self.scale, bias=self.bias, gelu=gelu,
                                   residual=residual, out=out, out_dtype=x.dtype)
        if ln is not None:
            x = ln(x)
        elif pre_gelu:
            x = F.gelu(x)
        y = F.linear(x, self.weight, self.bias)
        if gelu:
            y = F.gelu(y)
        if residual is not None:
            if out is not None and out.data_ptr() == residual.data_ptr():
                out.add_(y)
                return out
            y = y + residual
        if out is not None:
            out.copy_(y)
            return out
        return y


def _cat_linear(parts: list[Linear]) -> Linear:
    """Concatenate linears that share an input (rows of the weight)."""
    weight = torch.cat([p.weight for p in parts], dim=0)
    if all(p.bias is None for p in parts):
        bias = None
    else:
        bias = torch.cat([
            p.bias if p.bias is not None else torch.zeros(
                p.out_features, dtype=next(q.bias.dtype for q in parts if q.bias is not None), device=p.weight.device)
            for p in parts
        ])
    scale = torch.cat([p.scale for p in parts]) if parts[0].quantized else None
    return Linear(weight, bias, scale)


class EncoderLayer:
    __slots__ = ("attn_ln", "qkv", "out", "mlp_ln", "fc1", "fc2")


class DecoderLayer:
    __slots__ = ("attn_ln", "qkv", "out", "cross_ln", "cross_q", "cross_kv", "cross_out", "mlp_ln", "fc1", "fc2")


QUANT_FORMAT = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": K.CONVROT_GROUP}


def _attention(q, k, v, causal: bool = False):
    """[B, T, H, D] attention: flash-attn for fp16/bf16, SDPA otherwise (fp32 reference runs)."""
    if q.dtype in (torch.float16, torch.bfloat16) and flash_attn_func is not None:
        if _FLASH_ATTN_FORWARD is not None and q.shape[-1] % 8 == 0:
            # flash_attn_func's defaults: no dropout, scale 1/sqrt(head_dim), no window, no softcap
            return _FLASH_ATTN_FORWARD(q, k, v, 0.0, q.shape[-1] ** -0.5, causal, -1, -1, 0.0, None, False)[0]
        return flash_attn_func(q, k, v, causal=causal)
    if not causal and DA.supported(q, k):
        # A decode step's cross-attention (a few queries against the encoder frames): split-KV Triton kernels,
        # about a sixth of the time PyTorch's general attention kernel took.
        return DA.decode_attention(q, k, v)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal)
    return out.transpose(1, 2)


def _kv_cache_attention(q, k_cache, v_cache, k, v, cache_seqlens, cache_batch_idx=None, num_splits: int = 0):
    """``flash_attn_with_kvcache(..., causal=True)``: write ``k``/``v`` [B, P, H, D] into the cache rows at
    ``cache_seqlens`` and attend over each row's cache up to them.

    Without flash-attn, PyTorch SDPA does the same: a decoding step (P = 1) attends over the whole cache with a
    mask of every row's length, so the shapes stay fixed and the step is captured in the decode CUDA graph like
    the flash-attn kernel; a prompt (P > 1) is written into empty rows (``cache_seqlens`` 0, the only way the
    engine fills a cache) and attends causally over itself.
    """
    if flash_attn_with_kvcache is not None:
        extra = {"num_splits": num_splits} if num_splits else {}
        if cache_batch_idx is not None:
            extra["cache_batch_idx"] = cache_batch_idx
        return flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens, causal=True,
                                       **extra)
    batch, length = q.shape[:2]
    rows = cache_batch_idx.long() if cache_batch_idx is not None else torch.arange(batch, device=q.device)
    if length > 1:
        k_cache[rows, :length] = k
        v_cache[rows, :length] = v
        return _attention(q, k, v, causal=True)
    whole = cache_batch_idx is None and batch == k_cache.shape[0]
    if whole and DA.supported(q, k_cache):
        # Each row reads only its own cache length, as flash_attn_with_kvcache does.
        DA.write_kv(k_cache, v_cache, k[:, 0], v[:, 0], cache_seqlens)
        return DA.decode_attention(q, k_cache, v_cache, cache_seqlens, length_offset=1)
    positions = cache_seqlens.long()
    k_cache[rows, positions] = k[:, 0]
    v_cache[rows, positions] = v[:, 0]
    keys = k_cache if whole else k_cache[rows]
    values = v_cache if whole else v_cache[rows]
    mask = torch.arange(k_cache.shape[1], device=q.device)[None, :] <= positions[:, None]
    out = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                                         attn_mask=mask[:, None, None, :])
    return out.transpose(1, 2)


def _load_linear(tensors: dict, prefix: str, device, dtype=torch.float16) -> Linear:
    weight = tensors[prefix + ".weight"]
    bias = tensors.get(prefix + ".bias")
    scale = tensors.get(prefix + ".weight_scale")
    if scale is not None:
        if weight.dtype != torch.int8:
            raise ValueError(f"{prefix}: weight_scale present but weight dtype is {weight.dtype}")
        quant = tensors.get(prefix + ".comfy_quant")
        if quant is not None:
            meta = json.loads(bytes(quant.to(torch.uint8).tolist()).decode("utf-8"))
            if not meta.get("convrot") or int(meta.get("convrot_groupsize", 0)) != K.CONVROT_GROUP:
                raise ValueError(f"{prefix}: unsupported quantization metadata {meta}")
        return Linear(weight.to(device), bias.to(device, torch.float16) if bias is not None else None, scale.to(device))
    return Linear(weight.to(device, dtype), bias.to(device, dtype) if bias is not None else None)


class WhisperRuntime:
    """Encoder + decoder weights and the forward passes used by the engine."""

    def __init__(self, tensors: dict, dims: WhisperDims, device: str = "cuda", dtype: torch.dtype = torch.float16,
                 group_scales_encoder: bool = False, group_scales_decoder: bool = True,
                 decoder_weight_only: bool = False):
        self.dims = dims
        self.device = torch.device(device)
        self.dtype = dtype
        dt = dtype
        dev = self.device
        self.n_head = dims.n_text_head
        self.head_dim = dims.n_text_state // dims.n_text_head
        self.d_model = dims.n_text_state

        self.conv1_w = tensors["encoder.conv1.weight"].to(dev, dt).contiguous()
        self.conv1_b = tensors["encoder.conv1.bias"].to(dev, dt).contiguous()
        self.conv2_w = tensors["encoder.conv2.weight"].to(dev, dt).contiguous()
        self.conv2_b = tensors["encoder.conv2.bias"].to(dev, dt).contiguous()
        self.enc_pos = tensors["encoder.positional_embedding"].to(dev, dt).contiguous()
        self.ln_post = LayerNormParams(tensors["encoder.ln_post.weight"].to(dev), tensors["encoder.ln_post.bias"].to(dev))

        self.enc_layers = []
        for i in range(dims.n_audio_layer):
            p = f"encoder.blocks.{i}."
            layer = EncoderLayer()
            layer.attn_ln = LayerNormParams(tensors[p + "attn_ln.weight"].to(dev), tensors[p + "attn_ln.bias"].to(dev))
            layer.qkv = _cat_linear([_load_linear(tensors, p + "attn." + n, dev, dt) for n in ("query", "key", "value")])
            layer.out = _load_linear(tensors, p + "attn.out", dev, dt)
            layer.mlp_ln = LayerNormParams(tensors[p + "mlp_ln.weight"].to(dev), tensors[p + "mlp_ln.bias"].to(dev))
            layer.fc1 = _load_linear(tensors, p + "mlp.0", dev, dt)
            layer.fc2 = _load_linear(tensors, p + "mlp.2", dev, dt)
            self.enc_layers.append(layer)

        self.tok_emb = tensors["decoder.token_embedding.weight"].to(dev, dt).contiguous()
        self.dec_pos = tensors["decoder.positional_embedding"].to(dev, dt).contiguous()
        self.dec_ln = LayerNormParams(tensors["decoder.ln.weight"].to(dev), tensors["decoder.ln.bias"].to(dev))
        self.dec_layers = []
        for i in range(dims.n_text_layer):
            p = f"decoder.blocks.{i}."
            layer = DecoderLayer()
            layer.attn_ln = LayerNormParams(tensors[p + "attn_ln.weight"].to(dev), tensors[p + "attn_ln.bias"].to(dev))
            layer.qkv = _cat_linear([_load_linear(tensors, p + "attn." + n, dev, dt) for n in ("query", "key", "value")])
            layer.out = _load_linear(tensors, p + "attn.out", dev, dt)
            layer.cross_ln = LayerNormParams(tensors[p + "cross_attn_ln.weight"].to(dev),
                                             tensors[p + "cross_attn_ln.bias"].to(dev))
            layer.cross_q = _load_linear(tensors, p + "cross_attn.query", dev, dt)
            layer.cross_kv = _cat_linear([_load_linear(tensors, p + "cross_attn." + n, dev, dt) for n in ("key", "value")])
            layer.cross_out = _load_linear(tensors, p + "cross_attn.out", dev, dt)
            layer.mlp_ln = LayerNormParams(tensors[p + "mlp_ln.weight"].to(dev), tensors[p + "mlp_ln.bias"].to(dev))
            layer.fc1 = _load_linear(tensors, p + "mlp.0", dev, dt)
            layer.fc2 = _load_linear(tensors, p + "mlp.2", dev, dt)
            self.dec_layers.append(layer)
        self._cross_buffers: dict[tuple, torch.Tensor] = {}
        self.set_group_scales(group_scales_encoder, group_scales_decoder)
        self.set_decoder_weight_only(decoder_weight_only)

    def set_decoder_weight_only(self, flag: bool):
        """Decoder linears as weight-only INT8 (W8A16): the decode GEMMs are bound by weight bytes, so
        keeping fp16 activations costs no speed and removes their quantization error."""
        self.decoder_weight_only = bool(flag)
        for layer in self.dec_layers:
            for n in ("qkv", "out", "cross_q", "cross_kv", "cross_out", "fc1", "fc2"):
                lin = getattr(layer, n)
                if isinstance(lin, Linear):
                    lin.weight_only = bool(flag) and lin.quantized

    def set_group_scales(self, encoder: bool, decoder: bool):
        """Choose per-row or per-(row, 256-group) dynamic activation scales for each stack."""
        self.group_scales_encoder = bool(encoder)
        self.group_scales_decoder = bool(decoder)
        for layers, names, flag in ((self.enc_layers, ("qkv", "out", "fc1", "fc2"), encoder),
                                    (self.dec_layers, ("qkv", "out", "cross_q", "cross_kv", "cross_out", "fc1", "fc2"),
                                     decoder)):
            for layer in layers:
                for n in names:
                    lin = getattr(layer, n)
                    if isinstance(lin, Linear):
                        lin.group_scales = bool(flag)

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, n_mels, T] -> encoder output [B, T/2, d_model] fp16."""
        x = mel.to(self.device, self.dtype)
        x = F.gelu(F.conv1d(x, self.conv1_w, self.conv1_b, padding=1))
        x = F.gelu(F.conv1d(x, self.conv2_w, self.conv2_b, stride=2, padding=1))
        x = x.permute(0, 2, 1)
        b, t, c = x.shape
        x = (x + self.enc_pos[:t]).contiguous()
        x2 = x.view(b * t, c)
        h, d = self.dims.n_audio_head, c // self.dims.n_audio_head
        for layer in self.enc_layers:
            qkv = layer.qkv(x2, ln=layer.attn_ln).view(b, t, 3, h, d)
            attn = _attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
            layer.out(attn.reshape(b * t, c), residual=x2, out=x2)
            hidden = layer.fc1(x2, ln=layer.mlp_ln, gelu=True)
            layer.fc2(hidden, residual=x2, out=x2)
        return self.ln_post(x2).view(b, t, c)

    @torch.inference_mode()
    def to(self, device) -> None:
        """Move every weight to ``device`` ("cpu" parks them in RAM); scratch buffers are dropped."""
        device = torch.device(device)
        self._cross_buffers.clear()
        for name in ("conv1_w", "conv1_b", "conv2_w", "conv2_b", "enc_pos", "tok_emb", "dec_pos"):
            setattr(self, name, _move(getattr(self, name), device))
        self.ln_post.to(device)
        self.dec_ln.to(device)
        for layer in self.enc_layers:
            for name in EncoderLayer.__slots__:
                getattr(layer, name).to(device)
        for layer in self.dec_layers:
            for name in DecoderLayer.__slots__:
                getattr(layer, name).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.device = device

    # ------------------------------------------------------------------
    # Decoder helpers
    # ------------------------------------------------------------------
    def release_cross_buffers(self, keep=()) -> None:
        """Free the cross-attention buffers of batch shapes that no decoding session uses any more."""
        for key in [key for key in self._cross_buffers if key not in keep]:
            del self._cross_buffers[key]

    def cross_kv_buffer(self, nb: int, t: int) -> torch.Tensor:
        """Persistent [L, nb, T, 2*d_model] buffer (fixed addresses for CUDA graphs)."""
        key = (nb, t)
        buf = self._cross_buffers.get(key)
        if buf is None:
            buf = torch.empty((len(self.dec_layers), nb, t, 2 * self.d_model), device=self.device, dtype=self.dtype)
            self._cross_buffers[key] = buf
        return buf

    @torch.inference_mode()
    def compute_cross_kv(self, enc: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        nb, t, c = enc.shape
        if out is None:
            out = self.cross_kv_buffer(nb, t)
        e2 = enc.reshape(nb * t, c)
        if e2.dtype != self.dtype:
            e2 = e2.to(self.dtype)
        e2 = e2.contiguous()
        for l, layer in enumerate(self.dec_layers):
            layer.cross_kv(e2, out=out[l].view(nb * t, 2 * c))
        return out

    def cross_views(self, cross: torch.Tensor, l: int):
        nb, t, c2 = cross.shape[1:]
        kv = cross[l].view(nb, t, 2, self.n_head, self.head_dim)
        return kv[:, :, 0], kv[:, :, 1]

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.dec_ln(x), self.tok_emb).float()

    def decoder_layers(self, x: torch.Tensor, nb: int, q_per_item: int, cross: torch.Tensor,
                       self_attn, align_scores: dict | None = None):
        """Run all decoder blocks on x [nb*q_per_item, C] (in place).

        ``self_attn(l, q, k, v)`` returns the self-attention output [*, C].
        Queries of one batch item attend to that item's cross K/V.
        """
        c, h, d = self.d_model, self.n_head, self.head_dim
        for l, layer in enumerate(self.dec_layers):
            qkv = layer.qkv(x, ln=layer.attn_ln)
            attn = self_attn(l, qkv)
            layer.out(attn, residual=x, out=x)
            cq = layer.cross_q(x, ln=layer.cross_ln).view(nb, q_per_item, h, d)
            ck, cv = self.cross_views(cross, l)
            if align_scores is not None and l in align_scores:
                heads = align_scores[l]["heads"]
                qh = cq[:, :, heads].float().permute(0, 2, 1, 3)  # nb, nh, T, d
                kh = ck[:, :, heads].float().permute(0, 2, 3, 1)  # nb, nh, d, S
                align_scores[l]["scores"] = torch.matmul(qh, kh) / math.sqrt(d)
            ca = _attention(cq, ck, cv)
            layer.cross_out(ca.reshape(nb * q_per_item, c), residual=x, out=x)
            hidden = layer.fc1(x, ln=layer.mlp_ln, gelu=True)
            layer.fc2(hidden, residual=x, out=x)
        return x

    @torch.inference_mode()
    def forward_full(self, tokens: torch.Tensor, cross: torch.Tensor, align_heads: dict | None = None):
        """Causal decoder over complete sequences (no cache), used by align/detect_language.

        tokens: [nb, T] int64. Returns hidden states [nb, T, C] (before final LN).
        """
        nb, t = tokens.shape
        c, h, d = self.d_model, self.n_head, self.head_dim
        x = (F.embedding(tokens, self.tok_emb) + self.dec_pos[:t]).reshape(nb * t, c).contiguous()

        def self_attn(l, qkv):
            q = qkv.view(nb, t, 3, h, d)
            return _attention(q[:, :, 0], q[:, :, 1], q[:, :, 2], causal=True).reshape(nb * t, c)

        self.decoder_layers(x, nb, t, cross, self_attn, align_scores=align_heads)
        return x.view(nb, t, c)


class DecodeSession:
    """Static decoding state for ``nb`` audio items x ``group`` rows each.

    Owns the self-attention KV cache and, once warmed up, a CUDA graph for one
    decoding step. Row order is item-major: row = item * group + j.
    """

    def __init__(self, rt: WhisperRuntime, nb: int, group: int, cross: torch.Tensor, use_graph: bool = True):
        self.rt = rt
        self.nb = nb
        self.group = group
        self.rows = nb * group
        self.cross = cross
        dims = rt.dims
        L, S = dims.n_text_layer, dims.n_text_ctx
        dev = rt.device
        self.max_len = S
        # K and V caches share one tensor so a single kernel can reorder both.
        self.kv = torch.zeros((2, L, self.rows, S, rt.n_head, rt.head_dim), device=dev, dtype=rt.dtype)
        self.k_cache = self.kv[0]
        self.v_cache = self.kv[1]
        self.beam_state = None  # GpuBeamState: beam search processing captured in the step graph
        self.ids = torch.zeros((self.rows,), device=dev, dtype=torch.long)
        self.pos = torch.zeros((self.rows,), device=dev, dtype=torch.long)
        self.seqlens = torch.zeros((self.rows,), device=dev, dtype=torch.int32)
        self.logits_out = torch.empty((self.rows, dims.n_vocab), device=dev, dtype=torch.float32)
        self.use_graph = use_graph and dev.type == "cuda"
        self.graph = None
        self._warm = 0
        self.first_rows = torch.arange(nb, device=dev, dtype=torch.int32) * group

    # -- prompt ---------------------------------------------------------
    @torch.inference_mode()
    def prefill(self, prompt: torch.Tensor, want_positions: int | None = None):
        """Run the prompt [nb, P] for the first row of every item, replicate to the group.

        Returns hidden state (pre final LN) at ``want_positions`` [nb, C] if requested.
        """
        rt = self.rt
        nb, p = prompt.shape
        c, h, d = rt.d_model, rt.n_head, rt.head_dim
        zeros = torch.zeros((nb,), device=rt.device, dtype=torch.int32)
        x = (F.embedding(prompt, rt.tok_emb) + rt.dec_pos[:p]).reshape(nb * p, c).contiguous()
        first_rows = self.first_rows

        def self_attn(l, qkv):
            q = qkv.view(nb, p, 3, h, d)
            out = _kv_cache_attention(q[:, :, 0], self.k_cache[l], self.v_cache[l], q[:, :, 1], q[:, :, 2],
                                      cache_seqlens=zeros, cache_batch_idx=first_rows)
            return out.reshape(nb * p, c)

        rt.decoder_layers(x, nb, p, self.cross, self_attn)
        if self.group > 1:
            src = first_rows.long().repeat_interleave(self.group - 1)
            dst = (first_rows.long()[:, None] + torch.arange(1, self.group, device=rt.device)[None, :]).reshape(-1)
            self.k_cache[:, dst, :p] = self.k_cache[:, src, :p]
            self.v_cache[:, dst, :p] = self.v_cache[:, src, :p]
        self.seqlens.fill_(p)
        self.pos.fill_(p)
        if want_positions is not None:
            return x.view(nb, p, c)[:, want_positions]
        return None

    def reset_empty(self):
        self.seqlens.zero_()
        self.pos.zero_()

    # -- one decoding step ---------------------------------------------
    def _step_impl(self):
        rt = self.rt
        c, h, d = rt.d_model, rt.n_head, rt.head_dim
        rows = self.rows
        x = (F.embedding(self.ids, rt.tok_emb) + F.embedding(self.pos, rt.dec_pos)).contiguous()

        def self_attn(l, qkv):
            q = qkv.view(rows, 1, 3, h, d)
            out = _kv_cache_attention(q[:, :, 0], self.k_cache[l], self.v_cache[l], q[:, :, 1], q[:, :, 2],
                                      cache_seqlens=self.seqlens, num_splits=1)
            return out.reshape(rows, c)

        rt.decoder_layers(x, self.nb, self.group, self.cross, self_attn)
        self.logits_out.copy_(rt.logits(x))
        if self.beam_state is not None:
            self.beam_state.process(self.logits_out)
        self.seqlens.add_(1)
        self.pos.add_(1)

    @torch.inference_mode()
    def step(self) -> torch.Tensor:
        if self.graph is not None:
            self.graph.replay()
            return self.logits_out
        self._step_impl()
        if self.use_graph:
            self._warm += 1
            if self._warm >= 2:
                self._capture()
        return self.logits_out

    def _capture(self):
        # Capture records the kernels without executing them, so the decoding
        # state (cache, ids, positions) is untouched by the capture itself.
        torch.cuda.synchronize()
        graph = CudaGraph()
        with graph.capture():
            self._step_impl()
        torch.cuda.synchronize()
        self.graph = graph
