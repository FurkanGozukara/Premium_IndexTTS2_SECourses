"""CTranslate2-compatible Whisper model backed by the ConvRot PyTorch runtime.

``ConvRotWhisper`` exposes the subset of ``ctranslate2.models.Whisper`` that
faster-whisper uses (``encode``, ``generate``, ``align``, ``detect_language``,
``is_multilingual``, ``n_mels``, ``device``...). The decoding algorithms are
ported from CTranslate2 4.7.1 (``src/decoding.cc``, ``src/models/whisper.cc``,
``src/dtw.cc``): beam search with 2*beam candidates and patience, greedy and
random sampling, hard prefixes, the Whisper timestamp rules, no-speech
probabilities and cross-attention DTW alignment. Logits processing runs in
fp32 (CTranslate2 fp16 runs it in fp16).

Ported from Whisper-WebUI Premium (SECourses), modules/whisper/convrot at revision 613beba (September 2026).
"""

from __future__ import annotations

import json
import math
import os
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .gpu_beam import GpuBeamState
from .model import DecodeSession, WhisperDims, WhisperRuntime

NEG = float(torch.finfo(torch.float32).min)  # CTranslate2 disables tokens with lowest()

# Runtime defaults (stored in each model's config.json by the converter). Encoder: W8A8 on INT8
# tensor cores with dynamic activation scales per (row, 256-group). Decoder: weight-only INT8
# (fp16 activations); its decode GEMMs are bound by weight bytes, so INT8 activations would add
# error without speed.
DEFAULT_RUNTIME = {"group_scales_encoder": True, "group_scales_decoder": True, "decoder_weight_only": True}

try:
    import numba

    @numba.njit(cache=True)
    def _negative_dtw_numba(x):
        n, m = x.shape
        cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
        trace = np.full((n + 1, m + 1), -1, dtype=np.int32)
        cost[0, 0] = 0.0
        for j in range(1, m + 1):
            for i in range(1, n + 1):
                c0 = cost[i - 1, j - 1]
                c1 = cost[i - 1, j]
                c2 = cost[i, j - 1]
                if c0 < c1 and c0 < c2:
                    c = c0
                    t = 0
                elif c1 < c0 and c1 < c2:
                    c = c1
                    t = 1
                else:
                    c = c2
                    t = 2
                cost[i, j] = -x[i - 1, j - 1] + c
                trace[i, j] = t
        for k in range(m + 1):
            trace[0, k] = 2
        for k in range(n + 1):
            trace[k, 0] = 1
        i = n
        j = m
        out = np.empty((n + m, 2), dtype=np.int64)
        count = 0
        while i > 0 or j > 0:
            out[count, 0] = i - 1
            out[count, 1] = j - 1
            count += 1
            t = trace[i, j]
            if t == 0:
                i -= 1
                j -= 1
            elif t == 1:
                i -= 1
            else:
                j -= 1
        return out[:count][::-1]

except Exception:  # pragma: no cover - numba ships with openai-whisper
    _negative_dtw_numba = None


def negative_dtw(x: np.ndarray) -> list:
    x = np.ascontiguousarray(x, dtype=np.float32)
    if _negative_dtw_numba is not None:
        return [(int(a), int(b)) for a, b in _negative_dtw_numba(x)]
    n, m = x.shape
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    trace = np.full((n + 1, m + 1), -1, dtype=np.int32)
    cost[0, 0] = 0
    for j in range(1, m + 1):
        for i in range(1, n + 1):
            c0, c1, c2 = cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]
            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2
            cost[i, j] = -x[i - 1, j - 1] + c
            trace[i, j] = t
    trace[0, :] = 2
    trace[:, 0] = 1
    i, j, result = n, m, []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))
        t = trace[i, j]
        if t == 0:
            i, j = i - 1, j - 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    return result[::-1]


class WhisperGenerationResult:
    __slots__ = ("sequences", "sequences_ids", "scores", "no_speech_prob", "logits")

    def __init__(self, sequences, sequences_ids, scores, no_speech_prob=0.0):
        self.sequences = sequences
        self.sequences_ids = sequences_ids
        self.scores = scores
        self.no_speech_prob = no_speech_prob
        self.logits = []

    def __repr__(self):
        return (f"WhisperGenerationResult(sequences_ids={self.sequences_ids!r}, scores={self.scores!r}, "
                f"no_speech_prob={self.no_speech_prob!r})")


class WhisperAlignmentResult:
    __slots__ = ("alignments", "text_token_probs")

    def __init__(self, alignments, text_token_probs):
        self.alignments = alignments
        self.text_token_probs = text_token_probs


def _load_vocabulary(model_path: str) -> List[str]:
    json_path = os.path.join(model_path, "vocabulary.json")
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return list(json.load(f))
    txt_path = os.path.join(model_path, "vocabulary.txt")
    if os.path.isfile(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n").rstrip("\r") for line in f]
    import tokenizers

    tok = tokenizers.Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
    size = tok.get_vocab_size(with_added_tokens=True)
    return [tok.id_to_token(i) for i in range(size)]


def load_safetensors(path: str, device: str = "cpu") -> tuple[dict, dict]:
    from safetensors import safe_open

    tensors = {}
    with safe_open(path, framework="pt", device=device) as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors, metadata


class _BeamResult:
    __slots__ = ("scores", "hypotheses", "done")

    def __init__(self):
        self.scores: list = []
        self.hypotheses: list = []
        self.done = False


def _finalize_score(score: float, length: int, length_penalty: float) -> float:
    denom = float(length) ** float(length_penalty) if (length > 0 or length_penalty == 0) else 0.0
    if denom == 0.0:
        return -math.inf if score < 0 else (math.inf if score > 0 else math.nan)
    return score / denom


def _sort_hypotheses(result: _BeamResult, max_hypotheses: int, keep_scores: bool):
    order = sorted(range(len(result.hypotheses)), key=lambda i: result.scores[i], reverse=True)
    order = order[:max_hypotheses]
    result.hypotheses = [result.hypotheses[i] for i in order]
    result.scores = [result.scores[i] for i in order] if keep_scores else []


class ConvRotWhisper:
    """Drop-in replacement for ``ctranslate2.models.Whisper`` (subset used by faster-whisper)."""

    # Decoding sessions kept for reuse (see _evict_sessions). One is enough: a batched run over 7 files took the
    # same time with 1 or 2, and 1 halved the VRAM (15.6 instead of 29.4 GB at batch size 16, beam size 5).
    MAX_CACHED_SESSIONS = 1

    def __init__(self, model_path: str, device: str = "cuda", device_index: int | Sequence[int] = 0,
                 compute_type: str = "int8_convrot", use_cuda_graphs: bool = True,
                 weights_file: Optional[str] = None, tensors: Optional[dict] = None, dims: Optional[WhisperDims] = None,
                 dtype: torch.dtype = torch.float16, group_scales_encoder: Optional[bool] = None,
                 group_scales_decoder: Optional[bool] = None, decoder_weight_only: Optional[bool] = None,
                 gpu_beam_search: bool = True, **_ignored):
        if isinstance(device_index, (list, tuple)):
            device_index = int(device_index[0]) if device_index else 0
        if device not in ("cuda", "auto"):
            raise ValueError("The INT8 ConvRot Whisper runtime requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("The INT8 ConvRot Whisper runtime requires an NVIDIA CUDA GPU")
        major, minor = torch.cuda.get_device_capability(int(device_index))
        if (major, minor) < (8, 0):
            raise RuntimeError(
                "INT8 ConvRot Whisper models need an NVIDIA Ampere (RTX 30 series) or newer GPU "
                f"(INT8 tensor cores); this GPU is compute capability {major}.{minor}. "
                "The app uses its Transformers Whisper on this GPU instead.")
        self.device = "cuda"
        self.device_index = [int(device_index)]
        self._torch_device = torch.device("cuda", int(device_index))
        self.model_path = model_path
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.vocabulary = _load_vocabulary(model_path)
        self._vocab_index = {tok: i for i, tok in enumerate(self.vocabulary)}

        metadata = {}
        if tensors is None:
            path = weights_file or os.path.join(model_path, "model.safetensors")
            tensors, metadata = load_safetensors(path, device="cpu")
        if dims is None:
            dims_json = metadata.get("whisper_dims") or json.dumps(self.config.get("whisper_dims", {}))
            dims = WhisperDims(**json.loads(dims_json))
        self.dims = dims
        self.metadata = metadata
        self.compute_type = compute_type
        with torch.cuda.device(self._torch_device):
            rt_cfg = {**DEFAULT_RUNTIME, **self.config.get("convrot", {}).get("runtime", {})}
            gse = rt_cfg["group_scales_encoder"] if group_scales_encoder is None else group_scales_encoder
            gsd = rt_cfg["group_scales_decoder"] if group_scales_decoder is None else group_scales_decoder
            dwo = rt_cfg["decoder_weight_only"] if decoder_weight_only is None else decoder_weight_only
            self.runtime = WhisperRuntime(tensors, dims, device=str(self._torch_device), dtype=dtype,
                                          group_scales_encoder=gse, group_scales_decoder=gsd,
                                          decoder_weight_only=dwo)
        del tensors

        vocab = self._vocab_index
        self.sot_id = vocab["<|startoftranscript|>"]
        self.eot_id = vocab["<|endoftext|>"]
        self.no_timestamps_id = vocab["<|notimestamps|>"]
        self.no_speech_id = vocab.get("<|nospeech|>", vocab.get("<|nocaptions|>"))
        self.timestamp_begin_id = self.no_timestamps_id + 1
        self.timestamp_end_id = dims.n_vocab - 1
        self._is_multilingual = "<|en|>" in vocab and dims.n_vocab >= 51865
        self._num_languages = self.no_speech_id - self.sot_id - 5
        self.use_cuda_graphs = use_cuda_graphs
        # Beam search inside the decode graph (no per-step host round trip); the CPU
        # port stays the reference and handles prefixes, penalties and sampling.
        self.gpu_beam_search = gpu_beam_search and use_cuda_graphs
        self._sessions: dict = {}  # decoding sessions by (items, rows per item, audio frames, GPU beam search)
        self._suppress_cache: dict = {}
        self._arange_v = torch.arange(dims.n_vocab, device=self._torch_device)
        self._loaded = True
        self._weights_file = weights_file

    # ------------------------------------------------------------------
    # ctranslate2.models.Whisper properties
    # ------------------------------------------------------------------
    @property
    def is_multilingual(self) -> bool:
        return self._is_multilingual

    @property
    def n_mels(self) -> int:
        return self.dims.n_mels

    @property
    def num_languages(self) -> int:
        return self._num_languages

    @property
    def model_is_loaded(self) -> bool:
        return self._loaded

    def unload_model(self, to_cpu: bool = False):
        """Free the GPU memory (CTranslate2 semantics). With ``to_cpu`` the weights stay in RAM for a
        fast ``load_model()``; otherwise they are dropped and ``load_model()`` reloads them from disk."""
        if not self._loaded:
            return
        self._sessions.clear()  # CUDA graphs, KV caches and beam state
        self._suppress_cache.clear()
        self._arange_v = None
        if to_cpu:
            self.runtime.to("cpu")
        else:
            self.runtime = None
        self._loaded = False
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    def load_model(self, keep_cache: bool = False):
        if self._loaded:
            return
        with torch.cuda.device(self._torch_device):
            if self.runtime is None:
                path = self._weights_file or os.path.join(self.model_path, "model.safetensors")
                tensors, _ = load_safetensors(path, device="cpu")
                rt_cfg = {**DEFAULT_RUNTIME, **self.config.get("convrot", {}).get("runtime", {})}
                self.runtime = WhisperRuntime(tensors, self.dims, device=str(self._torch_device),
                                              group_scales_encoder=rt_cfg["group_scales_encoder"],
                                              group_scales_decoder=rt_cfg["group_scales_decoder"],
                                              decoder_weight_only=rt_cfg["decoder_weight_only"])
                del tensors
            else:
                self.runtime.to(self._torch_device)
            self._arange_v = torch.arange(self.dims.n_vocab, device=self._torch_device)
        self._loaded = True

    def _ensure_loaded(self):
        if not self._loaded:
            self.load_model()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def _to_tensor(self, features) -> torch.Tensor:
        if torch.is_tensor(features):
            t = features
        elif isinstance(features, np.ndarray):
            t = torch.from_numpy(features)
        else:
            dev = getattr(features, "device", "cpu")
            if dev == "cuda":
                t = torch.as_tensor(features, device=f"cuda:{getattr(features, 'device_index', 0)}")
            else:
                try:
                    t = torch.from_numpy(np.array(features))
                except Exception:
                    import ctranslate2

                    t = torch.from_numpy(np.array(features.to(ctranslate2.DataType.float32)))
        if t.dim() == 2:
            t = t.unsqueeze(0)
        return t

    def _encoder_output(self, features) -> torch.Tensor:
        t = self._to_tensor(features)
        if t.shape[1] == self.dims.n_mels and t.shape[2] != self.dims.n_audio_state:
            return self.runtime.encode(t.to(self._torch_device))
        return t.to(self._torch_device, self.runtime.dtype)

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def encode(self, features, to_cpu: bool = False):
        self._ensure_loaded()
        with torch.cuda.device(self._torch_device):
            mel = self._to_tensor(features).to(self._torch_device)
            out = self.runtime.encode(mel)
            if to_cpu:
                return out.float().cpu().numpy()
            return out

    # ------------------------------------------------------------------
    # detect_language
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def detect_language(self, features) -> List[List[tuple]]:
        if not self.is_multilingual:
            raise RuntimeError("detect_language can only be called on multilingual models")
        self._ensure_loaded()
        with torch.cuda.device(self._torch_device):
            enc = self._encoder_output(features)
            nb = enc.shape[0]
            cross = self.runtime.compute_cross_kv(enc, out=torch.empty(
                (self.dims.n_text_layer, nb, enc.shape[1], 2 * self.dims.n_text_state),
                device=self._torch_device, dtype=self.runtime.dtype))
            tokens = torch.full((nb, 1), self.sot_id, device=self._torch_device, dtype=torch.long)
            hidden = self.runtime.forward_full(tokens, cross)
            logits = self.runtime.logits(hidden[:, 0])
            lang_ids = [int(i) for i in self.config["lang_ids"]]
            probs = torch.softmax(logits[:, lang_ids], dim=-1).cpu().numpy()
        results = []
        for i in range(nb):
            pairs = [(self.vocabulary[lang_ids[j]], float(probs[i, j])) for j in range(len(lang_ids))]
            pairs.sort(key=lambda p: p[1], reverse=True)
            results.append(pairs)
        return results

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------
    def _session(self, nb: int, group: int, t_audio: int, gpu_beam: bool = False) -> DecodeSession:
        key = (nb, group, t_audio, gpu_beam)
        sess = self._sessions.pop(key, None)
        if sess is None:
            self._evict_sessions(self.MAX_CACHED_SESSIONS - 1)
            cross = self.runtime.cross_kv_buffer(nb, t_audio)
            sess = DecodeSession(self.runtime, nb, group, cross, use_graph=self.use_cuda_graphs)
            if gpu_beam:
                sess.beam_state = GpuBeamState(sess, self.eot_id, self.no_timestamps_id, self.timestamp_begin_id,
                                               self.timestamp_end_id, self.no_speech_id)
        self._sessions[key] = sess  # the most recently used session is the last one
        return sess

    def _evict_sessions(self, keep: int) -> None:
        """Drop the least recently used decoding sessions (and their cross-attention buffers) beyond ``keep``.

        A session holds the KV cache, cross-attention buffer and CUDA graph of one batch shape: about 10 GB at
        batch size 16 with beam size 5 for large-v3. Batched runs made one for every size of a file's last batch
        and kept them all, so a batch of files filled the VRAM (65 GB reserved after 7 files) and slowed down.
        """
        if len(self._sessions) <= keep:
            return
        while len(self._sessions) > keep:
            self._sessions.pop(next(iter(self._sessions)))
        self.runtime.release_cross_buffers(keep={(key[0], key[2]) for key in self._sessions})
        import gc

        gc.collect()  # a session and its beam state reference each other
        torch.cuda.empty_cache()

    def _get_prompt_ids(self, prompts) -> List[List[int]]:
        out = []
        for prompt in prompts:
            ids = []
            for tok in prompt:
                if isinstance(tok, str):
                    ids.append(self._vocab_index[tok])
                else:
                    ids.append(int(tok))
            out.append(ids)
        return out

    def _check_prompts(self, prompts: List[List[int]]):
        sot_index = prompt_length = None
        for prompt in prompts:
            try:
                idx = prompt.index(self.sot_id)
            except ValueError:
                raise ValueError("<|startoftranscript|> token was not found in the prompt")
            length = idx
            while length < len(prompt) and self.sot_id <= prompt[length] <= self.no_timestamps_id:
                length += 1
            if sot_index is None:
                sot_index, prompt_length = idx, length
            elif idx != sot_index:
                raise ValueError("The generate method currently requires the <|startoftranscript|> token to be at "
                                 "the same position in all batches.")
            elif length != prompt_length:
                raise ValueError("The generate method currently requires each batch to have the same number of "
                                 "task tokens after <|startoftranscript|>.")
        return sot_index, prompt_length

    def _suppress_mask(self, ids: tuple) -> torch.Tensor:
        mask = self._suppress_cache.get(ids)
        if mask is None:
            mask = torch.zeros(self.dims.n_vocab, dtype=torch.bool, device=self._torch_device)
            valid = [i for i in ids if 0 <= i < self.dims.n_vocab]
            if valid:
                mask[torch.tensor(valid, device=self._torch_device)] = True
            self._suppress_cache[ids] = mask
        return mask

    @torch.inference_mode()
    def generate(self, features, prompts, asynchronous: bool = False, beam_size: int = 5, patience: float = 1,
                 num_hypotheses: int = 1, length_penalty: float = 1, repetition_penalty: float = 1,
                 no_repeat_ngram_size: int = 0, max_length: int = 448, return_scores: bool = False,
                 return_logits_vocab: bool = False, return_no_speech_prob: bool = False,
                 max_initial_timestamp_index: int = 50, suppress_blank: bool = True,
                 suppress_tokens: Optional[Iterable[int]] = (-1,), sampling_topk: int = 1,
                 sampling_temperature: float = 1, **_ignored) -> List[WhisperGenerationResult]:
        if asynchronous:
            raise NotImplementedError("asynchronous generation is not supported by the ConvRot runtime")
        if return_logits_vocab:
            raise NotImplementedError("return_logits_vocab is not supported by the ConvRot runtime")
        if beam_size <= 0 or patience <= 0 or num_hypotheses <= 0:
            raise ValueError("beam_size, patience and num_hypotheses must be positive")
        if repetition_penalty <= 0:
            raise ValueError("The repetition penalty must be > 0")
        max_candidates = int(round(beam_size * patience))
        if num_hypotheses > max_candidates and not (beam_size == 1 and sampling_topk != 1):
            raise ValueError("The number of hypotheses cannot be greater than beam_size * patience")

        self._ensure_loaded()
        with torch.cuda.device(self._torch_device):
            prompts = self._get_prompt_ids(prompts)
            if not prompts:
                return []
            enc = self._encoder_output(features)
            nb = enc.shape[0]
            if len(prompts) != nb:
                raise ValueError(f"Got {len(prompts)} prompts for a batch of {nb} features")
            return self._generate(enc, prompts, beam_size, patience, num_hypotheses, length_penalty,
                                  repetition_penalty, no_repeat_ngram_size, max_length, return_scores,
                                  return_no_speech_prob, max_initial_timestamp_index, suppress_blank,
                                  list(suppress_tokens) if suppress_tokens is not None else [],
                                  sampling_topk, sampling_temperature)

    def _generate(self, enc, prompts, beam_size, patience, num_hypotheses, length_penalty, repetition_penalty,
                  no_repeat_ngram_size, max_length, return_scores, return_no_speech_prob,
                  max_initial_timestamp_index, suppress_blank, suppress_tokens, sampling_topk,
                  sampling_temperature):
        nb = enc.shape[0]
        sot_index, prompt_length = self._check_prompts(prompts)
        sot_is_start = sot_index == prompt_length - 1

        disable_ids = []
        for tok in suppress_tokens:
            if tok >= 0:
                disable_ids.append(int(tok))
            elif tok == -1:
                disable_ids.extend(int(i) for i in self.config.get("suppress_ids", []))
        disable_ids_begin = [int(i) for i in self.config.get("suppress_ids_begin", [])] if suppress_blank else []

        if prompt_length == 1:
            prompt_tokens = None
            start_tokens = [list(p) for p in prompts]
            start_step = 0
        else:
            prompt_tokens = [p[:prompt_length - 1] for p in prompts]
            start_tokens = [p[prompt_length - 1:] for p in prompts]
            start_step = prompt_length - 1
        decode_max_length = min(max_length // 2, max_length - start_step)
        if decode_max_length <= 0:
            raise ValueError("The maximum decoding length must be > 0")
        start_ids = [st[0] for st in start_tokens]
        prefix_ids = [st[1:] for st in start_tokens]
        if all(len(p) == 0 for p in prefix_ids):
            prefix_ids = None

        timestamps = prompts[0][prompt_length - 1] != self.no_timestamps_id
        ts_cfg = None
        if timestamps:
            ts_cfg = (self.timestamp_begin_id + max_initial_timestamp_index)

        greedy = beam_size == 1
        group = beam_size if not greedy else num_hypotheses
        random_sampler = not (sampling_topk == 1 or sampling_temperature == 0.0)
        use_gpu_beam = (self.gpu_beam_search and not greedy and prefix_ids is None and not random_sampler
                        and repetition_penalty == 1 and no_repeat_ngram_size == 0 and length_penalty != 0
                        and int(round(beam_size * patience)) <= 3 * beam_size and decode_max_length <= 224)
        sess = self._session(nb, group, enc.shape[1], gpu_beam=use_gpu_beam)
        self.runtime.compute_cross_kv(enc, out=sess.cross)

        no_speech_probs = None
        if prompt_tokens is not None:
            want = sot_index if (return_no_speech_prob and not sot_is_start) else None
            hidden = sess.prefill(torch.tensor(prompt_tokens, device=self._torch_device, dtype=torch.long),
                                  want_positions=want)
            if hidden is not None:
                logits = self.runtime.logits(hidden)
                no_speech_probs = torch.softmax(logits, dim=-1)[:, self.no_speech_id].cpu().tolist()
        else:
            sess.reset_empty()

        common = dict(sess=sess, nb=nb, group=group, start_ids=start_ids, prefix_ids=prefix_ids,
                      max_length=decode_max_length, start_step=start_step,
                      length_penalty=length_penalty, repetition_penalty=repetition_penalty,
                      no_repeat_ngram_size=no_repeat_ngram_size, disable_ids=tuple(sorted(set(disable_ids))),
                      disable_ids_begin=disable_ids_begin, ts_max_initial_id=ts_cfg,
                      want_first_no_speech=return_no_speech_prob and sot_is_start)
        if use_gpu_beam:
            results, first_ns = self._gpu_beam_search(sess, nb, beam_size, start_ids, decode_max_length, start_step,
                                                      length_penalty, patience, num_hypotheses, return_scores,
                                                      tuple(sorted(set(disable_ids))), disable_ids_begin, ts_cfg,
                                                      return_no_speech_prob and sot_is_start)
        elif greedy:
            results, first_ns = self._greedy_search(num_hypotheses=num_hypotheses, return_scores=return_scores,
                                                    sampling_topk=sampling_topk,
                                                    sampling_temperature=sampling_temperature, **common)
        else:
            results, first_ns = self._beam_search(beam=beam_size, patience=patience, num_hypotheses=num_hypotheses,
                                                  return_scores=return_scores, sampling_topk=sampling_topk,
                                                  sampling_temperature=sampling_temperature, **common)
        if first_ns is not None:
            no_speech_probs = first_ns

        final = []
        for i in range(nb):
            res = results[i]
            seqs_ids = [list(h) for h in res.hypotheses]
            seqs = [[self.vocabulary[t] for t in h] for h in seqs_ids]
            nsp = float(no_speech_probs[i]) if (return_no_speech_prob and no_speech_probs is not None) else 0.0
            final.append(WhisperGenerationResult(seqs, seqs_ids, list(res.scores), nsp))
        return final

    # ------------------------------------------------------------------
    # Logits processors (CTranslate2 order: apply_first, repetition penalty,
    # no-repeat-ngram, suppress tokens, suppress-begin, timestamp rules)
    # ------------------------------------------------------------------
    def _process_logits(self, logits: torch.Tensor, step: int, seqs: np.ndarray, rows_item: np.ndarray,
                        prefix_lens: np.ndarray, repetition_penalty: float, no_repeat_ngram_size: int,
                        disable_ids: tuple, disable_ids_begin: list, ts_max_initial_id):
        """In-place logits processing. ``seqs`` [rows, step] holds generated tokens."""
        rows = logits.shape[0]
        dev = logits.device
        if repetition_penalty != 1 and step > 0:
            prev = torch.from_numpy(seqs.astype(np.int64)).to(dev, non_blocking=True)
            scores = logits.gather(1, prev)
            scores = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
            logits.scatter_(1, prev, scores)
        if no_repeat_ngram_size > 0 and step >= no_repeat_ngram_size:
            r_idx, t_idx = [], []
            n = no_repeat_ngram_size
            for r in range(rows):
                seq = seqs[r]
                length = seq.shape[0]
                current = seq[length - n + 1:] if n > 1 else seq[length:]
                banned = set()
                for s in range(0, length - n + 1):
                    if n == 1 or np.array_equal(seq[s:s + n - 1], current):
                        banned.add(int(seq[s + n - 1]))
                for tok in banned:
                    r_idx.append(r)
                    t_idx.append(tok)
            if r_idx:
                logits[torch.tensor(r_idx, device=dev), torch.tensor(t_idx, device=dev)] = NEG
        if disable_ids:
            logits.masked_fill_(self._suppress_mask(disable_ids)[None, :], NEG)
        if disable_ids_begin:
            begin_rows = [r for r in range(rows) if step == prefix_lens[rows_item[r]]]
            if begin_rows:
                if len(begin_rows) == rows:
                    logits[:, disable_ids_begin] = NEG
                else:
                    br = torch.tensor(begin_rows, device=dev)
                    logits[br[:, None], torch.tensor(disable_ids_begin, device=dev)[None, :]] = NEG
        if ts_max_initial_id is None:
            return
        tb, te, eot, nt = self.timestamp_begin_id, self.timestamp_end_id, self.eot_id, self.no_timestamps_id
        lo = np.zeros((rows, 3), dtype=np.int64)
        hi = np.zeros((rows, 3), dtype=np.int64)
        check = np.zeros((rows,), dtype=bool)
        lo[:, 0], hi[:, 0] = nt, nt + 1
        for r in range(rows):
            sample_begin = int(prefix_lens[rows_item[r]])
            if step == sample_begin and step == 0:
                lo[r, 1], hi[r, 1] = 0, tb
                lo[r, 2], hi[r, 2] = ts_max_initial_id + 1, te + 1
            elif step > sample_begin:
                seq = seqs[r]
                last = int(seq[step - 1])
                if last >= tb:
                    pen = int(seq[step - 2]) if step - 1 > sample_begin else last
                    if pen >= tb:
                        lo[r, 1], hi[r, 1] = tb, te + 1
                    else:
                        lo[r, 1], hi[r, 1] = 0, eot
                        lo[r, 2], hi[r, 2] = tb, last
                        check[r] = True
                else:
                    check[r] = True
                    window = seq[sample_begin:step]
                    ts_pos = np.nonzero(window >= tb)[0]
                    if ts_pos.size:
                        tok = int(window[ts_pos[-1]])
                        lo[r, 1], hi[r, 1] = tb, tok + 1
        lo_t = torch.from_numpy(lo).to(dev, non_blocking=True)
        hi_t = torch.from_numpy(hi).to(dev, non_blocking=True)
        ar = self._arange_v
        mask = ((ar[None, None, :] >= lo_t[:, :, None]) & (ar[None, None, :] < hi_t[:, :, None])).any(dim=1)
        logits.masked_fill_(mask, NEG)
        if check.any():
            check_t = torch.from_numpy(check).to(dev, non_blocking=True)
            log_probs = torch.log_softmax(logits, dim=-1)
            ts_lp = torch.logsumexp(log_probs[:, tb:], dim=-1)
            max_text = log_probs[:, :tb].max(dim=-1).values
            force = check_t & (ts_lp > max_text)
            logits[:, :tb].masked_fill_(force[:, None], NEG)

    # ------------------------------------------------------------------
    # Beam search (CTranslate2 BeamSearch::search)
    # ------------------------------------------------------------------
    def _beam_search(self, sess: DecodeSession, nb, group, start_ids, prefix_ids, max_length, start_step,
                     length_penalty, repetition_penalty, no_repeat_ngram_size, disable_ids, disable_ids_begin,
                     ts_max_initial_id, want_first_no_speech, beam, patience, num_hypotheses, return_scores,
                     sampling_topk, sampling_temperature):
        V = self.dims.n_vocab
        eot = self.eot_id
        rows = nb * beam
        num_candidates = beam * 2
        max_cands = int(round(beam * patience))
        allow_early_exit = length_penalty == 0
        dev = self._torch_device
        prefix_lens = np.array([len(p) for p in prefix_ids] if prefix_ids is not None else [0] * nb, dtype=np.int64)
        max_step = int(max((int(pl) + max_length for pl in prefix_lens), default=max_length)) \
            if prefix_ids is not None else max_length
        rows_item = np.repeat(np.arange(nb), beam)
        random_sampler = not (sampling_topk == 1 or sampling_temperature == 0.0)

        sess.ids.copy_(torch.tensor(np.repeat(np.asarray(start_ids, dtype=np.int64), beam), device=dev))
        beam_scores = torch.full((rows,), NEG, device=dev, dtype=torch.float32)
        beam_scores[::beam] = 0.0
        alive = np.zeros((nb, beam, 0), dtype=np.int64)
        results = [_BeamResult() for _ in range(nb)]
        top_beam_finished = [False] * nb
        first_ns = None
        prompt_len = start_step

        for step in range(max_step):
            logits = sess.step()
            if want_first_no_speech and step == 0:
                probs = torch.softmax(logits, dim=-1)[:, self.no_speech_id]
                first_ns = probs[::beam].cpu().tolist()
            self._process_logits(logits, step, alive.reshape(rows, -1), rows_item, prefix_lens, repetition_penalty,
                                  no_repeat_ngram_size, disable_ids, disable_ids_begin, ts_max_initial_id)
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs += beam_scores[:, None]
            flat = log_probs.view(nb, beam * V)
            if random_sampler:
                top_scores, top_ids = self._random_sample(flat, num_candidates, sampling_topk, sampling_temperature)
            else:
                top_scores, top_ids = torch.topk(flat, num_candidates, dim=-1)
            cand = torch.stack((top_ids.to(torch.float64), top_scores.to(torch.float64)), dim=0).cpu().numpy()
            top_ids_np = cand[0].astype(np.int64)
            top_scores_np = cand[1].astype(np.float32)
            words = top_ids_np % V
            origins = top_ids_np // V  # beam index inside the item

            if prefix_ids is not None:
                for i in range(nb):
                    pl = int(prefix_lens[i])
                    if step > pl:
                        continue
                    for k in range(num_candidates):
                        if step < pl:
                            words[i, k] = prefix_ids[i][step]
                            top_scores_np[i, k] = 0.0 if k == 0 else -1e10
                            origins[i, k] = 0
                        elif k > 0 and words[i, k] == eot:
                            words[i, k] = 0
                            top_scores_np[i, k] = -1e10
                            origins[i, k] = 0

            cand_alive = np.concatenate(
                (np.take_along_axis(alive, origins[:, :, None].repeat(alive.shape[2], axis=2), axis=1)
                 if alive.shape[2] else np.zeros((nb, num_candidates, 0), dtype=np.int64),
                 words[:, :, None]), axis=2)

            active = np.tile(np.arange(beam), (nb, 1))
            for i in range(nb):
                res = results[i]
                if res.done:
                    continue
                pl = int(prefix_lens[i])
                is_last = step + 1 == max_length + pl
                secondary = beam
                for k in range(beam):
                    last_id = int(words[i, k])
                    next_beam = k
                    if (last_id == eot and step >= pl) or is_last:
                        if k == 0:
                            top_beam_finished[i] = True
                        ignore_last = last_id == eot
                        end = step if ignore_last else step + 1
                        res.scores.append(float(top_scores_np[i, k]))
                        res.hypotheses.append(cand_alive[i, k, pl:end].tolist())
                        for j in range(secondary, num_candidates):
                            if int(words[i, j]) != eot:
                                next_beam = j
                                secondary = j + 1
                                break
                    active[i, k] = next_beam
                if is_last:
                    finished = True
                elif allow_early_exit:
                    finished = top_beam_finished[i] and len(res.hypotheses) >= num_hypotheses
                else:
                    finished = len(res.hypotheses) >= max_cands
                if finished:
                    res.scores = [_finalize_score(s, len(h), length_penalty)
                                  for s, h in zip(res.scores, res.hypotheses)]
                    _sort_hypotheses(res, num_hypotheses, return_scores)
                    res.done = True
            if all(r.done for r in results):
                break

            sel_words = np.take_along_axis(words, active, axis=1)
            sel_scores = np.take_along_axis(top_scores_np, active, axis=1)
            sel_origins = np.take_along_axis(origins, active, axis=1)
            alive = np.take_along_axis(cand_alive, active[:, :, None].repeat(cand_alive.shape[2], axis=2), axis=1)
            src_rows = (np.arange(nb)[:, None] * beam + sel_origins).reshape(-1)
            for i in range(nb):
                if results[i].done:
                    src_rows[i * beam:(i + 1) * beam] = np.arange(i * beam, (i + 1) * beam)
                    sel_words[i] = eot
            changed = np.nonzero(src_rows != np.arange(rows))[0]
            if changed.size:
                length = prompt_len + step + 1
                src = torch.from_numpy(src_rows[changed]).to(dev)
                dst = torch.from_numpy(changed).to(dev)
                lo_pos = prompt_len
                sess.k_cache[:, dst, lo_pos:length] = sess.k_cache[:, src, lo_pos:length]
                sess.v_cache[:, dst, lo_pos:length] = sess.v_cache[:, src, lo_pos:length]
            sess.ids.copy_(torch.from_numpy(sel_words.reshape(-1)).to(dev, non_blocking=True))
            beam_scores = torch.from_numpy(sel_scores.reshape(-1).astype(np.float32)).to(dev, non_blocking=True)
        return results, first_ns

    def _gpu_beam_search(self, sess, nb, beam, start_ids, max_length, start_step, length_penalty, patience,
                         num_hypotheses, return_scores, disable_ids, disable_ids_begin, ts_max_initial_id,
                         want_first_no_speech):
        state: GpuBeamState = sess.beam_state
        begin_mask = self._suppress_mask(tuple(sorted(set(disable_ids_begin)))) if disable_ids_begin else \
            torch.zeros(self.dims.n_vocab, dtype=torch.bool, device=self._torch_device)
        suppress = self._suppress_mask(disable_ids) if disable_ids else \
            torch.zeros(self.dims.n_vocab, dtype=torch.bool, device=self._torch_device)
        state.reset(start_ids, max_length, int(round(beam * patience)), num_hypotheses, length_penalty,
                    ts_max_initial_id is not None, ts_max_initial_id if ts_max_initial_id is not None else 0,
                    start_step, suppress, begin_mask)
        first_ns = None
        pending = []
        flags = self._done_flags()
        for step in range(max_length):
            sess.step()
            if step == 0 and want_first_no_speech:
                first_ns = state.ns_prob[::beam].cpu().tolist()
            flag = flags[step % len(flags)]
            flag.copy_(state.done.all(), non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            pending.append((event, flag))
            if len(pending) > 2:
                ev, fl = pending.pop(0)
                ev.synchronize()
                if bool(fl):
                    break
        torch.cuda.current_stream().synchronize()
        results = []
        for hyps in state.results():
            res = _BeamResult()
            res.scores = [_finalize_score(sc, len(toks), length_penalty) for sc, toks in hyps]
            res.hypotheses = [toks for _, toks in hyps]
            _sort_hypotheses(res, num_hypotheses, return_scores)
            results.append(res)
        return results, first_ns

    def _done_flags(self):
        flags = getattr(self, "_pinned_flags", None)
        if flags is None:
            flags = [torch.zeros((), dtype=torch.bool).pin_memory() for _ in range(4)]
            self._pinned_flags = flags
        return flags

    def _random_sample(self, scores: torch.Tensor, num_samples: int, topk: int, temperature: float):
        """CTranslate2 RandomSampler: optional top-k restriction, temperature, Gumbel-max for >1 samples."""
        top_ids = None
        final = scores
        if topk > 0 and topk < scores.shape[-1]:
            final, top_ids = torch.topk(scores, topk, dim=-1)
        if temperature != 1:
            final = final / temperature
        if num_samples > 1:
            log_probs = torch.log_softmax(final, dim=-1)
            gumbel = -torch.log(-torch.log(torch.rand_like(log_probs).clamp_min(1e-20)))
            ids = torch.topk(log_probs + gumbel, num_samples, dim=-1).indices
        else:
            probs = torch.softmax(final, dim=-1)
            ids = torch.multinomial(probs, 1)
        if top_ids is not None:
            ids = top_ids.gather(-1, ids)
        return scores.gather(-1, ids), ids

    # ------------------------------------------------------------------
    # Greedy / random sampling (CTranslate2 GreedySearch::search)
    # ------------------------------------------------------------------
    def _greedy_search(self, sess: DecodeSession, nb, group, start_ids, prefix_ids, max_length, start_step,
                       length_penalty, repetition_penalty, no_repeat_ngram_size, disable_ids, disable_ids_begin,
                       ts_max_initial_id, want_first_no_speech, num_hypotheses, return_scores, sampling_topk,
                       sampling_temperature):
        eot = self.eot_id
        nh = group
        rows = nb * nh
        dev = self._torch_device
        keep_scores = return_scores or nh > 1
        prefix_lens_item = np.array([len(p) for p in prefix_ids] if prefix_ids is not None else [0] * nb,
                                    dtype=np.int64)
        rows_item = np.repeat(np.arange(nb), nh)
        max_step = int(max((int(pl) + max_length for pl in prefix_lens_item), default=max_length)) \
            if prefix_ids is not None else max_length
        random_sampler = not (sampling_topk == 1 or sampling_temperature == 0.0)

        sess.ids.copy_(torch.tensor(np.repeat(np.asarray(start_ids, dtype=np.int64), nh), device=dev))
        hyps = [[] for _ in range(rows)]
        scores = [0.0] * rows
        done = [False] * rows
        seqs = np.zeros((rows, 0), dtype=np.int64)
        first_ns = None

        for step in range(max_step):
            logits = sess.step()
            if want_first_no_speech and step == 0:
                probs = torch.softmax(logits, dim=-1)[:, self.no_speech_id]
                first_ns = probs[::nh].cpu().tolist()
            self._process_logits(logits, step, seqs, rows_item, prefix_lens_item, repetition_penalty,
                                  no_repeat_ngram_size, disable_ids, disable_ids_begin, ts_max_initial_id)
            scores_t = torch.log_softmax(logits, dim=-1) if keep_scores else logits
            if random_sampler:
                best, ids = self._random_sample(scores_t, 1, sampling_topk, sampling_temperature)
            else:
                best, ids = torch.max(scores_t, dim=-1, keepdim=True)
            pair = torch.cat((ids.to(torch.float64), best.to(torch.float64)), dim=1).cpu().numpy()
            word_ids = pair[:, 0].astype(np.int64)
            word_scores = pair[:, 1]
            if prefix_ids is not None:
                for r in range(rows):
                    pl = int(prefix_lens_item[rows_item[r]])
                    if step < pl:
                        word_ids[r] = prefix_ids[rows_item[r]][step]
                        word_scores[r] = 0.0
            seqs = np.concatenate((seqs, word_ids[:, None]), axis=1)
            for r in range(rows):
                if done[r]:
                    continue
                pl = int(prefix_lens_item[rows_item[r]])
                wid = int(word_ids[r])
                if wid != eot and step >= pl:
                    hyps[r].append(wid)
                if keep_scores:
                    scores[r] += float(word_scores[r])
                if (wid == eot and step >= pl) or (step + 1 == max_length + pl):
                    done[r] = True
            if all(done):
                break
            next_ids = np.where(np.asarray(done), eot, word_ids)
            sess.ids.copy_(torch.from_numpy(next_ids).to(dev, non_blocking=True))

        results = []
        for i in range(nb):
            res = _BeamResult()
            for j in range(nh):
                r = i * nh + j
                res.hypotheses.append(hyps[r])
                res.scores.append(_finalize_score(scores[r], len(hyps[r]), length_penalty) if keep_scores else 0.0)
            _sort_hypotheses(res, num_hypotheses, return_scores)
            results.append(res)
        return results, first_ns

    # ------------------------------------------------------------------
    # align (CTranslate2 WhisperReplica::align)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def align(self, features, start_sequence, text_tokens, num_frames, median_filter_width: int = 7):
        text_tokens = [list(map(int, t)) for t in text_tokens]
        nb = len(text_tokens)
        if nb == 0:
            return []
        if isinstance(num_frames, (int, np.integer)):
            num_frames = [int(num_frames)] * nb
        num_frames = [int(n) for n in num_frames]
        if len(num_frames) != nb:
            raise ValueError("Invalid batch size for argument num_frames")
        start_sequence = [int(t) for t in start_sequence]
        heads = self.config.get("alignment_heads")
        if not heads:
            raise RuntimeError("The model configuration does not contain 'alignment_heads'")
        self._ensure_loaded()
        with torch.cuda.device(self._torch_device):
            enc = self._encoder_output(features)
            if enc.shape[0] != nb:
                raise ValueError("align() expects one encoder output per text sequence")
            inputs = [start_sequence + [self.no_timestamps_id] + t + [self.eot_id] for t in text_tokens]
            outputs = [seq[1:] + [0] for seq in inputs]
            t_max = max(len(s) for s in inputs)
            tok = torch.zeros((nb, t_max), dtype=torch.long)
            out_tok = torch.zeros((nb, t_max), dtype=torch.long)
            for b, (inp, outp) in enumerate(zip(inputs, outputs)):
                tok[b, :len(inp)] = torch.tensor(inp)
                out_tok[b, :len(outp)] = torch.tensor(outp)
            tok = tok.to(self._torch_device)
            out_tok = out_tok.to(self._torch_device)
            cross = self.runtime.compute_cross_kv(enc, out=torch.empty(
                (self.dims.n_text_layer, nb, enc.shape[1], 2 * self.dims.n_text_state),
                device=self._torch_device, dtype=self.runtime.dtype))
            spec = {}
            for layer, head in heads:
                spec.setdefault(int(layer), {"heads": []})["heads"].append(int(head))
            hidden = self.runtime.forward_full(tok, cross, align_heads=spec)
            logits = self.runtime.logits(hidden.reshape(nb * t_max, -1)).view(nb, t_max, -1)
            text_logits = logits[..., :self.eot_id]
            log_z = torch.logsumexp(text_logits, dim=-1)
            gathered = logits.gather(-1, out_tok.clamp_max(self.eot_id - 1)[..., None]).squeeze(-1)
            token_probs = torch.exp(gathered - log_z).float().cpu().numpy()
            scores = torch.cat([spec[l]["scores"] for l in sorted(spec)], dim=1)  # nb, heads, T, S
            frames = [n // 2 for n in num_frames]
            alignments = []
            if len(set(frames)) == 1:
                w = torch.softmax(scores[..., :frames[0]], dim=-1)
                alignments = self._compute_alignments(w, start_sequence, text_tokens, median_filter_width)
            else:
                for b in range(nb):
                    w = torch.softmax(scores[b:b + 1, :, :len(inputs[b]), :frames[b]], dim=-1)
                    alignments.append(self._compute_alignments(w, start_sequence, [text_tokens[b]],
                                                               median_filter_width)[0])
        results = []
        offset = len(start_sequence)
        for b in range(nb):
            probs = [float(token_probs[b, offset + t]) for t in range(len(text_tokens[b]))]
            results.append(WhisperAlignmentResult(alignments[b], probs))
        return results

    @staticmethod
    def _median_filter(x: torch.Tensor, width: int) -> torch.Tensor:
        rank = width // 2
        if x.shape[-1] <= rank or width <= 1:
            return x
        padded = F.pad(x.reshape(-1, 1, x.shape[-1]), (rank, rank), mode="reflect")
        return padded.unfold(-1, width, 1).median(dim=-1).values.reshape(x.shape)

    def _compute_alignments(self, w: torch.Tensor, start_sequence, text_tokens, median_filter_width):
        w = w.float()
        mean = w.mean(dim=-2, keepdim=True)
        var = ((w - mean) ** 2).mean(dim=-2, keepdim=True)
        w = (w - mean) / torch.sqrt(var)
        w = self._median_filter(w, median_filter_width)
        w = w.mean(dim=1).cpu().numpy()
        out = []
        sot_len = len(start_sequence)
        for b, text in enumerate(text_tokens):
            matrix = w[b, sot_len:sot_len + len(text) + 1, :]
            out.append(negative_dtw(matrix))
        return out
