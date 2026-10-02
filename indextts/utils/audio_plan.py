"""Audio assembly shared by speech backends; timings are sample-rate aware."""

import math
import torch
import torch.nn.functional as F

SAMPLE_RATE = 22050

# Audio-plan items: ("segment", index) renders a text segment; the others insert silence:
# "silence" is the section gap, "pause" an explicit pause tag (never shortened afterwards),
# "sentence_gap" the pause between two sentences measured from the last word to the next.
PLAN_SILENCE_KINDS = frozenset({"silence", "pause", "sentence_gap"})
_EDGE_GATE_DBFS = -40.0


def _edge_quiet_samples(wav, sampling_rate=SAMPLE_RATE):
    """(leading, trailing) quiet samples of a segment: 10 ms frames below -40 dBFS at the edges."""

    if wav is None or wav.numel() == 0:
        return 0, 0
    audio = wav.detach().float()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    peak_scale = 32767.0 if audio.abs().max().item() > 2.0 else 1.0
    mono = audio.mean(dim=0) / peak_scale
    frame = max(1, int(round(float(sampling_rate) * 0.01)))
    count = mono.numel() // frame
    if count == 0:
        return 0, 0
    rms = mono[: count * frame].view(count, frame).square().mean(dim=1).add(1e-12).sqrt()
    loud = torch.nonzero(rms >= 10 ** (_EDGE_GATE_DBFS / 20.0), as_tuple=False).flatten()
    if loud.numel() == 0:
        return int(mono.numel()), 0
    leading = int(loud[0].item()) * frame
    trailing = int(mono.numel()) - (int(loud[-1].item()) + 1) * frame
    return max(0, leading), max(0, trailing)


def _stream_silence_samples(plan, index, segment_wavs, sampling_rate=SAMPLE_RATE):
    """Silence to stream for plan item ``index``; a sentence gap is shortened by the previous segment's tail."""

    kind, value = plan[index]
    if kind != "sentence_gap":
        return int(value)
    previous = next((plan[position][1] for position in range(index - 1, -1, -1) if plan[position][0] == "segment"), None)
    if previous is None or previous >= len(segment_wavs) or segment_wavs[previous] is None:
        return int(value)
    _leading, trailing = _edge_quiet_samples(segment_wavs[previous], sampling_rate)
    return max(0, int(value) - trailing)


def assemble_audio_plan(segment_wavs, plan, sampling_rate=SAMPLE_RATE):
    """Concatenate rendered segments and planned silences.

    Returns ``(wav, protected)`` where ``protected`` lists the sample ranges of explicit
    pause tags, so a later pause cap can leave them untouched. A ``sentence_gap`` item
    makes the pause from the last word of one segment to the first word of the next
    equal to its value: the quiet tails the model generated count towards it, extra
    tail is trimmed, and the rest is inserted as silence.
    """

    template = next((item for item in segment_wavs if item is not None), None)
    channels = int(template.shape[0]) if template is not None else 1
    dtype = template.dtype if template is not None else torch.float32
    parts = []
    protected = []
    offset = 0
    pending_gap = None
    for kind, value in plan:
        if kind == "segment":
            wav = segment_wavs[value]
            if pending_gap is not None and wav is not None:
                target = int(pending_gap)
                previous = parts[-1] if parts else None
                _, trailing = _edge_quiet_samples(previous, sampling_rate) if previous is not None else (0, 0)
                leading, _ = _edge_quiet_samples(wav, sampling_rate)
                gap = target - trailing - leading
                if gap >= 0:
                    if gap:
                        parts.append(torch.zeros(channels, gap, dtype=dtype))
                        offset += gap
                else:
                    excess = -gap
                    cut_previous = min(excess, trailing)
                    if cut_previous > 0 and previous is not None:
                        parts[-1] = previous[..., : previous.shape[-1] - cut_previous]
                        offset -= cut_previous
                        excess -= cut_previous
                    cut_next = min(excess, leading)
                    if cut_next > 0:
                        wav = wav[..., cut_next:]
            pending_gap = None
            if wav is not None:
                parts.append(wav)
                offset += int(wav.shape[-1])
            continue
        if kind == "sentence_gap":
            pending_gap = int(value)
            continue
        samples = int(value)
        if samples <= 0:
            continue
        if kind == "pause":
            protected.append((offset, offset + samples))
        parts.append(torch.zeros(channels, samples, dtype=dtype))
        offset += samples
    if not parts:
        return torch.zeros(channels, 0, dtype=dtype), protected
    return torch.cat(parts, dim=1), protected


def trim_segment_silence(wav, sampling_rate=SAMPLE_RATE, minimum_silence_ms=0):
    """Trim sufficiently long quiet leading/trailing runs using a fixed RMS gate."""

    minimum_ms = max(0.0, float(minimum_silence_ms or 0))
    if minimum_ms <= 0 or wav.numel() == 0:
        return wav
    audio = wav.detach().float()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    peak_scale = 32767.0 if audio.abs().max().item() > 2.0 else 1.0
    mono = audio.abs().mean(dim=0) / peak_scale
    frame_size = max(1, int(round(float(sampling_rate) * 0.01)))
    frame_count = int(math.ceil(mono.numel() / frame_size))
    padded = F.pad(mono, (0, frame_count * frame_size - mono.numel()))
    rms = padded.view(frame_count, frame_size).square().mean(dim=1).sqrt()
    active = torch.nonzero(rms >= 10 ** (-45.0 / 20.0), as_tuple=False).flatten()
    if active.numel() == 0:
        return wav

    first = min(mono.numel(), int(active[0].item()) * frame_size)
    last = min(mono.numel(), (int(active[-1].item()) + 1) * frame_size)
    minimum_samples = int(round(float(sampling_rate) * minimum_ms / 1000.0))
    start = first if first >= minimum_samples else 0
    end = last if mono.numel() - last >= minimum_samples else mono.numel()
    if start >= end:
        return wav
    return wav[..., start:end]

def fit_target_samples(wav, target_samples, mode):
    target_samples = max(0, int(target_samples))
    current = int(wav.shape[-1])
    if mode == "pad" and current < target_samples:
        return F.pad(wav, (0, target_samples - current))
    if mode == "trim" and current > target_samples:
        return wav[..., :target_samples]
    return wav
