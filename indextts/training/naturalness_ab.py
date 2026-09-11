"""Naturalness A/B harness: render one voice with several decoding or conditioning variants and compare.

The variants target the causes of flat delivery found on 10 September 2026:
mode-seeking beam sampling, the very high repetition penalty, the decoder
adapter, comma-driven pauses, the reference clip and the emotion prompt.
Every variant renders the same held-out sentences with the same seeds; the
clips are then measured against the person's real recordings (transcript
error, identity, style, pauses) and with the prosody statistics of
``prosody_metrics`` (pitch variability, loudness dynamics, articulation rate).
Listening bundles for blind ranking are written beside the report.

This module holds the CPU-only pieces (variant table, sentence selection,
report and bundle building); ``tools/naturalness_ab.py`` renders on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import re
import shutil
from typing import Any, Mapping, Sequence

from indextts.utils.atomic_json import write_json_atomic

from .prosody_metrics import PROSODY_KEYS, compare_prosody, prosody_table


@dataclass(frozen=True)
class Variant:
    """One rendering condition: inference overrides, decoder choice, text transform, prompts."""

    name: str
    description: str
    infer: dict[str, Any] = field(default_factory=dict)
    decoder: str | None = None  # None keeps the deployed choice; "none" removes it; a path selects a file
    decoder_strength: float | None = None
    text_transform: str = "none"  # none | strip_commas | strip_clause_marks
    reference: str | None = None  # speaker prompt override
    emotion_reference: str | None = None  # separate emotion prompt
    emo_alpha: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description, "infer": dict(self.infer), "decoder": self.decoder,
            "decoder_strength": self.decoder_strength, "text_transform": self.text_transform,
            "reference": self.reference, "emotion_reference": self.emotion_reference, "emo_alpha": self.emo_alpha,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Variant":
        return cls(
            name=str(value["name"]), description=str(value.get("description", "")), infer=dict(value.get("infer") or {}),
            decoder=value.get("decoder"), decoder_strength=value.get("decoder_strength"),
            text_transform=str(value.get("text_transform") or "none"), reference=value.get("reference"),
            emotion_reference=value.get("emotion_reference"), emo_alpha=value.get("emo_alpha"),
        )


DEFAULT_VARIANTS: tuple[Variant, ...] = (
    Variant("deployed", "As the app deploys the voice: 3 beams with sampling, temperature 0.8, top-p 0.8, top-k 30, repetition penalty 10, decoder adapter, saved speaking rate."),
    Variant("sampling", "Pure sampling: 1 beam, otherwise the deployed settings.", infer={"num_beams": 1}),
    Variant("sampling_warm", "Pure sampling with a wider distribution: 1 beam, temperature 1.0, top-p 0.95, top-k 50.",
            infer={"num_beams": 1, "temperature": 1.0, "top_p": 0.95, "top_k": 50}),
    Variant("repetition_3", "Deployed settings with repetition penalty 3 instead of 10.", infer={"repetition_penalty": 3.0}),
    Variant("sampling_repetition_1p5", "1 beam and repetition penalty 1.5 (the range every other open model uses).",
            infer={"num_beams": 1, "repetition_penalty": 1.5}),
    Variant("no_decoder", "Deployed GPT settings without the voice decoder adapter.", decoder="none"),
    Variant("decoder_0p6", "Voice decoder adapter at strength 0.6.", decoder_strength=0.6),
    Variant("no_commas", "Deployed settings; commas removed from the text so the model paces the clause itself.", text_transform="strip_commas"),
    Variant("silence_cap", "Deployed settings; runs of more than 10 silence tokens (about 200 ms) are trimmed.", infer={"max_consecutive_silence": 10}),
    Variant("cfg_0p5_steps_40", "Diffusion decoder at guidance 0.5 with 40 steps.", infer={"inference_cfg_rate": 0.5, "diffusion_steps": 40}),
    Variant("rate_1", "Speaking rate 1.0 (no mel stretch) when the saved rate differs from 1.", infer={"latent_multiplier": 1.72}),
)
REPETITION_VARIANTS: tuple[Variant, ...] = (
    Variant("window_8", "Deployed settings; repetition penalty 10 applied only to the last 8 codes (about 0.3 s).",
            infer={"repetition_window": 8}),
    Variant("window_16", "Deployed settings; repetition penalty 10 applied only to the last 16 codes (about 0.6 s).",
            infer={"repetition_window": 16}),
    Variant("window_32", "Deployed settings; repetition penalty 10 applied only to the last 32 codes (about 1.3 s).",
            infer={"repetition_window": 32}),
    Variant("window_64", "Deployed settings; repetition penalty 10 applied only to the last 64 codes (about 2.6 s).",
            infer={"repetition_window": 64}),
    Variant("window_16_penalty_3", "Repetition penalty 3 applied only to the last 16 codes.",
            infer={"repetition_window": 16, "repetition_penalty": 3.0}),
)
EXPRESSIVE_VARIANTS: tuple[Variant, ...] = (
    Variant("expressive_reference", "The most expressive clean training clip as the speaker and emotion prompt.", reference="{expressive}"),
    Variant("expressive_emotion", "Deployed reference for identity, the most expressive clip as the emotion prompt (alpha 0.65).",
            emotion_reference="{expressive}", emo_alpha=0.65),
    Variant("expressive_emotion_sampling", "Expressive emotion prompt with pure sampling.",
            emotion_reference="{expressive}", emo_alpha=0.65, infer={"num_beams": 1}),
)
_CLAUSE_MARKS = re.compile(r"\s*[,;:]\s*")
_COMMA = re.compile(r"\s*,\s*")


def transform_text(text: str, transform: str) -> str:
    """Apply a variant's text transform; punctuation removal keeps sentence ends."""

    value = str(text or "")
    if transform == "strip_commas":
        return _COMMA.sub(" ", value).strip()
    if transform == "strip_clause_marks":
        return _CLAUSE_MARKS.sub(" ", value).strip()
    return value


def load_variants(source: str | Path | None, *, expressive_reference: str | None = None,
                  include_extra: bool = False) -> list[Variant]:
    """The default variant table, the expressive additions when a clip is given, or a JSON file of variants.

    ``include_extra`` adds the optional tables (repetition windows) so they can be selected by name.
    """

    if source and str(source) not in {"default", ""}:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        items = payload.get("variants") if isinstance(payload, Mapping) else payload
        variants = [Variant.from_dict(item) for item in items]
    else:
        variants = list(DEFAULT_VARIANTS)
        if include_extra:
            variants.extend(REPETITION_VARIANTS)
        if expressive_reference:
            variants.extend(EXPRESSIVE_VARIANTS)
    resolved: list[Variant] = []
    for variant in variants:
        reference = variant.reference
        emotion = variant.emotion_reference
        if expressive_reference:
            reference = reference.replace("{expressive}", expressive_reference) if reference else reference
            emotion = emotion.replace("{expressive}", expressive_reference) if emotion else emotion
        if (reference and "{expressive}" in reference) or (emotion and "{expressive}" in emotion):
            continue  # needs an expressive clip that was not supplied
        resolved.append(Variant(variant.name, variant.description, dict(variant.infer), variant.decoder, variant.decoder_strength,
                                variant.text_transform, reference, emotion, variant.emo_alpha))
    return resolved


def load_sentences(run_dir: str | Path, source: str = "final_test", *, limit: int = 0) -> list[dict[str, Any]]:
    """Held-out sentences with the person's real recording: the final test, the development benchmark, or a JSON list."""

    root = Path(run_dir)
    if source in {"final_test", "development"}:
        report_path = root / "analysis" / "speech_evaluation" / ("final_test/report.json" if source == "final_test" else "report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        seen: dict[str, dict[str, Any]] = {}
        for cell in report.get("cells") or []:
            if cell.get("kind") != "matched" or not cell.get("real_audio") or cell.get("invalid_audio"):
                continue
            text = str(cell.get("text") or "").strip()
            if text and text not in seen and Path(str(cell["real_audio"])).is_file():
                seen[text] = {"id": str(cell.get("prompt_id") or len(seen)), "text": text, "real_audio": str(cell["real_audio"]),
                              "language": str(cell.get("language") or "EN")}
        sentences = list(seen.values())
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        items = payload.get("sentences") if isinstance(payload, Mapping) else payload
        sentences = []
        for item in items:
            audio = item.get("real_audio") or item.get("audio")
            if item.get("text") and audio and Path(str(audio)).is_file():
                sentences.append({"id": str(item.get("id") or len(sentences)), "text": str(item["text"]),
                                  "real_audio": str(audio), "language": str(item.get("language") or "EN"),
                                  "bucket": item.get("bucket")})
    sentences.sort(key=lambda item: item["id"])
    return sentences[:limit] if limit > 0 else sentences


def summarize_variant(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Speech metrics (from speech_metrics.measure_clips rows) and prosody comparison of one variant."""

    from .speech_metrics import summarize

    defaults = {"invalid_audio": False, "possible_truncation": False, "possible_repetition": False, "error_rate": 0.0,
                "errors": 0, "units": 0, "duration_s": 0.0, "start_matches": True, "end_matches": True}
    speech = summarize([{**defaults, **row} for row in rows]) if rows else {}
    pairs = [(row["prosody"], row["real_prosody"]) for row in rows if row.get("prosody") and row.get("real_prosody")]
    prosody = compare_prosody(pairs)
    real_similarity = [float(row["speaker_similarity_real"]) for row in rows if row.get("speaker_similarity_real") is not None]
    style = [float(row["style_similarity_real"]) for row in rows if row.get("style_similarity_real") is not None]
    pause_ratio = [float(row["pause_ratio_vs_real"]) for row in rows if row.get("pause_ratio_vs_real") is not None]
    return {
        "clips": len(rows),
        "speech": speech,
        "speaker_similarity_real_mean": round(sum(real_similarity) / len(real_similarity), 4) if real_similarity else None,
        "style_similarity_real_mean": round(sum(style) / len(style), 4) if style else None,
        "pause_ratio_vs_real_mean": round(sum(pause_ratio) / len(pause_ratio), 3) if pause_ratio else None,
        "prosody": prosody,
    }


def render_report_markdown(report: Mapping[str, Any]) -> str:
    """Human-readable report: one row per variant, then the prosody table."""

    lines = [f"# Naturalness A/B: {report.get('run_dir', '')}", ""]
    lines.append(f"Checkpoint: `{report.get('checkpoint', '')}` · sentences: {report.get('sentence_count', 0)} · seeds: {report.get('seeds', [])}")
    lines.append("")
    lines.append("| Variant | Corpus word error | Speaker sim vs real | Style sim vs real | Pause time vs real | Liveliness ratio | Pitch std (st) gen vs real | Energy std (dB) gen vs real | Words/s gen vs real |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|---|")
    summaries = report.get("variants") or {}
    for name, entry in summaries.items():
        speech = entry.get("speech") or {}
        prosody = entry.get("prosody") or {}
        f0 = prosody.get("f0_std_st") or {}
        energy = prosody.get("energy_std_db") or {}
        rate = prosody.get("words_per_s") or {}
        error = speech.get("corpus_error_rate")
        lines.append(
            f"| {name} | {(error * 100):.2f}% | {entry.get('speaker_similarity_real_mean') or 0:.3f} | "
            f"{entry.get('style_similarity_real_mean') or 0:.3f} | {entry.get('pause_ratio_vs_real_mean') or 0:.2f} | "
            f"{prosody.get('liveliness_ratio') or 0:.3f} | {f0.get('generated_mean', 0):.2f} vs {f0.get('real_mean', 0):.2f} | "
            f"{energy.get('generated_mean', 0):.2f} vs {energy.get('real_mean', 0):.2f} | "
            f"{rate.get('generated_mean', 0):.2f} vs {rate.get('real_mean', 0):.2f} |"
            if isinstance(error, (int, float)) else f"| {name} | – | – | – | – | – | – | – | – |"
        )
    lines.append("")
    lines.append("## Prosody against the real recordings")
    lines.append("")
    lines.append(prosody_table({name: (entry.get("prosody") or {}) for name, entry in summaries.items()}, PROSODY_KEYS))
    lines.append("")
    lines.append("Variants:")
    for variant in report.get("variant_table") or []:
        lines.append(f"- **{variant['name']}**: {variant['description']}")
    return "\n".join(lines) + "\n"


def build_listening_bundles(
    rows_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    out_dir: str | Path,
    *,
    variants: Sequence[str] | None = None,
    seed: int = 7,
    max_sets: int = 0,
) -> list[dict[str, Any]]:
    """Blind sets: Reference.wav (the real recording) plus shuffled A, B, C... clips of the chosen variants.

    The answer keys go to ``keys.json`` next to the ``blind`` folder so a listener
    that only sees a set folder cannot learn which letter is which variant.
    """

    root = Path(out_dir)
    blind = root / "blind"
    blind.mkdir(parents=True, exist_ok=True)
    chosen = list(variants or rows_by_variant)
    index: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for name in chosen:
        for row in rows_by_variant.get(name, []):
            key = (str(row.get("prompt_id")), int(row.get("seed", 0)))
            index.setdefault(key, {})[name] = row
    rng = random.Random(seed)
    keys: list[dict[str, Any]] = []
    for number, (key, clips) in enumerate(sorted(index.items()), start=1):
        if len(clips) < 2 or (max_sets and number > max_sets):
            continue
        set_dir = blind / f"set_{number:03d}"
        set_dir.mkdir(exist_ok=True)
        names = list(clips)
        rng.shuffle(names)
        letters = [chr(ord("A") + position) for position in range(len(names))]
        first = clips[names[0]]
        shutil.copy2(first["real_audio"], set_dir / "Reference.wav")
        for letter, name in zip(letters, names):
            shutil.copy2(clips[name]["audio"], set_dir / f"{letter}.wav")
        (set_dir / "prompt.md").write_text(
            "# Blind naturalness ranking\n\n"
            f"Expected text: {first.get('text', '')}\n\n"
            "Reference.wav is the real speaker. Rate every lettered clip for naturalness (1-5), voice similarity (1-5), "
            "pronunciation (1-5) and pace, then rank the letters from most to least natural; ties allowed.\n",
            encoding="utf-8",
        )
        keys.append({"set": set_dir.name, "prompt_id": key[0], "seed": key[1], "text": first.get("text", ""),
                     "letters": {letter: name for letter, name in zip(letters, names)}})
    write_json_atomic(root / "keys.json", {"sets": keys, "variants": chosen}, indent=2, ensure_ascii=False)
    return keys


__all__ = [
    "DEFAULT_VARIANTS",
    "EXPRESSIVE_VARIANTS",
    "Variant",
    "build_listening_bundles",
    "load_sentences",
    "load_variants",
    "render_report_markdown",
    "summarize_variant",
    "transform_text",
]
