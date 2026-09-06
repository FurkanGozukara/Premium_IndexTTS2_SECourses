"""Freeze a run's development speech benchmark using only its own dataset."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from .dataset_manifest import atomic_write_json, load_manifest


def record_identity(records: Sequence[Mapping[str, Any]]) -> str:
    fields = ("id", "audio", "text", "speaker", "language", "source_media", "split",
              "duration_s", "n_codes", "n_text_tokens")
    rows = [{key: row.get(key) for key in fields} for row in records]
    rows.sort(key=lambda row: str(row["id"]))
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def audio_path(dataset: str | Path, row: Mapping[str, Any]) -> Path:
    path = Path(str(row["audio"]))
    return (path if path.is_absolute() else Path(dataset) / path).resolve()


def audio_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_training_reference(records: Sequence[Mapping[str, Any]], dataset: str | Path) -> Mapping[str, Any] | None:
    """Choose a clear, moderate-length training clip without consulting past runs."""
    existing = [row for row in records if row.get("audio") and audio_path(dataset, row).is_file()]
    if not existing:
        return None
    return min(existing, key=lambda row: (
        float(row.get("asr_wer", 0) or 0),
        not bool(row.get("boundary_words_match", True)),
        abs(float(row.get("duration_s", 12) or 12) - 12),
        str(row["id"]),
    ))


def representative_records(records: Sequence[Mapping[str, Any]], count: int, seed: int) -> list[Mapping[str, Any]]:
    """Round-robin recordings, speakers and length bands instead of taking a prefix."""
    buckets: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        duration = float(row.get("duration_s", 0) or 0)
        band = 0 if duration < 10 else (1 if duration < 16 else 2)
        key = (str(row.get("speaker", "")), str(row.get("language", "EN")),
               str(row.get("source_media") or row["id"]), band)
        buckets[key].append(row)
    def order(value: Any) -> str:
        return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    for rows in buckets.values():
        rows.sort(key=lambda row: order(row["id"]))
    # Visit each recording once per pass; rotate its length bands on later passes.
    sources: dict[tuple[str, str, str], list[list[Mapping[str, Any]]]] = defaultdict(list)
    for key in sorted(buckets):
        sources[key[:3]].append(buckets[key])
    source_keys = sorted(sources, key=order)
    queues = {}
    for source in source_keys:
        bands = sources[source]
        queues[source] = [rows[i] for i in range(max(map(len, bands))) for rows in bands if i < len(rows)]
    chosen = []
    for index in range(max((len(rows) for rows in queues.values()), default=0)):
        for source in source_keys:
            if index < len(queues[source]):
                chosen.append(queues[source][index])
                if len(chosen) >= count:
                    return chosen
    return chosen


def build_speech_plan(config: Any, train_records: Sequence[Mapping[str, Any]],
                      val_records: Sequence[Mapping[str, Any]], run_dir: str | Path, *,
                      evaluation_dataset: str | Path | None = None, final_test: bool = False) -> dict[str, Any]:
    root = Path(run_dir) / "analysis" / "speech_evaluation"
    if final_test:
        root = root / "final_test"
    evaluation_dataset = evaluation_dataset or config.dataset_dir
    root.mkdir(parents=True, exist_ok=True)
    identity = record_identity([*train_records, *val_records])
    destination = root / "plan.json"
    if destination.is_file():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous.get("dataset_identity") != identity:
            raise ValueError("The saved speech evaluation plan belongs to a different dataset; use a new run name")
        return previous
    selected = representative_records(val_records, config.speech_eval_prompts, config.seed)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(str(row.get("speaker", "")), str(row.get("language") or "EN").upper())].append(row)
    groups = []
    for index, ((speaker, language), records) in enumerate(sorted(grouped.items()), 1):
        reference_row = choose_training_reference(
            [row for row in train_records if str(row.get("speaker", "")) == speaker], config.dataset_dir)
        if reference_row is None:
            raise ValueError(f"No training-only audio reference is available for speaker {speaker!r}")
        reference = root / "references" / f"speaker_{index:03d}.wav"
        reference.parent.mkdir(parents=True, exist_ok=True)
        # Prepared datasets contain WAVs. Preserve the source's format for imported datasets.
        source = audio_path(config.dataset_dir, reference_row)
        reference = reference.with_suffix(source.suffix)
        shutil.copy2(source, reference)
        prompts = [{"id": str(row["id"]), "text": str(row["text"]),
                    "audio": str(audio_path(evaluation_dataset, row)), "kind": "matched",
                    "audio_sha256": hashlib.sha256(audio_path(evaluation_dataset, row).read_bytes()).hexdigest(),
                    "source": str(row.get("source_media") or row["id"]),
                    "duration_s": float(row.get("duration_s", 0) or 0)} for row in records]
        if len(records) >= 3:
            long_rows = sorted(records, key=lambda row: (str(row.get("source_media", "")),
                                                       float(row.get("source_start_s", 0) or 0)))[:4]
            prompts.append({"id": f"long_{index}", "text": " ".join(str(row["text"]) for row in long_rows),
                            "audio": "", "kind": "long_form", "source": "held-out concatenation",
                            "duration_s": None})
        groups.append({"id": f"group_{index:03d}", "speaker": speaker, "language": language,
                       "reference": str(reference.resolve()), "reference_record_id": str(reference_row["id"]),
                       "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(), "prompts": prompts})
    train_sources = {str(row.get("source_media") or row["id"]) for row in train_records}
    val_sources = {str(row.get("source_media") or row["id"]) for row in val_records}
    warnings = []
    if train_sources & val_sources:
        warnings.append("Training and validation share source recordings; scores measure held-out clips, not new recording sessions.")
    if len(val_sources) < 3:
        label = "final-test" if final_test else "validation"
        warnings.append(f"Only {len(val_sources)} {label} recording(s); more recording sessions would improve coverage.")
    if len(selected) < config.speech_eval_prompts:
        warnings.append(f"The dataset supplies only {len(selected)} distinct validation prompts.")
    all_speakers = {str(row.get("speaker", "")) for row in train_records}
    evaluated_speakers = {group["speaker"] for group in groups}
    if all_speakers - evaluated_speakers:
        warnings.append(f"{len(all_speakers - evaluated_speakers)} training speaker(s) have no prompts in this benchmark; the recommendation does not establish their quality.")
    plan = {"version": 1, "dataset_identity": identity, "dataset_dir": str(Path(evaluation_dataset).resolve()),
            "training_items": len(train_records), "validation_items": len(val_records),
            "validation_sources": len(val_sources), "training_sources": len(train_sources),
            "source_overlap": sorted(train_sources & val_sources), "groups": groups,
            "seeds": [(int(config.seed) + 104729 * i) % 2**32 for i in range(config.speech_eval_seeds)],
            "candidate_limit": config.speech_eval_candidates, "warnings": warnings,
            "policy": {"max_wer_increase": config.speech_eval_max_wer_increase,
                       "max_speaker_drop": config.speech_eval_max_speaker_drop},
            "scope": "final test; used only after checkpoint selection is frozen" if final_test else
                     "development; no previous training or final-test targets are consulted"}
    atomic_write_json(destination, plan)
    return plan


def build_final_test_plan(config: Any, train_records: Sequence[Mapping[str, Any]],
                          development_records: Sequence[Mapping[str, Any]], run_dir: str | Path) -> dict[str, Any] | None:
    if not config.final_test_dataset:
        return None
    dataset = Path(config.final_test_dataset).expanduser().resolve()
    rows = load_manifest(dataset)
    if any("split" in row for row in rows):
        # Curated test datasets can carry training-reference rows for loss evaluation.
        rows = [row for row in rows if row.get("split") == "val"]
    if not rows:
        raise ValueError("The final-test dataset has no evaluation recordings")
    def source(row: Mapping[str, Any]) -> str:
        return str(row.get("source_media") or row.get("audio") or row["id"]).replace("\\", "/").casefold()
    development = [*train_records, *development_records]
    used_sources = {source(row) for row in development}
    if used_sources & {source(row) for row in rows}:
        raise ValueError("Final-test source recordings overlap training or validation; reserve separate recordings")
    used_audio = {str(audio_path(config.dataset_dir, row)) for row in development}
    if used_audio & {str(audio_path(dataset, row)) for row in rows}:
        raise ValueError("Final-test audio overlaps training or validation")
    test_hashes = {audio_digest(audio_path(dataset, row)) for row in rows}
    if any(audio_digest(audio_path(config.dataset_dir, row)) in test_hashes for row in development):
        raise ValueError("Final-test audio contains an exact copy of a training or validation file")
    return build_speech_plan(config, train_records, rows, run_dir, evaluation_dataset=dataset, final_test=True)


__all__ = ["audio_path", "build_speech_plan", "build_final_test_plan", "choose_training_reference", "record_identity", "representative_records"]
