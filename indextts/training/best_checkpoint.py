"""Naming of the lowest-validation-loss checkpoint and migration of older trainings.

A training keeps its best checkpoint as ``<run>/best/<run>_best.safetensors`` so the
word ``best`` is part of the file name itself, not only of the folder. Trainings
saved before this convention used ``<run>/best/<run>.safetensors``; the migration
below renames those files, their sibling files (resume state, decoder adapter,
reference copy), and every stored path reference inside the training folder's
JSON and Markdown reports, so "Use best checkpoint" and the Checkpoint Grid keep
resolving them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import tempfile

BEST_FOLDER = "best"
BEST_CHECKPOINT_SUFFIX = "_best"
DECODER_ADAPTER_TAIL = ".s2mel.safetensors"
# Files that travel with a checkpoint and share its stem, most specific tail first.
CHECKPOINT_SIBLING_TAILS = (DECODER_ADAPTER_TAIL, ".train_state.pt", "_reference.wav", ".safetensors")
TRAINING_TERMINAL_PHASES = frozenset(
    {"complete", "stopped", "failed", "error", "cancelled", "canceled"}
)
_REFERENCE_FILE_SUFFIXES = frozenset({".json", ".md"})


def best_checkpoint_path(adapter_dir: str | os.PathLike[str], name: str) -> Path:
    """Where a training saves its lowest-validation-loss checkpoint."""
    return Path(adapter_dir) / BEST_FOLDER / f"{name}{BEST_CHECKPOINT_SUFFIX}.safetensors"


def legacy_best_checkpoint_path(adapter_dir: str | os.PathLike[str], name: str) -> Path:
    """Where trainings saved that checkpoint before the ``_best`` suffix existed."""
    return Path(adapter_dir) / BEST_FOLDER / f"{name}.safetensors"


def is_best_checkpoint_name(path: str | os.PathLike[str]) -> bool:
    source = Path(path)
    return (
        source.parent.name.lower() == BEST_FOLDER
        and source.suffix.lower() == ".safetensors"
        and source.stem.lower().endswith(BEST_CHECKPOINT_SUFFIX)
    )


@dataclass
class BestCheckpointRename:
    run_dir: Path
    old: Path
    new: Path
    siblings: list[tuple[Path, Path]] = field(default_factory=list)
    rewritten_files: list[Path] = field(default_factory=list)
    skipped: str = ""

    @property
    def run(self) -> str:
        return self.run_dir.name


def _split_sibling(name: str) -> tuple[str, str] | None:
    lowered = name.lower()
    for tail in CHECKPOINT_SIBLING_TAILS:
        if lowered.endswith(tail):
            return name[: -len(tail)], name[-len(tail):]
    return None


def _run_is_active(run_dir: Path) -> bool:
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return False
    if not isinstance(status, dict):
        return False
    phase = str(status.get("phase", "")).strip().lower()
    return bool(phase and phase not in TRAINING_TERMINAL_PHASES)


def _atomic_write_text(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _reference_pattern(stem: str) -> re.Pattern[str]:
    tails = "|".join(re.escape(tail) for tail in CHECKPOINT_SIBLING_TAILS)
    # ``best/name.safetensors``, ``best\name.safetensors`` and the JSON-escaped ``best\\name.safetensors``.
    return re.compile(rf"({re.escape(BEST_FOLDER)}[\\/]+){re.escape(stem)}({tails})(?![A-Za-z0-9_.])", re.IGNORECASE)


def _rewrite_references(run_dir: Path, stem: str) -> list[Path]:
    pattern = _reference_pattern(stem)
    rewritten: list[Path] = []
    for candidate in sorted(run_dir.rglob("*"), key=lambda item: str(item).lower()):
        if not candidate.is_file() or candidate.suffix.lower() not in _REFERENCE_FILE_SUFFIXES:
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        updated = pattern.sub(lambda match: f"{match.group(1)}{stem}{BEST_CHECKPOINT_SUFFIX}{match.group(2)}", text)
        if updated == text:
            continue
        _atomic_write_text(candidate, updated)
        rewritten.append(candidate)
    return rewritten


def migrate_run_best_checkpoints(
    adapter_dir: str | os.PathLike[str], *, name: str | None = None, skip_active: bool = True
) -> list[BestCheckpointRename]:
    """Give one training folder's legacy best checkpoint the ``_best`` suffix.

    The legacy file is exactly ``best/<name>.safetensors`` where ``name`` is the
    training name (the folder name unless given), so a training whose own name
    ends in ``_best`` is still migrated correctly. Returns one record per legacy
    checkpoint found; a record with ``skipped`` set was left untouched because
    the training is still running or the target name already exists.
    """
    run_dir = Path(adapter_dir).expanduser().resolve()
    best_dir = run_dir / BEST_FOLDER
    stem = str(name or run_dir.name)
    if not best_dir.is_dir() or not stem:
        return []
    old = best_dir / f"{stem}.safetensors"
    new = best_dir / f"{stem}{BEST_CHECKPOINT_SUFFIX}.safetensors"
    if not old.is_file():
        return []
    record = BestCheckpointRename(run_dir=run_dir, old=old, new=new)
    if skip_active and _run_is_active(run_dir):
        record.skipped = "training is still running"
        return [record]
    if new.exists():
        record.skipped = f"{new.name} already exists"
        return [record]
    moves: list[tuple[Path, Path]] = []
    for sibling in sorted(best_dir.iterdir(), key=lambda item: item.name.lower()):
        if not sibling.is_file():
            continue
        parts = _split_sibling(sibling.name)
        if parts is None or parts[0] != stem:
            continue
        target = best_dir / f"{stem}{BEST_CHECKPOINT_SUFFIX}{parts[1]}"
        if not target.exists():
            moves.append((sibling, target))
    for source, target in moves:
        os.replace(source, target)
        if source != old:
            record.siblings.append((source, target))
    record.rewritten_files = _rewrite_references(run_dir, stem)
    return [record]


def migrate_legacy_best_checkpoints(root: str | os.PathLike[str]) -> list[BestCheckpointRename]:
    """Migrate every training folder under ``root`` (recursively, skipping analysis output)."""
    base = Path(root).expanduser()
    if not base.is_dir():
        return []
    records: list[BestCheckpointRename] = []
    for best_dir in sorted(base.rglob(BEST_FOLDER), key=lambda item: str(item).lower()):
        if not best_dir.is_dir() or best_dir.name != BEST_FOLDER:
            continue
        if any(part.lower() == "analysis" or part.startswith(".") for part in best_dir.relative_to(base).parts[:-1]):
            continue
        records.extend(migrate_run_best_checkpoints(best_dir.parent))
    return records


def describe_migration(records: list[BestCheckpointRename]) -> list[str]:
    lines: list[str] = []
    for record in records:
        if record.skipped:
            lines.append(f"kept {record.run}/best/{record.old.name}: {record.skipped}")
            continue
        extras = f", {len(record.siblings)} sibling file(s)" if record.siblings else ""
        refs = f", {len(record.rewritten_files)} report(s) updated" if record.rewritten_files else ""
        lines.append(f"renamed {record.run}/best/{record.old.name} -> {record.new.name}{extras}{refs}")
    return lines


__all__ = [
    "BEST_CHECKPOINT_SUFFIX",
    "BEST_FOLDER",
    "BestCheckpointRename",
    "TRAINING_TERMINAL_PHASES",
    "best_checkpoint_path",
    "describe_migration",
    "is_best_checkpoint_name",
    "legacy_best_checkpoint_path",
    "migrate_legacy_best_checkpoints",
    "migrate_run_best_checkpoints",
]
