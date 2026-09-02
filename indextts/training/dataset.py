"""Cached-feature dataset and length-bucketed batching."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, Dataset

from indextts.utils.tokenizer import lang_to_token

from .dataset_manifest import load_manifest


def _stable_unit_interval(*parts: Any) -> float:
    value = "\x1f".join(str(part) for part in parts).encode("utf-8", "surrogatepass")
    integer = int.from_bytes(hashlib.sha256(value).digest()[:8], "big")
    return integer / float(2**64)


def _cache_path(dataset_dir: Path, row: Mapping[str, Any]) -> Path:
    row_id = str(row["id"])
    direct = dataset_dir / "cache" / f"{row_id}.pt"
    if direct.is_file():
        return direct
    cache_value = row.get("cache")
    if isinstance(cache_value, Mapping):
        relative = cache_value.get("path") or cache_value.get("feature_path")
        if relative:
            candidate = Path(str(relative))
            return candidate if candidate.is_absolute() else dataset_dir / candidate
    raise FileNotFoundError(f"cached features are missing for {row_id}: {direct}")


def _read_cache_header(path: Path) -> tuple[int, int, str]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise TypeError(f"feature cache must be a dictionary: {path}")
    codes = value.get("codes")
    text = value.get("text_tokens")
    if not isinstance(codes, torch.Tensor) or not isinstance(text, torch.Tensor):
        raise ValueError(f"feature cache lacks codes/text_tokens: {path}")
    return int(codes.numel()), int(text.numel()), str(value.get("speaker", ""))


class LoraTrainDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        val_fraction: float = 0.05,
        seed: int = 42,
        max_codes: int = 1500,
        max_text_tokens: int = 600,
        speaker_ref_mode: str = "mixed",
        emo_ref_mode: str = "self",
    ) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.split = str(split).lower()
        if self.split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        self.val_fraction = min(0.5, max(0.0, float(val_fraction)))
        self.seed = int(seed)
        self.max_codes = int(max_codes)
        self.max_text_tokens = int(max_text_tokens)
        self.speaker_ref_mode = str(speaker_ref_mode).lower()
        if self.speaker_ref_mode not in {"self", "other", "mixed"}:
            raise ValueError("speaker_ref_mode must be self, other, or mixed")
        self.emo_ref_mode = str(emo_ref_mode).lower()
        if self.emo_ref_mode not in {"self", "other", "mixed", "follow_speaker"}:
            raise ValueError(
                "emo_ref_mode must be self, other, mixed, or follow_speaker"
            )
        self.epoch = 0

        manifest_rows = load_manifest(self.dataset_dir)
        if not manifest_rows:
            raise FileNotFoundError(f"manifest.jsonl is empty or missing in {self.dataset_dir}")

        all_records: list[dict[str, Any]] = []
        dropped = 0
        for row in manifest_rows:
            if not row.get("id"):
                continue
            path = _cache_path(self.dataset_dir, row)
            n_codes = row.get("n_codes")
            n_text = row.get("n_text_tokens")
            speaker = str(row.get("speaker", ""))
            if n_codes is None or n_text is None:
                n_codes, n_text, cached_speaker = _read_cache_header(path)
                speaker = speaker or cached_speaker
            n_codes = int(n_codes)
            n_text = int(n_text)
            if n_codes <= 0 or n_text <= 0 or n_codes > self.max_codes or n_text > self.max_text_tokens:
                dropped += 1
                continue
            record = dict(row)
            record.update(
                {
                    "id": str(row["id"]),
                    "cache_path": path,
                    "n_codes": n_codes,
                    "n_text_tokens": n_text,
                    "speaker": speaker,
                }
            )
            all_records.append(record)

        if not all_records:
            raise ValueError("no cached samples remain after length filtering")
        val_ids = self._validation_ids(all_records)
        self.records = [
            record
            for record in all_records
            if (record["id"] in val_ids) == (self.split == "val")
        ]
        # A zero validation fraction intentionally produces an empty validation
        # dataset. Training must still contain every record.
        if not self.records and self.split == "train":
            raise ValueError("the deterministic split produced an empty training set")

        self.lengths = [int(record["n_codes"]) for record in self.records]
        self.dropped_count = dropped
        self._all_records = all_records
        self._speaker_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in all_records:
            self._speaker_records[str(record["speaker"])].append(record)

    def _validation_ids(self, records: Sequence[Mapping[str, Any]]) -> set[str]:
        if self.val_fraction <= 0.0 or len(records) < 2:
            return set()
        selected = {
            str(record["id"])
            for record in records
            if _stable_unit_interval(self.seed, record["id"], "split") < self.val_fraction
        }
        # Small datasets still need a useful validation sample, while preserving
        # at least one training sample.
        if not selected:
            best = min(records, key=lambda record: _stable_unit_interval(self.seed, record["id"], "split"))
            selected.add(str(best["id"]))
        if len(selected) == len(records):
            keep_train = max(records, key=lambda record: _stable_unit_interval(self.seed, record["id"], "split"))
            selected.remove(str(keep_train["id"]))
        return selected

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))

    def __len__(self) -> int:
        return len(self.records)

    def _other_reference(
        self,
        record: Mapping[str, Any],
        index: int,
        tag: str = "reference",
    ) -> Mapping[str, Any] | None:
        choices = [
            candidate
            for candidate in self._speaker_records.get(str(record["speaker"]), [])
            if candidate["id"] != record["id"]
        ]
        if not choices:
            return None
        unit = _stable_unit_interval(self.seed, self.epoch, record["id"], index, tag)
        return choices[min(len(choices) - 1, int(unit * len(choices)))]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        cached = torch.load(record["cache_path"], map_location="cpu", weights_only=False)
        if not isinstance(cached, Mapping):
            raise TypeError(f"invalid feature cache {record['cache_path']}")

        use_other = self.speaker_ref_mode == "other"
        if self.speaker_ref_mode == "mixed":
            use_other = _stable_unit_interval(self.seed, self.epoch, record["id"], "mixed") < 0.5
        speaker_cache = cached
        reference_id = record["id"]
        if use_other:
            other = self._other_reference(record, index)
            if other is not None:
                speaker_cache = torch.load(other["cache_path"], map_location="cpu", weights_only=False)
                reference_id = other["id"]

        emo_cache = cached
        emo_reference_id = record["id"]
        use_other_emo = self.emo_ref_mode == "other"
        if self.emo_ref_mode == "mixed":
            use_other_emo = (
                _stable_unit_interval(
                    self.seed, self.epoch, record["id"], "emo_mixed"
                )
                < 0.5
            )
        if self.emo_ref_mode == "follow_speaker":
            emo_cache = speaker_cache
            emo_reference_id = reference_id
        elif use_other_emo:
            other_emo = self._other_reference(record, index, "emo_reference")
            if other_emo is not None:
                emo_cache = torch.load(
                    other_emo["cache_path"], map_location="cpu", weights_only=False
                )
                emo_reference_id = other_emo["id"]

        language = str(cached.get("language") or record.get("language") or "en")
        return {
            "id": record["id"],
            "reference_id": reference_id,
            "emo_reference_id": emo_reference_id,
            "text_tokens": torch.as_tensor(cached["text_tokens"], dtype=torch.long).flatten(),
            "codes": torch.as_tensor(cached["codes"], dtype=torch.long).flatten(),
            "lang_id": int(cached.get("lang_id", lang_to_token(language))),
            "campplus": torch.as_tensor(speaker_cache["campplus"], dtype=torch.float32).flatten(),
            "emo_raw": torch.as_tensor(emo_cache["emo_raw"], dtype=torch.float32).flatten(),
            "emo_vec": torch.as_tensor(emo_cache["emo_vec"], dtype=torch.float32).flatten(),
            "speaker": str(cached.get("speaker", record.get("speaker", ""))),
        }

    @property
    def fingerprint(self) -> str:
        value = "\n".join(f"{record['id']}:{record['n_codes']}:{record['n_text_tokens']}" for record in self.records)
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


def collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    text_values = [torch.as_tensor(item["text_tokens"], dtype=torch.long).flatten() for item in batch]
    code_values = [torch.as_tensor(item["codes"], dtype=torch.long).flatten() for item in batch]
    return {
        "text_tokens": pad_sequence(text_values, batch_first=True, padding_value=1),
        "text_lengths": torch.tensor([value.numel() for value in text_values], dtype=torch.long),
        "codes": pad_sequence(code_values, batch_first=True, padding_value=8193),
        "code_lengths": torch.tensor([value.numel() for value in code_values], dtype=torch.long),
        "lang_ids": torch.tensor([int(item["lang_id"]) for item in batch], dtype=torch.long),
        "campplus": torch.stack([torch.as_tensor(item["campplus"], dtype=torch.float32) for item in batch]),
        "emo_raw": torch.stack([torch.as_tensor(item["emo_raw"], dtype=torch.float32) for item in batch]),
        "emo_vec": torch.stack([torch.as_tensor(item["emo_vec"], dtype=torch.float32) for item in batch]),
        "ids": [str(item["id"]) for item in batch],
        "reference_ids": [str(item.get("reference_id", item["id"])) for item in batch],
        "emo_reference_ids": [
            str(item.get("emo_reference_id", item["id"])) for item in batch
        ],
    }


class LengthBucketBatchSampler(BatchSampler):
    """Sort into coarse length buckets, then shuffle batches and members."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        bucket_size: int | None = None,
    ) -> None:
        self.lengths = [int(value) for value in lengths]
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.bucket_size = max(self.batch_size, int(bucket_size or self.batch_size * 20))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        ordered = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        buckets = [ordered[start : start + self.bucket_size] for start in range(0, len(ordered), self.bucket_size)]
        batches: list[list[int]] = []
        for bucket in buckets:
            if self.shuffle:
                rng.shuffle(bucket)
            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def load_cache_index(dataset_dir: str | Path) -> dict[str, Any]:
    path = Path(dataset_dir) / "cache_index.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


__all__ = ["LengthBucketBatchSampler", "LoraTrainDataset", "collate", "load_cache_index"]
