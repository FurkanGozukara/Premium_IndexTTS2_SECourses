from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence


_REPO_TYPE_PREFIXES = {
    "model": "models",
    "dataset": "datasets",
    "space": "spaces",
}


def repo_cache_dir_name(repo_id: str, repo_type: str = "model") -> str:
    prefix = _REPO_TYPE_PREFIXES.get(repo_type, f"{repo_type}s")
    return f"{prefix}--{repo_id.replace('/', '--')}"


def repo_cache_dir(cache_dir: str, repo_id: str, repo_type: str = "model") -> str:
    return os.path.join(cache_dir, repo_cache_dir_name(repo_id, repo_type=repo_type))


def snapshot_dir(cache_dir: str, repo_id: str, revision: str = "main", repo_type: str = "model") -> Optional[str]:
    repo_dir = repo_cache_dir(cache_dir, repo_id, repo_type=repo_type)
    snapshots_root = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snapshots_root):
        return None

    ref_path = os.path.join(repo_dir, "refs", revision)
    if os.path.isfile(ref_path):
        with open(ref_path, "r", encoding="utf-8") as handle:
            commit_hash = handle.read().strip()
        if commit_hash:
            resolved = os.path.join(snapshots_root, commit_hash)
            if os.path.isdir(resolved):
                return resolved

    candidates = [
        os.path.join(snapshots_root, entry)
        for entry in os.listdir(snapshots_root)
        if os.path.isdir(os.path.join(snapshots_root, entry))
    ]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _has_required_files(base_dir: str, required_files: Optional[Iterable[str]]) -> bool:
    if not required_files:
        return True
    for relative_path in required_files:
        if not os.path.isfile(os.path.join(base_dir, relative_path)):
            return False
    return True


def cached_snapshot_dir(
    cache_dir: str,
    repo_id: str,
    required_files: Optional[Sequence[str]] = None,
    revision: str = "main",
    repo_type: str = "model",
) -> Optional[str]:
    resolved = snapshot_dir(cache_dir, repo_id, revision=revision, repo_type=repo_type)
    if resolved is None:
        return None
    if not _has_required_files(resolved, required_files):
        return None
    return resolved


def cached_file_path(
    cache_dir: str,
    repo_id: str,
    filename: str,
    revision: str = "main",
    repo_type: str = "model",
) -> Optional[str]:
    resolved = cached_snapshot_dir(
        cache_dir,
        repo_id,
        required_files=[filename],
        revision=revision,
        repo_type=repo_type,
    )
    if resolved is None:
        return None
    return os.path.join(resolved, filename)
