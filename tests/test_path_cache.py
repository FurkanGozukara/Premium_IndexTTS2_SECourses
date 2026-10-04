import os

from indextts.utils import path_cache


def test_a_new_run_folder_is_seen_even_when_the_parent_time_does_not_change(tmp_path):
    # NTFS does not always update a folder's modification time when a subfolder is created
    # in it, so a walk keyed by that time alone kept a new training run out of every list
    # (and out of the live training dashboard) for two seconds.
    path_cache.invalidate_tree()
    (tmp_path / "old_run").mkdir()
    (tmp_path / "old_run" / "status.json").write_text("{}", encoding="utf-8")
    first = path_cache.tree_files(tmp_path)
    assert [path.name for path in first] == ["status.json"]
    stamp = os.stat(tmp_path).st_mtime_ns

    (tmp_path / "new_run").mkdir()
    (tmp_path / "new_run" / "status.json").write_text("{}", encoding="utf-8")
    os.utime(tmp_path, ns=(stamp, stamp))  # the parent keeps its old time, as NTFS may leave it

    second = path_cache.tree_files(tmp_path)
    assert sorted(path.parent.name for path in second) == ["new_run", "old_run"]


def test_the_walk_is_shared_while_nothing_below_the_root_changes(tmp_path, monkeypatch):
    path_cache.invalidate_tree()
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "status.json").write_text("{}", encoding="utf-8")
    path_cache.tree_files(tmp_path)
    walks = []
    original = path_cache._walk
    monkeypatch.setattr(path_cache, "_walk", lambda top: walks.append(top) or original(top))
    path_cache.tree_files(tmp_path)
    assert walks == []
