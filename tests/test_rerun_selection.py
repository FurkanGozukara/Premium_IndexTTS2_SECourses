"""An evaluation rerun must not modify an active training folder."""

import json
import time

import pytest

from tools.rerun_selection import main


def test_active_run_is_rejected_before_saved_settings_change(tmp_path):
    config = tmp_path / "train_config.json"
    original = '{"name": "voice", "speech_eval_prompts": 7}\n'
    config.write_text(original, encoding="utf-8")
    (tmp_path / "status.json").write_text(
        json.dumps({"phase": "training", "updated_at": time.time()}), encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="appears to be active"):
        main(["--adapter-dir", str(tmp_path), "--defaults"])
    assert config.read_text(encoding="utf-8") == original
    assert {path.name for path in tmp_path.iterdir()} == {"train_config.json", "status.json"}
