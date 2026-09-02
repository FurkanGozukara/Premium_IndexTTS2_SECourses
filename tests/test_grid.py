from __future__ import annotations

from pathlib import Path
import wave

import pytest

import indextts.training.grid as grid_module
from indextts.training.grid import (
    GridCheckpoint,
    GridConfig,
    build_grid_cells,
    list_grids,
    load_grid,
    run_grid,
)
from ui.generation_tab import recent_outputs


def _wav(path: Path, frames: int = 2205) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22050)
        handle.writeframes(b"\0\0" * frames)


def test_cell_order_names_seed_and_grid_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    checkpoint = adapter / "voice_epoch_003.safetensors"
    checkpoint.write_bytes(b"test checkpoint")
    reference_a = tmp_path / "a.wav"
    reference_b = tmp_path / "b.wav"
    _wav(reference_a)
    _wav(reference_b)
    config = GridConfig(
        adapter_dir=str(adapter),
        checkpoints=[
            GridCheckpoint("Base model", ""),
            GridCheckpoint("epoch 3", str(checkpoint)),
        ],
        strengths=[1.0, 0.5],
        references=[str(reference_a), str(reference_b)],
        texts=["first", "second"],
        seed=-1,
        output_root=str(tmp_path / "outputs" / "grids"),
        grid_name="test_grid",
    )
    cells = build_grid_cells(config)
    assert len(cells) == 12
    assert [cell.checkpoint_label for cell in cells[:4]] == ["Base model"] * 4
    assert cells[0].filename == "base__s1__ref1__t1.wav"
    assert cells[4].filename == "epoch_003__s1__ref1__t1.wav"

    engine = object()
    monkeypatch.setattr(grid_module, "create_tts", lambda _runtime: engine)
    monkeypatch.setattr(grid_module.secrets, "randbelow", lambda _limit: 123)

    def fake_request(request, received_engine):
        assert received_engine is engine
        path = Path(request["task_layout"]["final_wav_path"])
        _wav(path)
        return {"output_path": str(path), "audio_seconds": 0.1, "seed": request["seed"]}

    monkeypatch.setattr(grid_module, "run_generation_request", fake_request)
    result = run_grid(config)
    assert result.status == "complete"
    assert {cell.seed for cell in result.cells} == {123}
    assert all(Path(cell.audio_path).is_file() for cell in result.cells)
    assert (Path(result.grid_dir) / "grid.json").is_file()
    assert (Path(result.grid_dir) / "grid.md").is_file()
    loaded = load_grid(result.grid_dir)
    assert loaded is not None and len(loaded.cells) == 12
    summaries = list_grids(config.output_root)
    assert [item.grid_name for item in summaries] == ["test_grid"]
    assert recent_outputs(tmp_path / "outputs", limit=20) == []
