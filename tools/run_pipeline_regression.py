import argparse
import json
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DummyProgress:
    def __call__(self, *args, **kwargs):
        return None


def import_webui():
    saved_argv = sys.argv[:]
    try:
        sys.argv = [saved_argv[0]]
        import webui  # noqa: PLC0415
    finally:
        sys.argv = saved_argv
    return webui


def read_wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    return int(round(frames * 1000.0 / rate))


def build_long_text() -> str:
    paragraphs = [
        "This is a long text regression pass for the single GPU batching path.",
        "It should split into multiple tokenizer sections and synthesize them in section micro batches.",
        "We need to verify that long form inference still works when section batch size is greater than one.",
        "The goal of this test is stability in the actual application path, not stylistic perfection.",
    ]
    return " ".join(paragraphs * 10)


def call_gen_single(webui, *, name: str, text: str, subtitle_mode: bool, subtitle_file: str | None, batch_size: int):
    output_update, subtitle_status = webui.gen_single(
        emo_control_method=0,
        prompt="demo_voice_for_test.mp3",
        text=text,
        subtitle_mode=subtitle_mode,
        subtitle_file=subtitle_file,
        save_used_audio=False,
        output_filename=name,
        emo_ref_path=None,
        emo_weight=1.0,
        vec1=0.0,
        vec2=0.0,
        vec3=0.0,
        vec4=0.0,
        vec5=0.0,
        vec6=0.0,
        vec7=0.0,
        vec8=0.0,
        emo_text="",
        emo_random=False,
        max_text_tokens_per_segment="120",
        save_as_mp3=False,
        diffusion_steps=25,
        inference_cfg_rate=0.7,
        interval_silence=200,
        max_speaker_audio_length=15,
        max_emotion_audio_length=15,
        autoregressive_batch_size=batch_size,
        apply_emo_bias=False,
        max_emotion_sum=0.8,
        latent_multiplier=1.72,
        max_consecutive_silence=0,
        mp3_bitrate="192k",
        do_sample=True,
        top_p=0.8,
        top_k=30,
        temperature=0.8,
        length_penalty=0.0,
        num_beams=3,
        repetition_penalty=10.0,
        max_mel_tokens=1500,
        low_memory_mode=False,
        prevent_vram_accumulation=False,
        semantic_layer=17,
        cfm_cache_length=8192,
        emo_bias_joy=1.0,
        emo_bias_anger=1.0,
        emo_bias_sad=1.0,
        emo_bias_fear=1.0,
        emo_bias_disgust=1.0,
        emo_bias_depression=1.0,
        emo_bias_surprise=1.0,
        emo_bias_calm=1.0,
        progress=DummyProgress(),
    )

    output_path = output_update.get("value") if isinstance(output_update, dict) else None
    if not output_path:
        raise RuntimeError(f"{name}: gen_single did not return an output path")

    output_path = (ROOT / output_path).resolve() if not Path(output_path).is_absolute() else Path(output_path)
    metadata_path = output_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    subtitle_status_value = subtitle_status.get("value") if isinstance(subtitle_status, dict) else None

    return output_path, metadata, subtitle_status_value


def run_core_batch_distinct(webui):
    texts = ["This is batch section one.", "This is batch section two."]
    outputs = webui.tts.infer_texts(
        spk_audio_prompt="demo_voice_for_test.mp3",
        texts=texts,
        section_batch_size=2,
        max_speaker_audio_length=15,
        max_emotion_audio_length=15,
        max_consecutive_silence=0,
    )
    (_, wav0), (_, wav1) = outputs
    mean_abs_diff = float(np.mean(np.abs(wav0.astype(np.int32) - wav1.astype(np.int32))))
    if mean_abs_diff <= 0.0:
        raise RuntimeError("core_batch_distinct: batched outputs are identical")
    return {
        "case": "core_batch_distinct",
        "result_count": len(outputs),
        "shape_0": list(wav0.shape),
        "shape_1": list(wav1.shape),
        "mean_abs_diff": mean_abs_diff,
    }


def run_text_case(webui, *, name: str, batch_size: int):
    output_path, metadata, _ = call_gen_single(
        webui,
        name=name,
        text=build_long_text(),
        subtitle_mode=False,
        subtitle_file=None,
        batch_size=batch_size,
    )
    duration_ms = read_wav_duration_ms(output_path)
    if metadata["status"] != "completed":
        raise RuntimeError(f"{name}: metadata status is {metadata['status']}")
    if duration_ms <= 0:
        raise RuntimeError(f"{name}: output duration is zero")
    return {
        "case": name,
        "output_path": str(output_path),
        "duration_ms": duration_ms,
        "status": metadata["status"],
        "mode": metadata["task"]["mode"],
        "section_batch_size": metadata["settings"]["resolved_generation_kwargs"]["section_batch_size"],
    }


def run_subtitle_case(webui, *, subtitle_name: str, batch_size: int):
    case_name = f"{Path(subtitle_name).stem}_cue_timing_b{batch_size}"
    output_path, metadata, subtitle_status = call_gen_single(
        webui,
        name=case_name,
        text="",
        subtitle_mode=True,
        subtitle_file=subtitle_name,
        batch_size=batch_size,
    )
    duration_ms = read_wav_duration_ms(output_path)
    subtitle_meta = metadata["subtitle"]
    timeline_end_ms = int(subtitle_meta["timeline_end_ms"])
    duration_delta_ms = duration_ms - timeline_end_ms

    if metadata["status"] != "completed":
        raise RuntimeError(f"{case_name}: metadata status is {metadata['status']}")
    if subtitle_meta["timing_issues"]:
        raise RuntimeError(f"{case_name}: timing issues present: {subtitle_meta['timing_issues']}")
    if abs(duration_delta_ms) > 2:
        raise RuntimeError(
            f"{case_name}: combined duration mismatch, got {duration_ms} ms vs expected {timeline_end_ms} ms"
        )

    return {
        "case": case_name,
        "output_path": str(output_path),
        "duration_ms": duration_ms,
        "timeline_end_ms": timeline_end_ms,
        "duration_delta_ms": duration_delta_ms,
        "cue_count": subtitle_meta["cue_count"],
        "render_unit_count": subtitle_meta["render_unit_count"],
        "timing_issues": subtitle_meta["timing_issues"],
        "section_batch_size": metadata["settings"]["resolved_generation_kwargs"]["section_batch_size"],
        "subtitle_status": subtitle_status,
    }


def main():
    parser = argparse.ArgumentParser(description="Run actual venv regression checks against the app pipeline.")
    parser.add_argument(
        "--skip-test2",
        action="store_true",
        help="Skip the very large test2.srt subtitle runs.",
    )
    args = parser.parse_args()

    webui = import_webui()
    webui.tts.hybrid_model_device = False

    report = {
        "started_at": datetime.now().isoformat(),
        "workspace": str(ROOT),
        "cases": [],
    }

    report["cases"].append(run_core_batch_distinct(webui))
    report["cases"].append(run_text_case(webui, name="long_text_b2", batch_size=2))
    report["cases"].append(run_subtitle_case(webui, subtitle_name="test.srt", batch_size=1))
    report["cases"].append(run_subtitle_case(webui, subtitle_name="test.srt", batch_size=2))

    if not args.skip_test2:
        report["cases"].append(run_subtitle_case(webui, subtitle_name="test2.srt", batch_size=1))
        report["cases"].append(run_subtitle_case(webui, subtitle_name="test2.srt", batch_size=2))

    report["completed_at"] = datetime.now().isoformat()

    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    report_path = outputs_dir / f"regression_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"report_path": str(report_path), "case_count": len(report["cases"])}, indent=2))


if __name__ == "__main__":
    main()
