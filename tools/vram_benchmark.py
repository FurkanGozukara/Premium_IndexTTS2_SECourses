"""Benchmark IndexTTS runtime presets under real or emulated VRAM budgets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from indextts.runtime.gpu import apply_vram_cap, memory_stats
from indextts.runtime.vram_presets import VRAM_TIERS, generation_hints, resolve_preset


TEXT = (
    "Every voice carries a history of tiny choices: a pause before an important word, a smile hidden inside a "
    "sentence, and a rhythm learned over many years. This benchmark asks the model to preserve those details "
    "while reading a practical passage about clear speech, patient listening, and the quiet confidence that "
    "comes from explaining a difficult idea in language that anyone can understand without rushing."
)
DEFAULT_IDLE_TIMEOUT_S = 1800.0


def _non_negative_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Idle timeout must be a finite, non-negative number of seconds") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("Idle timeout must be a finite, non-negative number of seconds")
    return seconds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=int, choices=VRAM_TIERS, default=32)
    parser.add_argument("--all", action="store_true", help="Run every tier in a clean subprocess")
    parser.add_argument("--variant", choices=["bf16", "int8_convrot"])
    parser.add_argument("--blocks-to-swap", type=int)
    parser.add_argument("--beams", type=int)
    parser.add_argument("--text-tokens", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--lora-path", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        help="Reference audio; defaults to the bundled example, then reference_audios/demo_voice.mp3",
    )
    parser.add_argument("--emulate", action="store_true")
    parser.add_argument("--subtitle", action="store_true", help="Also exercise the multi-text/subtitle path")
    parser.add_argument(
        "--idle-timeout", dest="idle_timeout_s", type=_non_negative_seconds,
        default=DEFAULT_IDLE_TIMEOUT_S,
        help="Maximum seconds to wait for an idle GPU (default: 1800); 0 checks once without waiting",
    )
    parser.add_argument("--json-out", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser


def _resolve_reference_audio(
    value: Path | None,
    *,
    root: Path = ROOT,
) -> Path:
    if value is not None:
        candidate = value.expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidates = [candidate.resolve()]
    else:
        candidates = [
            root / "examples" / "voice_01.wav",
            root / "reference_audios" / "demo_voice.mp3",
        ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"VRAM benchmark reference audio is missing. Tried: {attempted}. "
        "Pass --reference PATH to use another audio file."
    )


def _memory_value(value: str) -> float | None:
    try:
        result = float(value.strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _query_idle_memory(device_id: str) -> dict[str, Any]:
    """Read driver memory without creating a CUDA context in the benchmark."""
    # Older drivers may not expose memory.reserved. If memory.used itself is
    # unavailable, total-free is a conservative upper bound, not an idle pass.
    queries = (
        ("index", "memory.total", "memory.used", "memory.free", "memory.reserved"),
        ("index", "memory.total", "memory.used", "memory.free"),
        ("index", "memory.total", "memory.free"),
    )
    detail = "no memory row was returned"
    for fields in queries:
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--id", device_id,
                 "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=5.0, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Cannot verify idle GPU {device_id}: nvidia-smi failed ({exc})") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()[:500]
            continue
        rows = [row for row in csv.reader(completed.stdout.splitlines()) if row]
        if len(rows) != 1 or len(rows[0]) != len(fields):
            detail = "expected one complete memory row for the selected GPU"
            continue
        values = dict(zip(fields, rows[0]))
        total = _memory_value(values["memory.total"])
        free = _memory_value(values["memory.free"])
        used = _memory_value(values.get("memory.used", ""))
        reserved = _memory_value(values.get("memory.reserved", ""))
        if total is None or total <= 0 or free is None or free > total:
            detail = "the selected GPU reported invalid total/free memory"
            continue
        source = "memory.used"
        if used is None:
            if reserved is not None and reserved <= total - free:
                used = max(0.0, total - free - reserved)
                source = "total-free-reserved (fallback)"
            else:
                used = max(0.0, total - free)
                source = "total-free (conservative fallback; reserved may be included)"
        return {
            "device_id": device_id,
            "physical_index": values["index"].strip(),
            "total_gb": total / 1024.0,
            "free_gb": free / 1024.0,
            "used_gb": used / 1024.0,
            "reserved_gb": reserved / 1024.0 if reserved is not None else None,
            "usage_source": source,
        }
    raise RuntimeError(f"Cannot verify idle GPU {device_id}: {detail}")


def _wait_for_idle(timeout_s: float = DEFAULT_IDLE_TIMEOUT_S) -> dict[str, Any]:
    # Query the driver before Torch creates this process's CUDA context; otherwise
    # the context itself looks like roughly 1.5 GB of unrelated GPU use on WDDM.
    timeout_s = _non_negative_seconds(str(timeout_s))
    device_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0].strip()
    if not device_id or device_id == "-1":
        raise RuntimeError("The VRAM benchmark requires a CUDA-visible GPU; CUDA_VISIBLE_DEVICES disables it")
    started = time.monotonic()
    reported_at: float | None = None
    while True:
        snapshot = _query_idle_memory(device_id)
        total_gb, free_gb, used = (snapshot[key] for key in ("total_gb", "free_gb", "used_gb"))
        # A display-connected WDDM GPU commonly holds 1-3 GB for the desktop,
        # browser, and compositor even when no model workload is active.  Treat
        # up to ten percent of a large card as the idle display baseline while
        # retaining the original 1 GB limit on small cards.
        idle_limit_gb = max(1.0, total_gb * 0.10)
        elapsed = time.monotonic() - started
        remaining = max(0.0, timeout_s - elapsed)
        reserved = snapshot["reserved_gb"]
        reserved_text = "unavailable" if reserved is None else f"{reserved:.2f} GiB"
        detail = (
            f"GPU {snapshot['physical_index']} (CUDA-visible GPU 0): {used:.2f} GiB device use; "
            f"idle limit {idle_limit_gb:.2f} GiB; {free_gb:.2f}/{total_gb:.2f} GiB free; "
            f"driver-reserved {reserved_text}; source {snapshot['usage_source']}"
        )
        if used <= idle_limit_gb:
            print(f">> Idle check passed after {elapsed:.1f}s: {detail}.", flush=True)
            return {**snapshot, "idle_limit_gb": idle_limit_gb, "waited_s": elapsed, "timeout_s": timeout_s}
        if reported_at is None or elapsed - reported_at >= 10.0 or remaining == 0:
            print(f">> Waiting for idle {detail}; {elapsed:.1f}s elapsed, {remaining:.1f}s remaining.", flush=True)
            reported_at = elapsed
        if elapsed >= timeout_s:
            raise TimeoutError(f"GPU idle wait exceeded {timeout_s:g}s: {detail}")
        time.sleep(min(5.0, remaining))


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    config = resolve_preset(str(args.tier), float(args.tier))
    hints = generation_hints(args.tier)
    beams = max(1, int(args.beams if args.beams is not None else hints["num_beams_max"]))
    text_tokens = max(
        8,
        int(args.text_tokens if args.text_tokens is not None else hints["max_text_tokens_per_segment"]),
    )
    batch = max(1, int(args.batch if args.batch is not None else hints["section_batch_size_max"]))
    config.device = "cuda:0"
    if args.variant:
        config.model_variant = args.variant
    if args.blocks_to_swap is not None:
        config.blocks_to_swap = args.blocks_to_swap
    config.max_section_batch_size_hint = batch
    if args.lora_path:
        config.lora_path = str(args.lora_path)
    config.validate()

    result: dict[str, Any] = {
        "tier": args.tier,
        "variant": config.model_variant,
        "blocks_to_swap": config.blocks_to_swap,
        "beams": beams,
        "text_tokens": text_tokens,
        "batch": batch,
        "emulated": bool(args.emulate),
        "fit": False,
        "error": None,
    }
    output_dir = ROOT / "outputs" / "vram_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tier_{args.tier}_{os.getpid()}.wav"
    cuda_started = False
    measured_peaks: dict[str, float] = {}

    def record_measured_peaks(*_completed_text: Any) -> None:
        # Sequential infer_texts calls reset CUDA peak counters for each text.
        # Sample before the next text resets them, and retain allocated/reserved
        # maxima independently. This also preserves earlier peaks if a later
        # text fails before its completion callback.
        snapshot = memory_stats("cuda:0")
        for key in ("peak_allocated_gb", "peak_reserved_gb"):
            measured_peaks[key] = max(measured_peaks.get(key, 0.0), snapshot[key])

    try:
        reference_audio = _resolve_reference_audio(args.reference)
        result["reference_audio"] = str(reference_audio)

        result["idle_check"] = _wait_for_idle(args.idle_timeout_s)

        import librosa
        import torch
        if args.emulate:
            cap_gb = max(0.5, args.tier - config.vram_reserve_gb)
            fraction = apply_vram_cap("cuda:0", cap_gb)
            print(f">> Emulating {args.tier} GB tier with a {cap_gb:.2f} GB allocator cap ({fraction:.3f}).")

        cuda_started = True
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)
        load_started = time.perf_counter()
        from indextts.infer_v2_5 import IndexTTS2

        tts = IndexTTS2(
            cfg_path=str(ROOT / "models" / "config.yaml"),
            model_dir=str(ROOT / "models"),
            runtime=config,
            use_qwen_emo=True,
        )
        result["load_time_s"] = time.perf_counter() - load_started
        load_stats = memory_stats("cuda:0")
        result["load_allocated_gb"] = load_stats["allocated_gb"]
        result["load_reserved_gb"] = load_stats["reserved_gb"]

        torch.cuda.reset_peak_memory_stats(0)
        generation_started = time.perf_counter()
        common = {
            "spk_audio_prompt": str(reference_audio),
            "lang": "EN",
            "seed": 123,
            "max_text_tokens_per_segment": text_tokens,
            "num_beams": beams,
            "do_sample": False,
            "verbose": False,
        }
        if batch > 1 or args.subtitle:
            texts = [TEXT] * batch
            if args.subtitle:
                texts.append("A short second caption confirms that the batched subtitle synthesis path is operational.")
            generated = tts.infer_texts(
                texts=texts,
                section_batch_size=batch,
                on_text_complete=record_measured_peaks,
                **common,
            )
            audio_seconds = sum(item[1].shape[0] / float(item[0]) for item in generated if item is not None)
        else:
            tts.infer(text=TEXT, output_path=str(output_path), **common)
            audio_seconds = float(librosa.get_duration(path=str(output_path)))
        torch.cuda.synchronize(0)
        wall = time.perf_counter() - generation_started
        record_measured_peaks()
        generation_stats = getattr(tts, "last_generation_stats", {}) or {}
        # The engine also keeps an allocated maximum across its text units.
        # Accept only a finite non-negative value from that optional telemetry;
        # reserved memory is measured by the callback above, not inferred from it.
        engine_peak = _memory_value(str(generation_stats.get("peak_vram_gb", "")))
        if engine_peak is not None:
            measured_peaks["peak_allocated_gb"] = max(measured_peaks["peak_allocated_gb"], engine_peak)
        generated_tokens = int(generation_stats.get("generated_tokens", 0))
        gpt_time = float(generation_stats.get("gpt_time", 0.0))
        result.update(
            {
                "generation_wall_s": wall,
                "audio_seconds": audio_seconds,
                "rtf": wall / audio_seconds if audio_seconds > 0 else None,
                "generated_tokens": generated_tokens,
                "gpt_time_s": gpt_time,
                "tokens_per_s": generated_tokens / gpt_time if gpt_time > 0 else None,
                "mel_tokens_per_s": generated_tokens / gpt_time if gpt_time > 0 else None,
                **measured_peaks,
                "fit": True,
            }
        )
        tts.unload()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        try:
            if cuda_started and torch.cuda.is_available():
                record_measured_peaks()
        except Exception:
            pass
        result.update(measured_peaks)
        print(f">> Benchmark failed: {result['error']}")
    finally:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f">> Wrote {args.json_out}")
    elif not args.child:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = output_dir / f"{timestamp}_tier_{args.tier}.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f">> Wrote {result_path}")
    print("VRAM_BENCHMARK_JSON=" + json.dumps(result, separators=(",", ":")))
    return result


def _markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Tier | Variant | Swap | Fit | Load GB | Peak GB | Wall s | Audio s | RTF | Mel tok/s | Error |",
        "|---:|---|---:|:---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        value = lambda key: "" if item.get(key) is None else f"{item[key]:.2f}"
        error = str(item.get("error") or "").replace("|", "\\|")
        lines.append(
            f"| {item['tier']} | {item['variant']} | {item['blocks_to_swap']} | "
            f"{'yes' if item.get('fit') else 'no'} | {value('load_allocated_gb')} | "
            f"{value('peak_allocated_gb')} | {value('generation_wall_s')} | "
            f"{value('audio_seconds')} | {value('rtf')} | {value('tokens_per_s')} | {error} |"
        )
    return "\n".join(lines) + "\n"


def run_all(args: argparse.Namespace) -> int:
    output_dir = ROOT / "outputs" / "vram_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    reference_audio = _resolve_reference_audio(args.reference)
    for tier in VRAM_TIERS:
        command = [sys.executable, str(Path(__file__).resolve()), "--tier", str(tier), "--child"]
        for name, flag in (
            (args.variant, "--variant"),
            (args.blocks_to_swap, "--blocks-to-swap"),
            (args.beams, "--beams"),
            (args.text_tokens, "--text-tokens"),
            (args.batch, "--batch"),
        ):
            if name is not None:
                command.extend([flag, str(name)])
        if args.emulate:
            command.append("--emulate")
        if args.subtitle:
            command.append("--subtitle")
        if args.lora_path:
            command.extend(["--lora-path", str(args.lora_path)])
        command.extend(["--reference", str(reference_audio)])
        command.extend(["--idle-timeout", str(args.idle_timeout_s)])
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env["PYTHONUNBUFFERED"] = "1"
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        marker = next(
            (line.partition("=")[2] for line in reversed(completed.stdout.splitlines()) if line.startswith("VRAM_BENCHMARK_JSON=")),
            None,
        )
        if marker:
            results.append(json.loads(marker))
        else:
            results.append({"tier": tier, "variant": args.variant or "preset", "blocks_to_swap": -1, "fit": False,
                            "error": f"child exited {completed.returncode}"})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{timestamp}.json"
    markdown_path = output_dir / f"{timestamp}.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(results), encoding="utf-8")
    print(f">> Wrote {json_path}")
    print(f">> Wrote {markdown_path}")
    return 0 if all(item.get("fit") for item in results) else 1


def main() -> int:
    args = _parser().parse_args()
    if args.all and not args.child:
        return run_all(args)
    result = run_one(args)
    return 0 if result.get("fit") else 1


if __name__ == "__main__":
    raise SystemExit(main())
