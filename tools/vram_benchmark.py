"""Benchmark IndexTTS runtime presets under real or emulated VRAM budgets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    parser.add_argument("--emulate", action="store_true")
    parser.add_argument("--subtitle", action="store_true", help="Also exercise the multi-text/subtitle path")
    parser.add_argument("--json-out", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser


def _wait_for_idle(timeout_s: float = 1800.0) -> None:
    # Query the driver before Torch creates this process's CUDA context; otherwise
    # the context itself looks like roughly 1.5 GB of unrelated GPU use on WDDM.
    started = time.monotonic()
    reported_at = 0.0
    while True:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError:
            print(">> WARNING: nvidia-smi is unavailable; GPU-idle state could not be verified.")
            return
        row = next(
            (
                line.split(",")
                for line in completed.stdout.splitlines()
                if line.split(",", 1)[0].strip() == "0"
            ),
            None,
        )
        if row is None or len(row) < 3:
            print(">> WARNING: GPU 0 was not reported by nvidia-smi.")
            return
        total_gb, free_gb = float(row[1]) / 1024.0, float(row[2]) / 1024.0
        used = max(0.0, total_gb - free_gb)
        if used <= 1.0:
            return
        elapsed = time.monotonic() - started
        if elapsed - reported_at >= 30.0 or reported_at == 0.0:
            print(
                f">> Waiting for idle GPU 0 ({used:.2f} GB in use, "
                f"{free_gb:.2f}/{total_gb:.2f} GB free)."
            )
            reported_at = elapsed
        if elapsed >= timeout_s:
            raise TimeoutError("GPU 0 remained above 1 GB of external use for 30 minutes")
        time.sleep(5.0)


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    import librosa
    import torch

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    _wait_for_idle()
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

    try:
        if args.emulate:
            cap_gb = max(0.5, args.tier - config.vram_reserve_gb)
            fraction = apply_vram_cap("cuda:0", cap_gb)
            print(f">> Emulating {args.tier} GB tier with a {cap_gb:.2f} GB allocator cap ({fraction:.3f}).")

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
            "spk_audio_prompt": str(ROOT.parent / "demo_voice_for_test.mp3"),
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
                **common,
            )
            audio_seconds = sum(item[1].shape[0] / float(item[0]) for item in generated if item is not None)
        else:
            tts.infer(text=TEXT, output_path=str(output_path), **common)
            audio_seconds = float(librosa.get_duration(path=str(output_path)))
        torch.cuda.synchronize(0)
        wall = time.perf_counter() - generation_started
        peak = memory_stats("cuda:0")
        generated_tokens = int(getattr(tts, "last_generation_stats", {}).get("generated_tokens", 0))
        gpt_time = float(getattr(tts, "last_generation_stats", {}).get("gpt_time", 0.0))
        result.update(
            {
                "generation_wall_s": wall,
                "audio_seconds": audio_seconds,
                "rtf": wall / audio_seconds if audio_seconds > 0 else None,
                "generated_tokens": generated_tokens,
                "gpt_time_s": gpt_time,
                "tokens_per_s": generated_tokens / gpt_time if gpt_time > 0 else None,
                "mel_tokens_per_s": generated_tokens / gpt_time if gpt_time > 0 else None,
                "peak_allocated_gb": peak["peak_allocated_gb"],
                "peak_reserved_gb": peak["peak_reserved_gb"],
                "fit": True,
            }
        )
        tts.unload()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        try:
            import torch

            if torch.cuda.is_available():
                peak = memory_stats("cuda:0")
                result["peak_allocated_gb"] = peak["peak_allocated_gb"]
                result["peak_reserved_gb"] = peak["peak_reserved_gb"]
        except Exception:
            pass
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
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
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
