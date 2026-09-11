"""Render technical words with plain spelling and with ``<word|PHONES>`` annotations, then check them.

For every word the tool renders one carrier sentence three ways: the plain spelling, the
proposed ARPAbet reading with syllable dots (the documented format), and the same reading
without dots. Base and, when a run folder is given, the trained adapter both render every
condition. The recognizer's transcript of each clip is compared with the intended sentence,
the target word is checked in it, and speaker similarity to the reference is measured, so the
report answers three questions: do annotations make the recognizer hear the intended word,
do the dots matter, and do annotations change the voice. Blind listening sets are written too.

Example
    python tools/pronunciation_check.py --run-dir loras/<voice> --words Qwen ComfyUI Nunchaku
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import random
import re
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_WORDS = ("ComfyUI", "Qwen", "SwarmUI", "RunPod", "GGUF", "CUDA", "VRAM", "Nunchaku", "Hunyuan", "Kohya",
                 "xformers", "SageAttention", "Krea", "Musubi", "Zorbulax", "Vexmire")
DEFAULT_CARRIER = "In this video we will install {word} and then test it together."
_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _word_heard(word: str, hypothesis: str) -> tuple[bool, float, str]:
    """Whether the target word appears in the recognizer's transcript, with the closest token."""

    target = word.casefold()
    tokens = [token.casefold() for token in _WORD_RE.findall(hypothesis or "")]
    if not tokens:
        return False, 0.0, ""
    # Try single tokens and adjacent pairs (the recognizer may split "Comfy UI").
    candidates = tokens + [tokens[i] + tokens[i + 1] for i in range(len(tokens) - 1)]
    best = max(candidates, key=lambda item: difflib.SequenceMatcher(None, target, item).ratio())
    ratio = difflib.SequenceMatcher(None, target, best).ratio()
    return ratio >= 0.85, round(ratio, 3), best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default="", help="Training folder of the adapter to test beside Base (optional)")
    parser.add_argument("--words", nargs="*", default=list(DEFAULT_WORDS))
    parser.add_argument("--readings", default="", help="JSON file {word: reading} to use instead of the app's suggestions")
    parser.add_argument("--carrier", default=DEFAULT_CARRIER, help="Sentence with {word}")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 104771])
    parser.add_argument("--reference", default="", help="Speaker reference (default: the run's saved reference)")
    parser.add_argument("--language", default="EN")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    from indextts.runtime import ProgressReporter
    from indextts.training.grid import GridCheckpoint, GridConfig, run_grid
    from indextts.training.speech_metrics import measure_clips
    from indextts.utils.atomic_json import write_json_atomic
    from indextts.utils.pronunciation import builtin_entries, suggest_pronunciation

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    checkpoints = [GridCheckpoint("Base", "")]
    model_dir, model_config, device = str(REPO_ROOT / "models"), str(REPO_ROOT / "models" / "config.yaml"), "cuda:0"
    reference = args.reference
    decoder = "none"
    if run_dir is not None:
        from indextts.lora.decoder import find_decoder_adapter
        from indextts.training.train_config import TrainConfig

        config = TrainConfig.from_json(run_dir / "train_config.json")
        model_dir, model_config, device = config.model_dir, config.model_config, config.device
        report_path = run_dir / "analysis" / "speech_evaluation" / "report.json"
        checkpoint = ""
        if report_path.is_file():
            checkpoint = str(json.loads(report_path.read_text(encoding="utf-8")).get("recommended_checkpoint") or "")
        if not checkpoint or not Path(checkpoint).is_file():
            files = [item for item in sorted(run_dir.glob("*.safetensors")) if not item.name.endswith(".s2mel.safetensors")]
            checkpoint = str(files[-1]) if files else ""
        if checkpoint:
            checkpoints.append(GridCheckpoint(run_dir.name, checkpoint))
            decoder = find_decoder_adapter(checkpoint) or "none"
        if not reference:
            found = sorted(p for p in run_dir.glob("*_reference.wav") if "_expressive_reference" not in p.name)
            reference = str(found[0]) if found else ""
    if not reference:
        raise SystemExit("a speaker reference is required (--reference or a run folder with <name>_reference.wav)")

    readings: dict[str, str] = {}
    if args.readings:
        readings.update(json.loads(Path(args.readings).read_text(encoding="utf-8")))
    builtin = {entry.word.casefold(): entry.pronunciation for entry in builtin_entries()}
    for word in args.words:
        if word in readings:
            continue
        if word.casefold() in builtin:
            readings[word] = builtin[word.casefold()]
            continue
        suggestion = suggest_pronunciation(word)
        readings[word] = suggestion.pronunciation if suggestion and suggestion.kind == "phonemes" else ""
    conditions = []
    for word in args.words:
        reading = readings.get(word, "")
        conditions.append((word, "plain", args.carrier.format(word=word)))
        if reading:
            conditions.append((word, "phones_dots", args.carrier.format(word=f"<{word}|{reading}>")))
            conditions.append((word, "phones_flat", args.carrier.format(word=f"<{word}|{reading.replace(' . ', ' ')}>")))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(args.output).expanduser().resolve() if args.output else REPO_ROOT / "outputs" / "pronunciation_check" / stamp
    out.mkdir(parents=True, exist_ok=True)
    from indextts.runtime.vram_presets import RuntimeConfig, auto_tier, resolve_preset
    from indextts.runtime.gpu import gpu_free_gb, gpu_total_gb

    total = gpu_total_gb(0)
    runtime = resolve_preset(str(auto_tier(total)), total, gpu_free_gb(0)).to_dict() if total > 0 else RuntimeConfig(device="cpu").to_dict()
    runtime["device"] = device
    runtime["decoder_adapter"] = decoder
    runtime["decoder_adapter_strength"] = 1.0
    texts = [text for _, _, text in conditions]
    grid = GridConfig(
        adapter_dir=str(run_dir or REPO_ROOT / "loras"), checkpoints=checkpoints, references=[reference], texts=texts,
        language=args.language, seeds=list(args.seeds), seed=args.seeds[0], output_root=str(out / "grids"),
        grid_name="pronunciation", runtime={"runtime": runtime, "model_dir": model_dir, "cfg_path": model_config, "use_qwen_emo": False},
        infer_kwargs={"num_beams": 3, "temperature": 0.8, "top_p": 0.8, "top_k": 30, "repetition_penalty": 10.0},
        include_verdicts=False,
    )
    print(f">> rendering {len(texts)} texts x {len(checkpoints)} checkpoints x {len(args.seeds)} seeds", flush=True)
    result = run_grid(grid, reporter=ProgressReporter("pronunciation", progress_file=out / "progress.json"))
    if result.status != "complete":
        raise SystemExit(f"grid {result.status}")
    clips = []
    for cell in result.cells:
        word, condition, rendered = conditions[cell.text_index - 1]
        clips.append({"audio": cell.audio_path, "reference": reference, "text": args.carrier.format(word=word),
                      "spoken_text": rendered, "language": args.language, "kind": "pronunciation", "prompt_id": f"{word}:{condition}",
                      "source": "", "seed": cell.seed, "checkpoint": cell.checkpoint_label, "word": word, "condition": condition,
                      "reading": readings.get(word, "")})

    def update(message: str, completed: int, total_count: int) -> None:
        print(f">> {message}: {completed}/{total_count}", flush=True)

    measured = measure_clips(clips, model_dir=model_dir, model_config=model_config, device=device, output_dir=out / "measure",
                             update=update, cancelled=lambda: False)
    rows = []
    for row in measured:
        heard, ratio, closest = _word_heard(row["word"], row.get("asr_text", ""))
        rows.append({**row, "word_heard": heard, "word_match_ratio": ratio, "closest_token": closest})
    # Summary per checkpoint and condition.
    summary: dict[str, dict] = {}
    for row in rows:
        key = f"{row['checkpoint']} / {row['condition']}"
        entry = summary.setdefault(key, {"clips": 0, "word_heard": 0, "error_rate_sum": 0.0, "similarity_sum": 0.0, "similarity_n": 0})
        entry["clips"] += 1
        entry["word_heard"] += int(row["word_heard"])
        entry["error_rate_sum"] += float(row.get("error_rate") or 0.0)
        if row.get("speaker_similarity") is not None:
            entry["similarity_sum"] += float(row["speaker_similarity"])
            entry["similarity_n"] += 1
    lines = ["# Pronunciation check", "", f"Carrier: `{args.carrier}` · seeds {args.seeds}", "",
             "| Checkpoint / condition | Clips | Word heard | Mean word error | Speaker similarity |", "|---|---:|---:|---:|---:|"]
    for key, entry in summary.items():
        lines.append(f"| {key} | {entry['clips']} | {entry['word_heard']}/{entry['clips']} | {entry['error_rate_sum'] / entry['clips'] * 100:.1f}% | "
                     f"{(entry['similarity_sum'] / entry['similarity_n']) if entry['similarity_n'] else 0:.3f} |")
    lines += ["", "| Word | Reading | Checkpoint | Condition | Seed | Heard | Closest token | Recognizer transcript |", "|---|---|---|---|---:|---|---|---|"]
    for row in sorted(rows, key=lambda item: (item["word"], item["checkpoint"], item["condition"], item["seed"])):
        lines.append(f"| {row['word']} | `{row['reading']}` | {row['checkpoint']} | {row['condition']} | {row['seed']} | "
                     f"{'yes' if row['word_heard'] else 'no'} ({row['word_match_ratio']:.2f}) | {row['closest_token']} | {row.get('asr_text', '')} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json_atomic(out / "report.json", {"carrier": args.carrier, "readings": readings, "summary": summary, "rows": rows,
                                            "generated_at": datetime.now(timezone.utc).isoformat()}, indent=1)
    # Blind sets: for each word, checkpoint and seed, the conditions shuffled.
    blind = out / "listening" / "blind"
    blind.mkdir(parents=True, exist_ok=True)
    rng = random.Random(11)
    keys = []
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["word"], row["checkpoint"], int(row["seed"])), []).append(row)
    for number, (key, items) in enumerate(sorted(groups.items()), start=1):
        if len(items) < 2:
            continue
        set_dir = blind / f"set_{number:03d}"
        set_dir.mkdir(exist_ok=True)
        rng.shuffle(items)
        letters = {}
        for letter, item in zip("ABCDEFG", items):
            shutil.copy2(item["audio"], set_dir / f"{letter}.wav")
            letters[letter] = item["condition"]
        (set_dir / "prompt.md").write_text(
            f"# Pronunciation of the word \"{key[0]}\"\n\nEvery clip says: {args.carrier.format(word=key[0])}\n\n"
            f"Rate how correctly and naturally each lettered clip pronounces \"{key[0]}\" (1-5) and whether the voice sounds the same across clips.\n",
            encoding="utf-8")
        keys.append({"set": set_dir.name, "word": key[0], "checkpoint": key[1], "seed": key[2], "letters": letters})
    write_json_atomic(out / "listening" / "keys.json", {"sets": keys}, indent=2)
    print(f">> report: {out / 'report.md'} | blind sets: {len(keys)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
