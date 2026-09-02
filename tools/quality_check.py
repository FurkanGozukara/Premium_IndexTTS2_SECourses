r"""Objective quality check for IndexTTS 2.5 variants and adapters.

For every configuration it synthesises a fixed sentence set, transcribes the result with Whisper and
reports word error rate (WER) against the input text, plus CAMPPlus speaker-similarity (cosine) between
the generated audio and the reference clip.  Usage:

    venv\Scripts\python.exe tools\quality_check.py --reference ..\demo_voice_for_test.mp3 \
        --config bf16 --config int8_convrot --lora loras/NAME/NAME.safetensors --out outputs/_quality_check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SENTENCES = [
    "Greetings everyone, today I am going to show you how to install and use this application step by step.",
    "The model downloader verifies every file with a hash, so a corrupted download is detected immediately.",
    "You can reduce the resolution if you get an out of memory error, and it will still work on lower VRAM GPUs.",
    "This new version supports low rank adapters, block swapping, and an eight bit quantized model file.",
    "Please check the README file for the complete list of features and the recommended settings.",
    "It works on both Windows and Linux, and the installers set up everything automatically.",
    "This is the low VRAM tier test. [pause:400ms] Everything should sound natural.",
]


def _normalize(text: str) -> list[str]:
    text = re.sub(r"\[pause:[^\]]*\]", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return text.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev = d[:]
        d[0] = i
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + cost)
    return d[len(hyp)] / len(ref)


def speaker_embedding(tts, wav_path):
    import librosa
    import torch
    import torchaudio

    audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    audio = torch.from_numpy(audio).unsqueeze(0)
    feat = torchaudio.compliance.kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    model = tts.campplus_model
    device = next(model.parameters()).device
    with torch.no_grad():
        emb = model(feat.unsqueeze(0).to(device))
    return emb.float().cpu().squeeze(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--config", action="append", default=[], help="bf16 | int8_convrot (repeatable)")
    parser.add_argument("--lora", action="append", default=[], help="adapter safetensors to test on the bf16 base (repeatable)")
    parser.add_argument("--lora-reference", default="", help="reference clip to use with the adapters (default: --reference)")
    parser.add_argument("--lora-variant", default="bf16", help="base variant used for the adapter runs: bf16 | int8_convrot")
    parser.add_argument("--blocks-to-swap", type=int, default=0, help="GPT blocks to stream from CPU (block swap) for every run")
    parser.add_argument("--run-suffix", default="", help="suffix appended to run names (e.g. _int8_swap8)")
    parser.add_argument("--out", default=os.path.join("outputs", "_quality_check"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--whisper", default="openai/whisper-large-v3-turbo")
    args = parser.parse_args()

    import torch
    from indextts.infer_v2_5 import IndexTTS2
    from indextts.runtime.vram_presets import RuntimeConfig
    from indextts.training.whisper_asr import transcribe

    os.makedirs(args.out, exist_ok=True)
    runs = [(f"base_{c}{args.run_suffix}", c, "") for c in args.config]
    runs += [(f"lora_{os.path.basename(os.path.dirname(os.path.abspath(p)))}_{os.path.splitext(os.path.basename(p))[0]}"[:80] + args.run_suffix, args.lora_variant, p) for p in args.lora]
    if not runs:
        runs = [("base_bf16" + args.run_suffix, "bf16", "")]
    results = []
    for name, variant, lora in runs:
        runtime = RuntimeConfig(model_variant=variant, lora_path=lora, lora_strength=1.0 if lora else 0.0, blocks_to_swap=args.blocks_to_swap)
        t0 = time.perf_counter()
        tts = IndexTTS2(cfg_path=os.path.join("models", "config.yaml"), model_dir="models", runtime=runtime)
        load_s = time.perf_counter() - t0
        reference = args.lora_reference if (lora and args.lora_reference) else args.reference
        ref_emb = speaker_embedding(tts, reference)
        run_dir = os.path.join(args.out, name)
        os.makedirs(run_dir, exist_ok=True)
        rows = []
        gen_total = 0.0
        audio_total = 0.0
        for idx, sentence in enumerate(SENTENCES):
            out_path = os.path.join(run_dir, f"{idx:02d}.wav")
            t1 = time.perf_counter()
            tts.infer(
                spk_audio_prompt=reference,
                text=sentence,
                output_path=out_path,
                lang="EN",
                num_beams=args.beams,
                seed=args.seed,
                max_text_tokens_per_segment=60,
            )
            gen_s = time.perf_counter() - t1
            import soundfile as sf

            info = sf.info(out_path)
            audio_s = info.frames / info.samplerate
            gen_total += gen_s
            audio_total += audio_s
            emb = speaker_embedding(tts, out_path)
            sim = torch.nn.functional.cosine_similarity(emb, ref_emb, dim=0).item()
            rows.append({"idx": idx, "text": sentence, "wav": out_path, "audio_s": audio_s, "gen_s": gen_s, "spk_sim": sim})
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        try:
            tts.unload()
        except Exception:
            pass
        del tts
        torch.cuda.empty_cache()
        # Transcribe after unloading the engine to keep VRAM usage modest.
        wers = []
        for row in rows:
            tr = transcribe(row["wav"], sr=24000, language="EN", model_name=args.whisper, device="cuda:0")
            hyp = getattr(tr, "text", "") if not isinstance(tr, str) else tr
            row["hypothesis"] = hyp
            row["wer"] = word_error_rate(row["text"], hyp)
            wers.append(row["wer"])
        summary = {
            "name": name,
            "variant": variant,
            "lora": lora,
            "load_s": round(load_s, 2),
            "rtf": round(gen_total / max(audio_total, 1e-6), 3),
            "audio_s": round(audio_total, 2),
            "peak_vram_gb": round(peak, 2),
            "mean_wer": round(sum(wers) / len(wers), 4),
            "mean_spk_sim": round(sum(r["spk_sim"] for r in rows) / len(rows), 4),
            "rows": rows,
        }
        results.append(summary)
        print(f"== {name}: WER {summary['mean_wer']:.3f} | spk-sim {summary['mean_spk_sim']:.3f} | RTF {summary['rtf']} | peak VRAM {summary['peak_vram_gb']} GB")
        for r in rows:
            print(f"   [{r['idx']}] wer {r['wer']:.2f} sim {r['spk_sim']:.3f} | {r['hypothesis'][:110]}")
    with open(os.path.join(args.out, "quality_check.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("\n| config | mean WER | speaker sim | RTF | peak VRAM GB |")
    print("|---|---:|---:|---:|---:|")
    for s in results:
        print(f"| {s['name']} | {s['mean_wer']:.3f} | {s['mean_spk_sim']:.3f} | {s['rtf']} | {s['peak_vram_gb']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
