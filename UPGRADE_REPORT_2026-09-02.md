# IndexTTS 2.5 Premium — upgrade report (2 September 2026)

## What is ready
- App folder: `G:\Index_TTS_v4\Premium_IndexTTS2_SECourses` (IndexTTS 2.5 only; all 1.x/2.0 code removed).
- Distribution zip: `G:\Index_TTS_v4\Index_TTS_v5_20260902.zip` (installers, launchers, requirements, app source; no venv/models/outputs).
- The app is left running on http://127.0.0.1:7861 for testing; `Windows_Start_App.bat` starts it normally.
- Test suite: `venv\Scripts\python.exe -m pytest tests -q` → 192 passed (155 CPU + 37 GPU).

## Your DoRA (trained on `G:\Index_TTS_v4\Lora_Training_Dataset`, 220 sentence-aligned segments, 31.7 min)
- Final adapter: `loras/SECourses_Furkan_EN_DoRA_r32/SECourses_Furkan_EN_DoRA_r32.safetensors` (rank 32, 40 epochs, 1080 steps).
- Best-validation checkpoint: `loras/SECourses_Furkan_EN_DoRA_r32/best/SECourses_Furkan_EN_DoRA_r32.safetensors` (epoch 10).
- Recommended reference clip (auto-loaded when the adapter is selected): `..._reference.wav` next to the adapter.
- Objective check (7 sentences, Whisper WER + CAMPPlus speaker similarity):

| configuration | WER | speaker sim | RTF (RTX 5090) | peak VRAM |
|---|---:|---:|---:|---:|
| base BF16 | 4.3% | 0.932 | 0.70 | 7.9 GB |
| base INT8 ConvRot | 3.0% | 0.929 | 0.72 | 6.6 GB |
| DoRA best (epoch 10) on BF16 | 2.1% | 0.935 | 0.96 | 8.0 GB |
| DoRA final (epoch 40) on BF16 | 4.3% | 0.943 | 0.96 | 8.0 GB |
| DoRA final on INT8 + 8 swapped blocks | 5.6% | 0.940 | 1.17 | 5.2 GB |

  Try both: the best checkpoint is the most intelligible, the final one is closest to your timbre. For 30-minute
  datasets 10–20 epochs is usually enough; validation loss stopped improving after epoch 10.

## INT8 ConvRot model for Hugging Face
- Local file: `models/gpt_int8_convrot.safetensors` (1,159,030,322 bytes; weight error 0.81%, logits cosine 0.9994).
- Upload it to `MonsterMMORPG/Wan_GGUF` as `IndexTTS-2.5_gpt_int8_convrot.safetensors` (the names in `Models_Downloader.py`).
  Until it is uploaded, selecting INT8 on a fresh install falls back to BF16 with a clear message.

## VRAM tiers (measured with GPU 0 idle, 2 GB reserve kept free)
| tier | GPT | swapped blocks | reference encoders | peak VRAM |
|---|---|---:|---|---:|
| 6 GB | INT8 | 22 | CPU | 2.6 GB |
| 8 GB | INT8 | 8 | on demand | 4.9 GB |
| 10 GB | INT8 | 4 | on demand | 5.3 GB |
| 12 GB | BF16 | 8 | on demand | 5.5 GB |
| 16 GB | BF16 | 0 | on demand | 7.1 GB |
| 24 GB | BF16 | 0 | GPU | 10.0 GB |
| 32 GB | BF16 | 0 | GPU | 11.2 GB |

## Before publishing
- Push the app folder to the private GitHub repo used by the installers (`FurkanGozukara/Premium_IndexTTS2_SECourses`);
  the installers `git clone` it. Nothing was committed or pushed by the upgrade.
- Upload the INT8 file (above). Optionally upload `loras/SECourses_Furkan_EN_DoRA_r32` as a demo adapter.

## Notes for support
- Presets: `presets/system` (read-only, star-marked) and `presets/user`; selecting a preset loads it; the last-used
  preset and the last applied runtime tier are restored at startup.
- Every worker writes `status.json`/`progress.json` atomically with retries (Windows sharing violations are handled);
  the UI re-attaches to a running job after a page reload or from a second tab.
- Architecture and contracts: `ARCHITECTURE_NOTES.md`; UI integration notes: `ui/INTEGRATION_NOTES.md`.
