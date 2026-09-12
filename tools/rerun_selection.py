"""Re-run the post-training selection of an existing training folder with the current system, without retraining.

Example:
  python tools/rerun_selection.py --adapter-dir loras/my_voice --fresh-plan --restore-decoder --defaults

The phases are the trainer's own, each in a bounded child process that frees the GPU when it ends:
  1. With --fresh-plan, the frozen speech benchmark plan is rebuilt from the run's dataset with the current
     settings (automatic prompt count, interval guards, deployment settings); the previous
     analysis/speech_evaluation folder is kept as analysis/speech_evaluation_<timestamp>.
  2. The speech comparison renders Base and the shortlisted checkpoints (the probe-best file when present) and
     writes the recommendation and the matched speaking rate.
  3. The voice decoder adapter installed in the folder, or with --restore-decoder the newest quarantined one, is
     judged through the full pipeline with the recommended checkpoint, or with the best-scoring adapter when
     Base led; adapter + decoder is then judged against Base and every plain adapter (the joint deployment
     choice) and becomes the recommendation when it wins.
  4. The decoding sweep runs for the recommended checkpoint, and the independent final test when the run has
     a final-test dataset.
Nothing here needs a network connection: the same local models as training are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _newest_quarantined_decoder(run_dir: Path) -> Path | None:
    folder = run_dir / "analysis" / "quarantined_decoders"
    if not folder.is_dir():
        return None
    files = [path for path in folder.iterdir() if path.is_file() and ".s2mel.safetensors" in path.name]
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def main(argv: list[str] | None = None) -> int:
    from indextts.lora.decoder import DECODER_ADAPTER_SUFFIX
    from indextts.training.analysis import load_training_analysis
    from indextts.training.dataset import LoraTrainDataset
    from indextts.training.dataset_manifest import atomic_write_json
    from indextts.training.evaluation_plan import build_final_test_plan, build_speech_plan
    from indextts.training.speech_eval import load_speech_evaluation
    from indextts.training.train_config import TrainConfig
    from indextts.training.trainer import LoraTrainer
    from indextts.utils.atomic_json import read_json_retry

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter-dir", required=True, help="Training folder (contains train_config.json and the checkpoints)")
    parser.add_argument("--fresh-plan", action="store_true",
                        help="Rebuild the frozen speech benchmark plan with the current settings; the old folder is kept beside it")
    parser.add_argument("--defaults", action="store_true",
                        help="Reset the selection settings of the saved config to the current defaults (automatic prompt count, "
                             "interval guards, deployment settings, decoder gate with the best adapter) before running")
    parser.add_argument("--restore-decoder", action="store_true",
                        help="Bring back the newest quarantined voice decoder adapter and judge it again")
    parser.add_argument("--decoder", default="", help="Voice decoder adapter file to judge instead of the installed or quarantined one")
    parser.add_argument("--skip-speech", action="store_true", help="Keep the existing speech report; only run the later phases")
    parser.add_argument("--skip-decoder", action="store_true", help="Do not judge a voice decoder adapter")
    parser.add_argument("--skip-sweep", action="store_true", help="Skip the decoding sweep")
    parser.add_argument("--skip-final-test", action="store_true", help="Skip the independent final test")
    parser.add_argument("--device", default="", help="CUDA device for the phases (default: the run's training device)")
    args = parser.parse_args(argv)

    run_dir = Path(args.adapter_dir).expanduser().resolve()
    config_path = run_dir / "train_config.json"
    if not config_path.is_file():
        raise SystemExit(f"{config_path} is missing; this is not a training folder")
    # Check before changing the saved config or attaching a trainer to this folder.
    status = read_json_retry(run_dir / "status.json", {}) or {}
    original_phase = str(status.get("phase") or "complete")
    original_message = str(status.get("message") or "")
    if original_phase in {"training", "post_training", "evaluating", "evaluating_speech", "adapting_decoder", "calibrating_decoding",
                          "evaluating_final_test", "initializing"} and time.time() - float(status.get("updated_at") or 0) < 120:
        raise SystemExit("this training folder appears to be active; wait for it to finish")
    config = TrainConfig.from_json(config_path)
    config.output_dir = str(run_dir.parent)
    config.name = run_dir.name
    if args.device:
        config.device = args.device
    if args.defaults:
        config.speech_eval_enabled = True
        config.speech_eval_prompts = 0
        config.speech_eval_guard_mode = "interval"
        config.speech_eval_deployment_settings = True
        config.speech_eval_score_wer_weight = 4.0
        config.decoder_adapter_always_gate = True
    config.validate()
    atomic_write_json(config_path, config.to_dict())

    trainer = LoraTrainer(config, state_dir=run_dir, continue_existing=True)
    trainer.log(f">> rerun of the post-training selection started ({datetime.now(timezone.utc).isoformat()})")

    train_dataset = LoraTrainDataset(trainer.dataset_dir, split="train", val_fraction=config.val_fraction, seed=config.seed,
                                     max_codes=config.max_codes, max_text_tokens=config.max_text_tokens,
                                     speaker_ref_mode=config.speaker_ref_mode, emo_ref_mode=config.emo_ref_mode,
                                     val_split_mode=config.val_split_mode, reference_typical=config.reference_typical)
    val_speaker_ref_mode = "other" if config.val_reference_mode == "other" else "self"
    val_emo_ref_mode = "follow_speaker" if config.val_reference_mode == "other" else "self"
    val_dataset = LoraTrainDataset(trainer.dataset_dir, split="val", val_fraction=config.val_fraction, seed=config.seed,
                                   max_codes=config.max_codes, max_text_tokens=config.max_text_tokens,
                                   speaker_ref_mode=val_speaker_ref_mode, emo_ref_mode=val_emo_ref_mode,
                                   val_split_mode=config.val_split_mode)
    trainer.training_records = list(train_dataset.records)
    if not val_dataset.records:
        raise SystemExit("the run has no validation recordings; the speech benchmark cannot be rebuilt")
    trainer._prepare_reference()

    speech_root = run_dir / "analysis" / "speech_evaluation"
    if args.fresh_plan and speech_root.exists():
        backup = speech_root.with_name(f"speech_evaluation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
        shutil.move(str(speech_root), str(backup))
        trainer.log(f">> previous speech benchmark kept at {backup}")
    plan = build_speech_plan(config, trainer.training_records, val_dataset.records, run_dir)
    trainer.speech_plan_ready = bool(plan["groups"])
    trainer.log(f">> speech benchmark: {sum(len(g['prompts']) for g in plan['groups'])} prompts, {len(plan['seeds'])} seeds, "
                f"guard {plan['policy'].get('guard_mode', 'mean')}, deployment settings {plan.get('deployment_settings', False)}")
    if config.final_test_dataset:
        build_final_test_plan(config, trainer.training_records, val_dataset.records, run_dir)

    analysis = load_training_analysis(run_dir)
    recommended = str(getattr(analysis, "recommended_checkpoint", "") or status.get("recommended_checkpoint") or "")
    if recommended and not Path(recommended).is_file():
        recommended = ""
    post_phase, post_message = "post_training", "re-running the automatic quality checks"
    trainer.write_status(phase=post_phase, message=post_message, pipeline_status="running", speech_evaluation_status="pending",
                         decoder_adapter_status="", decoder_adapter_message="", decoder_test_status="", decoder_test_message="",
                         decoding_sweep_status="", decoding_sweep_message="", final_test_status="", final_test_message="",
                         recommended_deployment=None)
    if args.skip_speech and load_speech_evaluation(run_dir) is not None:
        report = load_speech_evaluation(run_dir)
        recommended = str(report.get("recommended_checkpoint") or "")
        trainer.write_status(speech_evaluation_status="complete", speech_evaluation_message=report.get("scope", ""),
                             speech_recommended_checkpoint=recommended, recommended_kind=report.get("recommended_kind"))
    else:
        recommended = trainer._run_automatic_speech_evaluation(terminal_phase=post_phase, terminal_message=post_message,
                                                               recommended_checkpoint=recommended)

    if not args.skip_decoder:
        installed = run_dir / f"{config.name}{DECODER_ADAPTER_SUFFIX}"
        decoder_path: Path | None = None
        if args.decoder:
            decoder_path = Path(args.decoder).expanduser().resolve()
            if decoder_path != installed.resolve():
                shutil.copy2(decoder_path, installed)
                decoder_path = installed
        elif installed.is_file():
            decoder_path = installed
        elif args.restore_decoder:
            quarantined = _newest_quarantined_decoder(run_dir)
            if quarantined is not None:
                shutil.copy2(quarantined, installed)
                decoder_path = installed
                trainer.log(f">> restored the quarantined voice decoder adapter from {quarantined.name} for a new full-pipeline gate")
        if decoder_path is None:
            trainer.write_status(decoder_adapter_status="skipped", decoder_adapter_message="no voice decoder adapter to judge")
            trainer.log(">> no voice decoder adapter to judge (none installed or quarantined)")
        else:
            trainer.recommended_after_decoder = recommended
            gate = recommended if recommended and Path(recommended).is_file() else ""
            if not gate and config.decoder_adapter_always_gate:
                gate = trainer._decoder_gate_checkpoint()
                if gate:
                    trainer.log(f">> the speech comparison preferred Base; the voice decoder adapter is judged with {Path(gate).name}")
            trainer.write_status(phase="adapting_decoder", decoder_adapter_status="running",
                                 decoder_adapter_message="Judging the voice decoder adapter through the full pipeline")
            verdict = None
            try:
                verdict = trainer._run_decoder_test(decoder_path, gate)
            except Exception as exc:
                trainer.write_status(decoder_test_status="failed", decoder_test_message=str(exc))
                trainer.log(f">> full-pipeline decoder test failed: {exc}")
            if verdict is None:
                gate_status = read_json_retry(trainer.status_path, {}) or {}
                trainer._quarantine_decoder(decoder_path, str(gate_status.get("decoder_test_message") or "no completed full-pipeline gate"),
                                            status="skipped" if gate_status.get("decoder_test_status") == "skipped" else "failed")
            elif not verdict["accepted"]:
                trainer._quarantine_decoder(decoder_path, "; ".join(verdict["reasons"]), status="rejected")
            else:
                gain = verdict["speaker_gain"].get("mean")
                summary = "Voice decoder adapter judged again through the full pipeline"
                if gain is not None:
                    summary += (f"; at strength {float(verdict.get('strength', 1.0)):g}: speaker similarity {float(gain):+.4f}, "
                                f"word error rate {100 * float(verdict['wer_increase']):+.2f} points")
                summary += trainer._apply_joint_selection(verdict, recommended)
                trainer.write_status(decoder_adapter_status="complete", decoder_adapter_message=summary,
                                     decoder_adapter_path=str(decoder_path.resolve()))
                trainer.log(">> " + summary)
            recommended = trainer.recommended_after_decoder or recommended
            trainer.write_status(phase=post_phase, message=post_message, recommended_checkpoint=recommended)

    if not args.skip_sweep and config.decoding_sweep_enabled:
        trainer._run_decoding_sweep(terminal_phase=post_phase, terminal_message=post_message, recommended_checkpoint=recommended)
    if not args.skip_final_test:
        trainer._run_final_test_assessment(terminal_phase=post_phase, terminal_message=post_message, recommended_checkpoint=recommended)

    final = read_json_retry(trainer.status_path, {}) or {}
    failed = [label for key, label in (("speech_evaluation_status", "speech evaluation"), ("decoder_adapter_status", "decoder validation"),
                                       ("decoding_sweep_status", "decoding sweep"), ("final_test_status", "final test"))
              if final.get(key) == "failed"]
    message = original_message or "training complete"
    message += f"; selection re-run {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    if failed:
        message += "; automatic checks incomplete: " + ", ".join(failed)
    trainer.write_status(phase=original_phase, message=message, recommended_checkpoint=recommended,
                         pipeline_status="completed_with_warnings" if failed else "complete")
    report = load_speech_evaluation(run_dir)
    print()
    print(f"Recommended checkpoint: {recommended or 'Base'}")
    if report:
        print(f"Speech recommendation: {report.get('recommended_label')}")
        joint = report.get("joint_selection")
        if joint:
            print(f"Deployment choice with the voice decoder: {joint.get('recommended_label')} ({joint.get('decision')})")
    print(json.dumps({key: final.get(key) for key in ("decoder_adapter_status", "decoding_sweep_status", "final_test_status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
