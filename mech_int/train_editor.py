"""Train the eight jointly optimized residual editors specified in TRAIN.md."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from mech_int.editor_data import GameSplits, collate
from mech_int.editor_model import EditedOthelloGPT
from mech_int.reproduce_linear_probe import CHECKPOINT_SHA256, write_json


REPO = Path(__file__).resolve().parents[1]
PLAYABLE = [square for square in range(64) if square not in (27, 28, 35, 36)]
TOKEN_TO_SQUARE = [-1] + PLAYABLE


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def uniform_legal_loss(logits, legal_mask, reduction="mean"):
    """Uniform legal targets, with normalization over every vocabulary token."""
    counts = legal_mask.sum(-1)
    if torch.any(counts == 0).item():
        raise ValueError("Terminal targets must be skipped before computing loss")
    values = -(F.log_softmax(logits, dim=-1) * legal_mask).sum(-1) / counts
    if reduction == "none":
        return values
    if reduction != "mean":
        raise ValueError(f"Unsupported reduction: {reduction}")
    return values.mean()


def aggregate(rows):
    count = len(rows)
    if not count:
        return {"examples": 0}
    perfect = sum(row["exact_legal_set"] for row in rows)
    false_positives = sum(row["false_positives"] for row in rows)
    false_negatives = sum(row["false_negatives"] for row in rows)
    return {
        "examples": count,
        "loss": sum(row["loss"] for row in rows) / count,
        "top_k_legal_set_accuracy": perfect / count,
        "exact_legal_sets": perfect,
        "mean_legal_move_recall": sum(row["legal_move_recall"] for row in rows) / count,
        "mean_false_positives": false_positives / count,
        "mean_false_negatives": false_negatives / count,
        "total_false_positives": false_positives,
        "total_false_negatives": false_negatives,
        "mean_set_errors": (false_positives + false_negatives) / count,
    }


@torch.no_grad()
def evaluate(model, examples, batch_size, device, bypass=False, details=False):
    model.eval()
    rows = []
    # Length sorting reduces padding; every saved example is still evaluated once.
    ordered = sorted(examples, key=lambda example: (example.prefix_length, example.game_index))
    for start in range(0, len(ordered), batch_size):
        batch = collate(ordered[start:start + batch_size], device=device)
        logits = model(batch.tokens, None if bypass else batch.rules, batch.lengths)
        losses = uniform_legal_loss(logits, batch.legal_mask, reduction="none").cpu().tolist()
        rankings = logits.argsort(dim=-1, descending=True).cpu().tolist()
        legal_masks = batch.legal_mask.cpu().tolist()
        for example, loss, ranking, mask in zip(batch.examples, losses, rankings, legal_masks):
            target = {token for token, legal in enumerate(mask) if legal}
            predicted = set(ranking[:len(target)])
            false_positive = sorted(predicted - target)
            false_negative = sorted(target - predicted)
            row = {
                "game_index": example.game_index, "action": example.action,
                "prefix_length": example.prefix_length, "loss": loss,
                "changed_legal_set": example.changed_legal_set,
                "legal_count": len(target), "false_positives": len(false_positive),
                "false_negatives": len(false_negative),
                "exact_legal_set": predicted == target,
                "legal_move_recall": len(predicted & target) / len(target),
            }
            if details:
                row.update({
                    "example": asdict(example), "target_tokens": sorted(target),
                    "predicted_tokens": ranking[:len(target)],
                    "predicted_squares": [TOKEN_TO_SQUARE[token] for token in ranking[:len(target)]],
                    "false_positive_tokens": false_positive, "false_negative_tokens": false_negative,
                })
            rows.append(row)
    result = aggregate(rows)
    result["by_action"] = {action: aggregate([row for row in rows if row["action"] == action])
                           for action in ("flip", "erase")}
    result["by_legality_change"] = {
        name: aggregate([row for row in rows if row["changed_legal_set"] is changed])
        for name, changed in (("changed", True), ("unchanged", False))
    }
    result["by_action_and_legality_change"] = {
        action: {name: aggregate([row for row in rows
                                 if row["action"] == action and row["changed_legal_set"] is changed])
                 for name, changed in (("changed", True), ("unchanged", False))}
        for action in ("flip", "erase")
    }
    result["ranking"] = "Top k over all 61 vocabulary tokens, k = edited legal-move count; token 0 counts as illegal"
    if details:
        result["per_example"] = sorted(rows, key=lambda row: row["game_index"])
    return result


def configuration(args, device, data):
    files = ["TRAIN.md", "mech_int/train_editor.py", "mech_int/editor_model.py",
             "mech_int/editor_data.py", "mech_int/reproduce_linear_probe.py", "data/othello.py"]
    return {
        "protocol": "TRAIN.md", "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "data": str(args.data.resolve()),
        "data_manifest": json.loads((args.data / "dataset_manifest.json").read_text()),
        "moves_sha256": {split: file_sha256(args.data / data.manifest["splits"][split]["moves_file"])
                         for split in ("train", "val", "test")},
        "source_sha256": {name: file_sha256(REPO / name) for name in files},
        "architecture": {"layers": 8, "d_model": 512, "rule_width": 128,
                         "bottleneck": args.bottleneck, "activation": "ReLU",
                         "patch_location": "post-block residual, final real input token only",
                         "initialization": "random D; zero U and b",
                         "frozen_transformer": True,
                         "gradients_through_frozen_blocks": True},
        "training": {"batch_size": args.batch_size, "optimizer": "AdamW",
                     "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                     "betas": [0.9, 0.999], "learning_rate_schedule": "constant",
                     "seed": args.seed, "validation_interval": args.validation_interval,
                     "max_steps": args.max_steps, "patience": args.patience,
                     "loss": "mean per-example uniform legal-move CE over full 61-token vocabulary"},
        "sampling": {"train_game": "candidate games uniform with replacement from train split only",
                     "prefix_length": "candidate lengths uniform integer 1..59; accept within chosen legality-change stratum",
                     "edit_square": "candidate square uniform occupied noncentral square",
                     "edit_action": "FLIP/ERASE equal probability, selected before rejection sampling",
                     "changed_target_fraction": args.changed_target_fraction,
                     "legality_change": "simulator edited legal set differs from original legal set; 90 percent by default",
                     "unchanged_controls": "remaining fraction of accepted examples; stratum selected before candidate rejection",
                     "rule_order": "FLIP[64]; ERASE[64], A1,A2,...,H8",
                     "turn": "preserved; original simulator implicit-pass convention",
                     "validation": "one fixed seeded accepted example per validation game",
                     "test": "one fixed seeded accepted example per test game",
                     "evaluation_rng": "independent generators per split; saved JSON examples"},
        "device": device, "precision": "float32", "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }


def fingerprint(config):
    # Device is allowed to change when resuming after a hardware interruption.
    comparable = {key: value for key, value in config.items()
                  if key not in ("device", "torch_version", "numpy_version")}
    return hashlib.sha256(json.dumps(comparable, sort_keys=True).encode()).hexdigest()


def cpu_editors(model):
    return {name: value.detach().cpu().clone() for name, value in model.editors.state_dict().items()}


def checkpoint_best(output, model, step, metrics, config_id):
    atomic_torch_save(output / "best_editors.pth", {
        "editors": cpu_editors(model), "step": step, "validation": metrics,
        "config_fingerprint": config_id,
    })


def save_resume(output, model, optimizer, rng, state):
    atomic_torch_save(output / "training_state.pth", {
        **state, "editors": cpu_editors(model), "optimizer": optimizer.state_dict(),
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
    })


def load_best(model, output, config_id):
    best = torch.load(output / "best_editors.pth", map_location="cpu", weights_only=True)
    if best["config_fingerprint"] != config_id:
        raise ValueError("Best checkpoint does not match this run's configuration")
    model.editors.load_state_dict(best["editors"])
    return best


def benchmark(args, device):
    data = GameSplits(args.data, changed_target_fraction=args.changed_target_fraction)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    model = EditedOthelloGPT(args.checkpoint, bottleneck=args.bottleneck).to(device)
    optimizer = torch.optim.AdamW(model.editors.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    measurements = []
    for step in range(6):
        started = time.monotonic()
        batch = data.sample_training_batch(args.batch_size, rng, device=device)
        sampled = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = uniform_legal_loss(model(batch.tokens, batch.rules, batch.lengths), batch.legal_mask)
        loss.backward()
        optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        row = {"step": step + 1, "seconds": elapsed, "sampling_seconds": sampled - started,
               "loss": loss.item(), "max_prefix_length": int(batch.lengths.max().item())}
        measurements.append(row)
        print(json.dumps(row), flush=True)
    summary = {"device": device, "batch_size": args.batch_size,
               "bottleneck": args.bottleneck,
               "changed_target_fraction": args.changed_target_fraction,
               "sampling_statistics": data.stats,
               "median_seconds_per_update": float(np.median([row["seconds"] for row in measurements[2:]])),
               "measurements": measurements}
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "benchmark.json", summary)
    print(json.dumps(summary), flush=True)


def run(args, device):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".training.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"A process is already using {output}") from None
    if args.mode == "train" and not args.resume and (output / "config.json").exists():
        raise ValueError("Existing run found: choose a new --output or use --resume")
    started = time.monotonic()
    status = {"pid": os.getpid(), "started_at": utc_now(), "device": device,
              "phase": "initializing", "step": 0, "max_steps": args.max_steps,
              "complete": False}

    def update_status(**fields):
        status.update(fields)
        status["updated_at"] = utc_now()
        status["elapsed_seconds_this_process"] = time.monotonic() - started
        write_json(output / "training_status.json", status)

    update_status()
    stop_requested = False

    def stop_handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"Signal {signum}: will save after the current update", flush=True)

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    try:
        data = GameSplits(args.data, changed_target_fraction=args.changed_target_fraction)
        config = configuration(args, device, data)
        config_id = fingerprint(config)
        if args.resume or args.mode == "evaluate":
            existing = json.loads((output / "config.json").read_text())
            if fingerprint(existing) != config_id:
                raise ValueError("Configuration or input source changed; refuse incompatible resume/evaluation")
        else:
            write_json(output / "config.json", config)
        rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        model = EditedOthelloGPT(args.checkpoint, bottleneck=args.bottleneck).to(device)
        optimizer = torch.optim.AdamW(model.editors.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
        update_status(phase="preparing_fixed_examples",
                      bottleneck=args.bottleneck,
                      changed_target_fraction=args.changed_target_fraction,
                      trainable_parameters=sum(parameter.numel() for parameter in model.editors.parameters()))
        validation = data.fixed_examples("val", seed=args.seed, path=output / "val_examples.json")
        test = data.fixed_examples("test", seed=args.seed, path=output / "test_examples.json")
        update_status(validation_examples=len(validation), test_examples=len(test))
        if args.mode == "evaluate":
            best = load_best(model, output, config_id)
            result = evaluate(model, test, args.batch_size, device, details=True)
            write_json(output / "test_results.json", {"complete": True, "best_step": best["step"], **result})
            update_status(phase="complete", complete=True, best_step=best["step"], test=aggregate(result["per_example"]))
            print(json.dumps({key: value for key, value in result.items() if key != "per_example"}), flush=True)
            return
        state = {"step": 0, "seen_examples": 0, "best_loss": float("inf"),
                 "best_step": 0, "patience_count": 0, "config_fingerprint": config_id,
                 "elapsed_training_seconds": 0.0, "sampling_statistics": data.stats}
        history = []
        if args.resume:
            # AdamW keeps scalar step counters on CPU; load_state_dict moves
            # momentum buffers to each parameter's device automatically.
            saved = torch.load(output / "training_state.pth", map_location="cpu", weights_only=False)
            if saved["config_fingerprint"] != config_id:
                raise ValueError("Resume checkpoint configuration mismatch")
            model.editors.load_state_dict(saved["editors"])
            optimizer.load_state_dict(saved["optimizer"])
            rng.bit_generator.state = saved["numpy_rng_state"]
            torch.set_rng_state(saved["torch_rng_state"].cpu())
            state = {key: saved[key] for key in state}
            data.stats = state["sampling_statistics"]
            history = json.loads((output / "training_history.json").read_text())
            print(f"Resumed at update={state['step']} best_update={state['best_step']}", flush=True)
        else:
            update_status(phase="baseline_validation")
            baseline = evaluate(model, validation, args.batch_size, device)
            write_json(output / "baseline_validation.json", baseline)
            state["best_loss"] = baseline["loss"]
            checkpoint_best(output, model, 0, baseline, config_id)
            history.append({"step": 0, "baseline": True, "improved": True, **baseline})
            write_json(output / "training_history.json", history)
            save_resume(output, model, optimizer, rng, state)
            print(f"Baseline validation loss={baseline['loss']:.6f} exact_sets={baseline['top_k_legal_set_accuracy']:.4f}", flush=True)
        prior_elapsed = state["elapsed_training_seconds"]
        train_started = time.monotonic()
        running_losses = []
        reason = "maximum_updates"
        update_status(phase="training", step=state["step"], best_validation_loss=state["best_loss"],
                      best_step=state["best_step"], patience_count=state["patience_count"])
        for step in range(state["step"] + 1, args.max_steps + 1):
            if state["patience_count"] >= args.patience or stop_requested:
                reason = "early_stopping" if not stop_requested else "interrupted"
                break
            model.train()
            batch = data.sample_training_batch(args.batch_size, rng, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.tokens, batch.rules, batch.lengths)
            loss = uniform_legal_loss(logits, batch.legal_mask)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite training loss at update {step}")
            loss.backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all().item()
                   for parameter in model.editors.parameters()):
                raise FloatingPointError(f"Missing or non-finite editor gradients at update {step}")
            optimizer.step()
            loss_value = loss.item()
            running_losses.append(loss_value)
            state["step"] = step
            state["seen_examples"] += len(batch.examples)
            state["elapsed_training_seconds"] = prior_elapsed + time.monotonic() - train_started
            if step % args.log_interval == 0 or step == 1:
                mean_loss = float(np.mean(running_losses))
                running_losses.clear()
                seconds_per_update = state["elapsed_training_seconds"] / step
                update_status(step=step, seen_examples=state["seen_examples"], train_loss=mean_loss,
                              best_validation_loss=state["best_loss"], best_step=state["best_step"],
                              patience_count=state["patience_count"], seconds_per_update=seconds_per_update,
                              sampling_statistics=data.stats,
                              remaining_seconds_upper_estimate=(args.max_steps - step) * seconds_per_update)
                print(f"update={step} train_loss={mean_loss:.6f} seconds_per_update={seconds_per_update:.3f}", flush=True)
            if step % args.validation_interval == 0 or step == args.max_steps:
                update_status(phase="validation", step=step)
                metrics = evaluate(model, validation, args.batch_size, device)
                improved = metrics["loss"] < state["best_loss"]
                if improved:
                    state["best_loss"], state["best_step"], state["patience_count"] = metrics["loss"], step, 0
                    checkpoint_best(output, model, step, metrics, config_id)
                else:
                    state["patience_count"] += 1
                history.append({"step": step, "improved": improved,
                                "patience_count": state["patience_count"], **metrics})
                write_json(output / "training_history.json", history)
                print(f"validation update={step} loss={metrics['loss']:.6f} "
                      f"exact_sets={metrics['top_k_legal_set_accuracy']:.4f} "
                      f"FP={metrics['mean_false_positives']:.4f} FN={metrics['mean_false_negatives']:.4f} "
                      f"improved={improved} patience={state['patience_count']}/{args.patience}", flush=True)
                update_status(phase="training", validation=metrics, best_validation_loss=state["best_loss"],
                              best_step=state["best_step"], patience_count=state["patience_count"])
            if step % args.save_interval == 0 or step % args.validation_interval == 0:
                save_resume(output, model, optimizer, rng, state)
        if stop_requested:
            reason = "interrupted"
        elif state["patience_count"] >= args.patience:
            reason = "early_stopping"
        state["elapsed_training_seconds"] = prior_elapsed + time.monotonic() - train_started
        save_resume(output, model, optimizer, rng, state)
        if reason == "interrupted":
            update_status(phase="interrupted", step=state["step"], reason=reason)
            return
        update_status(phase="testing_best_checkpoint", step=state["step"])
        best = load_best(model, output, config_id)
        test_metrics = evaluate(model, test, args.batch_size, device, details=True)
        write_json(output / "test_results.json", {"complete": True, "best_step": best["step"], **test_metrics})
        baseline_test = evaluate(model, test, args.batch_size, device, bypass=True)
        write_json(output / "baseline_test.json", baseline_test)
        summary = {"complete": True, "reason": reason, "updates": state["step"],
                   "seen_examples": state["seen_examples"], "best_step": best["step"],
                   "best_validation": best["validation"],
                   "test": {key: value for key, value in test_metrics.items() if key != "per_example"},
                   "baseline_test": baseline_test, "finished_at": utc_now(),
                   "elapsed_training_seconds": state["elapsed_training_seconds"]}
        summary["sampling_statistics"] = data.stats
        write_json(output / "run_summary.json", summary)
        update_status(phase="complete", complete=True, reason=reason, best_step=best["step"], test=summary["test"])
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        update_status(phase="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate", "benchmark"), default="train")
    parser.add_argument("--checkpoint", type=Path, default=Path("synthetic_model.pth"))
    parser.add_argument("--data", type=Path, default=Path("data/probe_reproduction"))
    parser.add_argument("--output", type=Path, default=Path("output/editor_mlp_128_changed90"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--changed-target-fraction", type=float, default=0.9,
                        help="Fraction of accepted training examples whose edit changes legal moves")
    parser.add_argument("--natural-training-sampling", action="store_true",
                        help="Use the original sampler without legality-change stratification")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("bottleneck", "batch_size", "validation_interval", "max_steps", "patience", "log_interval", "save_interval"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Learning rate must be positive and weight decay nonnegative")
    if not 0 <= args.changed_target_fraction <= 1:
        parser.error("--changed-target-fraction must be in [0,1]")
    if args.natural_training_sampling:
        args.changed_target_fraction = None
    torch.set_num_threads(4)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("Apple GPU unavailable in this process; run with GPU access or select --device cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.mode == "benchmark":
        benchmark(args, device)
    else:
        run(args, device)


if __name__ == "__main__":
    main()
