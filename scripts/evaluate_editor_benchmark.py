"""Evaluate a frozen trained editor on the released Nanda intervention cases.

This is held-out evaluation only: no released case is used to train an editor or
choose its checkpoint. The run's validation-selected best weights are snapshotted
before evaluation, so results remain reproducible if training continues.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mech_int.editor_model import EditedOthelloGPT
from mech_int.reproduce_interventions import (
    PAPER, PAPER_TABLE2, SQUARE_TO_TOKEN, TOKEN_TO_SQUARE, load_cases, score, sha256,
)
from mech_int.reproduce_linear_probe import write_json

BENCHMARK = REPO / "data" / "intervention_benchmark.pkl"
LOCAL_REFERENCE = REPO / "output" / "linear_probe_reproduction" / "interventions" / "results.json"


def configuration_fingerprint(config: dict) -> str:
    # Same definition as train_editor.fingerprint; device/version changes are OK.
    comparable = {key: value for key, value in config.items()
                  if key not in ("device", "torch_version", "numpy_version")}
    return hashlib.sha256(json.dumps(comparable, sort_keys=True).encode()).hexdigest()


def summarize(cases: list[dict]) -> dict:
    summary = {}
    for action in ("flip", "erase"):
        summary[action] = {}
        for method in ("baseline", "editor"):
            rows = [case["results"][action][method] for case in cases]
            errors = np.array([row["errors"] for row in rows], dtype=np.int64)
            perfect = int((errors == 0).sum())
            distribution = Counter(int(error) for error in errors)
            summary[action][method] = {
                "cases": len(rows),
                "mean_errors": float(errors.mean()),
                "total_errors": int(errors.sum()),
                "mean_false_positives": float(np.mean([row["false_positives"] for row in rows])),
                "mean_false_negatives": float(np.mean([row["false_negatives"] for row in rows])),
                "total_false_positives": sum(row["false_positives"] for row in rows),
                "total_false_negatives": sum(row["false_negatives"] for row in rows),
                "median_errors": float(np.median(errors)),
                "maximum_errors": int(errors.max()),
                "perfect_cases": perfect,
                "perfect_fraction": perfect / len(rows),
                "perfect_percent": 100 * perfect / len(rows),
                "error_distribution": {str(key): distribution[key] for key in sorted(distribution)},
                "loss": float(np.mean([row["loss"] for row in rows])),
                "target_entropy": float(np.mean([row["target_entropy"] for row in rows])),
                "legal_probability_mass": float(np.mean([row["legal_probability_mass"] for row in rows])),
                "padding_in_full_top_k": sum(row["padding_in_full_top_k"] for row in rows),
                "full_top_k_set_differences": sum(row["full_top_k_set_differs"] for row in rows),
                "full_vocabulary_mean_errors": float(np.mean([row["full_vocabulary_errors"] for row in rows])),
            }
    return summary


def compare(aggregates: dict, config: dict) -> dict:
    references = None
    if LOCAL_REFERENCE.exists():
        references = json.loads(LOCAL_REFERENCE.read_text())
        reference_config = references["config"]
        for key in ("benchmark_sha256", "checkpoint_sha256"):
            if reference_config[key] != config[key]:
                raise ValueError(f"Local probe comparison uses a different {key}")
    result = {}
    for aggregate, actions in aggregates.items():
        result[aggregate] = {}
        for action, methods in actions.items():
            editor_errors = methods["editor"]["mean_errors"]
            paper_errors = PAPER_TABLE2[action]["linear_addition"]
            row = {
                "paper_table2_mean_errors": paper_errors,
                "editor_difference_from_paper_mean_errors": editor_errors - paper_errors,
                "meets_paper_mean_errors": editor_errors <= paper_errors,
            }
            if references is not None and aggregate in references["aggregates"]:
                reference = references["aggregates"][aggregate][action]
                row.update({
                    "local_linear_addition": reference["linear_addition"],
                    "editor_difference_from_local_mean_errors": editor_errors - reference["linear_addition"]["mean_errors"],
                    "meets_local_mean_errors": editor_errors <= reference["linear_addition"]["mean_errors"],
                    "baseline_difference_from_local_mean_errors": methods["baseline"]["mean_errors"] - reference["baseline"]["mean_errors"],
                })
            result[aggregate][action] = row
    return result


@torch.inference_mode()
def run(args, device: str) -> dict:
    started = time.monotonic()
    run_config_path = args.run_dir / "config.json"
    run_config_bytes = run_config_path.read_bytes()
    run_config = json.loads(run_config_bytes)
    config_id = configuration_fingerprint(run_config)
    weights_path = args.run_dir / "best_editors.pth"
    # One read supplies both the loaded weights and their hash, even if the
    # trainer atomically replaces its best checkpoint during this evaluation.
    weights_bytes = weights_path.read_bytes()
    best = torch.load(io.BytesIO(weights_bytes), map_location="cpu", weights_only=True)
    if best["config_fingerprint"] != config_id:
        raise ValueError("Best editor checkpoint does not match the run configuration")
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != run_config["checkpoint_sha256"]:
        raise ValueError("Transformer checkpoint does not match the training run")
    bottleneck = int(run_config["architecture"]["bottleneck"])
    model = EditedOthelloGPT(args.checkpoint, bottleneck=bottleneck)
    model.editors.load_state_dict(best["editors"], strict=True)
    if any(not torch.isfinite(value).all().item() for value in best["editors"].values()):
        raise ValueError("Editor checkpoint contains non-finite weights")
    model.to(device=device, dtype=torch.float32).eval()
    cases = load_cases(BENCHMARK)
    output = args.output or args.run_dir / "interventions"
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "evaluated_editors.pth"
    temporary = snapshot.with_suffix(".pth.tmp")
    temporary.write_bytes(weights_bytes)
    temporary.replace(snapshot)
    config = {
        "paper": PAPER,
        "scope": "Released Table 2 FLIP and ERASE benchmark; validation-selected residual editor",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run_dir.resolve()),
        "run_config_sha256": hashlib.sha256(run_config_bytes).hexdigest(),
        "config_fingerprint": config_id,
        "architecture": run_config["architecture"],
        "editor_checkpoint": str(weights_path.resolve()),
        "editor_checkpoint_sha256": hashlib.sha256(weights_bytes).hexdigest(),
        "editor_step": int(best["step"]),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "benchmark": str(BENCHMARK.resolve()),
        "benchmark_sha256": sha256(BENCHMARK),
        "benchmark_records": len(cases),
        "benchmark_count_note": "Released file has 1001 records. All are evaluated; first1000_records is also reported because the paper describes 1000.",
        "device": device, "precision": "float32", "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "ranking": "Top k among playable tokens 1..60; k = counterfactual legal-move count. Full 61-token top k is also checked for padding and set differences.",
        "metric": "False positives plus false negatives relative to edited legal moves",
        "history": "Identical unchanged histories for baseline, FLIP and ERASE; simulator edits define targets",
        "training_on_benchmark": False,
        "checkpoint_selection": "Best validation CE checkpoint from the training run; no benchmark tuning",
        "runner_sha256": sha256(Path(__file__)),
        "editor_model_sha256": sha256(REPO / "mech_int" / "editor_model.py"),
        "frozen_model_sha256": sha256(REPO / "mech_int" / "reproduce_linear_probe.py"),
        "benchmark_loader_sha256": sha256(REPO / "mech_int" / "reproduce_interventions.py"),
        "original_engine_sha256": sha256(REPO / "data" / "othello.py"),
        "local_linear_probe_reference": str(LOCAL_REFERENCE.resolve()) if LOCAL_REFERENCE.exists() else None,
        "local_linear_probe_reference_sha256": sha256(LOCAL_REFERENCE) if LOCAL_REFERENCE.exists() else None,
    }
    groups = defaultdict(list)
    for case in cases:
        groups[case["history_length"]].append(case)
    for length in sorted(groups):
        group = groups[length]
        for offset in range(0, len(group), args.batch_size):
            batch = group[offset:offset + args.batch_size]
            tokens = torch.from_numpy(SQUARE_TO_TOKEN[np.array([case["history"] for case in batch])]).to(device)
            rows = torch.arange(len(batch), device=device)
            positions = torch.tensor([case["position"] for case in batch], device=device)
            rules = torch.zeros((len(batch), 128), device=device)
            rules[rows, positions] = 1
            baseline = model(tokens)
            outputs = {"baseline": baseline, "flip": model(tokens, rules)}
            rules.zero_()
            rules[rows, positions + 64] = 1
            outputs["erase"] = model(tokens, rules)
            if any(not torch.isfinite(logits).all().item() for logits in outputs.values()):
                raise FloatingPointError("Non-finite benchmark logits")
            rankings = {name: logits[:, 1:].topk(k=60).indices.cpu().numpy() + 1
                        for name, logits in outputs.items()}
            full_rankings = {name: logits.topk(k=61).indices.cpu().tolist()
                             for name, logits in outputs.items()}
            log_probs = {name: logits.log_softmax(dim=-1).cpu().numpy()
                         for name, logits in outputs.items()}
            for index, case in enumerate(batch):
                case["results"] = {}
                for action in ("flip", "erase"):
                    target = case[f"{action}_legal_moves"]
                    target_tokens = SQUARE_TO_TOKEN[np.array(target, dtype=np.int64)].tolist()
                    target_token_set = set(target_tokens)
                    count = len(target)
                    case["results"][action] = {}
                    for method, output_name in (("baseline", "baseline"), ("editor", action)):
                        selected = rankings[output_name][index, :count].tolist()
                        measurement = score(TOKEN_TO_SQUARE[selected].tolist(), target)
                        full_selected = full_rankings[output_name][index][:count]
                        full_set = set(full_selected)
                        full_fp = len(full_set - target_token_set)
                        full_fn = len(target_token_set - full_set)
                        probabilities = log_probs[output_name][index, target_tokens]
                        measurement.update({
                            "predicted_tokens": selected,
                            "full_top_k_tokens": full_selected,
                            "padding_in_full_top_k": 0 in full_set,
                            "full_top_k_set_differs": full_set != set(selected),
                            "full_vocabulary_false_positives": full_fp,
                            "full_vocabulary_false_negatives": full_fn,
                            "full_vocabulary_errors": full_fp + full_fn,
                            "loss": float(-probabilities.mean()) if count else 0.0,
                            "target_entropy": float(np.log(count)) if count else 0.0,
                            "legal_probability_mass": float(np.exp(probabilities).sum()),
                        })
                        case["results"][action][method] = measurement
    cases.sort(key=lambda case: case["case_id"])
    aggregates = {"all_records": summarize(cases)}
    if len(cases) >= 1000:
        aggregates["first1000_records"] = summarize(cases[:1000])
    comparisons = compare(aggregates, config)
    result = {
        "editor_step": int(best["step"]), "benchmark_records": len(cases),
        "history_length_range": [min(groups), max(groups)],
        "config": config, "aggregates": aggregates,
        "summary": aggregates["all_records"],
        "comparisons": comparisons,
        "padding_check": {
            "checked": True,
            "checks": len(cases) * 4,
            "padding_in_full_top_k": sum(row["padding_in_full_top_k"] for methods in aggregates["all_records"].values() for row in methods.values()),
            "full_top_k_set_differences": sum(row["full_top_k_set_differences"] for methods in aggregates["all_records"].values() for row in methods.values()),
        },
        "artifacts": {"evaluated_editors": snapshot.name, "per_case_json": "per_case_results.json"},
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "per_case_results.json", cases)
    write_json(output / "results.json", result)
    print(json.dumps({
        "output": str(output.resolve()), "editor_step": best["step"], "bottleneck": bottleneck,
        "benchmark_records": len(cases),
        "mean_errors": {action: {method: row["mean_errors"] for method, row in methods.items()}
                        for action, methods in aggregates["all_records"].items()},
        "padding_check": result["padding_check"], "elapsed_seconds": result["elapsed_seconds"],
    }), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=REPO / "output" / "editor_mlp_128_changed90")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, default=REPO / "synthetic_model.pth")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("Apple GPU unavailable; select --device cpu or run with GPU access")
    run(args, device)


if __name__ == "__main__":
    main()
