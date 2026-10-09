"""Evaluate released color-flip and erasure interventions with newly trained probes.

Uses the unchanged move history for clean, color-flip, and erasure model passes.
The original simulator supplies the legal moves of each counterfactual board.
No intervention strength or layer selection is tuned on the benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mech_int.reproduce_linear_probe import FrozenOthelloGPT, write_json
from scripts.verify_probe_games import original_engine


PAPER = "https://aclanthology.org/2023.blackboxnlp-1.2.pdf"
PAPER_TABLE2 = {
    "flip": {"baseline": 2.723, "linear_addition": 0.10},
    "erase": {"baseline": 2.73, "linear_addition": 0.02},
}
PROBE_LAYER = 5  # Zero-based residual probe: released resid_5_linear.pth.
FLIP_BLOCKS = tuple(range(2, 8))
ERASE_BLOCKS = tuple(range(8))
FLIP_SCALE = 2.3
ERASE_SCALE = 1.3
PLAYABLE_SQUARES = np.array([s for s in range(64) if s not in (27, 28, 35, 36)])
TOKEN_TO_SQUARE = np.concatenate(([-1], PLAYABLE_SQUARES))
SQUARE_TO_TOKEN = np.zeros(64, dtype=np.int64)
SQUARE_TO_TOKEN[PLAYABLE_SQUARES] = np.arange(1, 61)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def coordinate(square: int) -> str:
    """Repository convention: row letter first, then one-based column number."""
    return f"{chr(ord('a') + square // 8)}{square % 8 + 1}"


def flip_target_class(length: int, original_color: int) -> int:
    # Match released patch_heads exactly; original_color is WHITE=0 or BLACK=2,
    # whereas probe classes are EMPTY=0, YOURS=1, MINE=2.
    if original_color not in (0, 2):
        raise ValueError(f"Expected occupied target color 0 or 2, received {original_color}")
    if length % 2 == 0:
        return 2 if original_color == 0 else 1
    return 1 if original_color == 0 else 2


def load_cases(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        benchmark = pickle.load(handle)
    if not isinstance(benchmark, list) or not benchmark:
        raise ValueError("Benchmark must contain a nonempty list of records")
    board_type = original_engine(REPO)
    cases = []
    for index, record in enumerate(benchmark):
        history = [int(move) for move in record["history"]]
        if not 1 <= len(history) <= 59:
            raise ValueError(f"Case {index} has unsupported history length {len(history)}")
        if any(move not in PLAYABLE_SQUARES for move in history):
            raise ValueError(f"Case {index} contains invalid board move indices")
        position = int(record["pos_int"])
        if not 0 <= position < 64:
            raise ValueError(f"Case {index} has invalid intervention position")
        color_value = float(record["ori_color"])
        if color_value not in (0.0, 2.0):
            raise ValueError(f"Case {index} targets an empty or invalid tile")
        color = int(color_value)
        board = board_type()
        for move in history:
            if move not in board.get_valid_moves():
                raise ValueError(f"Case {index} contains an illegal history move: {move}")
            board.umpire(move)
        flat = board.state.reshape(64)
        if flat[position] != color - 1:
            raise ValueError(f"Case {index}: ori_color disagrees with the replayed board")
        original_legal = sorted(board.get_valid_moves())
        next_color = int(board.next_hand_color)
        flat[position] = (2 - color) - 1
        flip_legal = sorted(board.get_valid_moves())
        flat[position] = 0
        erase_legal = sorted(board.get_valid_moves())
        cases.append({
            "case_id": index,
            "history": history,
            "history_coordinates": [coordinate(move) for move in history],
            "history_length": len(history),
            "position": position,
            "position_coordinate": coordinate(position),
            "ori_color": color,
            "original_target_color": "white" if color == 0 else "black",
            "flipped_target_color": "black" if color == 0 else "white",
            "next_hand_color": next_color,
            "flip_probe_class": flip_target_class(len(history), color),
            "original_legal_moves": original_legal,
            "flip_legal_moves": flip_legal,
            "erase_legal_moves": erase_legal,
            "original_legal_coordinates": [coordinate(move) for move in original_legal],
            "flip_legal_coordinates": [coordinate(move) for move in flip_legal],
            "erase_legal_coordinates": [coordinate(move) for move in erase_legal],
        })
    return cases


def score(predicted: list[int], target: list[int]) -> dict:
    predicted_set, target_set = set(predicted), set(target)
    if len(predicted) != len(target) or len(predicted_set) != len(predicted):
        raise ValueError("Top-N predictions must contain N distinct candidates")
    false_positives = sorted(predicted_set - target_set)
    false_negatives = sorted(target_set - predicted_set)
    errors = len(false_positives) + len(false_negatives)
    return {
        "target_legal_moves": target,
        "target_legal_coordinates": [coordinate(move) for move in target],
        "predicted_moves": predicted,
        "predicted_coordinates": [coordinate(move) for move in predicted],
        "false_positive_moves": false_positives,
        "false_positive_coordinates": [coordinate(move) for move in false_positives],
        "false_negative_moves": false_negatives,
        "false_negative_coordinates": [coordinate(move) for move in false_negatives],
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "errors": errors,
        "perfect": errors == 0,
    }


def summarize(cases: list[dict]) -> dict:
    summary = {}
    for intervention in ("flip", "erase"):
        summary[intervention] = {}
        for method in ("baseline", "linear_addition"):
            measurements = [case["results"][intervention][method] for case in cases]
            errors = np.array([m["errors"] for m in measurements], dtype=np.int64)
            distribution = Counter(int(value) for value in errors)
            perfect = int((errors == 0).sum())
            paper_value = PAPER_TABLE2[intervention][method]
            summary[intervention][method] = {
                "cases": len(cases),
                "mean_errors": float(errors.mean()),
                "total_errors": int(errors.sum()),
                "mean_false_positives": float(np.mean([m["false_positives"] for m in measurements])),
                "mean_false_negatives": float(np.mean([m["false_negatives"] for m in measurements])),
                "median_errors": float(np.median(errors)),
                "maximum_errors": int(errors.max()),
                "perfect_cases": perfect,
                "perfect_fraction": perfect / len(cases),
                "perfect_percent": 100 * perfect / len(cases),
                "error_distribution": {str(key): distribution[key] for key in sorted(distribution)},
                "paper_table2_mean_errors": paper_value,
                "difference_from_paper_mean_errors": float(errors.mean()) - paper_value,
            }
    return summary


def csv_outputs(output: Path, cases: list[dict], aggregates: dict):
    cases_path = output / "per_case_results.csv"
    fields = ["case_id", "history_length", "position", "position_coordinate", "ori_color",
              "intervention", "method", "target_legal_count", "errors", "false_positives",
              "false_negatives", "perfect", "target_legal_coordinates", "predicted_coordinates",
              "false_positive_coordinates", "false_negative_coordinates"]
    temporary = cases_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            for intervention in ("flip", "erase"):
                for method in ("baseline", "linear_addition"):
                    measurement = case["results"][intervention][method]
                    row = {field: case[field] for field in fields[:5]}
                    row.update({"intervention": intervention, "method": method,
                                "target_legal_count": len(measurement["target_legal_moves"]),
                                **{field: measurement[field] for field in
                                   ("errors", "false_positives", "false_negatives", "perfect")},
                                **{field: " ".join(measurement[field]) for field in
                                   ("target_legal_coordinates", "predicted_coordinates",
                                    "false_positive_coordinates", "false_negative_coordinates")}})
                    writer.writerow(row)
    temporary.replace(cases_path)
    table_path = output / "table2_comparison.csv"
    fields = ["aggregate", "intervention", "method", "cases", "mean_errors",
              "paper_table2_mean_errors", "difference_from_paper_mean_errors",
              "perfect_cases", "perfect_percent"]
    temporary = table_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for aggregate, summaries in aggregates.items():
            for intervention, methods in summaries.items():
                for method, measurement in methods.items():
                    row = {field: measurement[field] for field in fields[3:]}
                    row.update({"aggregate": aggregate, "intervention": intervention, "method": method})
                    writer.writerow(row)
    temporary.replace(table_path)


@torch.inference_mode()
def run(args, device: str) -> dict:
    output = args.output or args.probe_dir / "interventions"
    output.mkdir(parents=True, exist_ok=True)
    if args.probe_dir.resolve() == (REPO / "mech_int" / "probes" / "linear").resolve():
        raise ValueError("Use the newly trained probes, not the bundled probes")
    probe_path = args.probe_dir / f"resid_{PROBE_LAYER}_linear.pth"
    weights = torch.load(probe_path, map_location="cpu", weights_only=True)
    if tuple(weights.shape) != (1, 512, 8, 8, 3):
        raise ValueError(f"Unexpected probe shape {tuple(weights.shape)}")
    if not torch.isfinite(weights).all():
        raise ValueError("Probe contains non-finite weights")
    weights = weights[0].to(device=device, dtype=torch.float32)
    cases = load_cases(args.benchmark)
    model = FrozenOthelloGPT(args.checkpoint).to(device=device, dtype=torch.float32)
    model.eval()
    config = {
        "paper": PAPER,
        "scope": "Table 2: linear probe addition for color flipping and erasure; null baselines",
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": sha256(args.benchmark),
        "benchmark_records": len(cases),
        "benchmark_count_note": "Released file has 1001 records; all are evaluated. The paper describes 1000; first1000 is an additional labeled aggregate.",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "probe": str(probe_path.resolve()),
        "probe_sha256": sha256(probe_path),
        "probe_layer_zero_based": PROBE_LAYER,
        "probe_classes": ["EMPTY", "YOURS", "MINE"],
        "device": device,
        "precision": "float32",
        "batch_size": args.batch_size,
        "torch_version": torch.__version__,
        "runner_sha256": sha256(Path(__file__)),
        "model_implementation_sha256": sha256(REPO / "mech_int" / "reproduce_linear_probe.py"),
        "original_engine_sha256": sha256(REPO / "data" / "othello.py"),
        "flip": {"scale": FLIP_SCALE, "block_indices_zero_based": list(FLIP_BLOCKS),
                 "direction": "Selected probe class vector normalized to unit L2 norm",
                 "class_parity": "even history length: white0->class2, black2->class1; odd: white0->class1, black2->class2"},
        "erase": {"scale": ERASE_SCALE, "block_indices_zero_based": list(ERASE_BLOCKS),
                  "direction": "EMPTY class0 vector normalized to unit L2 norm"},
        "patch_location": "Final token of attention output, before residual addition and subsequent MLP",
        "history": "Identical token history for baseline and both interventions; simulator board edits only define targets",
        "ranking": "Top N over playable vocabulary tokens1..60; padding token0 excluded. Assert full-vocabulary topN for flip contains no padding and matches candidate selection.",
        "metric": "False positives plus false negatives relative to counterfactual legal moves; N equals target legal move count",
        "strength_tuning": False,
        "table2_reference": PAPER_TABLE2,
    }
    training_config = args.probe_dir / "config.json"
    if training_config.exists():
        config["probe_training_config_sha256"] = sha256(training_config)
    write_json(output / "config.json", config)
    groups = defaultdict(list)
    for case in cases:
        groups[case["history_length"]].append(case)
    started = time.monotonic()
    completed = 0
    for length in sorted(groups):
        group = groups[length]
        for offset in range(0, len(group), args.batch_size):
            batch = group[offset:offset + args.batch_size]
            token_array = SQUARE_TO_TOKEN[np.array([case["history"] for case in batch])]
            tokens = torch.from_numpy(token_array).to(device)
            flip_vectors = []
            erase_vectors = []
            for case in batch:
                row, col = divmod(case["position"], 8)
                flip_vectors.append(weights[:, row, col, case["flip_probe_class"]])
                erase_vectors.append(weights[:, row, col, 0])
            flip_vectors = torch.stack(flip_vectors)
            erase_vectors = torch.stack(erase_vectors)
            flip_norms = flip_vectors.norm(dim=-1, keepdim=True)
            erase_norms = erase_vectors.norm(dim=-1, keepdim=True)
            if (flip_norms == 0).any() or (erase_norms == 0).any():
                raise ValueError("An intervention direction has zero norm")
            flip_delta = FLIP_SCALE * (flip_vectors / flip_norms)
            erase_delta = ERASE_SCALE * (erase_vectors / erase_norms)
            baseline_logits = model(tokens)[:, -1, :]
            flip_logits = model(tokens, attention_deltas={layer: flip_delta for layer in FLIP_BLOCKS})[:, -1, :]
            erase_logits = model(tokens, attention_deltas={layer: erase_delta for layer in ERASE_BLOCKS})[:, -1, :]
            outputs = {"baseline": baseline_logits, "flip": flip_logits, "erase": erase_logits}
            ranked_tokens = {name: value[:, 1:].topk(k=60).indices.cpu().numpy() + 1
                             for name, value in outputs.items()}
            for index, case in enumerate(batch):
                count = len(case["flip_legal_moves"])
                for name in ("baseline", "flip"):
                    full_top = outputs[name][index].topk(k=count).indices.cpu().tolist()
                    if 0 in full_top:
                        raise AssertionError(f"Padding appeared in full-vocabulary topN: case{case['case_id']} {name}")
                    if set(full_top) != set(ranked_tokens[name][index, :count].tolist()):
                        raise AssertionError("Candidate ranking differs from original full-vocabulary flip ranking")
                case["padding_in_full_vocab_top_n_flip"] = {"baseline": False, "linear_addition": False}
                case["results"] = {}
                for intervention in ("flip", "erase"):
                    target = case[f"{intervention}_legal_moves"]
                    count = len(target)
                    case["results"][intervention] = {}
                    for method, output_name in (("baseline", "baseline"), ("linear_addition", intervention)):
                        predictions = TOKEN_TO_SQUARE[ranked_tokens[output_name][index, :count]].tolist()
                        case["results"][intervention][method] = score(predictions, target)
                completed += 1
        print(json.dumps({"completed_cases": completed, "total_cases": len(cases),
                          "history_length": length, "elapsed_seconds": time.monotonic() - started}), flush=True)
    cases.sort(key=lambda case: case["case_id"])
    aggregates = {"all_records": summarize(cases)}
    if len(cases) >= 1000:
        aggregates["first1000_records"] = summarize(cases[:1000])
    result = {
        "benchmark_records": len(cases),
        "elapsed_seconds": time.monotonic() - started,
        "config": config,
        "aggregates": aggregates,
        "padding_check": {"passed": True, "cases": len(cases), "padding_in_full_vocab_top_n_flip": 0},
        "artifacts": {"per_case_json": "per_case_results.json", "per_case_csv": "per_case_results.csv",
                      "table2_comparison_csv": "table2_comparison.csv"},
    }
    write_json(output / "per_case_results.json", cases)
    csv_outputs(output, cases, aggregates)
    write_json(output / "results.json", result)
    print(json.dumps({"output": str(output.resolve()), "aggregates": aggregates}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-dir", type=Path, default=REPO / "output" / "linear_probe_reproduction")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark", type=Path, default=REPO / "data" / "intervention_benchmark.pkl")
    parser.add_argument("--checkpoint", type=Path, default=REPO / "synthetic_model.pth")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    run(args, device)


if __name__ == "__main__":
    main()
