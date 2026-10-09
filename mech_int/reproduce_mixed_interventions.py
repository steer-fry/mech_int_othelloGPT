"""Evaluate simultaneous random FLIP/ERASE edits at nested sizes k=2,4,8.

One reproducible eight-tile sample and eight independent fair coins are drawn for
each unique released benchmark prefix of length at least eight. Natural pure
coin outcomes are retained; no homogeneous control arms or null passes are run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
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
from mech_int.reproduce_interventions import (
    coordinate, flip_target_class, sha256, SQUARE_TO_TOKEN, TOKEN_TO_SQUARE,
)
from scripts.verify_probe_games import original_engine

KS = (2, 4, 8)
PROBE_LAYER = 5
FLIP_SCALE = 2.3
ERASE_SCALE = 1.3
CENTRAL_SQUARES = (27, 28, 35, 36)
METRICS = ("mean_errors", "exact_set_fraction", "mean_normalized_error")


def board_legal_metadata(board) -> dict:
    """The original engine implicitly lets the opponent move after a forced pass."""
    legal = sorted(int(move) for move in board.get_valid_moves())
    forced_pass = bool(legal) and all(board.tentative_move(move) == 2 for move in legal)
    return {
        "legal_moves": legal,
        "forced_pass": forced_pass,
        "terminal": not legal,
        "nominal_next_hand_color": int(board.next_hand_color),
        "effective_next_hand_color": -int(board.next_hand_color) if forced_pass else int(board.next_hand_color),
    }


def generate_cases(benchmark_path: Path, seed: int = 42) -> dict:
    """Generate the entire manifest without invoking any model or loading probes."""
    benchmark_path = Path(benchmark_path)
    with benchmark_path.open("rb") as handle:
        records = pickle.load(handle)
    board_type = original_engine(REPO)
    rng = np.random.default_rng(seed)
    seen = set()
    prefixes, cases = [], []
    for source_index, record in enumerate(records):
        history = tuple(int(move) for move in record["history"])
        if len(history) < 8 or history in seen:
            continue
        seen.add(history)
        board = board_type()
        history_passes = 0
        for move in history:
            if move not in board.get_valid_moves():
                raise ValueError(f"Illegal benchmark history: record {source_index}, move {move}")
            history_passes += board.tentative_move(move) == 2
            board.umpire(move)
        original = board.state.astype(np.int8)
        flat = original.reshape(64)
        eligible = np.array([square for square in range(64)
                             if flat[square] != 0 and square not in CENTRAL_SQUARES], dtype=np.int64)
        if len(eligible) < 8:
            raise ValueError("An eligible prefix has fewer than eight occupied noncentral squares")
        sampled = rng.permutation(eligible)[:8].tolist()
        coins = rng.integers(0, 2, size=8).tolist()
        colors = [int(flat[square]) + 1 for square in sampled]
        actions = ["flip" if coin == 1 else "erase" for coin in coins]
        original_meta = board_legal_metadata(board)
        prefix_id = len(prefixes)
        prefix = {
            "prefix_id": prefix_id,
            "source_benchmark_index": source_index,
            "history": list(history),
            "history_coordinates": [coordinate(move) for move in history],
            "history_length": len(history),
            "history_forced_passes": int(history_passes),
            "original_board": original.astype(int).tolist(),
            "original_next_hand_color": int(board.next_hand_color),
            "original_legal_moves": original_meta["legal_moves"],
            "original_forced_pass": original_meta["forced_pass"],
            "eligible_positions_ascending": eligible.tolist(),
            "sampled_positions_8": sampled,
            "sampled_coordinates_8": [coordinate(square) for square in sampled],
            "original_colors_8": colors,
            "coins_8": coins,
            "actions_8": actions,
            "case_ids": [],
        }
        for k in KS:
            target = board_type()
            target.state = original.copy()
            target.next_hand_color = board.next_hand_color
            target_flat = target.state.reshape(64)
            for square, action in zip(sampled[:k], actions[:k]):
                target_flat[square] = -flat[square] if action == "flip" else 0
            target_meta = board_legal_metadata(target)
            n_flip = actions[:k].count("flip")
            n_erase = k - n_flip
            composition = "mixed" if n_flip and n_erase else "all_flip" if n_flip else "all_erase"
            case_id = len(cases)
            prefix["case_ids"].append(case_id)
            cases.append({
                "case_id": case_id,
                "prefix_id": prefix_id,
                "source_benchmark_index": source_index,
                "history": list(history),
                "history_length": len(history),
                "k": k,
                "selected_positions": sampled[:k],
                "selected_coordinates": [coordinate(square) for square in sampled[:k]],
                "original_colors": colors[:k],
                "actions": actions[:k],
                "n_flip": n_flip,
                "n_erase": n_erase,
                "composition": composition,
                "original_next_hand_color": int(board.next_hand_color),
                "original_legal_moves": original_meta["legal_moves"],
                "target_board": target.state.astype(int).tolist(),
                "target_legal_moves": target_meta["legal_moves"],
                "target_legal_coordinates": [coordinate(move) for move in target_meta["legal_moves"]],
                "target_n": len(target_meta["legal_moves"]),
                "changed_legal_set": original_meta["legal_moves"] != target_meta["legal_moves"],
                "forced_pass": target_meta["forced_pass"],
                "terminal": target_meta["terminal"],
                "effective_next_hand_color": target_meta["effective_next_hand_color"],
            })
        prefixes.append(prefix)
    if benchmark_path.resolve() == (REPO / "data" / "intervention_benchmark.pkl").resolve() and len(prefixes) != 854:
        raise AssertionError(f"Expected 854 unique eligible released prefixes; received {len(prefixes)}")
    if not prefixes:
        raise ValueError("No eligible benchmark histories")
    counters = {}
    for k in KS:
        group = [case for case in cases if case["k"] == k]
        counters[str(k)] = {
            "cases": len(group),
            "terminal_n0": sum(case["terminal"] for case in group),
            "forced_pass_cases": sum(case["forced_pass"] for case in group),
            "changed_legal_set_cases": sum(case["changed_legal_set"] for case in group),
            "compositions": dict(Counter(case["composition"] for case in group)),
            "n_flip_distribution": dict(sorted(Counter(case["n_flip"] for case in group).items())),
        }
    return {
        "schema_version": 1,
        "benchmark_path": str(benchmark_path.resolve()),
        "benchmark_sha256": sha256(benchmark_path),
        "source_record_count": len(records),
        "seed": seed,
        "random_generator": "NumPy default_rng, PCG64",
        "sampler": "First-occurrence unique histories length>=8; ascending occupied noncentral indices; rng.permutation(candidates)[:8], then rng.integers(0,2,size=8),1=FLIP0=ERASE",
        "samples_per_prefix": 1,
        "ks": list(KS),
        "prefix_count": len(prefixes),
        "case_count": len(cases),
        "target_board_encoding": {"white": -1, "empty": 0, "black": 1},
        "original_colors_encoding": {"white": 0, "black": 2},
        "controls": "No additional homogeneous arms and no unpatched model passes",
        "counters_by_k": counters,
        "prefixes": prefixes,
        "cases": cases,
    }


def build_attention_deltas(case: dict, weights: torch.Tensor) -> tuple[dict, dict]:
    """Sum individually normalized tile vectors; accept full or stripped weights.

    Returned deltas are [512] tensors for all zero-based blocks0..7. Callers can
    stack cases into [B,512] and pass the result to FrozenOthelloGPT.forward.
    """
    if tuple(weights.shape) == (1, 512, 8, 8, 3):
        weights = weights[0]
    if tuple(weights.shape) != (512, 8, 8, 3):
        raise ValueError(f"Unexpected probe shape: {tuple(weights.shape)}")
    flip_sum = torch.zeros(512, device=weights.device, dtype=weights.dtype)
    erase_sum = torch.zeros_like(flip_sum)
    for square, color, action in zip(case["selected_positions"], case["original_colors"], case["actions"]):
        row, col = divmod(square, 8)
        selected_class = flip_target_class(case["history_length"], color) if action == "flip" else 0
        vector = weights[:, row, col, selected_class]
        norm = vector.norm()
        if not torch.isfinite(norm) or norm == 0:
            raise ValueError("Invalid individual probe direction norm")
        if action == "flip":
            flip_sum = flip_sum + FLIP_SCALE * (vector / norm)
        elif action == "erase":
            erase_sum = erase_sum + ERASE_SCALE * (vector / norm)
        else:
            raise ValueError(f"Unknown action {action}")
    early, late = erase_sum, erase_sum + flip_sum
    deltas = {block: early if block < 2 else late for block in range(8)}
    norms = {
        "early_blocks_l2": early.norm().item(),
        "late_blocks_l2": late.norm().item(),
        "flip_sum_l2": flip_sum.norm().item(),
        "erase_sum_l2": erase_sum.norm().item(),
        "individual_scaled_norm_sum_early": ERASE_SCALE * case["n_erase"],
        "individual_scaled_norm_sum_late": FLIP_SCALE * case["n_flip"] + ERASE_SCALE * case["n_erase"],
    }
    return deltas, norms


def score_case(predicted: list[int], case: dict) -> dict:
    if case["terminal"]:
        return {"predicted_moves": [], "predicted_coordinates": [], "false_positive_moves": [],
                "false_negative_moves": [], "false_positives": None, "false_negatives": None,
                "errors": None, "exact_set": None, "normalized_error": None,
                "scored": False, "exclusion_reason": "target has no legal moves (N=0)"}
    target = set(case["target_legal_moves"])
    predictions = set(predicted)
    if len(predicted) != case["target_n"] or len(predictions) != len(predicted):
        raise ValueError("Top-N must contain exactly N distinct playable candidates")
    fp, fn = sorted(predictions - target), sorted(target - predictions)
    return {"predicted_moves": predicted, "predicted_coordinates": [coordinate(move) for move in predicted],
            "false_positive_moves": fp, "false_negative_moves": fn,
            "false_positives": len(fp), "false_negatives": len(fn), "errors": len(fp) + len(fn),
            "exact_set": not fp and not fn, "normalized_error": len(fp) / case["target_n"],
            "scored": True, "exclusion_reason": None}


def bootstrap_interval(values: np.ndarray, indices: np.ndarray) -> dict:
    """Cluster draws index unique prefixes; NaN denotes excluded N0/subset rows."""
    sampled = values[indices]
    counts = np.isfinite(sampled).sum(axis=1)
    estimates = np.divide(np.nansum(sampled, axis=1), counts,
                          out=np.full(len(indices), np.nan), where=counts > 0)
    finite = estimates[np.isfinite(estimates)]
    bounds = np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else [None, None]
    return {"lower": bounds[0], "upper": bounds[1], "confidence": 0.95,
            "bootstrap_resamples": len(indices), "valid_bootstrap_resamples": len(finite)}


def metric_values(cases: list[dict], prefix_count: int, metric: str) -> np.ndarray:
    values = np.full(prefix_count, np.nan)
    for case in cases:
        if case["terminal"]:
            continue
        key = {"mean_errors": "errors", "exact_set_fraction": "exact_set",
               "mean_normalized_error": "normalized_error"}[metric]
        values[case["prefix_id"]] = float(case["result"][key])
    return values


def summarize_group(cases: list[dict], prefix_count: int, bootstrap_indices: np.ndarray) -> dict:
    scored = [case for case in cases if not case["terminal"]]
    distribution = Counter(case["result"]["errors"] for case in scored)
    result = {
        "cases": len(cases), "scored_cases": len(scored), "terminal_n0": len(cases) - len(scored),
        "forced_pass_cases": sum(case["forced_pass"] for case in cases),
        "changed_legal_set_cases": sum(case["changed_legal_set"] for case in cases),
        "natural_compositions": dict(Counter(case["composition"] for case in cases)),
        "exact_set_cases": sum(case["result"]["exact_set"] for case in scored),
        "total_errors": sum(case["result"]["errors"] for case in scored),
        "error_distribution": {str(key): distribution[key] for key in sorted(distribution)},
        "padding_in_full_vocab_top_n": sum(case.get("padding_in_full_vocab_top_n", False) for case in scored),
        "confidence_intervals": {},
    }
    for metric in METRICS:
        values = metric_values(cases, prefix_count, metric)
        valid = values[np.isfinite(values)]
        result[metric] = float(valid.mean()) if len(valid) else None
        result["confidence_intervals"][metric] = bootstrap_interval(values, bootstrap_indices)
    result["vector_norms"] = {}
    for key in ("early_blocks_l2", "late_blocks_l2", "flip_sum_l2", "erase_sum_l2"):
        values = [case["vector_norms"][key] for case in cases]
        result["vector_norms"][key] = {"mean": float(np.mean(values)), "min": float(np.min(values)),
                                         "max": float(np.max(values))} if values else None
    return result


def summarize_results(cases: list[dict], prefix_count: int, bootstrap_seed: int = 43,
                      bootstrap_resamples: int = 1000) -> dict:
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, prefix_count, size=(bootstrap_resamples, prefix_count))
    by_k = {}
    for k in KS:
        group = [case for case in cases if case["k"] == k]
        by_k[str(k)] = {
            "all_nonterminal": summarize_group(group, prefix_count, indices),
            "changed_legal_set": summarize_group([case for case in group if case["changed_legal_set"]], prefix_count, indices),
            "composition_strata": {},
            "pass_strata": {},
        }
        for n_flip in range(k + 1):
            subgroup = [case for case in group if case["n_flip"] == n_flip]
            by_k[str(k)]["composition_strata"][f"flip{n_flip}_erase{k - n_flip}"] = summarize_group(subgroup, prefix_count, indices)
        for forced_pass in (False, True):
            subgroup = [case for case in group if case["forced_pass"] == forced_pass]
            by_k[str(k)]["pass_strata"]["forced_pass" if forced_pass else "regular_turn"] = summarize_group(subgroup, prefix_count, indices)
    paired = {}
    case_lookup = {(case["prefix_id"], case["k"]): case for case in cases}
    for upper, lower in ((4, 2), (8, 4), (8, 2)):
        pair_cases = [prefix for prefix in range(prefix_count)
                      if not case_lookup[prefix, upper]["terminal"] and not case_lookup[prefix, lower]["terminal"]]
        result = {"paired_nonterminal_prefixes": len(pair_cases), "excluded_prefixes": prefix_count - len(pair_cases),
                  "direction": f"k{upper} minus k{lower}", "confidence_intervals": {}}
        for metric in METRICS:
            lower_values = metric_values([case_lookup[prefix, lower] for prefix in pair_cases], prefix_count, metric)
            upper_values = metric_values([case_lookup[prefix, upper] for prefix in pair_cases], prefix_count, metric)
            values = upper_values - lower_values
            valid = values[np.isfinite(values)]
            result[metric] = float(valid.mean()) if len(valid) else None
            result["confidence_intervals"][metric] = bootstrap_interval(values, indices)
        paired[f"k{upper}_minus_k{lower}"] = result
    return {"by_k": by_k, "paired_differences": paired,
            "bootstrap": {"unit": "unique prefix", "seed": bootstrap_seed,
                          "resamples": bootstrap_resamples, "confidence": 0.95,
                          "method": "Percentile bootstrap; resample prefix IDs with replacement jointly for all k; exclude N0 and condition on eligible subset within each draw"}}


def save_csvs(output: Path, cases: list[dict], summary: dict):
    path = output / "per_case_results.csv"
    fields = ["case_id", "prefix_id", "source_benchmark_index", "history_length", "k", "n_flip", "n_erase",
              "composition", "target_n", "changed_legal_set", "forced_pass", "terminal", "scored",
              "errors", "false_positives", "false_negatives", "exact_set", "normalized_error",
              "padding_in_full_vocab_top_n", "early_blocks_l2", "late_blocks_l2", "selected_coordinates",
              "actions", "target_legal_coordinates", "predicted_coordinates"]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            row = {key: case[key] for key in fields if key in case}
            row.update({key: case["result"][key] for key in fields if key in case["result"]})
            row.update({key: case["vector_norms"][key] for key in ("early_blocks_l2", "late_blocks_l2")})
            for key in ("selected_coordinates", "actions", "target_legal_coordinates", "predicted_coordinates"):
                row[key] = " ".join(row[key])
            writer.writerow(row)
    temporary.replace(path)
    path = output / "summary.csv"
    fields = ["k", "subset", "cases", "scored_cases", "terminal_n0", "forced_pass_cases", "changed_legal_set_cases",
              "mean_errors", "mean_errors_ci_low", "mean_errors_ci_high", "exact_set_fraction",
              "exact_set_fraction_ci_low", "exact_set_fraction_ci_high", "mean_normalized_error",
              "mean_normalized_error_ci_low", "mean_normalized_error_ci_high", "exact_set_cases"]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for k, summaries in summary["by_k"].items():
            subsets = [("all_nonterminal", summaries["all_nonterminal"]),
                       ("changed_legal_set", summaries["changed_legal_set"])]
            subsets.extend((f"composition:{name}", value) for name, value in summaries["composition_strata"].items())
            subsets.extend((f"pass:{name}", value) for name, value in summaries["pass_strata"].items())
            for subset, value in subsets:
                row = {key: value[key] for key in fields if key in value}
                row.update({"k": k, "subset": subset})
                for metric in METRICS:
                    bounds = value["confidence_intervals"][metric]
                    row[metric + "_ci_low"] = bounds["lower"]
                    row[metric + "_ci_high"] = bounds["upper"]
                writer.writerow(row)
    temporary.replace(path)


def prepare_manifest(args, output: Path) -> dict:
    path = output / "cases_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if (manifest["seed"] != args.seed or manifest["benchmark_sha256"] != sha256(args.benchmark)
                or manifest["samples_per_prefix"] != 1 or manifest["ks"] != list(KS)):
            raise ValueError("Existing manifest differs from requested protocol; choose a fresh output directory")
    else:
        manifest = generate_cases(args.benchmark, args.seed)
        write_json(path, manifest)
    return manifest


@torch.inference_mode()
def run(args, device: str) -> dict:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    with log_path.open("a") as log:
        def progress(event: dict):
            line = json.dumps(event)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        started = time.monotonic()
        manifest = prepare_manifest(args, output)
        manifest_path = output / "cases_manifest.json"
        probe_path = args.probe_dir / f"resid_{PROBE_LAYER}_linear.pth"
        if args.probe_dir.resolve() == (REPO / "mech_int" / "probes" / "linear").resolve():
            raise ValueError("Use newly trained probes, not the bundled probes")
        config = {
            "protocol": "Simultaneous random mixed FLIP/ERASE edits, one nested sample per unique prefix",
            "seed": args.seed, "samples_per_prefix": 1, "ks": list(KS),
            "prefix_count": manifest["prefix_count"], "case_count": manifest["case_count"],
            "sampler": manifest["sampler"], "random_generator": manifest["random_generator"],
            "benchmark_sha256": manifest["benchmark_sha256"], "cases_manifest_sha256": sha256(manifest_path),
            "probe_path": str(probe_path.resolve()), "probe_sha256": sha256(probe_path),
            "probe_layer_zero_based": PROBE_LAYER, "probe_classes": ["EMPTY", "YOURS", "MINE"],
            "checkpoint_sha256": sha256(args.checkpoint),
            "runner_sha256": sha256(Path(__file__)),
            "model_implementation_sha256": sha256(REPO / "mech_int" / "reproduce_linear_probe.py"),
            "original_engine_sha256": sha256(REPO / "data" / "othello.py"),
            "device": device, "precision": "float32", "batch_size": args.batch_size,
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "flip_scale": FLIP_SCALE, "flip_blocks_zero_based": list(range(2, 8)),
            "erase_scale": ERASE_SCALE, "erase_blocks_zero_based": list(range(8)),
            "normalization": "Normalize each individual tile class vector to unit L2, scale by its action alpha, then sum per block; never normalize the sums",
            "flip_class_parity": "Released rule: even history length white0->class2 black2->class1; odd white0->class1 black2->class2",
            "patch_location": "Final-token attention output before residual addition and subsequent MLP",
            "counterfactual": "Apply simultaneous edits to original board; preserve original next_hand_color; keep token history unchanged",
            "ranking": "Top N among playable vocabulary tokens1..60; padding0 excluded; full-vocabulary padding occurrence recorded diagnostically",
            "metrics": {"errors": "FP+FN", "exact_set_fraction": "Fraction of N>0 cases with predicted set equal to target legal set",
                        "normalized_error": "FP/N, equal to (FP+FN)/(2N) for top-N distinct candidates"},
            "N0": "Preserve and count terminal cases; exclude from error, normalized error and exact-set means",
            "forced_pass": "Retain and flag targets with no move for nominal current player but legal moves for opponent, following original engine",
            "controls": "No additional homogeneous arms and no null-reference forward or metrics",
            "natural_pure_outcomes": "Retain in primary aggregate, tag composition and report strata",
            "strength_tuning": False,
            "bootstrap_seed": args.bootstrap_seed, "bootstrap_resamples": args.bootstrap_resamples,
        }
        write_json(output / "config.json", config)
        progress({"event": "manifest_ready", "prefixes": manifest["prefix_count"], "cases": manifest["case_count"],
                  "counts_by_k": manifest["counters_by_k"]})
        if args.generate_only:
            return {"manifest": str(manifest_path), "config": str(output / "config.json"), "generated_only": True}
        # Construct the small direction sums on CPU to avoid thousands of GPU
        # synchronizations for individual norms; model evaluation stays float32
        # on the requested device, with one stacked transfer per block/batch.
        weights = torch.load(probe_path, map_location="cpu", weights_only=True).to(dtype=torch.float32)
        if not torch.isfinite(weights).all():
            raise ValueError("Probe contains non-finite weights")
        model = FrozenOthelloGPT(args.checkpoint).to(device=device, dtype=torch.float32)
        model.eval()
        cases = manifest["cases"]
        groups = defaultdict(list)
        for case in cases:
            groups[case["history_length"]].append(case)
        completed = 0
        for length in sorted(groups):
            group = groups[length]
            for offset in range(0, len(group), args.batch_size):
                batch = group[offset:offset + args.batch_size]
                tokens = torch.from_numpy(SQUARE_TO_TOKEN[np.array([case["history"] for case in batch])]).to(device)
                individual = [build_attention_deltas(case, weights) for case in batch]
                deltas = {block: torch.stack([item[0][block] for item in individual]).to(device) for block in range(8)}
                logits = model(tokens, attention_deltas=deltas)[:, -1, :]
                ranks = logits[:, 1:].topk(k=60).indices.cpu().numpy() + 1
                for index, case in enumerate(batch):
                    case["vector_norms"] = individual[index][1]
                    n = case["target_n"]
                    predictions = TOKEN_TO_SQUARE[ranks[index, :n]].tolist()
                    full_top = logits[index].topk(k=n).indices.cpu().tolist()
                    padding = 0 in full_top
                    case["padding_in_full_vocab_top_n"] = padding
                    case["full_vocab_top_n_matches_playable"] = set(full_top) == set(ranks[index, :n].tolist())
                    if not padding and not case["full_vocab_top_n_matches_playable"]:
                        raise AssertionError("Full-vocabulary and playable top-N differ without padding")
                    case["result"] = score_case(predictions, case)
                    completed += 1
            progress({"event": "progress", "completed_cases": completed, "total_cases": len(cases),
                      "history_length": length, "elapsed_seconds": time.monotonic() - started})
        cases.sort(key=lambda case: case["case_id"])
        summary = summarize_results(cases, manifest["prefix_count"], args.bootstrap_seed, args.bootstrap_resamples)
        result = {"config": config, "case_count": len(cases), "prefix_count": manifest["prefix_count"],
                  "counts_by_k": manifest["counters_by_k"], "summary": summary,
                  "elapsed_seconds": time.monotonic() - started,
                  "padding_in_full_vocab_top_n_total": sum(case["padding_in_full_vocab_top_n"] for case in cases)}
        write_json(output / "per_case_results.json", cases)
        save_csvs(output, cases, summary)
        write_json(output / "results.json", result)
        progress({"event": "complete", "cases": len(cases), "elapsed_seconds": result["elapsed_seconds"],
                  "means_by_k": {k: {metric: values["all_nonterminal"][metric] for metric in METRICS}
                                 for k, values in summary["by_k"].items()}})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-seed", type=int, default=43)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--probe-dir", type=Path, default=REPO / "output" / "linear_probe_reproduction")
    parser.add_argument("--benchmark", type=Path, default=REPO / "data" / "intervention_benchmark.pkl")
    parser.add_argument("--checkpoint", type=Path, default=REPO / "synthetic_model.pth")
    parser.add_argument("--output", type=Path, default=REPO / "output" / "linear_probe_reproduction" / "mixed_interventions")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap_resamples < 1:
        parser.error("Batch size and bootstrap resamples must be positive")
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
