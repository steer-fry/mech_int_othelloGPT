"""Evaluate one-time linear steering followed by unsteered KV-cache continuation.

Implements PERSISTENT_EVAL.md. Continuations are sampled on the evolving edited
board and supplied identically to independent patched and baseline KV caches.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mech_int.reproduce_interventions import (
    ERASE_BLOCKS, ERASE_SCALE, FLIP_BLOCKS, FLIP_SCALE, PROBE_LAYER,
    SQUARE_TO_TOKEN, TOKEN_TO_SQUARE, coordinate, load_cases, score, sha256,
)
from mech_int.reproduce_linear_probe import FrozenOthelloGPT, write_json
from scripts.verify_probe_games import original_engine

CENTRAL_SQUARES = (27, 28, 35, 36)
DEFAULT_OUTPUT = REPO / "output/linear_probe_reproduction/persistent_interventions"


def make_trajectories(cases: list[dict], seed: int = 42, horizon: int = 10,
                      max_input_length: int = 59) -> tuple[list[dict], list[dict]]:
    """Precompute sampling in case-ID/action order, independently of model batches."""
    if horizon < 0 or not 1 <= max_input_length <= 59:
        raise ValueError("Horizon must be nonnegative and input limit within 1..59")
    if any(case["history_length"] > max_input_length for case in cases):
        raise ValueError("A benchmark prefix exceeds the selected input length limit")
    board_type = original_engine(REPO)
    rng = np.random.default_rng(seed)
    trajectories, excluded = [], []
    for case in sorted(cases, key=lambda row: row["case_id"]):
        original = board_type()
        for move in case["history"]:
            original.umpire(move)
        for action in ("flip", "erase"):
            if action == "erase" and case["position"] in CENTRAL_SQUARES:
                excluded.append({"case_id": case["case_id"], "action": action,
                                 "reason": "central_square_has_no_move_token"})
                continue
            board = board_type()
            board.state = original.state.copy()
            board.next_hand_color = original.next_hand_color
            board.history = list(original.history)
            flat = board.state.reshape(64)
            flat[case["position"]] = -flat[case["position"]] if action == "flip" else 0
            original_branch = board_type()
            original_branch.state = original.state.copy()
            original_branch.next_hand_color = original.next_hand_color
            original_branch.history = list(original.history)
            episode = {**case, "action": action, "continuation_moves": [],
                       "continuation_tokens": [], "steps": []}
            incoming_move, implicit_pass = None, False
            for h in range(horizon + 1):
                legal = sorted(int(square) for square in board.get_valid_moves())
                if any(SQUARE_TO_TOKEN[square] == 0 for square in legal):
                    raise ValueError("Edited target includes an unrepresentable central move")
                original_legal = (sorted(original_branch.get_valid_moves())
                                  if original_branch is not None else None)
                length = case["history_length"] + h
                step = {
                    "horizon": h, "input_length": length,
                    "incoming_move": incoming_move,
                    "incoming_token": int(SQUARE_TO_TOKEN[incoming_move]) if incoming_move is not None else None,
                    "incoming_implicit_pass": bool(implicit_pass),
                    "board_state": board.state.astype(int).reshape(64).tolist(),
                    "next_hand_color": int(board.next_hand_color),
                    "edited_square_color": int(board.state.reshape(64)[case["position"]]),
                    "target_legal_moves": legal,
                    "target_legal_coordinates": [coordinate(square) for square in legal],
                    "target_legal_count": len(legal),
                    "original_board_legal_moves": original_legal,
                    "original_board_history_valid": original_branch is not None,
                    "legal_sets_differ_from_original_board": legal != original_legal if original_legal is not None else None,
                    "scored": bool(legal), "results": {},
                }
                episode["steps"].append(step)
                # Record a reached terminal board, but exclude N=0 from means.
                if not legal:
                    reason = "terminal_board"
                elif length >= max_input_length:
                    reason = "input_length_limit"
                elif h == horizon:
                    reason = "horizon_limit"
                else:
                    incoming_move = int(rng.choice(legal))
                    implicit_pass = board.tentative_move(incoming_move) == 2
                    board.umpire(incoming_move)
                    if original_branch is not None:
                        if incoming_move in original_branch.get_valid_moves():
                            original_branch.umpire(incoming_move)
                        else:
                            original_branch = None
                    episode["continuation_moves"].append(incoming_move)
                    episode["continuation_tokens"].append(int(SQUARE_TO_TOKEN[incoming_move]))
                    continue
                episode["stop_reason"] = reason
                episode["stop_horizon"] = h
                episode["final_input_length"] = length
                break
            trajectories.append(episode)
    return trajectories, excluded


def intervention_deltas(weights: torch.Tensor, episodes: list[dict]) -> dict:
    action = episodes[0]["action"]
    if any(episode["action"] != action for episode in episodes):
        raise ValueError("Intervention batches must share an action")
    vectors = torch.stack([
        weights[:, episode["position"] // 8, episode["position"] % 8,
                episode["flip_probe_class"] if action == "flip" else 0]
        for episode in episodes
    ])
    norms = vectors.norm(dim=-1, keepdim=True)
    if (norms == 0).any().item():
        raise ValueError("Intervention direction has zero norm")
    scale, blocks = (FLIP_SCALE, FLIP_BLOCKS) if action == "flip" else (ERASE_SCALE, ERASE_BLOCKS)
    delta = scale * vectors / norms
    return {layer: delta for layer in blocks}


def summarize(episodes: list[dict], horizon: int) -> list[dict]:
    rows = []
    for action in ("flip", "erase"):
        action_episodes = [episode for episode in episodes if episode["action"] == action]
        for h in range(horizon + 1):
            reached = [episode["steps"][h] for episode in action_episodes if len(episode["steps"]) > h]
            scored = [step for step in reached if step["scored"]]
            for method in ("patched", "baseline"):
                measurements = [step["results"][method] for step in scored]
                errors = [row["errors"] for row in measurements]
                count = len(measurements)
                perfect = sum(row["perfect"] for row in measurements)
                rows.append({
                    "action": action, "horizon": h, "method": method,
                    "eligible_cases": len(action_episodes), "reached_cases": len(reached),
                    "scored_cases": count, "terminal_cases_excluded": len(reached) - count,
                    "stopped_before_horizon": len(action_episodes) - len(reached),
                    "perfect_cases": perfect,
                    "exact_set_accuracy": perfect / count if count else None,
                    "exact_set_percent": 100 * perfect / count if count else None,
                    "mean_errors": sum(errors) / count if count else None,
                    "total_errors": sum(errors),
                    "mean_false_positives": sum(row["false_positives"] for row in measurements) / count if count else None,
                    "mean_false_negatives": sum(row["false_negatives"] for row in measurements) / count if count else None,
                    "original_board_history_valid_cases": sum(step["original_board_history_valid"] for step in scored),
                    "legal_sets_differ_from_original_board_cases": sum(step["legal_sets_differ_from_original_board"] is True for step in scored),
                    "error_distribution": dict(sorted(Counter(errors).items())),
                })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    temporary.replace(path)


def compare_h0(episodes: list[dict], config: dict, reference_dir: Path) -> dict:
    """Check all immediate predictions against the saved existing runner, if compatible."""
    path = reference_dir / "per_case_results.json"
    config_path = reference_dir / "config.json"
    if not path.exists() or not config_path.exists():
        return {"checked": False, "reason": "existing intervention artifacts unavailable"}
    reference_config = json.loads(config_path.read_text())
    for key in ("checkpoint_sha256", "probe_sha256", "benchmark_sha256"):
        if config[key] != reference_config[key]:
            return {"checked": False, "reason": f"existing intervention {key} differs"}
    references = {case["case_id"]: case for case in json.loads(path.read_text())}
    checked = 0
    for episode in episodes:
        step = episode["steps"][0]
        if not step["scored"]:
            continue
        for method, old_method in (("patched", "linear_addition"), ("baseline", "baseline")):
            row = step["results"][method]
            old = references[episode["case_id"]]["results"][episode["action"]][old_method]
            if set(row["predicted_moves"]) != set(old["predicted_moves"]) or row["errors"] != old["errors"]:
                raise AssertionError(f"Immediate prediction mismatch: {episode['case_id']}/{episode['action']}/{method}")
            checked += 1
    return {"checked": True, "passed": True, "method_case_predictions": checked,
            "reference_sha256": sha256(path)}


@torch.inference_mode()
def run(args, device: str) -> dict:
    started = time.monotonic()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    probe_path = args.probe_dir / f"resid_{PROBE_LAYER}_linear.pth"
    weights = torch.load(probe_path, map_location="cpu", weights_only=True)
    if tuple(weights.shape) != (1, 512, 8, 8, 3) or not torch.isfinite(weights).all().item():
        raise ValueError("Expected finite layer-5 probe with shape [1,512,8,8,3]")
    weights = weights[0].to(device=device, dtype=torch.float32)
    all_cases = load_cases(args.benchmark)
    cases = all_cases if args.limit_cases is None else all_cases[:args.limit_cases]
    episodes, excluded = make_trajectories(cases, args.seed, args.horizon, args.max_input_length)
    model = FrozenOthelloGPT(args.checkpoint).to(device=device, dtype=torch.float32).eval()
    config = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "PERSISTENT_EVAL.md", "protocol_sha256": sha256(REPO / "PERSISTENT_EVAL.md"),
        "scope": "One-time linear probe intervention, retained autoregressive KV cache, unsteered continuation",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "probe": str(probe_path.resolve()), "probe_sha256": sha256(probe_path), "probe_layer_zero_based": PROBE_LAYER,
        "benchmark": str(args.benchmark.resolve()), "benchmark_sha256": sha256(args.benchmark),
        "benchmark_records": len(all_cases), "evaluated_records": len(cases),
        "limited_run": args.limit_cases is not None,
        "seed": args.seed, "horizon": args.horizon, "max_input_length": args.max_input_length,
        "sampling": "numpy.default_rng(seed), case-ID order, FLIP then ERASE, sorted legal moves; trajectories precomputed before batching",
        "numpy_version": np.__version__, "torch_version": torch.__version__,
        "device": device, "precision": "float32", "batch_size": args.batch_size,
        "flip": {"strength": FLIP_SCALE, "blocks_zero_based": list(FLIP_BLOCKS)},
        "erase": {"strength": ERASE_SCALE, "blocks_zero_based": list(ERASE_BLOCKS)},
        "patch_location": "Attention output at original prefix final token, before residual/MLP; only during prefix pass",
        "continuation": "Identical sampled tokens for independent baseline/patched caches; no further deltas",
        "pass_convention": "Original simulator implicit passes; no pass token",
        "ranking": "Top N over playable tokens 1..60, N = evolving edited board legal count",
        "metric": "Exact-set accuracy and FP+FN; terminal N=0 excluded from averages",
        "runner_sha256": sha256(Path(__file__)),
        "model_implementation_sha256": sha256(REPO / "mech_int/reproduce_linear_probe.py"),
        "original_engine_sha256": sha256(REPO / "data/othello.py"),
    }
    write_json(output / "config.json", config)
    groups = defaultdict(list)
    for episode in episodes:
        groups[(episode["action"], episode["history_length"])].append(episode)
    completed = 0
    for (action, length), group in sorted(groups.items()):
        for offset in range(0, len(group), args.batch_size):
            batch = group[offset:offset + args.batch_size]
            tokens = torch.tensor(SQUARE_TO_TOKEN[np.array([episode["history"] for episode in batch])], device=device)
            clean_logits, clean_cache = model.forward_cached(tokens)
            patched_logits, patched_cache = model.forward_cached(tokens, attention_deltas=intervention_deltas(weights, batch))
            logits = {"baseline": clean_logits[:, -1], "patched": patched_logits[:, -1]}
            active = batch
            for h in range(args.horizon + 1):
                if clean_cache.length != length + h or patched_cache.length != length + h:
                    raise AssertionError("Cache length does not match trajectory horizon")
                rankings = {}
                for method, values in logits.items():
                    if not torch.isfinite(values).all().item():
                        raise FloatingPointError("Non-finite cached logits")
                    rankings[method] = (values[:, 1:].topk(60).indices + 1).cpu().tolist()
                for row, episode in enumerate(active):
                    step = episode["steps"][h]
                    for method in ("patched", "baseline"):
                        selected = rankings[method][row][:step["target_legal_count"]]
                        if step["scored"]:
                            measurement = score(TOKEN_TO_SQUARE[selected].tolist(), step["target_legal_moves"])
                            measurement.update({"predicted_tokens": selected, "scored": True})
                            if measurement["errors"] % 2:
                                raise AssertionError("Equal-size move-set error must be even")
                        else:
                            measurement = {"predicted_tokens": [], "predicted_moves": [],
                                           "errors": None, "false_positives": None,
                                           "false_negatives": None, "perfect": None, "scored": False}
                        step["results"][method] = measurement
                continuing = [i for i, episode in enumerate(active) if len(episode["steps"]) > h + 1]
                if not continuing:
                    break
                indices = torch.tensor(continuing, device=device, dtype=torch.long)
                clean_cache = clean_cache.index_select(indices)
                patched_cache = patched_cache.index_select(indices)
                active = [active[i] for i in continuing]
                incoming = torch.tensor([[episode["continuation_tokens"][h]] for episode in active], device=device)
                # No attention_deltas argument: steering ends after prefix pass.
                clean_logits, clean_cache = model.forward_cached(incoming, cache=clean_cache)
                patched_logits, patched_cache = model.forward_cached(incoming, cache=patched_cache)
                logits = {"baseline": clean_logits[:, -1], "patched": patched_logits[:, -1]}
            completed += len(batch)
        print(f"{action} prefix={length}: {completed}/{len(episodes)} trajectories ({time.monotonic()-started:.1f}s)", flush=True)
    summary = summarize(episodes, args.horizon)
    immediate_check = compare_h0(episodes, config, args.probe_dir / "interventions")
    stopping = {action: dict(Counter(episode["stop_reason"] for episode in episodes if episode["action"] == action))
                for action in ("flip", "erase")}
    step_rows = []
    for episode in episodes:
        for step in episode["steps"]:
            for method, measurement in step["results"].items():
                step_rows.append({
                    "case_id": episode["case_id"], "action": episode["action"], "method": method,
                    "position": episode["position"], "history_length": episode["history_length"],
                    **{key: step[key] for key in ("horizon", "input_length", "incoming_move", "incoming_implicit_pass", "target_legal_count", "scored")},
                    **{key: measurement[key] for key in ("errors", "false_positives", "false_negatives", "perfect", "predicted_moves", "predicted_tokens")},
                    "target_legal_moves": step["target_legal_moves"],
                    "stop_reason": episode["stop_reason"], "stop_horizon": episode["stop_horizon"],
                })
    result = {"config": config, "summary": summary, "stopping_reasons": stopping,
              "excluded_cases": excluded, "trajectories": len(episodes),
              "reached_steps": sum(len(episode["steps"]) for episode in episodes),
              "scored_steps": sum(step["scored"] for episode in episodes for step in episode["steps"]),
              "immediate_existing_runner_check": immediate_check,
              "elapsed_seconds": time.monotonic() - started,
              "artifacts": {"per_case_json": "per_case_results.json", "per_step_csv": "per_step_results.csv",
                            "summary_csv": "summary.csv"}}
    write_json(output / "per_case_results.json", episodes)
    write_csv(output / "per_step_results.csv", step_rows)
    write_csv(output / "summary.csv", summary)
    write_json(output / "results.json", result)
    print(json.dumps({"output": str(output.resolve()), "stopping_reasons": stopping,
                      "immediate_existing_runner_check": immediate_check,
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--probe-dir", type=Path, default=REPO / "output/linear_probe_reproduction")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark", type=Path, default=REPO / "data/intervention_benchmark.pkl")
    parser.add_argument("--checkpoint", type=Path, default=REPO / "synthetic_model.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-input-length", type=int, default=59)
    parser.add_argument("--limit-cases", type=int, help="Diagnostic run only; omit for the full benchmark")
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1 or args.horizon < 0:
        parser.error("Batch size/CPU threads must be positive and horizon nonnegative")
    if not 1 <= args.max_input_length <= 59:
        parser.error("--max-input-length must be between 1 and 59")
    if args.limit_cases is not None and args.limit_cases < 1:
        parser.error("--limit-cases must be positive")
    torch.set_num_threads(args.cpu_threads)
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
