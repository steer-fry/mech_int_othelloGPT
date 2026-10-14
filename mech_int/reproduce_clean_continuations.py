"""Evaluate ordinary legal-game continuation without board edits or steering.

Each benchmark record supplies its original prefix. The original simulator
samples a legal continuation, and an unmodified model retains its KV cache.
This control measures ordinary-game accuracy against that same original board.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
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
    SQUARE_TO_TOKEN, TOKEN_TO_SQUARE, coordinate, load_cases, score, sha256,
)
from mech_int.reproduce_linear_probe import FrozenOthelloGPT, write_json
from scripts.verify_probe_games import original_engine

DEFAULT_REFERENCE = REPO / "output/linear_probe_reproduction/persistent_interventions"
DEFAULT_OUTPUT = DEFAULT_REFERENCE / "ordinary_game_control"


def make_trajectories(cases: list[dict], seed: int = 42, horizon: int = 10,
                      max_input_length: int = 59) -> list[dict]:
    """Sample once per record, in case-ID order, before model batching.

    Repeated prefixes are deliberately retained to preserve benchmark weights.
    Only the original board is replayed; its state is never edited.
    """
    if horizon < 0 or not 1 <= max_input_length <= 59:
        raise ValueError("Horizon must be nonnegative and input limit within 1..59")
    if any(case["history_length"] > max_input_length for case in cases):
        raise ValueError("A benchmark prefix exceeds the selected input length limit")
    board_type = original_engine(REPO)
    rng = np.random.default_rng(seed)
    trajectories = []
    for case in sorted(cases, key=lambda row: row["case_id"]):
        board = board_type()
        for move in case["history"]:
            if move not in board.get_valid_moves():
                raise ValueError(f"Invalid original prefix for case {case['case_id']}")
            board.umpire(move)
        episode = {**case, "action": "ordinary_game", "continuation_moves": [],
                   "continuation_tokens": [], "steps": []}
        incoming_move, implicit_pass = None, False
        for h in range(horizon + 1):
            legal = sorted(int(square) for square in board.get_valid_moves())
            if any(SQUARE_TO_TOKEN[square] == 0 for square in legal):
                raise ValueError("Original target contains an unrepresentable central move")
            length = case["history_length"] + h
            episode["steps"].append({
                "horizon": h, "input_length": length,
                "incoming_move": incoming_move,
                "incoming_token": int(SQUARE_TO_TOKEN[incoming_move]) if incoming_move is not None else None,
                "incoming_implicit_pass": bool(implicit_pass),
                "board_state": board.state.astype(int).reshape(64).tolist(),
                "next_hand_color": int(board.next_hand_color),
                "target_legal_moves": legal,
                "target_legal_coordinates": [coordinate(square) for square in legal],
                "target_legal_count": len(legal), "scored": bool(legal), "results": {},
            })
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
                episode["continuation_moves"].append(incoming_move)
                episode["continuation_tokens"].append(int(SQUARE_TO_TOKEN[incoming_move]))
                continue
            episode["stop_reason"] = reason
            episode["stop_horizon"] = h
            episode["final_input_length"] = length
            break
        # Use the already sampled move whenever the episode continues. A
        # separate per-case stream supplies only the final lookahead label,
        # preserving the original continuation stream and every existing board.
        for h, step in enumerate(episode["steps"]):
            if not step["scored"]:
                target, source = None, None
            elif h < len(episode["continuation_moves"]):
                target, source = episode["continuation_moves"][h], "continuation"
            else:
                lookahead_rng = np.random.default_rng(np.random.SeedSequence([
                    seed, case["case_id"], 0x434C4541]))
                target = int(lookahead_rng.choice(step["target_legal_moves"]))
                source = "independent_lookahead"
            step.update({
                "next_target_move": target,
                "next_target_token": int(SQUARE_TO_TOKEN[target]) if target is not None else None,
                "next_target_source": source,
            })
        trajectories.append(episode)
    return trajectories


def summarize(episodes: list[dict], horizon: int) -> list[dict]:
    rows = []
    for h in range(horizon + 1):
        reached = [episode["steps"][h] for episode in episodes if len(episode["steps"]) > h]
        scored = [step for step in reached if step["scored"]]
        measurements = [step["results"]["clean"] for step in scored]
        errors = [row["errors"] for row in measurements]
        count = len(measurements)
        perfect = sum(row["perfect"] for row in measurements)
        top1_legal = sum(row["top1_is_legal"] for row in measurements)
        next_correct = sum(row["next_token_correct"] for row in measurements)
        rows.append({
            "action": "ordinary_game", "horizon": h, "method": "clean",
            "eligible_cases": len(episodes), "reached_cases": len(reached),
            "scored_cases": count, "terminal_cases_excluded": len(reached) - count,
            "stopped_before_horizon": len(episodes) - len(reached),
            "perfect_cases": perfect,
            "exact_set_accuracy": perfect / count if count else None,
            "exact_set_percent": 100 * perfect / count if count else None,
            "mean_errors": sum(errors) / count if count else None,
            "total_errors": sum(errors),
            "mean_false_positives": sum(row["false_positives"] for row in measurements) / count if count else None,
            "mean_false_negatives": sum(row["false_negatives"] for row in measurements) / count if count else None,
            "error_distribution": dict(sorted(Counter(errors).items())),
            "prediction_scored_cases": count,
            "top1_legal_cases": top1_legal,
            "top1_legal_accuracy": top1_legal / count if count else None,
            "top1_legal_percent": 100 * top1_legal / count if count else None,
            "next_token_correct_cases": next_correct,
            "next_token_accuracy": next_correct / count if count else None,
            "next_token_percent": 100 * next_correct / count if count else None,
            "mean_next_token_nll": sum(row["next_token_nll"] for row in measurements) / count if count else None,
            "mean_uniform_next_token_chance": sum(row["uniform_next_token_chance"] for row in measurements) / count if count else None,
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


def verification_cases(episodes: list[dict], count: int = 6) -> set[int]:
    """Spread checks across records, including an implicit-pass game if present."""
    selected = set(np.linspace(0, len(episodes) - 1, min(count, len(episodes)), dtype=int).tolist())
    pass_indices = [i for i, episode in enumerate(episodes)
                    if any(step["incoming_implicit_pass"] for step in episode["steps"])]
    if pass_indices and not any(index in selected for index in pass_indices):
        selected.remove(max(selected))
        selected.add(pass_indices[0])
    return {episodes[index]["case_id"] for index in selected}


def compare_full_logits(actual: torch.Tensor, expected: torch.Tensor, device: str,
                        count: int, target_token: int | None, **context) -> dict:
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    atol, rtol = (1e-4 if device == "cpu" else 1e-3), 1e-4
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    actual_tokens = (actual[1:].topk(count).indices + 1).tolist() if count else []
    expected_tokens = (expected[1:].topk(count).indices + 1).tolist() if count else []
    if set(actual_tokens) != set(expected_tokens):
        raise AssertionError(f"Cached/full top-N disagreement: {context}")
    actual_top1, expected_top1 = int(actual.argmax()), int(expected.argmax())
    if actual_top1 != expected_top1:
        raise AssertionError(f"Cached/full top-1 disagreement: {context}")
    actual_nll = -float(actual.log_softmax(dim=-1)[target_token]) if target_token is not None else None
    expected_nll = -float(expected.log_softmax(dim=-1)[target_token]) if target_token is not None else None
    if actual_nll is not None:
        np.testing.assert_allclose(actual_nll, expected_nll, atol=atol, rtol=rtol)
    difference = actual - expected
    return {**context, "max_absolute_error": difference.abs().max().item(),
            "rms_error": difference.square().mean().sqrt().item(),
            "logits_allclose": True,
            "max_tolerance_ratio": (difference.abs() / (atol + rtol * expected.abs())).max().item(),
            "top_n_agreement": True, "target_legal_count": count,
            "actual_top_n_tokens": actual_tokens, "expected_top_n_tokens": expected_tokens,
            "top1_agreement": True, "actual_top1_token": actual_top1,
            "expected_top1_token": expected_top1, "next_target_token": target_token,
            "actual_next_token_nll": actual_nll, "expected_next_token_nll": expected_nll}


def compare_h0(episodes: list[dict], config: dict, reference_dir: Path) -> dict:
    """Re-score clean h0 rankings against the existing baseline's edited targets.

    At h0 all runs have the identical untouched prefix, hence identical clean
    predictions. Targets and N differ, so comparisons take top-N from the full
    clean ranking using each saved baseline's edited-board legal count.
    """
    path = reference_dir / "per_case_results.json"
    config_path = reference_dir / "config.json"
    base = {"reference_directory": str(reference_dir.resolve())}
    if not path.exists() or not config_path.exists():
        return {**base, "checked": False, "reason": "Persistent baseline artifacts unavailable"}
    reference_config = json.loads(config_path.read_text())
    for key in ("checkpoint_sha256", "benchmark_sha256", "model_implementation_sha256",
                "original_engine_sha256", "precision"):
        if config[key] != reference_config[key]:
            return {**base, "checked": False, "reason": f"Persistent baseline {key} differs"}
    originals = {episode["case_id"]: episode for episode in episodes}
    expected_pairs = {(episode["case_id"], action) for episode in episodes
                      for action in ("flip", "erase")
                      if episode[f"{action}_legal_moves"]
                      and (action == "flip" or episode["position"] not in (27, 28, 35, 36))}
    checked_pairs = set()
    checked = 0
    action_counts = Counter()
    for reference in json.loads(path.read_text()):
        if reference["case_id"] not in originals:
            continue
        old_step = reference["steps"][0]
        if not old_step["scored"]:
            continue
        pair = (reference["case_id"], reference["action"])
        if pair in checked_pairs:
            raise AssertionError(f"Duplicate immediate baseline prediction: {pair}")
        checked_pairs.add(pair)
        episode = originals[reference["case_id"]]
        if episode["history"] != reference["history"]:
            raise AssertionError("Baseline/control prefixes differ")
        ranking = episode["steps"][0]["results"]["clean"]["ranked_tokens"]
        selected = ranking[:old_step["target_legal_count"]]
        measurement = score(TOKEN_TO_SQUARE[selected].tolist(), old_step["target_legal_moves"])
        old = old_step["results"]["baseline"]
        if set(measurement["predicted_moves"]) != set(old["predicted_moves"]) or measurement["errors"] != old["errors"]:
            raise AssertionError(f"Immediate baseline mismatch: {reference['case_id']}/{reference['action']}")
        checked += 1
        action_counts[reference["action"]] += 1
    if not checked:
        raise AssertionError("No comparable immediate baseline predictions")
    if checked_pairs != expected_pairs:
        raise AssertionError("Immediate baseline reference does not cover all eligible action/case pairs")
    return {**base, "checked": True, "passed": True, "method_case_predictions": checked,
            "predictions_by_action": dict(action_counts), "reference_sha256": sha256(path),
            "reference_config_sha256": sha256(config_path)}


@torch.inference_mode()
def run(args, device: str) -> dict:
    started = time.monotonic()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    all_cases = load_cases(args.benchmark)
    cases = all_cases if args.limit_cases is None else all_cases[:args.limit_cases]
    episodes = make_trajectories(cases, args.seed, args.horizon, args.max_input_length)
    model = FrozenOthelloGPT(args.checkpoint).to(device=device, dtype=torch.float32).eval()
    config = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Ordinary-game control: no board edit, no model steering, retained autoregressive KV cache",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "benchmark": str(args.benchmark.resolve()), "benchmark_sha256": sha256(args.benchmark),
        "benchmark_records": len(all_cases), "evaluated_records": len(cases),
        "unique_evaluated_prefixes": len({tuple(case["history"]) for case in cases}),
        "prefix_weighting": "One trajectory per benchmark record; repeated prefixes retained",
        "limited_run": args.limit_cases is not None,
        "seed": args.seed, "horizon": args.horizon, "max_input_length": args.max_input_length,
        "sampling": "numpy.default_rng(seed), ascending case-ID order, sorted original-board legal moves; trajectories precomputed before batching",
        "numpy_version": np.__version__, "torch_version": torch.__version__,
        "device": device, "precision": "float32", "batch_size": args.batch_size,
        "continuation": "Uniformly sampled legal tokens from evolving original board; no attention deltas in prefix or continuation",
        "trajectory_comparison": "Original-board trajectories are independent of edited-board trajectories after h0",
        "pass_convention": "Original simulator implicit passes; no pass token",
        "ranking": "Top N over playable tokens 1..60, N = evolving original board legal count; save all60 ranked tokens",
        "metric": "Exact legal-set accuracy, FP+FN, top-1 move legality, exact sampled-next-token accuracy and full-vocabulary NLL; terminal N=0 excluded from all averages",
        "top1_metric": "Argmax over all61 vocabulary tokens; token0 is padding and counts as illegal",
        "next_token_target": "The actual existing continuation token at h when available; the final reached nonterminal step uses one independently sampled legal lookahead token",
        "lookahead_sampling": "numpy.default_rng(numpy.SeedSequence([seed, case_id, 0x434C4541])), sorted legal moves; the trajectory RNG is unchanged",
        "next_token_accuracy_note": "Exact match to a uniformly sampled legal move; distinct from top-1 move legality and exact legal-set accuracy",
        "next_token_nll_definition": "-log_softmax(full61 logits)[next_target_token], in natural-log units",
        "uniform_next_token_chance_definition": "1/N for the evolving original board's N legal moves, averaged over scored cases; expected accuracy of any legal deterministic prediction against a uniform random legal target",
        "prediction_output": "Save all61 raw token logits for independent argmax and NLL verification; terminal prediction metrics and raw logits are null",
        "persistent_reference_dir": str(args.reference_dir.resolve()),
        "runner_sha256": sha256(Path(__file__)),
        "model_implementation_sha256": sha256(REPO / "mech_int/reproduce_linear_probe.py"),
        "original_engine_sha256": sha256(REPO / "data/othello.py"),
    }
    write_json(output / "config.json", config)
    selected_cases = verification_cases(episodes)
    checked_horizons = sorted({h for h in (0, 1, 10) if h <= args.horizon})
    full_check = {
        "checked": True, "passed": True, "case_ids": sorted(selected_cases),
        "requested_horizons": checked_horizons,
        "reference": "FrozenOthelloGPT.forward on the entire original prefix plus sampled continuation",
        "atol": 1e-4 if device == "cpu" else 1e-3, "rtol": 1e-4, "comparisons": [],
    }
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["history_length"]].append(episode)
    completed = 0
    for length, group in sorted(groups.items()):
        for offset in range(0, len(group), args.batch_size):
            batch = group[offset:offset + args.batch_size]
            tokens = torch.tensor(SQUARE_TO_TOKEN[np.array([episode["history"] for episode in batch])], device=device)
            logits, cache = model.forward_cached(tokens)
            logits = logits[:, -1]
            active = batch
            for h in range(args.horizon + 1):
                if cache.length != length + h:
                    raise AssertionError("Cache length does not match ordinary trajectory horizon")
                if not torch.isfinite(logits).all().item():
                    raise FloatingPointError("Non-finite clean cached logits")
                rankings = (logits[:, 1:].topk(60).indices + 1).cpu().tolist()
                top1_tokens = logits.argmax(dim=-1).cpu().tolist()
                all_logits = logits.cpu().tolist()
                log_probabilities = logits.log_softmax(dim=-1)
                for row, episode in enumerate(active):
                    step = episode["steps"][h]
                    selected = rankings[row][:step["target_legal_count"]]
                    if step["scored"]:
                        measurement = score(TOKEN_TO_SQUARE[selected].tolist(), step["target_legal_moves"])
                        measurement.update({"predicted_tokens": selected, "scored": True})
                        if measurement["errors"] % 2:
                            raise AssertionError("Equal-size move-set error must be even")
                    else:
                        measurement = {"target_legal_moves": [], "target_legal_coordinates": [],
                                       "predicted_tokens": [], "predicted_moves": [],
                                       "errors": None, "false_positives": None,
                                       "false_negatives": None, "perfect": None, "scored": False}
                    measurement["ranked_tokens"] = rankings[row]
                    if step["scored"]:
                        top1_token = top1_tokens[row]
                        top1_move = int(TOKEN_TO_SQUARE[top1_token])
                        measurement.update({
                            "token_logits": all_logits[row],
                            "top1_token": top1_token, "top1_move": top1_move,
                            "top1_is_legal": top1_move in step["target_legal_moves"],
                            "next_token_correct": top1_token == step["next_target_token"],
                            "next_token_nll": -float(log_probabilities[row, step["next_target_token"]]),
                            "uniform_next_token_chance": 1 / step["target_legal_count"],
                        })
                    else:
                        measurement.update({key: None for key in (
                            "token_logits", "top1_token", "top1_move", "top1_is_legal",
                            "next_token_correct", "next_token_nll", "uniform_next_token_chance")})
                    step["results"]["clean"] = measurement
                    if episode["case_id"] in selected_cases and h in checked_horizons:
                        history = episode["history"] + episode["continuation_moves"][:h]
                        full_tokens = torch.tensor(SQUARE_TO_TOKEN[np.array([history])], device=device)
                        expected = model(full_tokens)[0, -1]
                        full_check["comparisons"].append(compare_full_logits(
                            logits[row], expected, device, step["target_legal_count"], step["next_target_token"],
                            case_id=episode["case_id"], horizon=h, input_length=len(history),
                            input_tokens_sha256=hashlib.sha256(
                                np.asarray(SQUARE_TO_TOKEN[history], dtype="<i8").tobytes()).hexdigest()))
                continuing = [i for i, episode in enumerate(active) if len(episode["steps"]) > h + 1]
                if not continuing:
                    break
                indices = torch.tensor(continuing, device=device, dtype=torch.long)
                cache = cache.index_select(indices)
                active = [active[i] for i in continuing]
                incoming = torch.tensor([[episode["continuation_tokens"][h]] for episode in active], device=device)
                logits, cache = model.forward_cached(incoming, cache=cache)
                logits = logits[:, -1]
            completed += len(batch)
        print(f"ordinary_game prefix={length}: {completed}/{len(episodes)} trajectories ({time.monotonic()-started:.1f}s)", flush=True)
    summary = summarize(episodes, args.horizon)
    immediate_check = compare_h0(episodes, config, args.reference_dir)
    stopping = dict(Counter(episode["stop_reason"] for episode in episodes))
    step_rows = []
    for episode in episodes:
        for step in episode["steps"]:
            measurement = step["results"]["clean"]
            step_rows.append({
                "case_id": episode["case_id"], "action": "ordinary_game", "method": "clean",
                "position": episode["position"], "history_length": episode["history_length"],
                **{key: step[key] for key in ("horizon", "input_length", "incoming_move", "incoming_token", "incoming_implicit_pass", "next_hand_color", "board_state", "target_legal_count", "scored", "next_target_move", "next_target_token", "next_target_source")},
                **{key: measurement[key] for key in ("errors", "false_positives", "false_negatives", "perfect", "predicted_moves", "predicted_tokens", "ranked_tokens", "top1_token", "top1_move", "top1_is_legal", "next_token_correct", "next_token_nll", "uniform_next_token_chance")},
                "target_legal_moves": step["target_legal_moves"],
                "stop_reason": episode["stop_reason"], "stop_horizon": episode["stop_horizon"],
            })
    result = {
        "config": config, "summary": summary, "stopping_reasons": stopping,
        "trajectories": len(episodes),
        "reached_steps": sum(len(episode["steps"]) for episode in episodes),
        "scored_steps": sum(step["scored"] for episode in episodes for step in episode["steps"]),
        "verification": {"cached_vs_full": full_check, "immediate_baseline": immediate_check},
        "elapsed_seconds": time.monotonic() - started,
        "artifacts": {"per_case_json": "per_case_results.json", "per_step_csv": "per_step_results.csv",
                      "summary_csv": "summary.csv"},
    }
    write_json(output / "per_case_results.json", episodes)
    write_csv(output / "per_step_results.csv", step_rows)
    write_csv(output / "summary.csv", summary)
    write_json(output / "results.json", result)
    print(json.dumps({"output": str(output.resolve()), "stopping_reasons": stopping,
                      "immediate_baseline": immediate_check,
                      "cached_vs_full_comparisons": len(full_check["comparisons"]),
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
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
