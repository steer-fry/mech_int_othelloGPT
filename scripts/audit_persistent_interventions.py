#!/usr/bin/env python3
"""Independently replay and audit saved persistent-intervention artifacts.

No evaluator scoring, trajectory, or summary functions are imported. NumPy is
used only by the original simulator and to reproduce the documented RNG; all
scores and aggregates are recomputed with Python sets and integer arithmetic.
This verifies stored predictions, not model logits or KV-cache equivalence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.verify_probe_games import original_engine

CENTRAL = {27, 28, 35, 36}
PLAYABLE = [square for square in range(64) if square not in CENTRAL]
TOKEN_TO_SQUARE = {token: square for token, square in enumerate(PLAYABLE, 1)}
SQUARE_TO_TOKEN = {square: token for token, square in TOKEN_TO_SQUARE.items()}
DEFAULT_OUTPUT = REPO / "output/linear_probe_reproduction/persistent_interventions"


class PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Benchmark must contain only plain Python containers")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def same(actual, expected, label: str) -> None:
    if isinstance(expected, float):
        valid = isinstance(actual, (int, float)) and math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    elif isinstance(expected, bool):
        valid = type(actual) is bool and actual == expected
    elif isinstance(expected, int):
        valid = type(actual) is int and actual == expected
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{label}: dictionary keys differ")
        for key, value in expected.items():
            same(actual[key], value, f"{label}.{key}")
        return
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{label}: list lengths differ")
        for index, value in enumerate(expected):
            same(actual[index], value, f"{label}[{index}]")
        return
    else:
        valid = actual == expected
    if not valid:
        raise AssertionError(f"{label}: found {actual!r}, expected {expected!r}")


def fields(actual: dict, expected: dict, label: str) -> None:
    for key, value in expected.items():
        if key not in actual:
            raise AssertionError(f"{label}: missing {key}")
        same(actual[key], value, f"{label}.{key}")


def coordinate(square: int) -> str:
    return f"{chr(ord('a') + square // 8)}{square % 8 + 1}"


def clone(board, board_type):
    copied = board_type()
    copied.state = board.state.copy()
    copied.next_hand_color = board.next_hand_color
    copied.history = list(board.history)
    return copied


def measurement(row: dict, legal: list[int], label: str) -> dict:
    tokens = row["predicted_tokens"]
    if len(tokens) != len(legal) or len(set(tokens)) != len(tokens):
        raise AssertionError(f"{label}: predictions must contain N distinct tokens")
    if any(type(token) is not int or token not in TOKEN_TO_SQUARE for token in tokens):
        raise AssertionError(f"{label}: predicted tokens must be playable tokens 1..60")
    predicted = [TOKEN_TO_SQUARE[token] for token in tokens]
    positives = sorted(set(predicted) - set(legal))
    negatives = sorted(set(legal) - set(predicted))
    expected = {
        "predicted_tokens": tokens, "predicted_moves": predicted,
        "scored": bool(legal),
        "false_positives": len(positives) if legal else None,
        "false_negatives": len(negatives) if legal else None,
        "errors": len(positives) + len(negatives) if legal else None,
        "perfect": set(predicted) == set(legal) if legal else None,
    }
    if legal:
        expected.update({
            "target_legal_moves": legal,
            "target_legal_coordinates": [coordinate(square) for square in legal],
            "predicted_coordinates": [coordinate(square) for square in predicted],
            "false_positive_moves": positives,
            "false_positive_coordinates": [coordinate(square) for square in positives],
            "false_negative_moves": negatives,
            "false_negative_coordinates": [coordinate(square) for square in negatives],
        })
    fields(row, expected, label)
    return expected


def summary_rows(episodes: list[dict], horizon: int) -> list[dict]:
    """Aggregate independently recomputed measurements, including every h=0 case."""
    rows = []
    for action in ("flip", "erase"):
        eligible = [episode for episode in episodes if episode["action"] == action]
        for h in range(horizon + 1):
            reached = [episode["steps"][h] for episode in eligible if len(episode["steps"]) > h]
            scored = [step for step in reached if step["target_legal_count"] > 0]
            for method in ("patched", "baseline"):
                measurements = [step["results"][method] for step in scored]
                count = len(measurements)
                perfect = sum(row["perfect"] for row in measurements)
                errors = [row["errors"] for row in measurements]
                rows.append({
                    "action": action, "horizon": h, "method": method,
                    "eligible_cases": len(eligible), "reached_cases": len(reached),
                    "scored_cases": count, "terminal_cases_excluded": len(reached) - count,
                    "stopped_before_horizon": len(eligible) - len(reached),
                    "perfect_cases": perfect,
                    "exact_set_accuracy": perfect / count if count else None,
                    "exact_set_percent": 100 * perfect / count if count else None,
                    "mean_errors": sum(errors) / count if count else None,
                    "total_errors": sum(errors),
                    "mean_false_positives": sum(row["false_positives"] for row in measurements) / count if count else None,
                    "mean_false_negatives": sum(row["false_negatives"] for row in measurements) / count if count else None,
                    "original_board_history_valid_cases": sum(step["original_board_history_valid"] for step in scored),
                    "legal_sets_differ_from_original_board_cases": sum(step["legal_sets_differ_from_original_board"] is True for step in scored),
                    "error_distribution": {str(error): amount for error, amount in sorted(Counter(errors).items())},
                })
    return rows


def audit_csv(path: Path, expected_rows: list[dict]) -> None:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != set(expected_rows[0]):
            raise AssertionError(f"{path.name}: columns differ")
        count = 0
        for count, actual in enumerate(reader, 1):
            if count > len(expected_rows):
                raise AssertionError(f"{path.name}: too many rows")
            expected = expected_rows[count - 1]
            for key, value in expected.items():
                serialized = (json.dumps(value) if isinstance(value, (dict, list))
                              else "" if value is None else str(value))
                if actual[key] != serialized:
                    raise AssertionError(f"{path.name}:{count}:{key}: value differs")
        same(count, len(expected_rows), f"{path.name}: row count")


def audit(directory: Path) -> dict:
    config = json.loads((directory / "config.json").read_text())
    result = json.loads((directory / "results.json").read_text())
    episodes = json.loads((directory / "per_case_results.json").read_text())
    same(result["config"], config, "embedded config")
    fields(config, {
        "probe_layer_zero_based": 5,
        "flip": {"strength": 2.3, "blocks_zero_based": list(range(2, 8))},
        "erase": {"strength": 1.3, "blocks_zero_based": list(range(8))},
        "precision": "float32",
    }, "config")
    horizon, length_limit = config["horizon"], config["max_input_length"]
    if not 0 <= horizon or not 1 <= length_limit <= 59:
        raise AssertionError("Invalid horizon or model input length limit")
    input_paths = {
        "benchmark": Path(config["benchmark"]), "checkpoint": Path(config["checkpoint"]),
        "probe": Path(config["probe"]), "original_engine": REPO / "data/othello.py",
        "runner": REPO / "mech_int/reproduce_persistent_interventions.py",
        "model_implementation": REPO / "mech_int/reproduce_linear_probe.py",
    }
    for name, path in input_paths.items():
        same(sha256(path), config[f"{name}_sha256"], f"{name} provenance")
    with input_paths["benchmark"].open("rb") as handle:
        benchmark = PlainUnpickler(handle).load()
    same(len(benchmark), config["benchmark_records"], "benchmark record count")
    count = config["evaluated_records"]
    if not 1 <= count <= len(benchmark):
        raise AssertionError("Invalid evaluated benchmark record count")
    if not config["limited_run"]:
        same(count, len(benchmark), "full benchmark inclusion")
    expected_keys = [(case_id, action) for case_id in range(count)
                     for action in ("flip", "erase")
                     if action != "erase" or int(benchmark[case_id]["pos_int"]) not in CENTRAL]
    keys = [(episode["case_id"], episode["action"]) for episode in episodes]
    same(keys, expected_keys, "complete case/action coverage including h=0 failures")
    expected_exclusions = [{"case_id": case_id, "action": "erase",
                            "reason": "central_square_has_no_move_token"}
                           for case_id in range(count) if int(benchmark[case_id]["pos_int"]) in CENTRAL]
    same(result["excluded_cases"], expected_exclusions, "central ERASE exclusions")
    engine = original_engine(REPO)
    rng = np.random.default_rng(config["seed"])
    replayed = []
    csv_rows = []
    passes = repeated_history_moves = repeated_continuation_moves = edited_square_changes = 0
    for episode in episodes:
        case_id, action = episode["case_id"], episode["action"]
        label = f"case={case_id}/{action}"
        raw = benchmark[case_id]
        history = [int(move) for move in raw["history"]]
        position, original_color = int(raw["pos_int"]), int(raw["ori_color"])
        fields(episode, {"history": history, "history_length": len(history),
                         "position": position, "ori_color": original_color}, label)
        board = engine()
        for move in history:
            if move not in board.get_valid_moves():
                raise AssertionError(f"{label}: illegal original history move")
            board.umpire(move)
        same(int(board.state.reshape(64)[position]), original_color - 1, f"{label}: original color")
        same(episode["next_hand_color"], int(board.next_hand_color), f"{label}: original turn")
        original_branch = clone(board, engine)
        board.state.reshape(64)[position] = (-board.state.reshape(64)[position] if action == "flip" else 0)
        moves, tokens, steps = [], [], []
        incoming, implicit_pass = None, False
        previous_edited_color = int(board.state.reshape(64)[position])
        for h, stored in enumerate(episode["steps"]):
            step_label = f"{label}/h={h}"
            if h > horizon or len(history) + h > length_limit:
                raise AssertionError(f"{step_label}: continued past horizon/input limit")
            legal = sorted(int(move) for move in board.get_valid_moves())
            if any(move not in SQUARE_TO_TOKEN for move in legal):
                raise AssertionError(f"{step_label}: target contains a central move")
            original_legal = (sorted(int(move) for move in original_branch.get_valid_moves())
                              if original_branch is not None else None)
            edited_color = int(board.state.reshape(64)[position])
            edited_square_changes += edited_color != previous_edited_color
            previous_edited_color = edited_color
            expected_step = {
                "horizon": h, "input_length": len(history) + h,
                "incoming_move": incoming, "incoming_token": SQUARE_TO_TOKEN[incoming] if incoming is not None else None,
                "incoming_implicit_pass": implicit_pass,
                "board_state": board.state.astype(int).reshape(64).tolist(),
                "next_hand_color": int(board.next_hand_color), "edited_square_color": edited_color,
                "target_legal_moves": legal, "target_legal_coordinates": [coordinate(move) for move in legal],
                "target_legal_count": len(legal), "scored": bool(legal),
                "original_board_legal_moves": original_legal,
                "original_board_history_valid": original_branch is not None,
                "legal_sets_differ_from_original_board": legal != original_legal if original_legal is not None else None,
            }
            fields(stored, expected_step, step_label)
            same(set(stored["results"]), {"patched", "baseline"}, f"{step_label}: methods")
            expected_step["results"] = {}
            for method in ("patched", "baseline"):
                scored = measurement(stored["results"][method], legal, f"{step_label}/{method}")
                expected_step["results"][method] = scored
                csv_rows.append({
                    "case_id": case_id, "action": action, "method": method,
                    "position": position, "history_length": len(history),
                    **{key: expected_step[key] for key in ("horizon", "input_length", "incoming_move", "incoming_implicit_pass", "target_legal_count", "scored")},
                    **{key: scored[key] for key in ("errors", "false_positives", "false_negatives", "perfect", "predicted_moves", "predicted_tokens")},
                    "target_legal_moves": legal, "stop_reason": episode["stop_reason"], "stop_horizon": episode["stop_horizon"],
                })
            steps.append(expected_step)
            stop = ("terminal_board" if not legal else "input_length_limit" if len(history) + h >= length_limit
                    else "horizon_limit" if h == horizon else None)
            if stop is not None:
                fields(episode, {"stop_reason": stop, "stop_horizon": h,
                                 "final_input_length": len(history) + h}, label)
                same(len(episode["steps"]), h + 1, f"{label}: stop immediately at limit")
                break
            incoming = int(rng.choice(legal))
            implicit_pass = bool(board.tentative_move(incoming) == 2)
            passes += implicit_pass
            repeated_history_moves += incoming in history
            repeated_continuation_moves += incoming in moves
            board.umpire(incoming)  # Apply captures and passes; never reapply the edit.
            if original_branch is not None:
                if incoming in original_branch.get_valid_moves():
                    original_branch.umpire(incoming)
                else:
                    original_branch = None
            moves.append(incoming)
            tokens.append(SQUARE_TO_TOKEN[incoming])
        else:
            raise AssertionError(f"{label}: trajectory stopped without reaching any limit")
        fields(episode, {"continuation_moves": moves, "continuation_tokens": tokens}, label)
        same(len(moves), len(steps) - 1, f"{label}: one shared token per continuation step")
        replayed.append({"action": action, "steps": steps, "stop_reason": stop})
    summaries = summary_rows(replayed, horizon)
    stopping = {action: dict(Counter(episode["stop_reason"] for episode in replayed if episode["action"] == action))
                for action in ("flip", "erase")}
    reached_steps = sum(len(episode["steps"]) for episode in replayed)
    scored_steps = sum(step["target_legal_count"] > 0 for episode in replayed for step in episode["steps"])
    fields(result, {"summary": summaries, "stopping_reasons": stopping,
                    "trajectories": len(episodes), "reached_steps": reached_steps,
                    "scored_steps": scored_steps}, "results")
    audit_csv(directory / "summary.csv", summaries)
    audit_csv(directory / "per_step_results.csv", csv_rows)
    return {
        "passed": True, "audited_at": datetime.now(timezone.utc).isoformat(),
        "directory": str(directory.resolve()), "benchmark_records": len(benchmark),
        "evaluated_records": count, "limited_run": config["limited_run"],
        "trajectories": len(episodes), "reached_steps": reached_steps,
        "scored_steps": scored_steps, "prediction_scores_checked": 2 * scored_steps,
        "terminal_steps_excluded": reached_steps - scored_steps, "summary_rows_checked": len(summaries),
        "implicit_passes": passes, "continuation_moves_repeating_original_history": repeated_history_moves,
        "continuation_moves_repeating_continuation": repeated_continuation_moves,
        "natural_edited_square_color_changes": edited_square_changes,
        "stopping_reasons": stopping, "excluded_cases": expected_exclusions,
        "horizon_zero_failures_included": {
            action: next(row["scored_cases"] - row["perfect_cases"] for row in summaries
                         if row["action"] == action and row["horizon"] == 0 and row["method"] == "patched")
            for action in ("flip", "erase")
        },
        "checks": ["complete benchmark case/action coverage including h=0 failures",
                   "input hashes and central-square ERASE exclusions", "original-history simulator replay",
                   "one-time edit with preserved turn and natural subsequent captures",
                   "seeded legal continuation and shared token/move mapping",
                   "stored board states, turns, legal sets and implicit passes at every horizon",
                   "terminal exclusion and exact stopping conditions, input length <=59",
                   "every predicted token/move, FP/FN set and FP+FN score",
                   "every JSON/CSV aggregate and per-step CSV row"],
        "scope": "Artifact replay and arithmetic; model logits/top-N rankings and KV-cache equivalence are checked separately",
        "artifact_sha256": {name: sha256(directory / name) for name in
                            ("config.json", "per_case_results.json", "per_step_results.csv", "summary.csv", "results.json")},
        "audit_script_sha256": sha256(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit(args.directory)
    path = args.report or args.directory / "artifact_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps({key: report[key] for key in (
        "passed", "evaluated_records", "trajectories", "scored_steps", "prediction_scores_checked",
        "terminal_steps_excluded", "implicit_passes", "continuation_moves_repeating_original_history",
        "natural_edited_square_color_changes", "stopping_reasons", "horizon_zero_failures_included")}, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
