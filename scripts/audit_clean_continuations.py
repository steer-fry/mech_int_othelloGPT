#!/usr/bin/env python3
"""Independently replay and audit the ordinary-game continuation control.

Replays benchmark prefixes and seed-controlled legal moves on the original
simulator. No evaluator trajectory, scoring, or aggregation code is imported.
The saved full rankings and logits permit independent top-N, top-1 and sampled
next-token scoring, plus immediate-prefix comparison with the existing unsteered
counterfactual baseline. Numeric model
verification is reported separately; this auditor validates its coverage and
recorded outcomes rather than evaluating the model again.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import struct

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "output/linear_probe_reproduction/persistent_interventions/ordinary_game_control"
DEFAULT_REFERENCE = DEFAULT_OUTPUT.parent
CENTRAL = {27, 28, 35, 36}
PLAYABLE = [square for square in range(64) if square not in CENTRAL]
TOKEN_TO_SQUARE = dict(enumerate(PLAYABLE, 1))
SQUARE_TO_TOKEN = {square: token for token, square in TOKEN_TO_SQUARE.items()}


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
    if isinstance(expected, bool):
        valid = type(actual) is bool and actual == expected
    elif isinstance(expected, int):
        valid = type(actual) is int and actual == expected
    elif isinstance(expected, float):
        valid = isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-12)
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
    if not isinstance(actual, dict):
        raise AssertionError(f"{label}: expected a dictionary")
    for key, value in expected.items():
        if key not in actual:
            raise AssertionError(f"{label}: missing {key}")
        same(actual[key], value, f"{label}.{key}")


def coordinate(square: int) -> str:
    return f"{chr(ord('a') + square // 8)}{square % 8 + 1}"


def original_engine():
    """Load just the original class, avoiding unrelated plotting/PGN imports."""
    source = REPO / "data/othello.py"
    parsed = ast.parse(source.read_text())
    board_class = next(node for node in parsed.body
                       if isinstance(node, ast.ClassDef) and node.name == "OthelloBoardState")
    namespace = {
        "np": np, "BLACK": 1, "WHITE": -1, "EMPTY": 0,
        "eights": [[-1, 0], [-1, 1], [0, 1], [1, 1],
                   [1, 0], [1, -1], [0, -1], [-1, -1]],
    }
    exec(compile(ast.Module(body=[board_class], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["OthelloBoardState"]


def measurement(row: dict, legal: list[int], target_token: int | None, label: str) -> dict:
    ranked = row["ranked_tokens"]
    if (len(ranked) != 60 or any(type(token) is not int for token in ranked)
            or set(ranked) != set(TOKEN_TO_SQUARE)):
        raise AssertionError(f"{label}: full ranking must be a permutation of tokens 1..60")
    selected = ranked[:len(legal)]
    same(row["predicted_tokens"], selected, f"{label}: top-N ranking prefix")
    predicted = [TOKEN_TO_SQUARE[token] for token in selected]
    positives = sorted(set(predicted) - set(legal))
    negatives = sorted(set(legal) - set(predicted))
    expected = {
        "ranked_tokens": ranked, "predicted_tokens": selected,
        "predicted_moves": predicted, "scored": bool(legal),
        "false_positives": len(positives) if legal else None,
        "false_negatives": len(negatives) if legal else None,
        "errors": len(positives) + len(negatives) if legal else None,
        "perfect": set(predicted) == set(legal) if legal else None,
    }
    if legal:
        expected.update({
            "target_legal_moves": legal,
            "target_legal_coordinates": [coordinate(move) for move in legal],
            "predicted_coordinates": [coordinate(move) for move in predicted],
            "false_positive_moves": positives,
            "false_positive_coordinates": [coordinate(move) for move in positives],
            "false_negative_moves": negatives,
            "false_negative_coordinates": [coordinate(move) for move in negatives],
        })
    expected.update(prediction_metrics(row, legal, target_token, label))
    fields(row, expected, label)
    return expected


def prediction_metrics(row: dict, legal: list[int], target_token: int | None, label: str) -> dict:
    """Recompute full-vocabulary argmax and target likelihood from saved logits."""
    names = ("top1_token", "top1_move", "top1_is_legal", "next_token_correct",
             "next_token_nll", "uniform_next_token_chance")
    if not legal:
        expected = {key: None for key in names}
        expected["token_logits"] = None
        fields(row, expected, label)
        return expected
    logits = row["token_logits"]
    if (len(logits) != 61 or any(isinstance(value, bool) or not isinstance(value, (int, float))
                               or not math.isfinite(value) for value in logits)):
        raise AssertionError(f"{label}: expected 61 finite full-vocabulary logits")
    ranked = row["ranked_tokens"]
    if any(logits[first] < logits[second] for first, second in zip(ranked, ranked[1:])):
        raise AssertionError(f"{label}: playable ranking disagrees with saved logits")
    if type(target_token) is not int or target_token not in TOKEN_TO_SQUARE:
        raise AssertionError(f"{label}: invalid scored next-token target")
    top1 = logits.index(max(logits))  # torch.argmax also chooses the first exact tie.
    move = TOKEN_TO_SQUARE[top1] if top1 else -1
    maximum = max(logits)
    nll = maximum + math.log(math.fsum(math.exp(value - maximum) for value in logits)) - logits[target_token]
    saved_nll = row["next_token_nll"]
    if (isinstance(saved_nll, bool) or not isinstance(saved_nll, (int, float))
            or not math.isfinite(saved_nll) or saved_nll < 0
            or not math.isclose(saved_nll, nll, rel_tol=1e-6, abs_tol=3e-6)):
        raise AssertionError(f"{label}: saved next-token NLL disagrees with independently computed softmax")
    expected = {
        "token_logits": logits, "top1_token": top1, "top1_move": move,
        "top1_is_legal": move in set(legal), "next_token_correct": top1 == target_token,
        # Float32 evaluator NLL is checked above; retain its representation for CSV and means.
        "next_token_nll": saved_nll, "uniform_next_token_chance": 1 / len(legal),
    }
    fields(row, expected, label)
    return expected


def summary_rows(episodes: list[dict], horizon: int) -> list[dict]:
    rows = []
    for h in range(horizon + 1):
        reached = [episode["steps"][h] for episode in episodes if len(episode["steps"]) > h]
        scored = [step for step in reached if step["target_legal_count"] > 0]
        measurements = [step["results"]["clean"] for step in scored]
        count = len(measurements)
        perfect = sum(row["perfect"] for row in measurements)
        legal_top1 = sum(row["top1_is_legal"] for row in measurements)
        next_correct = sum(row["next_token_correct"] for row in measurements)
        errors = [row["errors"] for row in measurements]
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
            "error_distribution": {str(error): amount for error, amount in sorted(Counter(errors).items())},
            "prediction_scored_cases": count,
            "top1_legal_cases": legal_top1,
            "top1_legal_accuracy": legal_top1 / count if count else None,
            "top1_legal_percent": 100 * legal_top1 / count if count else None,
            "next_token_correct_cases": next_correct,
            "next_token_accuracy": next_correct / count if count else None,
            "next_token_percent": 100 * next_correct / count if count else None,
            "mean_next_token_nll": sum(row["next_token_nll"] for row in measurements) / count if count else None,
            "mean_uniform_next_token_chance": sum(row["uniform_next_token_chance"] for row in measurements) / count if count else None,
        })
    return rows


def audit_csv(path: Path, expected_rows: list[dict]) -> None:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if (len(reader.fieldnames or []) != len(expected_rows[0])
                or set(reader.fieldnames or []) != set(expected_rows[0])):
            raise AssertionError(f"{path.name}: columns differ")
        count = 0
        for count, actual in enumerate(reader, 1):
            if count > len(expected_rows):
                raise AssertionError(f"{path.name}: too many rows")
            for key, value in expected_rows[count - 1].items():
                serialized = (json.dumps(value) if isinstance(value, (dict, list))
                              else "" if value is None else str(value))
                if actual[key] != serialized:
                    raise AssertionError(f"{path.name}:{count}:{key}: value differs")
        same(count, len(expected_rows), f"{path.name}: row count")


def immediate_baseline(episodes: list[dict], config: dict, directory: Path) -> dict:
    """Re-score original-prefix rankings against saved counterfactual targets."""
    config_path = directory / "config.json"
    cases_path = directory / "per_case_results.json"
    if not config_path.exists() or not cases_path.exists():
        return {"checked": False, "reason": "Persistent baseline artifacts unavailable"}
    reference_config = json.loads(config_path.read_text())
    for key in ("checkpoint_sha256", "benchmark_sha256", "model_implementation_sha256",
                "original_engine_sha256", "precision"):
        if config[key] != reference_config.get(key):
            return {"checked": False, "reason": f"Persistent baseline {key} differs"}
    references = {}
    for row in json.loads(cases_path.read_text()):
        key = (row["case_id"], row["action"])
        if key in references:
            raise AssertionError(f"Duplicate persistent reference key {key}")
        references[key] = row
    checked = 0
    action_counts = Counter()
    for episode in episodes:
        ranked = episode["steps"][0]["results"]["clean"]["ranked_tokens"]
        for action in ("flip", "erase"):
            if action == "erase" and episode["position"] in CENTRAL:
                continue
            key = episode["case_id"], action
            if key not in references:
                return {"checked": False, "reason": "persistent intervention case coverage differs"}
            reference = references[key]
            same(reference["history"], episode["history"], f"h0 reference {key}: history")
            same(reference["position"], episode["position"], f"h0 reference {key}: position")
            step = reference["steps"][0]
            same(step["horizon"], 0, f"h0 reference {key}: horizon")
            legal = step["target_legal_moves"]
            if (any(type(move) is not int or move not in SQUARE_TO_TOKEN for move in legal)
                    or sorted(set(legal)) != legal):
                raise AssertionError(f"h0 reference {key}: invalid target legal moves")
            same(step["target_legal_count"], len(legal), f"h0 reference {key}: legal count")
            if f"{action}_legal_moves" in episode:
                same(legal, episode[f"{action}_legal_moves"], f"h0 reference {key}: saved target")
            if not legal:
                continue
            selected = ranked[:len(legal)]
            predicted = [TOKEN_TO_SQUARE[token] for token in selected]
            old = step["results"]["baseline"]
            old_tokens = old["predicted_tokens"]
            if (len(old_tokens) != len(legal) or len(set(old_tokens)) != len(legal)
                    or any(type(token) is not int or token not in TOKEN_TO_SQUARE for token in old_tokens)):
                raise AssertionError(f"h0 reference {key}: invalid saved baseline token selection")
            same(old["predicted_moves"], [TOKEN_TO_SQUARE[token] for token in old_tokens],
                 f"h0 reference {key}: baseline token/move mapping")
            if set(predicted) != set(old["predicted_moves"]):
                raise AssertionError(f"h0 reference {key}: unsteered top-N move sets differ")
            errors = len(set(predicted) - set(legal)) + len(set(legal) - set(predicted))
            fields(old, {"errors": errors, "perfect": errors == 0, "scored": True,
                         "false_positives": errors // 2, "false_negatives": errors // 2},
                   f"h0 reference {key}: independently re-scored baseline")
            checked += 1
            action_counts[action] += 1
    return {"checked": True, "passed": True, "method_case_predictions": checked,
            "predictions_by_action": dict(action_counts),
            "reference_sha256": sha256(cases_path),
            "reference_config_sha256": sha256(config_path),
            "reference_directory": str(directory.resolve())}


def numeric_verification(report: dict, episodes: list[dict], config: dict) -> dict:
    """Check saved cached-vs-full verification records, without reusing logits."""
    if not report.get("checked", False):
        same(report.get("checked"), False, "cached-vs-full checked")
        return {"checked": False}
    same(report.get("passed"), True, "cached-vs-full passed")
    case_ids = report["case_ids"]
    requested = report["requested_horizons"]
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise AssertionError("Numeric verification must include distinct cases")
    if (not requested or sorted(set(requested)) != requested
            or any(type(h) is not int or not 0 <= h <= config["horizon"] for h in requested)):
        raise AssertionError("Invalid numeric verification horizons")
    by_id = {episode["case_id"]: episode for episode in episodes}
    if any(type(case_id) is not int or case_id not in by_id for case_id in case_ids):
        raise AssertionError("Numeric verification case not in evaluated benchmark")
    expected = [(case_id, h) for case_id in case_ids for h in requested
                if h < len(by_id[case_id]["steps"])]
    rows = report["comparisons"]
    same(sorted((row["case_id"], row["horizon"]) for row in rows), sorted(expected),
         "cached-vs-full comparison coverage")
    atol, rtol = report["atol"], report["rtol"]
    for key, value in (("atol", atol), ("rtol", rtol)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise AssertionError(f"cached-vs-full: invalid {key}")
    for row in rows:
        label = f"cached-vs-full case={row['case_id']}/h={row['horizon']}"
        if type(row["case_id"]) is not int or type(row["horizon"]) is not int:
            raise AssertionError(f"{label}: case/horizon must be integers")
        for key in ("max_absolute_error", "rms_error"):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise AssertionError(f"{label}: invalid {key}")
        if row["rms_error"] > row["max_absolute_error"] + 1e-12:
            raise AssertionError(f"{label}: RMS exceeds maximum error")
        same(row["top_n_agreement"], True, f"{label}: top-N agreement")
        episode = by_id[row["case_id"]]
        step = episode["steps"][row["horizon"]]
        history = episode["history"] + episode["continuation_moves"][:row["horizon"]]
        input_tokens = [SQUARE_TO_TOKEN[move] for move in history]
        digest = hashlib.sha256(struct.pack(f"<{len(input_tokens)}q", *input_tokens)).hexdigest()
        fields(row, {"input_length": len(history), "input_tokens_sha256": digest,
                     "target_legal_count": step["target_legal_count"]}, label)
        count = step["target_legal_count"]
        expected_selected = step["results"]["clean"]["ranked_tokens"][:count]
        for key in ("actual_top_n_tokens", "expected_top_n_tokens"):
            selected = row[key]
            if (len(selected) != count or len(set(selected)) != count
                    or any(type(token) is not int or token not in TOKEN_TO_SQUARE for token in selected)
                    or set(selected) != set(expected_selected)):
                raise AssertionError(f"{label}: {key} differs from independently scored top-N set")
        fields(row, {"top1_agreement": True, "next_target_token": step["next_target_token"]}, label)
        actual_top1, expected_top1 = row["actual_top1_token"], row["expected_top1_token"]
        if (type(actual_top1) is not int or not 0 <= actual_top1 <= 60
                or type(expected_top1) is not int or actual_top1 != expected_top1):
            raise AssertionError(f"{label}: cached/full top1 tokens are invalid or disagree")
        if count:
            same(actual_top1, step["results"]["clean"]["top1_token"], f"{label}: saved argmax")
            for key in ("actual_next_token_nll", "expected_next_token_nll"):
                value = row[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise AssertionError(f"{label}: invalid {key}")
            actual_nll, expected_nll = row["actual_next_token_nll"], row["expected_next_token_nll"]
            if not math.isclose(actual_nll, step["results"]["clean"]["next_token_nll"],
                                rel_tol=1e-6, abs_tol=3e-6):
                raise AssertionError(f"{label}: numeric reference NLL disagrees with saved cached logits")
            if abs(actual_nll - expected_nll) > atol + rtol * abs(expected_nll) + 1e-12:
                raise AssertionError(f"{label}: cached/full target NLL exceeds documented tolerances")
        else:
            fields(row, {"actual_next_token_nll": None, "expected_next_token_nll": None}, label)
        if rtol == 0:
            if row["max_absolute_error"] > atol:
                raise AssertionError(f"{label}: error exceeds absolute tolerance")
        same(row["logits_allclose"], True, f"{label}: logits allclose")
        ratio = row["max_tolerance_ratio"]
        if (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
                or not math.isfinite(ratio) or not 0 <= ratio <= 1):
            raise AssertionError(f"{label}: logits difference exceeds documented tolerances")
    return {"checked": True, "passed": True, "comparisons": len(rows),
            "case_ids": case_ids, "requested_horizons": requested,
            "max_absolute_error": max(row["max_absolute_error"] for row in rows)}


def audit(directory: Path, reference_directory: Path | None = None) -> dict:
    config = json.loads((directory / "config.json").read_text())
    result = json.loads((directory / "results.json").read_text())
    episodes = json.loads((directory / "per_case_results.json").read_text())
    same(result["config"], config, "embedded config")
    same(config["precision"], "float32", "precision")
    horizon, length_limit = config["horizon"], config["max_input_length"]
    if (type(horizon) is not int or horizon < 0 or type(length_limit) is not int
            or not 1 <= length_limit <= 59):
        raise AssertionError("Invalid horizon or model input length limit")
    input_paths = {
        "benchmark": Path(config["benchmark"]), "checkpoint": Path(config["checkpoint"]),
        "original_engine": REPO / "data/othello.py",
        "runner": REPO / "mech_int/reproduce_clean_continuations.py",
        "model_implementation": REPO / "mech_int/reproduce_linear_probe.py",
    }
    for name, path in input_paths.items():
        same(sha256(path), config[f"{name}_sha256"], f"{name} provenance")
    with input_paths["benchmark"].open("rb") as handle:
        benchmark = PlainUnpickler(handle).load()
    if not isinstance(benchmark, list) or not benchmark:
        raise AssertionError("Benchmark must be a nonempty list")
    same(len(benchmark), config["benchmark_records"], "benchmark record count")
    count = config["evaluated_records"]
    if type(count) is not int or not 1 <= count <= len(benchmark):
        raise AssertionError("Invalid evaluated record count")
    if not config["limited_run"]:
        same(count, len(benchmark), "full benchmark inclusion")
    same([episode["case_id"] for episode in episodes], list(range(count)),
         "all records in case-ID order, including h0 failures")
    if "unique_evaluated_prefixes" in config:
        same(config["unique_evaluated_prefixes"], len({tuple(row["history"]) for row in episodes}),
             "unique original prefix count with repeated-record weighting")
    engine = original_engine()
    rng = np.random.default_rng(config["seed"])
    replayed, csv_rows = [], []
    passes = 0
    for episode in episodes:
        case_id = episode["case_id"]
        label = f"case={case_id}"
        raw = benchmark[case_id]
        history = [int(move) for move in raw["history"]]
        position, original_color = int(raw["pos_int"]), int(raw["ori_color"])
        if (not 1 <= len(history) <= length_limit or any(move not in SQUARE_TO_TOKEN for move in history)
                or not 0 <= position < 64 or original_color not in (0, 2)):
            raise AssertionError(f"{label}: invalid benchmark record")
        fields(episode, {"action": "ordinary_game", "history": history, "history_length": len(history),
                         "position": position, "ori_color": original_color}, label)
        board = engine()
        for move in history:
            if move not in board.get_valid_moves():
                raise AssertionError(f"{label}: illegal original history move")
            board.umpire(move)
        same(int(board.state.reshape(64)[position]), original_color - 1, f"{label}: original color")
        same(episode["next_hand_color"], int(board.next_hand_color), f"{label}: original turn")
        same(episode["original_legal_moves"], sorted(int(move) for move in board.get_valid_moves()),
             f"{label}: original legal moves")
        incoming, implicit_pass = None, False
        moves, tokens, steps = [], [], []
        for h, stored in enumerate(episode["steps"]):
            step_label = f"{label}/h={h}"
            if h > horizon or len(history) + h > length_limit:
                raise AssertionError(f"{step_label}: continued past horizon/input limit")
            legal = sorted(int(move) for move in board.get_valid_moves())
            if any(move not in SQUARE_TO_TOKEN for move in legal):
                raise AssertionError(f"{step_label}: unrepresentable central move")
            expected_step = {
                "horizon": h, "input_length": len(history) + h,
                "incoming_move": incoming,
                "incoming_token": SQUARE_TO_TOKEN[incoming] if incoming is not None else None,
                "incoming_implicit_pass": implicit_pass,
                "board_state": board.state.astype(int).reshape(64).tolist(),
                "next_hand_color": int(board.next_hand_color),
                "target_legal_moves": legal,
                "target_legal_coordinates": [coordinate(move) for move in legal],
                "target_legal_count": len(legal), "scored": bool(legal),
            }
            if not legal:
                target_move, target_token, target_source = None, None, None
            elif h + 1 < len(episode["steps"]):
                target_move = episode["continuation_moves"][h]
                if type(target_move) is not int or target_move not in legal:
                    raise AssertionError(f"{step_label}: next target is not a legal continuation move")
                target_token = SQUARE_TO_TOKEN[target_move]
                target_source = "continuation"
            else:
                lookahead = np.random.default_rng(np.random.SeedSequence([config["seed"], case_id, 0x434C4541]))
                target_move = int(lookahead.choice(legal))
                target_token = SQUARE_TO_TOKEN[target_move]
                target_source = "independent_lookahead"
            expected_step.update({"next_target_move": target_move, "next_target_token": target_token,
                                  "next_target_source": target_source})
            fields(stored, expected_step, step_label)
            same(set(stored["results"]), {"clean"}, f"{step_label}: methods")
            measured = measurement(stored["results"]["clean"], legal, target_token, f"{step_label}/clean")
            expected_step["results"] = {"clean": measured}
            csv_rows.append({
                "case_id": case_id, "action": "ordinary_game", "method": "clean",
                "position": position, "history_length": len(history),
                **{key: expected_step[key] for key in ("horizon", "input_length", "incoming_move", "incoming_token",
                                                      "incoming_implicit_pass", "next_hand_color", "board_state",
                                                      "target_legal_count", "scored", "next_target_move",
                                                      "next_target_token", "next_target_source")},
                **{key: measured[key] for key in ("errors", "false_positives", "false_negatives", "perfect",
                                                 "predicted_moves", "predicted_tokens", "ranked_tokens",
                                                 "top1_token", "top1_move", "top1_is_legal", "next_token_correct",
                                                 "next_token_nll", "uniform_next_token_chance")},
                "target_legal_moves": legal, "stop_reason": episode["stop_reason"],
                "stop_horizon": episode["stop_horizon"],
            })
            steps.append(expected_step)
            stop = ("terminal_board" if not legal else "input_length_limit" if len(history) + h >= length_limit
                    else "horizon_limit" if h == horizon else None)
            if stop is not None:
                fields(episode, {"stop_reason": stop, "stop_horizon": h,
                                 "final_input_length": len(history) + h}, label)
                same(len(episode["steps"]), h + 1, f"{label}: immediate stop at limit")
                break
            incoming = int(rng.choice(legal))
            if incoming in history or incoming in moves:
                raise AssertionError(f"{step_label}: ordinary play repeats an occupied square")
            implicit_pass = bool(board.tentative_move(incoming) == 2)
            passes += implicit_pass
            board.umpire(incoming)
            moves.append(incoming)
            tokens.append(SQUARE_TO_TOKEN[incoming])
        else:
            raise AssertionError(f"{label}: trajectory stopped before any declared limit")
        fields(episode, {"continuation_moves": moves, "continuation_tokens": tokens}, label)
        same(len(moves), len(steps) - 1, f"{label}: one move per continuation horizon")
        replayed.append({"action": "ordinary_game", "steps": steps, "stop_reason": stop})
    summaries = summary_rows(replayed, horizon)
    stopping = dict(Counter(episode["stop_reason"] for episode in replayed))
    reached_steps = sum(len(episode["steps"]) for episode in replayed)
    scored_steps = sum(step["target_legal_count"] > 0 for episode in replayed for step in episode["steps"])
    fields(result, {"summary": summaries, "stopping_reasons": stopping, "trajectories": len(episodes),
                    "reached_steps": reached_steps, "scored_steps": scored_steps}, "results")
    audit_csv(directory / "summary.csv", summaries)
    audit_csv(directory / "per_step_results.csv", csv_rows)
    reference = reference_directory or Path(config.get("persistent_reference_dir", DEFAULT_REFERENCE))
    baseline_check = immediate_baseline(episodes, config, reference)
    stored_baseline = result.get("verification", {}).get("immediate_baseline")
    if stored_baseline is not None:
        fields(stored_baseline, baseline_check, "saved immediate baseline verification")
    cached_check = numeric_verification(result.get("verification", {}).get("cached_vs_full", {"checked": False}),
                                        episodes, config)
    return {
        "passed": True, "audited_at": datetime.now(timezone.utc).isoformat(),
        "directory": str(directory.resolve()), "benchmark_records": len(benchmark),
        "evaluated_records": count, "limited_run": config["limited_run"],
        "trajectories": len(episodes), "reached_steps": reached_steps,
        "scored_steps": scored_steps, "prediction_scores_checked": scored_steps,
        "top1_legality_scores_checked": scored_steps, "next_token_scores_checked": scored_steps,
        "full_logit_vectors_checked": scored_steps, "next_token_nll_values_checked": scored_steps,
        "full_rankings_checked": reached_steps, "terminal_steps_excluded": reached_steps - scored_steps,
        "summary_rows_checked": len(summaries), "implicit_passes": passes,
        "stopping_reasons": stopping,
        "horizon_zero_failures_included": summaries[0]["scored_cases"] - summaries[0]["perfect_cases"],
        "immediate_baseline": baseline_check, "cached_vs_full": cached_check,
        "checks": ["every benchmark record in original case-ID order including h0 failures",
                   "benchmark/checkpoint/evaluator/model/original-engine provenance hashes",
                   "original history and evolving original board without simulator edits",
                   "independent seeded legal continuation and token mapping",
                   "board states, turns, legal sets and implicit passes at every horizon",
                   "terminal N=0 exclusions and exact horizon/input-length stopping",
                   "full token rankings and exact top-N prefixes",
                   "every FP/FN set, FP+FN score and exact-set outcome",
                   "saved 61-token logits, descending rankings and full-vocabulary argmax including padding",
                   "continuation targets and independent final-step lookahead targets",
                   "top1 move legality and sampled next-token accuracy, terminal exclusions and denominators",
                   "full-vocabulary next-token softmax NLL and uniform legal sampling chance",
                   "all JSON/CSV aggregates, counts, denominators and per-step CSV rows",
                   "h0 full rankings re-scored against compatible persistent unsteered baseline targets",
                   "coverage and reported outcomes of numeric cached-versus-full verification"],
        "scope": "Independent artifact replay, logit-based scoring and arithmetic; cached-versus-full model comparisons are recorded by the evaluator",
        "artifact_sha256": {name: sha256(directory / name) for name in
                            ("config.json", "per_case_results.json", "per_step_results.csv", "summary.csv", "results.json")},
        "audit_script_sha256": sha256(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit(args.directory, args.reference_dir)
    path = args.report or args.directory / "artifact_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps({key: report[key] for key in (
        "passed", "evaluated_records", "trajectories", "scored_steps", "prediction_scores_checked",
        "full_rankings_checked", "terminal_steps_excluded", "implicit_passes", "stopping_reasons",
        "horizon_zero_failures_included", "top1_legality_scores_checked", "next_token_scores_checked",
        "full_logit_vectors_checked", "next_token_nll_values_checked", "immediate_baseline", "cached_vs_full")}, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
