"""Combine the edited-board evaluation and ordinary-game reference for reporting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = REPO / "output/linear_probe_reproduction/persistent_interventions"
PREDICTION_METRICS = (
    "prediction_scored_cases", "top1_legal_cases", "top1_legal_accuracy", "top1_legal_percent",
    "next_token_correct_cases", "next_token_accuracy", "next_token_percent",
    "mean_next_token_nll", "mean_uniform_next_token_chance",
)


def combine(directory: Path) -> dict:
    edited_path = directory / "results.json"
    ordinary_path = directory / "ordinary_game_control/results.json"
    edited, ordinary = (json.loads(path.read_text()) for path in (edited_path, ordinary_path))
    for key in ("checkpoint_sha256", "benchmark_sha256", "model_implementation_sha256",
                "evaluated_records", "limited_run", "precision", "seed", "horizon", "max_input_length"):
        if edited["config"][key] != ordinary["config"][key]:
            raise ValueError(f"Comparison runs differ in {key}")
    summaries = []
    for result, target in ((edited, "edited"), (ordinary, "original")):
        for source in result["summary"]:
            method = source["method"]
            steering = "one_time" if method == "patched" else "none"
            arm = ("Patched model, edited board" if method == "patched" else
                   "Unsteered model, edited board" if target == "edited" else
                   "Unsteered model, original board")
            summaries.append({
                "arm": arm, "action": source["action"], "horizon": source["horizon"],
                "steering": steering, "scoring_board": target, "continuation_board": target,
                **{key: source[key] for key in (
                    "eligible_cases", "reached_cases", "scored_cases", "terminal_cases_excluded",
                    "perfect_cases", "exact_set_accuracy", "exact_set_percent", "mean_errors", "total_errors")},
                **{key: source.get(key) for key in PREDICTION_METRICS},
                "uniform_next_token_chance_percent": (
                    100 * source["mean_uniform_next_token_chance"]
                    if source.get("mean_uniform_next_token_chance") is not None else None
                ),
            })
    for result, actions, methods in ((edited, ("flip", "erase"), ("patched", "baseline")),
                                     (ordinary, ("ordinary_game",), ("clean",))):
        expected = {(action, h, method) for action in actions for h in range(result["config"]["horizon"] + 1)
                    for method in methods}
        keys = [(row["action"], row["horizon"], row["method"]) for row in result["summary"]]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError("Missing or duplicate summary rows")
    return {
        "description": "All arms start from the same benchmark prefixes. The ordinary-game reference uses its own legal original-board continuation. The two edited-board arms share continuation tokens and form the paired steering comparison.",
        "prediction_metrics": (
            "Ordinary-game prediction metrics use argmax over all 61 tokens. Top-1 legality asks whether "
            "that prediction is a legal move; next-token accuracy asks whether it matches the uniformly "
            "sampled next legal move. Mean uniform next-token chance averages 1 / legal_move_count. "
            "These differ from exact legal-set accuracy. Empty edited-arm prediction fields mean not measured."
        ),
        "benchmark_records": edited["config"]["evaluated_records"],
        "horizon": edited["config"]["horizon"], "seed": edited["config"]["seed"],
        "inputs": {label: {"file": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for label, path in (("edited_board_results", edited_path), ("ordinary_game_results", ordinary_path))},
        "summary": summaries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    result = combine(args.directory)
    json_path = args.directory / "comparison_with_ordinary_control.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(json_path)
    csv_path = args.directory / "comparison_with_ordinary_control.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["summary"][0]))
        writer.writeheader()
        writer.writerows(result["summary"])
    temporary.replace(csv_path)
    print(f"Saved {len(result['summary'])} comparison rows: {csv_path}")


if __name__ == "__main__":
    main()
