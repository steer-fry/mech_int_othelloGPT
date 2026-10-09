#!/usr/bin/env python3
"""Audit generated mixed edits and verify their inference against TL 1.6.0.

Checks the approved one-sample, nested k=2/4/8 benchmark and tests actual
TransformerLens attention-output hooks against compact CPU/MPS batches. Only
newly trained layer-five probes are used; no extra control cases are generated.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.metadata
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mech_int.reproduce_linear_probe import FrozenOthelloGPT
from scripts.verify_interventions import compare, flip_class, sha256
from scripts.verify_probe_games import original_engine

NEW_PROBE_SHA256 = "cbe404816ddb5711a4563ced365f7727b356aa0770f4290fe6eb99d88239f344"
CENTRAL = {27, 28, 35, 36}
PLAYABLE = [square for square in range(64) if square not in CENTRAL]
SQUARE_TO_TOKEN = {square: token for token, square in enumerate(PLAYABLE, 1)}


def independent_deltas(case, probe):
    """Normalize each edit direction separately, then sum scaled directions."""
    early, late = torch.zeros(512), torch.zeros(512)
    for position, original_colour, action in zip(
        case["selected_positions"], case["original_colors"], case["actions"]
    ):
        row, column = divmod(int(position), 8)
        label = 0 if action == "erase" else flip_class(case["history_length"], int(original_colour))
        vector = probe[:, row, column, label]
        strength = 1.3 if action == "erase" else 2.3
        scaled = strength * (vector / vector.norm())
        if action == "erase":
            early += scaled
        late += scaled
    return {layer: early.clone() if layer < 2 else late.clone() for layer in range(8)}


def audit_manifest(manifest):
    board_type = original_engine(REPO)
    grouped = defaultdict(list)
    for case in manifest["cases"]:
        grouped[case["prefix_id"]].append(case)
    if len(grouped) != 854 or len(manifest["cases"]) != 2562:
        raise AssertionError("Expected 854 prefixes and exactly 2,562 cases")
    if len({case["case_id"] for case in manifest["cases"]}) != 2562:
        raise AssertionError("Case identifiers must be unique")
    seen = set()
    composition = Counter()
    counts = Counter()
    for prefix_id, cases in grouped.items():
        cases.sort(key=lambda case: case["k"])
        if [case["k"] for case in cases] != [2, 4, 8]:
            raise AssertionError(f"Incorrect nested cases for prefix {prefix_id}")
        full = cases[-1]
        history = tuple(full["history"])
        if history in seen or not 8 <= len(history) <= 30:
            raise AssertionError("Prefixes must be unique and length 8..30")
        seen.add(history)
        board = board_type()
        for square in history:
            if int(square) not in board.get_valid_moves():
                raise AssertionError(f"Illegal history move in prefix {prefix_id}")
            counts["history_implicit_passes"] += board.tentative_move(int(square)) == 2
            board.umpire(int(square))
        original = board.state.copy()
        original_flat = original.reshape(-1)
        original_legal = set(board.get_valid_moves())
        for case in cases:
            k = case["k"]
            positions = list(map(int, case["selected_positions"]))
            colours = list(map(int, case["original_colors"]))
            actions = case["actions"]
            if tuple(case["history"]) != history or case["history_length"] != len(history):
                raise AssertionError("Model input history was changed")
            if len(positions) != k or len(set(positions)) != k or len(actions) != k or len(colours) != k:
                raise AssertionError("Edit sets must contain k distinct squares")
            if positions != full["selected_positions"][:k] or actions != full["actions"][:k]:
                raise AssertionError("Edit positions and coins must be nested")
            if colours != full["original_colors"][:k]:
                raise AssertionError("Original colours must be nested")
            if set(positions) & CENTRAL or any(original_flat[position] == 0 for position in positions):
                raise AssertionError("Edit targets must be occupied noncentral squares")
            target = board_type()
            target.state = original.copy()
            target.next_hand_color = board.next_hand_color
            for position, colour, action in zip(positions, colours, actions):
                if colour not in (0, 2) or original_flat[position] != colour - 1:
                    raise AssertionError("Incorrect original edit colour")
                if action not in ("flip", "erase"):
                    raise AssertionError("Only FLIP and ERASE edit coins are allowed")
                target.state.reshape(-1)[position] = -(colour - 1) if action == "flip" else 0
            if target.state.tolist() != case["target_board"]:
                raise AssertionError("Target board differs from simultaneous edits")
            legal = sorted(target.get_valid_moves())
            forced_pass = bool(legal and all(target.tentative_move(square) == 2 for square in legal))
            if legal != case["target_legal_moves"] or len(legal) != case["target_n"]:
                raise AssertionError("Target legal set differs from original engine")
            if bool(case["forced_pass"]) != forced_pass or bool(case["terminal"]) != (not legal):
                raise AssertionError("Incorrect forced-pass or zero-legal-move flag")
            if bool(case["changed_legal_set"]) != (set(legal) != original_legal):
                raise AssertionError("Incorrect changed-legal-set flag")
            flips = sum(action == "flip" for action in actions)
            if flips != case["n_flip"] or k - flips != case["n_erase"]:
                raise AssertionError("Incorrect edit composition counts")
            composition[f"k{k}:flip{flips}:erase{k-flips}"] += 1
            counts["forced_pass_cases"] += forced_pass
            counts["zero_legal_move_cases"] += not legal
            counts["changed_legal_set_cases"] += set(legal) != original_legal
    return grouped, {
        "passed": True, "unique_prefixes": len(grouped), "cases": len(manifest["cases"]),
        "checks": ["one sample per unique history", "k=2/4/8 nested positions and action coins",
                   "occupied noncentral distinct target squares", "unchanged original input histories",
                   "simultaneous target board edits", "original-engine legal sets and turn state",
                   "forced-pass and zero-legal-move flags"],
        **dict(counts), "composition_counts": dict(sorted(composition.items())),
    }


def selected_prefixes(grouped, count):
    """Use actual generated cases, including natural pure coin outcomes."""
    selected = []
    prefix_ids = sorted(grouped)

    def add(prefix):
        if prefix not in selected:
            selected.append(prefix)

    # A naturally all-FLIP or all-ERASE k8 case covers that composition at every k.
    for wanted_flips in (0, 8):
        candidates = [prefix for prefix in prefix_ids if grouped[prefix][-1]["n_flip"] == wanted_flips]
        if candidates:
            add(candidates[0])
        elif wanted_flips == 0:
            # The approved seed may have no all-ERASE k8 outcome; use a natural
            # all-ERASE k4 or k2 outcome from the same generated sample instead.
            for k in (4, 2):
                candidates = [prefix for prefix in prefix_ids
                              if any(case["k"] == k and case["n_flip"] == 0 for case in grouped[prefix])]
                if candidates:
                    add(candidates[0])
                    break
    mixed = [prefix for prefix in prefix_ids if 0 < grouped[prefix][-1]["n_flip"] < 8]
    if mixed:
        add(mixed[0])
    for field in ("forced_pass", "terminal"):
        candidates = [prefix for prefix in prefix_ids if any(case[field] for case in grouped[prefix])]
        if candidates:
            add(candidates[0])
    add(min(prefix_ids, key=lambda prefix: grouped[prefix][0]["history_length"]))
    add(max(prefix_ids, key=lambda prefix: grouped[prefix][0]["history_length"]))
    for parity in (0, 1):
        candidates = [prefix for prefix in prefix_ids if grouped[prefix][0]["history_length"] % 2 == parity]
        if candidates:
            add(candidates[0])
    # More selected prefixes increase same-length compact batches without creating new cases.
    for prefix in prefix_ids:
        if len(selected) >= count:
            break
        add(prefix)
    return selected


def top_moves(logits, number):
    if number == 0:
        return set()
    indices = (logits[1:].topk(number).indices + 1).tolist()
    return {PLAYABLE[index - 1] for index in indices}


def independent_score(predicted, target):
    target = set(target)
    if not target:
        if predicted:
            raise AssertionError("An N=0 case must not select any moves")
        return {"false_positives": None, "false_negatives": None, "errors": None,
                "exact_set": None, "normalized_error": None, "scored": False}
    fp, fn = len(predicted - target), len(target - predicted)
    if len(predicted) != len(target) or fp != fn:
        raise AssertionError("Top-N scores must have equal false positives and negatives")
    return {"false_positives": fp, "false_negatives": fn, "errors": fp + fn,
            "exact_set": fp + fn == 0, "normalized_error": fp / len(target), "scored": True}


def verify(args):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from mech_int.reproduce_mixed_interventions import build_attention_deltas, score_case
    if importlib.metadata.version("transformer-lens") != "1.6.0":
        raise ValueError("Reference requires TransformerLens 1.6.0")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(123)
    manifest = json.loads(args.manifest.read_text())
    grouped, audit = audit_manifest(manifest)
    prefixes = selected_prefixes(grouped, args.prefixes)
    cases = [case for prefix in prefixes for case in grouped[prefix]]
    probe_path = args.probe_directory / "resid_5_linear.pth"
    if sha256(probe_path) != NEW_PROBE_SHA256:
        raise ValueError("Probe differs from the newly trained layer-five checkpoint")
    full_probe = torch.load(probe_path, map_location="cpu", weights_only=True).detach()
    if tuple(full_probe.shape) != (1, 512, 8, 8, 3):
        raise ValueError("Unexpected layer-five probe shape")
    probe = full_probe[0]
    cfg = HookedTransformerConfig(n_layers=8, d_model=512, d_head=64, n_heads=8,
                                 d_mlp=2048, d_vocab=61, n_ctx=59, act_fn="gelu",
                                 normalization_type="LNPre", device="cpu")
    reference = HookedTransformer(cfg)
    reference.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    reference.eval().requires_grad_(False)
    devices = ["cpu", "mps"] if args.device == "both" else [args.device]
    if "mps" in devices and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this execution context")
    compacts = {device: FrozenOthelloGPT(args.checkpoint).to(device).eval() for device in devices}
    report = {
        "passed": True, "reference": "TransformerLens 1.6.0 run_with_hooks, CPU float32",
        "devices": devices, "manifest_sha256": sha256(args.manifest),
        "checkpoint_sha256": sha256(args.checkpoint), "probe_sha256": sha256(probe_path),
        "selected_prefix_ids": prefixes, "selected_case_ids": [case["case_id"] for case in cases],
        "selected_cases": len(cases), "manifest_audit": audit,
        "composition_rule": "Sum individually normalized edit vectors; no total renormalization",
        "flip": {"scale": 2.3, "blocks": list(range(2, 8))},
        "erase": {"scale": 1.3, "blocks": list(range(8))},
        "hook": "final-token hook_attn_out", "comparisons": [],
        "batch_sizes_by_length": {},
    }
    by_length = defaultdict(list)
    for case in cases:
        by_length[case["history_length"]].append(case)
    with torch.inference_mode():
        for length, batch_cases in sorted(by_length.items()):
            report["batch_sizes_by_length"][str(length)] = len(batch_cases)
            tokens = torch.tensor([[SQUARE_TO_TOKEN[int(square)] for square in case["history"]]
                                   for case in batch_cases], dtype=torch.long)
            expected_logits, deltas = [], []
            for index, case in enumerate(batch_cases):
                independent = independent_deltas(case, probe)
                actual_deltas, _ = build_attention_deltas(case, probe)
                for layer in range(8):
                    actual_delta = actual_deltas.get(layer, torch.zeros(512))
                    torch.testing.assert_close(actual_delta.cpu(), independent[layer], atol=1e-6, rtol=1e-5)
                    if case["n_erase"] == 0 and layer < 2 and actual_delta.abs().max().item() != 0:
                        raise AssertionError("FLIP must not change blocks zero and one")
                deltas.append(actual_deltas)
                # Independent sums drive the actual original TL reference hooks.
                def hook_for_layer(layer):
                    def patch(activation, hook):
                        result = activation.clone()
                        result[:, -1, :] += independent[layer]
                        return result
                    return patch
                expected_logits.append(reference.run_with_hooks(
                    tokens[index:index + 1],
                    fwd_hooks=[(f"blocks.{layer}.hook_attn_out", hook_for_layer(layer))
                               for layer in range(8)]))
            expected = torch.cat(expected_logits)
            for device, compact in compacts.items():
                # Recompute sums on the actual evaluation device, including its
                # vector norm reduction and summation order.
                device_probe = probe.to(device)
                device_deltas = [build_attention_deltas(case, device_probe)[0] for case in batch_cases]
                for index, delta in enumerate(device_deltas):
                    for layer in range(8):
                        torch.testing.assert_close(delta[layer].cpu(), deltas[index][layer].cpu(),
                                                   atol=1e-6, rtol=1e-5)
                batched_deltas = {layer: torch.stack([delta[layer] for delta in device_deltas])
                                  for layer in range(8)}
                actual = compact(tokens.to(device), attention_deltas=batched_deltas).cpu()
                comparison = compare(actual, expected, device)
                for index, case in enumerate(batch_cases):
                    n = case["target_n"]
                    expected_top = top_moves(expected[index, -1], n)
                    actual_top = top_moves(actual[index, -1], n)
                    if actual_top != expected_top:
                        raise AssertionError(f"Top-N mismatch for case {case['case_id']}/{device}")
                    score = independent_score(actual_top, case["target_legal_moves"])
                    reference_score = independent_score(expected_top, case["target_legal_moves"])
                    if score != reference_score:
                        raise AssertionError("Reference score differs")
                    runner_score = score_case(sorted(actual_top), case)
                    if any(runner_score[key] != value for key, value in score.items()):
                        raise AssertionError("Runner scoring differs from the independent metric")
                    report["comparisons"].append({
                        "mode": "mixed", "case_id": case["case_id"], "k": case["k"],
                        "length": length, "device": device, "n_flip": case["n_flip"],
                        "n_erase": case["n_erase"], "target_n": n,
                        "forced_pass": case["forced_pass"], "terminal": case["terminal"],
                        "top_n_agreement": True, "score": score, **comparison,
                    })
    report["maximum_absolute_logit_error"] = max(row["max_absolute_error"] for row in report["comparisons"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "output/linear_probe_reproduction/mixed_interventions/cases_manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=REPO / "synthetic_model.pth")
    parser.add_argument("--probe-directory", type=Path, default=REPO / "output/linear_probe_reproduction")
    parser.add_argument("--report", type=Path,
                        default=REPO / "output/linear_probe_reproduction/mixed_interventions/implementation_verification.json")
    parser.add_argument("--prefixes", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "mps", "both"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.prefixes < 1:
        parser.error("--prefixes must be positive")
    report = verify(args)
    print(json.dumps({key: report[key] for key in
                      ("passed", "selected_cases", "devices", "maximum_absolute_logit_error")}, indent=2))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
