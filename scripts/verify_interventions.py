#!/usr/bin/env python3
"""Compare compact intervention inference with original TransformerLens 1.6.0.

Uses newly trained layer-five probes and real benchmark prefixes. The reference
adds vectors through run_with_hooks at the released hook_attn_out locations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mech_int.reproduce_linear_probe import FrozenOthelloGPT
from scripts.verify_probe_games import original_engine


class PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Benchmark must contain only plain Python containers")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flip_class(length, original_colour):
    if original_colour not in (0, 2):
        raise ValueError(f"Unexpected original colour: {original_colour}")
    if length % 2 == 0:
        return 2 if original_colour == 0 else 1
    return 1 if original_colour == 0 else 2


def select_cases(data, count):
    # Include both original colours at both parities, plus different lengths.
    selected = set(np.linspace(0, len(data) - 1, min(count, len(data)), dtype=int).tolist())
    for parity in (0, 1):
        for colour in (0, 2):
            candidates = [i for i, case in enumerate(data)
                          if len(case["history"]) % 2 == parity and case["ori_color"] == colour]
            if not candidates:
                raise ValueError(f"Missing parity/colour group {parity}/{colour}")
            selected.add(candidates[0])
    selected.add(min(range(len(data)), key=lambda i: len(data[i]["history"])))
    selected.add(max(range(len(data)), key=lambda i: len(data[i]["history"])))
    return sorted(selected)


def get_top_indices(logits, number, exclude_padding):
    if number == 0:
        return set()
    if exclude_padding:
        return set((logits[1:].topk(number).indices + 1).tolist())
    return set(logits.topk(number).indices.tolist())


def compare(actual, expected, device):
    actual = actual.detach().cpu()
    expected = expected.detach().cpu()
    torch.testing.assert_close(actual, expected,
                               atol=1e-4 if device == "cpu" else 1e-3, rtol=1e-4)
    difference = actual - expected
    return {"max_absolute_error": difference.abs().max().item(),
            "rms_error": difference.square().mean().sqrt().item()}


def verify(args):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    version = importlib.metadata.version("transformer-lens")
    if version != "1.6.0":
        raise ValueError(f"Reference must use TransformerLens 1.6.0, found {version}")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(123)
    with args.benchmark.open("rb") as handle:
        data = PlainUnpickler(handle).load()
    if args.cases < 1:
        raise ValueError("--cases must be positive")
    selected = select_cases(data, args.cases)
    probe_path = args.probe_directory / "resid_5_linear.pth"
    probe = torch.load(probe_path, map_location="cpu", weights_only=True).detach()
    if tuple(probe.shape) != (1, 512, 8, 8, 3):
        raise ValueError(f"Unexpected probe shape: {tuple(probe.shape)}")
    cfg = HookedTransformerConfig(n_layers=8, d_model=512, d_head=64, n_heads=8,
                                 d_mlp=2048, d_vocab=61, n_ctx=59, act_fn="gelu",
                                 normalization_type="LNPre", device="cpu")
    reference = HookedTransformer(cfg)
    reference.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    reference.eval().requires_grad_(False)
    devices = ["cpu", "mps"] if args.device == "both" else [args.device]
    if "mps" in devices and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this execution context")
    compact_models = {device: FrozenOthelloGPT(args.checkpoint).to(device).eval() for device in devices}
    board_class = original_engine(REPO)
    square_to_token = {square: token for token, square in enumerate(
        [square for square in range(64) if square not in (27, 28, 35, 36)], 1)}
    report = {
        "passed": True, "reference": "TransformerLens 1.6.0 run_with_hooks, CPU float32",
        "checkpoint_sha256": sha256(args.checkpoint),
        "probe_path": str(probe_path.resolve()), "probe_sha256": sha256(probe_path),
        "probe_layer_reused_at_every_hook": 5,
        "benchmark_records": len(data), "checked_case_indices": selected,
        "checked_cases": len(selected), "devices": devices,
        "settings": {"flip": {"scale": 2.3, "blocks": list(range(2, 8))},
                     "erase": {"scale": 1.3, "blocks": list(range(8)), "class": 0}},
        "hook": "blocks.<layer>.hook_attn_out, final input token only",
        "comparisons": [],
    }
    with torch.inference_mode():
        for index in selected:
            case = data[index]
            length = len(case["history"])
            row, column = divmod(int(case["pos_int"]), 8)
            original_colour = int(case["ori_color"])
            tokens = torch.tensor([[square_to_token[int(square)] for square in case["history"]]])
            board = board_class()
            passes = 0
            for square in case["history"]:
                passes += board.tentative_move(int(square)) == 2
                board.umpire(int(square))
            if passes:
                raise AssertionError(f"Benchmark case {index} has an implicit pass")
            if board.state[row, column] != original_colour - 1:
                raise AssertionError(f"Incorrect original intervention colour in case {index}")
            unpatched = reference(tokens)
            for device, compact in compact_models.items():
                result = compare(compact(tokens.to(device)), unpatched, device)
                report["comparisons"].append({"case": index, "length": length, "mode": "original",
                                              "device": device, **result})
            for mode, scale, layers, label in (
                ("flip", 2.3, range(2, 8), flip_class(length, original_colour)),
                ("erase", 1.3, range(8), 0),
            ):
                vector = probe[0, :, row, column, label]
                delta = scale * vector / vector.norm()

                def patch(activation, hook):
                    result = activation.clone()
                    result[:, -1, :] += delta
                    return result

                expected = reference.run_with_hooks(
                    tokens, fwd_hooks=[(f"blocks.{layer}.hook_attn_out", patch) for layer in layers])
                target_board = board_class()
                target_board.state = board.state.copy()
                target_board.next_hand_color = board.next_hand_color
                target_board.state[row, column] = -(original_colour - 1) if mode == "flip" else 0
                legal = set(target_board.get_valid_moves())
                count = len(legal)
                expected_top = get_top_indices(expected[0, -1], count, True)
                expected_full = get_top_indices(expected[0, -1], count, False)
                if expected_top != expected_full:
                    raise AssertionError(f"Padding exclusion changes top-N predictions: {index}/{mode}")
                for device, compact in compact_models.items():
                    actual = compact(tokens.to(device),
                                     attention_deltas={layer: delta[None].to(device) for layer in layers})
                    result = compare(actual, expected, device)
                    actual_top = get_top_indices(actual[0, -1].cpu(), count, True)
                    actual_full = get_top_indices(actual[0, -1].cpu(), count, False)
                    if actual_top != expected_top or actual_full != expected_full:
                        raise AssertionError(f"Top-N move mismatch: {index}/{mode}/{device}")
                    legal_tokens = {square_to_token[square] for square in legal}
                    error = len(actual_top - legal_tokens) + len(legal_tokens - actual_top)
                    if error % 2:
                        raise AssertionError("Equal-size move-set error must be even")
                    report["comparisons"].append({
                        "case": index, "length": length, "mode": mode, "device": device,
                        "original_colour": original_colour, "probe_class": label,
                        "legal_moves": count, "fp_plus_fn": error,
                        "top_n_agreement": True, "padding_exclusion_changes_top_n": False,
                        **result,
                    })
    report["maximum_absolute_logit_error"] = max(row["max_absolute_error"] for row in report["comparisons"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO / "synthetic_model.pth")
    parser.add_argument("--benchmark", type=Path, default=REPO / "data/intervention_benchmark.pkl")
    parser.add_argument("--probe-directory", type=Path, default=REPO / "output/linear_probe_reproduction")
    parser.add_argument("--report", type=Path,
                        default=REPO / "output/linear_probe_reproduction/interventions/implementation_verification.json")
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "mps", "both"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    report = verify(args)
    print(json.dumps({key: report[key] for key in (
        "passed", "benchmark_records", "checked_cases", "devices", "maximum_absolute_logit_error")}, indent=2))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
