#!/usr/bin/env python3
"""Verify one-time intervention decoding against full TransformerLens histories.

Real benchmark prefixes and newly trained layer-five directions are checked at
h=0 and on legal edited-board continuations. Reference hooks always remain at
the original prefix position; cached decoding receives no subsequent steering.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mech_int.reproduce_interventions import (
    ERASE_BLOCKS, ERASE_SCALE, FLIP_BLOCKS, FLIP_SCALE, PROBE_LAYER,
    SQUARE_TO_TOKEN, TOKEN_TO_SQUARE, load_cases, sha256,
)
from mech_int.reproduce_linear_probe import FrozenOthelloGPT, write_json
from scripts.verify_interventions import compare, select_cases
from scripts.verify_probe_games import original_engine

CENTRAL = {27, 28, 35, 36}


def edited_board(case, intervention, board_type):
    board = board_type()
    for square in case["history"]:
        board.umpire(square)
    square = case["position"]
    original = board.state.reshape(-1)[square]
    board.state.reshape(-1)[square] = -original if intervention == "flip" else 0
    return board


def attention_deltas(case, intervention, probe):
    row, column = divmod(case["position"], 8)
    label = case["flip_probe_class"] if intervention == "flip" else 0
    vector = probe[:, row, column, label]
    if not torch.isfinite(vector).all() or vector.norm() == 0:
        raise ValueError("Intervention direction must be finite and nonzero")
    strength, layers = ((FLIP_SCALE, FLIP_BLOCKS) if intervention == "flip"
                        else (ERASE_SCALE, ERASE_BLOCKS))
    delta = strength * vector / vector.norm()
    return {layer: delta[None] for layer in layers}


def reference_logits(reference, tokens, deltas=None, position=None):
    if deltas is None:
        return reference(tokens)

    def hook_for(delta):
        def patch(activation, hook):
            result = activation.clone()
            result[:, position, :] += delta
            return result
        return patch

    return reference.run_with_hooks(tokens, fwd_hooks=[
        (f"blocks.{layer}.hook_attn_out", hook_for(delta))
        for layer, delta in deltas.items()
    ])


def top_moves(logits, count):
    if count == 0:
        return set()
    tokens = (logits[1:].topk(count).indices + 1).detach().cpu().numpy()
    return set(TOKEN_TO_SQUARE[tokens].tolist())


def check_logits(report, actual, expected, device, *, counts=None, **context):
    difference = compare(actual, expected, device)
    if counts is not None:
        for row, count in enumerate(counts):
            if top_moves(actual[row, -1], count) != top_moves(expected[row, -1], count):
                raise AssertionError(f"Top-N disagreement: {context}, row {row}")
        context["top_n_agreement"] = True
    report["comparisons"].append({"device": device, **context, **difference})


def assert_independent(first, second):
    first_storage = {tensor.untyped_storage().data_ptr()
                     for pair in first.layers for tensor in pair}
    second_storage = {tensor.untyped_storage().data_ptr()
                      for pair in second.layers for tensor in pair}
    if first_storage & second_storage:
        raise AssertionError("Independent caches share underlying tensor storage")


def trajectory(case, intervention, board_type, horizon, first_move=None):
    board = edited_board(case, intervention, board_type)
    rng = np.random.default_rng(np.random.SeedSequence([
        42, case["case_id"], int(intervention == "erase")]))
    moves, legal_sets, passes = [], [sorted(board.get_valid_moves())], []
    for step in range(min(horizon, 59 - case["history_length"])):
        legal = legal_sets[-1]
        if not legal:
            break
        move = first_move if step == 0 and first_move is not None else int(rng.choice(legal))
        if move not in legal or move in CENTRAL:
            raise AssertionError("Continuation must be a legal playable-square move")
        passes.append(board.tentative_move(move) == 2)
        board.umpire(move)
        moves.append(move)
        legal_sets.append(sorted(board.get_valid_moves()))
    return moves, legal_sets, passes


def verify_case(case, intervention, probe, board_type, reference, models, horizon,
                report, *, first_move=None, mode="benchmark"):
    moves, legal_sets, passes = trajectory(
        case, intervention, board_type, horizon, first_move)
    prefix = case["history_length"]
    deltas = attention_deltas(case, intervention, probe)
    caches = {}
    for step, legal in enumerate(legal_sets):
        history = case["history"] + moves[:step]
        tokens = torch.tensor(SQUARE_TO_TOKEN[np.array([history])])
        clean_reference = reference_logits(reference, tokens)
        patched_reference = reference_logits(reference, tokens, deltas, prefix - 1)
        for device, model in models.items():
            device_tokens = tokens.to(device)
            device_deltas = {layer: delta.to(device) for layer, delta in deltas.items()}
            if step == 0:
                clean_logits, clean_cache = model.forward_cached(device_tokens)
                patched_logits, patched_cache = model.forward_cached(
                    device_tokens, attention_deltas=device_deltas)
                ordinary = model(device_tokens, attention_deltas=device_deltas)
                check_logits(report, patched_logits, ordinary, device,
                             mode=f"{mode}_h0_existing_runner", case=case["case_id"],
                             intervention=intervention, horizon=0, counts=[len(legal)])
                assert_independent(clean_cache, patched_cache)
            else:
                clean_cache, patched_cache = caches[device]
                new_token = device_tokens[:, -1:]
                # Steering is deliberately absent after prefix ingestion.
                clean_logits, clean_cache = model.forward_cached(new_token, clean_cache)
                patched_logits, patched_cache = model.forward_cached(new_token, patched_cache)
            if clean_cache.length != len(history) or patched_cache.length != len(history):
                raise AssertionError("KV cache length does not equal input history length")
            caches[device] = clean_cache, patched_cache
            context = {"case": case["case_id"], "intervention": intervention,
                       "horizon": step, "input_length": len(history), "counts": [len(legal)]}
            check_logits(report, clean_logits[:, -1:], clean_reference[:, -1:], device,
                         mode=f"{mode}_clean_cached_vs_transformerlens", **context)
            check_logits(report, patched_logits[:, -1:], patched_reference[:, -1:], device,
                         mode=f"{mode}_patched_cached_vs_transformerlens", **context)
            full_clean = model(device_tokens)
            full_patched = model(device_tokens, attention_deltas=device_deltas,
                                 attention_position=prefix - 1)
            check_logits(report, clean_logits[:, -1:], full_clean[:, -1:], device,
                         mode=f"{mode}_clean_cached_vs_full", **context)
            check_logits(report, patched_logits[:, -1:], full_patched[:, -1:], device,
                         mode=f"{mode}_patched_cached_vs_fixed_position_full", **context)
    report["trajectories"].append({"case": case["case_id"], "intervention": intervention,
                                   "mode": mode, "steps": len(moves), "moves": moves,
                                   "implicit_passes": sum(passes),
                                   "terminal": not legal_sets[-1]})


def verify_batches(cases, probe, board_type, models, report):
    groups = defaultdict(list)
    for case in cases:
        if case["history_length"] <= 54:
            groups[case["history_length"]].append(case)
    group = next((group[:3] for length, group in sorted(groups.items()) if len(group) >= 3), None)
    if group is None:
        raise ValueError("Batched verification needs three same-length benchmark cases")
    prefix = group[0]["history_length"]
    for intervention in ("flip", "erase"):
        if intervention == "erase" and any(case["position"] in CENTRAL for case in group):
            continue
        continuations = [trajectory(case, intervention, board_type, 5)[0] for case in group]
        if any(len(moves) < 5 for moves in continuations):
            raise AssertionError("Selected batching checks need five legal continuation moves")
        token_array = SQUARE_TO_TOKEN[np.array([case["history"] for case in group])]
        continuation_array = SQUARE_TO_TOKEN[np.array(continuations)]
        individual = [attention_deltas(case, intervention, probe) for case in group]
        batch_deltas = {layer: torch.cat([deltas[layer] for deltas in individual])
                        for layer in individual[0]}
        for device, model in models.items():
            tokens = torch.tensor(token_array, device=device)
            continuation = torch.tensor(continuation_array, device=device)
            deltas = {layer: value.to(device) for layer, value in batch_deltas.items()}
            _, clean_cache = model.forward_cached(tokens)
            _, patched_cache = model.forward_cached(tokens, attention_deltas=deltas)
            assert_independent(clean_cache, patched_cache)
            prefix_snapshot = tuple((key.clone(), value.clone()) for key, value in patched_cache.layers)
            full_history = tokens
            for start, end in ((0, 2), (2, 5)):
                chunk = continuation[:, start:end]
                clean_logits, clean_cache = model.forward_cached(chunk, clean_cache)
                patched_logits, patched_cache = model.forward_cached(chunk, patched_cache)
                full_history = torch.cat((full_history, chunk), dim=1)
                expected_clean = model(full_history)[:, -chunk.shape[1]:]
                expected_patched = model(full_history, attention_deltas=deltas,
                                         attention_position=prefix - 1)[:, -chunk.shape[1]:]
                for method, actual, expected in (("clean", clean_logits, expected_clean),
                                                  ("patched", patched_logits, expected_patched)):
                    check_logits(report, actual, expected, device,
                                 mode=f"batch_multitoken_{method}", intervention=intervention,
                                 chunk_length=chunk.shape[1], input_length=full_history.shape[1])
            # Reorder and remove rows as trajectories stop, retaining their KV histories.
            indices = torch.tensor([2, 0], device=device)
            filtered = patched_cache.index_select(indices)
            assert_independent(filtered, patched_cache)
            for actual_pair, expected_pair in zip(filtered.layers, patched_cache.layers):
                for actual, expected in zip(actual_pair, expected_pair):
                    torch.testing.assert_close(actual, expected.index_select(0, indices), rtol=0, atol=0)
            extra = torch.tensor([[int(continuation_array[2, -1])],
                                  [int(continuation_array[0, -1])]], device=device)
            logits, result_cache = model.forward_cached(extra, filtered)
            filtered_history = torch.cat((full_history.index_select(0, indices), extra), dim=1)
            filtered_deltas = {layer: value.index_select(0, indices) for layer, value in deltas.items()}
            expected = model(filtered_history, attention_deltas=filtered_deltas,
                             attention_position=prefix - 1)[:, -1:]
            check_logits(report, logits, expected, device,
                         mode="batch_filter_reorder", intervention=intervention,
                         input_length=filtered_history.shape[1])
            if result_cache.length != prefix + 6:
                raise AssertionError("Filtered cache did not retain its prefix length")
            # Appending must not change the original prefix tensors.
            for pair, saved_pair in zip(patched_cache.layers, prefix_snapshot):
                for tensor, saved in zip(pair, saved_pair):
                    torch.testing.assert_close(tensor[:, :, :prefix], saved, rtol=0, atol=0)
    report["batch_checks"] = {"passed": True, "case_ids": [case["case_id"] for case in group],
                              "prefix_length": prefix, "chunk_lengths": [2, 3],
                              "filter_indices": [2, 0], "independent_storage": True}


def verify_boundaries(models, report):
    for device, model in models.items():
        generator = torch.Generator().manual_seed(42)
        tokens = torch.randint(1, 61, (2, 59), generator=generator).to(device)
        _, cache = model.forward_cached(tokens[:, :58])
        actual, cache = model.forward_cached(tokens[:, 58:], cache)
        check_logits(report, actual, model(tokens)[:, -1:], device,
                     mode="context_boundary_59_tokens", input_length=59)
        for action in (lambda: model.forward_cached(tokens[:, :1], cache),
                       lambda: model(torch.cat((tokens, tokens[:, :1]), dim=1))):
            try:
                action()
            except ValueError:
                pass
            else:
                raise AssertionError("A 60-token input must reject context overflow")
    report["context_boundary"] = {"passed": True, "accepted_length": 59,
                                   "rejected_length": 60}


def verify_simulator(board_type):
    board = board_type()
    board.state = np.zeros((8, 8))
    board.state[0, 0], board.state[0, 1] = 1, -1
    board.next_hand_color = -1
    if board.get_valid_moves() != [2] or board.tentative_move(2) != 2:
        raise AssertionError("Synthetic board must force an implicit pass")
    board.umpire(2)
    if board.next_hand_color != -1 or board.state[0, :3].tolist() != [1, 1, 1]:
        raise AssertionError("Implicit pass must capture and preserve the correct next side")
    if board.get_valid_moves():
        raise AssertionError("Both-sides-no-move board must be terminal")
    return {"passed": True, "implicit_pass_without_token": True,
            "captured_disks": 1, "terminal_after_move": True}


@torch.inference_mode()
def verify(args):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    if importlib.metadata.version("transformer-lens") != "1.6.0":
        raise ValueError("Reference requires TransformerLens 1.6.0")
    if args.probe_directory.resolve() == (REPO / "mech_int/probes/linear").resolve():
        raise ValueError("Use the newly trained layer-five probe")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = load_cases(args.benchmark)
    selected = select_cases(cases, args.cases)
    probe_path = args.probe_directory / f"resid_{PROBE_LAYER}_linear.pth"
    full_probe = torch.load(probe_path, map_location="cpu", weights_only=True)
    if tuple(full_probe.shape) != (1, 512, 8, 8, 3) or not torch.isfinite(full_probe).all():
        raise ValueError("Unexpected or non-finite layer-five probe")
    probe = full_probe[0]
    devices = ["cpu", "mps"] if args.device == "both" else [args.device]
    if "mps" in devices and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this execution context")
    reference = HookedTransformer(HookedTransformerConfig(
        n_layers=8, d_model=512, d_head=64, n_heads=8, d_mlp=2048,
        d_vocab=61, n_ctx=59, act_fn="gelu", normalization_type="LNPre", device="cpu"))
    reference.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    reference.eval().requires_grad_(False)
    models = {device: FrozenOthelloGPT(args.checkpoint).to(device).eval() for device in devices}
    board_type = original_engine(REPO)
    report = {"passed": True, "reference": "TransformerLens 1.6.0, CPU float32",
              "checkpoint_sha256": sha256(args.checkpoint), "probe_sha256": sha256(probe_path),
              "benchmark_sha256": sha256(args.benchmark), "benchmark_records": len(cases),
              "checked_case_indices": selected, "checked_cases": len(selected),
              "devices": devices, "horizon": args.horizon,
              "hook": "attention output, original prefix absolute position only",
              "cache": "autoregressive per-block K/V; no continuation steering",
              "tolerances": {"cpu_atol": 1e-4, "mps_atol": 1e-3, "rtol": 1e-4},
              "runner_sha256": sha256(Path(__file__)),
              "model_implementation_sha256": sha256(REPO / "mech_int/reproduce_linear_probe.py"),
              "comparisons": [], "trajectories": []}
    started = time.monotonic()
    for index in selected:
        for intervention in ("flip", "erase"):
            if intervention == "erase" and cases[index]["position"] in CENTRAL:
                continue
            verify_case(cases[index], intervention, probe, board_type, reference,
                        models, args.horizon, report)
        print(json.dumps({"verified_case": index, "elapsed_seconds": time.monotonic() - started}), flush=True)
    replay_case = next((case for case in cases if case["position"] not in CENTRAL and
                       case["position"] in edited_board(case, "erase", board_type).get_valid_moves()), None)
    if replay_case is None:
        raise AssertionError("Expected a benchmark ERASE case with immediate square reoccupation")
    if replay_case["position"] not in replay_case["history"]:
        raise AssertionError("Noncentral erased square must already occur in the token prefix")
    verify_case(replay_case, "erase", probe, board_type, reference, models, 1,
                report, first_move=replay_case["position"], mode="erase_repeated_token")
    report["repeated_erase_token"] = {"passed": True, "case_id": replay_case["case_id"],
                                       "square": replay_case["position"]}
    verify_batches(cases, probe, board_type, models, report)
    verify_boundaries(models, report)
    report["simulator_edge_cases"] = verify_simulator(board_type)
    report["maximum_absolute_logit_error"] = max(row["max_absolute_error"] for row in report["comparisons"])
    report["elapsed_seconds"] = time.monotonic() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO / "synthetic_model.pth")
    parser.add_argument("--benchmark", type=Path, default=REPO / "data/intervention_benchmark.pkl")
    parser.add_argument("--probe-directory", type=Path, default=REPO / "output/linear_probe_reproduction")
    parser.add_argument("--report", type=Path, default=REPO / "output/linear_probe_reproduction/persistent_interventions/implementation_verification.json")
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "mps", "both"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.cases < 1 or args.horizon < 0 or args.cpu_threads < 1:
        parser.error("--cases/--cpu-threads must be positive and --horizon nonnegative")
    report = verify(args)
    print(json.dumps({key: report[key] for key in (
        "passed", "checked_cases", "devices", "horizon", "maximum_absolute_logit_error", "elapsed_seconds")}, indent=2))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
