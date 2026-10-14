#!/usr/bin/env python3
"""Check editor targets, pass semantics, padding, masks and fixed split identities."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mech_int.editor_data import (
    CENTRAL_SQUARES, EditorExample, GameSplits, PLAYABLE_SQUARES, SQUARE_TO_TOKEN,
    TOKEN_TO_SQUARE, collate,
)
from scripts.verify_probe_games import original_engine


def assert_exact_target(example, board_type):
    board = board_type()
    passes = 0
    for move in example.history:
        assert move in board.get_valid_moves()
        passes += board.tentative_move(move) == 2
        board.umpire(move)
    assert int(board.next_hand_color) == example.nominal_next_hand_color
    assert passes == example.history_passes
    flat = board.state.reshape(64)
    assert int(flat[example.square]) == example.original_color
    assert example.square not in CENTRAL_SQUARES
    original_legal = tuple(sorted(int(move) for move in board.get_valid_moves()))
    assert example.original_legal_moves == original_legal
    flat[example.square] = -flat[example.square] if example.action == "flip" else 0
    before = int(board.next_hand_color)
    # Enumerate regular and fallback moves separately, independently of the
    # sampler's get_valid_moves call, to test implicit rather than explicit pass.
    regular = tuple(move for move in range(64) if board.tentative_move(move) == 1)
    fallback = tuple(move for move in range(64) if board.tentative_move(move) == 2)
    assert example.legal_moves == (regular or fallback)
    assert example.changed_legal_set == (example.legal_moves != original_legal)
    assert example.forced_pass == bool(not regular and fallback)
    assert example.effective_next_hand_color == (-before if example.forced_pass else before)
    assert int(board.next_hand_color) == before


def find_edge_cases(splits):
    forced = {}
    terminal = None
    for game_index in range(min(64, splits.counts["val"])):
        for prefix_length in range(59, 44, -1):
            board, _, _ = splits.replay("val", game_index, prefix_length)
            occupied = PLAYABLE_SQUARES[board.state.reshape(64)[PLAYABLE_SQUARES] != 0]
            for action in ("flip", "erase"):
                for square in occupied:
                    example = splits.example("val", game_index, prefix_length, int(square), action)
                    if example is None:
                        terminal = (game_index, prefix_length, int(square), action)
                    elif example.forced_pass and action not in forced:
                        forced[action] = example
                    if len(forced) == 2 and terminal is not None:
                        return forced, terminal
    raise AssertionError("Could not find forced-pass and terminal fixtures in the held-out games")


def verify(data: Path, seed: int = 42, samples: int = 64,
           changed_target_fraction: float | None = 0.9):
    start = time.monotonic()
    splits = GameSplits(data, changed_target_fraction)
    board_type = original_engine(REPO)
    rng = np.random.default_rng(seed)
    training = splits.sample_training_examples(samples, rng)
    state_before = json.dumps(rng.bit_generator.state, sort_keys=True)
    with tempfile.TemporaryDirectory(prefix="othello-editor-data-") as temporary:
        directory = Path(temporary)
        validation = splits.fixed_examples("val", seed, directory / "val.json")
        test = splits.fixed_examples("test", seed, directory / "test.json")
        assert validation == splits.fixed_examples("val", seed, directory / "val.json")
        assert test == splits.fixed_examples("test", seed, directory / "test.json")
        assert validation == splits.fixed_examples("val", seed)
        assert test == splits.fixed_examples("test", seed)
        # Old fixed-case JSON has no original-legal metadata. It remains a
        # valid cache and gains simulator metadata only in memory on reload.
        for name, cases in (("val", validation), ("test", test)):
            cache_path = directory / f"{name}.json"
            legacy_payload = json.loads(cache_path.read_text())
            for saved_case in legacy_payload["examples"]:
                saved_case.pop("original_legal_moves")
            assert EditorExample.from_dict(legacy_payload["examples"][0]).changed_legal_set is None
            cache_path.write_text(json.dumps(legacy_payload))
            reloaded = splits.fixed_examples(name, seed, cache_path)
            assert reloaded == cases
            assert all(case.original_legal_moves is not None for case in reloaded)
            assert cache_path.read_text() == json.dumps(legacy_payload)
        assert json.dumps(rng.bit_generator.state, sort_keys=True) == state_before
        # Cache configuration is part of its identity, so a different seed
        # cannot silently reuse already-written evaluation cases.
        try:
            splits.fixed_examples("val", seed + 1, directory / "val.json")
        except ValueError:
            pass
        else:
            raise AssertionError("A fixed cache accepted the wrong seed")

    # Independently preserve the original seeded proposal order for fixed
    # evaluation: game is fixed, then prefix, occupied square, then action.
    def legacy_draw(split, game_index, legacy_rng):
        length = int(legacy_rng.integers(1, 60))
        board, history, passes = splits.replay(split, game_index, length)
        flat = board.state.reshape(64)
        occupied = PLAYABLE_SQUARES[flat[PLAYABLE_SQUARES] != 0]
        square = int(legacy_rng.choice(occupied))
        action = "flip" if int(legacy_rng.integers(0, 2)) == 1 else "erase"
        return splits._edit_board(board, split, game_index, history, passes, square, action)

    for name, cases in (("val", validation), ("test", test)):
        legacy_rng = np.random.default_rng(np.random.SeedSequence([seed, 1 if name == "val" else 2]))
        for game_index, case in enumerate(cases):
            while (original_case := legacy_draw(name, game_index, legacy_rng)) is None:
                pass
            assert original_case == case
    for fraction in (None, 0.0, 1.0):
        other_sampler = GameSplits(data, fraction)
        assert other_sampler.fixed_examples("val", seed) == validation

    mixture_splits = GameSplits(data, 0.9)
    mixture_samples = max(1024, samples)
    mixture = mixture_splits.sample_training_examples(mixture_samples, np.random.default_rng(seed))
    changed_fraction = sum(case.changed_legal_set for case in mixture) / mixture_samples
    flip_fraction = sum(case.action == "flip" for case in mixture) / mixture_samples
    assert abs(changed_fraction - 0.9) <= 4 * np.sqrt(0.9 * 0.1 / mixture_samples)
    assert abs(flip_fraction - 0.5) <= 4 * np.sqrt(0.5 * 0.5 / mixture_samples)
    for action in ("flip", "erase"):
        action_cases = [case for case in mixture if case.action == action]
        action_changed_fraction = sum(case.changed_legal_set for case in action_cases) / len(action_cases)
        assert abs(action_changed_fraction - 0.9) <= 4 * np.sqrt(0.9 * 0.1 / len(action_cases))

    # Force one rejected candidate per selected stratum. Retrying must retain
    # the selected action/stratum instead of choosing either one again.
    fixtures = {(action, changed): next(case for case in mixture
                                      if case.action == action and case.changed_legal_set == changed)
                for action in ("flip", "erase") for changed in (False, True)}

    class ScriptedRng:
        selections = ((1, 0.0), (0, 0.95), (1, 0.95), (0, 0.0))

        def __init__(self):
            self.index = -1
            self.attempt = 0

        def integers(self, low, high):
            if high == 2:
                self.index += 1
                self.attempt = 0
                return self.selections[self.index][0]
            return 0

        def random(self):
            return self.selections[self.index][1]

    class ForcedStratumRejection(GameSplits):
        def _draw(self, split, game_index, rng, action=None):
            selected_action, selected_probability = rng.selections[rng.index]
            assert action == ("flip" if selected_action else "erase")
            desired = selected_probability < self.changed_target_fraction
            rng.attempt += 1
            return fixtures[(action, not desired if rng.attempt == 1 else desired)]

    scripted = ForcedStratumRejection(data, 0.9)
    scripted_cases = scripted.sample_training_examples(4, ScriptedRng())
    assert [(case.action, case.changed_legal_set) for case in scripted_cases] == [
        ("flip", True), ("erase", False), ("flip", False), ("erase", True)]
    assert scripted.stats["training_attempts"] == 8
    assert scripted.stats["training_stratum_rejections"] == 4
    assert all(count == 1 for counts in scripted.stats["training_rejections"].values()
               for count in counts.values())
    endpoint_stats = {}
    for fraction in (0.0, 1.0):
        endpoint = GameSplits(data, fraction)
        endpoint_cases = endpoint.sample_training_examples(64, np.random.default_rng(seed))
        assert all(case.changed_legal_set == bool(fraction) for case in endpoint_cases)
        assert {case.action for case in endpoint_cases} == {"flip", "erase"}
        for example in endpoint_cases:
            assert_exact_target(example, board_type)
        endpoint_stats[str(fraction)] = endpoint.stats
    natural = GameSplits(data, None)
    natural_rng = np.random.default_rng(seed)
    natural_cases = natural.sample_training_examples(samples, natural_rng)
    legacy_rng = np.random.default_rng(seed)
    expected_natural = []
    while len(expected_natural) < samples:
        game_index = int(legacy_rng.integers(0, splits.counts["train"]))
        case = legacy_draw("train", game_index, legacy_rng)
        if case is not None:
            expected_natural.append(case)
    assert natural_cases == expected_natural
    assert splits.sample_training_examples(samples, np.random.default_rng(seed + 1)) != training
    for invalid_fraction in (-0.1, 1.1, float("nan")):
        try:
            GameSplits(data, invalid_fraction)
        except ValueError:
            pass
        else:
            raise AssertionError("An invalid changed-target fraction was accepted")

    # One fixed example corresponds to every game in its actual held-out split.
    assert [case.game_index for case in validation] == list(range(splits.counts["val"]))
    assert [case.game_index for case in test] == list(range(splits.counts["test"]))
    assert all(case.split == "train" for case in training)
    held_out = {bytes(game) for name in ("val", "test") for game in splits.moves[name]}
    assert len(held_out) == splits.counts["val"] + splits.counts["test"]
    assert not any(bytes(game) in held_out for game in splits.moves["train"])

    all_cases = training + validation + test
    for example in all_cases:
        assert_exact_target(example, board_type)
        assert example.history == tuple(int(move) for move in
                                       splits.moves[example.split][example.game_index, :example.prefix_length])
    forced, terminal = find_edge_cases(splits)
    for example in forced.values():
        assert_exact_target(example, board_type)
    assert splits.example("val", *terminal) is None

    # The seed42 fixed cases need not randomly encounter a terminal edit. Feed
    # an actual terminal simulator fixture into each sampling path once to
    # check rejection without depending on a rare random event.
    class TerminalFirstDraw(GameSplits):
        def __init__(self, data, fraction):
            super().__init__(data, fraction)
            self.first_draw = True

        def _draw(self, split, game_index, rng, action=None):
            if self.first_draw:
                self.first_draw = False
                return self.example("val", *terminal)
            return super()._draw(split, game_index, rng, action=action)

    rejecting_train = TerminalFirstDraw(data, changed_target_fraction)
    accepted = rejecting_train.sample_training_examples(1, np.random.default_rng(seed))
    assert len(accepted) == 1 and accepted[0].split == "train"
    assert rejecting_train.stats["training_terminal_skips"] == 1
    rejecting_fixed = TerminalFirstDraw(data, changed_target_fraction)
    accepted_fixed = rejecting_fixed.fixed_examples("val", seed)
    assert len(accepted_fixed) == splits.counts["val"]
    assert rejecting_fixed.stats["fixed_terminal_skips"]["val"] >= 1

    # Exact simulator sides diverge from simple move-index parity after an odd
    # number of implicit passes; require such a case in the fixed evaluation.
    after_pass = [case for case in validation + test if case.history_passes % 2]
    assert after_pass
    assert all(case.nominal_next_hand_color != (-1) ** case.prefix_length for case in after_pass)

    batch_examples = training + list(forced.values())
    batch = collate(batch_examples, "cpu")
    assert batch.tokens.device.type == "cpu"
    assert batch.tokens.dtype == torch.int64 and batch.lengths.dtype == torch.int64
    assert batch.rules.dtype == torch.float32 and batch.legal_mask.dtype == torch.bool
    assert tuple(batch.rules.shape) == (len(batch_examples), 128)
    assert tuple(batch.legal_mask.shape) == (len(batch_examples), 61)
    assert not batch.legal_mask[:, 0].any()
    assert torch.all(batch.rules.sum(-1) == 1)
    for row, example in enumerate(batch_examples):
        length = example.prefix_length
        assert batch.lengths[row].item() == length
        assert batch.tokens[row, :length].tolist() == SQUARE_TO_TOKEN[list(example.history)].tolist()
        assert torch.all(batch.tokens[row, length:] == 0)
        assert batch.rules[row].nonzero().flatten().tolist() == [
            example.square + (64 if example.action == "erase" else 0)]
        selected_tokens = batch.legal_mask[row].nonzero().flatten().numpy()
        assert tuple(TOKEN_TO_SQUARE[selected_tokens]) == example.legal_moves
    assert SQUARE_TO_TOKEN[0] == 1 and SQUARE_TO_TOKEN[63] == 60
    assert np.all(SQUARE_TO_TOKEN[list(CENTRAL_SQUARES)] == 0)
    assert np.array_equal(TOKEN_TO_SQUARE[SQUARE_TO_TOKEN[PLAYABLE_SQUARES]], PLAYABLE_SQUARES)

    rng_again = np.random.default_rng(seed)
    assert training == splits.sample_training_examples(samples, rng_again)
    report = {
        "passed": True,
        "data": str(data.resolve()), "seed": seed,
        "seconds": time.monotonic() - start,
        "training_examples_verified": samples,
        "changed_target_fraction": changed_target_fraction,
        "mixture_distribution_samples": mixture_samples,
        "mixture_changed_fraction": changed_fraction,
        "mixture_flip_fraction": flip_fraction,
        "mixture_sampling_stats": mixture_splits.stats,
        "endpoint_sampling_stats": endpoint_stats,
        "fixed_validation_examples_verified": len(validation),
        "fixed_test_examples_verified": len(test),
        "actions": dict(Counter(case.action for case in all_cases)),
        "fixed_forced_passes": sum(case.forced_pass for case in validation + test),
        "fixed_histories_with_odd_implicit_passes": len(after_pass),
        "terminal_skips": splits.stats,
        "terminal_rejection_fixture_training_skips": rejecting_train.stats["training_terminal_skips"],
        "terminal_rejection_fixture_validation_skips": rejecting_fixed.stats["fixed_terminal_skips"]["val"],
        "forced_pass_fixtures": {action: case.to_dict() for action, case in forced.items()},
        "terminal_fixture": {"split": "val", "game_index": terminal[0],
                             "prefix_length": terminal[1], "square": terminal[2],
                             "action": terminal[3]},
        "checks": [
            "train games are disjoint from both held-out game splits",
            "one deterministic nonterminal validation/test case per game",
            "cache reload and fresh generation agree; wrong seed rejected",
            "legacy evaluation caches load and hydrate original legality without rewriting",
            "fixed cases exactly match the original seeded proposal order",
            "fixed evaluation is unchanged by the training sampler fraction",
            "fixed evaluation consumes no training RNG state",
            "exact edited simulator legal targets for both actions",
            "nominal turn preserved; implicit opponent moves after forced pass",
            "terminal edits skipped; actual terminal fixture excluded",
            "post-pass simulator side rather than prefix parity",
            "original history vocabulary mapping and right padding",
            "128-square rule encoding and full61 legal masks with token0 illegal",
            "training sampler is reproducible with the same seed",
            "fresh training candidates are drawn with a different seed",
            "90% changed-set targets with equiprobable actions including within each action",
            "candidate rejection preserves the initially chosen action and changed stratum",
            "zero and one changed-target endpoints produce only their requested stratum",
            "natural training sampler exactly preserves its original seeded cases",
        ],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data" / "probe_reproduction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--changed-target-fraction", type=float, default=0.9)
    parser.add_argument("--natural-training-sampling", action="store_true",
                        help="Verify the original unconstrained training sampler")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if not 0 <= args.changed_target_fraction <= 1:
        parser.error("--changed-target-fraction must be in [0, 1]")
    fraction = None if args.natural_training_sampling else args.changed_target_fraction
    report = verify(args.data, args.seed, args.samples, fraction)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("forced_pass_fixtures", "checks")}, indent=2))


if __name__ == "__main__":
    main()
