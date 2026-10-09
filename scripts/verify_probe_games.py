#!/usr/bin/env python3
"""Replay sampled generated games with the repository's original Othello engine.

Loads only OthelloBoardState via AST, avoiding the engine module's plotting/PGN imports.
Checks raw-file sizes, exact full-game uniqueness across every split, original-engine
move legality (including implicit passes), and parity-based labels after each move.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


def original_engine(repo: Path):
    source = repo / "data" / "othello.py"
    parsed = ast.parse(source.read_text())
    board_class = next(
        node for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == "OthelloBoardState"
    )
    module = ast.Module(body=[board_class], type_ignores=[])
    namespace = {
        "np": np,
        "BLACK": 1,
        "WHITE": -1,
        "EMPTY": 0,
        "eights": [[-1, 0], [-1, 1], [0, 1], [1, 1],
                   [1, 0], [1, -1], [0, -1], [-1, -1]],
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["OthelloBoardState"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(directory: Path, samples_per_split: int, seed: int) -> dict:
    repo = Path(__file__).resolve().parents[1]
    engine = original_engine(repo)
    manifest = json.loads((directory / "dataset_manifest.json").read_text())
    generator = np.random.default_rng(seed)
    seen: set[bytes] = set()
    report = {
        "passed": True,
        "engine": str(repo / "data" / "othello.py"),
        "engine_sha256": file_sha256(repo / "data" / "othello.py"),
        "generator_sha256": file_sha256(repo / "scripts" / "generate_probe_games.cpp"),
        "validation_seed": seed,
        "sampled_games_per_split": samples_per_split,
        "labels": ["empty", "yours", "mine"],
        "checks": ["raw file sizes", "exact full-game uniqueness across all splits",
                   "all games contain every noncentral square exactly once",
                   "all label values in 0..2 and occupied-square counts equal 5+t",
                   "all sampled move legalities against original engine",
                   "all sampled after-move labels against original engine board states",
                   "full board after 60 moves"],
        "splits": {},
    }
    for name in manifest["split_order"]:
        split = manifest["splits"][name]
        count = split["games"]
        moves_path = directory / split["moves_file"]
        labels_path = directory / split["labels_file"]
        assert moves_path.stat().st_size == count * 60, f"Incorrect moves file size: {name}"
        assert labels_path.stat().st_size == count * 59 * 64, f"Incorrect labels file size: {name}"
        if not count:
            report["splits"][name] = {"games": 0, "sampled_games": 0}
            continue
        moves = np.memmap(moves_path, mode="r", dtype=np.uint8, shape=(count, 60))
        labels = np.memmap(labels_path, mode="r", dtype=np.uint8, shape=(count, 59, 64))
        valid_squares = np.delete(np.arange(64, dtype=np.uint8), [27, 28, 35, 36])
        expected_occupied = np.arange(5, 64)
        for offset in range(0, count, 4096):
            chunk_moves = moves[offset:offset + 4096]
            chunk_labels = labels[offset:offset + 4096]
            assert np.all(np.sort(chunk_moves, axis=1) == valid_squares), f"Invalid complete-game move permutation: {name}:{offset}"
            assert chunk_labels.max() <= 2, f"Invalid label class: {name}:{offset}"
            assert np.all(np.count_nonzero(chunk_labels, axis=2) == expected_occupied), f"Incorrect occupied-square count: {name}:{offset}"
            assert np.all(chunk_labels[:, :, [27, 28, 35, 36]] != 0), f"Central square marked empty: {name}:{offset}"
        for sequence in moves:
            identity = sequence.tobytes()
            assert identity not in seen, f"Duplicate full-game sequence in split {name}"
            seen.add(identity)
        sample_indices = sorted(generator.choice(count, min(samples_per_split, count), replace=False).tolist())
        sampled_passes = 0
        sampled_games_with_passes = 0
        for index in sample_indices:
            board = engine()
            game_passes = 0
            for t, value in enumerate(moves[index]):
                cell = int(value)
                assert cell in board.get_valid_moves(), f"Illegal move {name}:{index}:{t}:{cell}"
                if board.tentative_move(cell) == 2:
                    game_passes += 1
                board.umpire(cell)
                if t < 59:
                    state = board.state.reshape(64)
                    expected = np.zeros(64, dtype=np.uint8)
                    expected[state == (1 if t % 2 == 0 else -1)] = 1
                    expected[state == (-1 if t % 2 == 0 else 1)] = 2
                    assert np.array_equal(labels[index, t], expected), f"Incorrect labels {name}:{index}:{t}"
            assert np.all(board.state != 0), f"Game did not fill board {name}:{index}"
            assert not board.get_valid_moves(), f"Game did not terminate {name}:{index}"
            sampled_passes += game_passes
            sampled_games_with_passes += game_passes > 0
        report["splits"][name] = {
            "games": count,
            "sampled_games": len(sample_indices),
            "sampled_game_indices": sample_indices,
            "sampled_passes": sampled_passes,
            "sampled_games_with_passes": sampled_games_with_passes,
            "moves_sha256": file_sha256(moves_path),
            "labels_sha256": file_sha256(labels_path),
        }
    report["distinct_games_total"] = len(seen)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--samples-per-split", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.samples_per_split < 1:
        parser.error("--samples-per-split must be positive")
    report = verify(args.directory, args.samples_per_split, args.seed)
    path = args.report or args.directory / "dataset_verification.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "distinct_games_total": report["distinct_games_total"],
                      "splits": {name: {key: value for key, value in split.items()
                                        if key in ("games", "sampled_games", "sampled_passes", "sampled_games_with_passes")}
                                 for name, split in report["splits"].items()}, "report": str(path)}, indent=2))


if __name__ == "__main__":
    main()
