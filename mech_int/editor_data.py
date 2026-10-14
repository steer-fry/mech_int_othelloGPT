"""Counterfactual single-square targets for the editors specified in TRAIN.md.

Game identities come exclusively from the existing probe-reproduction splits.
The simulator, rather than position parity or probe labels, determines the turn.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from scripts.verify_probe_games import original_engine


REPO = Path(__file__).resolve().parents[1]
CENTRAL_SQUARES = (27, 28, 35, 36)
PLAYABLE_SQUARES = np.array([s for s in range(64) if s not in CENTRAL_SQUARES])
SQUARE_TO_TOKEN = np.zeros(64, dtype=np.int64)
SQUARE_TO_TOKEN[PLAYABLE_SQUARES] = np.arange(1, 61)
TOKEN_TO_SQUARE = np.concatenate(([-1], PLAYABLE_SQUARES))
SPLIT_SEED_OFFSETS = {"val": 1, "test": 2}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class EditorExample:
    split: str
    game_index: int
    history: tuple[int, ...]
    prefix_length: int
    square: int
    action: str
    original_color: int
    legal_moves: tuple[int, ...]
    nominal_next_hand_color: int
    effective_next_hand_color: int
    forced_pass: bool
    history_passes: int
    # Evaluation caches from the original run omit this metadata. It does not
    # define the example's identity, so those cases still compare equal.
    original_legal_moves: tuple[int, ...] | None = field(default=None, compare=False)

    @property
    def changed_legal_set(self) -> bool | None:
        if self.original_legal_moves is None:
            return None
        return self.legal_moves != self.original_legal_moves

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, metadata: dict) -> EditorExample:
        fields = dict(metadata)
        fields["history"] = tuple(fields["history"])
        fields["legal_moves"] = tuple(fields["legal_moves"])
        if fields.get("original_legal_moves") is not None:
            fields["original_legal_moves"] = tuple(fields["original_legal_moves"])
        return cls(**fields)


@dataclass
class EditorBatch:
    tokens: torch.Tensor
    rules: torch.Tensor
    lengths: torch.Tensor
    legal_mask: torch.Tensor
    examples: tuple[EditorExample, ...]


def collate(examples: Sequence[EditorExample], device="cpu") -> EditorBatch:
    """Right-pad original histories and keep the entire 61-token target space."""
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    lengths = np.array([example.prefix_length for example in examples], dtype=np.int64)
    tokens = np.zeros((len(examples), int(lengths.max())), dtype=np.int64)
    rules = np.zeros((len(examples), 128), dtype=np.float32)
    legal = np.zeros((len(examples), 61), dtype=bool)
    for row, example in enumerate(examples):
        if not 1 <= example.prefix_length <= 59 or len(example.history) != example.prefix_length:
            raise ValueError("Editor examples require 1..59 original input moves")
        if example.square in CENTRAL_SQUARES or not 0 <= example.square < 64:
            raise ValueError("An editor target must be a noncentral board square")
        if example.action not in ("flip", "erase"):
            raise ValueError(f"Unknown editor action: {example.action}")
        if not example.legal_moves:
            raise ValueError("Terminal targets must be skipped before batching")
        if any(move not in PLAYABLE_SQUARES for move in (*example.history, *example.legal_moves)):
            raise ValueError("A history or target square has no vocabulary token")
        tokens[row, :example.prefix_length] = SQUARE_TO_TOKEN[list(example.history)]
        rules[row, example.square + (64 if example.action == "erase" else 0)] = 1
        legal[row, SQUARE_TO_TOKEN[list(example.legal_moves)]] = True
    return EditorBatch(
        tokens=torch.from_numpy(tokens).to(device),
        rules=torch.from_numpy(rules).to(device),
        lengths=torch.from_numpy(lengths).to(device),
        legal_mask=torch.from_numpy(legal).to(device),
        examples=tuple(examples),
    )


class GameSplits:
    def __init__(self, data: str | Path, changed_target_fraction: float | None = 0.9):
        if changed_target_fraction is not None and not 0 <= changed_target_fraction <= 1:
            raise ValueError("Changed-target fraction must be in [0, 1] or None")
        self.changed_target_fraction = changed_target_fraction
        self.data = Path(data)
        self.manifest_path = self.data / "dataset_manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        self.board_type = original_engine(REPO)
        self.moves = {}
        self.counts = {}
        for name in ("train", "val", "test"):
            spec = self.manifest["splits"][name]
            count = int(spec["games"])
            shape = tuple(spec["moves_shape"])
            path = self.data / spec["moves_file"]
            if shape != (count, 60) or spec["moves_dtype"] != "uint8":
                raise ValueError(f"Unexpected {name} move-array specification")
            if count < 1 or path.stat().st_size != count * 60:
                raise ValueError(f"Unexpected {name} move-file size")
            self.counts[name] = count
            self.moves[name] = np.memmap(path, dtype=np.uint8, mode="r", shape=shape)
        self.stats = {
            "requested_changed_target_fraction": changed_target_fraction,
            "training_attempts": 0,
            "training_terminal_skips": 0,
            "training_stratum_rejections": 0,
            "training_accepted_count": 0,
            "training_changed_count": 0,
            "training_changed_fraction": None,
            "training_accepted": {
                action: {"changed": 0, "unchanged": 0} for action in ("flip", "erase")
            },
            "training_rejections": {
                action: {"wanted_changed": 0, "wanted_unchanged": 0}
                for action in ("flip", "erase")
            },
            "training_terminal_skips_by_action": {"flip": 0, "erase": 0},
            "fixed_terminal_skips": {},
        }

    def replay(self, split: str, game_index: int, prefix_length: int):
        if split not in self.moves or not 0 <= game_index < self.counts[split]:
            raise ValueError("Unknown game split or out-of-range game index")
        if not 1 <= prefix_length <= 59:
            raise ValueError("Prefix length must be in 1..59")
        history = tuple(int(move) for move in self.moves[split][game_index, :prefix_length])
        board = self.board_type()
        passes = 0
        for move in history:
            before = board.next_hand_color
            board.umpire(move)
            # A forced pass followed by an opponent move returns the nominal
            # next side to its previous value; ordinary moves reverse it.
            passes += board.next_hand_color == before
        return board, history, int(passes)

    def example(self, split: str, game_index: int, prefix_length: int,
                square: int, action: str) -> EditorExample | None:
        """Build one exact simulator target; return None if both sides are stuck."""
        board, history, history_passes = self.replay(split, game_index, prefix_length)
        return self._edit_board(board, split, game_index, history, history_passes, square, action)

    def original_legal_moves(self, example: EditorExample) -> tuple[int, ...]:
        """Read the simulator's original target, including for legacy caches."""
        if example.original_legal_moves is not None:
            return example.original_legal_moves
        board, _, _ = self.replay(example.split, example.game_index, example.prefix_length)
        return tuple(sorted(int(move) for move in board.get_valid_moves()))

    def changed_legal_set(self, example: EditorExample) -> bool:
        return example.legal_moves != self.original_legal_moves(example)

    @staticmethod
    def _edit_board(board, split, game_index, history, history_passes, square, action):
        if square in CENTRAL_SQUARES or not 0 <= square < 64:
            raise ValueError("An editor target must be a noncentral square")
        flat = board.state.reshape(64)
        original_color = int(flat[square])
        if original_color not in (-1, 1):
            raise ValueError("An editor target must be occupied on the original board")
        nominal = int(board.next_hand_color)
        original_legal = tuple(sorted(int(move) for move in board.get_valid_moves()))
        if action == "flip":
            flat[square] = -original_color
        elif action == "erase":
            flat[square] = 0
        else:
            raise ValueError(f"Unknown editor action: {action}")
        legal = tuple(sorted(int(move) for move in board.get_valid_moves()))
        if int(board.next_hand_color) != nominal:
            raise AssertionError("Computing an edited target changed the nominal turn")
        if not legal:
            return None
        forced_pass = board.tentative_move(legal[0]) == 2
        return EditorExample(
            split=split, game_index=int(game_index), history=history,
            prefix_length=len(history), square=int(square), action=action,
            original_color=original_color, legal_moves=legal,
            nominal_next_hand_color=nominal,
            effective_next_hand_color=-nominal if forced_pass else nominal,
            forced_pass=bool(forced_pass), history_passes=history_passes,
            original_legal_moves=original_legal,
        )

    def _draw(self, split: str, game_index: int, rng: np.random.Generator,
              action: str | None = None):
        prefix_length = int(rng.integers(1, 60))
        board, history, passes = self.replay(split, game_index, prefix_length)
        flat = board.state.reshape(64)
        occupied = PLAYABLE_SQUARES[flat[PLAYABLE_SQUARES] != 0]
        square = int(rng.choice(occupied))
        if action is None:
            action = "flip" if int(rng.integers(0, 2)) == 1 else "erase"
        return self._edit_board(board, split, game_index, history, passes, square, action)

    def sample_training_examples(self, batch_size: int, rng: np.random.Generator):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        examples = []
        while len(examples) < batch_size:
            # Choose the output stratum before proposing candidates. Rechoosing
            # after rejection would bias FLIP/ERASE and the requested mixture
            # toward strata that happen to accept more often.
            action = None
            wanted_changed = None
            if self.changed_target_fraction is not None:
                action = "flip" if int(rng.integers(0, 2)) == 1 else "erase"
                wanted_changed = bool(rng.random() < self.changed_target_fraction)
            for _ in range(10000):
                game_index = int(rng.integers(0, self.counts["train"]))
                example = self._draw("train", game_index, rng, action=action)
                self.stats["training_attempts"] += 1
                if example is None:
                    self.stats["training_terminal_skips"] += 1
                    if action is not None:
                        self.stats["training_terminal_skips_by_action"][action] += 1
                    continue
                changed = self.changed_legal_set(example)
                if wanted_changed is not None and changed != wanted_changed:
                    self.stats["training_stratum_rejections"] += 1
                    key = "wanted_changed" if wanted_changed else "wanted_unchanged"
                    self.stats["training_rejections"][action][key] += 1
                    continue
                self.stats["training_accepted"][example.action][
                    "changed" if changed else "unchanged"] += 1
                self.stats["training_accepted_count"] += 1
                self.stats["training_changed_count"] += int(changed)
                self.stats["training_changed_fraction"] = (
                    self.stats["training_changed_count"] / self.stats["training_accepted_count"])
                examples.append(example)
                break
            else:
                raise RuntimeError(
                    f"Cannot find a nonterminal training edit for action={action}, "
                    f"changed_legal_set={wanted_changed}")
        return examples

    def sample_training_batch(self, batch_size: int, rng: np.random.Generator,
                              device="cpu") -> EditorBatch:
        return collate(self.sample_training_examples(batch_size, rng), device)

    def _fixed_identity(self, split: str, seed: int):
        spec = self.manifest["splits"][split]
        return {
            "split": split, "seed": seed,
            "split_seed_offset": SPLIT_SEED_OFFSETS[split],
            "game_count": self.counts[split],
            "manifest_sha256": _sha256(self.manifest_path),
            "moves_sha256": _sha256(self.data / spec["moves_file"]),
            "engine_sha256": _sha256(REPO / "data" / "othello.py"),
            "sampler": "one accepted case per game; prefix uniform1..59; occupied noncentral square uniform; FLIP/ERASE equiprobable; retry terminal in the same game",
            "random_generator": "NumPy PCG64; SeedSequence([seed,split_seed_offset])",
        }

    def fixed_examples(self, split: str, seed: int = 42,
                       path: str | Path | None = None) -> list[EditorExample]:
        if split not in SPLIT_SEED_OFFSETS:
            raise ValueError("Fixed evaluation examples require the val or test game split")
        identity = self._fixed_identity(split, seed)
        path = Path(path) if path is not None else None
        if path is not None and path.exists():
            saved = json.loads(path.read_text())
            if saved.get("schema_version") != 1 or saved.get("identity") != identity:
                raise ValueError(f"Fixed-example cache identity mismatch: {path}")
            examples = [EditorExample.from_dict(case) for case in saved["examples"]]
            if len(examples) != self.counts[split] or any(
                    case.split != split or case.game_index != index
                    for index, case in enumerate(examples)):
                raise ValueError("Fixed-example cache does not have exactly one case per split game")
            examples = [replace(case, original_legal_moves=self.original_legal_moves(case))
                        if case.original_legal_moves is None else case for case in examples]
            self.stats["fixed_terminal_skips"][split] = int(saved["terminal_skips"])
            return examples
        rng = np.random.default_rng(np.random.SeedSequence([seed, SPLIT_SEED_OFFSETS[split]]))
        examples = []
        skips = 0
        for game_index in range(self.counts[split]):
            for _ in range(10000):
                example = self._draw(split, game_index, rng)
                if example is not None:
                    examples.append(example)
                    break
                skips += 1
            else:
                raise RuntimeError(f"Cannot find a nonterminal edit for {split} game {game_index}")
        self.stats["fixed_terminal_skips"][split] = skips
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"schema_version": 1, "identity": identity, "terminal_skips": skips,
                       "example_count": len(examples),
                       "examples": [case.to_dict() for case in examples]}
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            temporary.replace(path)
        return examples
