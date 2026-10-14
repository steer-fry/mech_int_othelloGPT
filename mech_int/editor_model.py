"""Trainable post-block residual editors around the frozen OthelloGPT.

Only the editors contain parameters. Transformer weights remain buffers, while
autograd still differentiates through each block after an earlier layer's edit.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mech_int.reproduce_linear_probe import FrozenBlock, FrozenOthelloGPT


class ResidualEditor(nn.Module):
    """U ReLU(D [residual; FLIP mask; ERASE mask] + b)."""

    def __init__(self, bottleneck: int = 128):
        super().__init__()
        if bottleneck < 1:
            raise ValueError("The editor bottleneck must be positive")
        # nn.Linear's random initialization uses the caller's torch RNG seed.
        self.down = nn.Linear(512 + 128, bottleneck, bias=True)
        self.up = nn.Linear(bottleneck, 512, bias=False)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)

    def forward(self, residual: torch.Tensor, rules: torch.Tensor) -> torch.Tensor:
        return self.up(torch.relu(self.down(torch.cat((residual, rules), dim=-1))))


class EditedOthelloGPT(nn.Module):
    """Eight independent editors, applied to each row's final real input token.

    ``tokens`` has shape (batch, positions), padded on the right. ``lengths``
    counts real tokens, with full width assumed when omitted. ``rules`` has
    shape (batch, 128), with FLIP then ERASE in simulator A1..H8 square order.
    The forward output contains all 61 next-token logits for each example.
    Supplying no rule completely bypasses the editor modules.
    """

    def __init__(self, checkpoint: str | Path, bottleneck: int = 128):
        super().__init__()
        self.base = FrozenOthelloGPT(checkpoint)
        self.base.requires_grad_(False)
        self.editors = nn.ModuleList(ResidualEditor(bottleneck) for _ in self.base.blocks)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def _inputs(self, tokens, rules, lengths):
        if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError("tokens must be a nonempty (batch, positions) tensor")
        if tokens.dtype != torch.long:
            raise ValueError("tokens must have dtype torch.long")
        if tokens.shape[1] > self.base.position.shape[0]:
            raise ValueError("Token width exceeds the model context length")
        if tokens.device != self.base.embedding.device:
            raise ValueError("tokens and model must be on the same device")
        if lengths is None:
            lengths = torch.full((tokens.shape[0],), tokens.shape[1],
                                 dtype=torch.long, device=tokens.device)
        else:
            if lengths.shape != (tokens.shape[0],) or lengths.dtype != torch.long:
                raise ValueError("lengths must be a torch.long tensor of shape (batch,)")
            lengths = lengths.to(tokens.device)
            if torch.any((lengths < 1) | (lengths > tokens.shape[1])).item():
                raise ValueError("Every length must be between 1 and the padded token width")
        if rules is not None:
            if rules.shape != (tokens.shape[0], 128):
                raise ValueError("rules must have shape (batch, 128)")
            rules = rules.to(device=tokens.device, dtype=self.base.embedding.dtype)
        rows = torch.arange(tokens.shape[0], device=tokens.device)
        return rules, rows, lengths - 1

    def _edited_residuals(self, tokens, rules, rows, last):
        residual = self.base.embedding[tokens] + self.base.position[:tokens.shape[1]]
        for layer, (block, editor) in enumerate(zip(self.base.blocks, self.editors)):
            # Frozen block weights do not mean a no_grad forward: later blocks
            # must transmit the loss gradient to every preceding editor.
            residual = block(residual)
            if rules is not None:
                final = residual[rows, last]
                residual = residual.clone()
                residual[rows, last] = final + editor(final, rules)
            yield layer, residual

    def residuals(self, tokens, rules=None, lengths=None):
        """Yield full post-edit residuals, primarily for implementation checks."""
        rules, rows, last = self._inputs(tokens, rules, lengths)
        yield from self._edited_residuals(tokens, rules, rows, last)

    def forward(self, tokens, rules=None, lengths=None):
        rules, rows, last = self._inputs(tokens, rules, lengths)
        if rules is None:
            # This exact base call preserves bit-for-bit bypass behavior.
            return self.base(tokens)[rows, last]
        for _, residual in self._edited_residuals(tokens, rules, rows, last):
            pass
        # Unembed all positions before gathering, to preserve the base GEMM
        # dimensions and exact initial equality while U is zero.
        logits = FrozenBlock.normalize(residual) @ self.base.unembedding + self.base.unembedding_bias
        return logits[rows, last]
