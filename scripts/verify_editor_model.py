#!/usr/bin/env python3
"""Check editor placement, frozen weights, padding, gradients and TL parity."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mech_int.editor_model import EditedOthelloGPT
from mech_int.reproduce_linear_probe import CHECKPOINT_SHA256, write_json


def compare(actual, expected, device):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    torch.testing.assert_close(actual, expected,
                               atol=1e-4 if device == "cpu" else 1e-3, rtol=1e-4)
    return {"maximum_absolute_error": (actual - expected).abs().max().item(),
            "rms_error": (actual - expected).square().mean().sqrt().item()}


def uniform_legal_loss(logits, legal):
    return (-(logits.log_softmax(-1) * legal).sum(-1) / legal.sum(-1)).mean()


def verify(args):
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    version = importlib.metadata.version("transformer-lens")
    if version != "1.6.0":
        raise ValueError(f"Reference must use TransformerLens 1.6.0, found {version}")
    devices = ["cpu", "mps"] if args.device == "both" else [args.device]
    if "mps" in devices and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this execution context")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(42)
    reference = HookedTransformer(HookedTransformerConfig(
        n_layers=8, d_model=512, d_head=64, n_heads=8, d_mlp=2048,
        d_vocab=61, n_ctx=59, act_fn="gelu", normalization_type="LNPre", device="cpu"))
    reference.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    reference.eval().requires_grad_(False)
    model = EditedOthelloGPT(args.checkpoint, bottleneck=args.bottleneck).eval()
    lengths = torch.tensor([1, 2, 3, 7, 15, 31, 59])
    tokens = torch.randint(1, 61, (len(lengths), 59))
    padding = torch.arange(59)[None] >= lengths[:, None]
    tokens[padding] = 0
    rules = torch.zeros(len(lengths), 128)
    for row in range(len(lengths)):
        rules[row, (row % 2) * 64 + (row * 9) % 64] = 1
    rows, last = torch.arange(len(lengths)), lengths - 1
    expected_parameters = 8 * args.bottleneck * (640 + 512 + 1)
    report = {"passed": True, "checkpoint_sha256": CHECKPOINT_SHA256,
              "bottleneck": args.bottleneck, "trainable_parameters": expected_parameters,
              "reference": "TransformerLens 1.6.0 hook_resid_post, CPU float32",
              "prefix_lengths": lengths.tolist(), "vocabulary_logits_checked": 61,
              "devices": devices, "comparisons": []}
    with torch.inference_mode():
        base_logits = model.base(tokens)[rows, last]
        for device in devices:
            model.to(device)
            local_tokens, local_lengths, local_rules = tokens.to(device), lengths.to(device), rules.to(device)
            local_rows, local_last = rows.to(device), last.to(device)
            expected = model.base(local_tokens)[local_rows, local_last]
            torch.testing.assert_close(model(local_tokens, None, local_lengths), expected, rtol=0, atol=0)
            torch.testing.assert_close(model(local_tokens, local_rules, local_lengths), expected, rtol=0, atol=0)
            report["comparisons"].append({"device": device, "mode": "zero_initialization",
                                           "bit_exact": True, **compare(expected, base_logits, device)})
    model.to("cpu")
    # Nonzero U exercises all editor hooks and the propagation of previous edits.
    with torch.no_grad():
        for editor in model.editors:
            editor.up.weight.normal_(mean=0, std=.02)
    hooks = []
    for layer, editor in enumerate(model.editors):
        def patch(activation, hook, editor=editor):
            edited = activation.clone()
            final = activation[rows, last]
            edited[rows, last] = final + editor(final, rules)
            return edited
        hooks.append((f"blocks.{layer}.hook_resid_post", patch))
    with torch.inference_mode():
        reference_logits = reference.run_with_hooks(tokens, fwd_hooks=hooks)[rows, last]
        for device in devices:
            model.to(device)
            local_tokens, local_lengths, local_rules = tokens.to(device), lengths.to(device), rules.to(device)
            actual = model(local_tokens, local_rules, local_lengths)
            report["comparisons"].append({"device": device, "mode": "nonzero_editors",
                                           **compare(actual, reference_logits, device)})
            # Changing every pad cannot affect a causal model's real tokens.
            different_padding = local_tokens.clone()
            different_padding[padding.to(device)] = 60
            torch.testing.assert_close(model(different_padding, local_rules, local_lengths), actual,
                                       atol=0, rtol=0)
            base_residuals = list(model.base.residuals(local_tokens))
            for (layer, edited), (_, original) in zip(model.residuals(
                    local_tokens, local_rules, local_lengths), base_residuals):
                for row, length in enumerate(lengths.tolist()):
                    torch.testing.assert_close(edited[row, :length - 1], original[row, :length - 1],
                                               atol=0, rtol=0)
            # Independently truncated prefixes establish correct final indexing.
            for row, length in enumerate(lengths.tolist()):
                one = model(local_tokens[row:row + 1, :length], local_rules[row:row + 1])
                compare(one[0], actual[row], device)
            # Bypass remains exact even with nonzero editors.
            local_rows, local_last = rows.to(device), last.to(device)
            torch.testing.assert_close(model(local_tokens, None, local_lengths),
                                       model.base(local_tokens)[local_rows, local_last], atol=0, rtol=0)
    report["right_padding_independent"] = True
    report["earlier_tokens_bit_exact"] = True
    report["individually_truncated_prefixes_match"] = True

    # Reinitialize U and use the actual edited graph for two optimizer updates.
    # At zero U, D/b gradients vanish; after U's first update they must flow.
    for device in devices:
        torch.manual_seed(42)
        train_model = EditedOthelloGPT(args.checkpoint, bottleneck=args.bottleneck).to(device).train()
        assert not list(train_model.base.parameters())
        before_buffers = {name: value.detach().clone() for name, value in train_model.base.named_buffers()}
        assert sum(p.numel() for p in train_model.parameters()) == expected_parameters
        optimizer = torch.optim.AdamW(train_model.parameters(), lr=3e-4, weight_decay=1e-4)
        legal = torch.zeros(len(lengths), 61, device=device)
        for row in range(len(lengths)):
            legal[row, [row, row + 7, row + 15]] = 1
        local_tokens, local_lengths, local_rules = tokens.to(device), lengths.to(device), rules.to(device)
        gradients = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = uniform_legal_loss(train_model(local_tokens, local_rules, local_lengths), legal)
            loss.backward()
            for layer, editor in enumerate(train_model.editors):
                norms = {name: parameter.grad.norm().item()
                         for name, parameter in editor.named_parameters()}
                assert norms["up.weight"] > 0, (device, step, layer, norms)
                if step == 0:
                    assert norms["down.weight"] == norms["down.bias"] == 0, norms
                else:
                    assert norms["down.weight"] > 0 and norms["down.bias"] > 0, norms
                gradients.append({"step": step + 1, "layer": layer, "norms": norms})
            optimizer.step()
        for name, value in train_model.base.named_buffers():
            torch.testing.assert_close(value, before_buffers[name], atol=0, rtol=0)
            assert value.grad is None and not value.requires_grad
        report.setdefault("gradient_checks", []).append({"device": device, "passed": True,
                                                         "frozen_buffers_unchanged": True,
                                                         "gradients": gradients})
    # Full-vocabulary normalization matters: a large illegal logit raises loss.
    toy = torch.zeros(1, 61)
    legal = torch.zeros_like(toy)
    legal[:, [1, 2]] = 1
    torch.testing.assert_close(uniform_legal_loss(toy, legal), torch.tensor(61.).log())
    toy[0, 60] = 10
    assert uniform_legal_loss(toy, legal) > torch.tensor(61.).log()
    report["uniform_legal_full_vocabulary_loss_check"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO / "synthetic_model.pth")
    parser.add_argument("--report", type=Path,
                        default=REPO / "output/editor_mlp_128_changed90/model_verification.json")
    parser.add_argument("--device", choices=("cpu", "mps", "both"), default="cpu")
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    report = verify(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)
    print(json.dumps({key: report[key] for key in (
        "passed", "devices", "bottleneck", "trainable_parameters", "prefix_lengths", "comparisons", "right_padding_independent",
        "earlier_tokens_bit_exact", "individually_truncated_prefixes_match")}, indent=2))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
