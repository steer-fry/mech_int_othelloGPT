"""Train and evaluate only the relative-colour linear probes from Section 3.

The compact frozen transformer is checked against TransformerLens 1.6.0.
All eight independent probes share a frozen forward pass, never their weights.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PAPER_ACCURACY = [90.9, 94.8, 97.2, 98.3, 99.0, 99.4, 99.6, 99.5]
CHECKPOINT_SHA256 = "aef97afb1e7ec9dbef1d9007174458830a5c428d62f1e2338128879b179bd5c4"


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    temporary.replace(path)


class FrozenBlock(nn.Module):
    def __init__(self, sd, layer):
        super().__init__()
        p = f"blocks.{layer}."
        qkv = [sd[p + f"attn.W_{c}"].permute(1, 0, 2).reshape(512, 512) for c in "QKV"]
        self.register_buffer("qkv_weight", torch.cat(qkv, dim=1))
        self.register_buffer("qkv_bias", torch.cat([sd[p + f"attn.b_{c}"].reshape(512) for c in "QKV"]))
        self.register_buffer("output_weight", sd[p + "attn.W_O"].reshape(512, 512))
        self.register_buffer("output_bias", sd[p + "attn.b_O"])
        for name in ("W_in", "b_in", "W_out", "b_out"):
            self.register_buffer(name, sd[p + "mlp." + name])
        self.register_buffer("mask", sd[p + "attn.mask"].bool())

    @staticmethod
    def normalize(x):
        # Use the exact LayerNormPre operations, epsilon and biased variance.
        x = x - x.mean(dim=-1, keepdim=True)
        return x / (x.square().mean(dim=-1, keepdim=True) + 1e-5).sqrt()

    def forward(self, x, attention_delta=None):
        batch, positions, _ = x.shape
        qkv = self.normalize(x) @ self.qkv_weight + self.qkv_bias
        q, k, v = [a.reshape(batch, positions, 8, 64).transpose(1, 2) for a in qkv.chunk(3, dim=-1)]
        scores = (q @ k.transpose(-1, -2)) / 8.0
        scores = scores.masked_fill(~self.mask[:positions, :positions], -1e5)
        z = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, positions, 512)
        attention_out = z @ self.output_weight + self.output_bias
        if attention_delta is not None:
            # Equivalent to patching blocks.<layer>.hook_attn_out in the
            # released intervention scripts, at the final history token only.
            attention_out = attention_out.clone()
            attention_out[:, -1, :] = attention_out[:, -1, :] + attention_delta
        x = x + attention_out
        hidden = F.gelu(self.normalize(x) @ self.W_in + self.b_in)
        return x + (hidden @ self.W_out + self.b_out)


class FrozenOthelloGPT(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        actual = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if actual != CHECKPOINT_SHA256:
            raise ValueError(f"Unexpected checkpoint SHA256: {actual}")
        sd = torch.load(checkpoint, map_location="cpu")
        self.register_buffer("embedding", sd["embed.W_E"])
        self.register_buffer("position", sd["pos_embed.W_pos"])
        self.register_buffer("unembedding", sd["unembed.W_U"])
        self.register_buffer("unembedding_bias", sd["unembed.b_U"])
        self.blocks = nn.ModuleList(FrozenBlock(sd, l) for l in range(8))
        self.eval()

    def residuals(self, tokens, attention_deltas=None):
        x = self.embedding[tokens] + self.position[:tokens.shape[1]]
        for layer, block in enumerate(self.blocks):
            delta = None if attention_deltas is None else attention_deltas.get(layer)
            x = block(x, delta)
            yield layer, x

    def forward(self, tokens, attention_deltas=None):
        for _, residual in self.residuals(tokens, attention_deltas):
            pass
        return FrozenBlock.normalize(residual) @ self.unembedding + self.unembedding_bias


class Split:
    def __init__(self, directory, name, manifest):
        directory = Path(directory)
        spec = manifest["splits"][name]
        self.size = spec["games"]
        self.moves = np.memmap(directory / spec["moves_file"], dtype=np.uint8, mode="r", shape=(self.size, 60))
        self.labels = np.memmap(directory / spec["labels_file"], dtype=np.uint8, mode="r", shape=(self.size, 59, 64))

    def batch(self, indices, device):
        # Token zero is padding; the four initially occupied squares have no tokens.
        mapping = np.zeros(64, dtype=np.int64)
        mapping[[s for s in range(64) if s not in (27, 28, 35, 36)]] = np.arange(1, 61)
        tokens = torch.from_numpy(mapping[self.moves[indices, :59]]).to(device)
        labels = torch.from_numpy(np.array(self.labels[indices], dtype=np.int64)).to(device)
        return tokens, labels


def probe_logits(residual, weight):
    return (residual @ weight.reshape(512, 192)).reshape(*residual.shape[:2], 64, 3)


@torch.no_grad()
def cache_split(model, split, batch_size, device):
    cache = [[] for _ in range(8)]
    labels = []
    for start in range(0, split.size, batch_size):
        tokens, targets = split.batch(np.arange(start, min(start + batch_size, split.size)), device)
        for layer, residual in model.residuals(tokens):
            cache[layer].append(residual.cpu())
        labels.append(targets.cpu())
    return [torch.cat(c) for c in cache], torch.cat(labels)


@torch.no_grad()
def evaluate_cached(cache, labels, probes, batch_size, device):
    results = []
    for layer, weight in enumerate(probes):
        correct = 0
        nll = 0.0
        by_position = torch.zeros(59, dtype=torch.int64)
        confusion = torch.zeros(3, 3, dtype=torch.int64)
        for start in range(0, len(labels), batch_size):
            targets = labels[start:start + batch_size].to(device)
            residual = cache[layer][start:start + batch_size].to(device)
            logits = probe_logits(residual, weight)
            nll += F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1), reduction="sum").item()
            prediction = logits.argmax(-1)
            hits = (prediction == targets).cpu()
            correct += hits.sum().item()
            by_position += hits.sum(dim=(0, 2))
            indices = (targets * 3 + prediction).reshape(-1).cpu()
            confusion += torch.bincount(indices, minlength=9).reshape(3, 3)
        total = labels.numel()
        results.append({
            "layer": layer, "correct": correct, "total": total,
            "accuracy": correct / total, "accuracy_percent": 100 * correct / total,
            "loss": nll / (len(labels) * 59),
            "accuracy_by_position": (by_position.double() / (len(labels) * 64)).tolist(),
            "confusion_matrix_target_by_prediction": confusion.tolist(),
        })
    return results


def verify_model(args, device):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    torch.manual_seed(123)
    torch.set_num_threads(4)
    cfg = HookedTransformerConfig(n_layers=8, d_model=512, d_head=64, n_heads=8,
                                 d_mlp=2048, d_vocab=61, n_ctx=59, act_fn="gelu",
                                 normalization_type="LNPre", device="cpu")
    reference = HookedTransformer(cfg)
    reference.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    reference.eval()
    compact = FrozenOthelloGPT(args.checkpoint)
    tokens = torch.randint(1, 61, (3, 59))
    differences = []
    with torch.inference_mode():
        _, cache = reference.run_with_cache(tokens, return_type=None)
        for layer, actual in compact.residuals(tokens):
            expected = cache["resid_post", layer]
            max_error = (actual - expected).abs().max().item()
            rms_error = (actual - expected).square().mean().sqrt().item()
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
            differences.append({"layer": layer, "max_absolute_error": max_error, "rms_error": rms_error})
    result = {"passed": True, "reference": "TransformerLens 1.6.0, CPU float32", "differences": differences}
    if device != "cpu":
        compact.to(device)
        gpu_differences = []
        with torch.inference_mode():
            for layer, actual in compact.residuals(tokens.to(device)):
                expected = cache["resid_post", layer]
                actual = actual.cpu()
                torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-4)
                weight = torch.load(Path("mech_int/probes/linear") / f"resid_{layer}_linear.pth", map_location="cpu", weights_only=True)
                agreement = (probe_logits(actual, weight).argmax(-1) == probe_logits(expected, weight).argmax(-1)).float().mean().item()
                gpu_differences.append({"layer": layer, "max_absolute_error": (actual - expected).abs().max().item(),
                                        "rms_error": (actual - expected).square().mean().sqrt().item(),
                                        "bundled_probe_prediction_agreement": agreement})
        result["device"] = device
        result["device_differences"] = gpu_differences
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "model_verification.json", result)


def benchmark(args, device):
    torch.manual_seed(42)
    model = FrozenOthelloGPT(args.checkpoint).to(device)
    weights = [nn.Parameter(torch.randn(1, 512, 8, 8, 3, device=device) / math.sqrt(512)) for _ in range(8)]
    optimizers = [torch.optim.AdamW([w], lr=.01, betas=(.9, .99), weight_decay=.01) for w in weights]
    tokens = torch.randint(1, 61, (args.batch_size, 59), device=device)
    labels = torch.randint(0, 3, (args.batch_size, 59, 64), device=device)
    durations = []
    for step in range(6):
        started = time.monotonic()
        with torch.no_grad():
            residuals = list(model.residuals(tokens))
        for (layer, residual), weight, optimizer in zip(residuals, weights, optimizers):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe_logits(residual, weight).reshape(-1, 3), labels.reshape(-1)) * 64
            loss.backward()
            optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        duration = time.monotonic() - started
        durations.append(duration)
        print(f"benchmark step={step} seconds={duration:.3f}", flush=True)
    result = {"device": device, "batch_size": args.batch_size,
              "seconds_per_step": float(np.median(durations[2:])), "durations": durations}
    print(json.dumps(result, indent=2))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "benchmark.json", result)


def run(args, device):
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data / "dataset_manifest.json").read_text())
    validation = Split(args.data, "val", manifest)
    test = Split(args.data, "test", manifest)
    if validation.size != 512 or test.size != 1000:
        raise ValueError("Paper protocol requires exactly 512 validation and 1000 test games")
    model = FrozenOthelloGPT(args.checkpoint).to(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = {"paper": "https://aclanthology.org/2023.blackboxnlp-1.2.pdf",
              "scope": "Table 1: Linear {MINE, YOURS, EMPTY}, eight post-block residual probes",
              "optimizer": "AdamW", "learning_rate": .01, "weight_decay": .01, "betas": [.9, .99],
              "validation_every_optimizer_steps": args.validation_interval, "validation_patience": 10,
              "released_code_valid_every_sample_offset": 200,
              "validation_games": 512, "test_games": 1000, "training_pool_games": manifest["splits"]["train"]["games"],
              "batch_size": args.batch_size, "seed": args.seed, "device": device,
              "precision": "float32", "classes": ["EMPTY", "YOURS", "MINE"],
              "labels": "Released helper: position parity, also after implicit passes",
              "evaluation": "All 59 input positions and all 64 tiles; correct / total",
              "checkpoint_sha256": CHECKPOINT_SHA256,
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "transformer-lens")},
              "data_manifest": manifest,
              "protocol_notes": [
                  "Paper does not publish exact game identities, split seed, batch size or fixed training-pool size.",
                  "Seeded fresh games reproduce uniform random legal play and full-60-move filtering.",
                  "512 and 1000 game splits are fully disjoint from training and from each other.",
                  "Batch size128 and initialization std1/sqrt512 follow released code.",
                  "User selected released-code cadence: valid_every200 sample offsets means25 updates at batch128 (lcm128,200=3200).",
                  "Train until patience10 for every layer, with no one-epoch truncation.",
                  "Sharing frozen forward passes does not couple the eight probe parameters or optimizer states.",
              ]}
    write_json(args.output / "config.json", config)
    if args.mode == "train":
        train = Split(args.data, "train", manifest)
        probes = [nn.Parameter(torch.randn(1, 512, 8, 8, 3, device=device) / math.sqrt(512)) for _ in range(8)]
        optimizers = [torch.optim.AdamW([w], lr=.01, betas=(.9, .99), weight_decay=.01) for w in probes]
        validation_cache, validation_labels = cache_split(model, validation, args.batch_size, device)
        best = [float("inf")] * 8
        patience = [0] * 8
        active = [True] * 8
        best_steps = [0] * 8
        stopped_steps = [None] * 8
        step = 0
        seen = 0
        epoch = 0
        started = time.monotonic()
        history = []
        rng = torch.Generator().manual_seed(args.seed)
        next_offset = 0
        order = None
        if args.resume:
            state = torch.load(args.output / "training_state.pth", map_location=device, weights_only=False)
            step, seen, epoch = state["step"], state["seen"], state["epoch"]
            order, next_offset = state["order"], state["next_offset"]
            best, patience, active = state["best"], state["patience"], state["active"]
            best_steps, stopped_steps = state["best_steps"], state["stopped_steps"]
            rng.set_state(state["rng"].cpu())
            for w, value, optimizer, optimizer_state in zip(probes, state["probes"], optimizers, state["optimizers"]):
                w.data.copy_(value)
                optimizer.load_state_dict(optimizer_state)
            history = json.loads((args.output / "training_history.json").read_text())
            print(f"Resuming at step={step}, epoch={epoch}, offset={next_offset}", flush=True)
        while any(active):
            if order is None or next_offset >= train.size:
                order = torch.randperm(train.size, generator=rng).numpy()
                epoch += 1
                next_offset = 0
            for offset in range(next_offset, train.size, args.batch_size):
                indices = order[offset:offset + args.batch_size]
                tokens, targets = train.batch(indices, device)
                with torch.no_grad():
                    residuals = list(model.residuals(tokens))
                losses = [None] * 8
                for layer, residual in residuals:
                    if not active[layer]:
                        continue
                    optimizer = optimizers[layer]
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(probe_logits(residual, probes[layer]).reshape(-1, 3), targets.reshape(-1)) * 64
                    loss.backward()
                    optimizer.step()
                    losses[layer] = float(loss.detach().cpu())
                step += 1
                seen += len(indices)
                if step % 25 == 0:
                    print(f"step={step} games={seen} elapsed={time.monotonic()-started:.1f}s active={sum(active)} losses={losses}", flush=True)
                # The released trainer checks sample offset, including offset0
                # after the first update. Cadence also restarts at each epoch.
                validate_now = offset % 200 == 0 if args.validation_interval == 25 and args.batch_size == 128 else step % args.validation_interval == 0
                if validate_now:
                    metrics = evaluate_cached(validation_cache, validation_labels, probes, args.batch_size, device)
                    for layer, result in enumerate(metrics):
                        if not active[layer]:
                            continue
                        improved = result["loss"] < best[layer]
                        if improved:
                            best[layer] = result["loss"]
                            best_steps[layer] = step
                            patience[layer] = 0
                            torch.save(probes[layer].detach().cpu(), args.output / f"resid_{layer}_linear.pth")
                        else:
                            patience[layer] += 1
                        if patience[layer] >= 10:
                            active[layer] = False
                            stopped_steps[layer] = step
                        result.update({"step": step, "games_seen": seen, "patience": patience[layer], "improved": improved})
                        history.append(result)
                        print(f"validation layer={layer} accuracy={result['accuracy_percent']:.4f}% loss={result['loss']:.6f} patience={patience[layer]}", flush=True)
                    write_json(args.output / "training_history.json", history)
                    status = {"complete": not any(active), "step": step, "games_seen": seen, "epoch": epoch,
                              "elapsed_seconds": time.monotonic() - started, "best_validation_loss": best,
                              "best_steps": best_steps, "stopped_steps": stopped_steps, "patience": patience,
                              "active": active}
                    write_json(args.output / "training_status.json", status)
                    torch.save({"step": step, "seen": seen, "epoch": epoch, "order": order,
                                "next_offset": offset + len(indices), "probes": [w.detach().cpu() for w in probes],
                                "optimizers": [o.state_dict() for o in optimizers], "rng": rng.get_state(),
                                "best": best, "patience": patience, "active": active,
                                "best_steps": best_steps, "stopped_steps": stopped_steps}, args.output / "training_state.pth")
                    if not any(active):
                        break
            next_offset = train.size
        del validation_cache
    # Testing uses validation-selected checkpoints only, after all training ends.
    probes = [torch.load(args.output / f"resid_{l}_linear.pth", map_location=device) for l in range(8)]
    test_cache, test_labels = cache_split(model, test, args.batch_size, device)
    metrics = evaluate_cached(test_cache, test_labels, probes, args.batch_size, device)
    for layer, row in enumerate(metrics):
        row["paper_accuracy_percent"] = PAPER_ACCURACY[layer]
        row["difference_percentage_points"] = row["accuracy_percent"] - PAPER_ACCURACY[layer]
    write_json(args.output / "test_results.json", {"complete": True, "test_games": test.size,
               "positions_per_game": 59, "tiles_per_position": 64, "layers": metrics})
    with (args.output / "accuracy.csv").open("w", newline="") as f:
        fields = ["layer", "accuracy_percent", "paper_accuracy_percent", "difference_percentage_points", "correct", "total"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "evaluate", "verify", "benchmark"], default="train")
    parser.add_argument("--checkpoint", type=Path, default=Path("synthetic_model.pth"))
    parser.add_argument("--data", type=Path, default=Path("data/probe_reproduction"))
    parser.add_argument("--output", type=Path, default=Path("output/linear_probe_reproduction"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-interval", type=int, default=25,
                        help="Optimizer updates;25 matches released valid_every200 sample-offset code at batch128")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if args.mode == "verify":
        verify_model(args, device)
    elif args.mode == "benchmark":
        benchmark(args, device)
    else:
        run(args, device)


if __name__ == "__main__":
    main()
