"""Self-supervised pretraining for the road foundation model.

Objective: **masked-feature reconstruction.**

A random 15 % of nodes are 'masked' — their input feature vector is replaced
by a learned mask token (a single shared vector). The model must reconstruct
the original numeric features (MSE), binary tags (BCE), and highway class
(cross-entropy) of those masked nodes, using only graph context. This forces
the encoder to learn road-class semantics + geometric regularities from
context alone.

Importantly, the **`maxspeed`/`posted_speed_kph` is included in the masked
features** — so the model gets to predict speed from context. We exclude
nothing on principle. (Downstream misalignment scoring asks a *different*
question: not "what is the posted limit?" but "what is the modal posted
limit of geometrically similar segments?" — see model/embed.py.)

CLI:
    rt-pretrain --geojson webapp/tokens.geojson --epochs 200 \
                --runs-dir runs/cambridge_v1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from road_tokeniser.model.dataset import (
    BINARY_FEATURES,
    HIGHWAY_VOCAB,
    NUMERIC_FEATURES,
    build_from_geojson,
)
from road_tokeniser.model.encoder import RoadFoundationModel


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _compute_loss(
    out: dict,
    x: torch.Tensor,
    y_highway: torch.Tensor,
    mask_idx: torch.Tensor,
    n_numeric: int,
    n_binary: int,
) -> tuple[torch.Tensor, dict]:
    """Reconstruction loss on the masked nodes only."""
    target_numeric = x[mask_idx, :n_numeric]
    target_binary = x[mask_idx, n_numeric : n_numeric + n_binary]
    target_highway = y_highway[mask_idx]

    pred_numeric = out["numeric"][mask_idx]
    pred_binary = out["binary"][mask_idx]
    pred_highway = out["highway"][mask_idx]

    l_num = F.mse_loss(pred_numeric, target_numeric)
    l_bin = F.binary_cross_entropy_with_logits(pred_binary, target_binary)
    l_hw = F.cross_entropy(pred_highway, target_highway)

    loss = l_num + 0.5 * l_bin + 0.5 * l_hw
    parts = {
        "loss": float(loss.detach()),
        "loss_numeric": float(l_num.detach()),
        "loss_binary": float(l_bin.detach()),
        "loss_highway": float(l_hw.detach()),
        # Highway classification accuracy on masked nodes (sanity signal)
        "acc_highway": float((pred_highway.argmax(-1) == target_highway).float().mean()),
    }
    return loss, parts


def train(
    geojson_path: Path,
    runs_dir: Path,
    *,
    epochs: int = 200,
    embed_dim: int = 128,
    num_layers: int = 4,
    heads: int = 4,
    mask_rate: float = 0.15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    log_every: int = 10,
    device: torch.device | None = None,
) -> Path:
    """Train the foundation model and save the best checkpoint. Returns its path."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or _pick_device()
    runs_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(runs_dir))

    # ---- Dataset ----
    graph = build_from_geojson(geojson_path)
    print(graph.summary())
    data = graph.to_pyg().to(device)
    n_numeric = len(NUMERIC_FEATURES)
    n_binary = len(BINARY_FEATURES)
    n_highway = len(HIGHWAY_VOCAB)

    # ---- Model ----
    model = RoadFoundationModel(
        in_dim=graph.num_features,
        n_numeric=n_numeric,
        n_binary=n_binary,
        n_highway=n_highway,
        embed_dim=embed_dim,
        num_layers=num_layers,
        heads=heads,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {param_count:,}")

    # ---- Masked-feature setup ----
    # Learnable mask token (added to the input where masked)
    mask_token = torch.nn.Parameter(
        torch.zeros(1, graph.num_features, device=device)
    )
    torch.nn.init.normal_(mask_token, std=0.02)

    opt = torch.optim.AdamW(
        list(model.parameters()) + [mask_token],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss = float("inf")
    best_ckpt = runs_dir / "best.pt"
    config = {
        "geojson": str(geojson_path),
        "epochs": epochs,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "heads": heads,
        "mask_rate": mask_rate,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "device": str(device),
        "param_count": param_count,
        "num_nodes": graph.num_nodes,
        "num_features": graph.num_features,
        "num_edges": int(data.edge_index.shape[1]),
    }
    (runs_dir / "config.json").write_text(json.dumps(config, indent=2))

    x_clean = data.x.clone()

    print(f"\nstarting training on {device} for {epochs} epochs")
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()

        # Sample masked nodes
        n = data.num_nodes
        n_mask = max(1, int(n * mask_rate))
        mask_idx = torch.randperm(n, device=device)[:n_mask]

        # Build masked input: replace masked rows with the learnable mask token
        x_in = x_clean.clone()
        x_in[mask_idx] = mask_token

        out = model(x_in, data.edge_index)
        loss, parts = _compute_loss(
            out, x_clean, data.y_highway, mask_idx, n_numeric, n_binary
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(
                f"epoch {epoch:4d}/{epochs}  "
                f"loss={parts['loss']:.4f}  "
                f"num={parts['loss_numeric']:.4f}  "
                f"bin={parts['loss_binary']:.4f}  "
                f"hw={parts['loss_highway']:.4f}  "
                f"acc_hw={parts['acc_highway']:.3f}  "
                f"lr={scheduler.get_last_lr()[0]:.5f}  "
                f"({elapsed:.1f}s)"
            )

        for k, v in parts.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        if parts["loss"] < best_loss:
            best_loss = parts["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "mask_token": mask_token.detach().cpu(),
                    "config": config,
                    "epoch": epoch,
                    "best_loss": best_loss,
                },
                best_ckpt,
            )

    writer.close()
    print(f"\nbest loss {best_loss:.4f} → {best_ckpt}")
    return best_ckpt


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--geojson", type=Path, required=True)
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mask-rate", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    train(
        geojson_path=args.geojson,
        runs_dir=args.runs_dir,
        epochs=args.epochs,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        mask_rate=args.mask_rate,
        lr=args.lr,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
