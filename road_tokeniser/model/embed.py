"""Load a trained checkpoint, produce per-token embeddings + UMAP + clusters.

Outputs:
  - embeddings parquet (`embeddings.parquet`) with token_id + d-dim vector
  - UMAP 2-D coords (`umap.parquet`)
  - k-means cluster labels (`clusters.parquet`)
  - **`embeddings.geojson`** — original tokens.geojson with extra fields:
      `umap_x`, `umap_y`, `cluster`, and `ml_misalignment_score`.
    This is what the webapp loads to drive the embedding-based colour ramps.

CLI:
    rt-embed --geojson webapp/tokens.geojson \
             --ckpt runs/cambridge_v1/best.pt \
             --out-geojson webapp/embeddings.geojson \
             --n-clusters 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from road_tokeniser.model.dataset import (
    BINARY_FEATURES,
    HIGHWAY_VOCAB,
    NUMERIC_FEATURES,
    build_from_geojson,
)
from road_tokeniser.model.encoder import RoadFoundationModel


def _load_model(ckpt_path: Path, in_dim: int, device: torch.device) -> RoadFoundationModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = RoadFoundationModel(
        in_dim=in_dim,
        n_numeric=len(NUMERIC_FEATURES),
        n_binary=len(BINARY_FEATURES),
        n_highway=len(HIGHWAY_VOCAB),
        embed_dim=cfg["embed_dim"],
        num_layers=cfg["num_layers"],
        heads=cfg["heads"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def ml_misalignment(
    embeddings: np.ndarray, posted: np.ndarray, k: int = 50
) -> np.ndarray:
    """For each token, distance of its posted speed from the modal posted speed
    of its k-nearest neighbours in embedding space.

    The unsupervised misalignment signal: a segment whose embedding-neighbours
    have a different modal posted limit is flagged. Self-similarity ignored.
    """
    # Normalised cosine via L2-normalised embeddings → squared euclidean = 2(1-cos)
    e = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    # Distance matrix — fine at 12 k tokens. For Phase C we'll switch to FAISS.
    # Use top-k indexing via argpartition for speed.
    n = e.shape[0]
    sims = e @ e.T  # [N, N]
    # Mask self
    np.fill_diagonal(sims, -np.inf)
    nn_idx = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neighbours = posted[nn_idx[i]]
        # Mode = most common posted speed among neighbours
        vals, counts = np.unique(neighbours, return_counts=True)
        mode = vals[np.argmax(counts)]
        out[i] = abs(int(posted[i]) - int(mode))
    return out


def run(
    geojson_path: Path,
    ckpt_path: Path,
    out_geojson: Path,
    *,
    n_clusters: int = 12,
    knn_k: int = 50,
    umap_neighbours: int = 20,
    umap_min_dist: float = 0.1,
    seed: int = 42,
) -> dict:
    """End-to-end: load → embed → UMAP → cluster → enrich GeoJSON → write."""
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"[embed] device={device}")
    graph = build_from_geojson(geojson_path)
    print(f"[embed] {graph.summary()}")

    data = graph.to_pyg().to(device)
    model = _load_model(ckpt_path, graph.num_features, device)

    with torch.no_grad():
        emb = model.encode(data.x, data.edge_index).cpu().numpy()
    print(f"[embed] embeddings shape: {emb.shape}")

    # Save raw embeddings parquet
    emb_df = pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])
    emb_df["token_id"] = graph.token_ids
    emb_df.to_parquet(out_geojson.parent / "embeddings.parquet", index=False)

    # UMAP to 2D for visualisation
    import umap

    print("[embed] running UMAP")
    reducer = umap.UMAP(
        n_neighbors=umap_neighbours,
        min_dist=umap_min_dist,
        random_state=seed,
        n_components=2,
    )
    umap_xy = reducer.fit_transform(emb)

    # K-means cluster
    from sklearn.cluster import KMeans

    print(f"[embed] k-means k={n_clusters}")
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    clusters = km.fit_predict(emb)

    # ML misalignment score
    print(f"[embed] computing ml_misalignment_score (k={knn_k})")
    gj = json.loads(Path(geojson_path).read_text())
    features = gj["features"]
    posted = np.array(
        [f["properties"].get("posted_speed_kph") or 50 for f in features],
        dtype=np.int32,
    )
    ml_mis = ml_misalignment(emb, posted, k=knn_k)

    # Enrich GeoJSON
    for i, feat in enumerate(features):
        feat["properties"]["umap_x"] = float(umap_xy[i, 0])
        feat["properties"]["umap_y"] = float(umap_xy[i, 1])
        feat["properties"]["cluster"] = int(clusters[i])
        feat["properties"]["ml_misalignment_kph"] = float(ml_mis[i])

    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out_geojson.write_text(json.dumps(gj))
    print(f"[embed] wrote {out_geojson} ({out_geojson.stat().st_size/1e6:.1f} MB)")

    # Quick summary stats
    stats = {
        "n_tokens": len(features),
        "embed_dim": emb.shape[1],
        "n_clusters": n_clusters,
        "ml_misalignment_kph": {
            "mean": float(ml_mis.mean()),
            "median": float(np.median(ml_mis)),
            "p95": float(np.percentile(ml_mis, 95)),
            "max": float(ml_mis.max()),
            "share_ge_20": float((ml_mis >= 20).mean()),
        },
        "umap_x_range": [float(umap_xy[:, 0].min()), float(umap_xy[:, 0].max())],
        "umap_y_range": [float(umap_xy[:, 1].min()), float(umap_xy[:, 1].max())],
        "cluster_sizes": [int(c) for c in np.bincount(clusters)],
    }
    print(json.dumps(stats, indent=2))
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--geojson", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out-geojson", type=Path, required=True)
    p.add_argument("--n-clusters", type=int, default=12)
    p.add_argument("--knn-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    run(
        geojson_path=args.geojson,
        ckpt_path=args.ckpt,
        out_geojson=args.out_geojson,
        n_clusters=args.n_clusters,
        knn_k=args.knn_k,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
