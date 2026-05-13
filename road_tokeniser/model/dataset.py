"""Convert tokens.geojson into a torch_geometric Data object.

Node features per token = numerical geometric/policy features + one-hot
categorical (highway class). Edges have two types:

  1. **along-road**: consecutive tokens of the same OSM way (way_osmid +
     way_segment_index) are connected i↔i+1. Captures along-corridor context.
  2. **shares-junction**: tokens whose endpoints lie within `junction_eps_m`
     of each other share a graph edge. Captures network topology.

The output Data object has:
  - x:           [N, F]  node features (float32)
  - edge_index:  [2, E]  bidirectional edges (along-road + shares-junction)
  - edge_type:   [E]     0 = along-road, 1 = shares-junction
  - y_highway:   [N]     integer-encoded highway class (for downstream eval)
  - meta:        dict with feature names, highway-class vocab, token_ids
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Highway-class vocabulary in fixed order — index 0 is "other"/unknown
HIGHWAY_VOCAB = [
    "other",
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "pedestrian",
    "track",
    "road",
]
HIGHWAY_IDX = {c: i for i, c in enumerate(HIGHWAY_VOCAB)}

# Numerical feature names + (mean, std) normalisation constants tuned for
# typical urban+rural distributions. These are static so embeddings are
# comparable across runs.
NUMERIC_FEATURES: list[tuple[str, float, float]] = [
    # (key, mean, std)
    ("length_m", 25.0, 5.0),
    ("chord_m", 23.0, 6.0),
    ("sinuosity", 1.05, 0.15),
    ("mean_abs_curvature_rad_per_m", 0.02, 0.04),
    ("total_bearing_change_rad", 0.30, 0.50),
    ("dist_to_nearest_junction_m", 60.0, 80.0),
    ("vru_score", 0.30, 0.25),
    ("posted_speed_kph", 50.0, 25.0),
]

# Binary categorical features (already 0/1 in GeoJSON)
BINARY_FEATURES = [
    "oneway",
    "school_within_proximity",
    "crossing_within_proximity",
    "bus_stop_within_proximity",
]


@dataclass
class RoadGraph:
    """Container for the graph dataset + metadata. Methods convert to torch."""

    x: np.ndarray              # [N, F]
    edge_index: np.ndarray     # [2, E]
    edge_type: np.ndarray      # [E]
    y_highway: np.ndarray      # [N]
    token_ids: np.ndarray      # [N]
    feature_names: list[str]
    highway_vocab: list[str]
    n_along_road: int
    n_shares_junction: int

    def to_pyg(self):
        """Return a torch_geometric.data.Data object."""
        import torch
        from torch_geometric.data import Data

        return Data(
            x=torch.from_numpy(self.x).float(),
            edge_index=torch.from_numpy(self.edge_index).long(),
            edge_type=torch.from_numpy(self.edge_type).long(),
            y_highway=torch.from_numpy(self.y_highway).long(),
            token_ids=torch.from_numpy(self.token_ids).long(),
        )

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def num_features(self) -> int:
        return self.x.shape[1]

    def summary(self) -> str:
        return (
            f"RoadGraph(N={self.num_nodes}, F={self.num_features}, "
            f"E={self.edge_index.shape[1]} "
            f"[{self.n_along_road} along-road, {self.n_shares_junction} junction])"
        )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _numeric_vec(props: dict[str, Any]) -> np.ndarray:
    out = np.zeros(len(NUMERIC_FEATURES), dtype=np.float32)
    for i, (k, mu, sd) in enumerate(NUMERIC_FEATURES):
        v = props.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out[i] = 0.0
        else:
            out[i] = (float(v) - mu) / sd
    return out


def _binary_vec(props: dict[str, Any]) -> np.ndarray:
    out = np.zeros(len(BINARY_FEATURES), dtype=np.float32)
    for i, k in enumerate(BINARY_FEATURES):
        v = props.get(k)
        out[i] = 1.0 if v else 0.0
    return out


def _highway_onehot(highway: str | None) -> np.ndarray:
    out = np.zeros(len(HIGHWAY_VOCAB), dtype=np.float32)
    idx = HIGHWAY_IDX.get(highway, 0) if highway else 0
    out[idx] = 1.0
    return out


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------


def _along_road_edges(features: list[dict]) -> list[tuple[int, int]]:
    """Consecutive segments of the same OSM way are connected i↔i+1."""
    # group by way_osmid, sort by way_segment_index
    by_way: dict[Any, list[tuple[int, int]]] = {}  # way_id → [(seg_idx, node_idx)]
    for node_idx, feat in enumerate(features):
        p = feat["properties"]
        wid = p.get("way_osmid")
        if wid is None:
            continue
        by_way.setdefault(wid, []).append((p.get("way_segment_index", 0), node_idx))

    edges: list[tuple[int, int]] = []
    for items in by_way.values():
        items.sort()
        for (_, a), (_, b) in zip(items[:-1], items[1:]):
            edges.append((a, b))
            edges.append((b, a))
    return edges


def _endpoint_coords(feature: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    coords = feature["geometry"]["coordinates"]
    return tuple(coords[0]), tuple(coords[-1])


def _shares_junction_edges(features: list[dict], eps_deg: float) -> list[tuple[int, int]]:
    """Tokens whose endpoints are within `eps_deg` degrees of each other.

    We work in WGS84 lat/lon and use a small epsilon (~5 m at Cambridge
    latitudes ≈ 4.5e-5 deg). Junctions cluster the endpoints of multiple
    tokens — we connect any pair of tokens that share an endpoint location.
    """
    # Bucket endpoints by snapped coordinate
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, feat in enumerate(features):
        start, end = _endpoint_coords(feat)
        for x, y in (start, end):
            key = (round(x / eps_deg), round(y / eps_deg))
            buckets.setdefault(key, []).append(idx)

    edge_set: set[tuple[int, int]] = set()
    for nodes in buckets.values():
        # Deduplicate within a bucket (a token contributes both its start and end)
        uniq = list(dict.fromkeys(nodes))
        if len(uniq) < 2:
            continue
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = uniq[i], uniq[j]
                if a == b:
                    continue
                edge_set.add((a, b))
                edge_set.add((b, a))
    return list(edge_set)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_from_geojson(
    geojson_path: Path | str,
    *,
    junction_eps_m: float = 5.0,
) -> RoadGraph:
    """Build a RoadGraph from a tokens.geojson produced by `rt-tokenise`."""
    gj = json.loads(Path(geojson_path).read_text())
    features = list(gj.get("features", []))
    if not features:
        raise ValueError(f"{geojson_path}: no features")

    # Node features
    n = len(features)
    n_features = (
        len(NUMERIC_FEATURES) + len(BINARY_FEATURES) + len(HIGHWAY_VOCAB)
    )
    x = np.zeros((n, n_features), dtype=np.float32)
    y_highway = np.zeros(n, dtype=np.int64)
    token_ids = np.zeros(n, dtype=np.int64)

    feature_names = (
        [k for k, _, _ in NUMERIC_FEATURES]
        + BINARY_FEATURES
        + [f"highway={c}" for c in HIGHWAY_VOCAB]
    )

    for i, feat in enumerate(features):
        p = feat["properties"]
        token_ids[i] = int(p.get("token_id", i))
        x[i, : len(NUMERIC_FEATURES)] = _numeric_vec(p)
        x[
            i,
            len(NUMERIC_FEATURES) : len(NUMERIC_FEATURES) + len(BINARY_FEATURES),
        ] = _binary_vec(p)
        x[
            i,
            len(NUMERIC_FEATURES) + len(BINARY_FEATURES) :,
        ] = _highway_onehot(p.get("highway"))
        y_highway[i] = HIGHWAY_IDX.get(p.get("highway") or "other", 0)

    # Edges
    # Approx: 1 deg lat ≈ 111 km, so 5 m ≈ 4.5e-5 deg
    eps_deg = junction_eps_m / 111_000.0

    along = _along_road_edges(features)
    junction = _shares_junction_edges(features, eps_deg)

    if not along and not junction:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros(0, dtype=np.int64)
    else:
        # Dedup edges across the two types (keep along-road if both present)
        along_set = set(along)
        junction_only = [e for e in junction if e not in along_set]
        all_edges = along + junction_only
        edge_index = np.array(all_edges, dtype=np.int64).T  # [2, E]
        edge_type = np.concatenate(
            [
                np.zeros(len(along), dtype=np.int64),
                np.ones(len(junction_only), dtype=np.int64),
            ]
        )

    return RoadGraph(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y_highway=y_highway,
        token_ids=token_ids,
        feature_names=feature_names,
        highway_vocab=HIGHWAY_VOCAB,
        n_along_road=len(along),
        n_shares_junction=edge_index.shape[1] - len(along) if edge_index.size else 0,
    )
