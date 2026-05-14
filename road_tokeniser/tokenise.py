"""End-to-end tokenisation pipeline.

OSM extract  →  ~25 m road-segment tokens  →  geometric + OSM features  →
posted speed + VRU proxies + crash join  →  Safe System rule engine  →
GeoJSON output for the Leaflet viewer.

CLI:
    rt-tokenise --bbox W,S,E,N --site nz --out webapp/tokens.geojson
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import substring

from road_tokeniser.safe_system import (
    Rules,
    country_default_speed,
    load_rules,
    safe_system_speed,
    vru_score as compute_vru_score,
)

WGS84 = 4326

# OSM amenity / highway tags we treat as VRU proxies
VRU_FEATURE_QUERIES = {
    "school": {"amenity": ["school", "kindergarten"]},
    "crossing": {"highway": ["crossing"], "footway": ["crossing"]},
    "bus_stop": {"highway": ["bus_stop"], "public_transport": ["platform", "stop_position"]},
}


# ---------------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------------


def utm_epsg_for_bbox(bbox: tuple[float, float, float, float]) -> int:
    """Pick a sensible UTM zone for metric ops on a bbox `(W, S, E, N)`."""
    w, s, e, n = bbox
    lon = (w + e) / 2
    lat = (s + n) / 2
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be W,S,E,N")
    return tuple(parts)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# OSM fetch + tokenisation
# ---------------------------------------------------------------------------


def _first_value(v):
    """OSM tags can be a list when multiple tags exist on the way; pick the first."""
    if isinstance(v, list):
        return v[0] if v else None
    return v


def _parse_oneway(v) -> bool:
    """OSM `oneway` is a string ('yes'/'no'/'-1') or a list when ways merged.

    `bool([False, False])` is `True` — that bug silently flips two-way roads
    into 'divided'. Always unwrap the list first, then test against truthy
    string values.
    """
    v = _first_value(v)
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"yes", "true", "1", "-1"}


def split_line(line: LineString, length_m: float, line_length_m: float) -> list[LineString]:
    """Split a LineString (already in a metric CRS) into ~length_m pieces.

    The last piece can be longer than `length_m` if remainder < length_m/2 to
    avoid stub pieces, otherwise it becomes its own piece.
    """
    if line_length_m <= length_m * 1.5:
        return [line]

    n_full = max(1, int(line_length_m // length_m))
    # If the remainder of a clean division is small, fold it into the last piece
    remainder = line_length_m - n_full * length_m
    if remainder < length_m * 0.5:
        n_pieces = n_full
    else:
        n_pieces = n_full + 1

    out: list[LineString] = []
    for i in range(n_pieces):
        a = i / n_pieces
        b = (i + 1) / n_pieces
        out.append(substring(line, a, b, normalized=True))
    return out


def tokenise_edges(edges_metric: gpd.GeoDataFrame, length_m: float) -> gpd.GeoDataFrame:
    """Split each edge into ~length_m tokens. Returns one row per token."""
    rows: list[dict] = []
    for _, edge in edges_metric.iterrows():
        line: LineString = edge.geometry
        if line is None or line.is_empty or not isinstance(line, LineString):
            continue
        L = line.length
        pieces = split_line(line, length_m, L)
        for i, p in enumerate(pieces):
            rows.append(
                {
                    "geometry": p,
                    "way_osmid": _first_value(edge.get("osmid")),
                    "name": _first_value(edge.get("name")),
                    "highway": _first_value(edge.get("highway")),
                    "oneway": _parse_oneway(edge.get("oneway")),
                    "lanes": _first_value(edge.get("lanes")),
                    "maxspeed_raw": _first_value(edge.get("maxspeed")),
                    "surface": _first_value(edge.get("surface")),
                    "way_segment_index": i,
                    "way_segment_count": len(pieces),
                }
            )
    gdf = gpd.GeoDataFrame(rows, crs=edges_metric.crs)
    gdf["token_id"] = np.arange(len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# Geometric features
# ---------------------------------------------------------------------------


def _bearing(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Bearing in radians for a 2D vector (no Earth curvature — we are in UTM)."""
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def compute_geom_features(tokens_metric: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add length_m, chord_m, sinuosity, mean_abs_curvature, total_bearing_change.

    All in radians / metres (we're in a metric CRS).
    """
    lengths = tokens_metric.geometry.length.values
    chords = np.array(
        [
            Point(g.coords[0]).distance(Point(g.coords[-1])) if len(g.coords) >= 2 else 0.0
            for g in tokens_metric.geometry
        ]
    )
    sinuosity = np.where(chords > 0, lengths / np.maximum(chords, 1e-6), 1.0)

    bearing_changes: list[float] = []
    mean_curvatures: list[float] = []
    for g in tokens_metric.geometry:
        coords = list(g.coords)
        if len(coords) < 3:
            bearing_changes.append(0.0)
            mean_curvatures.append(0.0)
            continue
        bearings = [_bearing(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
        diffs = []
        for a, b in zip(bearings[:-1], bearings[1:]):
            d = b - a
            # wrap to [-pi, pi]
            d = (d + math.pi) % (2 * math.pi) - math.pi
            diffs.append(abs(d))
        total = float(sum(diffs))
        bearing_changes.append(total)
        # Mean curvature = total absolute angle change per unit length
        L = g.length
        mean_curvatures.append(total / max(L, 1e-6))

    out = tokens_metric.copy()
    out["length_m"] = lengths
    out["chord_m"] = chords
    out["sinuosity"] = sinuosity
    out["total_bearing_change_rad"] = bearing_changes
    out["mean_abs_curvature_rad_per_m"] = mean_curvatures
    return out


def attach_junction_proximity(
    tokens_metric: gpd.GeoDataFrame, junctions_metric: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Distance from each token centroid to the nearest OSM junction node."""
    centroids = tokens_metric.copy()
    centroids["geometry"] = centroids.geometry.centroid
    joined = gpd.sjoin_nearest(
        centroids[["token_id", "geometry"]],
        junctions_metric[["geometry"]],
        how="left",
        distance_col="dist_to_nearest_junction_m",
    )
    # sjoin_nearest may produce duplicates if there are ties; keep first per token
    joined = joined.drop_duplicates(subset=["token_id"], keep="first")
    out = tokens_metric.merge(
        joined[["token_id", "dist_to_nearest_junction_m"]], on="token_id", how="left"
    )
    return out


# ---------------------------------------------------------------------------
# Posted speed
# ---------------------------------------------------------------------------


_KMH_PER_MPH = 1.609344


def parse_maxspeed(raw, country: str, highway: str | None, rules: Rules) -> int:
    """Parse OSM `maxspeed` to km/h; fallback to country default by `highway` class.

    Accepts forms like '50', '50 mph', '30 kph', 'walk', 'none', or a list.
    """
    if isinstance(raw, list):
        # Multi-valued maxspeed — pick the smallest (conservative)
        parsed = [parse_maxspeed(x, country, highway, rules) for x in raw]
        return min(parsed) if parsed else country_default_speed(highway, country, rules)
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return country_default_speed(highway, country, rules)
    s = str(raw).strip().lower()
    if s in {"none", "signals", "walk", "no", ""}:
        return country_default_speed(highway, country, rules)
    try:
        # 'mph' suffix indicates imperial
        if "mph" in s:
            num = float(s.replace("mph", "").strip())
            return int(round(num * _KMH_PER_MPH))
        if "kph" in s or "km/h" in s:
            num = float(s.replace("kph", "").replace("km/h", "").strip())
            return int(round(num))
        # Bare number → country convention (UK = mph, NZ = kph, generic = kph)
        num = float(s)
        if country == "uk":
            return int(round(num * _KMH_PER_MPH))
        return int(round(num))
    except ValueError:
        return country_default_speed(highway, country, rules)


def attach_posted_speed(
    tokens: gpd.GeoDataFrame, country: str, rules: Rules
) -> gpd.GeoDataFrame:
    out = tokens.copy()
    out["posted_speed_kph"] = [
        parse_maxspeed(r, country, h, rules)
        for r, h in zip(out["maxspeed_raw"], out["highway"])
    ]
    return out


# ---------------------------------------------------------------------------
# VRU proxies
# ---------------------------------------------------------------------------


def fetch_vru_amenities(
    bbox: tuple[float, float, float, float],
) -> dict[str, gpd.GeoDataFrame]:
    """Fetch OSM amenity layers we use as VRU proxies. Returns {kind: GeoDataFrame}."""
    out: dict[str, gpd.GeoDataFrame] = {}
    for kind, tags in VRU_FEATURE_QUERIES.items():
        try:
            gdf = ox.features_from_bbox(bbox=bbox, tags=tags)
            # Reduce polygon amenities (e.g. school grounds) to centroids
            gdf = gdf.copy()
            gdf["geometry"] = gdf.geometry.representative_point()
            out[kind] = gdf[["geometry"]]
            print(f"[vru] {kind}: {len(gdf)} features", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[vru] {kind}: none ({type(e).__name__})", file=sys.stderr)
            out[kind] = gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{WGS84}")
    return out


def _any_within(tokens_metric: gpd.GeoDataFrame, points_metric: gpd.GeoDataFrame, buffer_m: float) -> np.ndarray:
    """For each token, is there any point of `points_metric` within `buffer_m` metres of its centroid?"""
    if len(points_metric) == 0:
        return np.zeros(len(tokens_metric), dtype=bool)
    centroids = tokens_metric.geometry.centroid
    buffered = gpd.GeoDataFrame(
        {"__tok_idx": np.arange(len(centroids))},
        geometry=centroids.buffer(buffer_m),
        crs=tokens_metric.crs,
    )
    # Clean the right side — OSM features have a MultiIndex which breaks sjoin column naming.
    pts = points_metric[["geometry"]].reset_index(drop=True)
    pts["__pt_idx"] = np.arange(len(pts))
    joined = gpd.sjoin(buffered, pts, how="inner", predicate="intersects")
    flag = np.zeros(len(tokens_metric), dtype=bool)
    if len(joined):
        hit_idx = joined["__tok_idx"].unique().astype(int)
        flag[hit_idx] = True
    return flag


def attach_vru_proxies(
    tokens_metric: gpd.GeoDataFrame,
    amenities_metric: dict[str, gpd.GeoDataFrame],
    rules: Rules,
) -> gpd.GeoDataFrame:
    out = tokens_metric.copy()
    out["school_within_proximity"] = _any_within(
        out, amenities_metric["school"], rules.thresholds["school_proximity_m"]
    )
    out["crossing_within_proximity"] = _any_within(
        out, amenities_metric["crossing"], rules.thresholds["crossing_proximity_m"]
    )
    out["bus_stop_within_proximity"] = _any_within(
        out, amenities_metric["bus_stop"], rules.thresholds["bus_stop_proximity_m"]
    )
    out["vru_score"] = [
        compute_vru_score(row.to_dict(), rules) for _, row in out.iterrows()
    ]
    return out


# ---------------------------------------------------------------------------
# Crash spatial join
# ---------------------------------------------------------------------------


def _check_required_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    """Fail loudly if a CSV schema drifted out from under us."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source}: required columns missing: {missing}. "
            f"Got columns: {list(df.columns)[:20]}..."
        )


def load_crashes_uk(csv_path: Path, bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    w, s, e, n = bbox
    # Read header first to defend against case / naming drift between STATS19
    # publication years (older snapshots use Title Case).
    header = pd.read_csv(csv_path, nrows=0).columns
    rename = {}
    for c in header:
        lc = c.lower()
        if lc in {"longitude", "long"}:
            rename[c] = "longitude"
        elif lc == "latitude":
            rename[c] = "latitude"
        elif lc in {"collision_severity", "accident_severity"}:
            rename[c] = "collision_severity"
        elif lc in {"collision_year", "accident_year"}:
            rename[c] = "collision_year"
        elif lc in {"collision_index", "accident_index"}:
            rename[c] = "collision_index"
    df = pd.read_csv(csv_path, low_memory=False).rename(columns=rename)
    _check_required_columns(
        df, ["longitude", "latitude", "collision_severity", "collision_index"], "STATS19"
    )
    df = df[(df.longitude.between(w, e)) & (df.latitude.between(s, n))].copy()
    df = df.dropna(subset=["longitude", "latitude"])
    # De-duplicate at source — same collision_index should never appear twice.
    df = df.drop_duplicates(subset=["collision_index"], keep="first")
    weights = {1: 5, 2: 3, 3: 1}
    df["severity_weight"] = df["collision_severity"].map(weights).fillna(1)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs=f"EPSG:{WGS84}",
    )
    keep = ["geometry", "severity_weight", "collision_index"]
    if "collision_year" in gdf.columns:
        keep.append("collision_year")
    return gdf[keep].rename(columns={"collision_index": "crash_id"})


def load_crashes_nz(csv_path: Path, bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    w, s, e, n = bbox
    # CAS exports columns 'X' and 'Y' for WGS84 when spatialRefId=4326 is set.
    # Defensive: accept 'longitude'/'latitude' aliases too.
    header = pd.read_csv(csv_path, nrows=0).columns
    rename = {}
    for c in header:
        lc = c.lower()
        if lc == "longitude" or c == "X":
            rename[c] = "X"
        elif lc == "latitude" or c == "Y":
            rename[c] = "Y"
    df = pd.read_csv(csv_path, low_memory=False).rename(columns=rename)
    _check_required_columns(df, ["X", "Y", "OBJECTID"], "CAS")
    df = df[(df.X.between(w, e)) & (df.Y.between(s, n))].copy()
    df = df.dropna(subset=["X", "Y"])
    df = df.drop_duplicates(subset=["OBJECTID"], keep="first")
    for col in ("fatalCount", "seriousInjuryCount", "minorInjuryCount"):
        if col not in df.columns:
            df[col] = 0
    df["severity_weight"] = (
        5 * df["fatalCount"].fillna(0)
        + 3 * df["seriousInjuryCount"].fillna(0)
        + 1 * df["minorInjuryCount"].fillna(0)
    ).clip(lower=1)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.X, df.Y),
        crs=f"EPSG:{WGS84}",
    )
    keep = ["geometry", "severity_weight", "OBJECTID"]
    if "crashYear" in gdf.columns:
        keep.append("crashYear")
    return gdf[keep].rename(columns={"OBJECTID": "crash_id"})


def attach_crashes(
    tokens_metric: gpd.GeoDataFrame,
    crashes_metric: gpd.GeoDataFrame,
    buffer_m: float = 15.0,
) -> gpd.GeoDataFrame:
    """Assign each crash to its single nearest token within `buffer_m`.

    Crashes at junctions are otherwise double-counted (a crash at a four-way
    intersection lies inside four token buffers). We solve this with
    ``sjoin_nearest`` instead of a buffered ``sjoin``, so each crash maps to
    exactly one token — the geometrically closest one within the cutoff.
    """
    out = tokens_metric.reset_index(drop=True).copy()
    out["__tok_idx"] = np.arange(len(out))

    if len(crashes_metric) == 0:
        out["crash_count"] = 0
        out["crash_score"] = 0.0
        return out.drop(columns=["__tok_idx"])

    cm = crashes_metric.reset_index(drop=True)[
        [c for c in ("geometry", "severity_weight", "crash_id") if c in crashes_metric.columns]
    ].copy()

    # For each crash, find the nearest token within buffer_m. Crashes farther
    # than that are dropped (filtered out via distance_col + cutoff).
    nearest = gpd.sjoin_nearest(
        cm,
        out[["__tok_idx", "geometry"]],
        how="left",
        max_distance=buffer_m,
        distance_col="__dist_m",
    )
    # Drop crashes that had no token within range
    nearest = nearest.dropna(subset=["__tok_idx"])
    # If ties produced duplicate rows for the same crash, keep the closest
    if "crash_id" in nearest.columns:
        nearest = nearest.sort_values("__dist_m").drop_duplicates(subset=["crash_id"], keep="first")

    agg = nearest.groupby("__tok_idx").agg(
        crash_count=("severity_weight", "size"),
        crash_score=("severity_weight", "sum"),
    )
    out["crash_count"] = out["__tok_idx"].map(agg["crash_count"]).fillna(0).astype(int)
    out["crash_score"] = out["__tok_idx"].map(agg["crash_score"]).fillna(0.0).astype(float)
    return out.drop(columns=["__tok_idx"])


# ---------------------------------------------------------------------------
# Apply Safe System rules
# ---------------------------------------------------------------------------


def apply_rule_engine(tokens: gpd.GeoDataFrame, rules: Rules) -> gpd.GeoDataFrame:
    out = tokens.copy()
    safe_speeds: list[int] = []
    rule_ids: list[str] = []
    for _, row in out.iterrows():
        s, rid = safe_system_speed(row.to_dict(), rules)
        safe_speeds.append(s)
        rule_ids.append(rid)
    out["safe_system_speed_kph"] = safe_speeds
    out["safe_system_rule"] = rule_ids
    out["misalignment_kph"] = out["posted_speed_kph"].fillna(0).astype(int) - out["safe_system_speed_kph"]
    # priority score: clamp(mis,0,50)/50 * (0.5 + 0.5*vru)
    mis_pos = out["misalignment_kph"].clip(lower=0, upper=50)
    out["priority_score"] = (mis_pos / 50.0) * (0.5 + 0.5 * out["vru_score"].fillna(0))
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


_OUTPUT_COLS = [
    "token_id",
    "name",
    "highway",
    "oneway",
    "lanes",
    "surface",
    "way_osmid",
    "way_segment_index",
    "way_segment_count",
    "length_m",
    "chord_m",
    "sinuosity",
    "total_bearing_change_rad",
    "mean_abs_curvature_rad_per_m",
    "dist_to_nearest_junction_m",
    "school_within_proximity",
    "crossing_within_proximity",
    "bus_stop_within_proximity",
    "vru_score",
    "maxspeed_raw",
    "posted_speed_kph",
    "crash_count",
    "crash_score",
    "safe_system_speed_kph",
    "safe_system_rule",
    "misalignment_kph",
    "priority_score",
    "geometry",
]


def write_geojson(tokens_4326: gpd.GeoDataFrame, out_path: Path) -> None:
    cols = [c for c in _OUTPUT_COLS if c in tokens_4326.columns]
    gdf = tokens_4326[cols].copy()
    # Make booleans JSON-friendly and round floats for smaller files
    for c in ["school_within_proximity", "crossing_within_proximity", "bus_stop_within_proximity", "oneway"]:
        if c in gdf:
            gdf[c] = gdf[c].astype(bool)
    for c in [
        "length_m",
        "chord_m",
        "sinuosity",
        "total_bearing_change_rad",
        "mean_abs_curvature_rad_per_m",
        "dist_to_nearest_junction_m",
        "vru_score",
        "priority_score",
        "crash_score",
    ]:
        if c in gdf:
            gdf[c] = gdf[c].astype(float).round(4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"[out] wrote {len(gdf)} tokens to {out_path} ({out_path.stat().st_size/1e6:.1f} MB)", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _resolve_crash_csv(site: str) -> Path | None:
    root = Path(__file__).resolve().parent.parent
    if site == "uk":
        p = root / "data" / "raw" / "uk" / "stats19_collision_last_5_years.csv"
    elif site == "nz":
        p = root / "data" / "raw" / "nz" / "CAS_Data_public.csv"
    else:
        return None
    return p if p.exists() else None


def run(bbox, site, token_len, out, rules_path=None):
    t0 = time.time()
    rules = load_rules(rules_path) if rules_path else load_rules()
    utm = utm_epsg_for_bbox(bbox)

    # One Overpass round-trip — return both edges and nodes from a single graph.
    print(f"[osm] fetching network for bbox={bbox}", file=sys.stderr)
    G = ox.graph_from_bbox(bbox, network_type="drive", simplify=True, retain_all=False)
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True).reset_index(drop=False)
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False).reset_index(drop=False)
    print(f"[osm] {len(edges)} edges, {len(nodes)} nodes", file=sys.stderr)
    if len(edges) == 0:
        raise RuntimeError(f"OSM returned no drivable edges for bbox={bbox}")

    edges_metric = edges.to_crs(epsg=utm)
    nodes_metric = nodes.to_crs(epsg=utm)

    print(f"[tok] splitting into ~{token_len} m pieces", file=sys.stderr)
    tokens = tokenise_edges(edges_metric, length_m=token_len)
    print(f"[tok] {len(tokens)} tokens", file=sys.stderr)

    print("[feat] geometric features", file=sys.stderr)
    tokens = compute_geom_features(tokens)
    tokens = attach_junction_proximity(tokens, nodes_metric)

    print("[osm] fetching VRU amenities", file=sys.stderr)
    amenities = fetch_vru_amenities(bbox)
    amenities_metric = {k: v.to_crs(epsg=utm) for k, v in amenities.items()}
    tokens = attach_vru_proxies(tokens, amenities_metric, rules)

    tokens = attach_posted_speed(tokens, site, rules)

    crash_csv = _resolve_crash_csv(site)
    if crash_csv is None:
        print(f"[crash] no crash CSV for site={site}; setting counts to 0", file=sys.stderr)
        tokens["crash_count"] = 0
        tokens["crash_score"] = 0.0
    else:
        loader = load_crashes_uk if site == "uk" else load_crashes_nz
        crashes = loader(crash_csv, bbox).to_crs(epsg=utm)
        print(f"[crash] {len(crashes)} crashes in bbox", file=sys.stderr)
        tokens = attach_crashes(tokens, crashes, buffer_m=15.0)

    tokens = apply_rule_engine(tokens, rules)

    out_4326 = tokens.to_crs(epsg=WGS84)
    write_geojson(out_4326, Path(out))

    print(f"[done] {time.time()-t0:.1f}s wall", file=sys.stderr)
    return out_4326


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tokenise OSM roads and score Safe System misalignment.")
    p.add_argument("--bbox", type=parse_bbox, required=True, help="W,S,E,N in WGS84 (degrees)")
    p.add_argument("--site", choices=["uk", "nz", "ph", "generic"], default="generic")
    p.add_argument("--token-len", type=float, default=25.0, help="Target token length in metres")
    p.add_argument("--out", type=Path, required=True, help="Output GeoJSON path")
    p.add_argument("--rules", type=Path, default=None, help="Override rules YAML path")
    args = p.parse_args(list(argv) if argv is not None else None)

    run(args.bbox, args.site, args.token_len, args.out, args.rules)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
