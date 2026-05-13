# road-tokeniser

> Where are posted speed limits inconsistent with Safe System principles?

A pipeline + visualisation that scores every ~25 m of road for how far its posted speed limit is from the speed a Safe-System assessment would recommend — given the road's geometry, function, and vulnerable-road-user exposure.

Entry for the **AI for Safer Roads Innovation Challenge** (Asian Development Bank × World Bank Development Impact Group × AI for Good × ITU, supported by JFPR and HLTF).

## What it does

1. Fetches road geometry from OpenStreetMap for a chosen bounding box.
2. Splits each road into ~25 m **tokens** (boundaries forced at every junction).
3. Computes geometric features per token (length, sinuosity, curvature, bearing change, junction proximity).
4. Attaches OSM `maxspeed` tags + country-default fallbacks for posted speed.
5. Attaches a **Vulnerable Road User (VRU)** exposure proxy from OSM amenities (schools, pedestrian crossings, bus stops; more layers in Phase B).
6. Spatial-joins crash records (UK STATS19 or NZ Waka Kotahi CAS), severity-weighted, where available.
7. Applies a **Safe System** rule engine to derive a recommended speed limit per token.
8. Flags segments where the posted limit is **misaligned** with the Safe System recommendation, prioritised by VRU exposure and crash history.
9. Renders the result as an interactive Leaflet webapp with click-to-explain feature breakdown.

Phase A (current) is fully deterministic — the rule engine ships day 1.
Phase B (planned) replaces the rule engine with an **unsupervised foundation model** (Graph Transformer + self-supervised pretraining on geometry) that flags segments whose posted limit is inconsistent with the *natural* posted-limit distribution for geometrically similar segments. See `docs/PLAN.md`.

## Headline results so far

| Site | Bbox | Tokens | Crashes joined | % road-km flagged ≥20 km/h |
|---|---|---|---|---|
| Cambridge UK (full) | `(0.10, 52.18, 0.16, 52.22)` | 12,008 | 803 | ~30% (urban primaries near schools dominate) |
| Wellington CBD NZ | `(174.770, -41.300, 174.795, -41.275)` | 3,472 | 10,617 | ~47% |
| Wellington + Hutt + SH1 NZ | `(174.70, -41.40, 175.20, -41.10)` | 141,978 | 68,862 | (scale-test only; too large for browser) |

Top-priority segments in Wellington CBD are dominated by **OSM tagging errors** (e.g. Cambridge Terrace tagged at 80 km/h in CBD), which is itself a useful diagnostic — the pipeline catches both true misalignment and bad tag data.

## Quickstart

```bash
# Setup (uses uv — fast Python package manager)
cd ~/Documents/road-tokeniser
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Download crash data — cached in data/raw/, idempotent
rt-fetch --site nz
rt-fetch --site uk

# Run on Cambridge UK
rt-tokenise --bbox 0.10,52.18,0.16,52.22 --site uk --out webapp/tokens.geojson

# Or Wellington CBD NZ (smaller demo)
rt-tokenise --bbox 174.770,-41.300,174.795,-41.275 --site nz --out webapp/tokens.geojson

# View in the browser
cd webapp && python -m http.server 8000
# → open http://localhost:8000
```

CLI:
- `rt-fetch [--site uk|nz|both] [--force]` — downloads crash data (~250 MB UK, ~270 MB NZ) into `data/raw/`, idempotent. Writes a SHA-256 manifest at `data/raw/manifest.json`.
- `rt-tokenise --bbox W,S,E,N --site uk|nz|generic --out tokens.geojson` — full pipeline.

Run `pytest` to execute the 23 unit tests on the rule engine.

## The Safe System decision tree

Per token, the rule engine fires the first matching rule and returns a recommended speed in km/h. Thresholds are tunable in `rules/safe_system.yml` — no code change needed to re-policy.

| Order | Rule | Trigger | Cap |
|---|---|---|---|
| 1 | `motorway_passthrough` | highway = motorway / motorway_link | trust posted (design speed) |
| 2 | `vru_high` | `vru_score ≥ 0.2` | **30** (pedestrian mix) |
| 3 | `vru_class` | highway = residential, living_street, service, pedestrian | **30** (pedestrian mix) |
| 4 | `junction_proximate` | within 30 m of a junction node | **50** (side-impact regime) |
| 5 | `high_curvature` | curvature ≥ 0.05 rad/m and posted ≥ 80 | **60** |
| 6 | `rural_arterial_undivided` | highway = trunk/primary and not oneway | **70** (head-on regime) |
| 7 | `rural_arterial_divided` | highway = trunk/primary and oneway (proxy for divided) | **90** |
| 8 | `secondary_tertiary` | highway = secondary/tertiary | **60** |
| 9 | `minor_road` | highway = unclassified/road/track | **50** |
| 10 | `fallthrough` | none of the above | trust posted (no opinion) |

**Output is clamped to ≤ posted** for all non-motorway rules. The challenge brief is about where limits are too high — we never recommend an increase. The `safe_system_rule` field in the output preserves which rule almost fired, so an analyst can see the rationale even when the recommendation matches posted.

## Data sources

| Layer | Source | Licence |
|---|---|---|
| Road geometry + tags | OpenStreetMap via OSMnx 2.x | ODbL 1.0 |
| UK crashes | UK DfT STATS19, 5-year rolling | Open Government Licence v3.0 |
| NZ crashes | Waka Kotahi Crash Analysis System (CAS) | CC-BY 4.0 |

Attribution must be preserved in any derivative work or publication. The webapp footer carries the required strings.

## Layout

```
road-tokeniser/
├── road_tokeniser/        # Python package
│   ├── safe_system.py     #   Pure-Python rule engine, no I/O deps
│   ├── fetch_data.py      #   CAS + STATS19 download + manifest
│   └── tokenise.py        #   End-to-end pipeline
├── rules/
│   └── safe_system.yml    # Tunable policy thresholds
├── tests/
│   └── test_safe_system.py
├── webapp/
│   ├── index.html         # Single-file Leaflet viewer
│   └── tokens.geojson     # Generated; gitignored
├── notebooks/             # Pipeline-stage exploration (Phase A: empty)
├── docs/
│   └── PLAN.md            # Full architectural plan including Phase B
├── data/raw/              # Crash CSVs (gitignored)
└── pyproject.toml
```

## Status

| Phase | Status |
|---|---|
| A — pipeline + rule engine + viz | ✅ ships a defensible answer |
| B — unsupervised foundation model + zero-shot transfer | 📋 planned, see `docs/PLAN.md` |
| C — APAC scale demo + crash-rate validation | 📋 planned |

## Licence

MIT for the code in this repository. Data attribution requirements above must be preserved in any derivative use.
