# road-tokeniser

Road-segment tokenisation + Safe System speed-limit misalignment detection.

Entry for the **AI for Safer Roads Innovation Challenge** (ADB × World Bank Development Impact Group × AI for Good × ITU).

## What this does

1. Fetches road geometry from OpenStreetMap for a chosen area.
2. Splits each road into ~25 m "tokens", forcing boundaries at every junction.
3. Computes geometric features per token (length, sinuosity, curvature, bearing change, junction proximity).
4. Attaches OSM `maxspeed` tags + country-default fallbacks for posted speed.
5. Attaches a Vulnerable Road User (VRU) exposure proxy from OSM amenities (schools, hospitals, markets, crossings, bus stops).
6. Spatial-joins crash records (UK STATS19 or NZ Waka Kotahi CAS) where available, severity-weighted.
7. Applies a **Safe System** rule engine to derive a recommended speed limit per token.
8. Flags segments where the posted limit is misaligned with Safe System recommendation, prioritised by VRU exposure and crash history.
9. Renders the result as an interactive Leaflet webapp.

The Safe System rule engine is the **Phase A** deliverable — fully deterministic, ships day 1. A Graph-Transformer foundation model with self-supervised pretraining is **Phase B** (separate branch).

## Quickstart

```bash
# Setup (uses uv — fast Python package manager)
cd ~/Documents/road-tokeniser
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Download crash data (cached in data/raw/, idempotent)
rt-fetch --site nz
rt-fetch --site uk

# Tokenise an area
rt-tokenise --bbox 174.70,-41.40,175.20,-41.10 --site nz --out webapp/tokens.geojson
# or Cambridge UK
rt-tokenise --bbox 0.10,52.18,0.16,52.22 --site uk --out webapp/tokens.geojson

# View
cd webapp && python -m http.server 8000
# open http://localhost:8000
```

## Sites supported in Phase A

| Site | Bbox `(W, S, E, N)` | Crash data | Use |
|---|---|---|---|
| Cambridge, UK | `(0.10, 52.18, 0.16, 52.22)` | DfT STATS19 (OGL v3.0) | Urban dev environment |
| Wellington + SH1, NZ | `(174.70, -41.40, 175.20, -41.10)` | Waka Kotahi CAS (CC-BY 4.0) | Demo: urban + rural-highway mix |

## Data sources

- **Road geometry**: © OpenStreetMap contributors, ODbL.
- **NZ crashes**: Waka Kotahi (NZ Transport Agency) Crash Analysis System, CC-BY 4.0.
- **UK crashes**: UK Department for Transport STATS19, Open Government Licence v3.0.

## Status

- Phase A: in progress.
- Phase B (foundation model): not started.

## Licence

MIT (code). Data attribution requirements above must be preserved in any derivative work.
