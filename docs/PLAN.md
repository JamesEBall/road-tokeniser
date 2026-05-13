# Plan — Road Safety Foundation Model for Speed-Limit Misalignment (NZ)

## Context

Entry for the **AI for Safer Roads Innovation Challenge** (ADB × World Bank DIG × AI for Good × ITU).

Brief:
> How might we use AI and mobility data to determine where speed limits are **misaligned with real-world road conditions**, supporting evidence-based speed management across Asia and the Pacific?

Explicit: this is about whether the *posted limit is wrong*, not whether drivers exceed it.

**Approach: an unsupervised road-safety foundation model.** Pretrain a Graph Transformer on the geometry and graph structure of an entire road network using self-supervised objectives — no crash labels, no speed-limit labels during pretraining. Then probe the learned embedding space for speed-limit *consistency*: a segment whose posted limit sits far from the modal posted limit for geometrically similar segments is flagged misaligned.

Why this beats a rule engine for the challenge:
- Rule engines encode *what experts say* (Safe System matrix). They're brittle when OSM attributes are missing.
- Embeddings encode *what the road network's own labelling says*. A misalignment flagged by both is high-confidence.
- The backbone is reusable for sidewalk inference, crash-rate prediction, intersection classification — one model, many downstream tasks. This is the "foundation model" contribution.
- Pretrain on dense regions, transfer zero-shot to data-sparse DMCs — directly answers the brief's scalability ask.

## Approach in one paragraph

Split every road into ~25 m tokens. Each token gets a geometric feature vector (curvature, sinuosity, length, bearing change, junction proximity) plus categorical features from OSM (`highway` class, `surface`, lane count if present). A Graph Transformer encodes tokens with two graph relations — *along-road neighbour* and *shares-junction neighbour* — into 256-d embeddings. We pretrain with three self-supervised losses (masked geometry, masked OSM-tag reconstruction, neighbour-contrastive) on the entire NZ + UK road network. Downstream, we project each token to its embedding and ask: among the K nearest neighbours in embedding space, what is the modal posted speed limit? If the segment's actual posted limit differs from that modal by ≥10 km/h, flag it. Validate against STATS19 (UK) and CAS (NZ) crash rates — flagged segments should have higher crash rates than unflagged segments of the same posted limit.

## Why "foundation model" is honest, not marketing

- **Pretrain once, many tasks**: the same encoder serves misalignment detection, crash-rate regression, missing-attribute inference (sidewalk/median/control), and intersection typing. We will demonstrate at least two downstream tasks.
- **Self-supervised at scale**: no human labels in pretraining; objective is reconstruction of masked features and graph-neighbour similarity.
- **Transferable**: trained on NZ + UK geometry, evaluated zero-shot on a DMC city (Manila) for misalignment scoring — no Manila-specific training data needed.
- **Open-weight**: model checkpoint + code released so DMC analysts can deploy it themselves.

## Architecture

```
[OSM tokenised network]
        │
        ▼
[Geometry MLP]  +  [Categorical embeds]  +  [Laplacian + sequence positional encoding]
        │
        ▼
[Graph Transformer, 6 layers, d=256, MHA=8, FFN=1024]
   ↑ edges: along-road + shares-junction
        │
        ├──▶ Masked-geometry head (SSL pretrain)
        ├──▶ Masked-tag head (SSL pretrain)
        ├──▶ Neighbour-contrastive projection (SSL pretrain)
        │
        ▼     (frozen after pretraining)
[Token embeddings]
        │
        ├──▶ Misalignment scorer (k-NN posted-speed mode)
        ├──▶ Crash-rate probe (linear, fit on STATS19 + CAS)
        └──▶ Attribute-inference probe (linear, fit on dense-OSM tags)
```

Size: ~5–8 M params. Fits on a single consumer GPU. Pretraining on the entire NZ + UK network (~5 M tokens) takes O(hours) on a laptop GPU.

### Self-supervised objectives

1. **Masked geometry modelling.** Mask 15 % of a token's geometric features; predict from context. MSE loss. Forces the encoder to learn geometric regularities.
2. **Masked tag reconstruction.** Mask categorical OSM attributes (`highway`, `surface`, `lit`, `oneway`); predict via classification heads. Cross-entropy. Forces semantic clustering by road class.
3. **Neighbour-contrastive (InfoNCE).** Positive pair = two tokens on the same OSM way OR within 3 hops on the road graph. Negative = random distant tokens. Pulls topologically related tokens together in embedding space.

These three together produce embeddings that capture geometry, semantics, and topology — the three things misalignment depends on. Crucially, `maxspeed` is **never seen by the encoder during pretraining** — only used downstream for misalignment scoring and validation, so misalignment signal is not a learning shortcut.

### Downstream: misalignment scoring

For each token *t* with posted limit *s\_t*:

1. Find the K=50 nearest neighbours of *t* in embedding space (excluding *t* itself and tokens on the same road, to avoid trivial neighbours).
2. Compute the posted-speed distribution over those neighbours: `p(s | embedding≈t)`.
3. `misalignment_score(t) = |s_t − mode(p)| + λ × entropy_penalty`, where the entropy penalty downweights tokens whose neighbours disagree on speed (geometrically ambiguous → low confidence in any flag).
4. Sort segments descending by misalignment score → priority list.

This is fully unsupervised at training time — the misalignment scorer just reads off the embedding space.

### Validation against crash data (no training signal)

Held-out validation only — never seen in training:
- For NZ: bin tokens by `misalignment_score` quintile, compute CAS crash rate per token-year per bin. Hypothesis: highest-misalignment quintile has ≥2× the crash rate of the lowest, *controlling for posted speed*. If true → misalignment captures a real safety signal, not just labelling noise.
- For Cambridge UK: same with STATS19.

This is the falsifiable test. If misalignment-flagged segments don't show elevated crash rates, the foundation model is just inventing inconsistencies and we'd need to pivot back to rule-based.

## Data sources (all verified open and downloadable)

| Layer | Source | Licence | Verified URL | Status |
|---|---|---|---|---|
| Road geometry | OSM via OSMnx 2.1.0 | ODbL 1.0 | `graph_from_bbox(bbox, network_type='drive')` | ✅ |
| Posted speed prior | OSM `maxspeed` tag | ODbL 1.0 | (same as above) | ✅ |
| NZ crashes (validation only) | Waka Kotahi CAS | CC-BY 4.0 | `https://opendata.arcgis.com/api/v3/datasets/8d684f1841fa4dbea6afaefc8a1ba0fc_0/downloads/data?format=csv&spatialRefId=4326` (`CAS_Data_public.csv`, ~150 MB, last updated 2026-05-06) | ✅ tested live HEAD |
| UK crashes (validation only) | DfT STATS19 5-year rolling | OGL v3.0 | `https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-{collision,casualty,vehicle}-last-5-years.csv` | ✅ |
| VRU exposure (for downstream context) | OSM amenities (schools, crossings); WorldPop 100 m | ODbL + CC-BY 4.0 | OSMnx + WorldPop direct | ✅ |
| Gradient (optional Phase B) | Copernicus DEM 30 m | ESA free | Copernicus Data Space | optional |

Footer attribution string in webapp:
> Road geometry © OpenStreetMap contributors, ODbL. Crash data: Waka Kotahi CAS (CC-BY 4.0); UK DfT STATS19 (OGL v3.0).

## Geography

| Site | Purpose | Bbox `(W, S, E, N)` | Token count est. |
|---|---|---|---|
| **Cambridge, UK** | Dev environment, urban dense | `(0.10, 52.18, 0.16, 52.22)` | ~2.5 k |
| **Wellington region + SH1 corridor, NZ** | Demo deliverable: mixed urban + rural-highway misalignment | `(174.70, -41.40, 175.20, -41.10)` (Wellington city + Hutt Valley + SH1 north to Paekākāriki) | ~30 k |
| **All of NZ** | Pretraining corpus, Phase B | (country bbox) | ~1.5 M |
| **All of UK** | Pretraining corpus, Phase B | (country bbox) | ~3.5 M |
| **Makati / BGC, Manila** | Zero-shot transfer demo, Phase C | `(121.00, 14.53, 121.07, 14.57)` | ~3 k |

Why Wellington + SH1 for the headline demo: covers the two regimes where misalignment matters most — urban arterials with VRU presence (Wellington CBD, Lower Hutt) and rural state highway with 100 km/h legal limit on winding terrain (SH1 north of Wellington, well-known for fatigue and head-on crashes). CAS has ~10 years of geocoded crashes for this corridor.

## Phased deliverables

### Phase A — Tokenisation pipeline + visual baseline (this PoC)
Goal: tokenise OSM, attach features, render in webapp, **compute a simple rule-engine misalignment score as the baseline**. Ships before any ML training. The rule engine is also the comparison point the foundation model will need to beat.

Files under `road-tokeniser/`:

- `pyproject.toml` — deps: `osmnx>=2.1`, `geopandas`, `shapely`, `numpy`, `pandas`, `pyarrow`, `requests`, `pyyaml`
- `fetch_data.py` — downloads CAS (NZ) + STATS19 (UK) CSVs to `data/raw/`, idempotent, with `--site nz|uk|both` flag
- `safe_system.py` — rule baseline: Safe System matrix → `safe_system_speed(token_features) -> int` (rules in `rules/safe_system.yml`)
- `tokenise.py` — CLI: `python tokenise.py --bbox W,S,E,N --site nz --token-len 25 --out webapp/tokens.geojson`
  - `fetch_osm(bbox)`, `split_to_tokens(gdf, length_m=25)`, `compute_geom_features(token)`, `attach_maxspeed(tokens, country='nz')`, `attach_vru_proxies(tokens)`, `attach_crashes(tokens, crashes_df, buffer_m=15)`, `apply_safe_system(tokens)`, `compute_priority_score(tokens)`, `write_geojson(...)`
  - Spatial ops use local UTM auto-picked from bbox centroid; final output in EPSG:4326
- `webapp/index.html` — Leaflet 1.9 (CDN). Colour ramps: `posted_speed_kph` / `safe_system_speed_kph` / `rule_misalignment_kph` / `crash_score` / `priority_score`. VRU amenity overlay toggle. Hover popup with full feature dict.
- `rules/safe_system.yml` — Safe System policy matrix
- `README.md` — quickstart

### Phase B — Foundation model pretraining
Files under `road-tokeniser/model/`:

- `dataset.py` — builds a `torch_geometric` graph dataset from tokenised GeoJSON; supports streaming for country-scale.
- `encoder.py` — Graph Transformer backbone (GraphGPS-style). Pure PyTorch + `torch_geometric`. ~5–8 M params.
- `pretrain.py` — three-objective SSL trainer (masked geometry, masked tag, neighbour-contrastive). Logs to `runs/`. Saves checkpoints to `weights/`.
- `embed.py` — loads a checkpoint, emits a `token_id → 256-d vector` parquet for downstream use.
- Hardware target: single laptop/desktop GPU (RTX 3060 / M-series Metal acceptable). Training time budget: < 24 h for combined NZ + UK pretraining.

### Phase C — Downstream misalignment scoring + validation
- `score_misalignment.py` — k-NN over embedding space → `ml_misalignment_score` per token. Writes back into `tokens.geojson`.
- `validate.py` — bins tokens by `ml_misalignment_score` quintile, computes crash-rate-per-token-year per bin per posted-speed cohort. Outputs a markdown report `reports/validation.md` with the headline number: relative-risk of top-misalignment quintile vs bottom, conditional on posted speed.
- Webapp adds `ml_misalignment_score` ramp + side-by-side comparison with `rule_misalignment_kph`.

### Phase D — Zero-shot transfer demo (Manila)
- Apply the NZ+UK-trained model to Manila bbox; produce the same misalignment map. No retraining, no Manila labels. This is the headline transferability claim for the challenge submission.

### Phase E — Submission packaging
- Methodology PDF, model card, reproducible Colab, hosted webapp, 3-min demo video. Model weights released under permissive licence.

## Critical files (Phase A only — what this plan commits to first)

| Path | Purpose |
|---|---|
| `road-tokeniser/pyproject.toml` | deps |
| `road-tokeniser/fetch_data.py` | CAS + STATS19 download, idempotent |
| `road-tokeniser/safe_system.py` | rule baseline |
| `road-tokeniser/rules/safe_system.yml` | rule thresholds as data |
| `road-tokeniser/tokenise.py` | end-to-end pipeline |
| `road-tokeniser/webapp/index.html` | Leaflet viewer with click-to-explain sidebar |
| `road-tokeniser/explorer/server.py` | FastAPI explorer backend (deferred to first time it's needed; Phase A static viewer works without it) |
| `road-tokeniser/notebooks/01_data_audit.ipynb` | first-stage data sanity notebook |
| `road-tokeniser/README.md` | run instructions |

Reused libraries: `osmnx.graph_from_bbox`, `shapely.ops.substring`, `geopandas`, Leaflet 1.9. Phase B will add: `torch`, `torch_geometric`.

## Verification

### Phase A
1. `pip install -e road-tokeniser/` clean.
2. `python road-tokeniser/fetch_data.py --site nz` downloads `CAS_Data_public.csv` (~150 MB) into `data/raw/`. Idempotent re-run.
3. `python road-tokeniser/tokenise.py --bbox 174.70,-41.40,175.20,-41.10 --site nz --out webapp/tokens.geojson` produces ~25–35 k features. Spot-check `length_m ∈ [5, 35]`.
4. Same script with Cambridge bbox + `--site uk` works without code changes.
5. `python -m http.server` in `webapp/`; open browser — map renders, all colour ramps switch, hover shows features.
6. **Sanity check on `rule_misalignment_kph`**: SH1 rural sections north of Wellington (legal 100 km/h, two-lane undivided, no median) should flag at +30 (Safe System says 70 km/h without median). Urban Cuba St in central Wellington with peds + no separation, posted 50 km/h → flags at +20 (Safe System says 30 km/h).
7. **Sanity check `crash_score`**: known black-spots (e.g. Ngauranga Gorge merge, Centennial Highway, the SH58 corner) light up.

### Phase B
1. Pretraining loss curves all decrease, no divergence. Final masked-geometry MSE < 0.3 on validation.
2. UMAP of embeddings clusters by `highway` class without `highway` being an input to the SSL loss → confirms the encoder learned road-class semantics from geometry + topology alone.

### Phase C
- The relative-risk headline: top-quintile-misalignment tokens have crash rate ≥2× bottom-quintile tokens *at the same posted speed*. This is the falsifiable validation. If it doesn't hit, the model is detecting noise, and we drop the ML claims and ship the rule engine alone.

### Phase D
- Apply NZ+UK model to Manila. Manual qualitative check: do high-`ml_misalignment_score` segments concentrate around schools, markets, and unsignalled intersections, as Safe System theory predicts? If yes, zero-shot transfer holds.

## Tooling & monitoring — live exploration is a first-class feature

The webapp isn't just the final demo. It's the **researcher cockpit** throughout. Three lightweight, local-only layers:

### 1. Pipeline-stage exploration (Jupyter)
One notebook per pipeline stage so we can stop and inspect the data at every step. Cheap to maintain because each notebook is just a thin wrapper that imports the same functions used by `tokenise.py`.

- `notebooks/01_data_audit.ipynb` — raw OSM + CAS / STATS19 sanity (row counts, CRS, missingness)
- `notebooks/02_tokenisation.ipynb` — visualise the same road before/after splitting at 25 m; histogram of `length_m`
- `notebooks/03_features.ipynb` — feature distributions per `highway` class; correlations
- `notebooks/04_pretrain_diagnostics.ipynb` — load a checkpoint, project embeddings with UMAP, colour by various tags, eyeball clustering
- `notebooks/05_misalignment_audit.ipynb` — top-100 flagged segments, side-by-side with crash-rate validation bins

### 2. Training monitoring (TensorBoard, local)
`pretrain.py` logs to `runs/{timestamp}/` via `torch.utils.tensorboard.SummaryWriter`. Standard, free, opens in browser at `localhost:6006`:

- Loss curves per SSL objective (masked-geom, masked-tag, contrastive) — separate plots so we can see if one objective is collapsing
- Validation masked-geometry MSE per epoch
- **Embedding Projector** (TB built-in): periodic dump of 5 k random tokens' embeddings + their metadata (`highway`, `posted_speed`, `region`) so you can rotate the 3D UMAP live in the browser as training progresses
- Sample-token panel: each epoch, pick 8 fixed validation tokens, log their nearest-neighbour posted-speed distributions as histograms — see misalignment scoring stabilise over training

### 3. Interactive map with explain-on-click (extended webapp + small FastAPI backend)
This is the main piece. The Leaflet webapp gains a sidebar that opens when you click a token, with a small Python backend serving live data:

`road-tokeniser/explorer/server.py` — FastAPI app, ~150 lines:
- `GET /api/tokens?bbox=W,S,E,N&ramp=...` — returns GeoJSON for the visible viewport (paginated for country-scale views)
- `GET /api/token/{id}` — full feature dict + Safe System breakdown + ML misalignment score
- `GET /api/token/{id}/neighbours?k=50` — the k embedding-space neighbours; the webapp highlights them on the map so you can *see* what the model thinks looks like this segment
- `GET /api/token/{id}/explain` — natural-language reason: "Flagged because: posted 100 km/h, but 47/50 geometric neighbours posted 70 km/h; nearest 3 km has 4 CAS fatal crashes since 2020."
- `WS /ws/training` — websocket that `pretrain.py` writes to via a callback; pushes `{epoch, loss, sample_token_scores}` so the map can recolour live during training (you can leave training running overnight and watch the misalignment map evolve)

Webapp side (`webapp/index.html`):
- Click any token → right sidebar opens with feature dict, Safe System verdict, ML verdict, "why" explanation
- "Show neighbours" button highlights the 50 nearest in embedding space — instant gut-check of whether the model has learned road-class semantics
- "Live training" toggle subscribes to the websocket and recolours the visible tokens each time a checkpoint lands
- Single-file HTML still — backend is optional; without `server.py` running, the webapp falls back to the static `tokens.geojson` and the sidebar shows feature dict only (no neighbours, no explanations)

This costs maybe 1 day of extra build over a plain static viewer and is worth it — it's the difference between "I built a model" and "I can audit any decision the model makes."

### 4. Reproducibility
- All scripts deterministic with a `--seed` flag
- `data/raw/` files have SHA-256 manifest in `data/manifest.json` (computed on first download); re-runs check against it
- Each training run writes `runs/{timestamp}/config.yml` + `runs/{timestamp}/git_sha.txt`

---

## Runs entirely on a local laptop

Nothing in this plan requires cloud compute, paid APIs, or a managed service. Concrete envelope on an Apple Silicon MacBook (assumed from `darwin` host):

| Phase | What it does | Where it runs | Wall time estimate |
|---|---|---|---|
| A — pipeline + viz | Tokenise, rule engine, GeoJSON, Leaflet | Pure Python CPU; webapp via `python -m http.server` | < 5 min for Wellington bbox; < 1 min for Cambridge |
| B — pretraining (small) | SSL pretrain on **Wellington + Cambridge tokens only** (~35 k) | PyTorch MPS (Metal) backend on M-series GPU, or CPU | 20–60 min on M2/M3; ~3 h on CPU |
| B — pretraining (full) | SSL pretrain on **all-NZ + all-UK** (~5 M tokens) | PyTorch MPS on M-series with ≥16 GB unified memory | 12–24 h overnight |
| C — misalignment scoring + validation | k-NN over embeddings, crash-rate bins, report | CPU, in-memory | < 5 min |
| D — Manila zero-shot | Inference only on Manila bbox | CPU or MPS | < 2 min |

**Default strategy: start small.** Phase B-small (Wellington + Cambridge corpus) is enough to validate the approach end-to-end in an afternoon. We only scale to Phase B-full if the small-corpus validation in Phase C looks promising. If it doesn't, we cut our losses and ship the rule engine baseline alone — no overnight runs wasted.

**External network is needed at extract time only**:
- OSMnx hits the Overpass API to fetch OSM extracts (cached locally after first run)
- `fetch_data.py` hits `data.dft.gov.uk` and the Waka Kotahi ArcGIS endpoint once (CSVs cached locally)

After those one-time downloads, everything runs offline.

No GPU vendor accounts, no Hugging Face Hub uploads, no Weights & Biases — `runs/` is a local directory of TensorBoard logs and PyTorch checkpoints.

## Open questions (non-blocking for Phase A)

**Q1.** Should the released model weights be ADB-branded and open, or held for a DMC pilot first? Affects how we structure Phase E.

**Q2.** Challenge submission deadline — what are we pacing against?
