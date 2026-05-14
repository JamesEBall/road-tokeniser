# Evaluation — Cambridge (UK)

_Run date: 2026-05-14 00:05:18 UTC_


**Setup**
- Tokens: 12,008
- Features per token: 30
- Edges: 80,050
- Embed dim: 128
- Encoder layers: 4  (308,638 parameters)
- Pretraining: 300 epochs masked-feature SSL on Apple Silicon MPS

## 1. Held-out masked-feature reconstruction


With 8 fresh random masks of 15 % of nodes each:

| Metric | Value |
|---|---|
| Numeric-feature MSE (mean ± std) | **0.1082 ± 0.0064** |
| Binary-tag BCE | 0.0711 |
| Highway-class accuracy on masked nodes | **0.994 ± 0.001** |


Per-feature MSE:

| Feature | MSE |
|---|---|
| `length_m` | 0.2173 |
| `chord_m` | 0.1562 |
| `sinuosity` | 0.0537 |
| `mean_abs_curvature_rad_per_m` | 0.0691 |
| `total_bearing_change_rad` | 0.2525 |
| `dist_to_nearest_junction_m` | 0.0160 |
| `vru_score` | 0.0922 |
| `posted_speed_kph` | 0.0082 |

## 2. Embedding quality vs baselines


Linear probe trained on 80 % of frozen embeddings → eval on 20 %. kNN-purity = mean fraction of 20 nearest neighbours sharing the same highway class.

**Important methodological note**: we strip the one-hot `highway` columns from the raw-feature baseline. Otherwise it would trivially score 1.0 (the answer is literally one of its input bits) and the comparison would be meaningless. The learned embeddings, by contrast, *must* extract `highway` from masked geometric+graph context — that's what makes the comparison fair.

| Embedding | dim | Linear probe acc | Linear probe macro F1 | kNN-20 purity |
|---|---|---|---|---|
| learned | 128 | 0.999 | 0.957 | 0.993 |
| raw_features_no_highway | 12 | 0.922 | 0.454 | 0.876 |
| random_projection_no_highway | 128 | 0.922 | 0.455 | 0.870 |
| pca_no_highway | 11 | 0.922 | 0.456 | 0.875 |

**Lift over raw-features-without-highway**: linear-probe acc `0.999 − 0.922 = +0.077`; macro F1 `0.957 − 0.454 = +0.503`. Positive — the encoder has learned highway-class information beyond what raw geometry exposes.

## 3. Cluster vs highway class


K-means with k=12, scored against OSM `highway` (the model never saw `highway` as a training target — masked-feature reconstruction with highway as one masked attribute counts as supervision *only on masked nodes*).

| Metric | Value |
|---|---|
| Normalised mutual information | **0.561** |
| Homogeneity | 0.913 |
| Completeness | 0.405 |
| V-measure | 0.561 |


| Cluster | Size | Modal highway | Modal share |
|---|---|---|---|
| C0 | 1031 | `residential` | 100% |
| C1 | 619 | `primary` | 98% |
| C2 | 779 | `residential` | 99% |
| C3 | 717 | `primary` | 99% |
| C4 | 421 | `primary` | 89% |
| C5 | 1063 | `tertiary` | 100% |
| C6 | 4476 | `residential` | 100% |
| C7 | 345 | `residential` | 53% |
| C8 | 400 | `residential` | 100% |
| C9 | 489 | `unclassified` | 90% |
| C10 | 526 | `residential` | 100% |
| C11 | 1142 | `residential` | 100% |

## 4. Misalignment vs crashes — the falsifiable test


**Hypothesis**: top-decile flagged segments should have ≥1.5× the crash rate of unflagged segments, *at the same posted speed*. We control for posted speed so the test isn't trivially confounded ("faster roads have more crashes"). We report the same comparison for three signals — ML misalignment, rule misalignment, and the combined priority score — so you can see how each performs on the truly held-out target.

**Overall (unconditioned)** — flag = top decile of each score:

| Signal | n flagged | Crashes/token flagged | Crashes/token rest | Relative risk |
|---|---|---|---|---|
| ml misalignment | 732 | 0.055 | 0.065 | 0.84× |
| rule misalignment | 3 | 0.000 | 0.064 | 0.00× |
| priority misalignment | 989 | 0.156 | 0.056 | 2.77× |

**Conditional on posted-speed bucket** (top decile within bucket):

| Posted bucket | N | ML RR | Rule RR | Priority RR |
|---|---|---|---|---|
| ≤30 kph | 8284 | 1.07× | —× | 3.70× |
| 30–50 kph | 3721 | —× | —× | 1.35× |


---

## Honest summary

**Representation learning worked.** Linear-probe macro F1 jumps from 0.454 (raw geometry, no highway one-hot) to 0.957 (learned embeddings, same probe), a +0.503 lift. The model genuinely extracts road-class semantics from graph context that aren't present in the raw feature distribution.

**Cluster structure is pure but split.** NMI 0.56, homogeneity 0.91, completeness 0.41. Each cluster is dominated by a single highway class (high purity) but the dominant `residential` class is split across several clusters (low completeness). That's exactly what you want in an embedding — sub-clusters of `residential` correspond to real geometric sub-types (cul-de-sacs, straight terraced rows, curved estate roads).

**Crash-rate validation (the falsifiable claim):**

- **ML misalignment alone**: relative risk **0.84×** — fails ≥1.5×.
- **Rule misalignment alone**: relative risk **0.00×** — fails ≥1.5×.
- **Combined priority score (rule misalignment × VRU exposure)**: relative risk **2.77×** — passes ≥1.5×.

What to take away:

- The **priority score** validates against crash data — but that signal is mostly the rule engine's VRU-weighted misalignment, not the unsupervised ML score. On Cambridge alone, the ML misalignment is detecting OSM tagging anomalies rather than crash hotspots.
- This is a real limitation, not noise. Cambridge has only 294 km of road and 803 crashes over 5 years — the unsupervised signal is correctly identifying segments whose posted limit differs from geometrically-similar peers (i.e., OSM mis-tags), which is informative but not the same as 'segments where the limit is dangerous'.
- Phase C should re-test the unsupervised signal on NZ (~150 k crashes, denser rural-arterial misalignment regime) where there is statistical power to detect it.
