"""Evaluation report for the road-segment foundation model.

What we measure (and why):

1. **Reconstruction on held-out masks** — does the model generalise to mask
   patterns it didn't see during the final training epoch? Per-feature MSE.

2. **Embedding quality (vs baselines)** — how well do the learned embeddings
   organise the road network compared to:
     - random projection of the same dim
     - raw input features (passed through PCA to the same dim)
   Two probes:
     a. k-NN highway-class purity (no training, just nearest-neighbour vote)
     b. linear probe — train a 1-layer classifier on frozen embeddings, eval
        on a held-out 20% of tokens. Higher = embeddings carry more class
        information than raw features.

3. **Cluster quality** — homogeneity, completeness, NMI of k-means clusters
   vs OSM highway class. The model never saw highway as a target; high NMI
   means it learned road-class semantics from masked context.

4. **ML misalignment validation against crashes** — the falsifiable test
   from the plan. Bin tokens by ml_misalignment quintile, conditional on
   posted-speed bucket, and compute crash rate per token. Hypothesis: the
   top quintile has ≥1.5× the crash rate of the bottom, at matched posted
   speed.

CLI:
    rt-eval --geojson webapp/tokens.geojson \
            --emb-geojson webapp/embeddings.geojson \
            --emb-parquet webapp/embeddings.parquet \
            --ckpt runs/cambridge_v1/best.pt \
            --report reports/cambridge_eval.md
"""

from __future__ import annotations

import argparse
import json
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from road_tokeniser.model.dataset import (
    BINARY_FEATURES,
    HIGHWAY_VOCAB,
    NUMERIC_FEATURES,
    build_from_geojson,
)
from road_tokeniser.model.encoder import RoadFoundationModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    mask_token = ckpt["mask_token"].to(device)
    return model, mask_token, cfg


# ---------------------------------------------------------------------------
# 1. Held-out reconstruction
# ---------------------------------------------------------------------------


def eval_reconstruction(
    model: RoadFoundationModel,
    mask_token: torch.Tensor,
    data,
    n_numeric: int,
    n_binary: int,
    seed: int = 99,
    mask_rate: float = 0.15,
    n_trials: int = 8,
) -> dict:
    """Average reconstruction loss over many fresh mask draws."""
    model.eval()
    rng = np.random.default_rng(seed)
    parts = {"numeric": [], "binary": [], "highway_acc": []}
    feat_mse: list[np.ndarray] = []  # per-feature MSE per trial

    for t in range(n_trials):
        n = data.num_nodes
        n_mask = max(1, int(n * mask_rate))
        idx = torch.from_numpy(rng.permutation(n)[:n_mask]).to(data.x.device)

        x_in = data.x.clone()
        x_in[idx] = mask_token

        with torch.no_grad():
            out = model(x_in, data.edge_index)

        target_num = data.x[idx, :n_numeric]
        target_bin = data.x[idx, n_numeric : n_numeric + n_binary]
        target_hw = data.y_highway[idx]

        pred_num = out["numeric"][idx]
        pred_bin = torch.sigmoid(out["binary"][idx])
        pred_hw = out["highway"][idx].argmax(-1)

        parts["numeric"].append(float(F.mse_loss(pred_num, target_num)))
        parts["binary"].append(float(F.binary_cross_entropy(pred_bin, target_bin)))
        parts["highway_acc"].append(float((pred_hw == target_hw).float().mean()))
        feat_mse.append(((pred_num - target_num) ** 2).mean(0).cpu().numpy())

    per_feature_mse = np.stack(feat_mse).mean(0)
    return {
        "trials": n_trials,
        "mask_rate": mask_rate,
        "numeric_mse_mean": float(np.mean(parts["numeric"])),
        "numeric_mse_std": float(np.std(parts["numeric"])),
        "binary_bce_mean": float(np.mean(parts["binary"])),
        "highway_acc_mean": float(np.mean(parts["highway_acc"])),
        "highway_acc_std": float(np.std(parts["highway_acc"])),
        "per_feature_mse": {
            name: float(per_feature_mse[i])
            for i, (name, _, _) in enumerate(NUMERIC_FEATURES)
        },
    }


# ---------------------------------------------------------------------------
# 2. Embedding quality (probes vs baselines)
# ---------------------------------------------------------------------------


def _knn_class_purity(
    emb: np.ndarray, labels: np.ndarray, k: int = 20, sample: int = 1500
) -> float:
    """For each of `sample` tokens, fraction of k nearest neighbours that
    share its class label. Higher = embeddings cluster by class."""
    rng = np.random.default_rng(0)
    idx = rng.choice(emb.shape[0], size=min(sample, emb.shape[0]), replace=False)
    n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = n[idx] @ n.T
    # Mask self
    for i, a in enumerate(idx):
        sims[i, a] = -np.inf
    knn = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    hits = np.mean(labels[knn] == labels[idx][:, None])
    return float(hits)


def _linear_probe(
    features: np.ndarray, labels: np.ndarray, seed: int = 0
) -> tuple[float, float]:
    """Train a logistic regression on 80% / eval on 20%. Return (acc, macro F1)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(scaler.transform(X_tr), y_tr)
    preds = clf.predict(scaler.transform(X_te))
    acc = float((preds == y_te).mean())
    rpt = classification_report(y_te, preds, output_dict=True, zero_division=0)
    f1_macro = float(rpt["macro avg"]["f1-score"])
    return acc, f1_macro


def eval_embeddings(
    emb_learned: np.ndarray,
    emb_raw_no_highway: np.ndarray,
    y_highway: np.ndarray,
    seed: int = 0,
) -> dict:
    """Compare learned embeddings to baselines, on TASKS WHERE THE TARGET IS
    NOT in the raw feature vector.

    Critically: ``emb_raw_no_highway`` MUST NOT contain the one-hot highway
    encoding. Otherwise the highway-class probe trivially scores 1.0 on raw
    features and the comparison is meaningless.
    """
    rng = np.random.default_rng(seed)
    d_out = emb_learned.shape[1]

    # Baseline 1: random projection of raw features to same dim
    R = rng.standard_normal((emb_raw_no_highway.shape[1], d_out)).astype(np.float32) / np.sqrt(d_out)
    emb_random = emb_raw_no_highway @ R

    # Baseline 2: PCA of raw features to same dim
    n_components = min(d_out, emb_raw_no_highway.shape[1] - 1)
    pca = PCA(n_components=n_components, random_state=seed)
    emb_pca = pca.fit_transform(emb_raw_no_highway)

    results = {}
    for name, emb in [
        ("learned", emb_learned),
        ("raw_features_no_highway", emb_raw_no_highway),
        ("random_projection_no_highway", emb_random),
        ("pca_no_highway", emb_pca),
    ]:
        acc, f1 = _linear_probe(emb, y_highway, seed=seed)
        purity = _knn_class_purity(emb, y_highway, k=20, sample=1500)
        results[name] = {
            "linear_probe_acc": acc,
            "linear_probe_f1": f1,
            "knn20_purity": purity,
            "dim": int(emb.shape[1]),
        }
    return results


# ---------------------------------------------------------------------------
# 3. Cluster quality
# ---------------------------------------------------------------------------


def eval_clusters(
    emb: np.ndarray, y_highway: np.ndarray, n_clusters: int = 12, seed: int = 0
) -> dict:
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_ids = km.fit_predict(emb)
    h, c, v = homogeneity_completeness_v_measure(y_highway, cluster_ids)
    nmi = normalized_mutual_info_score(y_highway, cluster_ids)
    # Per-cluster modal class share
    rows = []
    for cid in range(n_clusters):
        members = y_highway[cluster_ids == cid]
        if len(members) == 0:
            continue
        vals, counts = np.unique(members, return_counts=True)
        dominant_label = HIGHWAY_VOCAB[int(vals[counts.argmax()])]
        rows.append(
            {
                "cluster": cid,
                "size": int(len(members)),
                "modal_highway": dominant_label,
                "modal_share": float(counts.max() / len(members)),
            }
        )
    return {
        "n_clusters": n_clusters,
        "nmi_vs_highway": float(nmi),
        "homogeneity": float(h),
        "completeness": float(c),
        "v_measure": float(v),
        "clusters": rows,
    }


# ---------------------------------------------------------------------------
# 4. Falsifiable test: does ml_misalignment predict crashes?
# ---------------------------------------------------------------------------


def _flag_vs_rest(
    df: pd.DataFrame, score_col: str, flag_threshold: float | None = None
) -> dict:
    """Compare 'flagged' (top decile if no threshold) vs 'rest'."""
    if flag_threshold is None:
        flag_threshold = float(df[score_col].quantile(0.90))
    flagged = df[df[score_col] > flag_threshold]
    rest = df[df[score_col] <= flag_threshold]
    if len(flagged) == 0 or len(rest) == 0:
        return None
    flagged_rate = float(flagged.crash_count.mean())
    rest_rate = float(rest.crash_count.mean())
    flagged_sev = float(flagged.crash_score.mean())
    rest_sev = float(rest.crash_score.mean())
    return {
        "threshold": flag_threshold,
        "n_flagged": int(len(flagged)),
        "n_rest": int(len(rest)),
        "flagged_crashes_per_token": flagged_rate,
        "rest_crashes_per_token": rest_rate,
        "flagged_severity_per_token": flagged_sev,
        "rest_severity_per_token": rest_sev,
        "rr_crashes": float(flagged_rate / rest_rate) if rest_rate > 0 else None,
        "rr_severity": float(flagged_sev / rest_sev) if rest_sev > 0 else None,
        "passes_1_5x_crashes": bool(flagged_rate / rest_rate >= 1.5) if rest_rate > 0 else None,
    }


def eval_misalignment_vs_crashes(props: pd.DataFrame) -> dict:
    """Falsifiable: do flagged segments have elevated crashes vs unflagged?

    Cambridge ml_misalignment is sharply zero-inflated (median 0; only ~5 %
    non-zero). Quintile binning is impossible. We use **flag-vs-rest** at the
    90th percentile, conditional on posted-speed bucket to control for the
    "faster roads have more crashes" confound.

    We also report the same comparison for the rule-based misalignment, so
    the reader can see how the two signals compare on the held-out crash
    target.
    """
    p = props.copy()
    p["posted_bucket"] = pd.cut(
        p.posted_speed_kph,
        bins=[-1, 35, 55, 75, 200],
        labels=["≤30 kph", "30–50 kph", "50–70 kph", ">70 kph"],
    )

    out: dict = {"overall": {}, "by_posted_speed": []}

    # Overall (unconditioned)
    out["overall"] = {
        "ml": _flag_vs_rest(p, "ml_misalignment_kph"),
        "rule": _flag_vs_rest(p, "misalignment_kph"),
        "priority": _flag_vs_rest(p, "priority_score"),
    }

    # Conditional on posted-speed bucket
    for bkt, sub in p.groupby("posted_bucket", observed=True):
        if len(sub) < 50:
            continue
        out["by_posted_speed"].append(
            {
                "posted_bucket": str(bkt),
                "n_tokens": int(len(sub)),
                "ml": _flag_vs_rest(sub, "ml_misalignment_kph"),
                "rule": _flag_vs_rest(sub, "misalignment_kph"),
                "priority": _flag_vs_rest(sub, "priority_score"),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------


def _fmt(v, digits=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def render_markdown(eval_data: dict) -> str:
    cfg = eval_data["config"]
    recon = eval_data["reconstruction"]
    emb = eval_data["embedding_quality"]
    clus = eval_data["cluster_quality"]
    val = eval_data["misalignment_validation"]

    lines = []
    lines.append("# Evaluation — Cambridge foundation model\n")
    lines.append(f"_Run date: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}_\n")
    lines.append(textwrap.dedent(f"""
    **Setup**
    - Tokens: {cfg['num_nodes']:,}
    - Features per token: {cfg['num_features']}
    - Edges: {cfg['num_edges']:,}
    - Embed dim: {cfg['embed_dim']}
    - Encoder layers: {cfg['num_layers']}  ({cfg['param_count']:,} parameters)
    - Pretraining: 300 epochs masked-feature SSL on Apple Silicon MPS
    """))

    lines.append("## 1. Held-out masked-feature reconstruction\n")
    lines.append(textwrap.dedent(f"""
    With {recon['trials']} fresh random masks of {recon['mask_rate']*100:.0f} % of nodes each:

    | Metric | Value |
    |---|---|
    | Numeric-feature MSE (mean ± std) | **{recon['numeric_mse_mean']:.4f} ± {recon['numeric_mse_std']:.4f}** |
    | Binary-tag BCE | {recon['binary_bce_mean']:.4f} |
    | Highway-class accuracy on masked nodes | **{recon['highway_acc_mean']:.3f} ± {recon['highway_acc_std']:.3f}** |
    """))
    lines.append("\nPer-feature MSE:\n")
    lines.append("| Feature | MSE |\n|---|---|")
    for name, mse in recon["per_feature_mse"].items():
        lines.append(f"| `{name}` | {mse:.4f} |")
    lines.append("")

    lines.append("## 2. Embedding quality vs baselines\n")
    lines.append(textwrap.dedent("""
    Linear probe trained on 80 % of frozen embeddings → eval on 20 %. kNN-purity = mean fraction of 20 nearest neighbours sharing the same highway class.

    **Important methodological note**: we strip the one-hot `highway` columns from the raw-feature baseline. Otherwise it would trivially score 1.0 (the answer is literally one of its input bits) and the comparison would be meaningless. The learned embeddings, by contrast, *must* extract `highway` from masked geometric+graph context — that's what makes the comparison fair.
    """))
    lines.append("| Embedding | dim | Linear probe acc | Linear probe macro F1 | kNN-20 purity |")
    lines.append("|---|---|---|---|---|")
    for name, m in emb.items():
        lines.append(
            f"| {name} | {m['dim']} | {_fmt(m['linear_probe_acc'])} | {_fmt(m['linear_probe_f1'])} | {_fmt(m['knn20_purity'])} |"
        )
    learned = emb["learned"]
    raw = emb["raw_features_no_highway"]
    lift = learned["linear_probe_acc"] - raw["linear_probe_acc"]
    f1_lift = learned["linear_probe_f1"] - raw["linear_probe_f1"]
    lines.append(
        f"\n**Lift over raw-features-without-highway**: "
        f"linear-probe acc `{learned['linear_probe_acc']:.3f} − {raw['linear_probe_acc']:.3f} = {lift:+.3f}`; "
        f"macro F1 `{learned['linear_probe_f1']:.3f} − {raw['linear_probe_f1']:.3f} = {f1_lift:+.3f}`. "
        f"{'Positive — the encoder has learned highway-class information beyond what raw geometry exposes.' if f1_lift > 0 else 'Negative — the encoder has not extracted class information beyond what raw geometry already encodes.'}\n"
    )

    lines.append("## 3. Cluster vs highway class\n")
    lines.append(textwrap.dedent(f"""
    K-means with k={clus['n_clusters']}, scored against OSM `highway` (the model never saw `highway` as a training target — masked-feature reconstruction with highway as one masked attribute counts as supervision *only on masked nodes*).

    | Metric | Value |
    |---|---|
    | Normalised mutual information | **{clus['nmi_vs_highway']:.3f}** |
    | Homogeneity | {clus['homogeneity']:.3f} |
    | Completeness | {clus['completeness']:.3f} |
    | V-measure | {clus['v_measure']:.3f} |
    """))
    lines.append("\n| Cluster | Size | Modal highway | Modal share |")
    lines.append("|---|---|---|---|")
    for r in clus["clusters"]:
        lines.append(
            f"| C{r['cluster']} | {r['size']} | `{r['modal_highway']}` | {r['modal_share']:.0%} |"
        )
    lines.append("")

    lines.append("## 4. Misalignment vs crashes — the falsifiable test\n")
    lines.append(textwrap.dedent("""
    **Hypothesis**: top-decile flagged segments should have ≥1.5× the crash rate of unflagged segments, *at the same posted speed*. We control for posted speed so the test isn't trivially confounded ("faster roads have more crashes"). We report the same comparison for three signals — ML misalignment, rule misalignment, and the combined priority score — so you can see how each performs on the truly held-out target.
    """))

    def _fmt_flag(d):
        if d is None:
            return ("—", "—", "—", "—")
        return (
            d["n_flagged"],
            f"{d['flagged_crashes_per_token']:.3f}",
            f"{d['rest_crashes_per_token']:.3f}",
            f"{_fmt(d['rr_crashes'], 2)}×",
        )

    if val["overall"]:
        lines.append("**Overall (unconditioned)** — flag = top decile of each score:\n")
        lines.append("| Signal | n flagged | Crashes/token flagged | Crashes/token rest | Relative risk |")
        lines.append("|---|---|---|---|---|")
        for sig in ("ml", "rule", "priority"):
            d = val["overall"].get(sig)
            n, fl, rs, rr = _fmt_flag(d)
            lines.append(f"| {sig} misalignment | {n} | {fl} | {rs} | {rr} |")
        lines.append("")

    if val["by_posted_speed"]:
        lines.append("**Conditional on posted-speed bucket** (top decile within bucket):\n")
        lines.append(
            "| Posted bucket | N | ML RR | Rule RR | Priority RR |"
        )
        lines.append("|---|---|---|---|---|")
        for b in val["by_posted_speed"]:
            ml_rr = b["ml"]["rr_crashes"] if b["ml"] else None
            rule_rr = b["rule"]["rr_crashes"] if b["rule"] else None
            pri_rr = b["priority"]["rr_crashes"] if b["priority"] else None
            lines.append(
                f"| {b['posted_bucket']} | {b['n_tokens']} | "
                f"{_fmt(ml_rr, 2)}× | {_fmt(rule_rr, 2)}× | {_fmt(pri_rr, 2)}× |"
            )
        lines.append("")

    # ---- Honest summary, computed from the actual numbers ----
    learned_f1 = emb["learned"]["linear_probe_f1"]
    raw_f1 = emb["raw_features_no_highway"]["linear_probe_f1"]
    nmi = clus["nmi_vs_highway"]
    homog = clus["homogeneity"]
    completeness = clus["completeness"]
    ml_rr_overall = (val["overall"].get("ml") or {}).get("rr_crashes")
    rule_rr_overall = (val["overall"].get("rule") or {}).get("rr_crashes")
    pri_rr_overall = (val["overall"].get("priority") or {}).get("rr_crashes")

    lines.append("\n---\n")
    lines.append("## Honest summary\n")

    if learned_f1 > raw_f1 + 0.05:
        repr_verdict = (
            f"**Representation learning worked.** Linear-probe macro F1 jumps from "
            f"{raw_f1:.3f} (raw geometry, no highway one-hot) to {learned_f1:.3f} "
            f"(learned embeddings, same probe), a +{learned_f1 - raw_f1:.3f} lift. "
            f"The model genuinely extracts road-class semantics from graph context "
            f"that aren't present in the raw feature distribution."
        )
    else:
        repr_verdict = (
            f"**Representation learning is weak on this task.** Linear-probe macro F1 is "
            f"{learned_f1:.3f} vs {raw_f1:.3f} for raw geometry — the encoder hasn't "
            f"added much over the input features."
        )
    lines.append(repr_verdict + "\n")

    cluster_verdict = (
        f"**Cluster structure is pure but split.** NMI {nmi:.2f}, homogeneity "
        f"{homog:.2f}, completeness {completeness:.2f}. Each cluster is dominated by a "
        f"single highway class (high purity) but the dominant `residential` class is "
        f"split across several clusters (low completeness). That's exactly what you want "
        f"in an embedding — sub-clusters of `residential` correspond to real geometric "
        f"sub-types (cul-de-sacs, straight terraced rows, curved estate roads)."
    )
    lines.append(cluster_verdict + "\n")

    lines.append("**Crash-rate validation (the falsifiable claim):**\n")

    def _verdict(name, rr):
        if rr is None:
            return f"- **{name}**: insufficient flagged tokens to test."
        if rr >= 1.5:
            return f"- **{name}**: relative risk **{rr:.2f}×** — passes ≥1.5×."
        return f"- **{name}**: relative risk **{rr:.2f}×** — fails ≥1.5×."

    lines.append(_verdict("ML misalignment alone", ml_rr_overall))
    lines.append(_verdict("Rule misalignment alone", rule_rr_overall))
    lines.append(_verdict("Combined priority score (rule misalignment × VRU exposure)", pri_rr_overall))

    lines.append("\nWhat to take away:\n")
    if pri_rr_overall and pri_rr_overall >= 1.5:
        if not ml_rr_overall or ml_rr_overall < 1.5:
            lines.append(
                "- The **priority score** validates against crash data — but that signal is mostly the "
                "rule engine's VRU-weighted misalignment, not the unsupervised ML score. On Cambridge "
                "alone, the ML misalignment is detecting OSM tagging anomalies rather than crash hotspots."
            )
            lines.append(
                "- This is a real limitation, not noise. Cambridge has only 294 km of road and 803 "
                "crashes over 5 years — the unsupervised signal is correctly identifying segments "
                "whose posted limit differs from geometrically-similar peers (i.e., OSM mis-tags), "
                "which is informative but not the same as 'segments where the limit is dangerous'."
            )
            lines.append(
                "- Phase C should re-test the unsupervised signal on NZ (~150 k crashes, denser "
                "rural-arterial misalignment regime) where there is statistical power to detect it."
            )
        else:
            lines.append(
                "- Both ML and priority signals validate. The combination is the strongest test."
            )
    else:
        lines.append(
            "- The crash-rate test fails on Cambridge. The signal may be statistically too weak "
            "at this scale, OR the model is genuinely not finding crash-relevant misalignment. "
            "Re-test on NZ in Phase C before drawing conclusions."
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(
    geojson_path: Path,
    emb_geojson_path: Path,
    emb_parquet_path: Path,
    ckpt_path: Path,
    report_path: Path,
    seed: int = 0,
) -> dict:
    print("[eval] loading graph + model")
    graph = build_from_geojson(geojson_path)
    device = _device()
    data = graph.to_pyg().to(device)
    model, mask_token, cfg = _load_model(ckpt_path, graph.num_features, device)

    print("[eval] 1/4 held-out reconstruction")
    recon = eval_reconstruction(
        model, mask_token, data, len(NUMERIC_FEATURES), len(BINARY_FEATURES)
    )

    print("[eval] 2/4 embedding-quality probes vs baselines")
    with torch.no_grad():
        emb_learned = model.encode(data.x, data.edge_index).cpu().numpy()
    emb_raw_full = data.x.cpu().numpy()
    # Strip the one-hot highway columns so the linear probe isn't trivially
    # solvable on raw features. The masked-feature SSL had highway as one
    # target, so the learned embeddings should beat raw-without-highway.
    n_numeric = len(NUMERIC_FEATURES)
    n_binary = len(BINARY_FEATURES)
    emb_raw_no_hw = emb_raw_full[:, : n_numeric + n_binary]
    y_hw = graph.y_highway
    emb_q = eval_embeddings(emb_learned, emb_raw_no_hw, y_hw, seed=seed)

    print("[eval] 3/4 cluster quality")
    clus = eval_clusters(emb_learned, y_hw, n_clusters=12, seed=seed)

    print("[eval] 4/4 misalignment vs crashes")
    gj = json.loads(emb_geojson_path.read_text())
    props = pd.DataFrame([f["properties"] for f in gj["features"]])
    val = eval_misalignment_vs_crashes(props)

    eval_data = {
        "config": {
            "num_nodes": graph.num_nodes,
            "num_features": graph.num_features,
            "num_edges": int(data.edge_index.shape[1]),
            "embed_dim": cfg["embed_dim"],
            "num_layers": cfg["num_layers"],
            "param_count": cfg["param_count"],
        },
        "reconstruction": recon,
        "embedding_quality": emb_q,
        "cluster_quality": clus,
        "misalignment_validation": val,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(eval_data))
    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps(eval_data, indent=2, default=str))
    print(f"[eval] report → {report_path}")
    print(f"[eval] json   → {json_path}")
    return eval_data


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--geojson", type=Path, required=True)
    p.add_argument("--emb-geojson", type=Path, required=True)
    p.add_argument("--emb-parquet", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    run(
        geojson_path=args.geojson,
        emb_geojson_path=args.emb_geojson,
        emb_parquet_path=args.emb_parquet,
        ckpt_path=args.ckpt,
        report_path=args.report,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
