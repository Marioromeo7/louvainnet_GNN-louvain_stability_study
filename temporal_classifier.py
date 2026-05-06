"""
temporal_classifier.py — Cross-network binary classifier for community change

Research question:
  Can a logistic classifier trained on features from one temporal network
  predict high-change snapshots in a completely different network?

Design:
  - Pool snapshots from multiple temporal networks
  - Leave-one-network-out (LONO) cross-validation
  - Each fold: train on all networks except one, test on the held-out network
  - Features: mean_pairwise_NMI, N, E, density, clustering (current snapshot t)
  - Target: NMI(t, t+1) < 0.70  (high structural change next step)

Networks:
  1. email-Eu-core-temporal  (986 nodes, 332K edges, ~3 years)   → reuse computed data
  2. CollegeMsg              (1899 nodes,  60K edges, ~6 months)  → download + compute
  3. email-W3C               (community-structured; W3C mailing)  → if available

Run: conda run -n mario python temporal_classifier.py
Outputs: temporal_classifier_results.csv, fig_temporal_classifier.png
"""

import os
import gzip
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                              precision_recall_curve, average_precision_score,
                              confusion_matrix)
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# ── Config ────────────────────────────────────────────────────────────────────
N_PROBE    = 50
ARI_THRESH = 0.9999
MIN_NODES  = 30
HIGH_CHANGE_THRESH = 0.70   # NMI(t,t+1) < this → high-change event

FEATURE_COLS = ["mean_pairwise_nmi", "N", "E", "density", "clustering"]

DATASETS = [
    {
        "name":    "email-Eu-core",
        "url":     "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz",
        "gz":      "data/snap/email-Eu-core-temporal.txt.gz",
        "txt":     "data/snap/email-Eu-core-temporal.txt",
        "results": "temporal_results.csv",         # reuse if exists
        "rich":    "temporal_rich_metrics.csv",
        "window":  30 * 86400,
        "step":    7  * 86400,
    },
    {
        "name":    "CollegeMsg",
        "url":     "https://snap.stanford.edu/data/CollegeMsg.txt.gz",
        "gz":      "data/snap/CollegeMsg.txt.gz",
        "txt":     "data/snap/CollegeMsg.txt",
        "results": None,
        "rich":    None,
        "window":  14 * 86400,   # shorter window — only 6 months of data
        "step":    3  * 86400,
    },
]

# ── Utilities (identical to temporal_study / temporal_correlations) ───────────
def download(url, gz_path, txt_path):
    if os.path.exists(txt_path):
        print(f"  already at {txt_path}")
        return True
    os.makedirs(os.path.dirname(gz_path), exist_ok=True)
    print(f"  downloading {url} …")
    try:
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, "rb") as fi, open(txt_path, "wb") as fo:
            fo.write(fi.read())
        os.remove(gz_path)
        print("  done.")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

def load_edges(path):
    edges = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                u, v, t = int(parts[0]), int(parts[1]), int(parts[2])
                if u != v:
                    edges.append((u, v, t))
    return edges

def make_snapshots(edges, window, step):
    t_min = min(e[2] for e in edges)
    t_max = max(e[2] for e in edges)
    snaps = []
    t = t_min
    while t + window <= t_max:
        G = nx.Graph()
        for u, v, ts in edges:
            if t <= ts < t + window:
                G.add_edge(u, v)
        if G.number_of_nodes() >= MIN_NODES:
            snaps.append((t, G))
        t += step
    return snaps

def structural_metrics(G):
    part = community_louvain.best_partition(G, random_state=42)
    comms = defaultdict(list)
    for node, c in part.items():
        comms[c].append(node)
    sizes   = [len(v) for v in comms.values()]
    size_cv = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0.0
    mod     = community_louvain.modularity(part, G)
    dens    = nx.density(G)
    clust   = nx.average_clustering(G)
    return part, mod, size_cv, dens, clust

def rich_instability(G):
    nodes = sorted(G.nodes())
    partitions, k_vals = [], []
    for seed in range(N_PROBE):
        part = community_louvain.best_partition(G, random_state=seed)
        k_vals.append(len(set(part.values())))
        partitions.append([part[n] for n in nodes])
    rng  = np.random.default_rng(0)
    pairs = rng.choice(N_PROBE, size=(200, 2), replace=True)
    nmis  = [normalized_mutual_info_score(partitions[i], partitions[j])
             for i, j in pairs if i != j]
    return np.mean(nmis) if nmis else np.nan, np.array(k_vals)

def nmi_change(part_prev, nodes_prev, part_curr, nodes_curr):
    common = sorted(set(nodes_prev) & set(nodes_curr))
    if len(common) < 10:
        return np.nan
    return normalized_mutual_info_score(
        [part_prev[n] for n in common],
        [part_curr[n]  for n in common]
    )

# ── Process one dataset → DataFrame of snapshots ─────────────────────────────
def process_dataset(ds):
    name = ds["name"]

    # ── Reuse precomputed data if available ──
    if ds["results"] and ds["rich"] and \
       os.path.exists(ds["results"]) and os.path.exists(ds["rich"]):
        print(f"  Reusing {ds['results']} + {ds['rich']}")
        tdf = pd.read_csv(ds["results"])
        rdf = pd.read_csv(ds["rich"])
        df  = tdf.merge(rdf, on="snapshot", how="inner")
        df["network"] = name
        df.rename(columns={"modularity": "modularity",
                            "nmi_structural_change": "nmi_next_raw"}, inplace=True)
        # build target (shift nmi_structural_change by -1 to get NMI(t,t+1))
        df["nmi_next"] = df["nmi_next_raw"].shift(-1)
        df["high_change"] = (df["nmi_next"] < HIGH_CHANGE_THRESH).astype(int)
        return df

    # ── Download + compute ──
    if not download(ds["url"], ds["gz"], ds["txt"]):
        return None

    edges = load_edges(ds["txt"])
    snaps = make_snapshots(edges, ds["window"], ds["step"])
    print(f"  {len(snaps)} snapshots  (window={ds['window']//86400}d, step={ds['step']//86400}d)")

    rows      = []
    prev_part = prev_nodes = None

    for i, (t, G) in enumerate(snaps):
        lcc   = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(lcc).copy()
        N, E  = G_lcc.number_of_nodes(), G_lcc.number_of_edges()
        if N < MIN_NODES:
            prev_part = prev_nodes = None
            continue

        part, mod, scv, dens, clust = structural_metrics(G_lcc)
        mpnmi, k_arr = rich_instability(G_lcc)
        dnmi = nmi_change(prev_part, prev_nodes, part, list(G_lcc.nodes())) \
               if prev_part is not None else np.nan

        rows.append({
            "snapshot": i, "network": name,
            "N": N, "E": E, "density": dens, "clustering": clust,
            "modularity": mod, "size_cv": scv,
            "mean_pairwise_nmi": mpnmi,
            "K_mean": k_arr.mean(), "K_std": k_arr.std(),
            "nmi_next_raw": dnmi,
        })
        prev_part, prev_nodes = part, list(G_lcc.nodes())

        if (i + 1) % 10 == 0 or i == len(snaps) - 1:
            dnmi_str = f"{dnmi:.3f}" if not np.isnan(dnmi) else "nan"
            print(f"    [{i+1:3d}/{len(snaps)}]  N={N}  mpNMI={mpnmi:.3f}  ΔNMI={dnmi_str}")

    df = pd.DataFrame(rows)
    # NMI(t,t+1) = nmi_next_raw shifted back one step
    df["nmi_next"] = df["nmi_next_raw"].shift(-1)
    df["high_change"] = (df["nmi_next"] < HIGH_CHANGE_THRESH).astype(int)
    return df

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Collect all networks ──
    all_dfs = []
    for ds in DATASETS:
        print(f"\n── {ds['name']} ──")
        df = process_dataset(ds)
        if df is not None:
            all_dfs.append(df)
            print(f"  {len(df)} snapshots, {df['high_change'].sum()} high-change events "
                  f"({df['high_change'].mean()*100:.1f}%)")

    if len(all_dfs) < 2:
        print("\nNeed at least 2 networks for LONO CV. Exiting.")
        exit(1)

    pool = pd.concat(all_dfs, ignore_index=True)
    pool_clean = pool.dropna(subset=FEATURE_COLS + ["high_change", "nmi_next"]).copy()
    networks   = pool_clean["network"].unique()

    print(f"\nPooled: {len(pool_clean)} snapshots across {len(networks)} networks")
    for net in networks:
        sub = pool_clean[pool_clean["network"] == net]
        print(f"  {net:<22}  n={len(sub):3d}  high-change={sub['high_change'].sum():2d} "
              f"({sub['high_change'].mean()*100:.0f}%)")

    # ── Leave-one-network-out CV ──
    print("\n── Leave-One-Network-Out CV ──")
    lono_results = []

    for test_net in networks:
        train_mask = pool_clean["network"] != test_net
        test_mask  = pool_clean["network"] == test_net

        X_train = pool_clean.loc[train_mask, FEATURE_COLS].values
        y_train = pool_clean.loc[train_mask, "high_change"].values
        X_test  = pool_clean.loc[test_mask,  FEATURE_COLS].values
        y_test  = pool_clean.loc[test_mask,  "high_change"].values

        if y_test.sum() == 0:
            print(f"  {test_net}: no high-change events — skip")
            continue

        scaler  = StandardScaler().fit(X_train)
        clf     = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        clf.fit(scaler.transform(X_train), y_train)
        probs   = clf.predict_proba(scaler.transform(X_test))[:, 1]
        preds   = (probs >= 0.5).astype(int)

        bal_acc = balanced_accuracy_score(y_test, preds)
        ap      = average_precision_score(y_test, probs)
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel() \
                         if len(np.unique(y_test)) == 2 else (0, 0, 0, 0)
        recall  = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec    = tp / (tp + fp) if (tp + fp) > 0 else 0

        lono_results.append({
            "test_network": test_net,
            "n_test": len(y_test),
            "n_high": int(y_test.sum()),
            "balanced_acc": bal_acc,
            "avg_precision": ap,
            "recall": recall,
            "precision": prec,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "y_test": y_test,
            "probs":  probs,
        })

        print(f"  Test={test_net:<22}  bal_acc={bal_acc:.3f}  "
              f"recall={recall:.2f}  precision={prec:.2f}  AP={ap:.3f}  "
              f"(TP={tp} FP={fp} FN={fn})")

    # ── Coefficients across folds ──
    print("\n── Feature coefficients (train on all, inspect) ──")
    X_all = pool_clean[FEATURE_COLS].values
    y_all = pool_clean["high_change"].values
    sc    = StandardScaler().fit(X_all)
    clf_full = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf_full.fit(sc.transform(X_all), y_all)
    for feat, coef in zip(FEATURE_COLS, clf_full.coef_[0]):
        print(f"  {feat:<22}  {coef:+.4f}")

    # ── Baseline: always predict high-change ──
    print("\n── Baselines ──")
    for net in networks:
        sub = pool_clean[pool_clean["network"] == net]
        base_rate = sub["high_change"].mean()
        # majority-class: always predict 0
        y_t = sub["high_change"].values
        if y_t.sum() > 0:
            maj_bal  = balanced_accuracy_score(y_t, np.zeros_like(y_t))
            rand_bal = 0.5
            print(f"  {net:<22}  majority-class bal_acc={maj_bal:.3f}  random={rand_bal:.3f}  "
                  f"base_rate={base_rate:.2f}")

    # ── Save results ──
    res_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("y_test","probs")}
                            for r in lono_results])
    res_df.to_csv("temporal_classifier_results.csv", index=False)
    print(f"\nSaved temporal_classifier_results.csv")

    # ── Figure ───────────────────────────────────────────────────────────────
    n_nets = len(lono_results)
    fig    = plt.figure(figsize=(14, 4 + 3 * n_nets))
    gs     = gridspec.GridSpec(n_nets + 1, 3, figure=fig, hspace=0.55, wspace=0.38)

    COLS = {"email-Eu-core": "#993C1D", "CollegeMsg": "#534AB7",
            "email-W3C": "#0F6E56"}

    # Row per network: timeline + PR curve
    for row_i, res in enumerate(lono_results):
        test_net = res["test_network"]
        sub      = pool_clean[pool_clean["network"] == test_net].reset_index(drop=True)
        sub_clean = sub.dropna(subset=FEATURE_COLS + ["nmi_next"]).reset_index(drop=True)
        color    = COLS.get(test_net, "#888")

        # Timeline
        ax_t = fig.add_subplot(gs[row_i, :2])
        ax_t.plot(sub_clean.index, sub_clean["nmi_next"],
                  color="#333", lw=1.3, marker="o", ms=3, label="NMI(t,t+1) actual")
        ax_t.plot(sub_clean.index, res["probs"],
                  color=color, lw=1.3, linestyle="--", label="P(high-change)")
        ax_t.axhline(HIGH_CHANGE_THRESH, color="#aaa", lw=0.8, linestyle=":")

        hc_idx = np.where(sub_clean["high_change"].values == 1)[0]
        ax_t.scatter(hc_idx, sub_clean.loc[hc_idx, "nmi_next"],
                     color="red", zorder=5, s=40, label="True high-change")
        ax_t.set_ylabel("NMI / P(change)", fontsize=8)
        ax_t.set_xlabel("Snapshot", fontsize=8)
        ax_t.set_title(f"Test: {test_net}  —  bal_acc={res['balanced_acc']:.3f}  "
                        f"recall={res['recall']:.2f}  precision={res['precision']:.2f}",
                        fontsize=9, fontweight="bold")
        ax_t.legend(fontsize=7, loc="upper right")
        ax_t.set_ylim(0.35, 1.08)
        ax_t.grid(alpha=0.12)

        # PR curve
        ax_pr = fig.add_subplot(gs[row_i, 2])
        if res["n_high"] > 0:
            prec_c, rec_c, _ = precision_recall_curve(res["y_test"], res["probs"])
            ax_pr.plot(rec_c, prec_c, color=color, lw=1.5)
            ax_pr.axhline(res["n_high"] / res["n_test"], color="#aaa",
                          lw=0.8, linestyle="--", label="Random baseline")
            ax_pr.set_xlabel("Recall", fontsize=8)
            ax_pr.set_ylabel("Precision", fontsize=8)
            ax_pr.set_title(f"Precision-Recall\nAP={res['avg_precision']:.3f}", fontsize=9)
            ax_pr.set_xlim(0, 1.05); ax_pr.set_ylim(0, 1.05)
            ax_pr.legend(fontsize=7)
            ax_pr.grid(alpha=0.12)

    # Bottom row: coefficient bar
    ax_c = fig.add_subplot(gs[n_nets, :])
    coefs = dict(zip(FEATURE_COLS, clf_full.coef_[0]))
    sf    = sorted(coefs, key=lambda k: abs(coefs[k]), reverse=True)
    bar_colors = ["#993C1D" if coefs[f] > 0 else "#534AB7" for f in sf]
    ax_c.barh(sf, [coefs[f] for f in sf], color=bar_colors, edgecolor="white")
    ax_c.axvline(0, color="#aaa", lw=1)
    ax_c.set_xlabel("Standardised coefficient (positive → predicts high-change)", fontsize=9)
    ax_c.set_title("Logistic coefficients — trained on all networks", fontsize=9)
    ax_c.grid(alpha=0.15, axis="x")

    plt.suptitle(
        "Cross-network binary classifier: predict high-change snapshots\n"
        "Leave-one-network-out CV — does the signal generalise?",
        fontweight="bold", fontsize=11, y=1.01
    )
    plt.savefig("fig_temporal_classifier.png", dpi=150, bbox_inches="tight")
    print("Saved fig_temporal_classifier.png")
    plt.show()
