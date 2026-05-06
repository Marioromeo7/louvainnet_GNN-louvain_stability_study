"""
temporal_correlations.py
Quick analysis of existing temporal_results.csv + richer instability metrics.

Two questions:
1. Which existing metrics correlate with NMI_structural_change?
2. Do richer instability metrics (mean_pairwise_nmi, K_cv) show variance
   where K_distinct_norm was saturated at 1.0?
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.stats import spearmanr
from collections import defaultdict

# ── Part 1: correlations on existing data ─────────────────────────────────────
print("=" * 60)
print("PART 1 — Correlations with NMI_structural_change")
print("=" * 60)

df = pd.read_csv("temporal_results.csv")
print(f"\nLoaded {len(df)} snapshots")

print("\nDescriptive stats (metrics that vary):")
vary_cols = ["modularity", "size_cv", "deg_cv", "density", "clustering",
             "K_louvain", "k_distinct_norm", "nmi_structural_change", "N", "E"]
print(df[vary_cols].describe().round(3).to_string())

valid = df.dropna(subset=["nmi_structural_change"])
print(f"\nSpearman vs NMI_structural_change (n={len(valid)} snapshots with prior):")
print(f"  {'Feature':<22}  {'r':>6}  {'p':>7}  sig")
print("  " + "-"*45)
for col in ["modularity", "size_cv", "deg_cv", "density", "clustering",
            "K_louvain", "N", "E"]:
    r, p = spearmanr(valid[col], valid["nmi_structural_change"])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {col:<22}  {r:+.3f}  {p:.4f}  {sig}")

# ── Part 2: richer instability metrics ───────────────────────────────────────
print("\n" + "=" * 60)
print("PART 2 — Richer instability metrics (subset of snapshots)")
print("Metrics: mean_pairwise_NMI, K_mean, K_std, K_cv across 50 seeds")
print("=" * 60)

TXT_PATH = "data/snap/email-Eu-core-temporal.txt"
WINDOW   = 30 * 24 * 3600
STEP     = 7  * 24 * 3600
N_PROBE  = 50

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

def make_snapshots(edges):
    t_min = min(e[2] for e in edges)
    t_max = max(e[2] for e in edges)
    snaps = []
    t = t_min
    while t + WINDOW <= t_max:
        G = nx.Graph()
        for u, v, ts in edges:
            if t <= ts < t + WINDOW:
                G.add_edge(u, v)
        if G.number_of_nodes() >= 30:
            snaps.append((t, G))
        t += STEP
    return snaps

def rich_instability(G, n_probe=N_PROBE):
    nodes = sorted(G.nodes())
    partitions = []
    k_vals = []
    for seed in range(n_probe):
        part = community_louvain.best_partition(G, random_state=seed)
        k_vals.append(len(set(part.values())))
        partitions.append([part[n] for n in nodes])

    # mean pairwise NMI (sample 20 pairs to keep it fast)
    rng = np.random.default_rng(0)
    idx = rng.choice(n_probe, size=(min(200, n_probe*(n_probe-1)//2), 2), replace=True)
    nmis = []
    for i, j in idx:
        if i != j:
            nmis.append(normalized_mutual_info_score(partitions[i], partitions[j]))
    mean_nmi = np.mean(nmis) if nmis else np.nan

    k_arr = np.array(k_vals)
    return {
        "mean_pairwise_nmi": mean_nmi,
        "K_mean":  k_arr.mean(),
        "K_std":   k_arr.std(),
        "K_cv":    k_arr.std() / k_arr.mean() if k_arr.mean() > 0 else 0,
        "K_range": k_arr.max() - k_arr.min(),
    }

if not os.path.exists(TXT_PATH):
    print("Dataset not found — skipping Part 2.")
else:
    edges = load_edges(TXT_PATH)
    snaps = make_snapshots(edges)
    print(f"Processing {len(snaps)} snapshots...\n")

    rich_rows = []
    for i, (t, G) in enumerate(snaps):
        lcc   = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(lcc).copy()
        if G_lcc.number_of_nodes() < 30:
            continue
        metrics = rich_instability(G_lcc)
        metrics["snapshot"] = i
        rich_rows.append(metrics)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(snaps)}  mean_pairwise_nmi={metrics['mean_pairwise_nmi']:.3f}"
                  f"  K_mean={metrics['K_mean']:.1f}  K_cv={metrics['K_cv']:.3f}")

    rdf = pd.DataFrame(rich_rows)
    merged = df.merge(rdf, on="snapshot", how="inner")

    print("\nDescriptive stats — richer metrics:")
    rcols = ["mean_pairwise_nmi", "K_mean", "K_std", "K_cv", "K_range"]
    print(merged[rcols].describe().round(4).to_string())

    valid2 = merged.dropna(subset=["nmi_structural_change"])
    print(f"\nSpearman vs NMI_structural_change (n={len(valid2)}):")
    print(f"  {'Feature':<24}  {'r':>6}  {'p':>7}  sig")
    print("  " + "-"*47)
    for col in rcols + ["modularity", "size_cv", "deg_cv"]:
        r, p = spearmanr(valid2[col], valid2["nmi_structural_change"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {col:<24}  {r:+.3f}  {p:.4f}  {sig}")

    print(f"\nSpearman vs k_distinct_norm (n={len(merged)}):")
    print(f"  {'Feature':<24}  {'r':>6}  {'p':>7}  sig")
    print("  " + "-"*47)
    for col in rcols:
        r, p = spearmanr(merged[col], merged["k_distinct_norm"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {col:<24}  {r:+.3f}  {p:.4f}  {sig}")

    rdf.to_csv("temporal_rich_metrics.csv", index=False)
    print("\nSaved temporal_rich_metrics.csv")
