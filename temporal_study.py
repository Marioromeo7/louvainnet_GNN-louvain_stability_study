"""
temporal_study.py — Louvain stability on temporal network snapshots

Downloads email-Eu-core-temporal from SNAP, builds 30-day sliding-window
snapshots, computes K_distinct_norm (algorithmic instability) and
NMI(t, t-1) (structural change rate) for each snapshot.

Research question:
  Does size_cv predict between-snapshot structural volatility, not just
  within-snapshot algorithmic instability? If yes, size_cv is a leading
  indicator of community boundary instability in time.

Run: conda run -n mario python temporal_study.py
Outputs: temporal_results.csv, fig_temporal_study.png
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
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.stats import spearmanr

# ── Config ────────────────────────────────────────────────────────────────────
URL        = "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz"
GZ_PATH    = "data/snap/email-Eu-core-temporal.txt.gz"
TXT_PATH   = "data/snap/email-Eu-core-temporal.txt"
OUT_CSV    = "temporal_results.csv"
OUT_FIG    = "fig_temporal_study.png"

WINDOW_SEC = 30 * 24 * 3600   # 30-day sliding window
STEP_SEC   = 7  * 24 * 3600   # 7-day step between snapshots
N_PROBE    = 50                # seeds per snapshot for K_distinct_norm
ARI_THRESH = 0.9999            # two partitions identical iff ARI > this
MIN_NODES  = 30                # skip snapshots smaller than this

# ── Download ──────────────────────────────────────────────────────────────────
def download():
    os.makedirs("data/snap", exist_ok=True)
    if os.path.exists(TXT_PATH):
        print(f"Dataset already at {TXT_PATH}")
        return
    print(f"Downloading {URL} ...")
    try:
        urllib.request.urlretrieve(URL, GZ_PATH)
        with gzip.open(GZ_PATH, "rb") as f_in, open(TXT_PATH, "wb") as f_out:
            f_out.write(f_in.read())
        os.remove(GZ_PATH)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Place email-Eu-core-temporal.txt manually in data/snap/ and re-run.")
        raise

# ── Parse edge list ───────────────────────────────────────────────────────────
def load_edges(path):
    edges = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            u, v, t = int(parts[0]), int(parts[1]), int(parts[2])
            if u != v:
                edges.append((u, v, t))
    return edges

# ── Build 30-day sliding-window snapshots ────────────────────────────────────
def make_snapshots(edges):
    t_min = min(e[2] for e in edges)
    t_max = max(e[2] for e in edges)
    snaps = []
    t = t_min
    while t + WINDOW_SEC <= t_max:
        G = nx.Graph()
        for u, v, ts in edges:
            if t <= ts < t + WINDOW_SEC:
                G.add_edge(u, v)
        if G.number_of_nodes() >= MIN_NODES:
            snaps.append((t, G))
        t += STEP_SEC
    return snaps

# ── Structural metrics (one Louvain run, seed=42) ─────────────────────────────
def structural_metrics(G):
    part = community_louvain.best_partition(G, random_state=42)
    comms = defaultdict(list)
    for node, c in part.items():
        comms[c].append(node)
    sizes   = [len(v) for v in comms.values()]
    size_cv = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0.0
    mod     = community_louvain.modularity(part, G)
    K       = len(comms)
    density = nx.density(G)
    degs    = [d for _, d in G.degree()]
    deg_cv  = np.std(degs) / np.mean(degs) if np.mean(degs) > 0 else 0.0
    clust   = nx.average_clustering(G)
    return part, mod, size_cv, K, density, deg_cv, clust

# ── K_distinct_norm (50 seeds) ────────────────────────────────────────────────
def k_distinct_norm(G, n_probe=N_PROBE):
    nodes  = sorted(G.nodes())
    seen   = []
    for seed in range(n_probe):
        part   = community_louvain.best_partition(G, random_state=seed)
        labels = [part[n] for n in nodes]
        is_new = all(adjusted_rand_score(prev, labels) <= ARI_THRESH for prev in seen)
        if is_new:
            seen.append(labels)
    return len(seen) / n_probe

# ── Structural change NMI(t, t-1) ────────────────────────────────────────────
def structural_change(part_prev, nodes_prev, part_curr, nodes_curr):
    common = sorted(set(nodes_prev) & set(nodes_curr))
    if len(common) < 10:
        return np.nan
    a = [part_prev[n] for n in common]
    b = [part_curr[n]  for n in common]
    return normalized_mutual_info_score(a, b)

# ── Decision rule (from static study) ────────────────────────────────────────
def predict_stability(mod, size_cv):
    if mod > 0.6 and size_cv < 0.3:
        return "stable"
    if mod < 0.4 or size_cv > 1.0:
        return "unstable"
    return "intermediate"

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    download()
    edges = load_edges(TXT_PATH)
    print(f"Loaded {len(edges):,} temporal edges")

    snaps = make_snapshots(edges)
    print(f"Built {len(snaps)} snapshots  (window={WINDOW_SEC//86400}d, step={STEP_SEC//86400}d)")

    rows       = []
    prev_part  = None
    prev_nodes = None

    for i, (t_start, G) in enumerate(snaps):
        lcc   = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(lcc).copy()
        N, E  = G_lcc.number_of_nodes(), G_lcc.number_of_edges()

        if N < MIN_NODES:
            prev_part = prev_nodes = None
            continue

        print(f"  [{i+1:3d}/{len(snaps)}] N={N:4d}  E={E:6d}", end="  ", flush=True)

        part, mod, scv, K, dens, dcv, clust = structural_metrics(G_lcc)
        kdn  = k_distinct_norm(G_lcc)
        dnmi = structural_change(prev_part, prev_nodes, part, list(G_lcc.nodes())) \
               if prev_part is not None else np.nan
        pred = predict_stability(mod, scv)

        dnmi_str = f"{dnmi:.3f}" if not np.isnan(dnmi) else "nan"
        print(f"Q={mod:.3f}  size_cv={scv:.3f}  K_dn={kdn:.2f}  ΔNMI={dnmi_str}")

        rows.append({
            "snapshot":             i,
            "t_start":              t_start,
            "N":                    N,
            "E":                    E,
            "density":              dens,
            "deg_cv":               dcv,
            "clustering":           clust,
            "modularity":           mod,
            "size_cv":              scv,
            "K_louvain":            K,
            "k_distinct_norm":      kdn,
            "nmi_structural_change": dnmi,
            "prediction":           pred,
        })

        prev_part  = part
        prev_nodes = list(G_lcc.nodes())

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows → {OUT_CSV}")

    # ── Correlations ──────────────────────────────────────────────────────────
    valid = df.dropna(subset=["k_distinct_norm", "size_cv", "nmi_structural_change"])
    print("\n── Spearman correlations with K_distinct_norm ──")
    for col in ["size_cv", "modularity", "deg_cv", "density", "clustering"]:
        r, p = spearmanr(valid[col], valid["k_distinct_norm"])
        print(f"  {col:20s}  r={r:+.3f}  p={p:.4f}")

    print("\n── Spearman: size_cv vs structural change (NMI_t_t-1) ──")
    r_sc, p_sc = spearmanr(valid["size_cv"], valid["nmi_structural_change"])
    print(f"  size_cv vs NMI_change:  r={r_sc:+.3f}  p={p_sc:.4f}")
    r_kd, p_kd = spearmanr(valid["k_distinct_norm"], valid["nmi_structural_change"])
    print(f"  K_distinct vs NMI_change: r={r_kd:+.3f}  p={p_kd:.4f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    COLS = {"kdn": "#993C1D", "scv": "#534AB7", "mod": "#0F6E56",
            "dnmi": "#B8860B", "scatter": "#444"}

    x = df["snapshot"].values

    # Panel 1: K_distinct_norm over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, df["k_distinct_norm"], color=COLS["kdn"], marker="o", ms=3, lw=1.5)
    ax1.axhline(0.5, color="#aaa", lw=0.8, ls=":")
    ax1.set_ylabel("K_distinct_norm", fontsize=9)
    ax1.set_title("Algorithmic instability over time (50 seeds per snapshot)", fontsize=10)
    ax1.set_ylim(-0.05, 1.1)

    # Panel 2: size_cv over time
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(x, df["size_cv"], color=COLS["scv"], marker="o", ms=3, lw=1.5)
    ax2.set_ylabel("size_cv", fontsize=9)
    ax2.set_title("Community size heterogeneity", fontsize=10)

    # Panel 3: Modularity over time
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax3.plot(x, df["modularity"], color=COLS["mod"], marker="o", ms=3, lw=1.5)
    ax3.set_ylabel("Modularity Q", fontsize=9)
    ax3.set_title("Modularity", fontsize=10)

    # Panel 4: NMI structural change
    ax4 = fig.add_subplot(gs[2, :], sharex=ax1)
    ax4.plot(x, df["nmi_structural_change"], color=COLS["dnmi"], marker="o", ms=3, lw=1.5)
    ax4.set_ylabel("NMI(t, t−1)", fontsize=9)
    ax4.set_title("Structural change between consecutive snapshots (↓ = more change)", fontsize=10)
    ax4.set_ylim(0, 1.08)
    ax4.set_xlabel("Snapshot index", fontsize=9)

    # Panel 5: size_cv vs K_distinct scatter
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.scatter(df["size_cv"], df["k_distinct_norm"], c=COLS["scv"], s=25, alpha=0.7, edgecolors="white", lw=0.4)
    r, p = spearmanr(df["size_cv"].dropna(), df["k_distinct_norm"].dropna())
    ax5.set_xlabel("size_cv", fontsize=9)
    ax5.set_ylabel("K_distinct_norm", fontsize=9)
    ax5.set_title(f"size_cv vs algorithmic instability\nSpearman r={r:+.3f}, p={p:.3f}", fontsize=9)

    # Panel 6: size_cv vs NMI_change scatter
    ax6 = fig.add_subplot(gs[3, 1])
    vv  = df.dropna(subset=["size_cv", "nmi_structural_change"])
    ax6.scatter(vv["size_cv"], vv["nmi_structural_change"], c=COLS["dnmi"], s=25, alpha=0.7, edgecolors="white", lw=0.4)
    r2, p2 = spearmanr(vv["size_cv"], vv["nmi_structural_change"])
    ax6.set_xlabel("size_cv", fontsize=9)
    ax6.set_ylabel("NMI(t, t−1)", fontsize=9)
    ax6.set_title(f"size_cv vs structural change\nSpearman r={r2:+.3f}, p={p2:.3f}", fontsize=9)

    plt.suptitle("email-Eu-core-temporal: Louvain stability across 30-day snapshots",
                 fontweight="bold", fontsize=12, y=1.01)
    plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_FIG}")
    plt.show()
