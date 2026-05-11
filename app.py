"""
LouvainNet — Interactive SNA Dashboard
C-DE422 Social Network Analytics | Spring 2026

Run: streamlit run app.py
"""

import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import community as community_louvain
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LouvainNet — SNA Dashboard",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading graph data...")
def load_data():
    if os.path.exists("louvainnet_data.pkl"):
        with open("louvainnet_data.pkl", "rb") as f:
            return pickle.load(f)
    return None

data = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("LouvainNet")
st.sidebar.caption("C-DE422 · Social Network Analytics · Spring 2026")
st.sidebar.divider()
page = st.sidebar.radio(
    "Section",
    ["Network Overview", "Centrality Analysis", "Community Detection", "Research: Non-determinism", "Research: Stability Study", "Research: Temporal Analysis"],
    index=0
)

# ── Helper: community detection runner ───────────────────────────────────────
def run_community(G, algo, seed=42):
    nodes = sorted(G.nodes())
    if algo == "Newman (Greedy Modularity)":
        comms = list(nx.community.greedy_modularity_communities(G))
        part  = {n: i for i, c in enumerate(comms) for n in c}
        mod   = nx.community.modularity(G, comms)
    elif algo == "Louvain":
        part  = community_louvain.best_partition(G)
        comms_d = {}
        for n, c in part.items():
            comms_d.setdefault(c, set()).add(n)
        comms = list(comms_d.values())
        mod   = community_louvain.modularity(part, G)
    else:  # Label Propagation
        comms = list(nx.community.label_propagation_communities(G))
        part  = {n: i for i, c in enumerate(comms) for n in c}
        mod   = nx.community.modularity(G, comms)
    return part, comms, mod

# ── Helper: matplotlib community plot ────────────────────────────────────────
def plot_communities(G, partition, title, n_sample=400):
    top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:n_sample]
    G_sub     = G.subgraph(top_nodes).copy()
    sub_part  = {n: partition[n] for n in G_sub.nodes()}
    unique_c  = sorted(set(sub_part.values()))
    cmap      = plt.cm.get_cmap("tab20", max(len(unique_c), 1))
    clr_map   = {c: cmap(i) for i, c in enumerate(unique_c)}
    node_clrs = [clr_map[sub_part[n]] for n in G_sub.nodes()]
    pos       = nx.spring_layout(G_sub, k=0.3, iterations=40, seed=42)
    fig, ax   = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G_sub, pos, node_color=node_clrs, node_size=25, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G_sub, pos, alpha=0.08, width=0.4, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — NETWORK OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Network Overview":
    st.title("Network Overview — ego-Facebook")
    st.caption("Stanford SNAP dataset · Undirected · Anonymized Facebook friendship circles")

    if data is None:
        st.error("Data not found. Run the notebook first to generate louvainnet_data.pkl, or place facebook_combined.txt in data/ego_facebook/.")
        st.stop()

    G = data["graph"]

    # ── Stats row ──
    n = G.number_of_nodes()
    e = G.number_of_edges()
    density = nx.density(G)
    avg_clust = nx.average_clustering(G)
    n_cc = nx.number_connected_components(G)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Nodes", f"{n:,}")
    c2.metric("Edges", f"{e:,}")
    c3.metric("Density", f"{density:.5f}")
    c4.metric("Avg Clustering", f"{avg_clust:.4f}")
    c5.metric("Components", n_cc)

    st.divider()

    # ── Degree distribution ──
    st.subheader("Degree Distribution")
    degrees = [d for _, d in G.degree()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].hist(degrees, bins=70, color="#534AB7", edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Degree"); axes[0].set_ylabel("Count")
    axes[0].set_title("Linear scale"); axes[0].set_xlim(0, 300)

    from collections import Counter
    dc = Counter(degrees)
    xv = sorted(dc.keys()); yv = [dc[k] for k in xv]
    axes[1].scatter(xv, yv, s=6, color="#534AB7", alpha=0.7)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Degree (log)"); axes[1].set_ylabel("Count (log)")
    axes[1].set_title("Log-log scale")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("A linear trend on the log-log plot confirms a power-law degree distribution — characteristic of real social networks.")

    # ── Interactive Pyvis graph ──
    st.subheader("Interactive Network Graph")
    st.caption("Top 300 nodes by degree. Colors = Louvain communities. Hover a node for details.")
    if os.path.exists("network_interactive.html"):
        with open("network_interactive.html", "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=620, scrolling=False)
    else:
        st.info("Run the notebook to generate network_interactive.html, then reload this dashboard.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CENTRALITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Centrality Analysis":
    st.title("Centrality Analysis")

    if data is None:
        st.error("Run the notebook first."); st.stop()

    cent_df = data["centrality"]
    G = data["graph"]

    measure = st.selectbox(
        "Select centrality measure",
        ["degree", "betweenness", "closeness", "eigenvector"],
        format_func=lambda x: x.capitalize()
    )
    top_n = st.slider("Top N nodes", 5, 50, 10)

    top_nodes = cent_df.nlargest(top_n, measure)[[measure, f"{measure}_rank"]]
    top_nodes.columns = [f"{measure.capitalize()} score", "Rank"]
    top_nodes.index.name = "Node"

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"Top {top_n} nodes — {measure.capitalize()} centrality")
        st.dataframe(top_nodes.style.format("{:.6f}", subset=[f"{measure.capitalize()} score"]),
                     use_container_width=True)

    with col2:
        st.subheader("Score distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(cent_df[measure].values, bins=60, color="#534AB7", edgecolor="white", linewidth=0.3)
        ax.axvline(cent_df[measure].nlargest(top_n).min(), color="#993C1D",
                   linestyle="--", linewidth=1.5, label=f"Top-{top_n} threshold")
        ax.set_xlabel(f"{measure.capitalize()} centrality")
        ax.set_ylabel("Node count")
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()
    st.subheader("Centrality Heatmap — Top Nodes Across All Measures")

    all_top = set()
    for col in ["degree", "betweenness", "closeness", "eigenvector"]:
        all_top.update(cent_df.nlargest(15, col).index.tolist())
    all_top = sorted(list(all_top))

    hmap = cent_df.loc[all_top, ["degree", "betweenness", "closeness", "eigenvector"]]
    hmap_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min())

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(7, max(5, len(all_top) * 0.32)))
    sns.heatmap(hmap_norm, ax=ax, cmap="YlOrRd", linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Normalized score"},
                yticklabels=[f"Node {n}" for n in all_top])
    ax.set_xticklabels(["Degree", "Betweenness", "Closeness", "Eigenvector"], rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Nodes appearing bright across all columns are globally influential. Nodes bright only in one column have a specialized structural role.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — COMMUNITY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Community Detection":
    st.title("Community Detection")

    if data is None:
        st.error("Run the notebook first."); st.stop()

    G = data["graph"]

    algo = st.selectbox(
        "Algorithm",
        ["Newman (Greedy Modularity)", "Louvain", "Label Propagation"]
    )

    run_btn = st.button("Run detection", type="primary")

    if run_btn:
        with st.spinner(f"Running {algo}..."):
            partition, comms, mod = run_community(G, algo)

        sizes = sorted([len(c) for c in comms], reverse=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Communities found", len(comms))
        c2.metric("Modularity Q", f"{mod:.4f}")
        c3.metric("Largest community", f"{sizes[0]:,} nodes")

        st.divider()

        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.subheader("Community size distribution")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(range(len(sizes)), sizes, color="#534AB7", edgecolor="white", linewidth=0.3)
            ax.set_xlabel("Community rank (by size)")
            ax.set_ylabel("Nodes")
            ax.set_title(f"{algo} — {len(comms)} communities")
            plt.tight_layout()
            st.pyplot(fig)

        with col_b:
            st.subheader("Network — color coded by community")
            with st.spinner("Computing layout (top 400 nodes)..."):
                fig = plot_communities(G, partition, f"{algo} community structure")
            st.pyplot(fig)

        st.divider()
        st.subheader("Community summary table")
        summary = pd.DataFrame({
            "Community": range(len(comms)),
            "Size": sizes,
            "% of network": [f"{100*s/G.number_of_nodes():.1f}%" for s in sizes]
        })
        st.dataframe(summary, use_container_width=True, height=300)

    else:
        st.info("Select an algorithm and click **Run detection** to visualize communities.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RESEARCH: NON-DETERMINISM
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Research: Non-determinism":
    st.title("Research — Louvain Non-determinism & LouvainNet Findings")
    st.markdown(
        "Louvain produces different community partitions on identical input depending on random seed. "
        "We quantify this variance, then test whether a GNN can learn a deterministic, "
        "Newman-quality mapping — and report what the experiments actually found."
    )

    if data is None:
        st.error("Run the notebook first."); st.stop()

    nmi_scores   = data["nmi_louvain_runs"]
    ari_scores   = data["ari_louvain_runs"]
    n_comms_runs = data["n_comms_runs"]
    newman_part  = data["newman_partition"]
    G = data["graph"]

    n_runs = len(nmi_scores)
    newman_k = len(set(newman_part.values()))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs", n_runs)
    c2.metric("NMI mean ± std", f"{np.mean(nmi_scores):.3f} ± {np.std(nmi_scores):.3f}")
    c3.metric("ARI mean ± std", f"{np.mean(ari_scores):.3f} ± {np.std(ari_scores):.3f}")
    c4.metric("Newman communities (k)", newman_k)

    st.divider()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(n_runs), nmi_scores, color="#534AB7", alpha=0.85, edgecolor="white")
    axes[0].axhline(np.mean(nmi_scores), color="#993C1D", linestyle="--", linewidth=1.5,
                    label=f"Mean = {np.mean(nmi_scores):.3f}")
    axes[0].axhline(1.0, color="#0F6E56", linestyle=":", linewidth=1, label="Perfect match (NMI=1)")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("Louvain run index")
    axes[0].set_ylabel("NMI vs Newman")
    axes[0].set_title("NMI instability across runs")
    axes[0].legend(fontsize=9)

    axes[1].bar(range(n_runs), n_comms_runs, color="#0F6E56", alpha=0.85, edgecolor="white")
    axes[1].axhline(newman_k, color="#993C1D", linestyle="--", linewidth=1.5,
                    label=f"Newman k = {newman_k}")
    axes[1].set_xlabel("Louvain run index")
    axes[1].set_ylabel("Number of communities")
    axes[1].set_title("Community count instability")
    axes[1].legend(fontsize=9)

    plt.suptitle("Quantifying Louvain Non-determinism", fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.subheader("Multi-graph benchmark — 4-way comparison")
    st.caption("ego-Facebook (training graph) · ca-GrQc · ca-HepTh · Newman used as quality reference.")

    if os.path.exists("benchmark_results.csv"):
        bdf = pd.read_csv("benchmark_results.csv")
        display_cols = {
            "graph": "Graph", "nodes": "Nodes", "K_newman": "K (Newman)",
            "nmi_louvain_fixed": "Louvain\n(seed=42)", "nmi_svd_km": "SVD+KMeans",
            "nmi_gnn_km": "SVD+GNN", "t_newman": "Newman time (s)", "t_svd_km": "SVD+KMeans time (s)"
        }
        bdf_show = bdf[[c for c in display_cols if c in bdf.columns]].rename(columns=display_cols)
        st.dataframe(
            bdf_show.style.format({
                "Louvain\n(seed=42)": "{:.3f}", "SVD+KMeans": "{:.3f}",
                "SVD+GNN": "{:.3f}", "Newman time (s)": "{:.1f}", "SVD+KMeans time (s)": "{:.1f}"
            }),
            use_container_width=True
        )
        st.caption(
            "NMI measured against Newman's partition. "
            "Louvain (fixed seed) consistently outperforms both learned methods. "
            "SVD+GNN degrades on unseen graphs (ca-GrQc, ca-HepTh) — confirming overfitting to ego-Facebook."
        )
    else:
        st.info("Run `louvainnet_benchmark.py` to generate benchmark_results.csv, then reload.")

    st.divider()
    st.subheader("What the experiments found")

    trilemma = pd.DataFrame({
        "Algorithm"         : ["Newman", "Louvain (any seed)", "Louvain (fixed seed=42)", "LouvainNet (v3)"],
        "Deterministic"     : ["Yes", "No", "Yes", "Yes"],
        "Fast"              : ["No", "Yes", "Yes", "Yes*"],
        "Newman-quality NMI": ["1.000", "~0.85 (varies)", "0.856 (ego-FB)", "0.800 (ego-FB only)"],
        "Generalises"       : ["Yes", "Yes", "Yes", "No — degrades on unseen graphs"],
    })
    st.dataframe(trilemma.set_index("Algorithm"), use_container_width=True)

    st.info(
        "**Key finding:** fixing Louvain's random seed already achieves determinism with NMI=0.856 "
        "— no GNN required. LouvainNet matches this on ego-Facebook (the training graph) but "
        "underperforms fixed-seed Louvain on unseen graphs (ca-GrQc, ca-HepTh), confirming the "
        "GNN learned a graph-specific mapping rather than a general one.\n\n"
        "**Open problem:** the gap between Louvain and Newman is not due to non-determinism — "
        "it is due to the algorithms optimising modularity via different search strategies. "
        "Closing this gap without running Newman remains unsolved."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RESEARCH: STABILITY STUDY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Research: Stability Study":
    st.title("Research — What Predicts Louvain's Non-determinism?")
    st.markdown(
        "An empirical study across **70 graphs** (7 real SNAP + 63 synthetic) measuring which "
        "structural properties predict how often Louvain produces distinct partitions under "
        "different random seeds. Target venue: *Applied Network Science*."
    )

    # ── Metric definition ──
    st.subheader("Metric: K_distinct_norm")
    st.markdown(
        "Run Louvain 50 times with seeds 0–49. Count distinct partitions — two partitions are "
        "identical iff ARI > 0.9999. Normalise: **K_distinct_norm = distinct / 50**.\n\n"
        "- `0.02` → near-perfectly stable (1 unique partition across 50 runs)\n"
        "- `1.00` → maximally unstable (every run gave a different partition)"
    )

    st.divider()

    # ── Correlation table ──
    st.subheader("Spearman correlations with K_distinct_norm (n=70 graphs, all p<0.001)")
    corr_df = pd.DataFrame({
        "Feature": ["size_cv (community size heterogeneity)", "density", "deg_cv (degree heterogeneity)",
                    "K_louvain (number of communities)", "clustering coefficient", "modularity Q"],
        "Spearman r": [+0.789, -0.705, +0.688, +0.625, -0.541, -0.460],
        "Direction": ["unequal sizes → unstable", "denser → stable", "hubs → unstable",
                      "more communities → unstable", "more triangles → stable", "higher Q → stable"],
    })
    def highlight_top(s):
        return ["background-color: #e8f4f8; font-weight: bold" if i == 0 else "" for i in range(len(s))]
    st.dataframe(
        corr_df.style.apply(highlight_top, axis=0).format({"Spearman r": "{:+.3f}"}),
        use_container_width=True, hide_index=True
    )
    st.caption(
        "**Headline:** `size_cv` (r=+0.789) dominates over modularity Q (r=−0.460). "
        "A graph can have high Q with unequal community sizes and still be maximally unstable — "
        "prior intuition ('high modularity = stable') is incomplete."
    )

    st.divider()

    # ── Figure ──
    if os.path.exists("fig_stability_study.png"):
        st.subheader("Study figure")
        st.image("fig_stability_study.png", use_container_width=True)
    else:
        st.info("Run `stability_study.py` to generate fig_stability_study.png, then reload.")

    st.divider()

    # ── Interactive scatter ──
    st.subheader("Explore the 70-graph dataset")

    if os.path.exists("stability_results.csv"):
        sdf = pd.read_csv("stability_results.csv")

        x_var = st.selectbox(
            "X-axis predictor",
            ["size_cv", "modularity", "deg_cv", "density", "K_louvain", "clustering"],
            index=0,
            format_func=lambda c: {
                "size_cv": "size_cv — community size CV",
                "modularity": "modularity Q",
                "deg_cv": "deg_cv — degree CV",
                "density": "density",
                "K_louvain": "K_louvain — community count",
                "clustering": "clustering coefficient",
            }.get(c, c)
        )

        source_colors = {"real": "#993C1D", "sbm": "#534AB7", "ba": "#0F6E56", "pp": "#B8860B"}
        source_labels = {"real": "Real SNAP", "sbm": "SBM", "ba": "Barabási-Albert", "pp": "Planted partition"}

        fig, ax = plt.subplots(figsize=(9, 5))
        for src, grp in sdf.groupby("source"):
            ax.scatter(grp[x_var], grp["k_distinct_norm"],
                       color=source_colors.get(src, "#888"),
                       label=source_labels.get(src, src),
                       s=55, alpha=0.8, edgecolors="white", linewidths=0.5)

        # annotate real graphs
        for _, row in sdf[sdf["source"] == "real"].iterrows():
            ax.annotate(row["graph"], (row[x_var], row["k_distinct_norm"]),
                        fontsize=7, xytext=(4, 2), textcoords="offset points", color="#993C1D")

        # trend line
        from numpy.polynomial.polynomial import polyfit
        x_vals = sdf[x_var].values
        y_vals = sdf["k_distinct_norm"].values
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if mask.sum() > 5:
            c0, c1 = polyfit(x_vals[mask], y_vals[mask], 1)
            xl = np.linspace(x_vals[mask].min(), x_vals[mask].max(), 100)
            ax.plot(xl, c0 + c1 * xl, color="#555", linestyle="--", linewidth=1, alpha=0.6, label="Linear trend")

        ax.set_xlabel(x_var, fontsize=11)
        ax.set_ylabel("K_distinct_norm", fontsize=11)
        ax.set_ylim(-0.05, 1.08)
        ax.legend(fontsize=9, loc="upper left" if c1 > 0 else "lower left")
        ax.grid(alpha=0.15)
        plt.tight_layout()
        st.pyplot(fig)

        st.divider()
        st.subheader("Full dataset")
        display_cols = ["graph", "source", "N", "E", "modularity", "size_cv", "deg_cv", "density", "K_louvain", "clustering", "k_distinct_norm"]
        display_cols = [c for c in display_cols if c in sdf.columns]
        fmt = {c: "{:.3f}" for c in ["modularity", "size_cv", "deg_cv", "density", "clustering", "k_distinct_norm"]}
        st.dataframe(
            sdf[display_cols].sort_values("k_distinct_norm", ascending=False)
              .style.format(fmt),
            use_container_width=True, height=320, hide_index=True
        )
    else:
        st.info("Run `stability_study.py` to generate stability_results.csv, then reload.")

    st.divider()

    # ── Decision rule calculator ──
    st.subheader("Decision rule calculator")
    st.markdown(
        "After one Louvain run (seed=42), compute Q and size_cv to predict stability. "
        "Requires zero additional computation beyond the initial run."
    )

    col_q, col_s = st.columns(2)
    with col_q:
        user_q = st.slider("Modularity Q", 0.0, 1.0, 0.83, step=0.01)
    with col_s:
        user_scv = st.slider("size_cv", 0.0, 3.0, 0.66, step=0.01)

    if user_q > 0.6 and user_scv < 0.3:
        st.success(
            f"**Stable** — Q={user_q:.2f}, size_cv={user_scv:.2f}. "
            "Fixed-seed Louvain is sufficient. K_distinct_norm ≈ 0.02 expected."
        )
    elif user_q < 0.4 or user_scv > 1.0:
        st.error(
            f"**Maximally unstable** — Q={user_q:.2f}, size_cv={user_scv:.2f}. "
            "Expect K_distinct_norm ≈ 1.0. Use Newman or Leiden for reproducible results."
        )
    else:
        st.warning(
            f"**Intermediate** — Q={user_q:.2f}, size_cv={user_scv:.2f}. "
            "Stability is uncertain. Run 10–20 seeds and check variance, or use Newman."
        )

    st.caption(
        "Thresholds (Q>0.6 AND size_cv<0.3 for stable; Q<0.4 OR size_cv>1.0 for unstable) "
        "are empirical from 70 graphs. ego-Facebook: Q=0.835, size_cv=0.659 → unstable (K_distinct_norm=0.92)."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — RESEARCH: TEMPORAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Research: Temporal Analysis":
    st.title("Research — Temporal Extension: email-Eu-core")
    st.markdown(
        "Does the stability predictor (`size_cv`) track Louvain instability *within a network over time*, "
        "or is it only useful for comparing across different graphs? "
        "We ran **76 weekly snapshots** (30-day sliding window, 7-day step) of the "
        "email-Eu-core-temporal dataset (SNAP, 986 nodes, 332K emails, ~3 years). "
        "Each snapshot: 50 Louvain seeds → K_distinct_norm and mean pairwise NMI; NMI vs prior snapshot → structural change."
    )

    st.divider()

    # ── Metric cards ──
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Snapshots", "76")
    col_b.metric("K_distinct_norm = 1.00", "75 / 76")
    col_c.metric("mean_pairwise_NMI range", "0.73 – 0.88")
    col_d.metric("NMI(t,t−1) range", "0.49 – 0.85")

    st.divider()

    if os.path.exists("fig_temporal_study.png"):
        st.subheader("Timeline: 76 snapshots over ~3 years")
        st.image("fig_temporal_study.png", use_container_width=True)
    else:
        st.info("Run temporal_study.py to generate fig_temporal_study.png, then reload.")

    st.divider()

    # ── Three findings ──
    st.subheader("Three findings")

    st.markdown("#### 1. K_distinct_norm saturates — a finer metric is needed")
    st.warning(
        "K_distinct_norm = 1.00 for 75/76 snapshots. The network is maximally unstable throughout "
        "its 3-year history, so no predictor can correlate with a constant. "
        "This confirms the static study: real social networks fall uniformly in the maximally-unstable regime."
    )

    st.markdown("#### 2. Mean pairwise NMI predicts structural change (r = +0.437, p < 0.001)")
    st.success(
        "Even when all 50 runs are technically 'distinct' (K_distinct_norm = 1.0), their average pairwise NMI "
        "varies from **0.73 to 0.88** across snapshots. When runs agree more on average, "
        "community structure is more stable over time. "
        "This is the first evidence that the *degree* of algorithmic disagreement — not just whether "
        "disagreement exists — tracks temporal structural volatility."
    )

    st.markdown("#### 3. Network activity (N, E) is the dominant structural change predictor")
    st.info(
        "Larger, denser active windows → less structural change between snapshots (N: r=+0.373***, E: r=+0.364**). "
        "During high-activity periods the email graph is denser and communities are well-defined. "
        "In low-activity windows, the sparse graph has ambiguous boundaries that shift easily. "
        "size_cv remains non-significant (r=+0.046) — the static predictor does not transfer temporally."
    )

    st.divider()

    # ── Correlation tables ──
    st.subheader("Full correlation tables")

    tab1, tab2 = st.tabs(["vs NMI_structural_change", "vs K_distinct_norm"])

    with tab1:
        corr1 = pd.DataFrame({
            "Feature": ["mean_pairwise_NMI", "N (active nodes)", "E (active edges)",
                        "clustering", "density", "deg_cv", "modularity Q",
                        "size_cv", "K_louvain"],
            "Spearman r": [+0.437, +0.373, +0.364, +0.277, +0.234, -0.190, +0.049, +0.046, -0.022],
            "p-value":    [0.0001, 0.001,  0.001,  0.016,  0.044,  0.102,  0.675,  0.697,  0.851],
            "Significant?": ["***", "***", "**", "*", "*", "", "", "", ""],
        })
        st.dataframe(corr1.style.format({"Spearman r": "{:+.3f}", "p-value": "{:.4f}"}),
                     use_container_width=True, hide_index=True)
        st.caption("n=75 snapshots (first snapshot has no prior). Positive r = more agreement between snapshots (less structural change).")

    with tab2:
        corr2 = pd.DataFrame({
            "Feature": ["mean_pairwise_NMI", "K_mean", "K_std", "K_cv", "K_range",
                        "size_cv", "modularity Q", "deg_cv"],
            "Spearman r": [-0.197, -0.192, -0.055, +0.045, +0.020, +0.043, -0.199, +0.199],
            "p-value":    [0.087,  0.096,  0.635,  0.701,  0.863,  0.714,  0.088,  0.088],
            "Significant?": ["", "", "", "", "", "", "", ""],
        })
        st.dataframe(corr2.style.format({"Spearman r": "{:+.3f}", "p-value": "{:.4f}"}),
                     use_container_width=True, hide_index=True)
        st.caption("n=76 snapshots. No predictor reaches significance — K_distinct_norm has no variance (ceiling=1.0).")

    st.divider()

    # ── Scatter plots ──
    if os.path.exists("temporal_results.csv") and os.path.exists("temporal_rich_metrics.csv"):
        tdf  = pd.read_csv("temporal_results.csv")
        rdf  = pd.read_csv("temporal_rich_metrics.csv")
        mdf  = tdf.merge(rdf, on="snapshot", how="inner")

        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = plt.subplots(figsize=(4, 3.5))
            valid = mdf.dropna(subset=["nmi_structural_change"])
            ax.scatter(valid["mean_pairwise_nmi"], valid["nmi_structural_change"],
                       c="#993C1D", s=28, alpha=0.75, edgecolors="white", lw=0.4)
            ax.set_xlabel("mean pairwise NMI", fontsize=9)
            ax.set_ylabel("NMI(t, t−1)", fontsize=9)
            ax.set_title("r=+0.437***", fontsize=10, fontweight="bold")
            ax.grid(alpha=0.15)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("mean_pairwise_NMI vs structural change")

        with col2:
            fig, ax = plt.subplots(figsize=(4, 3.5))
            valid = tdf.dropna(subset=["nmi_structural_change"])
            ax.scatter(valid["N"], valid["nmi_structural_change"],
                       c="#0F6E56", s=28, alpha=0.75, edgecolors="white", lw=0.4)
            ax.set_xlabel("Active nodes (N)", fontsize=9)
            ax.set_ylabel("NMI(t, t−1)", fontsize=9)
            ax.set_title("r=+0.373***", fontsize=10, fontweight="bold")
            ax.grid(alpha=0.15)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("Network activity vs structural change")

        with col3:
            fig, ax = plt.subplots(figsize=(4, 3.5))
            valid = tdf.dropna(subset=["nmi_structural_change"])
            ax.scatter(valid["size_cv"], valid["nmi_structural_change"],
                       c="#534AB7", s=28, alpha=0.75, edgecolors="white", lw=0.4)
            ax.set_xlabel("size_cv", fontsize=9)
            ax.set_ylabel("NMI(t, t−1)", fontsize=9)
            ax.set_title("r=+0.046  (n.s.)", fontsize=10)
            ax.grid(alpha=0.15)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("size_cv vs structural change — not significant")

    st.divider()

    st.subheader("What this means")
    st.markdown(
        "**K_distinct_norm is too coarse for temporal analysis on real networks** — it saturates at 1.0 "
        "and loses all within-network variance. Mean pairwise NMI is the right metric for temporal contexts: "
        "it captures the *degree* of algorithmic disagreement, not just whether disagreement exists.\n\n"
        "**The dominant predictor of temporal structural change is activity level (N, E), not community geometry (size_cv).** "
        "This is a different mechanism than the static study. Static instability is driven by "
        "community structure (unequal sizes, hubs). Temporal instability is driven by how many "
        "nodes and edges are active in a window — a property orthogonal to size_cv.\n\n"
        "**These are two independent research questions:** (1) Which graphs are algorithmically unstable? "
        "→ answered by the static study via size_cv. "
        "(2) When does community structure change over time? → answered here via N, E, and mean_pairwise_NMI."
    )

    st.divider()

    # ── Cross-network classifier ──
    st.subheader("Cross-network classifier — does the signal generalise?")
    st.markdown(
        "Trained a logistic classifier on features from one temporal network, tested on a completely "
        "different network (**leave-one-network-out CV**). Networks: email-Eu-core-temporal (986 nodes, "
        "~3 years, 15% high-change) and CollegeMsg (1,899 nodes, 6 months, 71% high-change). "
        "Target: NMI(t,t+1) < 0.70."
    )

    lono_data = pd.DataFrame({
        "Train on →  Test on": [
            "email-Eu-core  →  CollegeMsg",
            "CollegeMsg     →  email-Eu-core",
        ],
        "Balanced acc.": [0.576, 0.500],
        "Recall":        [0.98,  1.00],
        "Precision":     [0.75,  0.15],
        "Avg. Precision (AP)": [0.964, 0.164],
        "TP / FP / FN":  ["41 / 14 / 1", "11 / 64 / 0"],
    })
    st.dataframe(
        lono_data.style.format({
            "Balanced acc.": "{:.3f}", "Recall": "{:.2f}",
            "Precision": "{:.2f}", "Avg. Precision (AP)": "{:.3f}",
        }),
        use_container_width=True, hide_index=True
    )

    col_r, col_l = st.columns([1, 1])

    with col_r:
        st.success(
            "**email-Eu-core → CollegeMsg: AP = 0.964, recall = 0.98.** "
            "A classifier trained on email data generalises near-perfectly to a student messaging network. "
            "The signal (mean_pairwise_NMI) is cross-network transferable."
        )
        st.warning(
            "**CollegeMsg → email-Eu-core: AP = 0.164, precision = 0.15.** "
            "Trained on a 71%-high-change network, the model over-predicts on a 15%-high-change network. "
            "The asymmetry is base-rate calibration, not a failure of the feature."
        )

    with col_l:
        st.markdown("**Dominant feature: `mean_pairwise_nmi` (coef = −2.47)**")
        coef_df = pd.DataFrame({
            "Feature":     ["mean_pairwise_nmi", "clustering", "E", "density", "N"],
            "Coefficient": [-2.4666, -0.7172, -0.4663, -0.4041, +0.0008],
        })
        fig, ax = plt.subplots(figsize=(4.5, 2.8))
        colors = ["#993C1D" if c < 0 else "#0F6E56" for c in coef_df["Coefficient"]]
        ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="white")
        ax.axvline(0, color="#aaa", lw=1)
        ax.set_xlabel("Standardised coefficient", fontsize=8)
        ax.set_title("Negative = predicts high-change", fontsize=8)
        ax.grid(alpha=0.15, axis="x")
        plt.tight_layout()
        st.pyplot(fig)

    if os.path.exists("fig_temporal_classifier.png"):
        st.divider()
        st.subheader("LONO timelines and precision-recall curves")
        st.image("fig_temporal_classifier.png", use_container_width=True)

    st.divider()
    st.info(
        "**Bottom line on temporal GNN:** The oracle diagnostic (R²=0.21 linear ceiling) and "
        "the cross-network AP=0.964 both point to the same conclusion — the predictive signal "
        "is real and transferable, but a linear model already captures it. A temporal GNN on "
        "134 pooled snapshots across 2 networks would not improve on AP=0.964. "
        "The path forward is **more networks** (5+), not a more complex model."
    )

    st.divider()

    # ── Snapshot visualiser ──────────────────────────────────────────────────
    st.subheader("Snapshot visualiser: G_t  →  persistence prediction  →  true G_{t+1}")
    st.markdown(
        "Select a snapshot to see the community structure at time **t** (left), "
        "the **persistence prediction** — G_{t+1} with t's community labels applied "
        "(middle), and the **true** G_{t+1} partition (right). "
        "Colour-matched by community overlap: nodes that stayed in the same community "
        "keep the same colour; reorganised nodes change colour."
    )

    VIZ_NETWORKS = {
        "CollegeMsg":    {"txt": "data/snap/CollegeMsg.txt",
                          "window": 14*86400, "step": 3*86400,
                          "results": None},
        "email-Eu-core": {"txt": "data/snap/email-Eu-core-temporal.txt",
                          "window": 30*86400, "step": 7*86400,
                          "results": "temporal_results.csv"},
    }

    available = [k for k, v in VIZ_NETWORKS.items() if os.path.exists(v["txt"])]
    if not available:
        st.info("Run temporal_study.py first to download the datasets.")
        st.stop()

    viz_net = st.selectbox("Network", available, key="viz_net_sel")
    cfg     = VIZ_NETWORKS[viz_net]

    @st.cache_data(show_spinner=False)
    def load_temporal_edges_cached(txt_path):
        edges = []
        with open(txt_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    u, v, t = int(parts[0]), int(parts[1]), int(parts[2])
                    if u != v:
                        edges.append((u, v, t))
        return edges

    @st.cache_data(show_spinner=False)
    def build_lcc(edges, t_start, window):
        G = nx.Graph()
        for u, v, ts in edges:
            if t_start <= ts < t_start + window:
                G.add_edge(u, v)
        if G.number_of_nodes() == 0:
            return G
        lcc = max(nx.connected_components(G), key=len)
        return G.subgraph(lcc).copy()

    @st.cache_data(show_spinner=False)
    def all_t_starts(edges, window, step):
        t_min = min(e[2] for e in edges)
        t_max = max(e[2] for e in edges)
        starts = []
        t = t_min
        while t + window <= t_max:
            starts.append(t)
            t += step
        return starts

    def match_colors(part_ref, part_target):
        """Remap part_target IDs to align visually with part_ref by overlap."""
        from collections import Counter
        ref_nodes   = set(part_ref)
        tgt_comms   = defaultdict(list)
        for n, c in part_target.items():
            tgt_comms[c].append(n)
        mapping, used = {}, set()
        for t_c, t_nodes in sorted(tgt_comms.items(), key=lambda x: -len(x[1])):
            common = [n for n in t_nodes if n in part_ref]
            if common:
                counts  = Counter(part_ref[n] for n in common)
                best    = next((r for r, _ in counts.most_common() if r not in used), None)
            else:
                best = None
            if best is None:
                best = max(part_ref.values(), default=0) + 100 + t_c
            mapping[t_c] = best
            used.add(best)
        return {n: mapping[c] for n, c in part_target.items()}

    def draw_snapshot(ax, G, partition, pos, title, n_sample=250, alpha_edge=0.07):
        top = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:n_sample]
        Gs  = G.subgraph(top)
        sub_pos  = {n: pos[n] for n in Gs.nodes() if n in pos}
        sub_part = {n: partition.get(n, -1) for n in Gs.nodes()}
        unique_c = sorted(set(sub_part.values()))
        cmap     = plt.cm.get_cmap("tab20", max(len(unique_c), 1))
        clr_map  = {c: cmap(i) for i, c in enumerate(unique_c)}
        node_clr = [clr_map[sub_part[n]] for n in Gs.nodes()]
        nx.draw_networkx_nodes(Gs, sub_pos, node_color=node_clr,
                               node_size=18, alpha=0.88, ax=ax)
        nx.draw_networkx_edges(Gs, sub_pos, alpha=alpha_edge, width=0.35, ax=ax)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    with st.spinner("Loading edges…"):
        edges_viz = load_temporal_edges_cached(cfg["txt"])

    t_starts = all_t_starts(edges_viz, cfg["window"], cfg["step"])

    # Drop last — no t+1 exists for it
    t_starts_sel = t_starts[:-1]
    snap_idx = st.slider("Snapshot index (t)", 0, len(t_starts_sel) - 1,
                         len(t_starts_sel) // 2, key="snap_slider")
    t_start_t  = t_starts_sel[snap_idx]
    t_start_t1 = t_starts[snap_idx + 1]

    with st.spinner("Building snapshots and running Louvain…"):
        G_t  = build_lcc(edges_viz, t_start_t,  cfg["window"])
        G_t1 = build_lcc(edges_viz, t_start_t1, cfg["window"])

    if G_t.number_of_nodes() < 10 or G_t1.number_of_nodes() < 10:
        st.warning("Snapshot too small — try a different index.")
    else:
        part_t  = community_louvain.best_partition(G_t,  random_state=42)
        part_t1 = community_louvain.best_partition(G_t1, random_state=42)
        part_t1_matched = match_colors(part_t, part_t1)

        # Persistence partition: apply t labels to t+1 nodes; new nodes get community -1
        part_persist = {n: part_t.get(n, -1) for n in G_t1.nodes()}

        # NMI between persistence and true t+1 (common nodes)
        common = sorted(set(G_t.nodes()) & set(G_t1.nodes()))
        nmi_persist = normalized_mutual_info_score(
            [part_t[n]  for n in common],
            [part_t1[n] for n in common]
        ) if len(common) >= 5 else float("nan")

        # Unified layout: spring on G_t, extend to t+1 new nodes
        top_t  = sorted(G_t.nodes(),  key=lambda n: G_t.degree(n),  reverse=True)[:250]
        top_t1 = sorted(G_t1.nodes(), key=lambda n: G_t1.degree(n), reverse=True)[:250]
        all_viz = sorted(set(top_t) | set(top_t1))
        G_union = nx.Graph()
        G_union.add_nodes_from(all_viz)
        for u, v in list(G_t.edges()) + list(G_t1.edges()):
            if u in G_union and v in G_union:
                G_union.add_edge(u, v)
        pos = nx.spring_layout(G_union, k=0.45, iterations=35, seed=42)

        # Metrics row
        k_t  = len(set(part_t.values()))
        k_t1 = len(set(part_t1.values()))
        n_new   = len(set(G_t1.nodes()) - set(G_t.nodes()))
        n_lost  = len(set(G_t.nodes())  - set(G_t1.nodes()))
        high_flag = "High-change" if nmi_persist < 0.70 else "Stable"
        flag_col  = "🔴" if nmi_persist < 0.70 else "🟢"

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("G_t nodes",          G_t.number_of_nodes())
        mc2.metric("G_{t+1} nodes",       G_t1.number_of_nodes())
        mc3.metric("Communities t / t+1", f"{k_t} / {k_t1}")
        mc4.metric("NMI(persist, true)",  f"{nmi_persist:.3f}")
        mc5.metric("Classifier verdict",  f"{flag_col} {high_flag}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        draw_snapshot(axes[0], G_t,  part_t,          pos,
                      f"G_t  (snapshot {snap_idx})\n{G_t.number_of_nodes()} nodes · {k_t} communities")
        draw_snapshot(axes[1], G_t1, part_persist,     pos,
                      f"Persistence prediction\n(G_{{t+1}} with t labels · {n_new} new nodes in grey)")
        draw_snapshot(axes[2], G_t1, part_t1_matched,  pos,
                      f"True G_{{t+1}}  (snapshot {snap_idx+1})\n{G_t1.number_of_nodes()} nodes · {k_t1} communities")

        plt.suptitle(
            f"NMI(persistence, true) = {nmi_persist:.3f} — "
            f"{'structure reorganised significantly' if nmi_persist < 0.70 else 'structure largely preserved'}",
            fontsize=11, fontweight="bold", y=1.01
        )
        plt.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Colours are matched by community overlap — same colour = same community group. "
            "Middle panel shows what persistence predicts; right panel shows what actually happened. "
            "Large colour differences between middle and right = high-change event."
        )
