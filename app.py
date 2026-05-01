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
    ["Network Overview", "Centrality Analysis", "Community Detection", "Research: Non-determinism"],
    index=0
)

# ── Helper: community detection runner ───────────────────────────────────────
def run_community(G, algo, seed=42):
    nodes = sorted(G.nodes())
    if algo == "Newman (Greedy Modularity)":
        comms = list(nx.community.greedy_modularity_communities(G, seed=seed))
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
