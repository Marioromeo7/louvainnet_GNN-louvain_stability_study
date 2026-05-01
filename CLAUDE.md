# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install core dependencies
pip install -r requirements.txt

# Additional dependencies for the GNN notebook (not in requirements.txt)
pip install torch torch_geometric

# Run the Streamlit dashboard
streamlit run app.py

# Launch Jupyter for notebooks
jupyter notebook
```

The dashboard runs at `http://localhost:8501`.

## Pipeline

Three sequential phases, each producing a pkl that feeds the next:

**Phase 1 — SNA Analysis** (`louvainnet_project.ipynb`)
- Downloads ego-Facebook from SNAP (4,039 nodes, 88,234 edges), or reads from `data/ego_facebook/facebook_combined.txt` if SNAP is blocked
- Runs Parts A–D: graph statistics, centrality, community detection, and Louvain non-determinism quantification (20 runs, NMI/ARI vs Newman)
- Saves `louvainnet_data.pkl` and `network_interactive.html`

**Phase 2 — GNN Training** (active version: `louvainnet_gnn_v3.ipynb`)
- Requires `louvainnet_data.pkl` from Phase 1
- Trains a 3-layer GraphSAGE with supervised contrastive loss
- Saves `louvainnet_results_v3.pkl`
- Estimated CPU runtime: ~25–35 min

**Phase 3 — Verdict** (`louvainnet_verdict.ipynb`)
- Loads `louvainnet_data.pkl` + `louvainnet_results_v3.pkl`
- Runs determinism test (20 reps each) and conservative speed test (LouvainNet × K_distinct runs vs Newman × 1)
- Saves `fig_verdict.png`

**Dashboard** (`app.py`) reads only from `louvainnet_data.pkl`. Does not require the GNN.

## GNN Architecture

**Goal:** Match Newman quality deterministically, faster than Newman's O(N² log N) runtime.

**Input features per node** (17 dims):
- 16 SVD features: top-16 left singular vectors of the N×N multi-resolution co-cluster matrix
- 1 normalized degree

**Co-cluster matrix** (`build_consensus_svd`): Run Louvain at 8 fixed resolutions `[0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]` with `random_state=42`. Build N×N fraction matrix of how often each node pair co-clustered. Decompose via sparse SVD. Also tracks `k_modpeak` — the K at the resolution with highest standard modularity Q (grounded in same objective Newman optimizes).

**Model** (`LouvainNet`): 3-layer GraphSAGE → 64-dim L2-normalized embeddings. No output head, no K in the architecture.

**Loss** (`contrastive_loss`): Margin-based supervised contrastive. Same-community pairs pulled together; different-community pairs pushed apart by `margin=1.0`. 512 pairs sampled per graph per step.

**Inference** (K-free architecture, K recovered from data):
1. Build 8-resolution consensus → SVD features + `k_modpeak`
2. GNN forward → 64-dim embeddings
3. Spectral clustering with `k_modpeak` as K → final partition

**Determinism sources**: fixed `random_state=42` in all Louvain calls + fixed `random_state=42` in SpectralClustering.

## Training Data (v3)

~125 graphs across four sources:

| Source | Count | Purpose |
|---|---|---|
| Built-in + planted-partition | 41 | Controlled community structure |
| SBM (stochastic block model) | 52 | K range 3–15, zero generation failures |
| Barabási-Albert | ~30 | Power-law degree distribution — matches ego-Facebook's structural scale |
| ego-Facebook (transductive) | 1 | Direct signal on target graph; Newman labels from Phase 1 |

**Why BA graphs**: ego-Facebook has a power-law degree distribution. SBM and planted-partition graphs do not. BA graphs close the distributional gap.

**Why transductive**: the GNN must generalize from small synthetic graphs to 4,039-node real networks. Including ego-Facebook in training gives the model direct signal on the target domain without changing the architecture.

## K Estimation: Modularity-Peak Scan

v2 used K from Louvain at `resolution=1.0` (gave K=15 vs Newman's 13). v3 replaces this with `k_modpeak`: the K returned by Louvain at the resolution that maximizes standard modularity Q across all 8 resolutions. Since Newman also maximizes Q, `k_modpeak` is structurally consistent with Newman's answer at zero extra compute cost (modularity is evaluated on the same partitions already built for the consensus matrix).

## Key Data Contracts

- `louvainnet_data.pkl` keys: `graph`, `centrality`, `newman_partition`, `nmi_louvain_runs`, `ari_louvain_runs`, `n_comms_runs`, `nodes_list`
- `louvainnet_results_v3.pkl` keys: `model_state_dict`, `louvainnet_partition`, `nmi_louvainnet`, `ari_louvainnet`, `k_estimated`, `best_val_nmi`, `train_losses`, `val_losses`, `val_nmis`, `config` (config includes `n_ba`, `n_sbm`, `n_real`, `transductive`)
- `lib/` contains only Pyvis JS assets — no Python code

## Dataset

ego-Facebook (Stanford SNAP): 4,039 nodes, 88,234 undirected edges. Newman detects 13 communities. If auto-download fails (403), place `facebook_combined.txt` in `data/ego_facebook/` and re-run from Cell 3.

---

## Stability Study (publishable contribution)

**Research question:** Which graph properties predict Louvain's non-determinism?

**Script:** `stability_study.py` — run standalone with `conda run -n mario python stability_study.py`
**Outputs:** `stability_results.csv`, `fig_stability_study.png`

**Metric:** `K_distinct_norm` = number of distinct partitions found in 50 Louvain runs / 50. Two partitions are distinct if ARI < 0.9999.

**Key findings across 70 graphs (7 real SNAP + 63 synthetic):**

| Feature | Spearman r | p-value |
|---|---|---|
| size_cv (community size heterogeneity) | **+0.789** | <0.001 |
| density | −0.705 | <0.001 |
| deg_cv (degree heterogeneity) | +0.688 | <0.001 |
| K_louvain | +0.625 | <0.001 |
| modularity Q | −0.460 | <0.001 |

**The non-obvious finding:** `size_cv` beats modularity Q as the dominant predictor. Unequal community sizes are more destabilising than weak modularity per se. All 7 real SNAP graphs scored K_distinct_norm ≥ 0.92 (maximally unstable) due to simultaneous high degree heterogeneity and unequal community sizes.

**GNN verdict (why LouvainNet was abandoned):**
- Fixed-seed Louvain (seed=42) already gives NMI=0.856 — no GNN needed for determinism
- LouvainNet v3: NMI=0.800 on ego-Facebook (training graph), degrades to 0.536/0.342 on unseen graphs
- Oracle K experiment confirmed K estimation is not the bottleneck — the representation is
- Modularity matrix eigenvalue gap does not reliably indicate Newman's K on real graphs

**Target venue:** Applied Network Science (same venue as Sobolevsky & Belyi 2022).
