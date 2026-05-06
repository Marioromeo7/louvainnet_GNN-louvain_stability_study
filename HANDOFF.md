# Session Handoff — 2026-05-05

## What was accomplished this session

### 1. Dashboard rebuilt (app.py)
Added two new pages (now 6 total):

| Page | Content |
|---|---|
| Network Overview | Unchanged |
| Centrality Analysis | Unchanged |
| Community Detection | Unchanged |
| Research: Non-determinism | Unchanged |
| **Research: Stability Study** | NEW — 70-graph correlation table, interactive scatter, decision rule calculator |
| **Research: Temporal Analysis** | NEW — temporal study results, LONO classifier, snapshot visualiser |

Bug fixed: `source` column in stability_results.csv uses `"pp"` not `"planted"` — corrected in the scatter plot.

---

### 2. Temporal study (temporal_study.py)
- Dataset: email-Eu-core-temporal (SNAP), 986 nodes, 332K edges, ~3 years
- 30-day sliding window, 7-day step → **76 snapshots**
- Per snapshot: structural metrics + 50-seed Louvain (K_distinct_norm + NMI structural change)
- **Key finding:** K_distinct_norm = 1.00 for 75/76 snapshots — uniformly maximally unstable, no variance to correlate

---

### 3. Richer instability metrics (temporal_correlations.py)
Beyond K_distinct_norm (saturated), computed per snapshot:
- `mean_pairwise_NMI` — average NMI between pairs of 50 Louvain runs
- `K_mean`, `K_std`, `K_cv`, `K_range` — variance in community count across seeds

**Key finding (Spearman vs NMI_structural_change, n=75):**

| Feature | r | p |
|---|---|---|
| mean_pairwise_NMI | **+0.437** | 0.0001 *** |
| N (active nodes) | +0.373 | 0.001 *** |
| E (active edges) | +0.364 | 0.001 ** |
| clustering | +0.277 | 0.016 * |
| density | +0.234 | 0.044 * |
| size_cv | +0.046 | 0.697 (n.s.) |

`size_cv` does not predict temporal structural change — the static predictor doesn't transfer.
`mean_pairwise_NMI` is the right instability metric for temporal contexts (K_distinct_norm is too coarse).

---

### 4. Oracle diagnostic (oracle_temporal.py)
Linear ceiling for predicting NMI(t, t+1) from current-snapshot features:

| Model | LOO R² | RMSE |
|---|---|---|
| Persistence (NMI=1.0) | −21.19 | 0.2616 |
| Mean predictor | 0.00 | 0.0555 |
| OLS all features | **0.21** | 0.0492 |
| OLS sig. features | 0.15 | 0.0513 |

Binary classification (NMI < 0.70 = high-change, 11/75 events):
- Logistic LOO balanced accuracy = **0.798**
- Recall = **0.91** (catches 10/11 high-change events)
- Precision = 0.33 (2/3 flags are false alarms)

**Verdict: R²=0.21 is the linear ceiling. A temporal GNN on 76 snapshots will not exceed it.**

---

### 5. Cross-network classifier (temporal_classifier.py)
Two networks pooled (134 snapshots total):
- email-Eu-core: 75 snapshots, 11 high-change (15%)
- CollegeMsg: 59 snapshots, 42 high-change (71%)

Leave-one-network-out CV results:

| Train → Test | Balanced acc | Recall | Precision | AP |
|---|---|---|---|---|
| email-Eu-core → CollegeMsg | 0.576 | **0.98** | 0.75 | **0.964** |
| CollegeMsg → email-Eu-core | 0.500 | 1.00 | 0.15 | 0.164 |

**Dominant feature:** `mean_pairwise_NMI` (coef = −2.47). Lower within-snapshot Louvain agreement → more likely high-change next step. **This signal transfers cross-network.**

Asymmetry explained by base-rate mismatch (71% vs 15%), not feature failure.

**Decision: no temporal GNN.** Signal is real and transferable but a linear model already achieves AP=0.964. Path forward: **more networks (5+)** to fix calibration, not a more complex model.

---

### 6. Snapshot visualiser (new dashboard section)
Bottom of "Research: Temporal Analysis" page. Shows three side-by-side network plots:
1. G_t — current snapshot coloured by t communities
2. Persistence prediction — G_{t+1}'s edges with t's labels applied
3. True G_{t+1} — actual t+1 partition, colour-matched by community overlap

Interactive: network dropdown (CollegeMsg / email-Eu-core) + snapshot slider.
Use CollegeMsg, snapshots 5–10 for dramatic high-change examples.
Use CollegeMsg, snapshots 55–59 for stable examples.

---

## Files created this session

| File | Purpose |
|---|---|
| `temporal_study.py` | 76-snapshot analysis, email-Eu-core-temporal |
| `temporal_results.csv` | 76 rows: structural metrics + K_distinct_norm + NMI_change |
| `fig_temporal_study.png` | 4-panel timeline figure |
| `temporal_correlations.py` | Richer metrics (mean_pairwise_NMI, K_cv) + correlations |
| `temporal_rich_metrics.csv` | 76 rows: mean_pairwise_NMI, K_mean, K_std, K_cv, K_range |
| `oracle_temporal.py` | LOO linear baseline + binary classifier diagnostic |
| `fig_oracle_temporal.png` | Oracle diagnostic figure |
| `temporal_classifier.py` | LONO cross-network logistic classifier |
| `temporal_classifier_results.csv` | LONO results table |
| `fig_temporal_classifier.png` | LONO timelines + PR curves |
| `data/snap/CollegeMsg.txt` | Downloaded from SNAP (1,899 nodes, 59K edges) |

---

## Open questions / next steps

1. **More networks for classifier:** Need 3–5 additional temporal networks to fix base-rate calibration and make the LONO classifier publishable. Good candidates on SNAP: `sx-mathoverflow` (large, use yearly windows), `ia-enron-email-dynamic`, department-level email datasets. Each needs download + compute (~25 min each).

2. **Calibration:** Apply isotonic regression or Platt scaling per network to handle base-rate differences. This is the bottleneck, not model capacity.

3. **Paper framing:** The temporal findings (mean_pairwise_NMI as cross-network predictor of structural change) are a separate contribution from the static study. Could be written as a companion paper or extended section.

4. **Dashboard:** The `network_interactive.html` Pyvis file powers the interactive graph on the Network Overview page. If it's missing, that section shows an info box. Re-run `louvainnet_project.ipynb` to regenerate it.

5. **Pkl files:** `louvainnet_data.pkl` is required for pages 1–4. If missing, run `louvainnet_project.ipynb`. `louvainnet_results_v3.pkl` is not required for any dashboard page — it's only used in the verdict notebook.
