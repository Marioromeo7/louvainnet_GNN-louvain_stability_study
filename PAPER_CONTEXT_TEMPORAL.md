# Paper Writing Context
## "Algorithmic Disagreement Predicts Community Structure Drift in Temporal Networks"

---

## One-line pitch

The degree to which Louvain runs disagree on the current snapshot — measured by mean pairwise NMI
across seeds — predicts how much community structure will change at the next time step, and this
signal transfers across structurally different networks without retraining.

---

## Relationship to the companion paper

**Companion (Paper 1):** "Predicting Louvain Non-determinism from Graph Structure"
→ Answers: *Which graphs* are algorithmically unstable? Predictor: size_cv (r=+0.789).

**This paper (Paper 2):** Temporal extension.
→ Answers: *When* does community structure change over time in a network you are already monitoring?
Predictor: mean_pairwise_NMI (r=+0.437, AP=0.964 cross-network).

The two papers share a method (Louvain multi-seed analysis) but address orthogonal questions.
size_cv does not predict temporal change (r=+0.046, n.s.). mean_pairwise_NMI does not appear
as a discriminator between graph families in the static study because all real graphs are already
at the ceiling. The two predictors operate on different timescales and different units of analysis.

---

## The problem (Introduction)

Community detection in temporal networks faces two distinct instability problems that are
often conflated:

1. **Algorithmic instability** — on a fixed snapshot, different random seeds produce
   different partitions. Paper 1 characterises this.

2. **Structural change** — the community structure itself evolves between time steps.
   Even with a fixed seed, the partition at t+1 differs from t because the graph changed.

Prior work addresses these separately: consensus methods (Lancichinetti 2012) for (1);
dynamic community detection algorithms (Mucha et al. 2010, Rossetti & Cazabet 2018) for (2).
Neither predicts *when* structural change will occur from observable current-snapshot features.

**The gap this paper fills:** Can we predict, before running Louvain on the next snapshot,
whether the community structure is about to change significantly? If yes, we can schedule
computationally expensive deterministic methods (Newman) only when needed.

---

## Key metric: mean pairwise NMI

**Definition:** Run Louvain N_probe=50 times on the current snapshot G_t with seeds 0..49.
Sample 200 random pairs of runs. Compute their average NMI.

- mean_pairwise_NMI ≈ 1.0 → all seeds agree on the same partition (low algorithmic disagreement)
- mean_pairwise_NMI ≈ 0.3 → seeds produce very different partitions (high disagreement)

**Why this is better than K_distinct_norm for temporal analysis:**
K_distinct_norm (Paper 1) is binary per pair: distinct or not. On real social networks it saturates
at 1.0 — every run is technically "distinct." Mean pairwise NMI captures the *degree* of
disagreement: even when all 50 runs produce different partitions, some may be nearly identical
(NMI=0.95) while others are radically different (NMI=0.30). This within-network variance is what
predicts temporal structural change.

---

## Dataset

**Two temporal networks, processed as sliding-window snapshots:**

| Network | Nodes | Edges | Time span | Window | Step | Snapshots | High-change |
|---|---|---|---|---|---|---|---|
| email-Eu-core-temporal | 986 | 332K | ~3 years | 30 days | 7 days | 76 | 11 (15%) |
| CollegeMsg | 1,899 | 59K | ~6 months | 14 days | 3 days | 59 | 42 (71%) |

Source: Stanford SNAP (snap.stanford.edu).
LCC (largest connected component) used for each snapshot.
High-change: NMI(G_t, G_{t+1}) < 0.70 with seed=42 fixed.

---

## Results

### Single-network correlations (email-Eu-core, n=75 consecutive pairs)

| Feature | Spearman r | p | Note |
|---|---|---|---|
| mean_pairwise_NMI | **+0.437** | 0.0001 *** | Key predictor |
| N (active nodes) | +0.373 | 0.001 *** | Activity level |
| E (active edges) | +0.364 | 0.001 ** | Activity level |
| clustering | +0.277 | 0.016 * | Cohesion |
| density | +0.234 | 0.044 * | Cohesion |
| size_cv | +0.046 | 0.697 | NOT significant |
| modularity Q | +0.049 | 0.675 | NOT significant |

Positive r = more stable next step.

### Oracle diagnostic (linear ceiling)

LOO linear regression, email-Eu-core, n=75:
- Best LOO R²: 0.21 (OLS, all features)
- Binary classifier (NMI < 0.70 = high-change): LOO balanced accuracy = 0.798, recall = 0.91

The R²=0.21 is the linear signal ceiling on 76 snapshots. This rules out complex models
(temporal GNN) on this dataset — the bottleneck is data volume, not model capacity.

### Cross-network classifier (leave-one-network-out)

Logistic regression trained on one network, tested on the other.
Features: mean_pairwise_NMI, N, E, density, clustering.

| Train → Test | Balanced acc | Recall | Precision | AP |
|---|---|---|---|---|
| email-Eu-core → CollegeMsg | 0.576 | **0.98** | 0.75 | **0.964** |
| CollegeMsg → email-Eu-core | 0.500 | 1.00 | 0.15 | 0.164 |

**Headline: AP=0.964.** A classifier trained exclusively on academic email data predicts
high-change events in a student messaging network with 98% recall.

**Dominant feature:** mean_pairwise_NMI (standardised coef = −2.47). Lower within-snapshot
Louvain agreement → higher probability of structural change next step.

**Asymmetry:** CollegeMsg → email-Eu-core gives AP=0.164 (essentially random). This is a
base-rate calibration problem — CollegeMsg has 71% high-change vs email-Eu-core's 15%.
The classifier trained on high base-rate data over-predicts on low base-rate data. Fix:
isotonic regression or Platt scaling per network. This is a calibration problem, not a
feature quality problem.

---

## Mechanistic interpretation (Discussion)

**Why does mean_pairwise_NMI predict structural change?**

When Louvain runs disagree strongly on the current snapshot, the community boundaries are
ambiguous — nodes near the boundary between two communities get assigned inconsistently.
These boundary nodes are also the most likely to change community when the graph evolves:
new edges arrive, old edges dissolve, and the boundary shifts. High algorithmic disagreement
at t is therefore a structural signal that community boundaries are "soft" — which makes
them more likely to reorganise by t+1.

This is distinct from the size_cv mechanism in Paper 1. size_cv captures a static structural
property (unequal community sizes). mean_pairwise_NMI captures boundary softness, which is
dynamic and sensitive to the current edge configuration.

**Why does activity level (N, E) predict structural change?**

During high-activity windows (many emails, many active nodes), the email graph is denser and
communities are better-defined — the same core people are emailing each other and the partition
is coherent. During low-activity windows, the sparse active subgraph has weaker community
signal, making it more sensitive to which specific edges happen to occur that week.

---

## Practical decision rule (Discussion)

Given a monitoring window for a temporal network you are tracking:

1. Run Louvain with 10 seeds (not 50 — fast approximation of mean_pairwise_NMI)
2. Compute mean pairwise NMI between the 10 runs
3. If mean_pairwise_NMI < 0.75: expect high structural change next step → flag for Newman
4. Also track N — if N drops significantly from the prior window, flag regardless of NMI

This rule requires 10 Louvain runs (~10× a single run) and no additional data.

---

## Related work to cite

1. **Blondel et al. (2008)** — Louvain algorithm. *J. Stat. Mech.*
2. **Lancichinetti & Fortunato (2012)** — Consensus clustering. *Scientific Reports.*
   (Addresses algorithmic instability by aggregation — we predict *when* to aggregate)
3. **Mucha et al. (2010)** — Generalized modularity for temporal networks. *Science.*
   (Optimises community structure across time slices jointly — orthogonal to our prediction task)
4. **Rossetti & Cazabet (2018)** — Survey of dynamic community detection. *ACM CSUR.*
   (Comprehensive review; our contribution is prediction of change onset, not tracking)
5. **Holme & Saramäki (2012)** — Temporal networks review. *Physics Reports.*
6. **Peixoto & Rosvall (2017)** — Modelling temporal community structure. *Nature Comms.*
7. **Newman (2006)** — Modularity and community structure. *PNAS.*
8. **Sobolevsky & Belyi (2022)** — GNN community detection. *Applied Network Science.*
   (Target venue; Paper 1 companion)
9. **[Paper 1 — this work]** — Predicting Louvain non-determinism from graph structure.
   *Applied Network Science.* (companion paper — cite as "in the companion study, we showed…")

---

## Paper structure (6–8 pages, Applied Network Science format)

1. **Abstract** (150 words) — temporal community change prediction, mean_pairwise_NMI,
   cross-network AP=0.964, practical monitoring rule
2. **Introduction** (1 page) — two types of instability, the gap, research question
3. **Background** (0.5 page) — Louvain, mean_pairwise_NMI definition, prior temporal work
4. **Methodology** (1 page) — sliding window snapshots, feature extraction, LONO CV design
5. **Results** (2 pages) — single-network correlations, oracle R²=0.21, LONO table,
   snapshot visualisation of high-change vs stable event
6. **Discussion** (1 page) — mechanism (boundary softness), activity level interpretation,
   base-rate calibration problem, practical rule
7. **Conclusion** (0.5 page) — summary, future work (5+ networks, node-level, Leiden)

---

## Figures already generated

- `fig_temporal_study.png` — 4-panel timeline: K_distinct_norm, size_cv, Q, NMI_change over 76 snapshots
- `fig_oracle_temporal.png` — oracle diagnostic: actual vs predicted, residuals, coefficient bar, model table
- `fig_temporal_classifier.png` — LONO timelines + PR curves for both network folds
- Snapshot visualiser (dashboard, page 6) — G_t / persistence / true G_{t+1} side-by-side

---

## Limitations to acknowledge honestly

1. Only 2 networks — LONO with 2 folds is not a robust generalization claim.
   Minimum for a publishable generalization result: 5 networks across ≥2 domains.
2. Base-rate mismatch (71% vs 15%) makes calibration a confound — need isotonic
   correction or more diverse base-rate networks.
3. Window size is a free parameter — results may change with different window/step choices.
   Sensitivity analysis needed.
4. We predict *when* change happens, not *where* (which nodes change community). A node-level
   analysis of boundary nodes would strengthen the mechanistic claim.
5. mean_pairwise_NMI requires 10–50 Louvain runs per snapshot — more expensive than a
   single-run statistic. Cost analysis vs benefit needed for practical deployment.
6. Correlation not causation: mean_pairwise_NMI predicts change but the causal mechanism
   (boundary softness) is hypothesised, not directly measured.

---

## What would make it stronger (next steps before submission)

1. **5+ temporal networks** from different domains (academic email, social messaging,
   peer-to-peer, collaboration, citation). Candidates on SNAP: sx-mathoverflow (use yearly
   windows), ia-enron-email-dynamic, ia-primary-school-proximity.
2. **Calibration fix:** isotonic regression per network to correct base-rate mismatch.
   If calibrated, the CollegeMsg → email-Eu-core AP should improve substantially.
3. **Window sensitivity:** repeat analysis with 7-day and 60-day windows. If mean_pairwise_NMI
   signal holds, the predictor is robust to window choice.
4. **Node-level analysis:** identify "boundary nodes" (nodes that appear in different communities
   across seeds). Test whether boundary-node density predicts structural change better than
   graph-level mean_pairwise_NMI.
5. **Leiden comparison:** does mean_pairwise_NMI computed from Leiden (instead of Louvain)
   also predict change? If yes, the predictor generalises beyond Louvain.
