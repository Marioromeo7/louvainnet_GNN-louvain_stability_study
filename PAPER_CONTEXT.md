# Paper Writing Context
## "Predicting Louvain Non-determinism from Graph Structure"

---

## One-line pitch

Community size heterogeneity — not modularity strength — is the dominant predictor of Louvain's
partition instability across 70 graphs spanning real social networks and three synthetic families.

---

## The problem (Introduction)

The Louvain algorithm is the dominant method for community detection in large networks: fast,
scalable, modularity-optimising. But it is non-deterministic — repeated runs on identical input
produce different partitions depending on the random seed.

The standard response is "just fix the seed." This is correct but incomplete. It raises a more
fundamental question: **when does the seed matter?** On some graphs Louvain is nearly
deterministic (K_distinct ≈ 0); on others, every seed produces a different partition
(K_distinct ≈ 1). No prior work has systematically measured what drives this difference.

Knowing when Louvain is unreliable has direct practical consequences: it tells you when you need
the more expensive but deterministic Newman (greedy modularity) algorithm instead.

---

## K_distinct metric (Methods)

**Definition:** Run Louvain N_probe=50 times with seeds 0..49. Count distinct partitions — two
partitions are identical iff ARI > 0.9999. Normalise: K_distinct_norm = distinct / N_probe.

- K_distinct_norm = 0.02 → near-perfectly stable (1 distinct partition found in 50 runs)
- K_distinct_norm = 1.0 → maximally unstable (every run gave a new partition)

This metric is cheap (50 Louvain runs), graph-agnostic, and interpretable.

---

## Dataset (Experiments)

**70 graphs total across four families:**

| Family | Count | Graph sizes | Purpose |
|---|---|---|---|
| Real SNAP | 7 | 4K–11K nodes | Ground truth for real social networks |
| SBM | 24 | 500 nodes | Vary modularity strength (pin/pout ratio) |
| Barabási-Albert | 18 | 300–800 nodes | Vary degree heterogeneity (m parameter) |
| Planted partition | 21 | 60–1,600 nodes | Vary community count and size |

**Real graphs used:**
- ego-Facebook: 4,039 nodes, 88,234 edges — K_distinct_norm=0.92
- ca-GrQc: 4,158 nodes, 13,428 edges — K_distinct_norm=1.00
- ca-HepTh: 8,638 nodes, 24,827 edges — K_distinct_norm=1.00
- ca-HepPh: 11,204 nodes, 117,649 edges — K_distinct_norm=1.00
- p2p-Gnutella08: 6,299 nodes, 20,776 edges — K_distinct_norm=1.00
- p2p-Gnutella09: 8,104 nodes, 26,008 edges — K_distinct_norm=1.00
- wiki-Vote: 7,066 nodes, 100,736 edges — K_distinct_norm=1.00

All datasets from Stanford SNAP (snap.stanford.edu).

---

## Results

### Correlation table (Spearman, n=70)

| Feature | r | p | Direction |
|---|---|---|---|
| size_cv (community size CV) | **+0.789** | <0.001 | unequal sizes → unstable |
| density | **−0.705** | <0.001 | denser → stable |
| deg_cv (degree CV) | **+0.688** | <0.001 | hubs → unstable |
| K_louvain | +0.625 | <0.001 | more communities → unstable |
| clustering coefficient | −0.541 | <0.001 | more triangles → stable |
| modularity Q | −0.460 | <0.001 | higher Q → stable |
| K_per_N | +0.076 | 0.533 | NOT significant |

### The headline finding

`size_cv` (r=0.789) dominates over modularity Q (r=0.460).

Prior intuition says "high modularity = stable Louvain." The data says **"equal-sized communities
= stable"** is a stronger and more specific claim. A graph can have high Q with unequal
community sizes and still be maximally unstable.

**Mechanistic interpretation:** When community sizes are unequal, boundary nodes between a large
community and a small one face ambiguous local incentives — their assignment changes the modularity
gain differently depending on processing order, which Louvain's random initialisation varies.

### SBM sweep (controlled experiment)

Across 24 SBM graphs varying pin and pout:
- pin=0.30, pout=0.005 → Q=0.737, K_distinct_norm=0.02 (stable)
- pin=0.10, pout=0.02  → Q=0.351, K_distinct_norm=0.90 (unstable)
- pin=0.08, pout=0.02  → Q=0.268, K_distinct_norm=1.00 (maximally unstable)

Clear threshold: Q < ~0.35 → K_distinct_norm > 0.8.

### BA sweep (degree heterogeneity)

- BA m=1 (tree-like, high Q): K_distinct_norm=0.06–0.80 (stable to moderate)
- BA m≥2 (hub-dominated, low Q): K_distinct_norm=1.00 (all maximally unstable)

### Real graphs

All 7 real social networks scored K_distinct_norm ≥ 0.92. Real social networks simultaneously
have high degree heterogeneity (hubs), unequal community sizes, and modest modularity — all three
instability drivers present together. This explains why fixed-seed Louvain is necessary in
practice: any seed is effectively arbitrary.

---

## Practical decision rule (Discussion)

Given a new graph, run Louvain once with seed=42. Compute:
- Q (modularity of resulting partition)
- size_cv (coefficient of variation of community sizes)

If Q > 0.6 AND size_cv < 0.3 → fixed-seed Louvain is sufficient (K_distinct_norm ≈ 0.02)
If Q < 0.4 OR size_cv > 1.0 → expect maximal instability, use Newman or Leiden

This rule requires zero additional computation beyond a single Louvain run.

---

## Related work to cite

1. **Blondel et al. (2008)** — Louvain algorithm original paper. *J. Stat. Mech.*
2. **Newman (2006)** — Modularity and community structure. *PNAS* 103(23).
3. **Traag, Waltman & van Eck (2019)** — From Louvain to Leiden. *Scientific Reports.*
   (Leiden fixes connectivity issues in Louvain but does not address size heterogeneity instability)
4. **Lancichinetti & Fortunato (2012)** — Consensus clustering across Louvain runs. *Scientific Reports.*
   (Addresses non-determinism by consensus — our work explains WHY non-determinism occurs)
5. **Fortunato & Barthélemy (2007)** — Resolution limit of modularity. *PNAS.*
   (Modularity can't detect small communities — our size_cv finding is related: small communities
   next to large ones are the unstable ones)
6. **Sobolevsky & Belyi (2022)** — GNN-inspired algorithm for community detection. *Applied Network Science.*
   (Our target venue; their method solves Newman quality at Louvain speed — we characterise when
   Louvain is insufficient)
7. **Newman & Girvan (2004)** — Finding and evaluating community structure. *Physical Review E.*
8. **Strehl & Ghosh (2002)** — Cluster ensembles. *JMLR.* (NMI definition)

---

## Paper structure (6–8 pages, Applied Network Science format)

1. **Abstract** (150 words) — problem, metric, key finding (size_cv), practical rule
2. **Introduction** (1 page) — Louvain dominance, non-determinism problem, "fix the seed" is incomplete, research question
3. **Background** (0.5 page) — Louvain algorithm, modularity Q, prior non-determinism work (Lancichinetti)
4. **Methodology** (1 page) — K_distinct_norm definition, graph families, graph properties measured
5. **Results** (2 pages) — correlation table, SBM sweep figure, BA figure, real graphs table, the size_cv vs Q comparison
6. **Discussion** (1 page) — mechanistic interpretation of size_cv finding, practical decision rule, limitations (N_probe=50, undirected only, no temporal graphs)
7. **Conclusion** (0.5 page) — restate finding, practical implication, future work (extend to Leiden, directed graphs, larger graphs)

---

## Figures already generated

- `fig_stability_study.png` — scatter plots of top predictors + correlation bar chart + distribution by graph type + modularity vs stability scatter
- `fig_benchmark.png` — 4-method NMI comparison across 3 graphs (context for why stability matters)
- `stability_results.csv` — full 70-graph dataset, ready for table in paper

---

## Limitations to acknowledge honestly

1. N_probe=50 may underestimate true K_distinct for large graphs — some graphs may have more
   distinct partitions discoverable with 500+ probes
2. All graphs are undirected — directed graph stability is unexplored
3. Synthetic graphs are small (60–1,600 nodes) vs real graphs (4K–11K) — size confound exists
4. Decision rule thresholds (Q>0.6, size_cv<0.3) are empirical, not derived — need cross-validation
   on held-out graphs for a rigorous paper
5. The study is correlational — does not prove size_cv *causes* instability (though the mechanistic
   interpretation is plausible)

---

## What would make it stronger (future work / reviewer questions to pre-empt)

- Validate decision rule on 20+ held-out graphs not in the study
- Include directed graphs (convert to undirected or use directed Louvain variant)
- Extend to Leiden algorithm — does size_cv predict Leiden stability similarly?
- Larger real graphs (100K+ nodes) using sparse consensus matrix
- Theoretical derivation of why size_cv drives instability (boundary node modularity gain analysis)
