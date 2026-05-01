"""
Louvain Stability Study
Research question: which graph properties predict Louvain non-determinism (K_distinct)?

Pipeline:
  1. Real SNAP graphs (diverse types)
  2. Synthetic graphs (SBM, BA, planted-partition) — vary one property at a time
  3. Correlation analysis + scatter plots
"""
import os, gzip, urllib.request, warnings, time
os.chdir('c:/Users/mario/Desktop/lovainnet')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain
from sklearn.metrics import adjusted_rand_score
from scipy import stats

N_PROBE  = 50   # Louvain seeds per graph
ARI_THRESH = 0.9999

# ── Helpers ────────────────────────────────────────────────────────────────────
def k_distinct(G, n_probe=N_PROBE):
    nodes = sorted(G.nodes())
    parts = []
    for s in range(n_probe):
        p = community_louvain.best_partition(G, random_state=s)
        parts.append(np.array([p[n] for n in nodes]))
    distinct = [0]
    for i in range(1, n_probe):
        if all(adjusted_rand_score(parts[j], parts[i]) < ARI_THRESH for j in distinct):
            distinct.append(i)
    return len(distinct) / n_probe   # normalised: 0=stable, 1=maximally unstable

def graph_stats(G):
    N  = G.number_of_nodes()
    E  = G.number_of_edges()
    m2 = 2 * E
    degs = [d for _, d in G.degree()]
    deg_mean = np.mean(degs)
    deg_std  = np.std(degs)

    part_fixed = community_louvain.best_partition(G, random_state=42)
    Q_fixed    = community_louvain.modularity(part_fixed, G)
    K_fixed    = len(set(part_fixed.values()))
    sizes      = list(pd.Series(list(part_fixed.values())).value_counts())
    size_cv    = np.std(sizes) / np.mean(sizes) if sizes else 0   # community size variation

    return {
        'N'          : N,
        'E'          : E,
        'density'    : nx.density(G),
        'deg_mean'   : deg_mean,
        'deg_cv'     : deg_std / deg_mean if deg_mean > 0 else 0,  # degree heterogeneity
        'clustering' : nx.average_clustering(G),
        'modularity' : Q_fixed,
        'K_louvain'  : K_fixed,
        'size_cv'    : size_cv,       # community size heterogeneity
        'K_per_N'    : K_fixed / N,   # community density
    }

def evaluate(name, G, source='real'):
    N, E = G.number_of_nodes(), G.number_of_edges()
    print(f'  {name}: {N:,} nodes, {E:,} edges', end=' ... ', flush=True)
    t0 = time.time()
    kd = k_distinct(G)
    stats_d = graph_stats(G)
    stats_d.update({'graph': name, 'source': source, 'k_distinct_norm': kd})
    print(f'K_distinct_norm={kd:.3f}  Q={stats_d["modularity"]:.3f}  t={time.time()-t0:.1f}s')
    return stats_d

def download_snap(url, dest):
    if os.path.exists(dest):
        return
    print(f'  Downloading {os.path.basename(dest)}...')
    gz = dest + '.gz'
    urllib.request.urlretrieve(url, gz)
    with gzip.open(gz, 'rb') as fi, open(dest, 'wb') as fo:
        fo.write(fi.read())
    os.remove(gz)

def load_snap(path, directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    if directed:
        G = G.to_undirected()
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.convert_node_labels_to_integers(G)

# ── 1. Real SNAP graphs ────────────────────────────────────────────────────────
os.makedirs('data/snap', exist_ok=True)

snap_datasets = [
    ('ego-Facebook',  'data/ego_facebook/facebook_combined.txt', None,       False),
    ('ca-GrQc',       'data/snap/ca-GrQc.txt',
     'https://snap.stanford.edu/data/ca-GrQc.txt.gz',                        False),
    ('ca-HepTh',      'data/snap/ca-HepTh.txt',
     'https://snap.stanford.edu/data/ca-HepTh.txt.gz',                       False),
    ('ca-HepPh',      'data/snap/ca-HepPh.txt',
     'https://snap.stanford.edu/data/ca-HepPh.txt.gz',                       False),
    ('p2p-Gnutella08','data/snap/p2p-Gnutella08.txt',
     'https://snap.stanford.edu/data/p2p-Gnutella08.txt.gz',                 True),
    ('p2p-Gnutella09','data/snap/p2p-Gnutella09.txt',
     'https://snap.stanford.edu/data/p2p-Gnutella09.txt.gz',                 True),
    ('wiki-Vote',     'data/snap/wiki-Vote.txt',
     'https://snap.stanford.edu/data/wiki-Vote.txt.gz',                      True),
]

print('=== REAL GRAPHS ===')
for name, path, url, directed in snap_datasets:
    if url:
        try:
            download_snap(url, path)
        except Exception as e:
            print(f'  Download failed for {name}: {e}')
            continue
    if not os.path.exists(path):
        print(f'  Skipping {name}: not found')
        continue

results = []
for name, path, url, directed in snap_datasets:
    if not os.path.exists(path):
        continue
    G = load_snap(path, directed)
    if G.number_of_nodes() > 25000:
        print(f'  Skipping {name}: too large ({G.number_of_nodes():,} nodes)')
        continue
    results.append(evaluate(name, G, source='real'))

# ── 2. Synthetic: SBM — vary modularity strength (pin vs pout) ────────────────
print('\n=== SYNTHETIC: SBM modularity sweep ===')
N_SBM, K_SBM = 500, 5
sizes_sbm = [N_SBM // K_SBM] * K_SBM
rng = np.random.RandomState(42)

for pin in [0.3, 0.2, 0.15, 0.10, 0.08, 0.06, 0.04, 0.03]:
    for pout in [0.005, 0.01, 0.02]:
        if pout >= pin:
            continue
        try:
            probs = [[pin if i == j else pout for j in range(K_SBM)]
                     for i in range(K_SBM)]
            G = nx.stochastic_block_model(sizes_sbm, probs, seed=42)
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            G = nx.convert_node_labels_to_integers(G)
            name = f'SBM_pin{pin}_pout{pout}'
            results.append(evaluate(name, G, source='sbm'))
        except Exception as e:
            print(f'  SBM failed pin={pin} pout={pout}: {e}')

# ── 3. Synthetic: BA — vary m (degree heterogeneity) ──────────────────────────
print('\n=== SYNTHETIC: BA m sweep ===')
for n in [300, 500, 800]:
    for m in [1, 2, 3, 5, 8, 12]:
        G = nx.barabasi_albert_graph(n, m, seed=42)
        name = f'BA_n{n}_m{m}'
        results.append(evaluate(name, G, source='ba'))

# ── 4. Synthetic: planted partition — vary community size ─────────────────────
print('\n=== SYNTHETIC: planted partition K sweep ===')
for k in [2, 3, 5, 8, 10, 15, 20]:
    for n_per in [30, 50, 80]:
        try:
            G = nx.planted_partition_graph(k, n_per, 0.5, 0.05, seed=42)
            G = nx.convert_node_labels_to_integers(G)
            name = f'PP_k{k}_n{n_per}'
            results.append(evaluate(name, G, source='pp'))
        except Exception as e:
            print(f'  PP failed k={k} n={n_per}: {e}')

# ── Analysis ───────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv('stability_results.csv', index=False)
print(f'\nTotal graphs evaluated: {len(df)}')
print(df[['graph','source','N','modularity','deg_cv','clustering','k_distinct_norm']].to_string(index=False))

# Correlation table
features = ['N', 'density', 'deg_mean', 'deg_cv', 'clustering', 'modularity',
            'K_louvain', 'size_cv', 'K_per_N']
target   = 'k_distinct_norm'

print('\n=== CORRELATIONS with k_distinct_norm ===')
corrs = []
for f in features:
    r, p = stats.pearsonr(df[f], df[target])
    rs, ps = stats.spearmanr(df[f], df[target])
    corrs.append({'feature': f, 'pearson_r': r, 'pearson_p': p,
                  'spearman_r': rs, 'spearman_p': ps})
    print(f'  {f:20s}  Pearson r={r:+.3f} (p={p:.3f})  Spearman r={rs:+.3f} (p={ps:.3f})')

corr_df = pd.DataFrame(corrs).sort_values('spearman_r', key=abs, ascending=False)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))

# Top predictors scatter plots
top_features = corr_df.head(6)['feature'].tolist()
src_colors   = {'real': '#993C1D', 'sbm': '#534AB7', 'ba': '#0F6E56', 'pp': '#E07B30'}

for i, feat in enumerate(top_features):
    ax = fig.add_subplot(3, 4, i + 1)
    for src in df['source'].unique():
        sub = df[df['source'] == src]
        ax.scatter(sub[feat], sub[target], c=src_colors.get(src, '#888888'),
                   alpha=0.7, s=25, label=src)
    r, p = stats.spearmanr(df[feat], df[target])
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel('K_distinct / N_probe', fontsize=9)
    ax.set_title(f'Spearman r={r:+.3f}  p={p:.3f}', fontsize=9, fontweight='bold')
    if i == 0:
        ax.legend(fontsize=8)

# Correlation bar chart
ax_bar = fig.add_subplot(3, 4, (7, 8))
colors_bar = ['#534AB7' if r > 0 else '#993C1D' for r in corr_df['spearman_r']]
ax_bar.barh(corr_df['feature'], corr_df['spearman_r'],
            color=colors_bar, edgecolor='white', alpha=0.85)
ax_bar.axvline(0, color='black', linewidth=0.8)
ax_bar.set_xlabel('Spearman correlation with K_distinct_norm')
ax_bar.set_title('Feature correlations', fontweight='bold')

# K_distinct distribution by source
ax_dist = fig.add_subplot(3, 4, (9, 10))
for src in df['source'].unique():
    sub = df[df['source'] == src]
    ax_dist.hist(sub[target], bins=15, alpha=0.6,
                 color=src_colors.get(src, '#888888'), label=src, edgecolor='white')
ax_dist.set_xlabel('K_distinct / N_probe (stability score)')
ax_dist.set_ylabel('Count')
ax_dist.set_title('Stability score distribution by graph type', fontweight='bold')
ax_dist.legend()

# Modularity vs K_distinct (main finding)
ax_main = fig.add_subplot(3, 4, (11, 12))
for src in df['source'].unique():
    sub = df[df['source'] == src]
    ax_main.scatter(sub['modularity'], sub[target],
                    c=src_colors.get(src, '#888888'), alpha=0.75, s=35, label=src)
r_mod, p_mod = stats.spearmanr(df['modularity'], df[target])
ax_main.set_xlabel('Modularity Q (fixed seed=42)')
ax_main.set_ylabel('K_distinct / N_probe')
ax_main.set_title(f'Modularity vs Stability  (r={r_mod:+.3f}, p={p_mod:.4f})',
                  fontweight='bold')
ax_main.legend()

plt.suptitle('Louvain Stability Study: what predicts non-determinism?',
             fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('fig_stability_study.png', dpi=150, bbox_inches='tight')
print('\nSaved: fig_stability_study.png')
print('Saved: stability_results.csv')
