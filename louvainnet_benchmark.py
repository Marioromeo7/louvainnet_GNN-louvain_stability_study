"""
LouvainNet — Multi-graph benchmark
Tests: Newman | Louvain(seed=42) | SVD+KMeans | SVD+GNN(inference-only)
Graphs: ego-Facebook, ca-GrQc, ca-HepTh
"""
import os, time, gzip, urllib.request, pickle, warnings
os.chdir('c:/Users/mario/Desktop/lovainnet')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering
from scipy.sparse.linalg import svds

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

SEED      = 42
SVD_DIM   = 16
EMBED_DIM = 64
RESOLUTIONS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
BASE_SEED   = 42

torch.manual_seed(SEED); np.random.seed(SEED)

# ── GNN ────────────────────────────────────────────────────────────────────────
class LouvainNet(nn.Module):
    def __init__(self, in_channels=17, hidden=64, embed_dim=EMBED_DIM, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, embed_dim)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.normalize(x, p=2, dim=1)

with open('louvainnet_results_v3.pkl', 'rb') as f:
    v3 = pickle.load(f)
model = LouvainNet()
model.load_state_dict(v3['model_state_dict'])
model.eval()
print(f'GNN loaded ({sum(p.numel() for p in model.parameters()):,} params)')

# ── Download helpers ───────────────────────────────────────────────────────────
def download_snap(url, dest_txt):
    if os.path.exists(dest_txt):
        print(f'  {dest_txt} already exists, skipping download')
        return
    gz = dest_txt + '.gz'
    print(f'  Downloading {url} ...')
    urllib.request.urlretrieve(url, gz)
    with gzip.open(gz, 'rb') as f_in, open(dest_txt, 'wb') as f_out:
        f_out.write(f_in.read())
    os.remove(gz)
    print(f'  Saved: {dest_txt}')

def load_edgelist(path):
    G = nx.Graph()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    return G

# ── Build features ─────────────────────────────────────────────────────────────
def build_features(G):
    nodes = sorted(G.nodes())
    n2i   = {n: i for i, n in enumerate(nodes)}
    N     = len(nodes)
    co    = np.zeros((N, N), dtype=np.float64)
    best_q, k_modpeak, k_baseline = -1.0, None, None
    for gamma in RESOLUTIONS:
        part = community_louvain.best_partition(G, resolution=gamma, random_state=BASE_SEED)
        k    = len(set(part.values()))
        q    = community_louvain.modularity(part, G)
        if abs(gamma - 1.0) < 1e-9:
            k_baseline = k
        if q > best_q:
            best_q = q; k_modpeak = k
        comms = sorted(set(part.values()))
        c2i   = {c: idx for idx, c in enumerate(comms)}
        ind   = np.zeros((N, len(comms)))
        for node, comm in part.items():
            ind[n2i[node], c2i[comm]] = 1.0
        co += ind @ ind.T
    co /= len(RESOLUTIONS)
    d = min(SVD_DIM, N - 1)
    U, s, _ = svds(co, k=d)
    svd_feat = U[:, np.argsort(-s)].astype(np.float32)
    deg = np.array([G.degree(n) for n in nodes], dtype=np.float32)
    x   = np.hstack([svd_feat, (deg / deg.max()).reshape(-1, 1)]).astype(np.float32)
    src, dst, ew = [], [], []
    for u, v in G.edges():
        i, j = n2i[u], n2i[v]
        w = float(co[i, j])
        src += [i, j]; dst += [j, i]; ew += [w, w]
    pyg = Data(
        x=torch.tensor(x),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(ew, dtype=torch.float).unsqueeze(1),
        num_nodes=N
    )
    return pyg, nodes, svd_feat, k_baseline, k_modpeak

# ── Evaluate one graph ──────────────────────────────────────────────────────────
def evaluate_graph(name, G):
    N = G.number_of_nodes()
    E = G.number_of_edges()
    print(f'\n{"="*60}')
    print(f'Graph: {name}  ({N:,} nodes, {E:,} edges)')

    # Newman (reference)
    t0 = time.time()
    comms_newman = list(nx.community.greedy_modularity_communities(G))
    t_newman = time.time() - t0
    part_newman = {node: i for i, c in enumerate(comms_newman) for node in c}
    nodes_sorted = sorted(G.nodes())
    y_ref = np.array([part_newman[n] for n in nodes_sorted])
    K_NEWMAN = len(comms_newman)
    print(f'  Newman: K={K_NEWMAN}, t={t_newman:.2f}s')

    # Build features
    t0 = time.time()
    pyg, nodes_inf, svd_feat, K_BASELINE, K_MODPEAK = build_features(G)
    t_cons = time.time() - t0
    K_USE = K_MODPEAK
    print(f'  Consensus: K_modpeak={K_MODPEAK}, K_baseline={K_BASELINE}, t={t_cons:.2f}s')

    # GNN forward
    t0 = time.time()
    with torch.no_grad():
        emb = model(pyg.x, pyg.edge_index).numpy()
    t_gnn_fwd = time.time() - t0

    # Louvain fixed seed
    lf_part  = community_louvain.best_partition(G, random_state=BASE_SEED)
    lf_labels = np.array([lf_part[n] for n in nodes_sorted])
    lf_nmi   = normalized_mutual_info_score(y_ref, lf_labels)
    K_LF     = len(set(lf_part.values()))

    # SVD + KMeans
    t0 = time.time()
    km = KMeans(n_clusters=K_USE, random_state=SEED, n_init=10)
    km_labels = km.fit_predict(svd_feat)
    t_km = time.time() - t0
    km_nmi = normalized_mutual_info_score(y_ref, km_labels)

    # SVD + GNN + KMeans (k-means on GNN embeddings — faster than SpectralClustering)
    t0 = time.time()
    km2 = KMeans(n_clusters=K_USE, random_state=SEED, n_init=10)
    gnn_labels = km2.fit_predict(emb)
    t_gnn_km = time.time() - t0
    gnn_nmi = normalized_mutual_info_score(y_ref, gnn_labels)

    print(f'  Louvain(seed=42): K={K_LF}, NMI={lf_nmi:.4f}')
    print(f'  SVD+KMeans:       K={K_USE}, NMI={km_nmi:.4f}, t={t_cons+t_km:.2f}s')
    print(f'  SVD+GNN+KMeans:   K={K_USE}, NMI={gnn_nmi:.4f}, t={t_cons+t_gnn_fwd+t_gnn_km:.2f}s')

    return {
        'graph': name, 'nodes': N, 'edges': E,
        'K_newman': K_NEWMAN, 'K_use': K_USE, 'K_lf': K_LF,
        'nmi_louvain_fixed': lf_nmi,
        'nmi_svd_km': km_nmi,
        'nmi_gnn_km': gnn_nmi,
        't_newman': t_newman,
        't_svd_km': t_cons + t_km,
        't_gnn_km': t_cons + t_gnn_fwd + t_gnn_km,
    }

# ── Download graphs ────────────────────────────────────────────────────────────
os.makedirs('data/snap', exist_ok=True)

datasets = [
    ('ego-Facebook',  'data/ego_facebook/facebook_combined.txt', None),
    ('ca-GrQc',       'data/snap/ca-GrQc.txt',
     'https://snap.stanford.edu/data/ca-GrQc.txt.gz'),
    ('ca-HepTh',      'data/snap/ca-HepTh.txt',
     'https://snap.stanford.edu/data/ca-HepTh.txt.gz'),
]

for name, path, url in datasets:
    if url:
        download_snap(url, path)

# ── Run benchmark ──────────────────────────────────────────────────────────────
results = []
for name, path, _ in datasets:
    if not os.path.exists(path):
        print(f'Skipping {name}: {path} not found')
        continue
    G = load_edgelist(path)
    row = evaluate_graph(name, G)
    results.append(row)

# ── Summary table ──────────────────────────────────────────────────────────────
print('\n\n' + '='*70)
print('BENCHMARK SUMMARY')
print('='*70)
df = pd.DataFrame(results)
cols = ['graph', 'nodes', 'edges', 'K_newman', 'K_use',
        'nmi_louvain_fixed', 'nmi_svd_km', 'nmi_gnn_km',
        't_newman', 't_svd_km', 't_gnn_km']
print(df[cols].to_string(index=False, float_format='{:.4f}'.format))
df.to_csv('benchmark_results.csv', index=False)
print('\nSaved: benchmark_results.csv')

# ── Figure ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
graphs = df['graph'].tolist()
x = np.arange(len(graphs))
w = 0.25

nmi_lf  = df['nmi_louvain_fixed'].tolist()
nmi_km  = df['nmi_svd_km'].tolist()
nmi_gnn = df['nmi_gnn_km'].tolist()

axes[0].bar(x - w, nmi_lf,  w, label='Louvain(seed=42)', color='#888888', edgecolor='white')
axes[0].bar(x,     nmi_km,  w, label='SVD+KMeans',       color='#E07B30', edgecolor='white')
axes[0].bar(x + w, nmi_gnn, w, label='SVD+GNN+KMeans',   color='#534AB7', edgecolor='white')
axes[0].set_xticks(x); axes[0].set_xticklabels(graphs)
axes[0].set_ylim(0, 1.15); axes[0].set_ylabel('NMI vs Newman')
axes[0].set_title('Quality (NMI vs Newman)', fontweight='bold')
axes[0].legend(fontsize=9)

t_n  = df['t_newman'].tolist()
t_km = df['t_svd_km'].tolist()
t_gn = df['t_gnn_km'].tolist()
axes[1].bar(x - w, t_n,  w, label='Newman',           color='#0F6E56', edgecolor='white')
axes[1].bar(x,     t_km, w, label='SVD+KMeans',       color='#E07B30', edgecolor='white')
axes[1].bar(x + w, t_gn, w, label='SVD+GNN+KMeans',   color='#534AB7', edgecolor='white')
axes[1].set_xticks(x); axes[1].set_xticklabels(graphs)
axes[1].set_ylabel('Wall-clock time (s)')
axes[1].set_title('Speed (single run)', fontweight='bold')
axes[1].legend(fontsize=9)

plt.suptitle('Multi-graph benchmark: Consensus SVD features vs Newman baseline',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('fig_benchmark.png', dpi=150, bbox_inches='tight')
print('Saved: fig_benchmark.png')
