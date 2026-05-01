"""
Oracle-K experiment: SVD+KMeans and SVD+GNN with K=Newman's K
Tests whether K estimation is the bottleneck or the features themselves.
"""
import os, time, pickle, warnings
os.chdir('c:/Users/mario/Desktop/lovainnet')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

SEED = 42
SVD_DIM = 16
EMBED_DIM = 64
RESOLUTIONS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
BASE_SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

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

def load_graph(path):
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
    return nx.convert_node_labels_to_integers(G)

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
    U, s, _ = svds(co, k=min(SVD_DIM, N-1))
    svd_feat = U[:, np.argsort(-s)].astype(np.float32)
    deg = np.array([G.degree(n) for n in nodes], dtype=np.float32)
    x   = np.hstack([svd_feat, (deg / deg.max()).reshape(-1, 1)]).astype(np.float32)
    src, dst, ew = [], [], []
    for u, v in G.edges():
        i, j = n2i[u], n2i[v]
        src += [i, j]; dst += [j, i]; ew += [float(co[i,j])]*2
    pyg = Data(
        x=torch.tensor(x),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(ew, dtype=torch.float).unsqueeze(1),
        num_nodes=N
    )
    return pyg, nodes, svd_feat, k_baseline, k_modpeak

def evaluate(name, G):
    nodes = sorted(G.nodes())
    print(f'\n{"="*60}')
    print(f'{name}  ({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)')

    # Newman reference
    comms = list(nx.community.greedy_modularity_communities(G))
    part_newman = {node: i for i, c in enumerate(comms) for node in c}
    y_ref = np.array([part_newman[n] for n in nodes])
    K_NEWMAN = len(comms)

    # Features
    pyg, nodes_inf, svd_feat, K_BASELINE, K_MODPEAK = build_features(G)
    with torch.no_grad():
        emb = model(pyg.x, pyg.edge_index).numpy()

    # Louvain fixed seed
    lf = community_louvain.best_partition(G, random_state=BASE_SEED)
    lf_labels = np.array([lf[n] for n in nodes])
    K_LF = len(set(lf.values()))

    def km_nmi(feats, k):
        return normalized_mutual_info_score(
            y_ref, KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(feats))

    # Estimated K (k_modpeak)
    nmi_svd_est   = km_nmi(svd_feat, K_MODPEAK)
    nmi_gnn_est   = km_nmi(emb,      K_MODPEAK)
    # Oracle K (Newman's K)
    nmi_svd_oracle = km_nmi(svd_feat, K_NEWMAN)
    nmi_gnn_oracle = km_nmi(emb,      K_NEWMAN)
    # Louvain fixed seed
    nmi_lf = normalized_mutual_info_score(y_ref, lf_labels)

    print(f'  K: Newman={K_NEWMAN}  k_modpeak={K_MODPEAK}  Louvain(fixed)={K_LF}')
    print(f'  Louvain(seed=42)          NMI={nmi_lf:.4f}  K={K_LF}')
    print(f'  SVD+KMeans  (K=modpeak)   NMI={nmi_svd_est:.4f}  K={K_MODPEAK}')
    print(f'  SVD+KMeans  (K=oracle)    NMI={nmi_svd_oracle:.4f}  K={K_NEWMAN}  <- oracle')
    print(f'  SVD+GNN     (K=modpeak)   NMI={nmi_gnn_est:.4f}  K={K_MODPEAK}')
    print(f'  SVD+GNN     (K=oracle)    NMI={nmi_gnn_oracle:.4f}  K={K_NEWMAN}  <- oracle')

    return {
        'graph': name, 'nodes': G.number_of_nodes(),
        'K_newman': K_NEWMAN, 'K_modpeak': K_MODPEAK, 'K_lf': K_LF,
        'nmi_lf':         nmi_lf,
        'nmi_svd_est':    nmi_svd_est,
        'nmi_svd_oracle': nmi_svd_oracle,
        'nmi_gnn_est':    nmi_gnn_est,
        'nmi_gnn_oracle': nmi_gnn_oracle,
    }

datasets = [
    ('ego-Facebook', 'data/ego_facebook/facebook_combined.txt'),
    ('ca-GrQc',      'data/snap/ca-GrQc.txt'),
    ('ca-HepTh',     'data/snap/ca-HepTh.txt'),
]

results = []
for name, path in datasets:
    G = load_graph(path)
    results.append(evaluate(name, G))

df = pd.DataFrame(results)

print('\n\n' + '='*70)
print('ORACLE-K SUMMARY')
print('='*70)
print(df[['graph','nodes','K_newman','K_modpeak',
          'nmi_lf','nmi_svd_est','nmi_svd_oracle',
          'nmi_gnn_est','nmi_gnn_oracle']].to_string(index=False, float_format='{:.4f}'.format))

# K estimation gap
df['k_gap'] = df['K_newman'] - df['K_modpeak']
df['svd_gain'] = df['nmi_svd_oracle'] - df['nmi_svd_est']
df['gnn_gain'] = df['nmi_gnn_oracle'] - df['nmi_gnn_est']
print('\nK gap (Newman - modpeak) and NMI gain from oracle K:')
print(df[['graph','k_gap','svd_gain','gnn_gain']].to_string(index=False, float_format='{:.4f}'.format))

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
graphs = df['graph'].tolist()
x = np.arange(len(graphs))
w = 0.18

cols = {
    'Louvain\n(seed=42)':      ('nmi_lf',         '#888888'),
    'SVD+KM\n(modpeak K)':     ('nmi_svd_est',     '#E07B30'),
    'SVD+KM\n(oracle K)':      ('nmi_svd_oracle',  '#E07B30'),
    'SVD+GNN\n(modpeak K)':    ('nmi_gnn_est',     '#534AB7'),
    'SVD+GNN\n(oracle K)':     ('nmi_gnn_oracle',  '#534AB7'),
}
offsets = [-2, -1, 0, 1, 2]
hatches = ['', '', '///', '', '///']

for ax_idx, (title, keys) in enumerate([
    ('All methods — estimated K vs oracle K',
     ['nmi_lf','nmi_svd_est','nmi_svd_oracle','nmi_gnn_est','nmi_gnn_oracle'])
]):
    ax = axes[0]
    labels = ['Louvain\n(seed=42)', 'SVD+KM\n(modpeak)', 'SVD+KM\n(oracle)',
              'SVD+GNN\n(modpeak)', 'SVD+GNN\n(oracle)']
    clrs   = ['#888888', '#E07B30', '#E07B30', '#534AB7', '#534AB7']
    htch   = ['', '', '///', '', '///']
    for i, (key, lbl, clr, h) in enumerate(zip(keys, labels, clrs, htch)):
        axes[0].bar(x + offsets[i]*w, df[key], w,
                    label=lbl, color=clr, hatch=h, edgecolor='white', alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(graphs)
    axes[0].set_ylim(0, 1.1); axes[0].set_ylabel('NMI vs Newman')
    axes[0].set_title('NMI: estimated K vs oracle K', fontweight='bold')
    axes[0].legend(fontsize=8, ncol=2)

# NMI gain from oracle K
gain_svd = df['svd_gain'].tolist()
gain_gnn = df['gnn_gain'].tolist()
axes[1].bar(x - w/2, gain_svd, w, label='SVD+KMeans gain', color='#E07B30', edgecolor='white', alpha=0.85)
axes[1].bar(x + w/2, gain_gnn, w, label='SVD+GNN gain',   color='#534AB7', edgecolor='white', alpha=0.85)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels(graphs)
axes[1].set_ylabel('NMI gain (oracle K − modpeak K)')
axes[1].set_title('NMI gain from knowing the correct K', fontweight='bold')
axes[1].legend(fontsize=9)
for j in range(len(graphs)):
    axes[1].text(j - w/2, gain_svd[j] + 0.005, f'+{gain_svd[j]:.3f}', ha='center', fontsize=8)
    axes[1].text(j + w/2, gain_gnn[j] + 0.005, f'+{gain_gnn[j]:.3f}', ha='center', fontsize=8)

plt.suptitle('Oracle-K experiment: is K estimation the bottleneck?',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('fig_oracle_k.png', dpi=150, bbox_inches='tight')
print('\nSaved: fig_oracle_k.png')
