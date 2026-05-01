"""
Modularity matrix spectrum analysis.
Q: does Newman's K correspond to a visible eigenvalue gap in B = A - kk^T/2m?
"""
import os, warnings
os.chdir('c:/Users/mario/Desktop/lovainnet')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

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

def modularity_matrix(G):
    """Build sparse modularity matrix B = A - kk^T / 2m"""
    N  = G.number_of_nodes()
    m2 = 2 * G.number_of_edges()
    nodes = sorted(G.nodes())
    n2i   = {n: i for i, n in enumerate(nodes)}
    rows, cols = [], []
    for u, v in G.edges():
        i, j = n2i[u], n2i[v]
        rows += [i, j]; cols += [j, i]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    deg = np.array(A.sum(axis=1)).flatten()
    # B = A - kk^T/2m  — keep sparse for eigsh, subtract null model via shift
    # eigsh can't handle dense outer product, so we use LinearOperator
    from scipy.sparse.linalg import LinearOperator
    def matvec(x):
        return A @ x - deg * (deg @ x) / m2
    B = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    return B, deg, m2, N

def top_eigenvalues(G, n_eigs=140):
    """Compute top n_eigs eigenvalues of B."""
    B, deg, m2, N = modularity_matrix(G)
    k = min(n_eigs, N - 2)
    vals, vecs = eigsh(B, k=k, which='LM')
    idx  = np.argsort(-vals)
    return vals[idx], vecs[:, idx]

datasets = [
    ('ego-Facebook', 'data/ego_facebook/facebook_combined.txt', 13),
    ('ca-GrQc',      'data/snap/ca-GrQc.txt',                   65),
    ('ca-HepTh',     'data/snap/ca-HepTh.txt',                 115),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for col, (name, path, K_NEWMAN) in enumerate(datasets):
    print(f'\n{name}  (Newman K={K_NEWMAN})')
    G    = load_graph(path)
    N, E = G.number_of_nodes(), G.number_of_edges()
    print(f'  {N:,} nodes, {E:,} edges')

    vals, vecs = top_eigenvalues(G, n_eigs=min(140, N-2))
    pos_vals   = vals[vals > 0]
    n_positive = len(pos_vals)
    print(f'  Positive eigenvalues: {n_positive}')
    print(f'  Top-20 eigenvalues: {np.round(vals[:20], 3).tolist()}')

    # ── Plot 1: eigenvalue spectrum with Newman K marked ──
    ax = axes[0, col]
    n_show = min(80, len(vals))
    x = np.arange(1, n_show + 1)
    colors = ['#534AB7' if v > 0 else '#cccccc' for v in vals[:n_show]]
    ax.bar(x, vals[:n_show], color=colors, width=0.8, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    if K_NEWMAN <= n_show:
        ax.axvline(K_NEWMAN + 0.5, color='#993C1D', linewidth=2,
                   linestyle='--', label=f'Newman K={K_NEWMAN}')
        ax.axvline(K_NEWMAN - 0.5, color='#993C1D', linewidth=2, linestyle='--')
    ax.set_xlabel('Eigenvalue rank')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'{name}\n({N:,} nodes, {E:,} edges)', fontweight='bold')
    ax.legend(fontsize=9)

    # ── Plot 2: eigenvalue gaps (consecutive differences) ──
    ax2 = axes[1, col]
    gaps = np.abs(np.diff(vals[:min(80, len(vals))]))
    gap_x = np.arange(1, len(gaps) + 1)
    ax2.bar(gap_x, gaps, color='#0F6E56', width=0.8, alpha=0.85)
    if K_NEWMAN <= len(gaps):
        ax2.axvline(K_NEWMAN, color='#993C1D', linewidth=2,
                    linestyle='--', label=f'Newman K={K_NEWMAN}')
    # Mark top-5 gaps
    top5 = np.argsort(-gaps)[:5] + 1
    for rank in top5:
        ax2.text(rank, gaps[rank-1] + gaps.max()*0.01,
                 str(rank), ha='center', fontsize=8, color='#534AB7', fontweight='bold')
    ax2.set_xlabel('Between eigenvalue rank i and i+1')
    ax2.set_ylabel('Gap size |λ_i − λ_{i+1}|')
    ax2.set_title(f'Eigenvalue gaps — largest gap at rank {top5[0]}', fontweight='bold')
    ax2.legend(fontsize=9)

    print(f'  Largest gaps at ranks: {sorted(top5.tolist())}')
    print(f'  Newman K={K_NEWMAN} gap rank: {K_NEWMAN}  '
          f'gap size={gaps[K_NEWMAN-1]:.4f}  '
          f'vs max gap={gaps.max():.4f}  '
          f'ratio={gaps[K_NEWMAN-1]/gaps.max():.3f}')

plt.suptitle('Modularity Matrix Spectrum: does Newman\'s K align with the eigenvalue gap?',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('fig_modularity_spectrum.png', dpi=150, bbox_inches='tight')
print('\nSaved: fig_modularity_spectrum.png')
