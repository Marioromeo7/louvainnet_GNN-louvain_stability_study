"""
oracle_temporal.py — Oracle diagnostic for temporal community prediction

Research question:
  Can we predict how much community structure will change next snapshot,
  using only current-snapshot features? If a linear model already captures
  most of the signal, a temporal GNN on 76 snapshots won't improve it.

Method:
  Features at time t  →  target: NMI(t, t+1)  (nmi_structural_change shifted by -1)

  Baselines:
    - Persistence: assume no change → predict NMI=1.0 always
    - Mean:        predict the mean NMI_change across all training snapshots

  Models:
    - Linear (OLS, all features)
    - Linear (significant features only: mean_pairwise_NMI, N, E)
    - Ridge (regularised)

  Evaluation: leave-one-out CV (n=74 usable pairs → LOO is exact and appropriate)

Outputs: printed table + fig_oracle_temporal.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from scipy.stats import spearmanr, pearsonr

# ── Load & merge ──────────────────────────────────────────────────────────────
tdf = pd.read_csv("temporal_results.csv")
rdf = pd.read_csv("temporal_rich_metrics.csv")
df  = tdf.merge(rdf, on="snapshot", how="inner")

# Target: NMI(t, t+1) = nmi_structural_change at snapshot t+1
# nmi_structural_change[t] = NMI(G_t, G_{t-1}), so shift by -1
df["target_nmi_next"] = df["nmi_structural_change"].shift(-1)

# Drop rows missing either features or target
feature_cols = ["mean_pairwise_nmi", "N", "E", "clustering",
                "modularity", "size_cv", "deg_cv", "density"]
sig_cols     = ["mean_pairwise_nmi", "N", "E"]   # only the significant ones

df_clean = df.dropna(subset=feature_cols + ["target_nmi_next"]).copy()
n = len(df_clean)
print(f"Usable snapshot pairs (t → t+1): {n}")
print(f"Target NMI(t,t+1): mean={df_clean['target_nmi_next'].mean():.3f}  "
      f"std={df_clean['target_nmi_next'].std():.3f}  "
      f"min={df_clean['target_nmi_next'].min():.3f}  "
      f"max={df_clean['target_nmi_next'].max():.3f}")
print()

y = df_clean["target_nmi_next"].values

# ── Baselines ─────────────────────────────────────────────────────────────────
rmse = lambda a, b: np.sqrt(mean_squared_error(a, b))

pred_persistence = np.ones(n)                    # always predict NMI=1.0
pred_mean        = np.full(n, y.mean())          # always predict the mean

rmse_persist = rmse(y, pred_persistence)
rmse_mean    = rmse(y, pred_mean)
r2_persist   = r2_score(y, pred_persistence)
r2_mean      = r2_score(y, pred_mean)            # by definition = 0

print("=" * 58)
print("BASELINES")
print("=" * 58)
print(f"  Persistence (predict NMI=1.0):  RMSE={rmse_persist:.4f}  R²={r2_persist:.4f}")
print(f"  Mean predictor:                 RMSE={rmse_mean:.4f}  R²={r2_mean:.4f}")
print()

# ── LOO cross-validated linear models ─────────────────────────────────────────
loo = LeaveOneOut()

def loo_eval(X, y, model, label):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    preds    = cross_val_predict(model, X_scaled, y, cv=loo)
    r2       = r2_score(y, preds)
    rmse_v   = rmse(y, preds)
    r_sp, p_sp = spearmanr(y, preds)
    r_pe, p_pe = pearsonr(y, preds)
    print(f"  {label:<38}  RMSE={rmse_v:.4f}  R²={r2:.4f}  "
          f"Spearman r={r_sp:+.3f} (p={p_sp:.3f})")
    return preds, r2, rmse_v

X_all = df_clean[feature_cols].values
X_sig = df_clean[sig_cols].values

print("=" * 58)
print("LOO CROSS-VALIDATED LINEAR MODELS")
print("=" * 58)
preds_ols_all, r2_ols_all, rmse_ols_all = loo_eval(X_all, y, LinearRegression(), "OLS — all features")
preds_ols_sig, r2_ols_sig, rmse_ols_sig = loo_eval(X_sig, y, LinearRegression(), "OLS — sig. features only")
preds_ridge,   r2_ridge,   rmse_ridge   = loo_eval(X_all, y, Ridge(alpha=1.0),   "Ridge (α=1) — all features")
print()

# ── Coefficient table (fit on full data for inspection) ───────────────────────
scaler  = StandardScaler()
X_s     = scaler.fit_transform(X_all)
ols_fit = LinearRegression().fit(X_s, y)

print("=" * 58)
print("OLS COEFFICIENTS (standardised, full-data fit)")
print("=" * 58)
for feat, coef in zip(feature_cols, ols_fit.coef_):
    print(f"  {feat:<22}  {coef:+.4f}")
print(f"  {'intercept':<22}  {ols_fit.intercept_:+.4f}")
print()

# ── Binary classification: high-change snapshots ──────────────────────────────
# Can we identify snapshots where NMI(t,t+1) < 0.70 (high structural change)?
threshold = 0.70
n_high    = (y < threshold).sum()
print("=" * 58)
print(f"BINARY TASK: detect high-change snapshots (NMI < {threshold})")
print(f"  {n_high}/{n} snapshots have NMI(t,t+1) < {threshold}")
print()

if n_high >= 3:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, balanced_accuracy_score

    y_bin = (y < threshold).astype(int)
    scaler2  = StandardScaler()
    X_s2     = scaler2.fit_transform(X_sig)
    preds_bin = cross_val_predict(
        LogisticRegression(class_weight="balanced", max_iter=500),
        X_s2, y_bin, cv=loo
    )
    bal_acc = balanced_accuracy_score(y_bin, preds_bin)
    print(f"  Logistic LOO balanced accuracy: {bal_acc:.3f}")
    print(classification_report(y_bin, preds_bin,
          target_names=["stable (≥0.70)", "high-change (<0.70)"], zero_division=0))
else:
    print(f"  Too few high-change snapshots ({n_high}) for binary classification.")
print()

# ── Summary verdict ───────────────────────────────────────────────────────────
print("=" * 58)
print("VERDICT")
print("=" * 58)
best_r2   = max(r2_ols_all, r2_ols_sig, r2_ridge)
best_rmse = min(rmse_ols_all, rmse_ols_sig, rmse_ridge)
improve   = (rmse_persist - best_rmse) / rmse_persist * 100

print(f"  Best LOO R²:              {best_r2:.4f}")
print(f"  Best LOO RMSE:            {best_rmse:.4f}")
print(f"  Persistence RMSE:         {rmse_persist:.4f}")
print(f"  Improvement over persist: {improve:.1f}%")
print()
if best_r2 > 0.25:
    print("  → Signal is substantial. A temporal model may add value,")
    print("    but verify on held-out snapshots before building a GNN.")
elif best_r2 > 0.10:
    print("  → Weak but non-zero signal. Linear model explains some variance.")
    print("    A GNN on 76 snapshots is unlikely to improve meaningfully.")
else:
    print("  → Signal is near zero. No model will reliably predict next-step change.")
    print("    Do not build a temporal GNN — the predictive signal isn't there.")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

x_idx = df_clean["snapshot"].values

# Panel 1: actual vs predicted (best model)
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(x_idx, y,              color="#333",    label="Actual NMI(t,t+1)",   lw=1.5, marker="o", ms=3)
ax1.plot(x_idx, preds_ols_sig,  color="#993C1D", label="OLS (sig. features)", lw=1.5, linestyle="--", marker="s", ms=3)
ax1.plot(x_idx, pred_persistence, color="#aaa",  label="Persistence baseline", lw=1, linestyle=":")
ax1.set_ylabel("NMI(t, t+1)", fontsize=10)
ax1.set_xlabel("Snapshot index", fontsize=10)
ax1.set_title("Actual vs predicted structural change (LOO CV)", fontsize=10)
ax1.legend(fontsize=8)
ax1.set_ylim(0.4, 1.1)
ax1.grid(alpha=0.15)

# Panel 2: residuals
ax2 = fig.add_subplot(gs[0, 2])
resid = y - preds_ols_sig
ax2.scatter(preds_ols_sig, resid, c="#534AB7", s=25, alpha=0.7, edgecolors="white", lw=0.3)
ax2.axhline(0, color="#aaa", lw=1, linestyle="--")
ax2.set_xlabel("Predicted NMI", fontsize=9)
ax2.set_ylabel("Residual", fontsize=9)
ax2.set_title(f"Residuals — OLS sig. features\nLOO R²={r2_ols_sig:.3f}", fontsize=9)
ax2.grid(alpha=0.15)

# Panel 3: scatter actual vs predicted
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(y, preds_ols_sig, c="#993C1D", s=30, alpha=0.75, edgecolors="white", lw=0.3)
ax3.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=1, alpha=0.5)
ax3.set_xlabel("Actual NMI(t,t+1)", fontsize=9)
ax3.set_ylabel("Predicted", fontsize=9)
ax3.set_title("Actual vs predicted scatter", fontsize=9)
ax3.grid(alpha=0.15)

# Panel 4: coefficient bar
ax4 = fig.add_subplot(gs[1, 1])
coefs = dict(zip(feature_cols, ols_fit.coef_))
sorted_feats = sorted(coefs, key=lambda k: abs(coefs[k]), reverse=True)
colors = ["#993C1D" if coefs[f] > 0 else "#534AB7" for f in sorted_feats]
ax4.barh(sorted_feats, [coefs[f] for f in sorted_feats], color=colors, edgecolor="white")
ax4.axvline(0, color="#aaa", lw=1)
ax4.set_xlabel("Standardised coefficient", fontsize=9)
ax4.set_title("OLS coefficients (standardised)", fontsize=9)
ax4.grid(alpha=0.15, axis="x")

# Panel 5: summary table
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
summary_data = [
    ["Model",                    "RMSE",           "LOO R²"],
    ["Persistence (NMI=1.0)",    f"{rmse_persist:.4f}", f"{r2_persist:.4f}"],
    ["Mean predictor",           f"{rmse_mean:.4f}",    "0.0000"],
    ["OLS — all features",       f"{rmse_ols_all:.4f}", f"{r2_ols_all:.4f}"],
    ["OLS — sig. only",          f"{rmse_ols_sig:.4f}", f"{r2_ols_sig:.4f}"],
    ["Ridge — all features",     f"{rmse_ridge:.4f}",   f"{r2_ridge:.4f}"],
]
tbl = ax5.table(cellText=summary_data[1:], colLabels=summary_data[0],
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.5)
ax5.set_title("Model comparison", fontsize=9)

plt.suptitle("Oracle diagnostic — can we predict next-snapshot structural change?",
             fontweight="bold", fontsize=11, y=1.01)
plt.savefig("fig_oracle_temporal.png", dpi=150, bbox_inches="tight")
print("\nSaved fig_oracle_temporal.png")
plt.show()
