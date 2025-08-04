import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

plot_dir = 'results/least_squares/plots/Random'
os.makedirs(plot_dir, exist_ok=True)

# ---- Ρυθμίσεις άξονα x ----
TOTAL_ITERS = 2000   # βάλε εδώ το it_max που είχες στο training
T = None             # ή βάλε έναν αριθμό, π.χ. 400, για να δείξεις 0..T

# Μέθοδοι και σταθερά σχήματα/χρώματα
methods = [
    ('AdGD',   'Adgd'),
    ('Accelerated AdGD','AdgdAccel'),
    ('AdGD+Nes⁽c⁾',    'AdgdNesCons'),
    ('AdGD+Nes',       'AdgdNes')
]
labels = [x[0] for x in methods]
tags   = [x[1] for x in methods]
colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green']
markers = ['s', 'D', 'o', '^']

# Φόρτωση histories από results/least_squares/data
hist = {}
for label, tag in methods:
    hist[label] = {
        'loss':     np.load(f'results/least_squares/data/ls_{tag}_loss.npy'),
        'gradnorm': np.array(
            np.load(f'results/least_squares/data/ls_{tag}_gradnorm.npy', allow_pickle=True),
            dtype=float
        ),
        'lr':       np.load(f'results/least_squares/data/ls_{tag}_lrs.npy')
    }

# Υπολογισμός του f* από όλα τα losses (global)
fstar_path = 'results/least_squares/data/ls_fstar.npy'
if os.path.exists(fstar_path):
    f_star = float(np.load(fstar_path))
else:
    f_star = min(np.min(hist[label]['loss']) for label in labels)
eps    = 1e-16  # μικρό offset για log-scale

def xvals(series, total_iters=TOTAL_ITERS, T=T):
    """Δώσε x-άξονα σε iterations με σωστό stride και (προαιρετικά) κόψε σε 0..T."""
    L = len(series)
    if L <= 1:
        return np.array([]), series
    stride = float(total_iters) / float(L - 1)  # π.χ. ~4 όταν L≈501 & total=2000
    x = np.arange(L, dtype=float) * stride
    if T is not None:
        m = x <= T
        return x[m], series[m]
    return x, series

def apply_major_horizontal_grid():
    """Major οριζόντιες γραμμές, χωρίς minor/vertical."""
    ax = plt.gca()
    ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.35)
    ax.xaxis.grid(False)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_locator(NullLocator())

# ---------------- Residual (f(w)-f*) σε log-scale ----------------
plt.figure(figsize=(7, 5))
for i, label in enumerate(labels):
    resid = np.maximum(hist[label]['loss'] - f_star, eps)
    x, y  = xvals(resid)
    plt.semilogy(
        x, y,
        label=label,
        color=colors[i],
        marker=markers[i],
        markevery=max(len(y)//25, 1),
        linewidth=1.3,
        markersize=8,
        alpha=0.88
    )
plt.xlabel('Iteration', fontsize=13)
plt.ylabel(r'$f(w) - f^*$', fontsize=14)
plt.title('Least Squares: Residual (log scale)', fontsize=15)
plt.legend(fontsize=10, loc='best', frameon=True, edgecolor='black')
apply_major_horizontal_grid()
plt.tight_layout()
if T is not None: plt.xlim(0, T)
plt.savefig(os.path.join(plot_dir, 'ls_plot_residual.png'), dpi=300)
plt.show()

# ---------------- Function value σε log-scale ----------------
plt.figure(figsize=(7, 5))
for i, label in enumerate(labels):
    loss = np.maximum(hist[label]['loss'], eps)
    x, y = xvals(loss)
    plt.semilogy(
        x, y,
        label=label,
        color=colors[i],
        marker=markers[i],
        markevery=max(len(y)//25, 1),
        linewidth=1.3,
        markersize=8,
        alpha=0.88
    )
plt.xlabel('Iteration', fontsize=13)
plt.ylabel(r'$f(w)$', fontsize=14)
plt.title('Least Squares: Function value (log scale)', fontsize=15)
plt.legend(fontsize=10, loc='best', frameon=True, edgecolor='black')
apply_major_horizontal_grid()
plt.tight_layout()
if T is not None: plt.xlim(0, T)
plt.savefig(os.path.join(plot_dir, 'ls_plot_loss_log.png'), dpi=300)
plt.show()

# ---------------- Gradient norm σε log-scale ----------------
plt.figure(figsize=(7, 5))
for i, label in enumerate(labels):
    grad = np.maximum(hist[label]['gradnorm'], eps)
    x, y = xvals(grad)
    plt.semilogy(
        x, y,
        label=label,
        color=colors[i],
        marker=markers[i],
        markevery=max(len(y)//25, 1),
        linewidth=1.3,
        markersize=8,
        alpha=0.88
    )
plt.xlabel('Iteration', fontsize=13)
plt.ylabel(r'$‖∇f(w)‖$', fontsize=14)
plt.title('Least Squares: Gradient norm (log scale)', fontsize=15)
plt.legend(fontsize=10, loc='best', frameon=True, edgecolor='black')
apply_major_horizontal_grid()
plt.tight_layout()
if T is not None: plt.xlim(0, T)
plt.savefig(os.path.join(plot_dir, 'ls_plot_gradnorm.png'), dpi=300)
plt.show()

# ---------------- Learning rate (γραμμική) ----------------
plt.figure(figsize=(7, 5))
for i, label in enumerate(labels):
    lr = hist[label]['lr']
    x, y = xvals(lr)
    plt.plot(
        x, y,
        label=label,
        color=colors[i],
        marker=markers[i],
        markevery=max(len(y)//25, 1),
        linewidth=1.3,
        markersize=8,
        alpha=0.75
    )
plt.xlabel('Iteration', fontsize=13)
plt.ylabel('Step size (lr)', fontsize=14)
plt.title('Least Squares: Learning rate per iteration', fontsize=15)
plt.legend(fontsize=10, loc='best', frameon=True, edgecolor='black')
apply_major_horizontal_grid()
plt.tight_layout()
if T is not None: plt.xlim(0, T)
plt.savefig(os.path.join(plot_dir, 'ls_plot_lr.png'), dpi=300)
plt.show()
