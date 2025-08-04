import os
import numpy as np
import matplotlib.pyplot as plt

# Δημιουργία φακέλου για τα plots
os.makedirs('results/cubic/plots', exist_ok=True)

# Ορισμός μεθόδων (χωρίς την απλή Nesterov)
methods = {
    'AdGD':        'Adgd',
    'AdGD+Nes⁽c⁾': 'AdgdNesCons',
    'AdGD+Nes':    'AdgdNes',
    'AdGD-Accel':  'AdgdAccel'
}

# Marker styles
marker_dict = {
    'AdGD': 's',
    'AdGD+Nes⁽c⁾': 'o',
    'AdGD+Nes': '^',
    'AdGD-Accel': 'v'
}

# Styling function
def beautify_plot(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.minorticks_on()
    ax.grid(True, which='both', linestyle='--', linewidth=0.8, color='lightgray')
    ax.legend(fontsize=10, loc='best', frameon=True, edgecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Φόρτωση δεδομένων
hist = {}
for label, tag in methods.items():
    hist[label] = {
        'loss':     np.load(f'results/cubic/data/cubic_{tag}_loss.npy')[:400],
        'gradnorm': np.array(np.load(f'results/cubic/data/cubic_{tag}_gradnorm.npy', allow_pickle=True), dtype=float)[:400],
        'lr':       np.load(f'results/cubic/data/cubic_{tag}_lrs.npy')[:400]
    }

# Υπολογισμός f*
f_star = min(hist[label]['loss'][-1] for label in methods)
eps = 1e-16

# 1) Residual plot
fig, ax = plt.subplots(figsize=(7, 5))
for label in methods:
    resid = np.maximum(hist[label]['loss'] - f_star, eps)
    marker = marker_dict.get(label, 'o')
    markevery = max(1, len(resid)//30)
    if label == 'AdGD+Nes':
        markevery = max(1, len(resid)//40)
    ax.semilogy(
        resid,
        label=label,
        alpha=0.9,
        marker=marker,
        markevery=markevery,
        markersize=8,
        markeredgecolor='black',
        markeredgewidth=1.2,
        linewidth=2
    )
beautify_plot(ax, 'Cubic: Residual (log scale)', 'Iteration', 'f(w) − f*')
fig.tight_layout()
fig.savefig('results/cubic/plots/cubic_plot_residual.png', dpi=300)
plt.close(fig)

# 2) Function value plot
fig, ax = plt.subplots(figsize=(7, 5))
for label in methods:
    vals = np.maximum(hist[label]['loss'], eps)
    marker = marker_dict.get(label, 'o')
    markevery = max(1, len(vals)//30)
    if label == 'AdGD+Nes':
        markevery = max(1, len(vals)//10)
    ax.semilogy(
        vals,
        label=label,
        alpha=0.9,
        marker=marker,
        markevery=markevery,
        markersize=8,
        markeredgecolor='black',
        markeredgewidth=1.2,
        linewidth=2
    )
beautify_plot(ax, 'Cubic: Function Value (log scale)', 'Iteration', 'f(w)')
fig.tight_layout()
fig.savefig('results/cubic/plots/cubic_plot_loss_log.png', dpi=300)
plt.close(fig)

# 3) Gradient norm plot
fig, ax = plt.subplots(figsize=(7, 5))
for label in methods:
    grad = np.maximum(hist[label]['gradnorm'], eps)
    marker = marker_dict.get(label, 'o')
    markevery = max(1, len(grad)//30)
    if label == 'AdGD+Nes':
        markevery = max(1, len(grad)//10)
    ax.semilogy(
        grad,
        label=label,
        alpha=0.9,
        marker=marker,
        markevery=markevery,
        markersize=8,
        markeredgecolor='black',
        markeredgewidth=1.2,
        linewidth=2
    )
beautify_plot(ax, 'Cubic: Gradient Norm (log scale)', 'Iteration', '‖∇f(w)‖ or PG–norm')
fig.tight_layout()
fig.savefig('results/cubic/plots/cubic_plot_gradnorm.png', dpi=300)
plt.close(fig)

# 4) Learning rate schedule
fig, ax = plt.subplots(figsize=(7, 5))
for label in methods:
    lrs = hist[label]['lr']
    marker = marker_dict.get(label, '.')
    markevery = max(1, len(lrs)//30)
    if label == 'AdGD+Nes':
        markevery = max(1, len(lrs)//10)
    ax.plot(
        lrs,
        label=label,
        marker=marker,
        markevery=markevery,
        markersize=8,
        markeredgecolor='black',
        markeredgewidth=1.1,
        linewidth=2,
        alpha=0.9
    )
beautify_plot(ax, 'Cubic: Learning Rate Schedule', 'Iteration', 'Step size (lr)')
fig.tight_layout()
fig.savefig('results/cubic/plots/cubic_plot_lr.png', dpi=300)
plt.close(fig)
