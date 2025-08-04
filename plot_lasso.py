import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Root directories
DATA_ROOT = 'results/lasso'
PLOTS_ROOT = 'results/lasso/plots'
os.makedirs(PLOTS_ROOT, exist_ok=True)

# Patterns
data_dir_pattern = re.compile(r'data_(?P<m>\d+)_(?P<n>\d+)_(?P<k>\d+)')
file_pattern = re.compile(
    r'lasso_m(?P<m>\d+)_n(?P<n>\d+)_k(?P<k>\d+)_' +
    r'(?P<method>\w+)_(?P<what>loss|gradnorm|lrs)\.npy'
)

# Marker styles
marker_dict = {
    'AdgdAccel': 'v',
    'AdgdNesCons': 'o',
    'AdgdNes': '^',
    'Adgd': 's',
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

# Iterate over each subfolder
for entry in os.listdir(DATA_ROOT):
    match = data_dir_pattern.match(entry)
    if not match:
        continue

    m, n, k = match.group('m'), match.group('n'), match.group('k')
    tag = f"data_{m}_{n}_{k}"
    data_dir = os.path.join(DATA_ROOT, entry)
    plot_dir = os.path.join(PLOTS_ROOT, tag)
    os.makedirs(plot_dir, exist_ok=True)

    histories = {}
    for fname in os.listdir(data_dir):
        fm = file_pattern.match(fname)
        if not fm:
            continue
        dd = fm.groupdict()
        method = dd['method']
        if method == 'Nes':
            continue  # skip 'Nes'
        what = dd['what']
        arr = np.load(os.path.join(data_dir, fname), allow_pickle=True)
        if what == 'gradnorm':
            arr = arr.astype(float)
        histories.setdefault(method, {})[what] = arr

    f_star = min(h['loss'][-1] for h in histories.values())
    eps = 1e-16

    # 1) Residual plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, h in histories.items():
        resid = np.maximum(h['loss'] - f_star, eps)[:400]
        marker = marker_dict.get(method, 'o')
        markevery = max(1, len(resid)//30)
        if method == 'AdgdNes':
            markevery = max(1, len(resid)//40)
        ax.semilogy(
            resid,
            label=method,
            alpha=0.9,
            marker=marker,
            markevery=markevery,
            markersize=8,
            markeredgecolor='black',
            markeredgewidth=1.2,
            linewidth=2
        )
    beautify_plot(ax, f'Lasso ({tag}): Residual (log scale)', 'Iteration', 'f(w) − f*')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'{tag}_residual.png'), dpi=300)
    plt.close(fig)

    # 2) Function value plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, h in histories.items():
        vals = np.maximum(h['loss'], eps)[:400]
        marker = marker_dict.get(method, 'o')
        markevery = max(1, len(vals)//30)
        if method == 'AdgdNes':
            markevery = max(1, len(vals)//10)
        ax.semilogy(
            vals,
            label=method,
            alpha=0.9,
            marker=marker,
            markevery=markevery,
            markersize=8,
            markeredgecolor='black',
            markeredgewidth=1.2,
            linewidth=2
        )
    beautify_plot(ax, f'Lasso ({tag}): Function Value (log scale)', 'Iteration', 'f(w)')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'{tag}_loss_log.png'), dpi=300)
    plt.close(fig)

    # 3) Gradient norm plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, h in histories.items():
        grad = np.maximum(h['gradnorm'], eps)[:400]
        marker = marker_dict.get(method, 'o')
        markevery = max(1, len(grad)//30)
        if method == 'AdgdNes':
            markevery = max(1, len(grad)//10)
        ax.semilogy(
            grad,
            label=method,
            alpha=0.9,
            marker=marker,
            markevery=markevery,
            markersize=8,
            markeredgecolor='black',
            markeredgewidth=1.2,
            linewidth=2
        )
    beautify_plot(ax, f'Lasso ({tag}): Gradient Norm (log scale)', 'Iteration', '‖∇f(w)‖')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'{tag}_gradnorm.png'), dpi=300)
    plt.close(fig)

    # 4) Learning rate schedule
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, h in histories.items():
        h['lrs'] = h['lrs'][:400]
        marker = marker_dict.get(method, '.')
        markevery = max(1, len(h['lrs'])//30)
        if method == 'AdgdNes':
            markevery = max(1, len(h['lrs'])//10)
        ax.plot(
            h['lrs'],
            label=method,
            marker=marker,
            markevery=markevery,
            markersize=8,
            markeredgecolor='black',
            markeredgewidth=1.1,
            linewidth=2,
            alpha=0.9
        )
    beautify_plot(ax, f'Lasso ({tag}): Learning Rate Schedule', 'Iteration', 'Step size (lr)')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'{tag}_lrs.png'), dpi=300)
    plt.close(fig)
