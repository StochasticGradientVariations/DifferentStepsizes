import os
import numpy as np
import matplotlib.pyplot as plt

# -- CONFIG --
DATA_ROOT  = 'results/logistic_l2/data'
PLOTS_ROOT = 'results/logistic_l2/plots'
os.makedirs(PLOTS_ROOT, exist_ok=True)

# Reference f* για κάθε dataset
fstar_dict = {
    'mushrooms': 0.02621597,
    'covtype': 0.65367992,
    'w8a': -47619.20934062,
}

methods = ['Adgd', 'AdgdAccel', 'AdgdNesCons', 'AdgdNes']
colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green']
markers = ['s', 'D', 'o', '^']

clip_floor = 1e-12

for ds_label in fstar_dict.keys():
    data_dir = os.path.join(DATA_ROOT, ds_label)
    plot_dir = os.path.join(PLOTS_ROOT, ds_label)
    os.makedirs(plot_dir, exist_ok=True)

    f_star = fstar_dict[ds_label]
    eps = 1e-16

    plt.figure(figsize=(7, 5))
    for i, method in enumerate(methods):
        loss_path = os.path.join(data_dir, f'{ds_label}_{method}_loss.npy')
        if not os.path.exists(loss_path):
            print(f"Λείπει το αρχείο: {loss_path}")
            continue
        losses = np.load(loss_path)
        residual = losses - f_star
        residual = np.maximum(residual, eps)

        residual_clipped = np.copy(residual)
        if ds_label in ['covtype', 'w8a'] and method in ['AdgdNes', 'AdgdNesCons']:
            mask = residual < clip_floor
            if np.any(mask):
                first_clip_idx = np.where(mask)[0][0]
                residual_clipped[first_clip_idx:] = clip_floor

        plt.semilogy(
            residual_clipped,
            label=method,
            color=colors[i],
            marker=markers[i],
            markevery=25,
            linewidth=1.3,
            markersize=7,
            alpha=0.85
        )

    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel(r'$f(w) - f^*$', fontsize=13)
    plt.title(f'Logistic L2 ({ds_label}): Residual', fontsize=15)
    plt.legend(fontsize=10, loc='best', frameon=True, edgecolor='black')
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.4)
    plt.savefig(os.path.join(plot_dir, f'{ds_label}_residual.png'), dpi=300)
    plt.show()
