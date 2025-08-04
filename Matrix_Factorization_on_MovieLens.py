#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import coo_matrix

from optimizers import Adgd, AdgdAccel, AdaPGNesterov

def load_movielens_u_data(path):
    df = pd.read_csv(path, sep='\t', names=['user','item','rating','ts'])
    n_users = df.user.max() + 1
    n_items = df.item.max() + 1
    rows = np.arange(len(df))
    cols_user = df.user.values
    cols_item = df.item.values + n_users
    data = np.ones(len(df))
    A = coo_matrix(
        (np.concatenate([data, data]),
         (np.concatenate([rows, rows]),
          np.concatenate([cols_user, cols_item]))),
        shape=(len(df), n_users + n_items)
    ).tocsr()
    b = df.rating.values.astype(float)
    return A, b

λ = 1e-6  # Regularization hyper-parameter

def main():
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams['mathtext.fontset'] = 'cm'

    # Save dir for MovieLens plots
    save_dir = 'results/matrix_fac/plots/MovieLens'
    os.makedirs(save_dir, exist_ok=True)

    repo_root = os.path.dirname(__file__)
    A, b = load_movielens_u_data(os.path.join(repo_root, 'datasets', 'u.data'))
    n, d = A.shape

    G = (A.T @ A).toarray()
    G /= n
    L = np.max(np.linalg.eigvalsh(G)) + λ

    def loss_fn(w):
        r = A.dot(w) - b
        return 0.5 * np.mean(r**2) + 0.5 * λ * np.dot(w, w)

    def grad_fn(w):
        return (A.T.dot(A.dot(w) - b)) / n + λ * w

    w0 = np.zeros(d)
    it_max = 1000

    OPTS = [
        ('AdGD',       Adgd,          {'lr0': 1.0 / L, 'eps': 0.0}),
        ('AdgdAccel',  AdgdAccel,     {}),
        ('AdGDNesCon', AdaPGNesterov, {'lr0': 1.0 / L, 'isConservative': True}),
        ('AdGDNes',    AdaPGNesterov, {'lr0': 1.0 / L, 'isConservative': False}),
    ]
    MARKERS = ['s', 'D', 'o', '^']
    COLORS  = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green']

    optimizers = []
    labels = []
    for lbl, Opt, extra in OPTS:
        kwargs = dict(loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
        kwargs.update(extra)
        optimizers.append(Opt(**kwargs))
        labels.append(lbl)

    # Run and collect histories
    for opt in optimizers:
        opt.run(w0.copy())
        if hasattr(opt, 'compute_loss_on_iterates'):
            opt.compute_loss_on_iterates()
        if not hasattr(opt, 'losses') and hasattr(opt, 'loss_hist'):
            opt.losses = np.array(opt.loss_hist)

    # Compute best loss for residuals
    f_star = min(np.min(opt.losses) for opt in optimizers)

    # --------- Plot residual ---------
    plt.figure(figsize=(8, 6))
    for opt, mk, lab, col in zip(optimizers, MARKERS, labels, COLORS):
        resid = np.maximum(opt.losses - f_star, 1e-16)
        x = np.arange(len(resid))
        plt.semilogy(
            x, resid,
            label=lab, color=col,
            marker=mk, markevery=max(len(resid) // 20, 1),
            linewidth=1.7, alpha=0.9
        )
    plt.xlabel('Iteration')
    plt.ylabel(r'$f(w) - f^*$')
    plt.title('MovieLens: Residual (log scale)')
    plt.legend(fontsize=11, loc='best', frameon=True, edgecolor='black')
    plt.grid(True, which='major', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'movielens_residual.png'), dpi=300)
    plt.show()

    # --------- Plot gradient norm ---------
    plt.figure(figsize=(8, 6))
    for opt, mk, lab, col in zip(optimizers, MARKERS, labels, COLORS):
        if hasattr(opt, 'grad_norm_hist') and opt.grad_norm_hist is not None:
            g = np.array(opt.grad_norm_hist, dtype=float)
            g = np.maximum(g, 1e-16)
            x = np.arange(len(g))
            plt.semilogy(
                x, g,
                label=lab, color=col,
                marker=mk, markevery=max(len(g) // 20, 1),
                linewidth=1.7, alpha=0.9
            )
    plt.xlabel('Iteration')
    plt.ylabel(r'$‖∇f(w)‖$')
    plt.title('MovieLens: Gradient norm (log scale)')
    plt.legend(fontsize=11, loc='best', frameon=True, edgecolor='black')
    plt.grid(True, which='major', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'movielens_gradnorm.png'), dpi=300)
    plt.show()

    # --------- Plot learning rate ---------
    plt.figure(figsize=(8, 6))
    for opt, mk, lab, col in zip(optimizers, MARKERS, labels, COLORS):
        lr = getattr(opt, 'lr_hist', getattr(opt, 'lrs', None))
        if lr is not None:
            lr = np.array(lr, dtype=float)
            x = np.arange(len(lr))
            plt.plot(
                x, lr,
                label=lab, color=col,
                marker=mk, markevery=max(len(lr) // 20, 1),
                linewidth=1.7, alpha=0.9
            )
    plt.xlabel('Iteration')
    plt.ylabel('Step size (lr)')
    plt.title('MovieLens: Learning rate per iteration')
    plt.legend(fontsize=11, loc='best', frameon=True, edgecolor='black')
    plt.grid(True, which='major', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'movielens_lr.png'), dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
