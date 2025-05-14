#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import coo_matrix

# Οι optimizers από το optimizers.py
from optimizers import (
    Gd, Nesterov, Adgd, AdgdAccel,
    AdgdK1OverK, AdgdKOverKplus3, AdaptiveGDK1onKNesterov,
    AdgdHybrid, AdgdHybrid2,
    ADPG_Momentum, ADPG_Momentum2, ADPG_Momentum3
)

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


λ =  1e-6  # αν θελουμε το αλλαζουμε το το regularization hyper-parameter ή πιο απλα το strong convexity term [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
def main():
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams['mathtext.fontset'] = 'cm'

    # 1) Φόρτωση A,b
    repo_root = os.path.dirname(__file__)
    A, b = load_movielens_u_data(os.path.join(repo_root, 'datasets', 'u.data'))
    n, d = A.shape

    # 2) Lipschitz constant για least‐squares: L = max eigenvalue((A^TA)/n)
    G = (A.T @ A).toarray()
    G /= n
    L = np.max(np.linalg.eigvalsh(G))
    L += λ

    # 3) Ορισμός loss & gradient
    def loss_fn(w):
        r = A.dot(w) - b
        return 0.5 * np.mean(r**2) + 0.5 * λ * np.dot(w, w)
    def grad_fn(w):
        # gradient of ½||Aw-b||² + (λ/2)||w||²
        return (A.T.dot(A.dot(w) - b)) / n + λ * w

    # 4) Σταθερές πειράματος
    w0 = np.zeros(d)
    it_max = 1000

    # 5) Δημιουργία optimizer instances
    OPTS = [
        Gd, Nesterov, Adgd,
        AdgdK1OverK, AdaptiveGDK1onKNesterov,
        AdgdHybrid, AdgdHybrid2,
        ADPG_Momentum, ADPG_Momentum2, ADPG_Momentum3
    ]
    LABELS = [
        'GD','Nesterov','AdGD',
        'AdGD (k+1)/k','AdGD+Nesterov',
        'AdGD Hybrid v1','AdGD Hybrid v2',
        'ADPG_M1','ADPG_M2','ADPG_M3'
    ]
    MARKERS = [',','o','*','<','x','d','D','h','p','H']

    optimizers = []
    for Opt in OPTS:
        kwargs = dict(loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
        sig = Opt.__init__.__code__.co_varnames
        if 'lr' in sig:
            kwargs['lr'] = 1.0 / L
        elif 'lr0' in sig:
            kwargs['lr0'] = 1.0 / L
        if Opt.__name__ == 'ADPG_Momentum3':
            kwargs['L_global'] = L
        optimizers.append(Opt(**kwargs))

    # 6) Τρέξιμο & συλλογή ιστορικού
    for opt in optimizers:
        opt.run(w0.copy())
        opt.compute_loss_on_iterates()

    # 7) Plot σύγκλισης
    f_star = min(np.min(opt.losses) for opt in optimizers)
    plt.figure(figsize=(8, 6))
    for opt, mk, lab in zip(optimizers, MARKERS, LABELS):
        opt.plot_losses(marker=mk, f_star=f_star, label=lab)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(r'$f(w^k)-f_*$')
    plt.title('Linear Least Squares on MovieLens u.data')
    plt.legend(ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
