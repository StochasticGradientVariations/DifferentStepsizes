#!/usr/bin/env python
# coding: utf-8

"""
Matrix factorization experiment με optimizers.py
Δοκιμάζει όλες τις custom μεθόδους μας πλάι στα κλασικά GD/Nesterov/AdGD,
συμπεριλαμβάνοντας τις ADPG_Momentum, ADPG_Momentum2 και ADPG_Momentum3.
"""

import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_svmlight_file

from optimizers import (
    Gd,
    Nesterov,
    Adgd,
    AdgdAccel,
    AdgdK1OverK,
    AdgdKOverKplus3,
    AdaptiveGDK1onKNesterov,
    AdgdHybrid,
    AdgdHybrid2,
    ADPG_Momentum,
    ADPG_Momentum2,
    ADPG_Momentum3
)
from loss_functions import logistic_loss, logistic_gradient

def main():
    # ─── Plot style ───────────────────────────────────────────────────
    sns.set(style="whitegrid", font_scale=1.2, context="talk")
    plt.rcParams['mathtext.fontset'] = 'cm'

    # # ─── Φόρτωση δεδομένων ──────────────────────────────────────────────
    # repo_root = os.path.dirname(__file__)
    # X_sp, y   = load_svmlight_file(os.path.join(repo_root, 'datasets', 'mushrooms'))
    # X, y      = X_sp.toarray(), y
    # if set(np.unique(y)) == {1, 2}:
    #     y = (y == 2).astype(float)

    # Φόρτωση δεδομένων w8a
    repo_root = os.path.dirname(__file__)
    X_sp, y = load_svmlight_file(os.path.join(repo_root, 'datasets', 'w8a'))
    X, y = X_sp.toarray(), y
    # Στο w8a οι ετικέτες είναι {-1,+1} → μετατρέπουμε σε {0,1}
    if set(np.unique(y)) == {-1, 1}:
        y = ((y + 1) // 2).astype(float)


    n, d   = X.shape
    # global Lipschitz estimate
    L      = 0.25 * np.max(la.eigvalsh((X.T @ X) / n))
    l2     = L / n
    w0     = np.zeros(d)
    it_max = 2000

    # ─── Βασικές NumPy‐based μέθοδοι ─────────────────────────────────
    gd     = Gd(
        lr=1.0/L,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    nest   = Nesterov(
        lr=1.0/L,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    adgd   = Adgd(
        eps=0.0, lr0=1.0/L,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    ad_acc = AdgdAccel(
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )

    # ─── Οι δικές μας παραλλαγές ───────────────────────────────────────
    opt_k1    = AdgdK1OverK(
        lr0=1.0/L,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    opt_k3    = AdgdKOverKplus3(
        lr0=1.0/L,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    opt_nes   = AdaptiveGDK1onKNesterov(
        lr0=1.0/L,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    hybrid1   = AdgdHybrid(
        lr0=1.0/L, b_lr=0.5, b_mu=0.5,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    hybrid2   = AdgdHybrid2(
        lr0=1.0/L, b_lr=0.5, b_mu=0.5,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    adpg_mom1 = ADPG_Momentum(
        lr0=1.0/L,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    adpg_mom2 = ADPG_Momentum2(
        lr0=1.0/L,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    adpg_mom3 = ADPG_Momentum3(
        lr0=1.0/L,
        L_global=L,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )

    # ─── Τρέξιμο όλων ─────────────────────────────────────────────────
    numpy_opts = [
        gd, nest, adgd, ad_acc,
        opt_k1, opt_k3, opt_nes,
        hybrid1, hybrid2,
        adpg_mom1, adpg_mom2, adpg_mom3
    ]
    labels_npy = [
        'GD',
        'Nesterov',
        'AdGD',
        'AdGD-accel',
        'AdGD (k+1)/k',
        'AdGD (k/(k+3))',
        'AdGD (k+1)/k + Nesterov',
        'AdGD Hybrid v1',
        'AdGD Hybrid v2',
        'ADPG_Momentum',
        'ADPG_Momentum2',
        'ADPG_Momentum3'
    ]
    markers_npy = [
        ',', 'o', '*', '^',
        '<', '>', 'x',
        'd', 'D',
        'h', 'p', 'H'
    ]

    for opt in numpy_opts:
        opt.run(w0.copy())
        opt.compute_loss_on_iterates()

    # ─── Plot όλων μαζί ────────────────────────────────────────────────
    f_star = min(np.min(opt.losses) for opt in numpy_opts)

    plt.figure(figsize=(8, 6))
    for opt, mk, lab in zip(numpy_opts, markers_npy, labels_npy):
        opt.plot_losses(marker=mk, f_star=f_star, label=lab)

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(r'$f(x^k)\!-\!f_*$')
    plt.legend(
        ncol=2, fontsize=9, frameon=False,
        handlelength=2.5, handletextpad=0.5,
        markerscale=1.5, markerfirst=True
    )
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
