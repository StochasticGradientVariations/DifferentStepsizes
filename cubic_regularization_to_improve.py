#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import numpy.linalg as la
import scipy
import scipy.special
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_svmlight_file
from sklearn.utils.extmath import safe_sparse_dot

from optimizers import (
    Gd, Nesterov, Adgd, AdgdAccel,
    AdgdK1OverK, AdgdKOverKplus3, AdaptiveGDK1onKNesterov,
    AdgdHybrid, AdgdHybrid2,
    ADPG_Momentum, ADPG_Momentum2, ADPG_Momentum3
)
from loss_functions import cubic_loss0, cubic_gradient0, logistic_gradient

# ─── Διάταξη γραφικών ───────────────────────────────────────────────────
sns.set(
    style="whitegrid", font_scale=1.2, context="talk",
    palette=sns.color_palette("bright"), color_codes=False
)
plt.rcParams['mathtext.fontset'] = 'cm'


def logistic_smoothness(X):
    """Υπολογίζει την global smoothness L της logistic loss."""
    return 0.25 * np.max(la.eigvalsh((X.T @ X) / X.shape[0]))


def logistic_hessian(w, X, y, l2):
    """
    Hessian της logistic loss: Xᵀ diag(p*(1-p)) X + l2·I
    Όπως στο notebook του cubic_regularization.ipynb.
    """
    activation = scipy.special.expit(
        safe_sparse_dot(X, w, dense_output=True).ravel()
    )
    weights = activation * (1 - activation)
    # X_weights: κάθε στήλη j του X πολλαπλασιασμένη με weights
    X_weights = X.T * weights
    return X_weights @ X + l2 * np.eye(len(w))


def main():
    # ─── Φόρτωση δεδομένων (CovType σε svmlight / bz2 μορφή) ───────────
    repo_root = os.path.dirname(__file__)
    data_path = os.path.join(repo_root, 'datasets', 'covtype.bz2')
    X_sp, y = load_svmlight_file(data_path)
    X, y = X_sp.toarray(), y
    # ετικέτες {1,2} → {0,1}
    if (np.unique(y) == [1, 2]).all():
        y = y - 1

    n, d = X.shape

    # ─── Υπολογισμός hyper-par. για cubic regularization ────────────────
    L = logistic_smoothness(X)
    scale = L * n
    l2 = 0.0           # δεν έχουμε καμία L2 regularization εδώ
    M = 100            # όπως στο notebook
    w0 = np.zeros(d)   # αρχική τιμή

    # αρχική κλίση & Hessian στο w0
    g0 = logistic_gradient(w0, X, y, l2, normalize=False)
    H0 = logistic_hessian(w0, X, y, l2)

    # loss & grad για το cubic subproblem
    def loss_func(w):
        return cubic_loss0(w, H0, g0, M, scale)

    def grad_func(w):
        return cubic_gradient0(w, H0, g0, M, scale)

    it_max = 1000

    # ─── 1) Tuning βήματος για GD ─────────────────────────────────────
    lrs = np.logspace(-1, 1, 10)
    losses = []
    for lr in lrs:
        tmp = Gd(
            lr=lr,
            loss_func=loss_func,
            grad_func=grad_func,
            it_max=it_max//2
        )
        tmp.run(w0)
        tmp.compute_loss_on_iterates()
        losses.append(tmp.losses[-1])
    best_lr_gd = lrs[np.nanargmin(losses)]
    gd = Gd(
        lr=best_lr_gd,
        loss_func=loss_func,
        grad_func=grad_func,
        it_max=it_max
    )
    gd.run(w0)

    # ─── 2) Tuning βήματος για Nesterov ──────────────────────────────
    lrs = np.logspace(-1, 1, 10)
    losses = []
    for lr in lrs:
        tmp = Nesterov(
            lr=lr,
            loss_func=loss_func,
            grad_func=grad_func,
            it_max=it_max//2
        )
        tmp.run(w0)
        tmp.compute_loss_on_iterates()
        losses.append(tmp.losses[-1])
    best_lr_nes = lrs[np.nanargmin(losses)]
    nest = Nesterov(
        lr=best_lr_nes,
        loss_func=loss_func,
        grad_func=grad_func,
        it_max=it_max
    )
    nest.run(w0)

    # ─── 3) Adgd & AdgdAccel ────────────────────────────────────────
    adgd    = Adgd( eps=0.0, lr0=1.0/L, loss_func=loss_func, grad_func=grad_func, it_max=it_max )
    ad_acc  = AdgdAccel( loss_func=loss_func, grad_func=grad_func, it_max=it_max )
    adgd.run(w0)
    ad_acc.run(w0)

    # ─── 4) Οι δικές μας παραλλαγές ───────────────────────────────────
    opt_k1    = AdgdK1OverK( lr0=1.0/L, loss_func=loss_func, grad_func=grad_func, it_max=it_max )
    opt_k3    = AdgdKOverKplus3( lr0=1.0/L, loss_func=loss_func, grad_func=grad_func, it_max=it_max )
    opt_nes   = AdaptiveGDK1onKNesterov( lr0=1.0/L, loss_func=loss_func, grad_func=grad_func, it_max=it_max )
    hybrid1   = AdgdHybrid( lr0=1.0/L, b_lr=0.5, b_mu=0.5, loss_func=loss_func, grad_func=grad_func, it_max=it_max )
    hybrid2   = AdgdHybrid2( lr0=1.0/L, b_lr=0.5, b_mu=0.5, loss_func=loss_func, grad_func=grad_func, it_max=it_max )
    adpg_m1   = ADPG_Momentum(  lr0=1.0/L, loss_func=loss_func, grad_func=grad_func, it_max=it_max )
    adpg_m2   = ADPG_Momentum2( lr0=1.0/L, loss_func=loss_func, grad_func=grad_func, it_max=it_max )
    adpg_m3   = ADPG_Momentum3( lr0=1.0/L, L_global=L, loss_func=loss_func, grad_func=grad_func, it_max=it_max )

    for opt in (opt_k1, opt_k3, opt_nes, hybrid1, hybrid2, adpg_m1, adpg_m2, adpg_m3):
        opt.run(w0)

    # ─── Συγκεντρώνουμε & plot ────────────────────────────────────────
    all_opts = [gd, nest, adgd, ad_acc,
                opt_k1, opt_k3, opt_nes, hybrid1, hybrid2,
                adpg_m1, adpg_m2, adpg_m3]
    labels    = [
        'GD', 'Nesterov', 'AdGD', 'AdGD-accel',
        'AdGD (k+1)/k', 'AdGD (k/(k+3))', 'AdGD+Nesterov',
        'AdGD Hybrid v1', 'AdGD Hybrid v2',
        'ADPG_M1', 'ADPG_M2', 'ADPG_M3'
    ]
    markers   = [',','o','*','D','<','>','x','d','D','h','p','H']

    for opt in all_opts:
        opt.compute_loss_on_iterates()

    f_star = min(np.min(opt.losses) for opt in all_opts)
    plt.figure(figsize=(8, 6))
    for opt, mk, lab in zip(all_opts, markers, labels):
        opt.plot_losses(marker=mk, f_star=f_star, label=lab)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(r'$f(x^k) - f_*$')
    plt.title('Cubic Regularization on covtype.bz2')
    plt.legend(ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
