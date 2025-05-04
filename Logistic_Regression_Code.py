# Logistic_Regression_Code.py

import os

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_svmlight_file

# NumPy‐based methods από το optimizers.py
from optimizers import (
    Gd,
    Nesterov,
    Adgd,
    AdgdAccel,
    AdgdK1OverK,
    AdgdKOverKplus3,
    AdaptiveGDK1onKNesterov
)
from loss_functions import logistic_loss, logistic_gradient

def main():
    # ─── Plot style ───────────────────────────────────────────────────
    sns.set(style="whitegrid", font_scale=1.2, context="talk")
    plt.rcParams['mathtext.fontset'] = 'cm'

    # ─── Φόρτωση δεδομένων ──────────────────────────────────────────────
    repo_root = os.path.dirname(__file__)
    X_sp, y   = load_svmlight_file(os.path.join(repo_root, 'datasets', 'mushrooms'))
    X, y      = X_sp.toarray(), y
    if set(np.unique(y)) == {1, 2}:
        y = (y == 2).astype(float)

    n, d   = X.shape
    L      = 0.25 * np.max(la.eigvalsh((X.T @ X) / n))
    l2     = L / n
    w0     = np.zeros(d)
    it_max = 2000

    # ─── Βασικές NumPy‐based μέθοδοι ─────────────────────────────────
    gd     = Gd(
        lr=1/L,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    nest   = Nesterov(
        lr=1/L,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    adgd   = Adgd(
        eps=0.0, lr0=1/L,
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
    opt_k1 = AdgdK1OverK(
        lr0=1/L,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    opt_k3 = AdgdKOverKplus3(
        lr0=1/L,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    opt_nes = AdaptiveGDK1onKNesterov(
        lr0=1/L,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )

    # ─── Τρέξιμο όλων ─────────────────────────────────────────────────
    numpy_opts  = [gd, nest, adgd, ad_acc, opt_k1, opt_k3, opt_nes]
    labels_npy  = [
        'GD',
        'Nesterov',
        'AdGD',
        'AdGD-accel',
        'AdGD (k+1)/k',
        'AdGD (k/(k+3))',
        'AdGD (k+1)/k + Nesterov'
    ]
    markers_npy = [',', 'o', '*', '^', '<', '>', 'x']

    for opt in numpy_opts:
        opt.run(w0.copy())
        opt.compute_loss_on_iterates()

    # ─── Plot όλων μαζί ────────────────────────────────────────────────
    f_star = min(np.min(opt.losses) for opt in numpy_opts)

    plt.figure(figsize=(8, 6))
    for opt, m, lab in zip(numpy_opts, markers_npy, labels_npy):
        opt.plot_losses(marker=m, f_star=f_star, label=lab)

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
