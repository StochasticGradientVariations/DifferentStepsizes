# cubic_regularization_to_improve.py
#!/usr/bin/env python
# coding: utf-8
"""
Cubic regularization experiment converted from Jupyter notebook to standalone Python script.
Compares first-order methods on the cubic model of the logistic loss around w0.
"""
import os
import numpy as np
import numpy.linalg as la
import scipy.special
from sklearn.datasets import load_svmlight_file
from sklearn.utils.extmath import safe_sparse_dot
import matplotlib.pyplot as plt
import seaborn as sns

from optimizers import (
    Gd, Nesterov, Adgd, AdgdAccel,
    AdgdK1OverK, AdgdKOverKplus3, AdaptiveGDK1onKNesterov,
    AdgdHybrid, AdgdHybrid2,
    ADPG_Momentum, ADPG_Momentum2, ADPG_Momentum3
)
from loss_functions import cubic_loss1, cubic_gradient1, logistic_gradient

# ─── Plot style ───────────────────────────────────────────────────
sns.set(
    style="whitegrid", font_scale=1.2, context="talk",
    palette=sns.color_palette("bright"), color_codes=False
)
plt.rcParams['mathtext.fontset'] = 'cm'

# ─── Helper functions ─────────────────────────────────────────────

def logistic_smoothness(X):
    """
    Global Lipschitz constant L for logistic loss: 0.25 * max eig(X^T X / n)
    """
    n = X.shape[0]
    return 0.25 * np.max(la.eigvalsh((X.T @ X) / n))


def logistic_hessian(w, X, y, l2):
    """
    Hessian of logistic loss at w: X^T diag(p*(1-p)) X + l2*I
    """
    z = safe_sparse_dot(X, w, dense_output=True).ravel()
    p = scipy.special.expit(z)
    W = p * (1 - p)
    # weighted X^T X
    Xw = X.T * W
    return Xw @ X + l2 * np.eye(len(w))

# ─── Main experiment ───────────────────────────────────────────────
def main():
    # 1) Load mushrooms classification dataset
    repo_root = os.path.dirname(__file__)
    data_path = os.path.join(repo_root, 'datasets', 'mushrooms')
    X_sp, y = load_svmlight_file(data_path)
    X, y = X_sp.toarray(), y
    if set(np.unique(y)) == {1, 2}:
        y = (y == 2).astype(float)

    n, d = X.shape

    # 2) Compute logistic smoothness L and scale
    L = logistic_smoothness(X)
    scale = L * n
    l2 = 0.0

    # 3) Linearization point w0, gradient g0 and Hessian H0
    w0 = np.zeros(d)
    g0 = logistic_gradient(w0, X, y, l2)
    H0 = logistic_hessian(w0, X, y, l2)

    # 4) Cubic model parameter
    # M = 100  ετσι το είχανε στο notebook, στο paper https://arxiv.org/pdf/2301.04431#page=20&zoom=100,105,489 βάζουν M=1
    M = 0.1 * L


    # 5) Define cubic model loss and gradient
    def loss_fn(w):
        return cubic_loss1(w, H0, g0, M, scale)

    def grad_fn(w):
        return cubic_gradient1(w, H0, g0, M, scale)

    # 6) Iteration budgets
    it_max = 1000
    tune_iters = it_max // 2

    # 7) Step-size tuning for GD
    lrs = np.logspace(-6, -1, 12)  # π.χ. από 1e-6 ως 1e-1
    final_losses = []
    for lr in lrs:
        tmp = Gd(lr=lr, loss_func=loss_fn, grad_func=grad_fn, it_max=tune_iters)
        tmp.run(w0)
        tmp.compute_loss_on_iterates()
        final_losses.append(tmp.losses[-1])
    best_lr_gd = lrs[np.nanargmin(final_losses)]
    gd = Gd(lr=best_lr_gd, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    gd.run(w0)

    # 8) Step-size tuning for Nesterov
    final_losses = []
    for lr in lrs:
        tmp = Nesterov(lr=lr, loss_func=loss_fn, grad_func=grad_fn, it_max=tune_iters)
        tmp.run(w0)
        tmp.compute_loss_on_iterates()
        final_losses.append(tmp.losses[-1])
    best_lr_nes = lrs[np.nanargmin(final_losses)]
    nest = Nesterov(lr=best_lr_nes, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    nest.run(w0)

    # 9) First-order baselines
    adgd   = Adgd(loss_func=loss_fn, grad_func=grad_fn, eps=0.0, lr0=1.0/L, it_max=it_max)
    ad_acc = AdgdAccel(loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    adgd.run(w0)
    ad_acc.run(w0)

    # 10) Custom adaptive variants
    opts = []
    labels = []
    markers = []

    add = [
        (AdgdK1OverK, 'AdGD (k+1)/k', '<'),
        (AdgdKOverKplus3, 'AdGD (k/(k+3))', '>'),
        (AdaptiveGDK1onKNesterov, 'AdGD+Nesterov', 'x'),
        (AdgdHybrid, 'AdGD Hybrid v1', 'd'),
        (AdgdHybrid2, 'AdGD Hybrid v2', 'D'),
        (ADPG_Momentum, 'ADPG_M1', 'h'),
        (ADPG_Momentum2,'ADPG_M2','p'),
        (ADPG_Momentum3,'ADPG_M3','H'),
    ]
    for cls, lab, mk in add:
        kwargs = dict(loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
        sig = cls.__init__.__code__.co_varnames
        if 'lr0' in sig: kwargs['lr0'] = 1.0/L
        if cls.__name__ == 'ADPG_Momentum3': kwargs['L_global'] = L
        opt = cls(**kwargs)
        opt.run(w0)
        opts.append(opt)
        labels.append(lab)
        markers.append(mk)

    # 11) Collect all methods
    all_methods = [gd, nest, adgd, ad_acc] + opts
    all_labels  = ['GD', 'Nesterov', 'AdGD', 'AdGD-accel'] + labels
    all_markers = [',','o','*','^'] + markers

    for opt in all_methods:
        opt.compute_loss_on_iterates()

    f_star = min(np.min(opt.losses) for opt in all_methods)

    # 12) Plot convergence (από το covtype example)
    plt.figure(figsize=(8, 6))
    for opt, mk, lab in zip(all_methods, all_markers, all_labels):
        opt.plot_losses(marker=mk, f_star=f_star, label=lab)

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(r'$f(x^k) - f_*$')
    plt.title('Cubic Regularization Model on Mushrooms')
    plt.legend(ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
