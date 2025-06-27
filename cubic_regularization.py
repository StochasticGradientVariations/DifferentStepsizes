#!/usr/bin/env python
# coding: utf-8
"""
Cubic–Newton subproblem (M=1, no scaling) όπως στο Patrinos et al. [53].
Λύνουμε   min_w  ½ wᵀ Q w + qᵀ w + (1/6)||w||³
όπου Q=∇²ℓ(0), q=∇ℓ(0) για την logistic loss στο w=0.
Στη συνέχεια συγκρίνουμε GD, Nesterov, AdGD, AdGD-accel,
AdaptiveGDK1onKNesterov, ADPG_Momentum και AdaptiveNPGM.
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
    AdaptiveGDK1onKNesterov,
    ADPG_Momentum,
    AdaptiveNPGM
)

from loss_functions import logistic_gradient  # μόνο αυτό χρειαζόμαστε

# ─── Plot style ───────────────────────────────────────────────────
sns.set(style="whitegrid", font_scale=1.2, context="talk")
plt.rcParams['mathtext.fontset'] = 'cm'


def logistic_hessian(w, X, y, l2=0.0):
    """Hessian της logistic στο w."""
    z = safe_sparse_dot(X, w, dense_output=True).ravel()
    p = scipy.special.expit(z)
    W = p * (1 - p)
    return (X.T * W) @ X + l2 * np.eye(len(w))


def main():
    # 1) Φόρτωση mushrooms
    repo = os.path.dirname(__file__)
    X_sp, y = load_svmlight_file(os.path.join(repo, 'datasets', 'mushrooms'))
    X, y = X_sp.toarray(), y
    if set(np.unique(y)) == {1, 2}:
        y = (y == 2).astype(float)

    n, d = X.shape

    # 2) Q, q στο w0=0
    w0 = np.zeros(d)
    l2 = 0.0
    q = logistic_gradient(w0, X, y, l2=l2)
    Q = logistic_hessian(w0, X, y, l2=l2)

    # 3) ορισμός m(w) και ∇m(w)
    def loss_fn(w):
        return q @ w + 0.5 * (w @ (Q @ w)) + (1/6) * la.norm(w) ** 3

    def grad_fn(w):
        normw = la.norm(w)
        return q + Q @ w + 0.5 * normw * w

    # 4) budgets & tuning
    it_max = 1000
    tune = it_max // 2

    # 5) Lipschitz constant L0 = ‖Q‖₂
    L0 = np.max(np.linalg.eigvalsh(Q))
    lrs = np.logspace(-4, -1, 10) / L0

    # 6) grid‐search για AdaptiveNPGM
    gammas = [0.01, 0.1, 1.0, 10.0]
    best = None
    best_loss = np.inf
    for g0 in gammas:
        tmp = AdaptiveNPGM(gamma0=g0, loss_func=loss_fn, grad_func=grad_fn, it_max=tune)
        tmp.run(w0)
        tmp.compute_loss_on_iterates()
        if tmp.losses[-1] < best_loss:
            best_loss = tmp.losses[-1]
            best = g0
    print("Best gamma0 for AdaptiveNPGM:", best)
    best_gamma0 = best

    # 7) Tuning για GD
    vals = []
    for lr in lrs:
        tmp = Gd(lr=lr, loss_func=loss_fn, grad_func=grad_fn, it_max=tune)
        tmp.run(w0)
        tmp.compute_loss_on_iterates()
        vals.append(tmp.losses[-1])
    best_lr = lrs[np.nanargmin(vals)]
    gd = Gd(lr=best_lr, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    gd.run(w0)

    # 8) Tuning για Nesterov
    vals = []
    for lr in lrs:
        tmp = Nesterov(lr=lr, loss_func=loss_fn, grad_func=grad_fn, it_max=tune)
        tmp.run(w0)
        tmp.compute_loss_on_iterates()
        vals.append(tmp.losses[-1])
    best_lr_nes = lrs[np.nanargmin(vals)]
    nest = Nesterov(lr=best_lr_nes, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    nest.run(w0)

    # 9) Κλασικά AdGD & AdGD-accel
    adgd = Adgd(eps=0.0, lr0=1.0, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    adacc = AdgdAccel(loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    adgd.run(w0)
    adacc.run(w0)

    # 10) Οι επιλεγμένες μέθοδοι προς σύγκριση
    extras, labs, mks = [], [], []
    variants = [
        (AdaptiveGDK1onKNesterov, 'AdGD+Nes', 'x'),
        (ADPG_Momentum,              'ADPG_M1',    'h'),
        (AdaptiveNPGM,               'Adaptive NPGM', 's'),
    ]

    for cls, lab, mk in variants:
        kwargs = dict(loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
        if cls is AdaptiveNPGM:
            kwargs['gamma0'] = best_gamma0
            kwargs['gamma_prev'] = best_gamma0
        opt = cls(**kwargs)
        opt.run(w0)
        extras.append(opt)
        labs.append(lab)
        mks.append(mk)

    # 11) Plot όλων
    methods = [gd, nest, adgd, adacc] + extras
    labels = ['GD', 'Nesterov', 'AdGD', 'AdGD-accel'] + labs
    marks = [',', 'o', '*', '^'] + mks

    for m in methods:
        m.compute_loss_on_iterates()
    fstar = min(np.min(m.losses) for m in methods)

    plt.figure(figsize=(12, 6))
    for m, lab, mk in zip(methods, labels, marks):
        m.plot_losses(marker=mk, markevery=50, f_star=fstar, label=lab)

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(r'$m(w^k) - m^*$')
    plt.legend(ncol=1, frameon=False, loc='upper left', bbox_to_anchor=(1.05, 1.0))
    plt.subplots_adjust(right=0.7)
    plt.show()


if __name__ == '__main__':
    main()
