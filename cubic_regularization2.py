#!/usr/bin/env python
# coding: utf-8
"""
Cubic–Newton subproblem (M=10, 20, 100) όπως στο Patrinos et al. [53].
Λύνουμε   min_w  ½ wᵀ Q w + qᵀ w + (M/6)||w||³
όπου Q=∇²ℓ(0), q=∇ℓ(0) για την logistic loss στο w=0.
Στη συνέχεια συγκρίνουμε GD, Nesterov, AdGD, AdGD-accel,
AdGD+Nes⁽c⁾, AdGD+Nes.
"""
import os
import numpy as np
import numpy.linalg as la
import scipy.special
from sklearn.datasets import load_svmlight_file
from sklearn.utils.extmath import safe_sparse_dot
import matplotlib.pyplot as plt
import seaborn as sns

# Import optimizers
from optimizers import *
from loss_functions import logistic_gradient

# ─── Plot style ───────────────────────────────────────────────────
sns.set(style="whitegrid", font_scale=1.2, context="talk")
plt.rcParams['mathtext.fontset'] = 'cm'

def logistic_hessian(w, X, y, l2=0.0):
    z = safe_sparse_dot(X, w, dense_output=True).ravel()
    p = scipy.special.expit(z)
    W = p * (1 - p)
    return (X.T * W) @ X + l2 * np.eye(len(w))

def main():
    # 1) Load covtype dataset
    fmin = 0
    repo = os.path.dirname(__file__)
    print(repo)
    os.makedirs('results/cubic/data', exist_ok=True)

    X_sp, y = load_svmlight_file(os.path.join(repo, 'datasets', 'covtype.bz2'))
    X, y = X_sp.toarray(), y
    n, d = X.shape

    # 2) Loop over M values
    M_values = [10, 20, 100]
    iters = {10: 4000, 20: 3000, 100: 1000}

    for M in M_values:
        print(f"\n--- Running for M={M} ---\n")
        it_max = iters[M]
        tune = it_max // 2

        w0 = np.zeros(d)
        l2 = 0.0
        q = logistic_gradient(w0, X, y, l2=l2)
        Q = logistic_hessian(w0, X, y, l2=l2)

        def loss_fn(w):
            return q @ w + 0.5 * (w @ (Q @ w)) + (M/6) * la.norm(w) ** 3

        def grad_fn(w):
            normw = la.norm(w)
            return q + Q @ w + 0.5 * M * normw * w / 3

        # Lipschitz constant L0 = ‖Q‖₂
        L0 = np.max(np.linalg.eigvalsh(Q))
        lrs = np.logspace(-4, -1, 10) / L0

        # Tuning GD
        vals = []
        for lr in lrs:
            tmp = Gd(lr=lr, loss_func=loss_fn, grad_func=grad_fn, it_max=tune)
            tmp.run(w0)
            tmp.compute_loss_on_iterates()
            vals.append(tmp.losses[-1])
        best_lr = lrs[np.nanargmin(vals)]
        gd = Gd(lr=best_lr, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
        gd.run(w0)

        # Tuning Nesterov
        vals = []
        for lr in lrs:
            tmp = Nesterov(lr=lr, loss_func=loss_fn, grad_func=grad_fn, it_max=tune)
            tmp.run(w0)
            tmp.compute_loss_on_iterates()
            vals.append(tmp.losses[-1])
        best_lr_nes = lrs[np.nanargmin(vals)]
        nest = Nesterov(lr=best_lr_nes, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
        nest.run(w0)

        # AdGD family
        adgd = Adgd(eps=0.0, lr0=1e-6, prox_type="none", fmin=fmin, reg_param=1,
                    isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
        adgd.run(w0)
        adacc = AdgdAccel(fmin=fmin, reg_param=1, isVerbose=True, loss_func=loss_fn,
                          grad_func=grad_fn, it_max=it_max)
        adacc.run(w0)
        adgdNesCons = AdaPGNesterov(lr0=1e-6, prox_type="none", fmin=fmin, reg_param=1,
                                    isConservative=True, isVerbose=True, loss_func=loss_fn,
                                    grad_func=grad_fn, it_max=it_max)
        adgdNesCons.run(w0)
        adgdNes = AdaPGNesterov(lr0=1e-6, prox_type="none", fmin=fmin, reg_param=1,
                                isConservative=False, isVerbose=True, loss_func=loss_fn,
                                grad_func=grad_fn, it_max=it_max)
        adgdNes.run(w0)

        # Save results με _M{M} στο όνομα
        np.save(f'results/cubic/data/cubic_Gd_loss_M{M}.npy', gd.loss_hist)
        np.save(f'results/cubic/data/cubic_Gd_gradnorm_M{M}.npy', gd.grad_norm_hist)
        np.save(f'results/cubic/data/cubic_Gd_lrs_M{M}.npy', gd.lr_hist)

        np.save(f'results/cubic/data/cubic_Nes_loss_M{M}.npy', nest.loss_hist)
        np.save(f'results/cubic/data/cubic_Nes_gradnorm_M{M}.npy', nest.grad_norm_hist)
        np.save(f'results/cubic/data/cubic_Nes_lrs_M{M}.npy', nest.lr_hist)

        np.save(f'results/cubic/data/cubic_Adgd_loss_M{M}.npy', adgd.loss_hist)
        np.save(f'results/cubic/data/cubic_Adgd_gradnorm_M{M}.npy', adgd.grad_norm_hist)
        np.save(f'results/cubic/data/cubic_Adgd_lrs_M{M}.npy', adgd.lr_hist)

        np.save(f'results/cubic/data/cubic_AdgdAccel_loss_M{M}.npy', adacc.loss_hist)
        np.save(f'results/cubic/data/cubic_AdgdAccel_gradnorm_M{M}.npy', adacc.grad_norm_hist)
        np.save(f'results/cubic/data/cubic_AdgdAccel_lrs_M{M}.npy', adacc.lr_hist)

        np.save(f'results/cubic/data/cubic_AdgdNesCons_loss_M{M}.npy', adgdNesCons.loss_hist)
        np.save(f'results/cubic/data/cubic_AdgdNesCons_gradnorm_M{M}.npy', adgdNesCons.grad_norm_hist)
        np.save(f'results/cubic/data/cubic_AdgdNesCons_lrs_M{M}.npy', adgdNesCons.lr_hist)

        np.save(f'results/cubic/data/cubic_AdgdNes_loss_M{M}.npy', adgdNes.loss_hist)
        np.save(f'results/cubic/data/cubic_AdgdNes_gradnorm_M{M}.npy', adgdNes.grad_norm_hist)
        np.save(f'results/cubic/data/cubic_AdgdNes_lrs_M{M}.npy', adgdNes.lr_hist)

if __name__ == '__main__':
    main()
