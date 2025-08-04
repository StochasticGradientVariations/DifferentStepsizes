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

# Import optimizers
from optimizers import *

from loss_functions import logistic_gradient  # μόνο αυτό χρειαζόμαστε

# ─── Plot style ───────────────────────────────────────────────────
sns.set(style="whitegrid", font_scale=1.2, context="talk")
plt.rcParams['mathtext.fontset'] = 'cm'


def logistic_hessian(w, X, y, l2=0.0):
    z = safe_sparse_dot(X, w, dense_output=True).ravel()
    p = scipy.special.expit(z)
    W = p * (1 - p)
    return (X.T * W) @ X + l2 * np.eye(len(w))


def main():
    # 1) Φόρτωση mushrooms
    fmin = 0
    repo = os.path.dirname(__file__)
    print(repo)
    # create the folder where data will be saved (if it doesn’t already exist)
    os.makedirs('results/cubic/data', exist_ok=True)

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
    it_max = 2000
    tune = it_max // 2

    # 5) Lipschitz constant L0 = ‖Q‖₂
    # L0 = np.max(np.linalg.eigvalsh(Q))
    # lrs = np.logspace(-4, -1, 10) / L0

    # # 6) grid‐search για AdaptiveNPGM
    # gammas = [0.01, 0.1, 1.0, 10.0]
    # best = None
    # best_loss = np.inf
    # for g0 in gammas:
    #     tmp = AdaptiveNPGM(gamma0=g0, loss_func=loss_fn, grad_func=grad_fn, it_max=tune)
    #     tmp.run(w0)
    #     tmp.compute_loss_on_iterates()
    #     if tmp.losses[-1] < best_loss:
    #         best_loss = tmp.losses[-1]
    #         best = g0
    # print("Best gamma0 for AdaptiveNPGM:", best)
    # best_gamma0 = best
    #
    # # 7) Tuning για GD
    # vals = []
    # for lr in lrs:
    #     tmp = Gd(lr=lr, loss_func=loss_fn, grad_func=grad_fn, it_max=tune)
    #     tmp.run(w0)
    #     tmp.compute_loss_on_iterates()
    #     vals.append(tmp.losses[-1])
    # best_lr = lrs[np.nanargmin(vals)]
    # gd = Gd(lr=best_lr, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    # gd.run(w0)
    #
    # # 8) Tuning για Nesterov
    # vals = []
    # for lr in lrs:
    #     tmp = Nesterov(lr=lr, loss_func=loss_fn, grad_func=grad_fn, it_max=tune)
    #     tmp.run(w0)
    #     tmp.compute_loss_on_iterates()
    #     vals.append(tmp.losses[-1])
    # best_lr_nes = lrs[np.nanargmin(vals)]
    # nest = Nesterov(lr=best_lr_nes, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    # nest.run(w0)

    # These are the main algorithms we want to compare. Standard AdGD, AccAdGD, AdGD with Nesterov
    # momentum but conservative stepsize, AdGD with Nesterov and actual stepsize

    adgd = Adgd(eps=0.0, lr0=1e-06, prox_type="none", fmin=fmin, reg_param=1, isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    adacc = AdgdAccel(fmin=fmin, reg_param=1, isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    adgdNesCons = AdaPGNesterov(lr0=1e-06, prox_type="none", fmin=fmin, reg_param=1, isConservative=True, isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    adgdNes = AdaPGNesterov(lr0=1e-06, prox_type="none", fmin=fmin, reg_param=1, isConservative=False, isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    # nes = Nesterov(lr=1/L, strongly_convex=False, mu=0, reg_param=1, isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    # initialize the Nesterov optimizer with custom settings
    nes = Nesterov(
        lr=1.0 / la.norm(Q, 2),
        strongly_convex=False,
        mu=0,
        loss_func=loss_fn,
        grad_func=grad_fn,
        it_max=it_max,
        isVerbose=True,
        prox_type="none",
        reg_param=1
    )

    # print("## Nesterov with 1/L ##")
    # # nes.run(w0)

    # run Nesterov optimizer and save training metrics
    print("## Nesterov with 1/L ##")
    nes.run(w0)
    np.save('results/cubic/data/cubic_Nes_loss.npy', nes.loss_hist)
    np.save('results/cubic/data/cubic_Nes_gradnorm.npy', nes.grad_norm_hist)
    np.save('results/cubic/data/cubic_Nes_lrs.npy', nes.lr_hist)

    # print("## AdGD ##")
    # adgd.run(w0)
    # run AdGD and save training history
    print("## AdGD ##")
    adgd.run(w0)
    np.save('results/cubic/data/cubic_Adgd_loss.npy', adgd.loss_hist)
    np.save('results/cubic/data/cubic_Adgd_gradnorm.npy', adgd.grad_norm_hist)
    np.save('results/cubic/data/cubic_Adgd_lrs.npy', adgd.lr_hist)

    # print("## Nesterov AdGD conservative ##")
    # adgdNesCons.run(w0)
    # run AdGD + Nesterov (conservative variant) and save training history
    print("## AdGD + Nesterov (conservative) ##")
    adgdNesCons.run(w0)
    np.save('results/cubic/data/cubic_AdgdNesCons_loss.npy', adgdNesCons.loss_hist)
    np.save('results/cubic/data/cubic_AdgdNesCons_gradnorm.npy', adgdNesCons.grad_norm_hist)
    np.save('results/cubic/data/cubic_AdgdNesCons_lrs.npy', adgdNesCons.lr_hist)

    # print("## Nesterov AdGD ##")
    # adgdNes.run(w0)
    # run AdGD + Nesterov (standard variant) and save training history
    print("## AdGD + Nesterov ##")
    adgdNes.run(w0)
    np.save('results/cubic/data/cubic_AdgdNes_loss.npy', adgdNes.loss_hist)
    np.save('results/cubic/data/cubic_AdgdNes_gradnorm.npy', adgdNes.grad_norm_hist)
    np.save('results/cubic/data/cubic_AdgdNes_lrs.npy', adgdNes.lr_hist)

    # print("## Accelerated AdGD ##")
    # # adacc.run(w0)
    # run AdGD-Accel and save training history
    print("## AdGD-Accel ##")
    adacc.run(w0)
    np.save('results/cubic/data/cubic_AdgdAccel_loss.npy', adacc.loss_hist)
    np.save('results/cubic/data/cubic_AdgdAccel_gradnorm.npy', adacc.grad_norm_hist)
    np.save('results/cubic/data/cubic_AdgdAccel_lrs.npy', adacc.lr_hist)


    # # 10) Οι επιλεγμένες μέθοδοι προς σύγκριση
    # extras, labs, mks = [], [], []
    # variants = [
    #     (AdaptiveGDK1onKNesterov, 'AdGD+Nes', 'x'),
    #     (ADPG_Momentum,              'ADPG_M1',    'h'),
    #     (AdaptiveNPGM,               'Adaptive NPGM', 's'),
    # ]
    #
    # for cls, lab, mk in variants:
    #     kwargs = dict(loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    #     if cls is AdaptiveNPGM:
    #         kwargs['gamma0'] = best_gamma0
    #         kwargs['gamma_prev'] = best_gamma0
    #     opt = cls(**kwargs)
    #     opt.run(w0)
    #     extras.append(opt)
    #     labs.append(lab)
    #     mks.append(mk)

    # Write plotting script
    # # 11) Plot όλων
    # methods = [gd, nest, adgd, adacc] + extras
    # labels = ['GD', 'Nesterov', 'AdGD', 'AdGD-accel'] + labs
    # marks = [',', 'o', '*', '^'] + mks
    #
    # for m in methods:
    #     m.compute_loss_on_iterates()
    # fstar = min(np.min(m.losses) for m in methods)
    #
    # plt.figure(figsize=(12, 6))
    # for m, lab, mk in zip(methods, labels, marks):
    #     m.plot_losses(marker=mk, markevery=50, f_star=fstar, label=lab)
    #
    # plt.yscale('log')
    # plt.xlabel('Iteration')
    # plt.ylabel(r'$m(w^k) - m^*$')
    # plt.legend(ncol=1, frameon=False, loc='upper left', bbox_to_anchor=(1.05, 1.0))
    # plt.subplots_adjust(right=0.7)
    # plt.show()


if __name__ == '__main__':
    main()
