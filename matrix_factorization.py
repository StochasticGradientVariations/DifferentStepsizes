#!/usr/bin/env python
# coding: utf-8

"""
Matrix factorization experiment με optimizers.py
Δοκιμάζει όλες τις custom μεθόδους μας πλάι στα κλασικά GD/Nesterov/AdGD.
"""

import os
import numpy as np
import scipy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt

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
    ADPG_Momentum
)

def load_data(path="datasets/u.data"):
    names = ['user_id','item_id','rating','timestamp']
    df    = pd.read_csv(path, sep='\t', names=names)
    m, n  = df.user_id.max(), df.item_id.max()
    A     = np.zeros((m, n))
    for u,i,r,_ in df.itertuples(index=False):
        A[u-1, i-1] = r
    return A

def make_problem(A, r):
    m,n = A.shape
    def f(X):
        U = X[:m].reshape(m,r)
        V = X[m:].reshape(n,r)
        return 0.5 * LA.norm(U@V.T - A,'fro')**2
    def df(X):
        U = X[:m].reshape(m,r)
        V = X[m:].reshape(n,r)
        R = U@V.T - A
        return np.vstack([R@V, R.T@U])
    return f, df, m, n

def plot_all(opts, labels):
    plt.figure(figsize=(8,6))
    markers = [',','o','*','D','<','>','x','d','p','h']
    f_star = min(np.min(opt.losses) for opt in opts)
    for opt, lab, mk in zip(opts, labels, markers):
        opt.plot_losses(label=lab, marker=mk, f_star=f_star)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\|\nabla f(x^k)\|$')
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    plt.show()

def main():
    A = load_data()
    r = 20
    f, df, m, n = make_problem(A, r)

    # αρχική τυχαία θέση
    np.random.seed(0)
    X0 = np.random.randn(m+n, r)
    N  = 30000

    # βήματα για GD / Nesterov (πειραματικά)
    L_gd  = 1000
    L_nes = 30000

    # loss/grad interface
    loss_grad = lambda X: LA.norm(df(X))
    grad_func = df

    # ——— Δημιουργία optimizer instances ——————————————————————
    opts = [
        Gd(   lr=1.0/L_gd,
              loss_func=loss_grad, grad_func=grad_func, it_max=N),
        Nesterov(lr=1.0/L_nes,
                 loss_func=loss_grad, grad_func=grad_func, it_max=N),
        Adgd(   eps=0.0, lr0=1e-9,
                loss_func=loss_grad, grad_func=grad_func, it_max=N),
        AdgdAccel(a_lr=0.5,a_mu=0.5,b_lr=0.5,b_mu=0.5,
                  loss_func=loss_grad, grad_func=grad_func, it_max=N),
        AdgdK1OverK(   lr0=1.0/L_gd,
                       loss_func=loss_grad, grad_func=grad_func, it_max=N),
        AdgdKOverKplus3(lr0=1.0/L_gd,
                        loss_func=loss_grad, grad_func=grad_func, it_max=N),
        AdaptiveGDK1onKNesterov(lr0=1.0/L_gd,
                                loss_func=loss_grad, grad_func=grad_func, it_max=N),
        AdgdHybrid(   lr0=1.0/L_gd, b_lr=0.5, b_mu=0.5,
                      loss_func=loss_grad, grad_func=grad_func, it_max=N),
        AdgdHybrid2(  lr0=1.0/L_gd, b_lr=0.5, b_mu=0.5,
                      loss_func=loss_grad, grad_func=grad_func, it_max=N),
        ADPG_Momentum(lr0=1.0/L_gd,
                      loss_func=loss_grad, grad_func=grad_func, it_max=N)
    ]

    labels = [
        'GD',
        'Nesterov',
        'AdGD',
        'AdGD-accel',
        'AdGD (k+1)/k',
        'AdGD (k/(k+3))',
        'AdGD (k+1)/k + Nesterov',
        'AdGD Hybrid v1',
        'AdGD Hybrid v2',
        'ADPG_Momentum'
    ]

    # ——— Τρέξιμο & συλλογή losses ——————————————————————————
    for opt in opts:
        opt.run(X0.copy())
        opt.compute_loss_on_iterates()

    # ——— Plot σύγκρισης ———————————————————————————————
    plot_all(opts, labels)

if __name__ == "__main__":
    main()
