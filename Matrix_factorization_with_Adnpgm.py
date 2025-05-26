#!/usr/bin/env python
# coding: utf-8

"""
Γρήγορη matrix factorization με optimizers.py
Μόνο οι επιλεγμένες μέθοδοι, με λιγότερες επαναλήψεις.
"""

import time
import numpy as np
import scipy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt

from optimizers import (
    Gd,
    Nesterov,
    Adgd,
    AdgdK1OverK,
    AdaptiveGDK1onKNesterov,
    ADPG_Momentum,
    AdaptiveNPGM    # <--- προσθήκη της νέας μεθόδου
)

def load_data(path="datasets/u.data"):
    names = ['user_id','item_id','rating','timestamp']
    df = pd.read_csv(path, sep='\t', names=names)
    m, n = df.user_id.max(), df.item_id.max()
    A = np.zeros((m, n))
    for u, i, r, _ in df.itertuples(index=False):
        A[u-1, i-1] = r
    return A

def make_problem(A, r):
    m, n = A.shape
    def df(X):
        U = X[:m].reshape(m, r)
        V = X[m:].reshape(n, r)
        R = U @ V.T - A
        return np.vstack([R @ V, R.T @ U])
    return df, m, n

def plot_all(opts, labels):
    plt.figure(figsize=(7,5))
    markers = [',','o','*','D','<','>','s']  # προστέθηκε ακόμα ένα marker για AdaptiveNPGM
    f_star = min(opt.losses.min() for opt in opts)
    for opt, lab, mk in zip(opts, labels, markers):
        opt.plot_losses(label=lab, marker=mk, f_star=f_star)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel(r'$\|\nabla f(x^k)\|$')
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    plt.show()

def main():
    # Φόρτωση δεδομένων & πρόβλημα
    A = load_data()
    r = 20
    df_func, m, n = make_problem(A, r)

    # Αρχικοποίηση
    np.random.seed(0)
    X0 = np.random.randn(m+n, r)

    # Params
    N     = 5000    # λιγότερες επαναλήψεις
    L_gd  = 1000
    L_nes = 30000

    # Συναρτήσεις διεπαφής
    loss_grad = lambda X: LA.norm(df_func(X))
    grad_func = df_func

    # Επιλεγμένες μέθοδοι
    names_and_opts = [
        ('GD',               Gd(lr=1.0/L_gd,
                                loss_func=loss_grad,
                                grad_func=grad_func,
                                it_max=N)),
        ('Nesterov',         Nesterov(lr=1.0/L_nes,
                                      loss_func=loss_grad,
                                      grad_func=grad_func,
                                      it_max=N)),
        ('AdGD',             Adgd(eps=0.0,
                                lr0=1e-9,
                                loss_func=loss_grad,
                                grad_func=grad_func,
                                it_max=N)),
        ('AdGD (k+1)/k',     AdgdK1OverK(lr0=1.0/L_gd,
                                       loss_func=loss_grad,
                                       grad_func=grad_func,
                                       it_max=N)),
        ('AdGD + Nesterov',  AdaptiveGDK1onKNesterov(lr0=1.0/L_gd,
                                                   loss_func=loss_grad,
                                                   grad_func=grad_func,
                                                   it_max=N)),
        ('ADPG_Momentum',    ADPG_Momentum(lr0=1.0/L_gd,
                                        loss_func=loss_grad,
                                        grad_func=grad_func,
                                        it_max=N)),
        ('Adaptive NPGM',    AdaptiveNPGM(gamma0=1.0/L_gd,
                                          gamma_prev=1.0/L_gd,
                                          loss_func=loss_grad,
                                          grad_func=grad_func,
                                          it_max=N)),
    ]

    opts, labels = [], []
    for name, opt in names_and_opts:
        print(f"Running {name:20s}...", end=" ", flush=True)
        t0 = time.time()
        opt.run(X0.copy())
        opt.compute_loss_on_iterates()
        print(f"{time.time()-t0:.1f}s")
        opts.append(opt)
        labels.append(name)

    plot_all(opts, labels)

if __name__ == "__main__":
    main()
