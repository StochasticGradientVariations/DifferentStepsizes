#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

# Οι optimizers από το optimizers.py
from optimizers import *

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


def generate_simple_quadratic(m, n):
    # Generate problem solution and normalize such that $x_\star \in B_(0, 1)$
    x_star = np.random.rand(n)
    x_star = x_star / np.linalg.norm(x_star)

    # Generate matrix A
    A = np.random.rand(m,n)

    # Generate vector b
    b = A @ x_star

    return A, b

def main():
    # sns.set(style="whitegrid", font_scale=1.2)
    # plt.rcParams['mathtext.fontset'] = 'cm'

    # 1) Φόρτωση A,b
    repo_root = os.path.dirname(__file__)

    # Set variable numbers
    n = 4000
    m = 1000
    # A, b = load_movielens_u_data(os.path.join(repo_root, 'datasets', 'u.data'))

    A, b = generate_simple_quadratic(m, n)
    At = A.T
    # n, d = A.shape

    # 2) Lipschitz constant για least‐squares: L = max eigenvalue((A^TA)/n)
    G = A.T @ A
    G /= n
    L = np.max(np.linalg.eigvalsh(G))

    # 3) Ορισμός loss & gradient
    def loss_fn(w):
        r = A.dot(w) - b
        return 0.5 * pow(np.linalg.norm(r), 2)
    def grad_fn(w):
        return At.dot(A.dot(w) - b)

    # 4) Σταθερές πειράματος
    w0 = np.zeros(n)
    it_max = 10000

    # 5) Δημιουργία optimizer instances
    # OPTS = [
    #     Gd, Nesterov, Adgd, AdgdAccel,
    #     AdgdK1OverK, AdgdKOverKplus3, AdaptiveGDK1onKNesterov,
    #     AdgdHybrid, AdgdHybrid2,
    #     ADPG_Momentum, ADPG_Momentum2, ADPG_Momentum3
    # ]
    # LABELS = [
    #     'GD','Nesterov','AdGD','AdGD-accel',
    #     'AdGD (k+1)/k','AdGD (k/(k+3))','AdGD+Nesterov',
    #     'AdGD Hybrid v1','AdGD Hybrid v2',
    #     'ADPG_M1','ADPG_M2','ADPG_M3'
    # ]
    # MARKERS = [',','o','*','^','<','>','x','d','D','h','p','H']
    #
    # optimizers = []
    # for Opt in OPTS:
    #     kwargs = dict(loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    #     sig = Opt.__init__.__code__.co_varnames
    #     if 'lr' in sig:
    #         kwargs['lr'] = 1.0 / L
    #     elif 'lr0' in sig:
    #         kwargs['lr0'] = 1.0 / L
    #     if Opt.__name__ == 'ADPG_Momentum3':
    #         kwargs['L_global'] = L
    #     optimizers.append(Opt(**kwargs))
    #
    # # 6) Τρέξιμο & συλλογή ιστορικού
    # for opt in optimizers:
    #     opt.run(w0.copy())
    #     opt.compute_loss_on_iterates()

    # 7) Plot σύγκλισης
    # f_star = min(np.min(opt.losses) for opt in optimizers)
    # plt.figure(figsize=(8, 6))
    # for opt, mk, lab in zip(optimizers, MARKERS, LABELS):
    #     opt.plot_losses(marker=mk, f_star=f_star, label=lab)
    # plt.yscale('log')
    # plt.xlabel('Iteration')
    # plt.ylabel(r'$f(w^k)-f_*$')
    # plt.title('Linear Least Squares on MovieLens u.data')
    # plt.legend(ncol=2, fontsize=9, frameon=False)
    # plt.tight_layout()
    # plt.show()
    kwargs = dict(loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    AdaPGNesterov(**kwargs).run(w0.copy())

if __name__ == '__main__':
    main()
