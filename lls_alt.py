#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

# Import optimizers
from optimizers import *


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

    # Set variable numbers
    n = 4000
    m = 1000

    A, b = generate_simple_quadratic(m, n)
    At = A.T

    # 2) Lipschitz constant για least‐squares: L = max eigenvalue((A^TA)/n)
    G = A.T @ A
    # G /= n
    L = np.max(np.linalg.eigvalsh(G))

    # 3) Ορισμός loss & gradient
    def loss_fn(w):
        r = A.dot(w) - b
        return 0.5 * pow(np.linalg.norm(r), 2)
    def grad_fn(w):
        return At.dot(A.dot(w) - b)

    # 4) Σταθερές πειράματος
    w0 = np.zeros(n)
    it_max = 8000

    adgd = Adgd(eps=0.0, lr0=1e-06, isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    adacc = AdgdAccel(isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    adgdNesCons = AdaPGNesterov(lr0=1e-06, isConservative=True, isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    adgdNes = AdaPGNesterov(lr0=1e-06, isConservative=False, isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
    nes = Nesterov(lr=1/L, strongly_convex=False, mu=0, isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)

    print("## Nesterov with 1/L ##")
    # nes.run(w0)    
    print("## AdGD ##")
    # adgd.run(w0)
    print("## Nesterov AdGD conservative ##")
    adgdNesCons.run(w0)
    print("## Nesterov AdGD ##")
    adgdNes.run(w0)
    print("## Accelerated AdGD ##")
    # adacc.run(w0)

if __name__ == '__main__':
    main()