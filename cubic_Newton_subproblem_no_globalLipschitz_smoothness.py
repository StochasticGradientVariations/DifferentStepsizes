#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from optimizers import (
    Gd,
    Nesterov,
    Adgd,
    AdgdAccel,
    AdaptiveGDK1onKNesterov,
    ADPG_Momentum,
    AdaptiveNPGM
)

# 1) Συνθέτουμε τυχαίο PSD matrix H0 και gradient g0
np.random.seed(0)
d = 50
A = np.random.randn(d, d)
H0 = A.T @ A          # PSD
g0 = np.random.randn(d)

# 2) Ορισμός M>0
M = 1.0

def grad_fn(p):
    normp = la.norm(p)
    return g0 + H0 @ p + 0.5 * M * normp * p

# Για το ιστορικό, μετράμε απλώς την norm του gradient:
loss_norm = lambda p: la.norm(grad_fn(p))

# 3) Τρέχουμε τους optimizer με ίδιο budget
it_max = 1000
x0 = np.zeros(d)

optimizers = [
    ("GD",               Gd(lr=1e-3,        loss_func=loss_norm, grad_func=grad_fn, it_max=it_max)),
    ("Nesterov",         Nesterov(lr=1e-3,  loss_func=loss_norm, grad_func=grad_fn, it_max=it_max)),
    ("AdGD",             Adgd(eps=0.0, lr0=1e-3, loss_func=loss_norm, grad_func=grad_fn, it_max=it_max)),
    ("AdGD-accel",       AdgdAccel(loss_func=loss_norm, grad_func=grad_fn, it_max=it_max)),
    ("AdGD + Nesterov",  AdaptiveGDK1onKNesterov(lr0=1e-3, loss_func=loss_norm, grad_func=grad_fn, it_max=it_max)),
    ("ADPG_Momentum1",   ADPG_Momentum(lr0=1e-3, loss_func=loss_norm, grad_func=grad_fn, it_max=it_max)),
    ("Adaptive NPGM",    AdaptiveNPGM(gamma0=1.0, gamma_prev=1.0,
                                      loss_func=loss_norm, grad_func=grad_fn, it_max=it_max)),
]

for name, opt in optimizers:
    print(f"Running {name:20s}...", end=" ", flush=True)
    opt.run(x0.copy())
    opt.compute_loss_on_iterates()
    print("done")

# 4) Plot
plt.figure(figsize=(8,5))
markers = ['o','v','^','<','>','s','p']
for (name, opt), mk in zip(optimizers, markers):
    plt.semilogy(opt.losses, marker=mk, markevery=100, label=name)
plt.xlabel("Iteration")
plt.ylabel(r"$\|\nabla m(p^k)\|$")
plt.legend(ncol=2, fontsize=8)
plt.title("Cubic–Newton Subproblem (non‐smooth Hessian)")
plt.tight_layout()
plt.show()
