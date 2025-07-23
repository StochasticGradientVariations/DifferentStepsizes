# Code for Lasso problems (possibly with different powers): 1/p||Ax-b||_r^p + λ||x||_1
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

# Import optimizers
from optimizers import *

# Problem parameters
n = 300
m = 100
k = 30
# Set the p and r parameters
power = 2
norm = 2
# Set the λ parameter
lamb = 1

# Generate problem
assert k < n
assert n > m

B = np.random.uniform(-1., 1., [m, n])

v = np.random.uniform(0, 1., m)
yopt = v / np.linalg.norm(v, 2)

p = np.dot(B.T, yopt)
perm = np.argsort(np.abs(p))[::-1]

alpha = np.zeros(n)
xi = np.random.uniform(0, 1, n)

xopt = np.zeros(n)

for i in range(n):
    if i < k:
        alpha[perm[i]] = lamb / np.abs(p[perm[i]])
    elif np.abs(p[perm[i]]) < 0.1 * lamb:
        alpha[perm[i]] = lamb
    else:
        alpha[perm[i]] = lamb * xi[perm[i]] / np.abs(p[perm[i]])

A = np.matmul(B, np.diag(alpha))
xi = np.random.uniform(0, 1 / np.sqrt(k), n)
xopt = np.zeros(n)

q = np.dot(A.T, yopt)
for i in range(n):
    if i < k:
        xopt[perm[i]] = xi[perm[i]] * np.sign(q[perm[i]])

conj_power = power / (power - 1)
conj_norm = norm / (norm - 1)

b = np.sign(yopt) * np.power(np.abs(yopt), conj_power - 1) + np.dot(A, xopt)

fmin = np.sum(np.power(np.abs(yopt), conj_power)) / conj_power + lamb * np.sum(np.abs(xopt))

# loss and gradient functions
def loss_fn(w):
    r = A.dot(w) - b
    return np.sum(np.power(np.abs(r), power)) / power + lamb * np.sum(np.abs(w))
def grad_fn(w):
    if norm == power:
        return np.sign(A.T.dot(A.dot(w) - b)) * np.power(np.abs(A.T.dot(A.dot(w) - b)), power - 1)
    else:
        return np.power(np.linalg.norm(A.T.dot(A.dot(w) - b) - b, 2), power - 2) * (A.T.dot(A.dot(w) - b) - b)

w0 = np.random.rand(n)
it_max = 2000

adgd = Adgd(eps=0.0, lr0=1e-06, prox_type="l1", fmin=fmin, reg_param=1, isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
adacc = AdgdAccel(fmin=fmin, reg_param=1, isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
adgdNesCons = AdaPGNesterov(lr0=1e-06, prox_type="l1", fmin=fmin, reg_param=1, isConservative=True, isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
adgdNes = AdaPGNesterov(lr0=1e-06, prox_type="l1", fmin=fmin, reg_param=1, isConservative=False, isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)
# nes = Nesterov(lr=1/L, strongly_convex=False, mu=0, reg_param=1, isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)

print("## Nesterov with 1/L ##")
# nes.run(w0)    
print("## AdGD ##")
# adgd.run(w0)
print("## Nesterov AdGD conservative ##")
# adgdNesCons.run(w0)
print("## Nesterov AdGD ##")
adgdNes.run(w0)
print("## Accelerated AdGD ##")
# adacc.run(w0)
