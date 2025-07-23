# Code for logistic regression with l2 regularization problem

import os

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_svmlight_file

# Import optimizers
from optimizers import *

from loss_functions import logistic_loss, logistic_gradient

# Generate problem
repo_root = os.path.dirname(__file__)
X_sp, y   = load_svmlight_file(os.path.join(repo_root, 'datasets', 'mushrooms'))
X, y      = X_sp.toarray(), y
if set(np.unique(y)) == {1, 2}:
    y = (y == 2).astype(float)

n, d   = X.shape
L      = 0.25 * np.max(la.eigvalsh((X.T @ X) / n))
l2     = L / n
w0     = np.zeros(d)
it_max = 2000

fmin = 0

adgd = Adgd(eps=0.0, lr0=1e-06, prox_type="none", fmin=fmin, reg_param=1, 
isVerbose=True, loss_func=lambda w: logistic_loss(w, X, y, l2),
grad_func=lambda w: logistic_gradient(w, X, y, l2), it_max=it_max)

adacc = AdgdAccel(fmin=fmin, reg_param=1, isVerbose=True, 
loss_func=lambda w: logistic_loss(w, X, y, l2),
grad_func=lambda w: logistic_gradient(w, X, y, l2), it_max=it_max)

adgdNesCons = AdaPGNesterov(lr0=1e-06, prox_type="none", fmin=fmin, reg_param=1, 
isConservative=True, isVerbose=True, loss_func=lambda w: logistic_loss(w, X, y, l2),
grad_func=lambda w: logistic_gradient(w, X, y, l2), it_max=it_max)

adgdNes = AdaPGNesterov(lr0=1e-06, prox_type="none", fmin=fmin, reg_param=1, 
isConservative=False, isVerbose=True, loss_func=lambda w: logistic_loss(w, X, y, l2),
grad_func=lambda w: logistic_gradient(w, X, y, l2), it_max=it_max)
# nes = Nesterov(lr=1/L, strongly_convex=False, mu=0, reg_param=1, isVerbose=True, loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)

print("## Nesterov with 1/L ##")
# nes.run(w0)    
print("## AdGD ##")
adgd.run(w0)
print("## Nesterov AdGD conservative ##")
adgdNesCons.run(w0)
print("## Nesterov AdGD ##")
adgdNes.run(w0)
print("## Accelerated AdGD ##")
# adacc.run(w0)