#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import numpy.linalg as la
from sklearn.datasets import load_svmlight_file

# Import optimizers
from optimizers import Adgd, AdgdAccel, AdaPGNesterov, Nesterov
from loss_functions import logistic_loss, logistic_gradient

# 1)
DATASETS    = ['mushrooms', 'covtype.bz2', 'w8a']
RESULTS_DIR = 'results/logistic_l2/data'
os.makedirs(RESULTS_DIR, exist_ok=True)

it_max = 2000

# Loop through each dataset
for ds in DATASETS:
    # Strip extension for folder/file name
    ds_name = ds.replace('.bz2', '')
    print(f"\n=== Running on dataset: {ds} (folder: {ds_name}) ===")
    # Output folder for this dataset
    ds_dir = os.path.join(RESULTS_DIR, ds_name)
    os.makedirs(ds_dir, exist_ok=True)

    # load data
    repo_root = os.path.dirname(__file__)
    path = os.path.join(repo_root, 'datasets', ds)
    X_sp, y    = load_svmlight_file(path)
    X, y       = X_sp.toarray(), y
    if set(np.unique(y)) == {1, 2}:
        y = (y == 2).astype(float)

    # Parameters logistic‐L2
    n, d = X.shape
    L    = 0.25 * np.max(la.eigvalsh((X.T @ X) / n))
    l2   = L / n
    w0   = np.zeros(d)

    # Θέτουμε f* = 0
    fmin = 0

    # 3)  optimizers
    nes = Nesterov(
        lr=1.0/L,
        strongly_convex=False,
        mu=0,
        isVerbose=False,
        prox_type="none",
        reg_param=1,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    adgd = Adgd(
        eps=0.0, lr0=1e-6,
        prox_type="none", fmin=fmin, reg_param=1,
        isVerbose=False,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    adacc = AdgdAccel(
        fmin=fmin, reg_param=1, isVerbose=False,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    adgdNesCons = AdaPGNesterov(
        lr0= 1e-6, prox_type="none", fmin=fmin, reg_param=1,   #try 1e-3,  previous: 1e-6
        isConservative=True, isVerbose=False,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )
    adgdNes = AdaPGNesterov(
        lr0=1e-6, prox_type="none", fmin=fmin, reg_param=1,
        isConservative=False, isVerbose=False,
        loss_func=lambda w: logistic_loss(w, X, y, l2),
        grad_func=lambda w: logistic_gradient(w, X, y, l2),
        it_max=it_max
    )

    # 4) Εκτέλεση και αποθήκευση ιστορικών
    for name, opt in [
        ('Nes', nes),
        ('Adgd', adgd),
        ('AdgdAccel', adacc),
        ('AdgdNesCons', adgdNesCons),
        ('AdgdNes', adgdNes),
    ]:
        print(f"--> {name} …")
        opt.run(w0)
        np.save(os.path.join(ds_dir, f'{ds_name}_{name}_loss.npy'), opt.loss_hist)
        np.save(os.path.join(ds_dir, f'{ds_name}_{name}_gradnorm.npy'), opt.grad_norm_hist)
        np.save(os.path.join(ds_dir, f'{ds_name}_{name}_lrs.npy'), opt.lr_hist)
