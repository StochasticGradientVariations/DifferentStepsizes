#!/usr/bin/env python
# coding: utf-8

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Root directories
DATA_ROOT  = 'results/logistic_l2/data'
PLOTS_ROOT = 'results/logistic_l2/plots'
os.makedirs(PLOTS_ROOT, exist_ok=True)

# Ρυθμίσεις για σταθερά σχήματα & χρώματα
methods = ['Nes', 'Adgd', 'AdgdAccel', 'AdgdNesCons', 'AdgdNes']
colors = ['tab:purple', 'tab:red', 'tab:blue', 'tab:orange', 'tab:green']
markers = ['v', 's', 'D', 'o', '^']

pattern = re.compile(
    r'(?P<ds>[\w\.]+)_'
    r'(?P<method>\w+)_'
    r'(?P<what>loss|gradnorm|lrs)\.npy'
)

# Για κάθε dataset
for ds_folder in os.listdir(DATA_ROOT):
    ds_label = ds_folder[:-4] if ds_folder.endswith('.bz2') else ds_folder
    ds_dir = os.path.join(DATA_ROOT, ds_label)
    if not os.path.isdir(ds_dir):
        continue

    plot_dir = os.path.join(PLOTS_ROOT, ds_label)
    os.makedirs(plot_dir, exist_ok=True)

    # φορτώνουμε τα αρχεία
    histories = {}
    for fname in os.listdir(ds_dir):
        m = pattern.match(fname)
        if not m:
            continue
        dd = m.groupdict()
        method = dd['method']
        what = dd['what']
        if method not in methods:
            continue
        arr = np.load(os.path.join(ds_dir, fname), allow_pickle=True)
        if what == 'gradnorm':
            arr = arr.astype(float)
        histories.setdefault(method, {})[what] = arr

    if not histories:
        print(f"Warning: δεν βρέθηκαν αρχεία στο {ds_dir}")
        continue

    eps = 1e-16

    # 1) Loss (log scale, με markers)
    plt.figure(figsize=(7, 5))
    for i, method in enumerate(methods):
        if method not in histories: continue
        vals = np.maximum(histories[method]['loss'], eps)
        plt.semilogy(
            vals,
            label=method,
            color=colors[i],
            marker=markers[i],
            markevery=25,
            linewidth=2,
            markersize=9,
            alpha=0.95,
            markerfacecolor=colors[i],
            markeredgecolor='black'
        )
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('f(w)', fontsize=14)
    plt.title(f'Logistic L2 ({ds_label}): Loss (log scale)', fontsize=16)
    plt.legend(fontsize=11, loc='best', frameon=True, edgecolor='black')
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.savefig(os.path.join(plot_dir, f'{ds_label}_loss_log.png'), dpi=300)
    plt.close()

    # 2) Gradient Norm (log scale, με markers)
    plt.figure(figsize=(7, 5))
    for i, method in enumerate(methods):
        if method not in histories: continue
        g = np.maximum(histories[method]['gradnorm'], eps)
        plt.semilogy(
            g,
            label=method,
            color=colors[i],
            marker=markers[i],
            markevery=25,
            linewidth=2,
            markersize=9,
            alpha=0.95,
            markerfacecolor=colors[i],
            markeredgecolor='black'
        )
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel(r'$‖∇f(w)‖$', fontsize=14)
    plt.title(f'Logistic L2 ({ds_label}): Gradient Norm', fontsize=16)
    plt.legend(fontsize=11, loc='best', frameon=True, edgecolor='black')
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.savefig(os.path.join(plot_dir, f'{ds_label}_gradnorm.png'), dpi=300)
    plt.close()

    # 3) Learning Rate (linear scale, με markers)
    plt.figure(figsize=(7, 5))
    for i, method in enumerate(methods):
        if method not in histories: continue
        plt.plot(
            histories[method]['lrs'],
            label=method,
            color=colors[i],
            marker=markers[i],
            markevery=25,
            linewidth=2,
            markersize=9,
            alpha=0.95,
            markerfacecolor=colors[i],
            markeredgecolor='black'
        )
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.title(f'Logistic L2 ({ds_label}): Learning Rate', fontsize=16)
    plt.legend(fontsize=11, loc='best', frameon=True, edgecolor='black')
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.savefig(os.path.join(plot_dir, f'{ds_label}_lrs.png'), dpi=300)
    plt.close()
