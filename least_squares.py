#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import numpy.linalg as la

# Import optimizers
from optimizers import *


def generate_simple_quadratic(m, n):
    """
    Δημιουργεί πρόβλημα least-squares:
      min_w 0.5 * ||A w - b||^2
    με b = A x_star και ||x_star|| = 1.
    """
    x_star = np.random.rand(n)
    x_star = x_star / np.linalg.norm(x_star)
    A = np.random.rand(m, n)
    b = A @ x_star
    return A, b


def main():
    # ------------------------
    # Ρυθμίσεις πειράματος
    # ------------------------
    n = 4000
    m = 1000
    it_max = 2000

    results_dir = 'results/least_squares/data'
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------
    # Πρόβλημα
    # ------------------------
    A, b = generate_simple_quadratic(m, n)
    At = A.T

    # Loss & gradient για 0.5 * ||A w - b||^2
    def loss_fn(w):
        r = A.dot(w) - b
        return 0.5 * np.linalg.norm(r) ** 2

    def grad_fn(w):
        return At.dot(A.dot(w) - b)

    # Υπολογισμός Lipschitz (για Nesterov αν χρειαστεί)
    G = A.T @ A
    L = np.max(np.linalg.eigvalsh(G))

    # ------------------------
    # Υπολογισμός και αποθήκευση "πραγματικού" f*
    # (λύση least-squares μέσω lstsq)
    # ------------------------
    w_star, *_ = np.linalg.lstsq(A, b, rcond=None)
    f_star_true = 0.5 * np.linalg.norm(A.dot(w_star) - b) ** 2
    np.save(f'{results_dir}/ls_fstar.npy', np.array(f_star_true, dtype=float))
    print(f"[info] Saved f* = {f_star_true:.3e}")

    # ------------------------
    # Αρχικές συνθήκες
    # ------------------------
    w0 = np.zeros(n)

    adgd = Adgd(
        eps=0.0, lr0=1e-6, isVerbose=True,
        loss_func=loss_fn, grad_func=grad_fn, it_max=it_max
    )
    adacc = AdgdAccel(
        isVerbose=True,
        loss_func=loss_fn, grad_func=grad_fn, it_max=it_max
    )
    adgdNesCons = AdaPGNesterov(
        lr0=1e-6, isConservative=True, isVerbose=True,
        loss_func=loss_fn, grad_func=grad_fn, it_max=it_max
    )
    adgdNes = AdaPGNesterov(
        lr0=1e-6, isConservative=False, isVerbose=True,
        loss_func=loss_fn, grad_func=grad_fn, it_max=it_max
    )
    # nes = Nesterov(
    #     lr=1/L, strongly_convex=False, mu=0, isVerbose=True,
    #     loss_func=loss_fn, grad_func=grad_fn, it_max=it_max
    # )


    init_loss = loss_fn(w0)
    init_grad = np.linalg.norm(grad_fn(w0))
    init_lr = 1e-6

    for opt in [adgd, adacc, adgdNesCons, adgdNes]:
        opt.loss_hist = [init_loss]
        opt.grad_norm_hist = [init_grad]
        opt.lr_hist = [init_lr]

    # ------------------------
    # Helper γιαrun+save
    # Αποθηκεύει και meta[it_max, stride] ώστε τα plots να φτιάχνουν σωστό x axis
    # ------------------------
    def run_and_save(tag, opt):
        print(f"## {tag} ##")
        opt.run(w0)
        loss = np.asarray(opt.loss_hist)
        grad = np.asarray(opt.grad_norm_hist, dtype=float)
        lrs  = np.asarray(opt.lr_hist)

        np.save(f'{results_dir}/ls_{tag}_loss.npy', loss)
        np.save(f'{results_dir}/ls_{tag}_gradnorm.npy', grad)
        np.save(f'{results_dir}/ls_{tag}_lrs.npy', lrs)

        # stride = it_max / (len(loss)-1) (σε περίπτωση αραιού logging)
        if len(loss) > 1:
            stride = float(it_max) / float(len(loss) - 1)
        else:
            stride = float('nan')
        np.save(f'{results_dir}/ls_{tag}_meta.npy', np.array([float(it_max), stride], dtype=float))
        print(f"[saved] {tag}: points={len(loss)}, it_max={it_max}, stride={stride:.6f}")


    run_and_save('Adgd', adgd)
    run_and_save('AdgdAccel', adacc)
    run_and_save('AdgdNesCons', adgdNesCons)
    run_and_save('AdgdNes', adgdNes)

    # Αν θες και Nesterov:
    # run_and_save('Nes', nes)


if __name__ == '__main__':
    main()
