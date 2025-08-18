# scripts/run_matrix_fac_r20_r30_silent.py
import time
import warnings
import argparse
import os
import numpy as np
import scipy.linalg as LA
import pandas as pd
from scipy.linalg import norm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Import from your existing repo (must be on PYTHONPATH or same folder)
from optimizers import Adgd, AdgdAccel, AdaPGNesterov

# =========================
# Problem: Matrix Factorization on MovieLens 100K
# =========================

# Read MovieLens 100K (u.data) from your repo's datasets/
names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('./datasets/u.data', sep='\t', names=names)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

# Build ratings matrix A
ratings = np.zeros((n_users, n_items))
for row in df.itertuples():
    ratings[row[1] - 1, row[2] - 1] = row[3]

A = ratings
m, n = A.shape

# Loss and gradient in the factorized variables X = [U; V] ∈ R^{(m+n)×r}
def f(X):
    U, V = X[:m], X[m:]
    R = U @ V.T - A
    return 0.5 * norm(R)**2

def df_func(X):
    U, V = X[:m], X[m:]
    R = U @ V.T - A
    grad_U = R @ V
    grad_V = R.T @ U
    return np.vstack([grad_U, grad_V])

# -------------------------
# Malitsky–Mishchenko-style λ0
# -------------------------
def mm_initial_stepsize(X0, grad_func, probe_scale=1e-6):
    g0 = grad_func(X0)
    gnorm = LA.norm(g0)
    if not np.isfinite(gnorm) or gnorm == 0.0:
        return 1e-3
    delta = -(probe_scale / (gnorm + 1e-18)) * g0
    X1 = X0 + delta
    g1 = grad_func(X1)
    denom = LA.norm(delta)
    L0 = LA.norm(g1 - g0) / (denom + 1e-18)
    if not np.isfinite(L0) or L0 <= 0:
        L0 = max(gnorm / (denom + 1e-18), 1e3)
    return 1.0 / L0

# -------------------------
# Adaptive NPGM (inline)
# -------------------------
class AdaptiveNPGM:
    def __init__(self, loss_func, grad_func, it_max=4000, lr0=None,
                 a_lr=0.5, a_mu=0.5, b_lr=0.5, b_mu=0.5,
                 fmin=0.0, isVerbose=False, tag='AdNPGM'):
        self.loss_func = loss_func
        self.grad_func = grad_func
        self.it_max = it_max
        self.lr0 = lr0
        self.a_lr = a_lr
        self.a_mu = a_mu
        self.b_lr = b_lr
        self.b_mu = b_mu
        self.fmin = fmin
        self.isVerbose = isVerbose
        self.tag = tag

        self.loss_hist = []
        self.grad_norm_hist = []
        self.lr_hist = []

    def run(self, X0):
        t0 = time.time()
        x = X0.copy()
        y_prev = x.copy()
        x_prev = None
        g_prev = None

        g = self.grad_func(x)
        self.lr = mm_initial_stepsize(X0, self.grad_func) if self.lr0 is None else self.lr0
        self.mu = 1e-12
        self.theta_lr = np.inf
        self.theta_mu = np.inf

        self.loss_hist.append(self.loss_func(x))
        self.grad_norm_hist.append(LA.norm(g))
        self.lr_hist.append(self.lr)

        for _ in range(self.it_max):
            g = self.grad_func(x)

            if x_prev is not None:
                dx = x - x_prev
                dg = g - g_prev
                denom = LA.norm(dx)
                Lloc = LA.norm(dg) / (denom + 1e-18)

                lr_new = min(np.sqrt(1 + self.a_lr * self.theta_lr) * self.lr,
                             self.b_lr / (Lloc + 1e-18))
                mu_new = min(np.sqrt(1 + self.a_mu * self.theta_mu) * self.mu,
                             self.b_mu * (Lloc + 0.0))

                self.theta_lr = lr_new / self.lr
                self.theta_mu = mu_new / self.mu
                self.lr = lr_new
                self.mu = mu_new

            y_next = x - self.lr * g

            inv_sqrt_lr = 1.0 / np.sqrt(max(self.lr, 1e-30))
            sqrt_mu = np.sqrt(max(self.mu, 0.0))
            beta = (inv_sqrt_lr - sqrt_mu) / (inv_sqrt_lr + sqrt_mu)
            beta = float(np.clip(beta, 0.0, 0.999999))

            x_next = y_next + beta * (y_next - y_prev)

            x_prev, g_prev = x, g
            y_prev = y_next
            x = x_next

            # logs (kept in memory only)
            self.loss_hist.append(self.loss_func(x))
            self.grad_norm_hist.append(LA.norm(self.grad_func(x)))
            self.lr_hist.append(self.lr)

        self.runtime = time.time() - t0

# =========================
# Saving helpers
# =========================
def save_histories(out_dir, tag, opt):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'mf_{tag}_loss.npy'),     np.array(opt.loss_hist))
    np.save(os.path.join(out_dir, f'mf_{tag}_gradnorm.npy'), np.array(opt.grad_norm_hist, dtype=float))
    np.save(os.path.join(out_dir, f'mf_{tag}_lrs.npy'),      np.array(opt.lr_hist))

# =========================
# Run for r ∈ {20, 30} with the requested iteration budgets
# =========================
r_values = (20, 30)
fmin = 0.0

# iteration budget per rank
iter_for_rank = {
    20: 30000,
    30: 100000,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="results/matrix_fac/data",
                        help="Base directory to save histories for plotting later.")
    args = parser.parse_args()

    for r in r_values:
        # reproducibility per rank
        np.random.seed(1 + r)
        X0 = np.random.randn(m + n, r)

        # MM-style initial stepsize per rank
        lambda0 = mm_initial_stepsize(X0, df_func)

        # iters for this rank
        it_max = iter_for_rank[r]

        # instantiate optimizers (silent)
        adgd = Adgd(eps=0.0, lr0=lambda0, prox_type="none",
                    fmin=fmin, reg_param=1, isVerbose=False,
                    loss_func=f, grad_func=df_func, it_max=it_max)

        adacc = AdgdAccel(fmin=fmin, reg_param=1, isVerbose=False,
                          loss_func=f, grad_func=df_func, it_max=it_max)

        adgdNesCons = AdaPGNesterov(lr0=lambda0, prox_type="none", fmin=fmin, reg_param=1,
                                    isConservative=True, isVerbose=False,
                                    loss_func=f, grad_func=df_func, it_max=it_max)

        adgdNes = AdaPGNesterov(lr0=lambda0, prox_type="none", fmin=fmin, reg_param=1,
                                isConservative=False, isVerbose=False,
                                loss_func=f, grad_func=df_func, it_max=it_max)

        adnpgm = AdaptiveNPGM(loss_func=f, grad_func=df_func, it_max=it_max,
                              lr0=lambda0, a_lr=0.5, a_mu=0.5, b_lr=0.5, b_mu=0.5,
                              fmin=fmin, isVerbose=False, tag='AdNPGM')

        # --- Run all (no prints) ---
        adgd.run(X0)
        adacc.run(X0)
        adgdNesCons.run(X0)
        adgdNes.run(X0)
        adnpgm.run(X0)

        # --- Save histories for plotting later ---
        out_dir = os.path.join(args.save_dir, f"r{r}")
        save_histories(out_dir, 'Adgd',        adgd)
        save_histories(out_dir, 'AdgdAccel',   adacc)
        save_histories(out_dir, 'AdgdNesCons', adgdNesCons)
        save_histories(out_dir, 'AdgdNes',     adgdNes)
        save_histories(out_dir, 'AdNPGM',      adnpgm)

if __name__ == "__main__":
    main()
