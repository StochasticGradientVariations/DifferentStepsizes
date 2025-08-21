import time
import numpy as np
import scipy.linalg as LA
import os
import pandas as pd
from scipy.linalg import norm

# --- Import from your existing repo (no changes there)
from optimizers import Adgd, AdgdAccel, AdaPGNesterov

# =========================
# Problem: Matrix Factorization
# =========================
r = 30

# Read MovieLens 100K (u.data)
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
                 fmin=0.0, isVerbose=True, tag='AdNPGM'):
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

        for k in range(self.it_max):
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

            # logs
            self.loss_hist.append(self.loss_func(x))
            self.grad_norm_hist.append(LA.norm(self.grad_func(x)))
            self.lr_hist.append(self.lr)

            if self.isVerbose and (k % 100 == 0 or k == self.it_max - 1):
                print(f"[{self.tag}] it={k:5d}  f={self.loss_hist[-1]:.6e}  "
                      f"||g||={self.grad_norm_hist[-1]:.3e}  lr={self.lr:.3e}  mu={self.mu:.3e}")

        self.runtime = time.time() - t0

# =========================
# Run all methods with the SAME λ0
# =========================
np.random.seed(1)
X0 = np.random.randn(m + n, r)
fmin = 0.0
it_max = 55000  # <-- all run for 4000 iterations

lambda0 = mm_initial_stepsize(X0, df_func)
print(f"[Init] Common MM-style λ0 = {lambda0:.3e}")

adgd = Adgd(eps=0.0, lr0=lambda0, prox_type="none",
            fmin=fmin, reg_param=1, isVerbose=True,
            loss_func=f, grad_func=df_func, it_max=it_max)

adacc = AdgdAccel(fmin=fmin, reg_param=1, isVerbose=True,
                  loss_func=f, grad_func=df_func, it_max=it_max)

adgdNesCons = AdaPGNesterov(lr0=lambda0, prox_type="none", fmin=fmin, reg_param=1,
                             isConservative=True, isVerbose=True,
                             loss_func=f, grad_func=df_func, it_max=it_max)

adgdNes = AdaPGNesterov(lr0=lambda0, prox_type="none", fmin=fmin, reg_param=1,
                         isConservative=False, isVerbose=True,
                         loss_func=f, grad_func=df_func, it_max=it_max)

adnpgm = AdaptiveNPGM(loss_func=f, grad_func=df_func, it_max=it_max,
                      lr0=lambda0, a_lr=0.5, a_mu=0.5, b_lr=0.5, b_mu=0.5,
                      fmin=fmin, isVerbose=True, tag='AdNPGM')

os.makedirs('results/matrix_fac/data', exist_ok=True)

print("## AdGD ##")
adgd.run(X0)
np.save('results/matrix_fac/data/mf_Adgd_loss.npy',     np.array(adgd.loss_hist))
np.save('results/matrix_fac/data/mf_Adgd_gradnorm.npy', np.array(adgd.grad_norm_hist))
np.save('results/matrix_fac/data/mf_Adgd_lrs.npy',      np.array(adgd.lr_hist))

print("## Accelerated AdGD ##")
adacc.run(X0)
np.save('results/matrix_fac/data/mf_AdgdAccel_loss.npy',     np.array(adacc.loss_hist))
np.save('results/matrix_fac/data/mf_AdgdAccel_gradnorm.npy', np.array(adacc.grad_norm_hist))
np.save('results/matrix_fac/data/mf_AdgdAccel_lrs.npy',      np.array(adacc.lr_hist))

print("## AdGD + Nesterov (conservative) ##")
adgdNesCons.run(X0)
np.save('results/matrix_fac/data/mf_AdgdNesCons_loss.npy',     np.array(adgdNesCons.loss_hist))
np.save('results/matrix_fac/data/mf_AdgdNesCons_gradnorm.npy', np.array(adgdNesCons.grad_norm_hist))
np.save('results/matrix_fac/data/mf_AdgdNesCons_lrs.npy',      np.array(adgdNesCons.lr_hist))

print("## AdGD + Nesterov (non-conservative) ##")
adgdNes.run(X0)
np.save('results/matrix_fac/data/mf_AdgdNes_loss.npy',     np.array(adgdNes.loss_hist))
np.save('results/matrix_fac/data/mf_AdgdNes_gradnorm.npy', np.array(adgdNes.grad_norm_hist))
np.save('results/matrix_fac/data/mf_AdgdNes_lrs.npy',      np.array(adgdNes.lr_hist))

print("## Adaptive NPGM ##")
adnpgm.run(X0)
np.save('results/matrix_fac/data/mf_AdNPGM_loss.npy',     np.array(adnpgm.loss_hist))
np.save('results/matrix_fac/data/mf_AdNPGM_gradnorm.npy', np.array(adnpgm.grad_norm_hist))
np.save('results/matrix_fac/data/mf_AdNPGM_lrs.npy',      np.array(adnpgm.lr_hist))

print("Done.")
