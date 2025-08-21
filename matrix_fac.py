import time
import numpy as np
import scipy.linalg as LA
import os
import pandas as pd
from scipy.linalg import norm

# Import optimizers
from optimizers import *

# >>> ΝΕΟ: φέρνουμε τη δική μας συνάρτηση από algorithms.py
from algorithms import adaptive_npgm_simple

class IdentityPrec:
    def prec(self, g):
        return g

# >>> ΝΕΟ: απλός logger για να ταιριάζει με το hist.add(grad_norm, func_val, i)
class SimpleHist:
    def __init__(self):
        self.grad_norm_hist = []
        self.loss_hist = []
    def add(self, grad_norm, func_val, i):
        self.grad_norm_hist.append(grad_norm)
        self.loss_hist.append(func_val)

# Setup problem
r = 10

# Read data
names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('./datasets/u.data', sep='\t', names=names)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]

# Create r_{ui}, our ratings matrix
ratings = np.zeros((n_users, n_items))
for row in df.itertuples():
    ratings[row[1] - 1, row[2] - 1] = row[3]

A = ratings
m, n = A.shape

# Function and gradient
def f(X):
    U, V = X[:m], X[m:]
    return 0.5 * norm(U @ V.T - A) ** 2

def df(X):
    U, V = X[:m], X[m:]
    res = U @ V.T - A
    grad_U = res @ V
    grad_V = res.T @ U
    return np.vstack([grad_U, grad_V])

np.random.seed(1)
X0 = np.random.randn(m+n, r)
# Nonconvex problem!
fmin = 0
it_max = 5000

# --- Optimizers από το πακέτο σου ---
adgd = Adgd(eps=0.0, lr0=1e-03, prox_type="none", fmin=fmin, reg_param=1, isVerbose=True, loss_func=f, grad_func=df, it_max=it_max)
adacc = AdgdAccel(fmin=fmin, reg_param=1, isVerbose=True, loss_func=f, grad_func=df, it_max=it_max)
adgdNesCons = AdaPGNesterov(lr0=1e-03, prox_type="none", fmin=fmin, reg_param=1, isConservative=True, isVerbose=True, loss_func=f, grad_func=df, it_max=it_max)
adgdNes = AdaPGNesterov(lr0=1e-03, prox_type="none", fmin=fmin, reg_param=1, isConservative=False, isVerbose=True, loss_func=f, grad_func=df, it_max=it_max)

# --- ΑΦΑΙΡΕΘΗΚΕ ο απλός Nesterov ---
# nes = Nesterov(...)

os.makedirs('results/matrix_fac/data', exist_ok=True)

print("## AdGD ##")
adgd.run(X0)
np.save('results/matrix_fac/data/mf_Adgd_loss.npy',     adgd.loss_hist)
np.save('results/matrix_fac/data/mf_Adgd_gradnorm.npy', adgd.grad_norm_hist)
np.save('results/matrix_fac/data/mf_Adgd_lrs.npy',      adgd.lr_hist)

print("## Accelerated AdGD ##")
adacc.run(X0)
np.save('results/matrix_fac/data/mf_AdgdAccel_loss.npy',     adacc.loss_hist)
np.save('results/matrix_fac/data/mf_AdgdAccel_gradnorm.npy', adacc.grad_norm_hist)
np.save('results/matrix_fac/data/mf_AdgdAccel_lrs.npy',      adacc.lr_hist)

print("## AdGD + Nesterov (c.) ##")
adgdNesCons.run(X0)
np.save('results/matrix_fac/data/mf_AdgdNesCons_loss.npy',     adgdNesCons.loss_hist)
np.save('results/matrix_fac/data/mf_AdgdNesCons_gradnorm.npy', adgdNesCons.grad_norm_hist)
np.save('results/matrix_fac/data/mf_AdgdNesCons_lrs.npy',      adgdNesCons.lr_hist)

print("## AdGD + Nesterov ##")
adgdNes.run(X0)
np.save('results/matrix_fac/data/mf_AdgdNes_loss.npy',     adgdNes.loss_hist)
np.save('results/matrix_fac/data/mf_AdgdNes_gradnorm.npy', adgdNes.grad_norm_hist)
np.save('results/matrix_fac/data/mf_AdgdNes_lrs.npy',      adgdNes.lr_hist)

# >>> ΝΕΟ: Adaptive NPGM (simple loop από algorithms.py)
print("## Adaptive NPGM (simple) ##")
prec = IdentityPrec()         # <<— ταυτότητα, σύμφωνη με το algorithms.py
hist = SimpleHist()
X_final = adaptive_npgm_simple(
    f=f, df=df, prec=prec,
    x0=X0, la_0=1e-3, numb_iter=it_max,
    hist=hist, verbose=True
)
# αποθήκευση όπως στα άλλα (δεν έχουμε lrs μέσα στο loop, οπότε το παραλείπουμε ή γράφουμε NaN)
np.save('results/matrix_fac/data/mf_AdNPGM_simple_loss.npy',     np.array(hist.loss_hist))
np.save('results/matrix_fac/data/mf_AdNPGM_simple_gradnorm.npy', np.array(hist.grad_norm_hist))
# προαιρετικά, αν θες ομοιομορφία αρχείων:
# np.save('results/matrix_fac/data/mf_AdNPGM_simple_lrs.npy', np.full(len(hist.loss_hist), np.nan))
