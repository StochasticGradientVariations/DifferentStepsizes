import os
import numpy as np
from numpy.linalg import norm
from sklearn.datasets import load_svmlight_file
from scipy.optimize import minimize

def logsig(x):
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out

def logistic_loss(w, X, y, l2):
    z = np.dot(X, w)
    y = np.asarray(y)
    return np.mean((1-y)*z - logsig(z)) + l2/2 * norm(w)**2

def logistic_grad(w, X, y, l2):
    z = np.dot(X, w)
    sig = 1/(1 + np.exp(-z))
    grad = np.dot(X.T, (sig - y)) / X.shape[0] + l2 * w
    return grad

DATASETS = ['mushrooms', 'covtype.bz2', 'w8a']
DATASET_DIR = 'datasets'
optimal_losses = {}

for ds in DATASETS:
    ds_name = ds.replace('.bz2', '')
    print(f"\n=== Υπολογισμός f* για το dataset: {ds} ===")

    path = os.path.join(DATASET_DIR, ds)
    X_sp, y = load_svmlight_file(path)
    X, y = X_sp.toarray(), y

    if set(np.unique(y)) == {1, 2}:
        y = (y == 2).astype(float)

    n, d = X.shape
    L = 0.25 * np.max(np.linalg.eigvalsh((X.T @ X) / n))
    l2 = L / n
    w0 = np.zeros(d)

    result = minimize(
        logistic_loss, w0, args=(X, y, l2),
        method='L-BFGS-B', jac=logistic_grad,
        options={'maxiter': 2000, 'ftol': 1e-8, 'disp': True}
    )
    f_star = result.fun
    optimal_losses[ds_name] = f_star
    print(f"f* για {ds_name}: {f_star:.8f}")

print("\nΤιμές f* για όλα τα datasets:")
for name, val in optimal_losses.items():
    print(f"{name}: {val:.8f}")
