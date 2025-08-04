import os
import numpy as np
import numpy.linalg as la
from optimizers import *
from scipy.sparse import coo_matrix

# λίστα με (m,n,k) που θέλεις να δοκιμάσεις
configs = [
    (100, 300, 30),
    (200, 500, 60),
    (500, 1000, 100),
    (500, 1000, 200),
]

# σταθερές
power  = 2
norm   = 2
lamb   = 1
it_max = 2000

for m, n, k in configs:
    assert k < n and n > m

    # reproducibility
    np.random.seed(0)

    # κατασκευή του προβλήματος
    B = np.random.uniform(-1., 1., [m, n])
    v = np.random.uniform(0, 1., m)
    yopt = v / np.linalg.norm(v)

    p    = B.T @ yopt
    perm = np.argsort(np.abs(p))[::-1]

    alpha = np.zeros(n)
    xi    = np.random.uniform(0, 1, n)
    for i in range(n):
        if i < k:
            alpha[perm[i]] = lamb / np.abs(p[perm[i]])
        elif np.abs(p[perm[i]]) < 0.1 * lamb:
            alpha[perm[i]] = lamb
        else:
            alpha[perm[i]] = lamb * xi[perm[i]] / np.abs(p[perm[i]])

    A = B @ np.diag(alpha)
    xi = np.random.uniform(0, 1 / np.sqrt(k), n)
    xopt = np.zeros(n)
    q    = A.T @ yopt
    for i in range(n):
        if i < k:
            xopt[perm[i]] = xi[perm[i]] * np.sign(q[perm[i]])

    conj_power = power / (power - 1)
    b = np.sign(yopt) * np.abs(yopt)**(conj_power - 1) + A @ xopt
    fmin = np.sum(np.abs(yopt)**conj_power)/conj_power + lamb * np.sum(np.abs(xopt))

    # loss & grad functions
    def loss_fn(w):
        r = A.dot(w) - b
        return np.sum(np.abs(r)**power)/power + lamb * np.sum(np.abs(w))

    def grad_fn(w):
        if norm == power:
            return np.sign(A.T @ (A @ w - b)) * np.abs(A.T @ (A @ w - b))**(power-1)
        else:
            return (np.linalg.norm(A.T @ (A @ w - b) - b)**(power-2)) * (A.T @ (A @ w - b) - b)

    # Lipschitz για Nesterov
    L = la.norm(A.T @ A, 2)

    # ορισμός optimizers
    w0 = np.random.rand(n)
    nes = Nesterov(lr=1/L, strongly_convex=False, mu=0,
                   loss_func=loss_fn, grad_func=grad_fn,
                   it_max=it_max, isVerbose=False,
                   prox_type="l1", reg_param=1)

    adgd = Adgd(eps=0.0, lr0=1e-6, prox_type="l1", fmin=fmin, reg_param=1,
                loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)

    adacc = AdgdAccel(fmin=fmin, reg_param=1,
                      loss_func=loss_fn, grad_func=grad_fn, it_max=it_max)

    adgdNesCons = AdaPGNesterov(lr0=1e-6, prox_type="l1", fmin=fmin, reg_param=1,
                                isConservative=True,
                                loss_func=loss_fn, grad_func=grad_fn,
                                it_max=it_max)

    adgdNes = AdaPGNesterov(lr0=1e-6, prox_type="l1", fmin=fmin, reg_param=1,
                            isConservative=False,
                            loss_func=loss_fn, grad_func=grad_fn,
                            it_max=it_max)

    # φτιάχνουμε ξεχωριστό φάκελο για κάθε (m,n,k)
    data_dir = f"results/lasso/data_{m}_{n}_{k}"
    os.makedirs(data_dir, exist_ok=True)

    tag = f"m{m}_n{n}_k{k}"
    for name, opt in [
        ('Nes', nes),
        ('Adgd', adgd),
        ('AdgdAccel', adacc),
        ('AdgdNesCons', adgdNesCons),
        ('AdgdNes', adgdNes),
    ]:
        print(f"Running {name} on config {tag} …")
        opt.run(w0)
        # σώζουμε μέσα στον αντίστοιχο φάκελο
        np.save(f"{data_dir}/lasso_{tag}_{name}_loss.npy",     opt.loss_hist)
        np.save(f"{data_dir}/lasso_{tag}_{name}_gradnorm.npy", opt.grad_norm_hist)
        np.save(f"{data_dir}/lasso_{tag}_{name}_lrs.npy",      opt.lr_hist)
