import os
import numpy as np
import matplotlib.pyplot as plt

# Create folder for the plots
os.makedirs('results/matrix_fac/plots', exist_ok=True)

# 1) Descriptive names of the methods and corresponding file tags
methods = {
    'AdGD':          'Adgd',
    'Accelerated AdGD':      'AdgdAccel',
    'AdGD + Nesterov (c.)':  'AdgdNesCons',
    'AdGD + Nesterov':       'AdgdNes',
}

# 2) Load history arrays from results/matrix_fac/data
hist = {}
for label, tag in methods.items():
    hist[label] = {
        'loss':     np.load(f'results/matrix_fac/data/mf_{tag}_loss.npy'),
        'gradnorm': np.array(
                          np.load(f'results/matrix_fac/data/mf_{tag}_gradnorm.npy',
                                  allow_pickle=True),
                          dtype=float
                      ),
        'lr':       np.load(f'results/matrix_fac/data/mf_{tag}_lrs.npy')
    }

# 3) Compute f* as the minimum achieved loss across all methods
f_star = min(np.min(hist[label]['loss']) for label in methods)
eps    = 1e-16  # small offset to avoid zero values in log scale

# 4) Plot residual (f(X) - f*) on a log scale
plt.figure(figsize=(6,4))
for label in methods:
    residual = hist[label]['loss'] - f_star
    residual = np.maximum(residual, eps)
    plt.semilogy(residual, label=label, alpha=0.8)
plt.xlabel('Iteration')
plt.ylabel('f(X) − f*')
plt.title('Matrix Factorization: Residual (log scale)')
plt.legend(fontsize=10, loc='best', frameon=True, edgecolor='black')
plt.tight_layout()
plt.savefig('results/matrix_fac/plots/mf_plot_residual.png', dpi=300)
plt.show()



# 5) Plot raw function value on a log scale
plt.figure(figsize=(6,4))
for label in methods:
    loss_vals = np.maximum(hist[label]['loss'], eps)
    plt.semilogy(loss_vals, label=label, alpha=0.8)
plt.xlabel('Iteration')
plt.ylabel('f(X)')
plt.title('Matrix Factorization: Function Value (log scale)')
plt.legend(fontsize=10, loc='best', frameon=True, edgecolor='black')
plt.tight_layout()
plt.savefig('results/matrix_fac/plots/mf_plot_loss_log.png', dpi=300)
plt.show()

# 6) Plot gradient norm on a log scale
plt.figure(figsize=(6,4))
for label in methods:
    grad_vals = np.maximum(hist[label]['gradnorm'], eps)
    plt.semilogy(grad_vals, label=label, alpha=0.8)
plt.xlabel('Iteration')
plt.ylabel('‖∇f(X)‖')
plt.title('Matrix Factorization: Gradient Norm (log scale)')
plt.legend(fontsize=10, loc='best', frameon=True, edgecolor='black')
plt.tight_layout()
plt.savefig('results/matrix_fac/plots/mf_plot_gradnorm.png', dpi=300)
plt.show()

# 7) Plot learning rate schedule on a linear scale with markers
plt.figure(figsize=(6,4))
for label in methods:
    plt.plot(hist[label]['lr'], label=label, marker='.', markevery=100, alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Step size (lr)')
plt.title('Matrix Factorization: Learning Rate Schedule')
plt.legend(fontsize=10, loc='best', frameon=True, edgecolor='black')
plt.tight_layout()
plt.savefig('results/matrix_fac/plots/mf_plot_lr.png', dpi=300)
plt.show()
