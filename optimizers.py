import numpy as np

import numpy.linalg as la

from trainer import Trainer

def eval_prox(type, x, step_size, reg_param):
    if type == "l1":
        return np.maximum(0., np.abs(x) - reg_param*step_size) * np.sign(x)
    else:
        return x


class Gd(Trainer):
    """
    Gradient descent with constant learning rate.
    
    Arguments:
        lr (float): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr, *args, **kwargs):
        super(Gd, self).__init__(*args, **kwargs)
        self.lr = lr
        
    def step(self):
        return self.w - self.lr * self.grad
    
    def init_run(self, *args, **kwargs):
        super(Gd, self).init_run(*args, **kwargs)
    
    
class Nesterov(Trainer):
    """
    Nesterov's accelerated gradient descent with constant learning rate.
    
    Arguments:
        lr (float): an estimate of the inverse smoothness constant
        strongly_convex (boolean, optional): if true, uses the variant
            for strongly convex functions, which requires mu>0 (default: False)
    """
    def __init__(self, lr, strongly_convex=False, mu=0, *args, **kwargs):
        super(Nesterov, self).__init__(*args, **kwargs)
        self.lr = lr
        if mu < 0:
            raise ValueError("Invalid mu: {}".format(mu))
        if strongly_convex and mu == 0:
            raise ValueError("""Mu must be larger than 0 for strongly_convex=True,
                             invalid value: {}""".format(mu))
        if strongly_convex:
            self.mu = mu
            kappa = (1/self.lr)/self.mu
            self.momentum = (np.sqrt(kappa)-1) / (np.sqrt(kappa)+1)
        self.strongly_convex = strongly_convex
        
    def step(self):
        if not self.strongly_convex:
            alpha_new = 0.5 * (1 + np.sqrt(1 + 4 * self.alpha ** 2))
            self.momentum = (self.alpha - 1) / alpha_new
            self.alpha = alpha_new
        self.w_nesterov_old = self.w_nesterov.copy()
        self.w_nesterov = self.w - self.lr * self.grad
        return self.w_nesterov + self.momentum * (self.w_nesterov - self.w_nesterov_old)
    
    def init_run(self, *args, **kwargs):
        super(Nesterov, self).init_run(*args, **kwargs)
        self.w_nesterov = self.w.copy()
        self.alpha = 1.
    
    
class Adgd(Trainer):
    """
    Adaptive gradient descent based on the local smoothness constant
    
    Arguments:
        eps (float, optional): an estimate of 1 / L^2, where L is the global smoothness constant (default: 0)
    """
    def __init__(self, eps=0.0, lr0=None, *args, **kwargs):
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {}".format(eps))
        super(Adgd, self).__init__(*args, **kwargs)
        self.eps = eps
        self.lr0 = lr0
        
    def estimate_stepsize(self):
        L = la.norm(self.grad - self.grad_old) / la.norm(self.w - self.w_old)
        if np.isinf(self.theta):
            lr_new = 0.5 / L
        else:
            lr_new = min(np.sqrt(1 + self.theta) * self.lr, self.eps / self.lr + 0.5 / L)
        self.theta = lr_new / self.lr
        self.lr = lr_new
        
    def step(self):
        self.w_old = self.w.copy()
        self.grad_old = self.grad.copy()
        return eval_prox(self.prox_type, self.w - self.lr * self.grad, self.lr, self.reg_param)
        
        
    def init_run(self, *args, **kwargs):
        super(Adgd, self).init_run(*args, **kwargs)
        self.theta = np.inf
        grad = self.grad_func(self.w)
        if self.lr0 is None:
            self.lr0 = 1e-10
        self.lr = self.lr0
        self.lrs = [self.lr]
        self.w_old = self.w.copy()
        self.grad_old = grad
        # self.w -= self.lr * grad
        self.w = eval_prox(self.prox_type, self.w - self.lr * grad, self.lr, self.reg_param)
        self.save_checkpoint()
        
    def update_logs(self):
        super(Adgd, self).update_logs()
        self.lrs.append(self.lr)
        
        
class AdgdAccel(Trainer):
    """
    Adaptive gradient descent with heuristic Nesterov's acceleration
    Targeted at locally strongly convex functions, so by default uses
    estimation with min(sqrt(1 + theta_{k-1} / 2) * la_{k-1}, 0.5 / L_k)
    
    Arguments:
        a_lr (float, optional): increase parameter for learning rate (default: 0.5)
        a_mu (float, optional): increase parameter for strong convexity (default: 0.5)
        b_lr (float, optional): local smoothness scaling (default: 0.5)
        b_mu (float, optional): local strong convexity scaling (default: 0.5)
    """
    def __init__(self, a_lr=0.5, a_mu=0.5, b_lr=0.5, b_mu=0.5, isVerbose=False, *args, **kwargs):
        super(AdgdAccel, self).__init__(*args, **kwargs)
        self.a_lr = a_lr
        self.a_mu = a_mu
        self.b_lr = b_lr
        self.b_mu = b_mu
        self.isVerbose = isVerbose
        
    def estimate_stepsize(self):
        denom = la.norm(self.w - self.w_old)
        L = la.norm(self.grad - self.grad_old) / (denom + 1e-12)
        lr_new = min(np.sqrt(1 + self.a_lr * self.theta_lr) * self.lr, self.b_lr / L)
        self.theta_lr = lr_new / self.lr
        self.lr = lr_new
        mu_new = min(np.sqrt(1 + self.a_mu * self.theta_mu) * self.mu,self.b_mu * L)
        self.theta_mu = mu_new / self.mu
        self.mu = mu_new
        
    def step(self):
        self.w_old = self.w.copy()
        self.grad_old = self.grad.copy()
        momentum = (np.sqrt(1 / self.lr) - np.sqrt(self.mu)) / (np.sqrt(1 / self.lr) + np.sqrt(self.mu))
        self.w_nesterov_old = self.w_nesterov.copy()
        self.w_nesterov = self.w - self.lr * self.grad
        return self.w_nesterov + momentum * (self.w_nesterov - self.w_nesterov_old)
        
    def init_run(self, *args, **kwargs):
        super(AdgdAccel, self).init_run(*args, **kwargs)
        self.theta_lr = np.inf
        self.theta_mu = np.inf
        grad = self.grad_func(self.w)
        # The first estimate is normalized gradient with a small coefficient
        self.lr = 1e-5 / la.norm(grad)
        self.lrs = [self.lr]
        self.mu = 1 / self.lr
        self.w_old = self.w.copy()
        self.w_nesterov = self.w.copy()
        self.grad_old = grad
        self.w -= self.lr * grad
        self.save_checkpoint()
        
    def update_logs(self):
        super(AdgdAccel, self).update_logs()
        self.lrs.append(self.lr)
        
        
class Armijo(Trainer):
    """
    Adaptive gradient descent based on the local smoothness constant
    
    Arguments:
        eps (float): an estimate of 1 / L^2, where L is the global smoothness constant
    """
    def __init__(self, backtracking=0.5, armijo_const=0.5, lr0=None, *args, **kwargs):
        if lr0 < 0:
            raise ValueError("Invalid lr0: {}".format(lr0))
        super(Armijo, self).__init__(*args, **kwargs)
        self.lr = lr0
        self.backtracking = backtracking
        self.armijo_const = armijo_const
        
    def estimate_stepsize(self):
        f = self.loss_func(self.w)
        lr = self.lr / self.backtracking
        w_new = self.w - lr * self.grad
        f_new = self.loss_func(w_new)
        armijo_condition = f_new <= f - self.lr * self.armijo_const * la.norm(self.grad)**2
        while not armijo_condition:
            lr *= self.backtracking
            w_new = self.w - lr * self.grad
            f_new = self.loss_func(w_new)
            armijo_condition = f_new <= f - lr * self.armijo_const * la.norm(self.grad)**2
            self.it += 1
            
        self.lr = lr
        
    def step(self):
        self.grad = self.grad_func(self.w)
        self.estimate_stepsize()
        return self.w - self.lr * self.grad
        
    def init_run(self, *args, **kwargs):
        super(Armijo, self).init_run(*args, **kwargs)
        self.w_ave = self.w.copy()
        self.ws_ave = [self.w_ave.copy()]
        self.lr_sum = 0
        self.lrs = []
        
    def update_logs(self):
        super(Armijo, self).update_logs()
        self.lrs.append(self.lr)
        self.ws_ave.append(self.w_ave.copy())
        


class AdaPGNesterov(Trainer):
    """
    Adaptive GD with τ₁=(k+1)/k * lr_prev and τ₂=0.5*||Δx||/||Δg||,
    plus Nesterov‐type momentum.
    """
    def __init__(self, lr0=None, isConservative=False, *args, **kwargs):
        super(AdaPGNesterov, self).__init__(*args, **kwargs)
        self.lr0 = lr0
        self.isConservative = isConservative

    def init_run(self, w0):
        # First initialization
        super(AdaPGNesterov, self).init_run(w0)
        # Set theta to infinity to avoid first computation
        self.theta = np.inf

        # First step
        if self.lr0 is None:
            self.lr0 = 1e-10
        self.lr = self.lr0

        # Main variable

        # Nesterov momentum variable
        self.y = self.w.copy()
        self.y_old = self.w.copy()

        # Compute gradient
        grad = self.grad_func(self.w)
        self.grad = grad

        # Αρχικές τιμές για curvature
        self.w_old    = self.w.copy()
        self.grad_old = grad

        # Initial step
        # self.w -= self.lr * grad
        self.w = eval_prox(self.prox_type, self.w - self.lr * self.grad, self.lr, self.reg_param)
        self.save_checkpoint()

    def step(self):
        # Compute new extrapolation point and store the previous one
        self.y_old = self.y.copy()
        self.y = self.w + self.it / (self.it + 3) * (self.w - self.w_old)
        # Compute gradient at extrapolated point and store the previous one
        self.grad_old = self.grad_func(self.y_old)
        self.grad = self.grad_func(self.y)

        # Compute stepsize
        k = max(1, self.it)
        if self.isConservative:
            tau1 = (k+1)/k * self.lr
        else:
            tau1 = np.sqrt(1 + self.theta) * self.lr
        denom = la.norm(self.y - self.y_old) + 1e-12
        Lloc = la.norm(self.grad - self.grad_old) / denom
        tau2 = 0.5 / Lloc
        lr_new = min(tau1, tau2)
        self.theta = lr_new / self.lr
        self.lr = lr_new

        # Perform gradient step
        self.w_old = self.w.copy()
        self.w = eval_prox(self.prox_type, self.y - self.lr * self.grad, self.lr, self.reg_param)
        # self.w = self.y - self.lr * self.grad
        return self.w
