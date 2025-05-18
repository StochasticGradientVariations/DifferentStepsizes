import numpy as np

import numpy.linalg as la

from trainer import Trainer


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
        return self.w - self.lr * self.grad
        
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
        self.w -= self.lr * grad
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
    def __init__(self, a_lr=0.5, a_mu=0.5, b_lr=0.5, b_mu=0.5, *args, **kwargs):
        super(AdgdAccel, self).__init__(*args, **kwargs)
        self.a_lr = a_lr
        self.a_mu = a_mu
        self.b_lr = b_lr
        self.b_mu = b_mu
        
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

        
class Adagrad(Trainer):
    """
    Implement Adagrad from Duchi et. al, 2011
    "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    
    Arguments:
        primal_dual (boolean, optional): if true, uses the dual averaging method of Nesterov, 
            otherwise uses gradient descent update (default: False)
        eta (float, optional): learning rate scaling, but needs to be tuned to
            get better performance (default: 1)
        delta (float, optional): another learning rate parameter, slows down performance if
            chosen too large, otherwise requires tuning (default: 0)
    """
    def __init__(self, primal_dual=False, eta=1, delta=0, *args, **kwargs):
        super(Adagrad, self).__init__(*args, **kwargs)
        self.primal_dual = primal_dual
        self.eta = eta
        self.delta = delta
        
    def estimate_stepsize(self):
        self.s = np.sqrt(self.s ** 2 + self.grad ** 2)
        self.inv_lr = self.delta + self.s
        assert len(self.inv_lr) == len(self.w)
        
    def step(self):
        if self.primal_dual:
            self.sum_grad += self.grad
            return self.w0 - self.eta * np.divide(self.sum_grad, self.inv_lr, out=np.zeros_like(self.inv_lr), where=self.inv_lr != 0)
        else:
            return self.w - self.eta * np.divide(self.grad, self.inv_lr, out=np.zeros_like(self.inv_lr), where=self.inv_lr != 0)
        
    def init_run(self, *args, **kwargs):
        super(Adagrad, self).init_run(*args, **kwargs)
        self.w0 = self.w.copy()
        self.s = np.zeros(len(self.w))
        self.sum_grad = np.zeros(self.d)
        
        
class MirrorDescent(Trainer):
    """
    Gradient descent with constant learning rate.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr, mirror_step, *args, **kwargs):
        super(MirrorDescent, self).__init__(*args, **kwargs)
        self.lr = lr
        self.mirror_step = mirror_step
        
    def step(self):
        return self.mirror_step(self.w, self.lr, self.grad)
    
    def init_run(self, *args, **kwargs):
        super(MirrorDescent, self).init_run(*args, **kwargs)
        
        
class Bb(Trainer):
    """
    Barzilai-Borwein Adaptive gradient descent based on the local smoothness constant
    """
    def __init__(self, lr0=1, option='1', *args, **kwargs):
        if not 0.0 < lr0:
            raise ValueError("Invalid lr0: {}".format(lr0))
        super(Bb, self).__init__(*args, **kwargs)
        self.lr0 = lr0
        self.option = option
        
    def estimate_stepsize(self):
        if self.option == '1':
            L = (self.w-self.w_old) @ (self.grad-self.grad_old) / la.norm(self.w-self.w_old)**2
        else:
            L = la.norm(self.grad-self.grad_old)**2 / ((self.grad-self.grad_old) @ (self.w-self.w_old))
        self.lr = self.lr0/L
        
    def step(self):
        self.grad = self.grad_func(self.w)
        self.estimate_stepsize()
        self.w_old = self.w.copy()
        self.grad_old = self.grad.copy()
        return self.w - self.lr*self.grad
        
    def init_run(self, *args, **kwargs):
        super(Bb, self).init_run(*args, **kwargs)
        self.lrs = []
        self.theta = np.inf
        grad = self.grad_func(self.w)
        # The first estimate is normalized gradient with a small coefficient
        self.lr = 1 / la.norm(grad)
        self.w_old = self.w.copy()
        self.grad_old = grad
        self.w -= self.lr * grad
        self.save_checkpoint()
        
    def update_logs(self):
        super(Bb, self).update_logs()
        self.lrs.append(self.lr)
        
        
class Polyak(Trainer):
    """
    Adaptive gradient descent based on the local smoothness constant
    
    Arguments:
        eps (float): an estimate of 1 / L^2, where L is the global smoothness constant
    """
    def __init__(self, f_opt=0, lr_min=0.0, *args, **kwargs):
        if lr_min < 0:
            raise ValueError("Invalid lr_min: {}".format(lr_min))
        super(Polyak, self).__init__(*args, **kwargs)
        self.lr_min = lr_min
        self.f_opt = f_opt
        
    def estimate_stepsize(self):
        f = self.loss_func(self.w)
        self.lr = max(self.lr_min, (f-self.f_opt) / la.norm(self.grad)**2)
        
    def step(self):
        self.grad = self.grad_func(self.w)
        self.estimate_stepsize()
        return self.w - self.lr * self.grad
        
    def init_run(self, *args, **kwargs):
        super(Polyak, self).init_run(*args, **kwargs)
        self.w_ave = self.w.copy()
        self.ws_ave = [self.w_ave.copy()]
        self.lr_sum = 0
        self.lrs = []
        
    def update_logs(self):
        super(Polyak, self).update_logs()
        self.lrs.append(self.lr)
        self.ws_ave.append(self.w_ave.copy())
        
        
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
        

class NestLine(Trainer):
    """
    Nesterov's accelerated gradient descent with line search.
    
    Arguments:
        lr0 (float, optional): an estimate of the inverse smoothness constant
            to initialize the stepsize
        strongly_convex (boolean, optional): if true, uses the variant
            for strongly convex functions, which requires mu>0 (default: False)
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr0=1, mu=0, backtracking=0.5, tolerance=0., *args, **kwargs):
        super(NestLine, self).__init__(*args, **kwargs)
        self.lr = lr0
        if mu < 0:
            raise ValueError("Invalid mu: {}".format(mu))
        self.mu = mu
        self.backtracking = backtracking
        self.tolerance = tolerance
        
    def condition(self, y, w_new):
        grad_new = self.grad_func(w_new)
        return grad_new @ (y-w_new) >= self.lr * la.norm(grad_new)**2 - self.tolerance
        
    def step(self):
        self.lr = self.lr / self.backtracking
        # Find a from quadratic equation a^2/(A+a) = 2*lr*(1 + mu*A)
        discriminant = (self.lr * (1+self.mu*self.A))**2 + self.A * self.lr * (1+self.mu*self.A)
        a = self.lr * (1+self.mu*self.A) + np.sqrt(discriminant)
        y = (self.A*self.w + a*self.v) / (self.A+a)
        gradient = self.grad_func(y)
        w_new = y - self.lr * gradient
        nest_condition_met = self.condition(y, w_new)
        self.it += 1
        
        it_extra = 0
        while not nest_condition_met and it_extra < 2 * self.it_max:
            self.lr *= self.backtracking
            discriminant = (self.lr * (1+self.mu*self.A))**2 + self.A * self.lr * (1+self.mu*self.A)
            a = self.lr * (1+self.mu*self.A) + np.sqrt(discriminant)
            y = self.A / (self.A+a) * self.w + a / (self.A+a) * self.v
            gradient = self.grad_func(y)
            w_new = y - self.lr * gradient
            nest_condition_met = self.condition(y, w_new)
            it_extra += 2
            if self.lr * self.backtracking == 0:
                break
        
        self.it += it_extra
        self.w = w_new
        self.A += a
        self.grad = self.grad_func(self.w)
        self.v -= a * self.grad
        
        return self.w
    
    def init_run(self, *args, **kwargs):
        super(NestLine, self).init_run(*args, **kwargs)
        self.A = 0
        self.v = self.w.copy()

class AdgdK1OverK(Trainer):
    """
    Adaptive gradient descent με τ1 = (k+1)/k * lr_prev και τ2 = 0.5 * ||Δx||/||Δg||.
    """
    def __init__(self, lr0=None, *args, **kwargs):
        super(AdgdK1OverK, self).__init__(*args, **kwargs)
        self.lr0 = lr0

    def init_run(self, w0):
        # Καλούμε πρώτα το base init_run ώστε να οριστούν self.w, self.it, κλπ.
        super(AdgdK1OverK, self).init_run(w0)

        # Αρχικός learning rate
        if self.lr0 is None:
            self.lr0 = 1e-10
        self.lr = self.lr0
        self.theta = np.inf

        # Υπολογίζουμε την πρώτη κλίση και την αποθηκεύουμε σε self.grad
        grad = self.grad_func(self.w)
        self.grad = grad

        # Αρχικές τιμές για curvature‐bound
        self.w_old    = self.w.copy()
        self.grad_old = grad

        # Πρώτο βήμα
        self.w -= self.lr * grad
        self.save_checkpoint()

    def estimate_stepsize(self):
        k = max(1, self.it)
        # τ1 = (k+1)/k * lr_prev
        tau1 = (k+1)/k * self.lr
        # τ2 = 0.5 * ||Δx|| / ||Δg||
        denom = la.norm(self.w - self.w_old) + 1e-12
        Lloc  = la.norm(self.grad - self.grad_old) / denom
        tau2  = 0.5 / Lloc
        lr_new = min(tau1, tau2)

        # Ενημέρωση του learning rate
        self.theta = lr_new / self.lr
        self.lr = lr_new

    def step(self):
        # Αποθήκευση παλιών τιμών για το επόμενο estimate
        self.w_old    = self.w.copy()
        self.grad_old = self.grad.copy()

        # Κανονικό πλήρες‐gradient update
        return self.w - self.lr * self.grad

class AdgdKOverKplus3(Trainer):
    """
    Adaptive gradient descent με τ1 = k/(k+3) * lr_prev και τ2 = 0.5 * ||Δx||/||Δg||.
    """
    def __init__(self, lr0=None, *args, **kwargs):
        super(AdgdKOverKplus3, self).__init__(*args, **kwargs)
        self.lr0 = lr0

    def init_run(self, w0):
        # 1) θεμέλιο init
        super(AdgdKOverKplus3, self).init_run(w0)

        # 2) αρχικός LR
        if self.lr0 is None:
            self.lr0 = 1e-10
        self.lr = self.lr0

        # 3) υπολογισμός και αποθήκευση πρώτης κλίσης
        grad      = self.grad_func(self.w)
        self.grad = grad

        # 4) curvature‐state
        self.w_old    = self.w.copy()
        self.grad_old = grad

        # 5) πρώτο βήμα
        self.w -= self.lr * grad
        self.save_checkpoint()

    def estimate_stepsize(self):
        k = max(1, self.it)
        tau1 = k/(k+3) * self.lr
        denom = la.norm(self.w - self.w_old) + 1e-12
        Lloc  = la.norm(self.grad - self.grad_old) / denom
        tau2  = 0.5 / Lloc
        self.lr = min(tau1, tau2)

    def step(self):
        self.w_old    = self.w.copy()
        self.grad_old = self.grad.copy()
        return self.w - self.lr * self.grad

class AdaptiveGDK1onKNesterov(Trainer):
    """
    Adaptive GD with τ₁=(k+1)/k * lr_prev and τ₂=0.5*||Δx||/||Δg||,
    plus Nesterov‐type momentum.
    """
    def __init__(self, lr0=None, *args, **kwargs):
        super(AdaptiveGDK1onKNesterov, self).__init__(*args, **kwargs)
        self.lr0 = lr0

    def init_run(self, w0):
        # Θεμέλιο init
        super(AdaptiveGDK1onKNesterov, self).init_run(w0)

        # Αρχικό βήμα
        if self.lr0 is None:
            self.lr0 = 1e-10
        self.lr = self.lr0

        # Μεταβλητές Nesterov
        self.w_nesterov      = self.w.copy()
        self.w_nesterov_old  = self.w.copy()
        self.alpha           = 1.0
        self.momentum        = 0.0

        # Υπολογισμός αρχικής κλίσης
        grad = self.grad_func(self.w)
        self.grad = grad

        # Αρχικές τιμές για curvature
        self.w_old    = self.w.copy()
        self.grad_old = grad

        # Πρώτο GD βήμα
        self.w -= self.lr * grad
        self.save_checkpoint()

    def estimate_stepsize(self):
        k = max(1, self.it)
        # τ1 = (k+1)/k * lr_prev
        tau1 = (k+1)/k * self.lr
        # τ2 = 0.5 * ||Δx|| / ||Δg||
        denom = la.norm(self.w - self.w_old) + 1e-12
        Lloc  = la.norm(self.grad - self.grad_old) / denom
        tau2  = 0.5 / Lloc

        # Ενημέρωση lr
        self.lr = min(tau1, tau2)

    def step(self):
        # 0) πρώτα ξαναϋπολογίζουμε τη νέα κλίση
        self.grad = self.grad_func(self.w)

        # 1) Υπολογισμός νέου momentum
        alpha_new = 0.5 * (1 + np.sqrt(1 + 4 * self.alpha ** 2))
        self.momentum = (self.alpha - 1) / alpha_new
        self.alpha = alpha_new

        # 2) Αποθήκευση παλιών τιμών
        self.w_old = self.w.copy()
        self.grad_old = self.grad.copy()
        self.w_nesterov_old = self.w_nesterov.copy()

        # 3) Look‐ahead GD βήμα
        self.w_nesterov = self.w - self.lr * self.grad
        # 4) Nesterov combination
        w_new = self.w_nesterov + self.momentum * (self.w_nesterov - self.w_nesterov_old)

        return w_new


class AdaPGNesterov(Trainer):
    """
    Adaptive GD with τ₁=(k+1)/k * lr_prev and τ₂=0.5*||Δx||/||Δg||,
    plus Nesterov‐type momentum.
    """
    def __init__(self, lr0=None, *args, **kwargs):
        super(AdaPGNesterov, self).__init__(*args, **kwargs)
        self.lr0 = lr0

    def init_run(self, w0):
        # Θεμέλιο init
        super(AdaPGNesterov, self).init_run(w0)

        # Αρχικό βήμα
        if self.lr0 is None:
            self.lr0 = 1e-10
        self.lr = self.lr0

        # Main variable

        # Nesterov momentum variable
        self.y = self.w.copy()
        self.y_old = self.w.copy()

        # Υπολογισμός αρχικής κλίσης
        grad = self.grad_func(self.w)
        self.grad = grad

        # Αρχικές τιμές για curvature
        self.w_old    = self.w.copy()
        self.grad_old = grad

        # Πρώτο GD βήμα
        self.w -= self.lr * grad
        self.save_checkpoint()

    def step(self):
        # Compute new extrapolation point and store the previous one
        self.y_old = self.y.copy()
        self.y = self.w + self.it / (self.it + 3) * (self.w - self.w_old)
        # Compute gradient at extrapolated point and store the previous one
        self.grad_old = self.grad.copy()
        self.grad = self.grad_func(self.y)

        # Compute stepsize
        k = max(1, self.it)
        # tau1 = (k + 1) / k * self.lr
        tau1 = (k+1)/k * self.lr
        denom = la.norm(self.y - self.y_old) + 1e-12
        Lloc = la.norm(self.grad - self.grad_old) / denom
        tau2 = 0.5 / Lloc
        self.lr = min(tau1, tau2)

        # Perform gradient step
        self.w_old = self.w.copy()
        self.w = self.y - self.lr * self.grad
        return self.w

class AdgdHybrid(Trainer):
    """
    Hybrid adaptive gradient descent:
      - βήμα με τ1 = (k+1)/k · η_{k−1}, τ2 = b_lr / L_loc
      - τοπική εκτίμηση μ_k = min((k+1)/k · μ_{k−1}, b_mu * L_loc)
      - Nesterov‐τύπου momentum με β_k = (1/η_k − μ_k)/(1/η_k + μ_k)

    Arguments:
      lr0 (float, optional): αρχικό βήμα (default: 1e-10)
      b_lr (float, optional): scaling παράγοντας για το curvature‐bound στο βήμα (default: 0.5)
      b_mu (float, optional): scaling παράγοντας για το curvature‐bound στο μ (default: 0.5)
    """
    def __init__(self, lr0=None, b_lr=0.5, b_mu=0.5, *args, **kwargs):
        super(AdgdHybrid, self).__init__(*args, **kwargs)
        self.lr0 = lr0
        self.b_lr = b_lr
        self.b_mu = b_mu

    def init_run(self, w0):
        super(AdgdHybrid, self).init_run(w0)
        # αρχικό learning rate
        if self.lr0 is None:
            self.lr0 = 1e-10
        self.lr = self.lr0
        # αρχική μ
        self.mu = 1.0 / self.lr
        # curvature state
        grad = self.grad_func(self.w)
        self.grad = grad
        self.w_old    = self.w.copy()
        self.grad_old = grad
        # για Nesterov: y_prev
        self.y_prev = self.w.copy()
        # πρώτο βήμα (GD)
        self.w -= self.lr * grad
        self.save_checkpoint()

    def estimate_stepsize(self):
        k = max(1, self.it)
        # τοπικός L_loc
        denom = la.norm(self.w - self.w_old) + 1e-12
        L_loc = la.norm(self.grad - self.grad_old) / denom
        # τ1 rule
        tau1 = (k+1)/k * self.lr
        # τ2 curvature‐bound
        tau2 = self.b_lr / L_loc
        lr_new = min(tau1, tau2)
        # αντίστοιχα για μ
        mu1 = (k+1)/k * self.mu
        mu2 = self.b_mu * L_loc
        mu_new = min(mu1, mu2)
        # ενημέρωση
        self.lr = lr_new
        self.mu = mu_new

    def step(self):
        # ξαναϋπολογίζουμε grad στο τρέχον w
        self.grad = self.grad_func(self.w)
        # momentum coefficient
        beta = (1.0/self.lr - self.mu) / (1.0/self.lr + self.mu)
        # αποθηκεύουμε παλιές τιμές
        self.w_old      = self.w.copy()
        self.grad_old   = self.grad.copy()
        y_old = self.y_prev.copy()
        # look‐ahead GD
        y = self.w - self.lr * self.grad
        # Nesterov‐update
        w_new = y + beta * (y - y_old)
        # ενημέρωση y_prev
        self.y_prev = y
        return w_new

    def update_logs(self):
        super(AdgdHybrid, self).update_logs()
        # κρατάμε και τα lr αν θέλουμε να τα βλέπουμε
        if not hasattr(self, 'lrs'):
            self.lrs = []
        self.lrs.append(self.lr)


class AdgdHybrid2(Trainer):
    """
    Hybrid adaptive gradient descent v2:
      1) βήμα: η_k = min( sqrt(1+theta_lr)*η_{k-1},
                           (k+1)/k*η_{k-1},
                           b_lr / L_loc )
      2) τοπική mu_k = min( sqrt(1+theta_mu)*μ_{k-1},
                              b_mu * L_loc )
      3) momentum: beta_k = (1/η_k - μ_k) / (1/η_k + μ_k)

    Arguments:
      lr0 (float, optional): αρχικό βήμα (default: 1e-10)
      b_lr (float, optional): scaling παράγοντας για τ2 στο βήμα (default: 0.5)
      b_mu (float, optional): scaling παράγοντας για bound στο μ (default: 0.5)
    """
    def __init__(self, lr0=None, b_lr=0.5, b_mu=0.5, *args, **kwargs):
        super(AdgdHybrid2, self).__init__(*args, **kwargs)
        self.lr0  = lr0
        self.b_lr = b_lr
        self.b_mu = b_mu

    def init_run(self, w0):
        super(AdgdHybrid2, self).init_run(w0)
        # αρχικό learning rate
        if self.lr0 is None:
            self.lr0 = 1e-10
        self.lr      = self.lr0
        self.theta_lr = np.inf
        # αρχική mu
        self.mu       = 1.0/self.lr
        self.theta_mu = np.inf

        # curvature‐state
        grad = self.grad_func(self.w)
        self.grad     = grad
        self.w_old    = self.w.copy()
        self.grad_old = grad

        # για Nesterov
        self.y_prev = self.w.copy()

        # πρώτο βήμα
        self.w -= self.lr * grad
        self.save_checkpoint()

    def estimate_stepsize(self):
        k = max(1, self.it)
        # curvature estimate
        denom = la.norm(self.w - self.w_old) + 1e-12
        L_loc = la.norm(self.grad - self.grad_old) / denom

        # (α) γεωμετρική αύξηση
        tau_g = np.sqrt(1 + self.theta_lr) * self.lr
        # (β) γραμμική αύξηση
        tau_l = (k+1)/k * self.lr
        # curvature‐bound
        tau_c = self.b_lr / L_loc
        # νέο lr
        lr_new = min(tau_g, tau_l, tau_c)
        self.theta_lr = lr_new / self.lr
        self.lr       = lr_new

        # τώρα το τοπικό mu
        mu_g = np.sqrt(1 + self.theta_mu) * self.mu
        mu_c = self.b_mu * L_loc
        mu_new = min(mu_g, mu_c)
        self.theta_mu = mu_new / self.mu
        self.mu       = mu_new

    def step(self):
        # ξαναϋπολογίζουμε grad
        self.grad = self.grad_func(self.w)

        # υπολογίζουμε beta
        beta = (1.0/self.lr - self.mu) / (1.0/self.lr + self.mu)

        # αποθηκεύουμε παλιές τιμές
        self.w_old    = self.w.copy()
        self.grad_old = self.grad.copy()
        y_old         = self.y_prev.copy()

        # look‐ahead GD
        y = self.w - self.lr * self.grad
        # Nesterov‐update
        w_new = y + beta * (y - y_old)

        # ενημέρωση για την επόμενη επανάληψη
        self.y_prev = y
        return w_new

    def update_logs(self):
        super(AdgdHybrid2, self).update_logs()
        # αν θέλεις να κρατάς και ιστορικό των βημάτων
        if not hasattr(self, 'lrs'):
            self.lrs = []
        self.lrs.append(self.lr)


class ADPG_Momentum(Trainer):
    """
    Adaptive GD + Nesterov momentum με:
      – βήμα γ_k = min((k+1)/k * γ_{k-1}, 0.5 / L_loc)
      – look‐ahead y^k = x^k + (k/(k+3))*(x^k - x^{k-1})
      – update x^{k+1} = y^k - γ_k * ∇f(y^k)
    """
    def __init__(self, lr0=None, *args, **kwargs):
        super(ADPG_Momentum, self).__init__(*args, **kwargs)
        self.lr0 = lr0

    def init_run(self, w0):
        # 1) θεμέλιο init
        super(ADPG_Momentum, self).init_run(w0)

        # 2) αρχικό βήμα
        self.lr = self.lr0 if self.lr0 is not None else 1e-6

        # 3) πρώτος gradient & αρχικοποίηση grad, grad_old
        g0 = self.grad_func(self.w)
        self.grad_old = g0.copy()
        self.grad     = g0.copy()

        # 4) αποθήκευση παλαιάς θέσης
        self.x_old = self.w.copy()

        # 5) πρώτος απλός βηματισμός
        self.w -= self.lr * g0
        self.save_checkpoint()

    def estimate_stepsize(self):
        k = max(1, self.it)
        # curvature estimate
        diff_x = self.w - self.x_old
        diff_g = self.grad - self.grad_old
        L_loc  = la.norm(diff_g) / (la.norm(diff_x) + 1e-12)
        # κανόνες για το γ
        tau1 = (k+1)/k * self.lr
        tau2 = 0.5 / L_loc
        self.lr = min(tau1, tau2)

    def step(self):
        # 1) υπολογίζουμε grad στο τρέχον x
        gk = self.grad_func(self.w)
        self.grad = gk.copy()

        # 2) εκτιμούμε το καινούριο βήμα
        self.estimate_stepsize()

        # 3) υπολογισμός momentum συντελεστή
        k = max(1, self.it)
        m = k / (k + 3)

        # 4) look‐ahead
        yk = self.w + m * (self.w - self.x_old)

        # 5) αποθήκευση για την επόμενη iter
        self.x_old    = self.w.copy()
        self.grad_old = gk.copy()

        # 6) τελική ενημέρωση
        x_new = yk - self.lr * self.grad_func(yk)

        # 7) Ενημέρωση της κατάστασης του Trainer
        self.w = x_new
        self.save_checkpoint()

        return x_new


class ADPG_Momentum2(Trainer):
    """
    Adaptive GD + Nesterov momentum v2 με:
      – global safeguard: lr ≤ 1/L_global
      – τ1 = (k+1)/k · lr_prev
      – τ2 = 0.5 / L_loc (με L_loc ≥ 1e-3·L_global)
      – γεωμετρική αύξηση τ_geom = sqrt(1+θ_lr)·lr_prev
      – lr_new = min(τ1, τ2, τ_geom, 1/L_global)
      – momentum m = k/(k+3)
      – look-ahead y^k = x^k + m·(x^k − x^{k−1})
      – update x^{k+1} = y^k − lr_new·∇f(y^k)
    """
    def __init__(self, lr0=None, *args, **kwargs):
        super(ADPG_Momentum2, self).__init__(*args, **kwargs)
        self.lr0 = lr0

    def init_run(self, w0):
        super(ADPG_Momentum2, self).init_run(w0)
        # αρχικό βήμα και global bound = 1/L_global
        self.lr         = self.lr0 if self.lr0 is not None else 1e-6
        self.gamma_max  = self.lr   # = 1/L_global
        self.theta_lr   = np.inf

        # init grad & ιστορικό x_old
        g0             = self.grad_func(self.w)
        self.grad_old  = g0.copy()
        self.grad      = g0.copy()
        self.x_old     = self.w.copy()

        # πρώτο απλό βήμα
        self.w        -= self.lr * g0
        self.save_checkpoint()

    def estimate_stepsize(self):
        k = max(1, self.it)
        # curvature
        dx    = self.w - self.x_old
        dg    = self.grad - self.grad_old
        L_loc = np.linalg.norm(dg) / (np.linalg.norm(dx) + 1e-12)
        L_loc = max(L_loc, 1.0/self.gamma_max * 1e-3)

        # κανόνες βήματος
        tau1   = (k+1)/k * self.lr
        tau2   = 0.5       / L_loc
        tau_g  = np.sqrt(1 + self.theta_lr) * self.lr

        # νέο lr με global safeguard
        new_lr      = min(tau1, tau2, tau_g, self.gamma_max)
        self.theta_lr = new_lr / self.lr
        self.lr      = new_lr

    def step(self):
        # update grad
        gk           = self.grad_func(self.w)
        self.grad    = gk.copy()

        # επανεκτίμηση βήματος
        self.estimate_stepsize()

        # momentum & look-ahead
        k           = max(1, self.it)
        m           = k/(k+3)
        yk          = self.w + m * (self.w - self.x_old)

        # store history
        self.x_old  = self.w.copy()
        self.grad_old = gk.copy()

        # τελικό update
        x_new       = yk - self.lr * self.grad_func(yk)

        # ενημέρωση Trainer
        self.w      = x_new
        self.save_checkpoint()

        return x_new


class ADPG_Momentum3(Trainer):
    """
    Adaptive GD + Nesterov momentum v3 με:
      – global safeguard: lr ≤ 1/L_global
      – τ1 = (k+1)/k · lr_prev
      – τ2 = 0.5 / L_loc, με L_loc ≥ L_global·1e-3
      – γεωμετρική αύξηση τ_geom = sqrt(1+θ_lr)·lr_prev
      – lr_new = min(τ1, τ2, τ_geom, 1/L_global)
      – momentum m = k/(k+3)
      – look-ahead y^k = x^k + m·(x^k − x^{k−1})
      – update x^{k+1} = y^k − lr_new·∇f(y^k)
    """
    def __init__(self, lr0=None, L_global=None, *args, **kwargs):
        super(ADPG_Momentum3, self).__init__(*args, **kwargs)
        self.lr0      = lr0
        # πρέπει να δώσετε το global Lipschitz L_global
        self.L_global = L_global

    def init_run(self, w0):
        super(ADPG_Momentum3, self).init_run(w0)
        # 1) αρχικό βήμα & global cap
        if self.lr0 is None:
            raise ValueError("Πρέπει να δώσετε lr0=1/L_global")
        self.lr         = self.lr0
        self.gamma_max  = self.lr0    # = 1/L_global
        self.theta_lr   = np.inf

        # 2) init history για curvature
        g0             = self.grad_func(self.w)
        self.grad_old  = g0.copy()
        self.grad      = g0.copy()
        self.x_old     = self.w.copy()

        # 3) πρώτο απλό step
        self.w        -= self.lr * g0
        self.save_checkpoint()

    def estimate_stepsize(self):
        k = max(1, self.it)
        # 1) τοπική curvature
        dx    = self.w - self.x_old
        dg    = self.grad - self.grad_old
        L_loc = la.norm(dg) / (la.norm(dx) + 1e-12)
        # κατώφλι για να μην πέσει πολύ χαμηλά
        L_loc = max(L_loc, self.L_global * 1e-3)

        # 2) οι τρεις κανόνες
        tau1  = (k+1)/k * self.lr
        tau2  = 0.5      / L_loc
        tau_g = np.sqrt(1 + self.theta_lr) * self.lr

        # 3) global safeguard
        new_lr      = min(tau1, tau2, tau_g, self.gamma_max)
        self.theta_lr = new_lr / self.lr
        self.lr      = new_lr

    def step(self):
        # 1) ξαναϋπολογισμός gradient
        gk           = self.grad_func(self.w)
        self.grad    = gk.copy()

        # 2) update βήμα
        self.estimate_stepsize()

        # 3) Nesterov momentum
        k    = max(1, self.it)
        m    = k/(k+3)
        yk   = self.w + m * (self.w - self.x_old)

        # 4) save history for next curvature
        self.x_old    = self.w.copy()
        self.grad_old = gk.copy()

        # 5) final update at yk
        x_new = yk - self.lr * self.grad_func(yk)

        # 6) update Trainer state
        self.w = x_new
        self.save_checkpoint()
        return x_new

class ADPG_Momentum4(Trainer):
    """
    Adaptive GD + Nesterov momentum v4:
      – global safeguard: lr ≤ 1/L_global
      – τ1 = (k+1)/k · lr_prev
      – τ2 = 0.5 / L_loc_avg, with L_loc_avg = ρ·L_loc_avg_prev + (1-ρ)·L_loc
      – τ_geom = sqrt(1+θ_lr)·lr_prev
      – lr_new = min(τ1, τ2, τ_geom, 1/L_global)
      – momentum m = k/(k+3)
      – look-ahead y^k = x^k + m·(x^k − x^{k−1})
      – update x^{k+1} = y^k − lr_new·∇f(y^k)
    """
    def __init__(self, lr0=None, L_global=None, rho=0.95, *args, **kwargs):
        super(ADPG_Momentum4, self).__init__(*args, **kwargs)
        if lr0 is None or L_global is None:
            raise ValueError("Δώστε lr0=1/L_global και L_global")
        self.lr0      = lr0
        self.L_global = L_global
        self.gamma_max = lr0      # = 1/L_global
        self.rho      = rho       # decay for L_loc smoothing

    def init_run(self, w0):
        super(ADPG_Momentum4, self).init_run(w0)
        # 1) αρχικό lr
        self.lr        = self.lr0
        self.theta_lr  = np.inf
        # 2) init history
        g0            = self.grad_func(self.w)
        self.grad_old = g0.copy()
        self.grad     = g0.copy()
        self.x_old    = self.w.copy()
        # 3) init smoothed L_loc
        self.L_loc_avg = 1.0 / self.lr0  # initial guess
        # 4) πρώτο απλό βήμα
        self.w       -= self.lr * g0
        self.save_checkpoint()

    def estimate_stepsize(self):
        k = max(1, self.it)
        # a) τοπική εκτίμηση curvature
        dx    = self.w - self.x_old
        dg    = self.grad - self.grad_old
        L_loc = np.linalg.norm(dg) / (np.linalg.norm(dx) + 1e-12)
        # b) smoothing με running average
        self.L_loc_avg = self.rho * self.L_loc_avg + (1 - self.rho) * L_loc
        # c) κανόνες βήματος
        tau1  = (k+1)/k * self.lr
        tau2  = 0.5       / self.L_loc_avg
        tau_g = np.sqrt(1 + self.theta_lr) * self.lr
        # d) global safeguard
        new_lr       = min(tau1, tau2, tau_g, self.gamma_max)
        self.theta_lr = new_lr / self.lr
        self.lr       = new_lr

    def step(self):
        # 1) υπολογισμός gradient
        gk           = self.grad_func(self.w)
        self.grad    = gk.copy()
        # 2) επανεκτίμηση βήματος
        self.estimate_stepsize()
        # 3) momentum look-ahead
        k    = max(1, self.it)
        m    = k/(k+3)
        yk   = self.w + m * (self.w - self.x_old)
        # 4) save history
        self.x_old    = self.w.copy()
        self.grad_old = gk.copy()
        # 5) τελική ενημέρωση
        x_new = yk - self.lr * self.grad_func(yk)
        # 6) checkpoint
        self.w = x_new
        self.save_checkpoint()
        return x_new
