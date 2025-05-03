# optimizer_adgrad_nesterov2.py
#“Introduces Nesterov momentum updates combined with the ad_grad step‐size rule for smoother, accelerated convergence by leveraging both curvature‐based learning rates and lookahead momentum.”
import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required

class AdsgdAdGradNesterov2(Optimizer):
    """
    Adaptive SGD + Nesterov momentum + ad_grad step-size.
    - tau1: original or modified rule
    - επιτρέπει set_tau2() για εξωτερικό override του τ₂
    """
    def __init__(self, params, lr=1e-6, weight_decay=0, tau_rule='original', momentum=0.0):
        if lr is not required and lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if tau_rule not in ('original', 'mod'):
            raise ValueError(f"Invalid tau_rule: {tau_rule}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        defaults = dict(lr=lr, weight_decay=weight_decay,
                        tau_rule=tau_rule, momentum=momentum)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for g in self.param_groups:
            g.setdefault('lr', 1e-6)
            g.setdefault('weight_decay', 0)
            g.setdefault('tau_rule', 'original')
            g.setdefault('momentum', 0.0)

    def set_tau2(self, tau2: float):
        """Override τ₂ (που θα χρησιμοποιηθεί στο επόμενο step)."""
        for g in self.param_groups:
            # Ορίζουμε προσωρινά τον ρυθμό στο τ₂
            g['lr'] = tau2
            # Και το κρατάμε σαν παλιό βήμα ώστε το επόμενο tau1 να το χρησιμοποιήσει
            st = self.state.setdefault(id(g), {})
            st['la_old'] = tau2

    def step(self, closure=None):
        loss = None
        if closure:
            loss = closure()
        for group in self.param_groups:
            state = self.state.setdefault(id(group), {})

            # init counter & last-la
            if 'k' not in state:
                state['k']      = 1
                state['la_old'] = group['lr']
            else:
                state['k'] += 1

            k      = state['k']
            la_old = state['la_old']
            rule   = group['tau_rule']
            mu     = group['momentum']

            # υπολογισμός τ₁
            if rule == 'mod':
                tau1 = k/(k+3) * la_old
            else:
                tau1 = (k+1)/k * la_old

            # τ₂ έχει είτε οριστεί από set_tau2, είτε είναι ίσο με τ₁
            tau2 = group['lr']

            # νέο βήμα = min(τ₁, τ₂)
            la_new = min(tau1, tau2)
            state['la_old'] = la_new

            wd = group['weight_decay']
            # Nesterov momentum update:
            # v ← μ·v + la_new·g
            # p ← p - μ·v_prev - la_new·g
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if wd != 0:
                    d_p = d_p.add(wd, p.data)

                # buffer στο state για το momentum
                buf = state.setdefault('momentum_buffer', torch.zeros_like(p.data))
                v_prev = buf.clone()
                buf.mul_(mu).add_(d_p, alpha=la_new)

                # actual Nesterov update:
                p.data.add_(v_prev, alpha=-mu).add_(d_p, alpha=-la_new)

            group['lr'] = la_new

        return loss
