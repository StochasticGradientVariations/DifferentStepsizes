# optimizer_adgrad2.py

import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required

class AdsgdAdGrad2(Optimizer):
    """
    AdGrad with external hook set_tau2.
    """
    def __init__(self, params, lr=1e-3, weight_decay=0.0, tau_rule='original'):
        defaults = dict(lr=lr, weight_decay=weight_decay, tau_rule=tau_rule)
        super().__init__(params, defaults)

    def set_tau2(self, tau2: float):
        """Override τ2 BEFORE calling step()."""
        for group in self.param_groups:
            group['lr'] = tau2
            self.state[id(group)]['la_old'] = tau2

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            state = self.state.setdefault(id(group), {})
            k      = state.get('k', 0) + 1
            state['k'] = k
            la_old = state.get('la_old', group['lr'])
            rule   = group['tau_rule']

            # τ1 rule
            tau1 = (k+1)/k * la_old if rule=='original' else k/(k+3)*la_old
            # τ2 is already in group['lr'] via set_tau2 (or equals τ1 on first iter)
            tau2 = group['lr']
            la_new = min(tau1, tau2)
            state['la_old'] = la_new

            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad.data
                if wd!=0: g = g.add(wd, p.data)
                p.data.add_(g, alpha=-la_new)

            group['lr'] = la_new

        return loss
