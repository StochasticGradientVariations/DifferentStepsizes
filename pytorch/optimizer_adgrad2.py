# pytorch/optimizer_adgrad2.py

import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required

class AdsgdAdGrad2(Optimizer):
    """
    Adaptive SGD implementing ad_grad logic, with selectable tau1 rule:
      - 'original': tau1 = (k+1)/k * la_old
      - 'mod':      tau1 = k/(k+3) * la_old
    Adds public `set_tau2(tau2)` to override tau2 before step().
    """
    def __init__(self, params, lr=1e-6, weight_decay=0, tau_rule='original'):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid initial learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if tau_rule not in ('original', 'mod'):
            raise ValueError(f"Invalid tau_rule: {tau_rule}")
        defaults = dict(lr=lr, weight_decay=weight_decay, tau_rule=tau_rule)
        super(AdsgdAdGrad2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdsgdAdGrad2, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('lr', 1e-6)
            group.setdefault('weight_decay', 0)
            group.setdefault('tau_rule', 'original')

    def compute_dif_norms(self, prev_optimizer):
        """
        Must be called BEFORE step(), to populate:
          group['grad_diff_norm'], group['param_diff_norm']
        """
        for group, prev_group in zip(self.param_groups, prev_optimizer.param_groups):
            grad_diff_sq  = 0.0
            param_diff_sq = 0.0
            for p, prev_p in zip(group['params'], prev_group['params']):
                if p.grad is None or prev_p.grad is None:
                    continue
                grad_diff_sq  += (p.grad.data - prev_p.grad.data).norm().item()**2
                param_diff_sq += (p.data       - prev_p.data     ).norm().item()**2
            group['grad_diff_norm']  = np.sqrt(grad_diff_sq)
            group['param_diff_norm'] = np.sqrt(param_diff_sq)

    def set_tau2(self, tau2: float):
        """
        Public hook: store τ₂ override in state, without mutating group['lr'] directly.
        """
        for group in self.param_groups:
            st = self.state.setdefault(id(group), {})
            st['tau2_override'] = tau2

    def step(self, closure=None):
        """
        Performs one optimization step using ad_grad rules, with overrideable tau2.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            state = self.state.setdefault(id(group), {})

            # iteration count
            k = state.get('k', 0) + 1
            state['k'] = k

            la_old = state.get('la_old', group['lr'])
            rule   = group['tau_rule']

            # τ1 rule
            if rule == 'mod':
                tau1 = k / (k + 3) * la_old
            else:
                tau1 = (k + 1) / k * la_old

            # τ2: either override or fallback to τ1
            tau2 = state.get('tau2_override', tau1)

            la_new = min(tau1, tau2)
            state['la_old'] = la_new

            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                if wd != 0:
                    g = g.add(wd, p.data)
                p.data.add_(g, alpha=-la_new)

            group['lr'] = la_new

        return loss
