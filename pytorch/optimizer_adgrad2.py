# optimizer_adgrad2.py for easier use on the jupyter notebook problems (eg. mushrooms...etc)
#“Adds a public hook for externally overriding the τ₂ adaptive step-size, simplifying integration and ensuring the correct curvature-based adjustment without fallback hacks.”

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

    def set_tau2(self, tau2: float):
        """
        Public hook: override tau2 before calling step().
        """
        for group in self.param_groups:
            # directly set lr to new tau2
            group['lr'] = tau2
            # also update state so next tau1 uses this as la_old
            state = self.state.setdefault(id(group), {})
            state['la_old'] = tau2

    def step(self, closure=None):
        """
        Performs one optimization step using ad_grad rules, with selectable tau1,
        but tau2 can be overridden via set_tau2() before calling.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            state = self.state.setdefault(id(group), {})
            # init or update iteration count
            if 'k' not in state:
                state['k']      = 1
                state['la_old'] = group['lr']
            else:
                state['k'] += 1

            k      = state['k']
            la_old = state['la_old']

            # compute tau1 according to rule
            rule = group['tau_rule']
            if rule == 'mod':
                tau1 = k/(k + 3) * la_old
            else:
                tau1 = (k + 1)/k * la_old

            # tau2 is already in group['lr'] if user called set_tau2(),
            # otherwise we fall back to tau1
            tau2 = group['lr']

            # choose the min(tau1, tau2)
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
