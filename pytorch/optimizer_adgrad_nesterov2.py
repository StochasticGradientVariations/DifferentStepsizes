# optimizer_adgrad_nesterov2.py

import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required

class AdsgdAdGradNesterov2(Optimizer):
    """
    AdGrad2 + Nesterov momentum.
    """
    def __init__(self, params, lr=1e-3, weight_decay=0.0,
                 tau_rule='original', momentum=0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay,
                        tau_rule=tau_rule, momentum=momentum)
        super().__init__(params, defaults)

    def set_tau2(self, tau2: float):
        for g in self.param_groups:
            g['lr'] = tau2
            self.state[id(g)]['la_old'] = tau2

    def step(self, closure=None):
        loss = None
        if closure: loss = closure()

        for group in self.param_groups:
            state = self.state.setdefault(id(group), {})
            k      = state.get('k',0) + 1
            la_old = state.get('la_old', group['lr'])
            rule   = group['tau_rule']
            mu     = group['momentum']
            state['k'] = k

            tau1 = (k+1)/k*la_old if rule=='original' else k/(k+3)*la_old
            tau2 = group['lr']
            la_new = min(tau1, tau2)
            state['la_old'] = la_new

            # ensure one buffer per parameter
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = [torch.zeros_like(p.data) for p in group['params']]

            wd = group['weight_decay']
            for idx,p in enumerate(group['params']):
                if p.grad is None: continue
                d_p = p.grad.data
                if wd!=0: d_p = d_p.add(wd, p.data)

                buf = state['momentum_buffer'][idx]
                v_prev = buf.clone()
                buf.mul_(mu).add_(d_p, alpha=la_new)

                # Nesterov update:
                p.data.add_(v_prev, alpha=-mu).add_(d_p, alpha=-la_new)

            group['lr'] = la_new

        return loss
