import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required

class AdsgdAdGrad(Optimizer):
    """
    Adaptive SGD implementing ad_grad logic, with selectable tau1 rule:
      - 'original': tau1 = (k+1)/k * la_old
      - 'mod':      tau1 = k/(k+3) * la_old
    """
    def __init__(self, params, lr=1e-6, weight_decay=0, tau_rule='original'):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid initial learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if tau_rule not in ('original', 'mod'):
            raise ValueError(f"Invalid tau_rule: {tau_rule}")
        defaults = dict(lr=lr, weight_decay=weight_decay, tau_rule=tau_rule)
        super(AdsgdAdGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdsgdAdGrad, self).__setstate__(state)
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

    def step(self, closure=None):
        """
        Performs one optimization step using ad_grad rules, with selectable tau1.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            state = self.state.setdefault(id(group), {})
            # init or update iteration count and last step size
            if 'k' not in state:
                state['k']      = 1
                state['la_old'] = group['lr']
            else:
                state['k'] += 1

            k       = state['k']
            la_old  = state['la_old']
            grad_d  = group.get('grad_diff_norm',  0.0)
            param_d = group.get('param_diff_norm', 0.0)
            rule    = group.get('tau_rule', 'original')

            # compute tau1 according to rule
            if rule == 'mod':
                tau1 = k/(k + 3) * la_old
            else:
                tau1 = (k + 1)/k * la_old
            tau2 = 0.5 * (param_d/grad_d) if grad_d > 0 else tau1
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
