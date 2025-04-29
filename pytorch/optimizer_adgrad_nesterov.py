import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required

class AdsgdAdGradNesterov(Optimizer):
    """
    Adaptive SGD with Nesterov-type acceleration.
    Combines ad_grad adaptive step-size (tau_rule: 'original' or 'mod')
    with Nesterov momentum computed via local curvature estimate.
    """
    def __init__(self, params, lr=1e-6, weight_decay=0, tau_rule='original'):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid initial learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if tau_rule not in ('original', 'mod'):
            raise ValueError(f"Invalid tau_rule: {tau_rule}")
        defaults = dict(lr=lr, weight_decay=weight_decay, tau_rule=tau_rule)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('lr', 1e-6)
            group.setdefault('weight_decay', 0)
            group.setdefault('tau_rule', 'original')

    def compute_dif_norms(self, prev_optimizer):
        # same as before
        for group, prev_group in zip(self.param_groups, prev_optimizer.param_groups):
            grad_diff_sq, param_diff_sq = 0.0, 0.0
            for p, prev_p in zip(group['params'], prev_group['params']):
                if p.grad is None or prev_p.grad is None:
                    continue
                grad_diff_sq  += (p.grad.data - prev_p.grad.data).norm().item()**2
                param_diff_sq += (p.data       - prev_p.data     ).norm().item()**2
            group['grad_diff_norm']  = np.sqrt(grad_diff_sq)
            group['param_diff_norm'] = np.sqrt(param_diff_sq)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            state = self.state.setdefault(id(group), {})
            # Initialize state variables
            if 'k' not in state:
                state['k'] = 1
                state['la_old'] = group['lr']
                state['Lambda_old'] = group['lr']  # start same
                # y_prev, x_prev per param
                state['y_prev'] = [p.data.clone() for p in group['params']]
            else:
                state['k'] += 1

            k = state['k']
            la_old = state['la_old']
            Lambda_old = state['Lambda_old']

            grad_d = group.get('grad_diff_norm', 0.0)
            param_d = group.get('param_diff_norm', 0.0)
            rule   = group.get('tau_rule', 'original')

            # adaptive lambda (as in AdGrad)
            if rule == 'mod':
                tau1 = k/(k+3) * la_old
            else:
                tau1 = (k+1)/k * la_old
            tau2 = 0.5 * (param_d/grad_d) if grad_d>0 else tau1
            la_new = min(tau1, tau2)

            # adaptive Lambda (for momentum)
            if 'Theta_old' not in state:
                state['Theta_old'] = 1.0
            Theta_old = state['Theta_old']
            # mirror rule: swap grad and param norms
            mu1 = Theta_old
            if rule == 'mod':
                mu_tau1 = k/(k+3) * Lambda_old
            else:
                mu_tau1 = (k+1)/k * Lambda_old
            mu_tau2 = 0.5 * (grad_d/param_d) if param_d>0 else mu_tau1
            Lambda_new = min(mu_tau1, mu_tau2)

            # compute momentum coefficient beta
            t_val = np.sqrt(la_new * Lambda_new)
            beta = (1.0 - t_val)/(1.0 + t_val)

            # update state
            state['la_old'] = la_new
            state['Lambda_old'] = Lambda_new
            state['Theta_old'] = Lambda_new/Lambda_old

            # perform Nesterov update
            new_y = []
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    new_y.append(state['y_prev'][idx])
                    continue
                g = p.grad.data
                wd = group['weight_decay']
                if wd!=0:
                    g = g.add(wd, p.data)
                # y_k+1 = x_k - la_new * g
                y1 = p.data - la_new * g
                # x_k+1 = y_k+1 + beta*(y_k+1 - y_k)
                y_prev = state['y_prev'][idx]
                x_new = y1 + beta * (y1 - y_prev)
                # write back
                p.data.copy_(x_new)
                new_y.append(y1)

            # save new y
            state['y_prev'] = new_y
            # logging
            group['lr'] = la_new

        return loss
