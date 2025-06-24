import math
import torch
from torch.optim.optimizer import Optimizer

class AdaptiveSGD(Optimizer):
    """
    Minimal adaptation of AdaptiveNPGM to implement Adaptive Gradient Descent
    (Algorithm 1 of https://arxiv.org/abs/1910.09529) for debugging.
    """
    def __init__(self, params, lr=1.0, weight_decay=0.0, epsilon=1e-12):
        if lr <= 0:
            raise ValueError("lr must be positive")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            epsilon=epsilon
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("Closure must be provided to evaluate f(x)")
        # Evaluate loss + gradients
        loss = closure()

        for group in self.param_groups:
            wd  = group['weight_decay']
            eps = group['epsilon']

            # Gather parameters and raw gradients
            params, grads = [], []
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                if wd != 0:
                    g = g.add(p.data, alpha=wd)
                params.append(p)
                grads.append(g.view(-1))
            if not grads:
                continue

            g_k = torch.cat(grads)         # raw gradient
            x_k = torch.cat([p.data.view(-1) for p in params])

            state = group.setdefault('state', {})
            # Initial step
            if 'gamma' not in state:
                gamma0 = 1.0
                state.update({
                    'gamma':      gamma0,
                    'gamma_prev': gamma0,
                    'g_old':      g_k.clone(),
                    'x_old':      x_k.clone()
                })
                # x1 = x0 - gamma0 * g_k
                with torch.no_grad():
                    offset = 0
                    for p in params:
                        numel = p.numel()
                        seg = g_k[offset:offset+numel].view_as(p)
                        p.data.add_(seg, alpha=-gamma0)
                        offset += numel
                continue

            # Estimate local Lipschitz: Lk = ||Δx|| / ||Δg||
            delta_x = x_k - state['x_old']
            delta_g = g_k - state['g_old']
            Lk = delta_x.norm() / (delta_g.norm() + eps)

            # Previous step-sizes
            gamma_k   = state['gamma']
            gamma_km1 = state['gamma_prev']

            # Compute τ = γ_k * sqrt(1 + γ_k/γ_{k-1})
            tau = gamma_k * math.sqrt(1 + gamma_k / gamma_km1)
            # λ_k = min(τ, 0.5/Lk)
            gamma_new = min(tau, 0.5 / Lk.item())

            # Parameter update: x_{k+1} = x_k - γ_new * raw gradient
            with torch.no_grad():
                offset = 0
                for p in params:
                    numel = p.numel()
                    seg = g_k[offset:offset+numel].view_as(p)
                    p.data.add_(seg, alpha=-gamma_new)
                    offset += numel

            # Shift state
            state['x_old']      = x_k.clone()
            state['g_old']      = g_k.clone()
            state['gamma_prev'] = gamma_k
            state['gamma']      = gamma_new
            group['lr']         = gamma_new

        return loss