import torch
from torch.optim.optimizer import Optimizer
import math

class AdaptiveNPGM(Optimizer):
    def __init__(self, params, lr=1.0, epsilon=1e-12):
        if lr <= 0:
            raise ValueError("lr must be positive")
        defaults = dict(
            lr=lr,
            epsilon=epsilon
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        # if closure is None:
            # raise RuntimeError("Closure must be provided to evaluate f(x)")
        # Evaluate loss + gradients
        # loss = closure()

        for group in self.param_groups:
            eps = group['epsilon']

            # Gather parameters and raw gradients
            params, grads = [], []
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                params.append(p)
                grads.append(g.view(-1))
            if not grads:
                continue

            g_k = torch.cat(grads)         # raw gradient
            x_k = torch.cat([p.data.view(-1) for p in params])

            norm_gk = g_k.norm()
            s_k = torch.arcsinh(norm_gk) / norm_gk * g_k
            norm_k = torch.arcsinh(norm_gk) / norm_gk
            # param_norm = x_k.norm()
            # max_norm = max(1, param_norm)
            # print(param_norm)

            state = group.setdefault('state', {})
            # Initial step
            if 'gamma' not in state:
                gamma0 = 1.0
                state.update({
                    'gamma':      gamma0,
                    'gamma_prev': gamma0,
                    'g_old':      s_k.clone(),
                    'x_old':      x_k.clone(),
                    'normal_prev':norm_k
                })
                # x1 = x0 - gamma0 * g_k
                with torch.no_grad():
                    offset = 0
                    for p in params:
                        numel = p.numel()
                        seg = s_k[offset:offset+numel].view_as(p)
                        p.data.add_(seg, alpha=-gamma0)
                        offset += numel
                continue

            # Estimate local Lipschitz: Lk = ||Δx|| / ||Δg||
            delta_x = x_k - state['x_old']
            delta_g = s_k - state['g_old']
            Lk = delta_x.norm() / (delta_g.norm() + eps)

            # Previous step-sizes
            gamma_k   = state['gamma']
            gamma_km1 = state['gamma_prev']

            # norm_k = state['normal']
            norm_km1 = state['normal_prev']

            # Compute τ = γ_k * sqrt(1 + γ_k/γ_{k-1})
            tau = gamma_k * math.sqrt(norm_km1 / norm_k * (1 + gamma_k / gamma_km1))
            # λ_k = min(τ, 0.5/Lk)
            gamma_new = min(tau, 0.5 / Lk.item())

            # Parameter update: x_{k+1} = x_k - γ_new * raw gradient
            with torch.no_grad():
                offset = 0
                for p in params:
                    numel = p.numel()
                    seg = s_k[offset:offset+numel].view_as(p)
                    p.data.add_(seg, alpha=-gamma_new)
                    # p.data.mul_(1/max_norm)
                    offset += numel

            # xk_new = torch.cat([p.data.view(-1) for p in params])
            # param_norm = xk_new.norm()
            # max_norm = max(.001, .001*param_norm)
            # with torch.no_grad():
                # offset = 0
                # for p in params:
                    # p.data.mul_(1/max_norm)
            # xk_new = torch.cat([p.data.view(-1) for p in params])
            # print(xk_new.norm())

            # Shift state
            state['x_old']      = x_k.clone()
            state['g_old']      = s_k.clone()
            state['gamma_prev'] = gamma_k
            state['gamma']      = gamma_new
            group['lr']         = gamma_new
            state['normal_prev']= norm_k

    
    def __repr__(self):
        return f"adaptive NPGM"
