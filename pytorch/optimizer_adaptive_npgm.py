import torch
from torch.optim.optimizer import Optimizer

class AdaptiveNPGM(Optimizer):
    """
    Fully faithful implementation of Algorithm 1: Adaptive NPGM
    """
    def __init__(self, params, lr=1e-2, weight_decay=0.0, epsilon=1e-12):
        if lr <= 0:
            raise ValueError("lr must be positive")
        defaults = dict(lr=lr, weight_decay=weight_decay, epsilon=epsilon)
        super().__init__(params, defaults)

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("Closure must be provided to evaluate f(x)")
        loss = closure()

        for group in self.param_groups:
            wd  = group['weight_decay']
            eps = group['epsilon']

            # Gather parameters and gradients
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

            g_k = torch.cat(grads)
            norm_g = g_k.norm()
            s_k = torch.asinh(norm_g) / norm_g
            scaled_gk = s_k * g_k
            x_k = torch.cat([p.data.view(-1) for p in params])

            # Access or initialize state
            # νέα γραμμή — ξεχωριστό dict για κάθε group
            state = group.setdefault('state', {})

            if 'gamma' not in state:
                gamma0 = group['lr']
                state.update({
                    'gamma': gamma0,
                    'gamma_prev': gamma0,
                    's_old': s_k,
                    'norm_old': norm_g.clone(),
                    'scaled_g_old': scaled_gk.clone(),
                    'x_old': x_k.clone()
                })

                # First update: x¹ = x⁰ − γ₀ * scaled_g₀
                with torch.no_grad():
                    offset = 0
                    for p in params:
                        numel = p.numel()
                        seg = scaled_gk[offset:offset + numel].view_as(p)
                        p.data.add_(seg, alpha=-gamma0)
                        offset += numel
                continue

            # Eq. (11): L_k = ‖ŷₖ − ŷₖ₋₁‖ / ‖xₖ − xₖ₋₁‖
            delta_y = scaled_gk - state['scaled_g_old']
            delta_x = x_k - state['x_old']
            Lk = delta_y.norm() / (delta_x.norm() + eps)

            # Eq. (12): τ = γₖ · sqrt( (sₖ₋₁ / sₖ) · (nₖ / nₖ₋₁) · (1 + γₖ / γₖ₋₁) )
            gamma_k   = state['gamma']
            gamma_km1 = state['gamma_prev']
            norm_old  = state['norm_old']
            s_old     = state['s_old']

            tau = gamma_k * torch.sqrt(
                (s_old / s_k)
              * (norm_g / norm_old)
              * (1 + gamma_k / gamma_km1)
            )

            # γₖ₊₁ = min(τ, 1 / (2·Lₖ))
            gamma_new = min(tau.item(), 1.0 / (2 * Lk.item()))

            # Eq. (13): xᵏ⁺¹ = xᵏ − γₖ₊₁·ŷₖ
            with torch.no_grad():
                offset = 0
                for p in params:
                    numel = p.numel()
                    seg = scaled_gk[offset:offset + numel].view_as(p)
                    p.data.add_(seg, alpha=-gamma_new)
                    offset += numel

            # Shift state for next iteration
            state['x_old'] = x_k.clone()
            state['scaled_g_old'] = scaled_gk.clone()
            state['s_old'] = s_k
            state['norm_old'] = norm_g.clone()
            state['gamma_prev'] = gamma_k
            state['gamma'] = gamma_new
            group['lr'] = gamma_new

        return loss
