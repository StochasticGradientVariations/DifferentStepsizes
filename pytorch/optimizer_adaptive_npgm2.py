import torch
import numpy as np
from torch.optim.optimizer import Optimizer

class AdaptiveNPGM(Optimizer):
    """
    Adaptive SGD optimizer implementing the Adaptive NPGM logic from Algorithm 1.
    Scaling factor sₖ = arsinh(||gₖ||)/||gₖ||, adaptive rule for γₖ₊₁.
    """
    def __init__(self, params, lr=1e-2, weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid initial learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('lr', 1e-2)
            group.setdefault('weight_decay', 0)

    def compute_dif_norms(self, prev_optimizer):
        # unchanged
        for group, prev_group in zip(self.param_groups, prev_optimizer.param_groups):
            grad_diff_sq = 0.0
            param_diff_sq = 0.0
            for p, prev_p in zip(group['params'], prev_group['params']):
                if p.grad is None or prev_p.grad is None:
                    continue
                grad_diff_sq  += (p.grad.data - prev_p.grad.data).norm().item()**2
                param_diff_sq += (p.data       - prev_p.data     ).norm().item()**2
            group['grad_diff_norm'] = np.sqrt(grad_diff_sq)
            group['param_diff_norm'] = np.sqrt(param_diff_sq)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            wd = group['weight_decay']

            # 1) build global gradient vector ∇f(xᵏ)
            params = []
            grads  = []
            for p in group['params']:
                if p.grad is None:
                    continue
                params.append(p)
                g = p.grad.data
                if wd != 0:
                    g = g.add(p.data, alpha=wd)
                grads.append(g.view(-1))
            if not grads:
                continue
            g_k     = torch.cat(grads)
            norm_g  = g_k.norm()
            s_k     = torch.asinh(norm_g) / (norm_g + 1e-12)
            scaled_g = s_k * g_k

            # 2) get or init optimizer state
            opt_state = self.state.setdefault(group_idx, {})
            if 'gamma' not in opt_state:
                x_vec = torch.cat([p.data.view(-1) for p in params])
                opt_state.update({
                    'gamma': group['lr'],
                    'gamma_prev': group['lr'],
                    'scaled_g_old': scaled_g.clone(),
                    's_old': s_k.clone(),
                    'x_old': x_vec.clone(),
                    'norm_g_old': norm_g.item(),
                })
                # first update (Eq.13)
                with torch.no_grad():
                    γ = opt_state['gamma']
                    idx = 0
                    for p in params:
                        numel = p.numel()
                        seg = scaled_g[idx:idx + numel].view_as(p)
                        p.data.add_(seg, alpha=-γ)
                        idx += numel
                continue

            # 3) load old state BEFORE update
            γ_k = opt_state['gamma']
            γ_km1 = opt_state['gamma_prev']
            s_old = opt_state['s_old']
            scaled_g_old = opt_state['scaled_g_old']
            x_old = opt_state['x_old']
            norm_g_old = opt_state['norm_g_old']

            # 4) compute local curvature L_k (Eq.11) with clamp_min
            x_vec = torch.cat([p.data.view(-1) for p in params])
            dg = scaled_g - scaled_g_old
            dx = x_vec - x_old
            eps = 1e-6
            eps = 1e-6
            norm_dx = dx.norm()
            norm_dg = dg.norm()

            if norm_dx.item() < eps or norm_dg.item() < eps:
                L_k = 1.0 / (2 * eps)  # βάζουμε ένα fixed μεγάλο L_k για να είναι μικρό το βήμα
            else:
                L_k = norm_dg / norm_dx

            # 5) adaptive rule for γₖ₊₁ (Eq.12) with minimum gamma
            τ = γ_k * (s_old / s_k) * (norm_g.item() / norm_g_old) * (1 + γ_k / γ_km1)
            gamma_min = 1e-6
            # πρώτα παίρνουμε το μικρότερο ανάμεσα σε τ και 1/(2L_k)
            γ_new = min(τ, 1.0 / (2 * L_k))
            # μετά εξασφαλίζουμε ότι δεν πέφτει ποτέ κάτω από gamma_min
            γ_new = max(γ_new, gamma_min)
            # 6) apply update (Eq.13)
            with torch.no_grad():
                idx = 0
                for p in params:
                    numel = p.numel()
                    seg = scaled_g[idx:idx + numel].view_as(p)
                    p.data.add_(seg, alpha=-γ_new)
                    idx += numel

            # 7) update state AFTER update
            opt_state.update({
                'gamma_prev': γ_k,
                'gamma': γ_new,
                'scaled_g_old': scaled_g.clone(),
                's_old': s_k.clone(),
                'x_old': x_vec.clone(),
                'norm_g_old': norm_g.item(),
            })
            # mirror γ_new into group['lr'] for logging
            group['lr'] = γ_new

        return loss
