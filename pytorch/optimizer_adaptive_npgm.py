import torch
import numpy as np
from torch.optim.optimizer import Optimizer


class AdaptiveNPGM(Optimizer):
    def __init__(self, params, lr=1e-2, weight_decay=0, epsilon=1e-12):
        if lr < 0.0:
            raise ValueError(f"Invalid initial learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, weight_decay=weight_decay, epsilon=epsilon)
        super().__init__(params, defaults)

    def compute_dif_norms(self, prev_optimizer):
        # Η μέθοδος υπάρχει για συμβατότητα, αλλά δεν χρησιμοποιείται πλέον.
        pass

    def step(self, closure=None):
        loss = closure() if closure else None

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            epsilon = group['epsilon']

            params_with_grad = []
            grads = []
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                params_with_grad.append(p)
                grads.append(grad.view(-1))

            if not grads:
                continue

            g_k = torch.cat(grads)
            norm_g = g_k.norm()
            s_k = torch.asinh(norm_g) / (norm_g + epsilon)
            scaled_g = s_k * g_k

            state = self.state.setdefault('global_state', {})

            if len(state) == 0:
                gamma = group['lr']
                state.update({
                    'gamma': gamma,
                    'gamma_prev': gamma,
                    'scaled_g_old': scaled_g.clone(),
                    's_old': s_k,
                    'norm_g_old': norm_g.item(),
                    'x_old': torch.cat([p.data.view(-1) for p in params_with_grad]).clone(),
                    'norm_g_hist': [norm_g.item()]  # ✅ Προσθέτεις αυτό
                })

                with torch.no_grad():
                    idx = 0
                    for p in params_with_grad:
                        numel = p.numel()
                        grad_segment = scaled_g[idx:idx + numel].view_as(p)
                        p.data.add_(grad_segment, alpha=-gamma)
                        idx += numel
                continue

            gamma_k = state['gamma']
            gamma_km1 = state['gamma_prev']
            s_old = state['s_old']
            scaled_g_old = state['scaled_g_old']
            norm_g_old = state['norm_g_old']
            x_old = state['x_old']

            x_k = torch.cat([p.data.view(-1) for p in params_with_grad])

            delta_g = scaled_g - scaled_g_old
            delta_x = x_k - x_old
            L_k = delta_g.norm() / (delta_x.norm() + epsilon)

            # Διορθωμένος υπολογισμός arsinh_ratio
            # ✅ Ενημερώνουμε ιστορικό με την τρέχουσα norm_g
            hist = state.get('norm_g_hist', [])
            hist.append(norm_g.item())
            if len(hist) > 4:
                hist.pop(0)
            state['norm_g_hist'] = hist

            # ✅ Υπολογισμός arsinh_ratio με βάση το ιστορικό
            if len(hist) == 4:
                num = torch.asinh(torch.tensor(hist[1])) * torch.asinh(torch.tensor(hist[3]))
                denom = torch.asinh(torch.tensor(hist[0])) * torch.asinh(torch.tensor(hist[2]))
                arsinh_ratio = (num / (denom + epsilon)).item()
            else:
                arsinh_ratio = (torch.asinh(norm_g) / torch.asinh(torch.tensor(norm_g_old))).item()
            tau = gamma_k * arsinh_ratio
            gamma_min = 1e-6
            gamma_new = max(min(tau, 1.0 / (2 * L_k)), gamma_min)

            with torch.no_grad():
                idx = 0
                for p in params_with_grad:
                    numel = p.numel()
                    grad_segment = scaled_g[idx:idx + numel].view_as(p)
                    p.data.add_(grad_segment, alpha=-gamma_new)
                    idx += numel

            state.update({
                'gamma_prev': gamma_k,
                'gamma': gamma_new,
                'scaled_g_old': scaled_g.clone(),
                's_old': s_k,
                'norm_g_old': norm_g.item(),
                'x_old': x_k.clone(),
            })

            group['lr'] = gamma_new

        return loss