import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required

class AdsgdAdaptiveNPGM(Optimizer):
    """
    Adaptive SGD optimizer implementing the Adaptive NPGM logic from Algorithm 1.
    Scaling factor sₖ = arsinh(||gₖ||)/||gₖ||, adaptive rule for gammaₖ₊₁.
    """
    def __init__(self, params, lr=1e-2, weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid initial learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(AdsgdAdaptiveNPGM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdsgdAdaptiveNPGM, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('lr', 1e-2)
            group.setdefault('weight_decay', 0)

    def compute_dif_norms(self, prev_optimizer):
        for group, prev_group in zip(self.param_groups, prev_optimizer.param_groups):
            grad_diff_sq = 0.0
            param_diff_sq = 0.0
            for p, prev_p in zip(group['params'], prev_group['params']):
                if p.grad is None or prev_p.grad is None:
                    continue
                grad_diff_sq += (p.grad.data - prev_p.grad.data).norm().item()**2
                param_diff_sq += (p.data - prev_p.data).norm().item()**2
            group['grad_diff_norm'] = np.sqrt(grad_diff_sq)
            group['param_diff_norm'] = np.sqrt(param_diff_sq)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                # 1) Βασικό gradient
                grad = p.grad.data

                # 2) Προσθέτουμε weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # 3) Υπολογίζουμε scaling & scaled_grad με το decayed grad
                norm_g = grad.norm().item()
                scaling = np.arcsinh(norm_g) / norm_g if norm_g > 0 else 0.0
                scaled_grad = scaling * grad

                # Τώρα παίρνουμε το state για αυτό το p
                state = self.state[p]

                # Initialization
                if len(state) == 0:
                    gamma = group['lr']
                    norm_g = grad.norm().item()
                    scaling = np.arcsinh(norm_g) / norm_g if norm_g > 0 else 0.0
                    scaled_grad = scaling * grad

                    state['step'] = 0
                    state['gamma'] = gamma
                    state['gamma_prev'] = gamma
                    state['scaling_old'] = scaling
                    state['scaled_grad_old'] = scaled_grad.clone()
                    state['norm_grad_old'] = norm_g
                    state['x_old'] = p.data.clone()


                    # Parameter update without tracking in autograd
                    with torch.no_grad():
                        p.data.add_(scaled_grad, alpha=-gamma)
                    continue

                # Load history
                gamma = state['gamma']
                gamma_prev = state['gamma_prev']
                scaling_old = state['scaling_old']
                scaled_grad_old = state['scaled_grad_old']
                norm_grad_old = state['norm_grad_old']
                x_old = state['x_old']

                # Compute new scaled gradient
                norm_g = grad.norm().item()
                scaling = np.arcsinh(norm_g) / norm_g if norm_g > 0 else 0.0
                scaled_grad = scaling * grad

                # Estimate curvature
                delta_g = scaled_grad - scaled_grad_old
                delta_x = p.data - x_old
                Lk = delta_g.norm().item() / (delta_x.norm().item() + 1e-12)

                # Adaptive rule for gamma
                tau = gamma * (scaling_old / scaling) * (norm_g / norm_grad_old) * (1 + gamma / gamma_prev)
                gamma_new = min(tau, 1.0 / (2 * Lk))



                # Parameter update with no_grad
                with torch.no_grad():
                    p.data.add_(scaled_grad, alpha=-gamma_new)

                # Update state
                state['step'] += 1
                state['gamma_prev'] = gamma
                state['gamma'] = gamma_new
                state['scaling_old'] = scaling
                state['scaled_grad_old'] = scaled_grad.clone()
                state['norm_grad_old'] = norm_g
                state['x_old'] = p.data.clone()

        return loss
