# pytorch/optimizer_adgrad_nesterov2.py

import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required

class AdsgdAdGradNesterov2(Optimizer):
    """
    AdGrad2 + Nesterov momentum, με εξωτερικό hook set_tau2.
    """
    def __init__(self, params, lr=1e-3, weight_decay=0.0,
                 tau_rule='original', momentum=0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay,
                        tau_rule=tau_rule, momentum=momentum)
        super().__init__(params, defaults)

    def set_tau2(self, tau2: float):
        """
        Public hook: αποθηκεύει το override για τ₂ στο state, χωρίς να αλλάζει
        αμέσως group['lr'].
        """
        for group in self.param_groups:
            st = self.state.setdefault(id(group), {})
            st['tau2_override'] = tau2

    def step(self, closure=None):
        """
        Μία βήμα βελτιστοποίησης που συνδυάζει ad_grad learning‐rate και
        Nesterov momentum.
        """
        loss = None
        if closure:
            loss = closure()

        for group in self.param_groups:
            state = self.state.setdefault(id(group), {})

            # α) αύξηση iter count
            k = state.get('k', 0) + 1
            state['k'] = k

            la_old = state.get('la_old', group['lr'])
            rule   = group['tau_rule']
            mu     = group['momentum']

            # β) υπολογισμός τ1
            if rule == 'mod':
                tau1 = k / (k + 3) * la_old
            else:
                tau1 = (k + 1) / k * la_old

            # γ) τ2: είτε override είτε fallback σε τ1
            tau2 = state.get('tau2_override', tau1)

            # νέο βήμα
            la_new = min(tau1, tau2)
            state['la_old'] = la_new

            # δ) προετοιμασία momentum buffer
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = [
                    torch.zeros_like(p.data) for p in group['params']
                ]

            wd = group['weight_decay']
            # ε) ενημέρωση κάθε παραμέτρου
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if wd != 0:
                    d_p = d_p.add(wd, p.data)

                buf = state['momentum_buffer'][idx]
                v_prev = buf.clone()

                # ενημέρωση buffer
                buf.mul_(mu).add_(d_p, alpha=la_new)
                # Nesterov update
                p.data.add_(v_prev, alpha=-mu).add_(d_p, alpha=-la_new)

            # στ) αποθήκευση του νέου lr
            group['lr'] = la_new

        return loss

