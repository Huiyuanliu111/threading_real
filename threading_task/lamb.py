"""Small native PyTorch LAMB optimizer used by the original ARP recipe."""
from __future__ import annotations

import torch


class Lamb(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad
                if grad.is_sparse: raise RuntimeError("LAMB does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                mhat = m / (1 - beta1 ** state["step"])
                vhat = v / (1 - beta2 ** state["step"])
                update = mhat / (vhat.sqrt() + group["eps"])
                if group["weight_decay"]: update.add_(p, alpha=group["weight_decay"])
                p_norm, u_norm = p.norm(), update.norm()
                trust = torch.where((p_norm > 0) & (u_norm > 0), p_norm / u_norm,
                                    torch.ones_like(p_norm))
                p.add_(update * trust, alpha=-group["lr"])
        return loss
