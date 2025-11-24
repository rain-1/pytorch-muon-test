import torch
from torch.optim import Optimizer


class SimpleSGD(Optimizer):
    """Minimal reference SGD optimizer (no momentum)."""

    def __init__(self, params, lr: float = 1e-2, weight_decay: float = 0.0):
        if lr <= 0:
            raise ValueError("Learning rate must be positive.")
        if weight_decay < 0:
            raise ValueError("Weight decay must be non-negative.")
        defaults = {"lr": lr, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)
                p.add_(grad, alpha=-lr)
        return loss

    def zero_grad(self, set_to_none: bool = False):
        return super().zero_grad(set_to_none=set_to_none)
