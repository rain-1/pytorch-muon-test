import torch
from torch.optim import Optimizer


class Muon(Optimizer):
    """Skeleton optimizer; fill in your custom logic."""

    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.0):
        # Flatten param groups the same way most torch optimizers do
        defaults = {"lr": lr, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Perform a single optimization step.

        TODO: implement your update rule here.
        """
        raise NotImplementedError("Implement Muon.step()")

    def zero_grad(self, set_to_none: bool = False):
        return super().zero_grad(set_to_none=set_to_none)
