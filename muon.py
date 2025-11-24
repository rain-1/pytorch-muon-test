from click import group
import torch
from torch.optim import Optimizer

import torch

def newton_schulz_5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Performs 5-step Newton-Schulz iteration on a non-square matrix G 
    to approximate its orthogonal factor.
    """
    # Empirically derived coefficients for the 5-step polynomial (NS5)
    # The original paper uses: a=3.4445, b=-4.7750, c=2.0315
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # 1. Start with a clone of the gradient/momentum (X)
    X = G.clone()
    
    # 2. Normalize the matrix by its Frobenius norm
    X_norm = X.norm(p='fro')
    X = X / (X_norm + eps)
    
    # 3. Transpose if 'tall' to ensure X @ X.T is the smaller dimension (Computational efficiency)
    transpose_needed = X.size(0) > X.size(1)
    if transpose_needed:
        X = X.T

    # NS Iteration Loop (5 steps)
    for _ in range(steps):
        # A is the square matrix: X @ X.T
        A = X @ X.T 
        # B applies the b and c polynomial coefficients
        B = b * A + c * (A @ A)
        # Final update step
        X = a * X + B @ X
    
    # 4. Transpose back if the initial matrix was tall
    if transpose_needed:
        X = X.T
        
    return X

class Muon(Optimizer):
    """Skeleton optimizer; fill in your custom logic."""

    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.0):
        # Flatten param groups the same way most torch optimizers do
        defaults = {"lr": lr, "weight_decay": weight_decay}
        # super().__init__(params, defaults)

        plain, geom = [], []
        for p in params:
            (geom if p.ndim == 2 else plain).append(p)
        self.geom_params = geom
        self.adamw = torch.optim.AdamW(
            plain, lr=lr, weight_decay=weight_decay
        )
        super().__init__(geom, defaults)

        self.b = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # AdamW for non-2D params
        self.adamw.step()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                #if len(p.grad.shape) != 2:
                    # do normal SVD on non-matrix parameters
                    # if wd != 0:
                    #     grad = grad.add(p, alpha=wd)
                    #     p.add_(grad, alpha=-lr)
                #    continue
                #print(grad.shape)
                # orthogonalize grad using SVG
                muongrad = newton_schulz_5(grad)
                p.add_(muongrad, alpha=-lr)
       
        
        return loss


        raise NotImplementedError("Implement Muon.step()")

    def zero_grad(self, set_to_none: bool = False):
        self.adamw.zero_grad(set_to_none=set_to_none)
        return super().zero_grad(set_to_none=set_to_none)
