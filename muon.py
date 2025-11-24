from click import group
import torch
from torch.optim import Optimizer, AdamW

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
        # Ensure AdamW and Muon receive valid parameter lists
        if not plain:
            plain = [torch.nn.Parameter(torch.zeros(1))]  # Dummy parameter for AdamW
        if not geom:
            geom = [torch.nn.Parameter(torch.zeros(1, 1))]  # Dummy parameter for Muon
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

class CombinedAdamWMuon(Optimizer):
    """Combines AdamW and Muon optimizers."""

    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
        }
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
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            epsilon = group["epsilon"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay (AdamW)
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                
                # Compute step size
                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

                # Update parameters (AdamW + Muon-like step)
                p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add(epsilon), value=-step_size)

        return loss

class CombinedOfficialMuonAdamW(Optimizer):
    """Combines official Muon and AdamW optimizers."""

    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        # Materialize params in case we were given a generator; we need to iterate twice.
        params = list(params)
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
        }
        super().__init__(params, defaults)

        plain, geom = [], []
        for p in params:
            (geom if p.ndim == 2 else plain).append(p)

        # Only construct inner optimizers when there are parameters to optimize
        self.adamw = AdamW(plain, lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), eps=epsilon) if plain else None
        self.muon = torch.optim.Muon(geom, lr=lr, weight_decay=weight_decay) if geom else None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Perform AdamW step
        if self.adamw is not None:
            self.adamw.step()

        # Perform Muon step
        if self.muon is not None:
            self.muon.step()

        return loss
