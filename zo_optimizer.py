"""Zero-order optimizer for CIFAR100 ResNet18 fine-tuning.

Final strategy: tune only the classification head with an antithetic SPSA
estimator and Adam-style moments.  No gradients/backward calls are used.
"""
from __future__ import annotations

from typing import Callable
import math
import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-2,
        eps: float = 3e-3,
        perturbation_mode: str = "uniform",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        if perturbation_mode not in ("gaussian", "uniform"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'uniform', got {perturbation_mode!r}"
            )
        self.perturbation_mode = perturbation_mode

        # The head has only ~51k parameters.  Tuning deeper layers makes the
        # SPSA signal much noisier under the tiny sample budget.
        self.layer_names: list[str] = ["fc.weight", "fc.bias"]

        self.step_idx = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_eps = 1e-8
        self.weight_decay = 2e-4
        self.max_update_rms = 2.5e-2

        # Averaging several antithetic directions reduces SPSA variance.  This
        # is intentionally small enough to keep validation reasonably fast.
        self.n_directions = 6

        self._m: dict[str, torch.Tensor] = {}
        self._v: dict[str, torch.Tensor] = {}

    def _active_params(self) -> dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: {missing}. "
                "Use [n for n, _ in model.named_parameters()] to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        if self.perturbation_mode == "uniform":
            # Rademacher SPSA direction: inverse is itself and no normalization
            # is used, which is the standard SPSA estimator.
            return torch.empty_like(param).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        return torch.randn_like(param)

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        grads = {name: torch.zeros_like(p) for name, p in params.items()}

        # Mild annealing: larger perturbations early, more precise later.
        eps_t = self.eps / math.sqrt(1.0 + 0.03 * self.step_idx)

        with torch.no_grad():
            for _ in range(self.n_directions):
                dirs = {name: self._sample_direction(p) for name, p in params.items()}

                for name, p in params.items():
                    p.add_(eps_t * dirs[name])
                f_plus = loss_fn()

                for name, p in params.items():
                    p.add_(-2.0 * eps_t * dirs[name])
                f_minus = loss_fn()

                for name, p in params.items():
                    p.add_(eps_t * dirs[name])

                coeff = (f_plus - f_minus) / (2.0 * eps_t)
                for name in grads:
                    grads[name].add_(coeff * dirs[name])

            inv = 1.0 / float(self.n_directions)
            for name in grads:
                grads[name].mul_(inv)
                if name.endswith("weight") and self.weight_decay > 0:
                    grads[name].add_(self.weight_decay * params[name])

        return grads

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        self.step_idx += 1
        with torch.no_grad():
            for name, p in params.items():
                g = grads[name]
                if name not in self._m:
                    self._m[name] = torch.zeros_like(p)
                    self._v[name] = torch.zeros_like(p)

                m = self._m[name]
                v = self._v[name]
                m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

                m_hat = m / (1.0 - self.beta1 ** self.step_idx)
                v_hat = v / (1.0 - self.beta2 ** self.step_idx)
                update = self.lr * m_hat / (v_hat.sqrt().add_(self.adam_eps))

                # Clip update RMS per tensor.  This is more stable than clipping
                # the raw SPSA gradient, whose scale varies strongly by batch.
                rms = update.pow(2).mean().sqrt()
                if torch.isfinite(rms) and rms > self.max_update_rms:
                    update.mul_(self.max_update_rms / (rms + 1e-12))

                p.sub_(update)

    def step(self, loss_fn: Callable[[], float]) -> float:
        params = self._active_params()
        with torch.no_grad():
            loss_before = loss_fn()
        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)
        return float(loss_before)



import torch


class ZeroOrderOptimizer:
    def __init__(self, model):
        self.model = model

        # Tuned by validate.py for reporting.
        self.layer_names = ["fc.weight", "fc.bias"]

        self.params = []
        self.names = []
        for name, p in self.model.named_parameters():
            if name in self.layer_names:
                p.requires_grad_(False)
                self.names.append(name)
                self.params.append(p)

        self.step_idx = 0

        # Conservative SPSA hyperparameters.
        self.eps = 1e-3
        self.lr = 2e-2
        self.lr_decay = 0.985
        self.max_update_norm = 0.05

    @torch.no_grad()
    def _sample_directions(self):
        directions = []
        for p in self.params:
            # Rademacher noise: {-1, +1}
            u = torch.empty_like(p)
            u.bernoulli_(0.5)
            u.mul_(2.0).sub_(1.0)
            directions.append(u)
        return directions

    @torch.no_grad()
    def _add_perturbation(self, directions, scale):
        for p, u in zip(self.params, directions):
            p.add_(u, alpha=scale)

    @torch.no_grad()
    def _apply_update(self, directions, grad_scalar, lr):
        # SPSA pseudo-gradient: g * u
        # Clip global update norm for stability.
        sq_norm = torch.zeros((), device=self.params[0].device)
        for u in directions:
            sq_norm += torch.sum((lr * grad_scalar * u) ** 2)

        update_norm = torch.sqrt(sq_norm).clamp_min(1e-12)
        clip = min(1.0, float(self.max_update_norm / update_norm.item()))

        for p, u in zip(self.params, directions):
            p.add_(u, alpha=-lr * grad_scalar * clip)

    @torch.no_grad()
    def step(self, loss_fn):
        """
        Zero-order SPSA step with accept/reject rollback.

        Uses only scalar loss queries:
        1. f(theta)
        2. f(theta + eps*u)
        3. f(theta - eps*u)
        4. f(theta_new) for accepting/rejecting

        No gradients / backward are used.
        """
        self.model.eval()

        lr = self.lr * (self.lr_decay ** self.step_idx)
        eps = self.eps

        # Current loss on the same batch.
        base_loss = float(loss_fn().detach().item())

        directions = self._sample_directions()

        # f(theta + eps*u)
        self._add_perturbation(directions, eps)
        loss_plus = float(loss_fn().detach().item())

        # f(theta - eps*u)
        self._add_perturbation(directions, -2.0 * eps)
        loss_minus = float(loss_fn().detach().item())

        # Restore theta.
        self._add_perturbation(directions, eps)

        grad_scalar = (loss_plus - loss_minus) / (2.0 * eps)

        # Save old parameters for rollback.
        old_params = [p.detach().clone() for p in self.params]

        # Try update.
        self._apply_update(directions, grad_scalar, lr)

        new_loss = float(loss_fn().detach().item())

        # Accept only if current batch loss improves.
        if new_loss <= base_loss:
            accepted_loss = new_loss
        else:
            for p, old in zip(self.params, old_params):
                p.copy_(old)

            # Try a smaller step once.
            small_lr = 0.25 * lr
            self._apply_update(directions, grad_scalar, small_lr)
            small_loss = float(loss_fn().detach().item())

            if small_loss <= base_loss:
                accepted_loss = small_loss
            else:
                for p, old in zip(self.params, old_params):
                    p.copy_(old)
                accepted_loss = base_loss

        self.step_idx += 1

        return torch.tensor(accepted_loss, device=self.params[0].device)