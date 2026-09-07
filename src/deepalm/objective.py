"""Training and evaluation losses for the no-swap Deep ALM workflow."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from deepalm.constraints import constraint_penalty
from deepalm.runner import OperationalRunError


class ObjectiveError(OperationalRunError):
    """Raised when objective inputs do not form an auditable path batch."""


@dataclass(frozen=True)
class ObjectiveParameters:
    """One sampled or fixed return target and penalty weight per path."""

    mu: torch.Tensor
    penalty_weight: torch.Tensor

    def __post_init__(self) -> None:
        if self.mu.ndim != 1 or self.penalty_weight.shape != self.mu.shape:
            raise ObjectiveError("Objective parameters must have matching shape [paths]")
        if not torch.isfinite(self.mu).all() or not torch.isfinite(self.penalty_weight).all():
            raise ObjectiveError("Objective parameters must be finite")
        if torch.any(self.penalty_weight <= 0):
            raise ObjectiveError("Objective penalty weight must be positive")


@dataclass(frozen=True)
class ObjectiveResult:
    """Per-path training loss components and evaluation-only CRRA loss."""

    target: torch.Tensor
    penalty: torch.Tensor
    total: torch.Tensor
    crra: torch.Tensor
    gamma: float
    equity_ratio_floor: float


def sample_training_objective_parameters(
    paths: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ObjectiveParameters:
    """Sample the paper's per-path return target and penalty weight ranges."""

    if paths <= 0:
        raise ObjectiveError("Objective sampling requires positive paths")
    mu = 0.02 + 0.05 * torch.rand(
        paths, generator=generator, device=device, dtype=dtype
    )
    penalty_weight = 0.05 + 24.95 * torch.rand(
        paths, generator=generator, device=device, dtype=dtype
    )
    return ObjectiveParameters(mu=mu, penalty_weight=penalty_weight)


def evaluation_objective_parameters(
    paths: int,
    *,
    horizon_years: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ObjectiveParameters:
    """Return the fixed parameters used for selection and locked evaluation."""

    if paths <= 0 or horizon_years not in {5, 15}:
        raise ObjectiveError("Evaluation parameters require positive paths and a 5 or 15 year horizon")
    mu = 0.0406 if horizon_years == 5 else 0.0400
    return ObjectiveParameters(
        mu=torch.full((paths,), mu, device=device, dtype=dtype),
        penalty_weight=torch.full((paths,), 3.5, device=device, dtype=dtype),
    )


def evaluate_objective(
    initial_equity: torch.Tensor,
    terminal_equity: torch.Tensor,
    *,
    horizon_years: int,
    violations: torch.Tensor,
    parameters: ObjectiveParameters,
    gamma: float = 10.0,
    equity_ratio_floor: float = 1e-8,
) -> ObjectiveResult:
    """Return asymmetric training loss and separately reported CRRA loss."""

    if horizon_years not in {5, 15}:
        raise ObjectiveError("Objective horizon must be 5 or 15 years")
    if initial_equity.ndim != 1 or terminal_equity.shape != initial_equity.shape:
        raise ObjectiveError("Initial and terminal equity must have matching shape [paths]")
    if parameters.mu.shape != initial_equity.shape:
        raise ObjectiveError("Objective parameters must match the equity path count")
    if not torch.isfinite(initial_equity).all() or not torch.isfinite(terminal_equity).all():
        raise ObjectiveError("Objective equity values must be finite")
    if torch.any(initial_equity <= 0):
        raise ObjectiveError("Initial equity must be positive")
    if gamma <= 0 or gamma == 1 or equity_ratio_floor <= 0:
        raise ObjectiveError("CRRA gamma and equity-ratio floor must be positive, with gamma not one")

    target_equity = (1.0 + parameters.mu).pow(horizon_years) * initial_equity
    target = torch.clamp_max(terminal_equity - target_equity, 0.0).square()
    penalty = constraint_penalty(violations)
    if penalty.shape != initial_equity.shape:
        raise ObjectiveError("Constraint penalty must provide one value per path")
    total = target + parameters.penalty_weight * penalty
    equity_ratio = torch.clamp_min(terminal_equity / initial_equity, equity_ratio_floor)
    utility = (equity_ratio.pow(1.0 - gamma) - 1.0) / (1.0 - gamma)
    return ObjectiveResult(
        target=target,
        penalty=penalty,
        total=total,
        crra=-utility,
        gamma=gamma,
        equity_ratio_floor=equity_ratio_floor,
    )
