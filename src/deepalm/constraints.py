"""Differentiable regulatory constraint formulas for economic-value bank states."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from deepalm.runner import OperationalRunError

CONSTRAINT_NAMES = ("lcr", "nsfr", "cmr", "equity_rwa", "irs", "eyr")
PENALTY_COEFFICIENTS = (1.0, 0.2, 1.0, 2.5, 2.0, 0.002)


class ConstraintError(OperationalRunError):
    """Raised when a state cannot be evaluated against the constraint contract."""


@dataclass(frozen=True)
class ConstraintState:
    """Six economic-value ladders and cash at one decision state."""

    cash: torch.Tensor
    investments: torch.Tensor
    mortgages: torch.Tensor
    enterprise_loans: torch.Tensor
    non_maturity_deposits: torch.Tensor
    term_deposits: torch.Tensor
    funding: torch.Tensor
    discounts: torch.Tensor


@dataclass(frozen=True)
class ConstraintValues:
    """Constraint quantities and nonlinear breach magnitudes for one state."""

    values: torch.Tensor
    violations: torch.Tensor
    annual_mask: torch.Tensor


def evaluate_constraints(
    state: ConstraintState,
    *,
    previous_annual_equity: torch.Tensor | None = None,
) -> ConstraintValues:
    """Evaluate all six ratios at a post-restructuring decision state."""

    _validate_state(state)
    values = _portfolio_values(state)
    investments, mortgages, enterprise_loans, nmd, term_deposits, funding = values
    equity = state.cash + investments + mortgages + enterprise_loans - nmd - term_deposits - funding

    risk_weighted_assets = 0.10 * investments + 0.35 * mortgages + enterprise_loans
    hqla = 0.71 * state.cash + 0.89 * investments
    net_outflow = 0.176 * nmd + 0.13 * term_deposits + 0.01 * funding
    available_stable_funding = 0.95 * nmd + 0.90 * term_deposits + 0.60 * funding + equity
    required_stable_funding = 0.12 * investments + 0.71 * (mortgages + enterprise_loans)
    one_month_yield = -12.0 * torch.log(state.discounts[:, 0])
    funding_eligible_share = torch.where(
        one_month_yield >= 0,
        torch.full_like(one_month_yield, 0.20),
        torch.ones_like(one_month_yield),
    )
    minimum_reserves = 0.025 * (nmd + funding_eligible_share * term_deposits)

    result = torch.stack(
        (
            hqla / _positive_denominator("thirty-day net outflow", net_outflow),
            available_stable_funding
            / _positive_denominator("required stable funding", required_stable_funding),
            state.cash / _positive_denominator("minimum reserves", minimum_reserves),
            equity / _positive_denominator("risk-weighted assets", risk_weighted_assets),
            _interest_rate_sensitivity(state, equity),
            _equity_yield(equity, previous_annual_equity),
        ),
        dim=1,
    )
    annual_mask = torch.full_like(equity, previous_annual_equity is not None, dtype=torch.bool)
    return ConstraintValues(
        values=result,
        violations=constraint_violations(result, annual_mask=annual_mask),
        annual_mask=annual_mask,
    )


def constraint_violations(
    values: torch.Tensor, *, annual_mask: torch.Tensor
) -> torch.Tensor:
    """Return paper-style squared breaches, with EYR active only at annual closes."""

    if values.ndim != 2 or values.shape[1] != len(CONSTRAINT_NAMES):
        raise ConstraintError("Constraint values must have shape [paths, 6]")
    if annual_mask.shape != values.shape[:1] or annual_mask.dtype != torch.bool:
        raise ConstraintError("Annual mask must be boolean with shape [paths]")
    if not torch.isfinite(values).all():
        raise ConstraintError("Constraint values must be finite")
    lower_bounds = values.new_tensor((1.05, 1.05, 1.00, 0.17, 0.0, 0.0))
    shortfall = torch.clamp_min(lower_bounds - values, 0.0)
    shortfall[:, 4] = torch.clamp_min(values[:, 4] - 0.085, 0.0)
    shortfall[:, 5] = torch.where(annual_mask, shortfall[:, 5], torch.zeros_like(shortfall[:, 5]))
    return (1.0 + shortfall).square() - 1.0


def constraint_penalty(violations: torch.Tensor) -> torch.Tensor:
    """Aggregate weighted violations over time before applying the outer square."""

    if violations.ndim not in {2, 3} or violations.shape[-1] != len(CONSTRAINT_NAMES):
        raise ConstraintError("Constraint violations must have shape [paths, 6] or [paths, time, 6]")
    if not torch.isfinite(violations).all() or torch.any(violations < 0):
        raise ConstraintError("Constraint violations must be finite and non-negative")
    weighted = violations * violations.new_tensor(PENALTY_COEFFICIENTS)
    cumulative = weighted.sum(dim=tuple(range(1, weighted.ndim)))
    return (1.0 + cumulative).square() - 1.0


def _portfolio_values(state: ConstraintState) -> tuple[torch.Tensor, ...]:
    return tuple(
        (ladder * state.discounts).sum(dim=1)
        for ladder in (
            state.investments,
            state.mortgages,
            state.enterprise_loans,
            state.non_maturity_deposits,
            state.term_deposits,
            state.funding,
        )
    )


def _interest_rate_sensitivity(state: ConstraintState, equity: torch.Tensor) -> torch.Tensor:
    months = torch.arange(1, 181, dtype=state.discounts.dtype, device=state.discounts.device) / 12.0
    up_discounts = state.discounts * torch.exp(-0.01 * months)
    down_discounts = state.discounts * torch.exp(0.01 * months)
    shifted_equities = []
    for discounts in (up_discounts, down_discounts):
        asset_value = sum(
            (ladder * discounts).sum(dim=1)
            for ladder in (state.investments, state.mortgages, state.enterprise_loans)
        )
        liability_value = sum(
            (ladder * discounts).sum(dim=1)
            for ladder in (
                state.non_maturity_deposits,
                state.term_deposits,
                state.funding,
            )
        )
        shifted_equities.append(state.cash + asset_value - liability_value)
    sensitivity = torch.maximum(
        (shifted_equities[0] - equity).abs(), (shifted_equities[1] - equity).abs()
    )
    return sensitivity / _equity_scale(equity)


def _equity_yield(
    equity: torch.Tensor, previous_annual_equity: torch.Tensor | None
) -> torch.Tensor:
    if previous_annual_equity is None:
        return torch.zeros_like(equity)
    if previous_annual_equity.shape != equity.shape or not torch.isfinite(previous_annual_equity).all():
        raise ConstraintError("Previous annual equity must provide one finite value per path")
    if torch.any(previous_annual_equity == 0):
        raise ConstraintError("Previous annual equity cannot be zero for EYR")
    return (equity - previous_annual_equity - 6.0) / previous_annual_equity


def _validate_state(state: ConstraintState) -> None:
    if state.cash.ndim != 1 or state.cash.shape[0] == 0:
        raise ConstraintError("Constraint cash must have shape [positive paths]")
    ladders = (
        state.investments,
        state.mortgages,
        state.enterprise_loans,
        state.non_maturity_deposits,
        state.term_deposits,
        state.funding,
        state.discounts,
    )
    expected_shape = (state.cash.shape[0], 180)
    if any(ladder.shape != expected_shape for ladder in ladders):
        raise ConstraintError("Constraint ladders and discounts must share shape [paths, 180]")
    if not torch.isfinite(state.cash).all() or any(not torch.isfinite(ladder).all() for ladder in ladders):
        raise ConstraintError("Constraint state must be finite")
    if torch.any(state.discounts <= 0):
        raise ConstraintError("Constraint discounts must be positive")


def _require_positive(name: str, values: torch.Tensor) -> None:
    if torch.any(values <= 0):
        raise ConstraintError(f"{name} must be positive for constraint evaluation")


def _positive_denominator(name: str, values: torch.Tensor) -> torch.Tensor:
    _require_positive(name, values)
    return values


def _equity_scale(equity: torch.Tensor) -> torch.Tensor:
    """Keep a failed-capital path finite without altering its equity state."""

    return equity.abs().clamp_min(torch.finfo(equity.dtype).eps)
