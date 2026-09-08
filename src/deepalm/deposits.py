"""Deterministic deposit rolling, operating costs, cash charges, and dividends."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from deepalm.runner import OperationalRunError


class DepositDynamicsError(OperationalRunError):
    """Raised when deposit or cash-flow inputs are economically invalid."""


@dataclass(frozen=True)
class DepositConfiguration:
    non_maturity_growth: float = 0.04
    term_growth: float = 0.01
    reference_terms_months: tuple[int, ...] = (1, 2, 12, 120)
    non_maturity_weights: tuple[float, ...] = (0.40, 0.30, 0.25, 0.05)
    term_weights: tuple[float, ...] = (0.10, 0.10, 0.50, 0.30)
    personnel_annual_growth: float = 0.02
    dividend_share: float = 0.50
    reserve_ratio: float = 0.025


DEFAULT_DEPOSIT_CONFIGURATION = DepositConfiguration()


@dataclass(frozen=True)
class DepositRates:
    non_maturity: torch.Tensor
    term: torch.Tensor


def deposit_rates(
    six_month_yield_history: torch.Tensor, current_six_month_yield: torch.Tensor
) -> DepositRates:
    """Implement equations 11a--c from the paper on batch-first rate history."""

    if six_month_yield_history.ndim != 2 or six_month_yield_history.shape[1] != 3:
        raise DepositDynamicsError("Deposit-rate history must have shape [paths, 3]")
    if current_six_month_yield.shape != six_month_yield_history.shape[:1]:
        raise DepositDynamicsError("Current six-month yield must have one value per path")
    if not torch.isfinite(six_month_yield_history).all() or not torch.isfinite(current_six_month_yield).all():
        raise DepositDynamicsError("Deposit reference yields must be finite")
    reference = six_month_yield_history.mean(dim=1)
    return DepositRates(
        non_maturity=torch.minimum(0.60 * reference, torch.full_like(reference, 0.03)),
        term=torch.minimum(
            torch.maximum(0.85 * reference, current_six_month_yield - 0.0025),
            torch.full_like(reference, 0.05),
        ),
    )


def monthly_deposit_interest(rate: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(rate).all():
        raise DepositDynamicsError("Deposit rate must be finite")
    return torch.exp(rate / 12.0) - 1.0


def allocate_deposit_tranches(
    amounts: torch.Tensor, terms: tuple[int, ...], weights: tuple[float, ...]
) -> torch.Tensor:
    """Split deposits into equal monthly tranches up to each reference maturity."""

    if amounts.ndim != 1 or not torch.isfinite(amounts).all() or torch.any(amounts < 0):
        raise DepositDynamicsError("Deposit allocation amounts must be finite and non-negative")
    if len(terms) != len(weights) or not terms or abs(sum(weights) - 1.0) > 1e-12:
        raise DepositDynamicsError("Invalid deposit maturity distribution")
    result = torch.zeros((amounts.shape[0], 180), device=amounts.device, dtype=amounts.dtype)
    for term, weight in zip(terms, weights, strict=True):
        if term < 1 or term > 180 or weight < 0:
            raise DepositDynamicsError("Invalid deposit reference maturity")
        result[:, :term] += (amounts * weight / term).unsqueeze(1)
    return result


def operating_cost(
    *, personnel_cost: float, material_cost: float, completed_years: int, configuration: DepositConfiguration = DEFAULT_DEPOSIT_CONFIGURATION
) -> float:
    if personnel_cost < 0 or material_cost < 0 or completed_years < 0:
        raise DepositDynamicsError("Operating-cost inputs must be non-negative")
    return personnel_cost * (1.0 + configuration.personnel_annual_growth) ** completed_years + material_cost


def cash_penalty(cash: torch.Tensor, minimum_reserves: torch.Tensor, one_month_discount: torch.Tensor) -> torch.Tensor:
    if not (cash.shape == minimum_reserves.shape == one_month_discount.shape):
        raise DepositDynamicsError("Cash penalty inputs must share one value per path")
    if not torch.isfinite(cash).all() or not torch.isfinite(minimum_reserves).all() or not torch.isfinite(one_month_discount).all() or torch.any(one_month_discount <= 0):
        raise DepositDynamicsError("Cash penalty inputs must be finite with positive discounts")
    # E-03: this is a non-negative cost, subtracted by the cash ledger.
    return torch.clamp_min(cash - 30.0 * minimum_reserves, 0.0) * (1.0 - torch.minimum(1.0 / one_month_discount, torch.ones_like(one_month_discount)))
