"""Loan-originations, interest, and credit-impairment rules for Deep ALM."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from deepalm.runner import OperationalRunError


class LoanDynamicsError(OperationalRunError):
    """Raised when a loan transition cannot be constructed or reconciled."""


@dataclass(frozen=True)
class LoanConfiguration:
    """The disclosed/reference-case rules for loans, in decimal and mCHF units."""

    spread: float = 0.015
    annual_growth: float = 0.03
    mortgage_terms_months: tuple[int, ...] = tuple(range(24, 145, 12))
    mortgage_weights: tuple[float, ...] = (0.06,) * 8 + (0.40,) + (0.06,) * 2
    enterprise_terms_months: tuple[int, ...] = (1, 2, 3)
    enterprise_weights: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)
    mortgage_growth_share: float = 55 / 75
    enterprise_growth_share: float = 20 / 75


@dataclass(frozen=True)
class LoanTransition:
    """Cash and ladder updates for one post-settlement monthly loan event."""

    mortgages: torch.Tensor
    enterprise_loans: torch.Tensor
    originations: torch.Tensor
    interest_cash_flow: torch.Tensor
    impairment_factor: torch.Tensor


DEFAULT_LOAN_CONFIGURATION = LoanConfiguration()


def monthly_loan_interest_rate(
    six_month_yield: torch.Tensor, *, spread: float, convention: str
) -> torch.Tensor:
    """Return the locked Paper or Corrected loan-interest expression.

    The Paper expression is literally annual-effective and therefore is not
    divided by twelve.  Corrected divides the continuous rate before applying
    the exponential.  This is the sole loan-interest convention difference.
    """

    _validate_convention(convention)
    _validate_finite("six_month_yield", six_month_yield)
    if not torch.isfinite(torch.tensor(spread)):
        raise LoanDynamicsError("Loan spread must be finite")
    exponent = six_month_yield + spread
    if convention == "corrected":
        exponent = exponent / 12.0
    result = torch.clamp_min(torch.exp(exponent) - 1.0, 0.0)
    _validate_finite("loan_interest_rate", result)
    return result


def apply_loan_transition(
    mortgages: torch.Tensor,
    enterprise_loans: torch.Tensor,
    *,
    six_month_yield: torch.Tensor,
    six_month_yield_one_year_ago: torch.Tensor | None,
    initial_total_loan_value: float,
    convention: str,
    annual_close: bool,
    configuration: LoanConfiguration = DEFAULT_LOAN_CONFIGURATION,
) -> LoanTransition:
    """Replace maturing loans, add deterministic growth, interest, and impairment."""

    _validate_configuration(configuration)
    _validate_ladder("mortgages", mortgages)
    _validate_ladder("enterprise_loans", enterprise_loans)
    if mortgages.shape[0] != enterprise_loans.shape[0]:
        raise LoanDynamicsError("Mortgage and enterprise-loan path counts differ")
    if six_month_yield.shape != (mortgages.shape[0],):
        raise LoanDynamicsError("Six-month yield must provide one value per path")
    _validate_finite("six_month_yield", six_month_yield)
    if initial_total_loan_value <= 0:
        raise LoanDynamicsError("Initial total loan value must be positive")

    growth = initial_total_loan_value * configuration.annual_growth / 12.0
    mortgage_originations = mortgages[:, 0] + growth * configuration.mortgage_growth_share
    enterprise_originations = (
        enterprise_loans[:, 0] + growth * configuration.enterprise_growth_share
    )
    shifted_mortgages = _shift_left(mortgages)
    shifted_enterprise = _shift_left(enterprise_loans)
    new_mortgages = _add_bullet_originations(
        shifted_mortgages,
        mortgage_originations,
        configuration.mortgage_terms_months,
        configuration.mortgage_weights,
    )
    new_enterprise = _add_bullet_originations(
        shifted_enterprise,
        enterprise_originations,
        configuration.enterprise_terms_months,
        configuration.enterprise_weights,
    )

    impairment = torch.zeros_like(enterprise_originations)
    if annual_close:
        if six_month_yield_one_year_ago is None:
            raise LoanDynamicsError("Annual loan impairment requires one-year-ago yield")
        if six_month_yield_one_year_ago.shape != six_month_yield.shape:
            raise LoanDynamicsError("One-year-ago yield shape does not match paths")
        _validate_finite("six_month_yield_one_year_ago", six_month_yield_one_year_ago)
        impairment = torch.clamp_min(
            six_month_yield - six_month_yield_one_year_ago - 0.02, 0.0
        )
        new_enterprise = new_enterprise * (1.0 - impairment.unsqueeze(1))

    interest_rate = monthly_loan_interest_rate(
        six_month_yield, spread=configuration.spread, convention=convention
    )
    interest_cash_flow = interest_rate * (
        new_mortgages.sum(dim=1) + new_enterprise.sum(dim=1)
    )
    _validate_finite("loan_interest_cash_flow", interest_cash_flow)
    return LoanTransition(
        mortgages=new_mortgages,
        enterprise_loans=new_enterprise,
        originations=mortgage_originations + enterprise_originations,
        interest_cash_flow=interest_cash_flow,
        impairment_factor=impairment,
    )


def _add_bullet_originations(
    ladder: torch.Tensor,
    originations: torch.Tensor,
    terms: tuple[int, ...],
    weights: tuple[float, ...],
) -> torch.Tensor:
    result = ladder.clone()
    for term, weight in zip(terms, weights, strict=True):
        result[:, term - 1] = result[:, term - 1] + originations * weight
    return result


def _shift_left(ladder: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.pad(ladder[:, 1:], (0, 1))


def _validate_configuration(configuration: LoanConfiguration) -> None:
    if configuration.annual_growth < 0 or not torch.isfinite(
        torch.tensor(configuration.annual_growth)
    ):
        raise LoanDynamicsError("Loan annual growth must be finite and non-negative")
    for name, terms, weights in (
        ("mortgage", configuration.mortgage_terms_months, configuration.mortgage_weights),
        ("enterprise", configuration.enterprise_terms_months, configuration.enterprise_weights),
    ):
        if (
            not terms
            or len(terms) != len(weights)
            or any(term < 1 or term > 180 for term in terms)
            or any(weight < 0 for weight in weights)
            or abs(sum(weights) - 1.0) > 1e-12
        ):
            raise LoanDynamicsError(f"Invalid {name} loan maturity distribution")
    if abs(configuration.mortgage_growth_share + configuration.enterprise_growth_share - 1.0) > 1e-12:
        raise LoanDynamicsError("Loan growth shares must sum to one")


def _validate_ladder(name: str, ladder: torch.Tensor) -> None:
    if ladder.ndim != 2 or ladder.shape[1] != 180:
        raise LoanDynamicsError(f"{name} ladder must have shape [paths, 180]")
    if torch.any(ladder < 0):
        raise LoanDynamicsError(f"{name} ladder cannot contain negative cash flows")
    _validate_finite(name, ladder)


def _validate_finite(name: str, values: torch.Tensor) -> None:
    if not torch.isfinite(values).all():
        raise LoanDynamicsError(f"{name} must be finite")


def _validate_convention(convention: str) -> None:
    if convention not in {"paper", "corrected"}:
        raise LoanDynamicsError("Loan convention must be paper or corrected")
