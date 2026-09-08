"""Loan-originations, interest, and credit-impairment rules for Deep ALM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
class LoanCohortState:
    """Principal schedules and irrevocably fixed monthly coupons by cohort."""

    principal_cash_flows: torch.Tensor
    monthly_coupon_rates: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.principal_cash_flows.ndim != 3
            or self.principal_cash_flows.shape[2] != 180
            or self.monthly_coupon_rates.shape
            != self.principal_cash_flows.shape[:2]
        ):
            raise LoanDynamicsError(
                "Loan cohorts require principal [paths, cohorts, 180] and rates [paths, cohorts]"
            )
        if (
            not torch.isfinite(self.principal_cash_flows).all()
            or not torch.isfinite(self.monthly_coupon_rates).all()
            or torch.any(self.principal_cash_flows < 0)
            or torch.any(self.monthly_coupon_rates < 0)
        ):
            raise LoanDynamicsError(
                "Loan cohort principal and coupon rates must be finite and non-negative"
            )

    @classmethod
    def empty(
        cls,
        *,
        paths: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> LoanCohortState:
        return cls(
            principal_cash_flows=torch.zeros(
                (paths, 0, 180), device=device, dtype=dtype
            ),
            monthly_coupon_rates=torch.zeros((paths, 0), device=device, dtype=dtype),
        )

    @property
    def paths(self) -> int:
        return int(self.principal_cash_flows.shape[0])

    @property
    def outstanding_principal(self) -> torch.Tensor:
        return self.principal_cash_flows.sum(dim=(1, 2))

    @property
    def matured_principal(self) -> torch.Tensor:
        return self.principal_cash_flows[:, :, 0].sum(dim=1)

    @property
    def cash_flows(self) -> torch.Tensor:
        outstanding = torch.flip(
            torch.cumsum(
                torch.flip(self.principal_cash_flows, dims=(2,)), dim=2
            ),
            dims=(2,),
        )
        return self.principal_cash_flows.sum(dim=1) + (
            outstanding * self.monthly_coupon_rates.unsqueeze(2)
        ).sum(dim=1)

    @property
    def settled_interest(self) -> torch.Tensor:
        return self.cash_flows[:, 0] - self.matured_principal

    def roll_forward(self) -> LoanCohortState:
        return LoanCohortState(
            principal_cash_flows=torch.nn.functional.pad(
                self.principal_cash_flows[:, :, 1:], (0, 1)
            ),
            monthly_coupon_rates=self.monthly_coupon_rates,
        )

    def originate(
        self,
        principal: torch.Tensor,
        monthly_coupon_rate: torch.Tensor,
        *,
        terms: tuple[int, ...],
        weights: tuple[float, ...],
    ) -> LoanCohortState:
        if principal.shape != (self.paths,) or monthly_coupon_rate.shape != (
            self.paths,
        ):
            raise LoanDynamicsError(
                "Loan origination amounts and rates must have one value per path"
            )
        schedule = _principal_schedule(principal, terms, weights)
        return LoanCohortState(
            principal_cash_flows=torch.cat(
                (self.principal_cash_flows, schedule.unsqueeze(1)), dim=1
            ),
            monthly_coupon_rates=torch.cat(
                (self.monthly_coupon_rates, monthly_coupon_rate.unsqueeze(1)), dim=1
            ),
        )

    def impair(self, factor: torch.Tensor) -> LoanCohortState:
        if factor.shape != (self.paths,):
            raise LoanDynamicsError("Loan impairment must have one value per path")
        return LoanCohortState(
            principal_cash_flows=self.principal_cash_flows
            * (1.0 - factor[:, None, None]),
            monthly_coupon_rates=self.monthly_coupon_rates,
        )


@dataclass(frozen=True)
class LoanTransition:
    """Cash and ladder updates for one post-settlement monthly loan event."""

    mortgage_cohorts: LoanCohortState
    enterprise_cohorts: LoanCohortState
    mortgages: torch.Tensor
    enterprise_loans: torch.Tensor
    originations: torch.Tensor
    interest_cash_flow: torch.Tensor
    impairment_factor: torch.Tensor


DEFAULT_LOAN_CONFIGURATION = LoanConfiguration()


def initial_loan_cohort_state(
    cohorts: object,
    *,
    paths: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> LoanCohortState:
    """Create one independent cohort state per simulated path from a snapshot."""

    try:
        principal = torch.tensor(
            np.stack([cohort.principal_cash_flows for cohort in cohorts]),
            device=device,
            dtype=dtype,
        )
        rates = torch.tensor(
            [cohort.monthly_coupon_rate for cohort in cohorts],
            device=device,
            dtype=dtype,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise LoanDynamicsError("Reference Bank fixed-rate loan cohorts are invalid") from error
    return LoanCohortState(
        principal_cash_flows=principal.unsqueeze(0).expand(paths, -1, -1).clone(),
        monthly_coupon_rates=rates.unsqueeze(0).expand(paths, -1).clone(),
    )


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
    mortgages: LoanCohortState,
    enterprise_loans: LoanCohortState,
    *,
    six_month_yield: torch.Tensor,
    six_month_yield_one_year_ago: torch.Tensor | None,
    convention: str,
    annual_close: bool,
    configuration: LoanConfiguration = DEFAULT_LOAN_CONFIGURATION,
) -> LoanTransition:
    """Settle fixed coupons, replace principal, and lock rates on new loans."""

    _validate_configuration(configuration)
    if mortgages.paths != enterprise_loans.paths:
        raise LoanDynamicsError("Mortgage and enterprise-loan path counts differ")
    if six_month_yield.shape != (mortgages.paths,):
        raise LoanDynamicsError("Six-month yield must provide one value per path")
    _validate_finite("six_month_yield", six_month_yield)
    interest = mortgages.settled_interest + enterprise_loans.settled_interest
    growth = (
        (mortgages.outstanding_principal + enterprise_loans.outstanding_principal)
        * configuration.annual_growth / 12.0
    )
    mortgage_originations = (
        mortgages.matured_principal + growth * configuration.mortgage_growth_share
    )
    enterprise_originations = (
        enterprise_loans.matured_principal
        + growth * configuration.enterprise_growth_share
    )
    coupon = monthly_loan_interest_rate(
        six_month_yield, spread=configuration.spread, convention=convention
    )
    new_mortgages = mortgages.roll_forward().originate(
        mortgage_originations,
        coupon,
        terms=configuration.mortgage_terms_months,
        weights=configuration.mortgage_weights,
    )
    new_enterprise = enterprise_loans.roll_forward().originate(
        enterprise_originations,
        coupon,
        terms=configuration.enterprise_terms_months,
        weights=configuration.enterprise_weights,
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
        new_enterprise = new_enterprise.impair(impairment)

    _validate_finite("loan_interest_cash_flow", interest)
    return LoanTransition(
        mortgage_cohorts=new_mortgages,
        enterprise_cohorts=new_enterprise,
        mortgages=new_mortgages.cash_flows,
        enterprise_loans=new_enterprise.cash_flows,
        originations=mortgage_originations + enterprise_originations,
        interest_cash_flow=interest,
        impairment_factor=impairment,
    )


def _principal_schedule(
    principal: torch.Tensor,
    terms: tuple[int, ...],
    weights: tuple[float, ...],
) -> torch.Tensor:
    result = torch.zeros(
        (principal.shape[0], 180), device=principal.device, dtype=principal.dtype
    )
    for term, weight in zip(terms, weights, strict=True):
        result[:, term - 1] = result[:, term - 1] + principal * weight
    return result


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


def _validate_finite(name: str, values: torch.Tensor) -> None:
    if not torch.isfinite(values).all():
        raise LoanDynamicsError(f"{name} must be finite")


def _validate_convention(convention: str) -> None:
    if convention not in {"paper", "corrected"}:
        raise LoanDynamicsError("Loan convention must be paper or corrected")
