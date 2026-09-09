from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepalm.loans import (
    LoanCohortState,
    LoanConfiguration,
    LoanDynamicsError,
    apply_loan_transition,
    monthly_loan_interest_rate,
)
from deepalm.reference_bank import ReferenceBankProvider
from deepalm.runoff import ALMSimulator
from deepalm.term_structures import MarketScenarioModel

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
)


def _cohort(principal: float, *, monthly_coupon_rate: float = 0.0) -> LoanCohortState:
    schedule = torch.zeros((1, 1, 180), dtype=torch.float64)
    schedule[0, 0, 0] = principal
    return LoanCohortState(
        principal_cash_flows=schedule,
        monthly_coupon_rates=torch.tensor([[monthly_coupon_rate]], dtype=torch.float64),
    )


def test_monthly_loan_interest_rate_has_one_monthly_interpretation() -> None:
    yield_rate = torch.tensor([0.03], dtype=torch.float64)

    monthly_rate = monthly_loan_interest_rate(yield_rate, spread=0.015)

    assert monthly_rate.item() == pytest.approx(np.exp(0.045 / 12.0) - 1.0)


def test_fixed_rate_cohort_keeps_its_coupon_when_market_rate_changes() -> None:
    principal = torch.zeros((1, 1, 180), dtype=torch.float64)
    principal[0, 0, 1] = 100.0
    cohorts = LoanCohortState(
        principal_cash_flows=principal,
        monthly_coupon_rates=torch.tensor([[0.01]], dtype=torch.float64),
    )
    result = apply_loan_transition(
        cohorts,
        LoanCohortState.empty(paths=1, dtype=torch.float64),
        six_month_yield=torch.tensor([0.50], dtype=torch.float64),
        six_month_yield_one_year_ago=None,
        annual_close=False,
        configuration=LoanConfiguration(annual_growth=0.0),
    )
    # The existing cohort pays its locked 1% coupon; 50% market yield is only
    # relevant when a new loan is originated.
    assert result.interest_cash_flow.item() == pytest.approx(1.0)
    assert result.mortgages[0, 0].item() == pytest.approx(101.0)


def test_simulator_keeps_legacy_coupons_fixed_and_locks_new_cohorts() -> None:
    historical = MarketScenarioModel().load_historical_term_structures(SOURCE)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    discounts = np.broadcast_to(historical.initial_curve.discount_factors, (1, 3, 180)).copy()
    base_spots = np.broadcast_to(historical.initial_curve.spot_rates, (1, 3, 180)).copy()
    stressed_spots = base_spots.copy()
    stressed_spots[0, 1, 5] = 0.50
    base = ALMSimulator().rollout(
        snapshot,
        SimpleNamespace(discount_factors=discounts, spot_rates=base_spots, initial_curve_identity=snapshot.initial_curve_identity, as_of_date=snapshot.as_of_date),
        include_loan_dynamics=True,
    )
    stressed = ALMSimulator().rollout(
        snapshot,
        SimpleNamespace(discount_factors=discounts, spot_rates=stressed_spots, initial_curve_identity=snapshot.initial_curve_identity, as_of_date=snapshot.as_of_date),
        include_loan_dynamics=True,
    )
    assert base.loan_interest_cash_flows is not None
    assert stressed.loan_interest_cash_flows is not None
    # The first settlement comes wholly from legacy cohorts, so its coupon is
    # unchanged by the market shock. Later payments include newly locked loans.
    assert stressed.loan_interest_cash_flows[0, 0] == pytest.approx(
        base.loan_interest_cash_flows[0, 0]
    )
    assert stressed.loan_interest_cash_flows[0, 1] > base.loan_interest_cash_flows[0, 1]
    assert stressed.cash_reconciliation_error.abs().max().item() < 1e-8


def test_loan_transition_replaces_maturities_adds_growth_and_impairs_enterprise() -> (
    None
):
    mortgages = _cohort(10.0, monthly_coupon_rate=0.01)
    enterprise = _cohort(3.0, monthly_coupon_rate=0.01)

    result = apply_loan_transition(
        mortgages,
        enterprise,
        six_month_yield=torch.tensor([0.06], dtype=torch.float64),
        six_month_yield_one_year_ago=torch.tensor([0.01], dtype=torch.float64),
        annual_close=True,
    )

    mortgage_originations = 10.0 + 13 * 0.03 / 12 * (55 / 75)
    enterprise_originations = 3.0 + 13 * 0.03 / 12 * (20 / 75)
    assert result.originations.item() == pytest.approx(
        mortgage_originations + enterprise_originations
    )
    assert result.mortgage_cohorts.principal_cash_flows[0, -1, 119].item() == pytest.approx(
        0.4 * mortgage_originations
    )
    assert result.enterprise_cohorts.principal_cash_flows[0, -1, 0].item() == pytest.approx(
        enterprise_originations / 3 * 0.97
    )
    assert result.impairment_factor.item() == pytest.approx(0.03)
    assert result.interest_cash_flow.item() > 0


def test_invalid_loan_distribution_fails_with_domain_error() -> None:
    cohorts = LoanCohortState.empty(paths=1)
    invalid = LoanConfiguration(mortgage_weights=(1.0,))

    with pytest.raises(LoanDynamicsError, match="maturity distribution"):
        apply_loan_transition(
            cohorts,
            cohorts,
            six_month_yield=torch.zeros(1, dtype=torch.float64),
            six_month_yield_one_year_ago=None,
            annual_close=False,
            configuration=invalid,
        )


def test_simulator_reconciles_loan_events_and_applies_annual_impairment() -> None:
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    discounts = np.broadcast_to(
        historical.initial_curve.discount_factors, (1, 14, 180)
    ).copy()
    spots = np.broadcast_to(historical.initial_curve.spot_rates, (1, 14, 180)).copy()
    spots[0, 0, 5] = 0.01
    spots[0, 12, 5] = 0.06

    result = ALMSimulator().rollout(
        snapshot,
        SimpleNamespace(
            discount_factors=discounts,
            spot_rates=spots,
            initial_curve_identity=snapshot.initial_curve_identity,
            as_of_date=snapshot.as_of_date,
        ),
        include_loan_dynamics=True,
    )

    assert result.loan_originations is not None
    assert result.loan_interest_cash_flows is not None
    assert result.enterprise_impairment_factors is not None
    assert result.enterprise_impairment_factors[0, 11].item() == pytest.approx(0.03)
    assert result.loan_originations[0, 0].item() > 0
    assert torch.max(result.cash_reconciliation_error.abs()).item() <= 1e-6
