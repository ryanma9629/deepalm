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
from deepalm.objective import ObjectiveParameters
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


def _flat_curve(rate: float) -> torch.Tensor:
    return torch.full((1, 180), rate, dtype=torch.float64)


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
        spot_curve=_flat_curve(0.50),
        six_month_yield_one_year_ago=None,
        annual_impairment=False,
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
    stressed_spots[0, 1, :] = 0.50
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
        spot_curve=_flat_curve(0.06),
        six_month_yield_one_year_ago=torch.tensor([0.01], dtype=torch.float64),
        annual_impairment=True,
    )

    mortgage_originations = 10.0 + 13 * 0.03 / 12 * (55 / 75)
    enterprise_originations = 3.0 + 13 * 0.03 / 12 * (20 / 75)
    assert result.originations.item() == pytest.approx(
        mortgage_originations + enterprise_originations
    )
    assert result.mortgage_cohorts.outstanding_principal.item() == pytest.approx(
        mortgage_originations
    )
    assert result.enterprise_cohorts.outstanding_principal.item() == pytest.approx(
        enterprise_originations * 0.97
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
            spot_curve=_flat_curve(0.0),
            six_month_yield_one_year_ago=None,
            annual_impairment=False,
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


def test_new_loans_lock_maturity_specific_coupons() -> None:
    mortgages = _cohort(100.0)
    enterprise = LoanCohortState.empty(paths=1, dtype=torch.float64)
    configuration = LoanConfiguration(
        annual_growth=0.0,
        spread=0.0,
        mortgage_terms_months=(2, 4),
        mortgage_weights=(0.25, 0.75),
    )
    desired_monthly_coupons = torch.tensor([0.001, 0.003], dtype=torch.float64)
    curve_at_origination = _flat_curve(0.0)
    curve_at_origination[
        0, tuple(term - 1 for term in configuration.mortgage_terms_months)
    ] = 12.0 * torch.log1p(desired_monthly_coupons)

    originated = apply_loan_transition(
        mortgages,
        enterprise,
        spot_curve=curve_at_origination,
        six_month_yield_one_year_ago=None,
        annual_impairment=False,
        configuration=configuration,
    )
    assert originated.mortgages[0, :4].tolist() == pytest.approx(
        [0.25, 25.25, 0.225, 75.225]
    )
    repriced_curve = _flat_curve(0.50)
    settled = apply_loan_transition(
        originated.mortgage_cohorts,
        originated.enterprise_cohorts,
        spot_curve=repriced_curve,
        six_month_yield_one_year_ago=None,
        annual_impairment=False,
        configuration=configuration,
    )

    assert settled.interest_cash_flow.item() == pytest.approx(0.25)


def test_short_enterprise_loans_lock_each_maturitys_coupon() -> None:
    desired_monthly_coupons = torch.tensor([0.001, 0.002, 0.003], dtype=torch.float64)
    curve_at_origination = _flat_curve(0.0)
    curve_at_origination[0, :3] = 12.0 * torch.log1p(desired_monthly_coupons)
    configuration = LoanConfiguration(
        annual_growth=0.0,
        spread=0.0,
        enterprise_weights=(0.2, 0.3, 0.5),
    )

    originated = apply_loan_transition(
        LoanCohortState.empty(paths=1, dtype=torch.float64),
        _cohort(100.0),
        spot_curve=curve_at_origination,
        six_month_yield_one_year_ago=None,
        annual_impairment=False,
        configuration=configuration,
    )
    assert originated.enterprise_loans[0, :3].tolist() == pytest.approx(
        [20.23, 30.21, 50.15]
    )
    settled = apply_loan_transition(
        originated.mortgage_cohorts,
        originated.enterprise_cohorts,
        spot_curve=_flat_curve(0.50),
        six_month_yield_one_year_ago=None,
        annual_impairment=False,
        configuration=configuration,
    )

    assert settled.interest_cash_flow.item() == pytest.approx(0.23)


def test_enterprise_impairment_reduces_the_next_interest_cash_flow() -> None:
    principal = torch.zeros((1, 1, 180), dtype=torch.float64)
    principal[0, 0, 1] = 100.0
    enterprise = LoanCohortState(
        principal_cash_flows=principal,
        monthly_coupon_rates=torch.tensor([[0.01]], dtype=torch.float64),
    )
    configuration = LoanConfiguration(annual_growth=0.0)

    impaired = apply_loan_transition(
        LoanCohortState.empty(paths=1, dtype=torch.float64),
        enterprise,
        spot_curve=_flat_curve(0.06),
        six_month_yield_one_year_ago=torch.tensor([0.01], dtype=torch.float64),
        annual_impairment=True,
        configuration=configuration,
    )
    unimpaired = apply_loan_transition(
        LoanCohortState.empty(paths=1, dtype=torch.float64),
        enterprise,
        spot_curve=_flat_curve(0.06),
        six_month_yield_one_year_ago=None,
        annual_impairment=False,
        configuration=configuration,
    )

    impaired_next = apply_loan_transition(
        impaired.mortgage_cohorts,
        impaired.enterprise_cohorts,
        spot_curve=_flat_curve(0.06),
        six_month_yield_one_year_ago=None,
        annual_impairment=False,
        configuration=configuration,
    )
    unimpaired_next = apply_loan_transition(
        unimpaired.mortgage_cohorts,
        unimpaired.enterprise_cohorts,
        spot_curve=_flat_curve(0.06),
        six_month_yield_one_year_ago=None,
        annual_impairment=False,
        configuration=configuration,
    )

    assert impaired.enterprise_cohorts.outstanding_principal.item() == pytest.approx(97.0)
    assert impaired_next.interest_cash_flow.item() == pytest.approx(0.97)
    assert unimpaired_next.interest_cash_flow.item() == pytest.approx(1.0)


def test_same_annual_shock_impairs_loans_at_terminal_and_nonterminal_dates() -> None:
    historical = MarketScenarioModel().load_historical_term_structures(SOURCE)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    terminal_discounts = np.broadcast_to(
        historical.initial_curve.discount_factors, (1, 13, 180)
    ).copy()
    terminal_spots = np.broadcast_to(
        historical.initial_curve.spot_rates, (1, 13, 180)
    ).copy()
    terminal_spots[0, 0, 5] = 0.01
    terminal_spots[0, 12, 5] = 0.06

    terminal = ALMSimulator().rollout(
        snapshot,
        SimpleNamespace(
            discount_factors=terminal_discounts,
            spot_rates=terminal_spots,
            initial_curve_identity=snapshot.initial_curve_identity,
            as_of_date=snapshot.as_of_date,
        ),
        include_loan_dynamics=True,
    )
    nonterminal = ALMSimulator().rollout(
        snapshot,
        SimpleNamespace(
            discount_factors=np.concatenate(
                (terminal_discounts, terminal_discounts[:, -1:]), axis=1
            ),
            spot_rates=np.concatenate(
                (terminal_spots, terminal_spots[:, -1:]), axis=1
            ),
            initial_curve_identity=snapshot.initial_curve_identity,
            as_of_date=snapshot.as_of_date,
        ),
        include_loan_dynamics=True,
    )

    assert terminal.enterprise_impairment_factors is not None
    assert nonterminal.enterprise_impairment_factors is not None
    assert terminal.enterprise_impairment_factors[0, 11].item() == pytest.approx(0.03)
    assert nonterminal.enterprise_impairment_factors[0, 11].item() == pytest.approx(0.03)
    assert terminal.portfolio_values["enterprise_loans"][0, 12].item() == pytest.approx(
        nonterminal.portfolio_values["enterprise_loans"][0, 12].item()
    )
    assert terminal.equity[0, 12].item() == pytest.approx(
        nonterminal.equity[0, 12].item()
    )


def test_terminal_impairment_reduces_loan_value_equity_and_training_target() -> None:
    historical = MarketScenarioModel().load_historical_term_structures(SOURCE)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    discounts = np.broadcast_to(
        historical.initial_curve.discount_factors, (1, 61, 180)
    ).copy()
    base_spots = np.broadcast_to(
        historical.initial_curve.spot_rates, (1, 61, 180)
    ).copy()
    base_spots[0, :, 5] = 0.01
    shocked_spots = base_spots.copy()
    shocked_spots[0, 60, 5] = 0.06
    parameters = ObjectiveParameters(
        mu=torch.tensor([1.0], dtype=torch.float64),
        penalty_weight=torch.tensor([3.5], dtype=torch.float64),
    )

    def rollout(spots: np.ndarray) -> object:
        return ALMSimulator().rollout(
            snapshot,
            SimpleNamespace(
                discount_factors=discounts,
                spot_rates=spots,
                initial_curve_identity=snapshot.initial_curve_identity,
                as_of_date=snapshot.as_of_date,
            ),
            include_loan_dynamics=True,
            objective_parameters=parameters,
        )

    base = rollout(base_spots)
    shocked = rollout(shocked_spots)

    assert shocked.enterprise_impairment_factors is not None
    assert shocked.enterprise_impairment_factors[0, -1].item() == pytest.approx(0.03)
    assert shocked.portfolio_values["enterprise_loans"][0, -1].item() == pytest.approx(
        0.97 * base.portfolio_values["enterprise_loans"][0, -1].item()
    )
    assert shocked.equity[0, -1] < base.equity[0, -1]
    assert shocked.objective is not None
    assert base.objective is not None
    assert shocked.objective.target[0] > base.objective.target[0]
    assert torch.max(shocked.cash_reconciliation_error.abs()).item() <= 1e-6
