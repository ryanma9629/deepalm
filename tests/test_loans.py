from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepalm.loans import (
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


def test_paper_and_corrected_interest_differ_only_by_annualization() -> None:
    yield_rate = torch.tensor([0.03], dtype=torch.float64)

    paper = monthly_loan_interest_rate(
        yield_rate, spread=0.015, convention="paper"
    )
    corrected = monthly_loan_interest_rate(
        yield_rate, spread=0.015, convention="corrected"
    )

    assert paper.item() == pytest.approx(np.exp(0.045) - 1.0)
    assert corrected.item() == pytest.approx(np.exp(0.045 / 12.0) - 1.0)
    assert paper.item() > corrected.item() > 0


def test_loan_transition_replaces_maturities_adds_growth_and_impairs_enterprise() -> None:
    mortgages = torch.zeros((1, 180), dtype=torch.float64)
    enterprise = torch.zeros((1, 180), dtype=torch.float64)
    mortgages[0, 0] = 10.0
    enterprise[0, 0] = 3.0

    result = apply_loan_transition(
        mortgages,
        enterprise,
        six_month_yield=torch.tensor([0.06], dtype=torch.float64),
        six_month_yield_one_year_ago=torch.tensor([0.01], dtype=torch.float64),
        initial_total_loan_value=7_500.0,
        convention="corrected",
        annual_close=True,
    )

    mortgage_originations = 10.0 + 7_500 * 0.03 / 12 * (55 / 75)
    enterprise_originations = 3.0 + 7_500 * 0.03 / 12 * (20 / 75)
    assert result.originations.item() == pytest.approx(
        mortgage_originations + enterprise_originations
    )
    assert result.mortgages[0, 119].item() == pytest.approx(0.4 * mortgage_originations)
    assert result.enterprise_loans[0, 0].item() == pytest.approx(
        enterprise_originations / 3 * 0.97
    )
    assert result.impairment_factor.item() == pytest.approx(0.03)
    assert result.interest_cash_flow.item() > 0


def test_invalid_loan_distribution_fails_with_domain_error() -> None:
    ladder = torch.zeros((1, 180), dtype=torch.float64)
    invalid = LoanConfiguration(mortgage_weights=(1.0,))

    with pytest.raises(LoanDynamicsError, match="maturity distribution"):
        apply_loan_transition(
            ladder,
            ladder,
            six_month_yield=torch.zeros(1, dtype=torch.float64),
            six_month_yield_one_year_ago=None,
            initial_total_loan_value=7_500,
            convention="corrected",
            annual_close=False,
            configuration=invalid,
        )


def test_simulator_reconciles_loan_events_and_applies_annual_impairment() -> None:
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    discounts = np.broadcast_to(historical.initial_curve.discount_factors, (1, 14, 180)).copy()
    spots = np.broadcast_to(historical.initial_curve.spot_rates, (1, 14, 180)).copy()
    spots[0, 0, 5] = 0.01
    spots[0, 12, 5] = 0.06

    result = ALMSimulator().rollout(
        snapshot,
        SimpleNamespace(discount_factors=discounts, spot_rates=spots),
        include_loan_dynamics=True,
        convention="corrected",
    )

    assert result.loan_originations is not None
    assert result.loan_interest_cash_flows is not None
    assert result.enterprise_impairment_factors is not None
    assert result.enterprise_impairment_factors[0, 11].item() == pytest.approx(0.03)
    assert result.loan_originations[0, 0].item() > 0
    assert torch.max(result.cash_reconciliation_error.abs()).item() <= 1e-6
