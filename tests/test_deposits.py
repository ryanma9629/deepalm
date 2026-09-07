from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepalm.deposits import (
    allocate_deposit_tranches,
    cash_penalty,
    deposit_rates,
    monthly_deposit_interest,
    operating_cost,
)
from deepalm.reference_bank import ReferenceBankProvider
from deepalm.runoff import ALMSimulator
from deepalm.term_structures import MarketScenarioModel


def test_deposit_rate_caps_and_interest_reinvestment_formula() -> None:
    rates = deposit_rates(
        torch.tensor([[0.10, 0.10, 0.10]], dtype=torch.float64),
        torch.tensor([0.10], dtype=torch.float64),
    )
    assert rates.non_maturity.item() == pytest.approx(0.03)
    assert rates.term.item() == pytest.approx(0.05)
    assert monthly_deposit_interest(rates.term).item() == pytest.approx(
        np.exp(0.05 / 12) - 1
    )
    tranches = allocate_deposit_tranches(
        torch.tensor([12.0], dtype=torch.float64), (1, 2), (0.5, 0.5)
    )
    assert tranches[0, :2].tolist() == pytest.approx([9.0, 3.0])


def test_cash_penalty_and_annual_operating_cost() -> None:
    penalty = cash_penalty(
        torch.tensor([100.0]), torch.tensor([2.0]), torch.tensor([1.01])
    )
    assert penalty.item() == pytest.approx(40.0 * (1 / 1.01 - 1))
    assert operating_cost(personnel_cost=3, material_cost=1, completed_years=1) == pytest.approx(4.06)


def test_simulator_tracks_deposit_cost_cash_charge_and_nonterminal_dividend() -> None:
    source = Path(__file__).resolve().parents[1] / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(source)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    discounts = np.broadcast_to(historical.initial_curve.discount_factors, (1, 14, 180)).copy()
    spots = np.broadcast_to(historical.initial_curve.spot_rates, (1, 14, 180)).copy()
    discounts[:, :, 0] = 1.01
    result = ALMSimulator().rollout(snapshot, SimpleNamespace(discount_factors=discounts, spot_rates=spots), include_deposit_dynamics=True)
    assert result.deposit_growth is not None and torch.all(result.deposit_growth > 0)
    assert result.operating_costs is not None and result.operating_costs[0, 12] == pytest.approx(4.06)
    assert result.cash_penalties is not None and torch.all(result.cash_penalties >= 0)
    assert result.dividends is not None and result.dividends[0, 11] >= 0
    assert torch.max(result.accounting_error.abs()).item() <= 1e-6
