from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepalm.deposits import (
    DepositConfiguration,
    allocate_deposit_tranches,
    cash_penalty,
    deposit_rates,
    monthly_deposit_interest,
    operating_cost,
)
from deepalm.reference_bank import ReferenceBankProvider
from deepalm.runoff import ALMSimulator
from deepalm.term_structures import MarketScenarioModel


@pytest.mark.parametrize("convention", ["paper", "corrected"])
def test_deposit_rollover_preserves_equity_except_operating_costs(
    convention: str,
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
    )
    historical = MarketScenarioModel().load_historical_term_structures(source)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    market = SimpleNamespace(
        discount_factors=np.ones((1, 4, 180)),
        spot_rates=np.zeros((1, 4, 180)),
        initial_curve_identity=snapshot.initial_curve_identity,
        as_of_date=snapshot.as_of_date,
    )
    result = ALMSimulator().rollout(
        snapshot,
        market,
        convention=convention,
        include_deposit_dynamics=True,
        deposit_configuration=DepositConfiguration(
            non_maturity_growth=0, term_growth=0
        ),
    )
    # With no interest, growth, trading or dividends, rollover cannot destroy equity.
    assert torch.diff(result.equity, dim=1).tolist()[0] == pytest.approx([-4.0] * 3)
    assert result.cash_reconciliation_error.abs().max().item() < 1e-8


@pytest.mark.parametrize("convention", ["paper", "corrected"])
def test_excess_cash_charge_reduces_simulated_cash_and_keeps_trade_gradients(
    convention: str,
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
    )
    historical = MarketScenarioModel().load_historical_term_structures(source)
    snapshot = ReferenceBankProvider().build_canonical(historical)

    def simulate(discount: float, funding: torch.Tensor):
        discounts = np.ones((1, 2, 180))
        discounts[0, 0, 0] = discount
        market = SimpleNamespace(
            discount_factors=discounts,
            spot_rates=np.zeros((1, 2, 180)),
            initial_curve_identity=snapshot.initial_curve_identity,
            as_of_date=snapshot.as_of_date,
        )
        mask = torch.zeros((1, 1, 29), dtype=torch.float64)
        mask[0, 0, 13] = 1
        return ALMSimulator().rollout(
            snapshot,
            market,
            actions=funding * mask,
            convention=convention,
            include_deposit_dynamics=True,
            deposit_configuration=DepositConfiguration(
                non_maturity_growth=0, term_growth=0
            ),
        )

    funding = torch.tensor(10_000.0, dtype=torch.float64, requires_grad=True)
    charged, uncharged = simulate(1.01, funding), simulate(1.0, funding)
    assert charged.cash_penalties[0, 0] > 0
    assert charged.cash[0, 1] < uncharged.cash[0, 1]
    charged.cash[0, 1].backward()
    epsilon = 0.01
    finite_difference = (
        simulate(1.01, funding.detach() + epsilon).cash[0, 1]
        - simulate(1.01, funding.detach() - epsilon).cash[0, 1]
    ) / (2 * epsilon)
    assert funding.grad.item() == pytest.approx(finite_difference.item(), rel=1e-7)


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
    assert penalty.item() == pytest.approx(0.3960396039603964)
    assert operating_cost(
        personnel_cost=3, material_cost=1, completed_years=1
    ) == pytest.approx(4.06)


@pytest.mark.parametrize(
    "cash,discount", [(60.0, 1.01), (50.0, 1.01), (100.0, 1.0), (100.0, 0.99)]
)
def test_cash_penalty_does_not_charge_exempt_cash_or_pay_positive_interest(
    cash: float, discount: float
) -> None:
    assert (
        cash_penalty(
            torch.tensor([cash]), torch.tensor([2.0]), torch.tensor([discount])
        ).item()
        == 0
    )


def test_simulator_tracks_deposit_cost_cash_charge_and_nonterminal_dividend() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
    )
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(source)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    discounts = np.broadcast_to(
        historical.initial_curve.discount_factors, (1, 14, 180)
    ).copy()
    spots = np.broadcast_to(historical.initial_curve.spot_rates, (1, 14, 180)).copy()
    discounts[:, :, 0] = 1.01
    result = ALMSimulator().rollout(
        snapshot,
        SimpleNamespace(
            discount_factors=discounts,
            spot_rates=spots,
            initial_curve_identity=snapshot.initial_curve_identity,
            as_of_date=snapshot.as_of_date,
        ),
        include_deposit_dynamics=True,
    )
    assert result.deposit_growth is not None and torch.all(result.deposit_growth > 0)
    assert result.operating_costs is not None and result.operating_costs[
        0, 12
    ] == pytest.approx(4.06)
    assert result.cash_penalties is not None and torch.all(result.cash_penalties >= 0)
    assert result.dividends is not None and result.dividends[0, 11] >= 0
    assert torch.max(result.accounting_error.abs()).item() <= 1e-6
