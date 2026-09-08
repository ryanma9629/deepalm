from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepalm.reference_bank import ReferenceBankProvider
from deepalm.runoff import ALMSimulator, PassiveRunoffSimulator, RunoffSimulationError
from deepalm.term_structures import MarketScenarioModel

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
)


@pytest.fixture(scope="module")
def canonical_inputs() -> tuple[object, object, MarketScenarioModel]:
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    return ReferenceBankProvider().build_canonical(historical), historical, model


def test_passive_runoff_settles_shifts_revalues_and_preserves_snapshot(
    canonical_inputs: tuple[object, object, MarketScenarioModel],
) -> None:
    snapshot, historical, _ = canonical_inputs
    source_mortgages = snapshot.ladders["mortgages"].copy()
    discounts = np.broadcast_to(
        historical.initial_curve.discount_factors, (1, 3, 180)
    ).copy()

    result = PassiveRunoffSimulator().rollout(
        snapshot,
        SimpleNamespace(
            discount_factors=discounts,
            initial_curve_identity=snapshot.initial_curve_identity,
            as_of_date=snapshot.as_of_date,
        ),
    )

    asset_settled = sum(
        snapshot.ladders[name][0]
        for name in ("investments", "mortgages", "enterprise_loans")
    )
    liability_settled = sum(
        snapshot.ladders[name][0]
        for name in ("non_maturity_deposits", "term_deposits", "funding")
    )
    expected_cash = snapshot.cash + asset_settled - liability_settled
    expected_mortgage_value = float(
        snapshot.ladders["mortgages"][1:]
        @ historical.initial_curve.discount_factors[:-1]
    )

    assert result.cash.shape == (1, 3)
    assert result.cash[0, 1].item() == pytest.approx(expected_cash)
    assert result.portfolio_values["mortgages"][0, 1].item() == pytest.approx(
        expected_mortgage_value
    )
    assert result.settled_cash_flows["mortgages"][0, 0].item() == pytest.approx(
        snapshot.ladders["mortgages"][0]
    )
    assert np.array_equal(snapshot.ladders["mortgages"], source_mortgages)
    assert not snapshot.ladders["mortgages"].flags.writeable
    assert torch.max(result.accounting_error.abs()).item() <= 1e-6
    assert torch.max(result.cash_reconciliation_error.abs()).item() <= 1e-6


@pytest.mark.parametrize("horizon_years", [5, 15])
def test_passive_runoff_accepts_complete_generated_market_paths(
    canonical_inputs: tuple[object, object, MarketScenarioModel], horizon_years: int
) -> None:
    snapshot, historical, model = canonical_inputs
    calibration = model.calibrate_hjm_pca(historical)
    market = model.generate_hjm_scenarios(
        historical,
        calibration,
        convention="corrected",
        horizon_years=horizon_years,
        paths=1,
        seed=42,
    )

    result = PassiveRunoffSimulator().rollout(snapshot, market)

    assert result.transitions == horizon_years * 12
    assert result.cash.shape == (1, horizon_years * 12 + 1)
    assert torch.isfinite(result.equity).all()
    assert torch.allclose(
        result.assets - result.liabilities - result.equity,
        torch.zeros_like(result.equity),
        atol=1e-6,
    )


def test_passive_runoff_identifies_first_invalid_market_component(
    canonical_inputs: tuple[object, object, MarketScenarioModel],
) -> None:
    snapshot, historical, _ = canonical_inputs
    discounts = np.broadcast_to(
        historical.initial_curve.discount_factors, (1, 2, 180)
    ).copy()
    discounts[0, 1, 7] = np.nan

    with pytest.raises(RunoffSimulationError, match="discount_factors") as error:
        PassiveRunoffSimulator().rollout(
            snapshot, SimpleNamespace(discount_factors=discounts)
        )

    assert error.value.diagnostics == {
        "path": 0,
        "time": 1,
        "component": "discount_factors[7]",
    }


def test_zero_actions_match_passive_runoff_and_nonzero_actions_reach_terminal_equity(
    canonical_inputs: tuple[object, object, MarketScenarioModel],
) -> None:
    snapshot, historical, _ = canonical_inputs
    discounts = np.broadcast_to(
        historical.initial_curve.discount_factors, (1, 3, 180)
    ).copy()
    market = SimpleNamespace(
        discount_factors=discounts,
        initial_curve_identity=snapshot.initial_curve_identity,
        as_of_date=snapshot.as_of_date,
    )
    simulator = ALMSimulator()

    passive = simulator.rollout(snapshot, market)
    zero = simulator.rollout(
        snapshot, market, actions=torch.zeros((1, 2, 29), dtype=torch.float64)
    )
    action_values = torch.zeros((1, 2, 29), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        action_values[0, 0, 0] = 1.0
        action_values[0, 1, 13] = 2.0
    active = simulator.rollout(snapshot, market, actions=action_values)
    active.equity[:, -1].sum().backward()

    assert torch.allclose(passive.cash, zero.cash)
    assert torch.allclose(passive.equity, zero.equity)
    assert active.treasury_cash_settlements is not None
    assert torch.max(active.cash_reconciliation_error.abs()).item() <= 1e-6
    assert action_values.grad is not None
    assert action_values.grad.abs().sum().item() > 0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an available MPS device"
)
def test_float32_mps_rollout_accepts_machine_precision_reconciliation(
    canonical_inputs: tuple[object, object, MarketScenarioModel],
) -> None:
    snapshot, historical, _ = canonical_inputs
    discounts = np.broadcast_to(
        historical.initial_curve.discount_factors, (1, 3, 180)
    ).copy()

    result = ALMSimulator().rollout(
        snapshot,
        SimpleNamespace(
            discount_factors=discounts,
            initial_curve_identity=snapshot.initial_curve_identity,
            as_of_date=snapshot.as_of_date,
        ),
        device="mps",
        dtype=torch.float32,
    )

    assert torch.isfinite(result.cash).all()
    assert torch.isfinite(result.equity).all()


def test_active_rollout_gradient_matches_central_difference(
    canonical_inputs: tuple[object, object, MarketScenarioModel],
) -> None:
    snapshot, historical, _ = canonical_inputs
    discounts = np.broadcast_to(
        historical.initial_curve.discount_factors, (1, 3, 180)
    ).copy()
    market = SimpleNamespace(
        discount_factors=discounts,
        initial_curve_identity=snapshot.initial_curve_identity,
        as_of_date=snapshot.as_of_date,
    )
    simulator = ALMSimulator()
    direction = torch.zeros((1, 2, 29), dtype=torch.float64)
    direction[0, 0, 0] = 1.0

    amount = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    loss = (
        simulator.rollout(snapshot, market, actions=amount * direction)
        .equity[:, -1]
        .sum()
    )
    loss.backward()

    def terminal_equity(value: float) -> float:
        return float(
            simulator.rollout(snapshot, market, actions=value * direction)
            .equity[:, -1]
            .sum()
        )

    epsilon = 1e-5
    finite_difference = (
        terminal_equity(1.0 + epsilon) - terminal_equity(1.0 - epsilon)
    ) / (2.0 * epsilon)
    assert amount.grad is not None
    assert abs(amount.grad.item() - finite_difference) <= 1e-6 + 1e-4 * abs(
        finite_difference
    )


@pytest.mark.parametrize("horizon_months", [60, 180])
def test_final_action_is_followed_by_one_passive_roll(
    canonical_inputs: tuple[object, object, MarketScenarioModel], horizon_months: int
) -> None:
    snapshot, historical, _ = canonical_inputs
    discounts = np.broadcast_to(
        historical.initial_curve.discount_factors, (1, horizon_months + 1, 180)
    ).copy()
    market = SimpleNamespace(
        discount_factors=discounts,
        initial_curve_identity=snapshot.initial_curve_identity,
        as_of_date=snapshot.as_of_date,
    )
    actions = torch.zeros((1, horizon_months, 29), dtype=torch.float64)
    actions[0, -1, 13] = 1.0
    simulator = ALMSimulator()

    passive = simulator.rollout(snapshot, market)
    active = simulator.rollout(snapshot, market, actions=actions)

    assert active.treasury_cash_settlements is not None
    final_settlement = active.treasury_cash_settlements[0, -1]
    assert final_settlement > 0
    assert active.cash[0, -2] - passive.cash[0, -2] == pytest.approx(
        final_settlement.item()
    )
    assert active.cash[0, -1] - passive.cash[0, -1] == pytest.approx(
        final_settlement.item()
    )
    assert active.cash[0, -1] - passive.cash[0, -1] == pytest.approx(
        (active.cash[0, -1] - passive.cash[0, -1]).item()
    )
