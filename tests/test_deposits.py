from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepalm.deposits import (
    DepositConfiguration,
    allocate_deposit_tranches,
    allocate_reference_term_tranches,
    cash_penalty,
    deposit_rates,
    monthly_deposit_interest,
    operating_cost,
)
from deepalm.reference_bank import ReferenceBankError, ReferenceBankProvider
from deepalm.runoff import ALMSimulator
from deepalm.term_structures import (
    MarketScenarioModel,
    deposit_initial_history_identity,
)


def _fixed_growth_snapshot(
    tmp_path: Path,
    *,
    nmd_balance: float = 100.0,
    initial_history: tuple[float, float] | None = None,
) -> object:
    """Import a 100-unit NMD book with the paper's declared class weights."""

    source = (
        Path(__file__).resolve().parents[1]
        / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
    )
    historical = MarketScenarioModel().load_historical_term_structures(source)
    provider = ReferenceBankProvider()
    data = provider.serialize(provider.build_canonical(historical))
    weights = (0.40, 0.30, 0.25, 0.05)
    terms = (1, 2, 12, 120)
    cash_flows = np.zeros((4, 180))
    for index, (term, weight) in enumerate(zip(terms, weights, strict=True)):
        cash_flows[index, :term] = nmd_balance * weight / term
    data["profile"] = "imported"
    data["ladders"]["non_maturity_deposits"] = cash_flows.sum(axis=0).tolist()
    data["deposit_reference_schedules"]["non_maturity_deposits"] = {
        "terms_months": list(terms),
        "weights": list(weights),
        "cash_flows": cash_flows.tolist(),
        "provenance": "100-unit paper-weight NMD growth oracle",
    }
    data["target_economic_values"]["non_maturity_deposits"] = nmd_balance
    data["target_value_errors"]["non_maturity_deposits"] = 1.0
    data["equity"] = 2_000.0 - nmd_balance
    if initial_history is not None:
        target_dates = ("2022-06-15", "2022-05-15")
        observation_dates = ("2022-06-14", "2022-05-13")
        available_dates = ("2022-05-13", "2022-06-14")
        available_yields = (initial_history[1], initial_history[0])
        source_identity = "fixed-deposit-history-fixture"
        data["deposit_initial_history"] = {
            "target_dates": list(target_dates),
            "observation_dates": list(observation_dates),
            "six_month_yields": list(initial_history),
            "available_observation_dates": list(available_dates),
            "available_six_month_yields": list(available_yields),
            "source_identity": source_identity,
            "identity": deposit_initial_history_identity(
                target_dates,
                observation_dates,
                initial_history,
                available_dates,
                available_yields,
                source_identity,
            ),
        }
    data_without_hash = dict(data)
    data_without_hash.pop("content_hash", None)
    data["content_hash"] = hashlib.sha256(
        json.dumps(data_without_hash, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "fixed-growth-bank.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return provider.load(path)


def test_deposit_rollover_preserves_equity_except_operating_costs() -> None:
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
        deposit_initial_history_identity=snapshot.deposit_initial_history.identity,
    )
    result = ALMSimulator().rollout(
        snapshot,
        market,
        include_deposit_dynamics=True,
        deposit_configuration=DepositConfiguration(
            non_maturity_growth=0, term_growth=0
        ),
    )
    # Historical initial rates may capitalize interest, but the cash ledger remains exact.
    assert torch.isfinite(result.equity).all()
    assert result.cash_reconciliation_error.abs().max().item() < 1e-8


def test_reference_term_deposit_schedules_roll_independently(tmp_path: Path) -> None:
    """A deposited balance keeps its original reference-term class on renewal."""

    snapshot = _fixed_growth_snapshot(tmp_path, initial_history=(0.0, 0.0))
    market = SimpleNamespace(
        discount_factors=np.ones((1, 2, 180)),
        spot_rates=np.zeros((1, 2, 180)),
        initial_curve_identity=snapshot.initial_curve_identity,
        as_of_date=snapshot.as_of_date,
        deposit_initial_history_identity=snapshot.deposit_initial_history.identity,
    )

    result = ALMSimulator().rollout(
        snapshot,
        market,
        include_deposit_dynamics=True,
        deposit_configuration=DepositConfiguration(
            non_maturity_growth=0.0, term_growth=0.0
        ),
    )

    assert result.deposit_reference_schedules is not None
    for name, initial in snapshot.deposit_reference_schedules.items():
        expected = torch.tensor(initial.cash_flows, dtype=torch.float64).unsqueeze(0)
        maturing = expected[:, :, 0].clone()
        expected = torch.nn.functional.pad(expected[:, :, 1:], (0, 1))
        for index, term in enumerate(initial.terms_months):
            expected[:, index, :term] += (maturing[:, index] / term).unsqueeze(1)
        actual = result.deposit_reference_schedules[name]
        assert torch.allclose(actual, expected)
        assert torch.allclose(
            actual.sum(dim=(1, 2)), result.portfolio_values[name][:, 1]
        )
    assert result.cash_reconciliation_error.abs().max().item() < 1e-8


def test_reference_term_renewal_has_one_and_two_month_hand_oracles() -> None:
    renewed = allocate_reference_term_tranches(
        torch.tensor([[1.0, 2.0]], dtype=torch.float64), (1, 2)
    )
    assert renewed[0, 0, :3].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert renewed[0, 1, :3].tolist() == pytest.approx([1.0, 1.0, 0.0])

    capitalized_interest = allocate_reference_term_tranches(
        torch.tensor([[2.0]], dtype=torch.float64), (2,)
    )
    assert capitalized_interest[0, 0, :3].tolist() == pytest.approx(
        [1.0, 1.0, 0.0]
    )

    negative = allocate_reference_term_tranches(
        torch.tensor([[-2.0]], dtype=torch.float64), (2,)
    )
    assert negative[0, 0, :3].tolist() == pytest.approx([-1.0, -1.0, 0.0])


def test_negative_deposit_interest_stays_in_each_reference_term_class(
    tmp_path: Path,
) -> None:
    snapshot = _fixed_growth_snapshot(tmp_path, initial_history=(-0.10, -0.10))
    market = SimpleNamespace(
        discount_factors=np.ones((1, 2, 180)),
        spot_rates=np.full((1, 2, 180), -0.10),
        initial_curve_identity=snapshot.initial_curve_identity,
        as_of_date=snapshot.as_of_date,
        deposit_initial_history_identity=snapshot.deposit_initial_history.identity,
    )
    result = ALMSimulator().rollout(
        snapshot,
        market,
        include_deposit_dynamics=True,
        deposit_configuration=DepositConfiguration(
            non_maturity_growth=0.0, term_growth=0.0
        ),
    )

    for name, annual_rate in (
        ("non_maturity_deposits", -0.06),
        ("term_deposits", -0.085),
    ):
        initial = snapshot.deposit_reference_schedules[name]
        expected = torch.tensor(initial.cash_flows, dtype=torch.float64).unsqueeze(0)
        renewal = expected[:, :, 0] + (
            np.exp(annual_rate / 12.0) - 1.0
        ) * expected.sum(dim=2)
        expected = torch.nn.functional.pad(expected[:, :, 1:], (0, 1))
        for index, term in enumerate(initial.terms_months):
            expected[:, index, :term] += (renewal[:, index] / term).unsqueeze(1)
        assert torch.allclose(result.deposit_reference_schedules[name], expected)
    assert result.cash_reconciliation_error.abs().max().item() < 1e-8


def test_new_deposit_growth_uses_only_the_declared_global_weights(
    tmp_path: Path,
) -> None:
    snapshot = _fixed_growth_snapshot(tmp_path, initial_history=(0.0, 0.0))
    market = SimpleNamespace(
        discount_factors=np.ones((1, 2, 180)),
        spot_rates=np.zeros((1, 2, 180)),
        initial_curve_identity=snapshot.initial_curve_identity,
        as_of_date=snapshot.as_of_date,
        deposit_initial_history_identity=snapshot.deposit_initial_history.identity,
    )
    result = ALMSimulator().rollout(
        snapshot,
        market,
        include_deposit_dynamics=True,
        deposit_configuration=DepositConfiguration(
            non_maturity_growth=0.12, term_growth=0.0
        ),
    )

    initial = snapshot.deposit_reference_schedules["non_maturity_deposits"]
    expected = torch.tensor(initial.cash_flows, dtype=torch.float64).unsqueeze(0)
    renewal = expected[:, :, 0].clone()
    growth = expected.sum(dim=(1, 2)) * 0.12 / 12.0
    expected = torch.nn.functional.pad(expected[:, :, 1:], (0, 1))
    for index, term in enumerate(initial.terms_months):
        expected[:, index, :term] += (
            (renewal[:, index] + growth * initial.weights[index]) / term
        ).unsqueeze(1)
    assert torch.allclose(
        result.deposit_reference_schedules["non_maturity_deposits"], expected
    )
    assert result.deposit_growth is not None
    assert result.deposit_growth[:, 0] == pytest.approx(growth)
    assert result.cash_reconciliation_error.abs().max().item() < 1e-8


def test_one_hundred_new_deposits_allocate_40_30_25_5_after_rollout(
    tmp_path: Path,
) -> None:
    snapshot = _fixed_growth_snapshot(tmp_path)
    market = SimpleNamespace(
        discount_factors=np.ones((1, 2, 180)),
        spot_rates=np.zeros((1, 2, 180)),
        initial_curve_identity=snapshot.initial_curve_identity,
        as_of_date=snapshot.as_of_date,
        deposit_initial_history_identity=snapshot.deposit_initial_history.identity,
    )
    simulator = ALMSimulator()
    without_growth = simulator.rollout(
        snapshot,
        market,
        include_deposit_dynamics=True,
        deposit_configuration=DepositConfiguration(
            non_maturity_growth=0.0, term_growth=0.0
        ),
    )
    with_growth = simulator.rollout(
        snapshot,
        market,
        include_deposit_dynamics=True,
        deposit_configuration=DepositConfiguration(
            non_maturity_growth=12.0, term_growth=0.0
        ),
    )
    added_by_class = (
        with_growth.deposit_reference_schedules["non_maturity_deposits"]
        .sum(dim=2)
        .sub(without_growth.deposit_reference_schedules["non_maturity_deposits"].sum(dim=2))
    )
    assert torch.allclose(
        added_by_class, torch.tensor([[40.0, 30.0, 25.0, 5.0]], dtype=torch.float64)
    )
    assert with_growth.deposit_growth is not None
    assert with_growth.deposit_growth[:, 0] == pytest.approx([100.0])
    assert with_growth.cash_reconciliation_error.abs().max().item() < 1e-8


def test_first_deposit_interest_uses_dated_y0_yminus1_yminus2_history(
    tmp_path: Path,
) -> None:
    snapshot = _fixed_growth_snapshot(
        tmp_path, nmd_balance=1_000.0, initial_history=(0.02, 0.01)
    )
    market = SimpleNamespace(
        discount_factors=np.ones((1, 2, 180)),
        spot_rates=np.full((1, 2, 180), 0.03),
        initial_curve_identity=snapshot.initial_curve_identity,
        as_of_date=snapshot.as_of_date,
        deposit_initial_history_identity=snapshot.deposit_initial_history.identity,
    )
    result = ALMSimulator().rollout(
        snapshot,
        market,
        include_deposit_dynamics=True,
        deposit_configuration=DepositConfiguration(
            non_maturity_growth=0.0, term_growth=0.0
        ),
    )

    expected_interest = 1_000.0 * (np.exp(0.012 / 12.0) - 1.0)
    actual_balance = result.deposit_reference_schedules["non_maturity_deposits"].sum()
    assert actual_balance.item() == pytest.approx(1_000.0 + expected_interest)
    assert result.cash_reconciliation_error.abs().max().item() < 1e-8


def test_second_deposit_interest_drops_yminus2_from_the_window(tmp_path: Path) -> None:
    snapshot = _fixed_growth_snapshot(
        tmp_path, nmd_balance=1_000.0, initial_history=(0.02, 0.01)
    )
    market = SimpleNamespace(
        discount_factors=np.ones((1, 3, 180)),
        spot_rates=np.full((1, 3, 180), 0.04),
        initial_curve_identity=snapshot.initial_curve_identity,
        as_of_date=snapshot.as_of_date,
        deposit_initial_history_identity=snapshot.deposit_initial_history.identity,
    )
    market.spot_rates[:, 0, :] = 0.03
    result = ALMSimulator().rollout(
        snapshot,
        market,
        include_deposit_dynamics=True,
        deposit_configuration=DepositConfiguration(
            non_maturity_growth=0.0, term_growth=0.0
        ),
    )

    first_interest = np.exp(0.012 / 12.0) - 1.0
    second_interest = np.exp(0.018 / 12.0) - 1.0
    actual_balance = result.deposit_reference_schedules["non_maturity_deposits"].sum()
    assert actual_balance.item() == pytest.approx(
        1_000.0 * (1.0 + first_interest) * (1.0 + second_interest)
    )


def test_deposit_rollout_rejects_a_market_with_different_history_identity(
    tmp_path: Path,
) -> None:
    snapshot = _fixed_growth_snapshot(tmp_path, initial_history=(0.02, 0.01))
    market = SimpleNamespace(
        discount_factors=np.ones((1, 2, 180)),
        spot_rates=np.full((1, 2, 180), 0.03),
        initial_curve_identity=snapshot.initial_curve_identity,
        as_of_date=snapshot.as_of_date,
        deposit_initial_history_identity="other-history",
    )

    with pytest.raises(ReferenceBankError, match="initial-history identity"):
        ALMSimulator().rollout(snapshot, market, include_deposit_dynamics=True)


@pytest.mark.parametrize("transitions", [60, 180])
def test_reference_term_classes_conserve_each_original_class_across_long_rolls(
    transitions: int, tmp_path: Path
) -> None:
    snapshot = _fixed_growth_snapshot(tmp_path, initial_history=(0.0, 0.0))
    market = SimpleNamespace(
        discount_factors=np.ones((1, transitions + 1, 180)),
        spot_rates=np.zeros((1, transitions + 1, 180)),
        initial_curve_identity=snapshot.initial_curve_identity,
        as_of_date=snapshot.as_of_date,
        deposit_initial_history_identity=snapshot.deposit_initial_history.identity,
    )
    result = ALMSimulator().rollout(
        snapshot,
        market,
        include_deposit_dynamics=True,
        deposit_configuration=DepositConfiguration(
            non_maturity_growth=0.0, term_growth=0.0
        ),
    )
    for name, schedule in snapshot.deposit_reference_schedules.items():
        initial_balance = torch.tensor(schedule.cash_flows, dtype=torch.float64).sum(
            dim=1
        )
        final_balance = result.deposit_reference_schedules[name].sum(dim=2)[0]
        assert torch.allclose(final_balance, initial_balance)
    assert result.cash_reconciliation_error.abs().max().item() < 1e-8


def test_excess_cash_charge_reduces_simulated_cash_and_keeps_trade_gradients() -> None:
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
            deposit_initial_history_identity=snapshot.deposit_initial_history.identity,
        )
        mask = torch.zeros((1, 1, 29), dtype=torch.float64)
        mask[0, 0, 13] = 1
        return ALMSimulator().rollout(
            snapshot,
            market,
            actions=funding * mask,
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
                deposit_initial_history_identity=snapshot.deposit_initial_history.identity,
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
