from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from deepalm.constraints import (
    ConstraintError,
    ConstraintState,
    constraint_violations,
    evaluate_constraints,
)
from deepalm.objective import (
    ObjectiveParameters,
    evaluate_objective,
    evaluation_objective_parameters,
    sample_training_objective_parameters,
)
from deepalm.reference_bank import ReferenceBankProvider
from deepalm.runoff import ALMSimulator
from deepalm.term_structures import MarketScenarioModel

SNB_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
)


def _ladder(value: float, *, month: int = 1) -> torch.Tensor:
    ladder = torch.zeros((1, 180), dtype=torch.float64)
    ladder[0, month - 1] = value
    return ladder


def test_constraint_values_match_independent_hand_calculation() -> None:
    state = ConstraintState(
        cash=torch.tensor([100.0], dtype=torch.float64),
        investments=_ladder(100.0),
        mortgages=_ladder(100.0),
        enterprise_loans=_ladder(100.0),
        non_maturity_deposits=_ladder(100.0),
        term_deposits=_ladder(100.0),
        funding=_ladder(100.0),
        discounts=torch.ones((1, 180), dtype=torch.float64),
    )

    result = evaluate_constraints(state)

    assert result.values[0, 0].item() == pytest.approx(
        (0.71 * 100 + 0.89 * 100) / (0.176 * 100 + 0.13 * 100 + 0.01 * 100), abs=1e-10
    )
    assert result.values[0, 1].item() == pytest.approx(
        (0.95 * 100 + 0.90 * 100 + 0.60 * 100 + 100) / (0.12 * 100 + 0.71 * 200),
        abs=1e-10,
    )
    assert result.values[0, 2].item() == pytest.approx(
        100 / (0.025 * (100 + 0.20 * 100)), abs=1e-10
    )
    assert result.values[0, 3].item() == pytest.approx(
        100 / (0.10 * 100 + 0.35 * 100 + 100), abs=1e-10
    )
    assert result.values[0, 4].item() == pytest.approx(0.0, abs=1e-10)
    assert result.values[0, 5].item() == pytest.approx(0.0, abs=1e-10)


def test_constraint_breach_transforms_respect_bounds_and_annual_mask() -> None:
    values = torch.tensor(
        [
            [1.04, 1.05, 0.99, 0.16, 0.086, -0.01],
            [1.05, 1.05, 1.00, 0.17, 0.085, 0.00],
            [1.06, 1.06, 1.01, 0.18, 0.084, 0.01],
        ],
        dtype=torch.float64,
    )

    violations = constraint_violations(
        values, annual_mask=torch.tensor([True, False, True])
    )

    assert violations[0, 0].item() == pytest.approx((1.0 + 0.01) ** 2 - 1.0, abs=1e-10)
    assert violations[0, 4].item() == pytest.approx((1.0 + 0.001) ** 2 - 1.0, abs=1e-10)
    assert violations[0, 5].item() == pytest.approx((1.0 + 0.01) ** 2 - 1.0, abs=1e-10)
    assert torch.allclose(violations[1], torch.zeros(6, dtype=torch.float64))
    assert torch.allclose(violations[2], torch.zeros(6, dtype=torch.float64))


def test_irs_revalues_ladders_under_parallel_curve_shifts() -> None:
    state = ConstraintState(
        cash=torch.tensor([100.0], dtype=torch.float64),
        investments=_ladder(100.0, month=180),
        mortgages=_ladder(0.0),
        enterprise_loans=_ladder(0.0),
        non_maturity_deposits=_ladder(20.0),
        term_deposits=_ladder(20.0),
        funding=_ladder(20.0),
        discounts=torch.ones((1, 180), dtype=torch.float64),
    )

    result = evaluate_constraints(state)

    equity = 100.0 + 100.0 - 20.0 - 20.0 - 20.0
    equity_up = 100.0 + 100.0 * math.exp(-0.01 * 15.0) - 60.0 * math.exp(-0.01 / 12.0)
    equity_down = 100.0 + 100.0 * math.exp(0.01 * 15.0) - 60.0 * math.exp(0.01 / 12.0)
    expected = max(abs(equity_up - equity), abs(equity_down - equity)) / equity
    assert result.values[0, 4].item() == pytest.approx(expected, abs=1e-10)


def test_eyr_uses_pre_dividend_equity_only_at_an_annual_close() -> None:
    state = ConstraintState(
        cash=torch.tensor([100.0], dtype=torch.float64),
        investments=_ladder(100.0),
        mortgages=_ladder(100.0),
        enterprise_loans=_ladder(100.0),
        non_maturity_deposits=_ladder(100.0),
        term_deposits=_ladder(100.0),
        funding=_ladder(100.0),
        discounts=torch.ones((1, 180), dtype=torch.float64),
    )

    result = evaluate_constraints(
        state, previous_annual_equity=torch.tensor([90.0], dtype=torch.float64)
    )

    assert result.annual_mask.item()
    assert result.values[0, 5].item() == pytest.approx(
        (100.0 - 90.0 - 6.0) / 90.0, abs=1e-10
    )


def test_eyr_preserves_negative_prior_equity_and_rejects_zero_denominator() -> None:
    state = ConstraintState(
        cash=torch.tensor([100.0], dtype=torch.float64),
        investments=_ladder(100.0),
        mortgages=_ladder(100.0),
        enterprise_loans=_ladder(100.0),
        non_maturity_deposits=_ladder(100.0),
        term_deposits=_ladder(100.0),
        funding=_ladder(290.0),
        discounts=torch.ones((1, 180), dtype=torch.float64),
    )

    negative_prior = evaluate_constraints(
        state, previous_annual_equity=torch.tensor([-100.0], dtype=torch.float64)
    )
    assert negative_prior.values[0, 5].item() == pytest.approx(
        (-90.0 - (-100.0) - 6.0) / -100.0, abs=1e-10
    )
    with pytest.raises(ConstraintError, match="cannot be zero"):
        evaluate_constraints(
            state, previous_annual_equity=torch.tensor([0.0], dtype=torch.float64)
        )


def test_objective_uses_asymmetric_target_and_excludes_crra_from_training_loss() -> (
    None
):
    parameters = ObjectiveParameters(
        mu=torch.tensor([0.02, 0.02], dtype=torch.float64),
        penalty_weight=torch.tensor([3.5, 3.5], dtype=torch.float64),
    )
    initial_equity = torch.tensor([100.0, 100.0], dtype=torch.float64)
    terminal_equity = torch.tensor([120.0, 90.0], dtype=torch.float64)
    violations = torch.zeros((2, 2, 6), dtype=torch.float64)
    violations[1, 0, 0] = (1.0 + 0.01) ** 2 - 1.0

    result = evaluate_objective(
        initial_equity,
        terminal_equity,
        horizon_years=5,
        violations=violations,
        parameters=parameters,
    )

    expected_target = (90.0 - (1.02**5) * 100.0) ** 2
    expected_penalty = (1.0 + ((1.0 + 0.01) ** 2 - 1.0)) ** 2 - 1.0
    assert result.target[0].item() == pytest.approx(0.0, abs=1e-10)
    assert result.target[1].item() == pytest.approx(expected_target, abs=1e-10)
    assert result.penalty[1].item() == pytest.approx(expected_penalty, abs=1e-10)
    assert result.total[1].item() == pytest.approx(
        expected_target + 3.5 * expected_penalty, abs=1e-10
    )
    assert result.crra[1].item() > 0.0
    assert result.gamma == 10.0
    assert result.equity_ratio_floor == 1e-8


@pytest.mark.parametrize(
    ("one_month_discount", "expected_cmr"),
    [
        (1.0, 10_000.0 / (0.025 * (100.0 + 0.20 * 50.0))),
        (1.001, 10_000.0 / (0.025 * 1.001 * (100.0 + 50.0))),
    ],
)
def test_cmr_uses_term_deposits_and_the_sign_dependent_reserve_base(
    one_month_discount: float, expected_cmr: float
) -> None:
    discounts = torch.ones((1, 180), dtype=torch.float64)
    discounts[0, 0] = one_month_discount
    state = ConstraintState(
        cash=torch.tensor([10_000.0], dtype=torch.float64),
        investments=_ladder(100.0),
        mortgages=_ladder(100.0),
        enterprise_loans=_ladder(100.0),
        non_maturity_deposits=_ladder(100.0),
        term_deposits=_ladder(50.0),
        funding=_ladder(9_000.0),
        discounts=discounts,
    )

    assert evaluate_constraints(state).values[0, 2].item() == pytest.approx(
        expected_cmr, abs=1e-10
    )


def test_training_parameter_sampling_and_fixed_evaluation_parameters() -> None:
    sampled = sample_training_objective_parameters(
        3,
        generator=torch.Generator().manual_seed(7),
        dtype=torch.float64,
    )

    assert torch.all((sampled.mu >= 0.02) & (sampled.mu <= 0.07))
    assert torch.all(
        (sampled.penalty_weight >= 0.05) & (sampled.penalty_weight <= 25.0)
    )
    assert torch.allclose(
        evaluation_objective_parameters(2, horizon_years=5, dtype=torch.float64).mu,
        torch.tensor([0.0406, 0.0406], dtype=torch.float64),
    )
    assert torch.allclose(
        evaluation_objective_parameters(
            2, horizon_years=15, dtype=torch.float64
        ).penalty_weight,
        torch.tensor([3.5, 3.5], dtype=torch.float64),
    )


@pytest.mark.parametrize("horizon_months", [60, 180])
def test_simulator_returns_constraint_trajectory_and_terminal_objective(
    horizon_months: int,
) -> None:
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SNB_SOURCE)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    monthly_tenors = torch.arange(1, 181, dtype=torch.float64) / 12.0
    discounts = torch.exp(-0.03 * monthly_tenors).repeat(1, horizon_months + 1, 1)
    spots = torch.full_like(discounts, 0.03)
    parameters = ObjectiveParameters(
        mu=torch.tensor([0.04], dtype=torch.float64),
        penalty_weight=torch.tensor([3.5], dtype=torch.float64),
    )

    result = ALMSimulator().rollout(
        snapshot,
        SimpleNamespace(
            discount_factors=discounts,
            spot_rates=spots,
            initial_curve_identity=snapshot.initial_curve_identity,
            as_of_date=snapshot.as_of_date,
            deposit_initial_history_identity=snapshot.deposit_initial_history.identity,
        ),
        actions=torch.zeros((1, horizon_months, 29), dtype=torch.float64),
        include_deposit_dynamics=True,
        include_constraints=True,
        objective_parameters=parameters,
    )

    assert result.initial_constraint_values is not None
    assert result.constraint_values is not None
    assert result.constraint_violations is not None
    assert result.constraint_annual_mask is not None
    assert result.objective is not None
    assert result.initial_constraint_values.shape == (1, 6)
    assert result.constraint_values.shape == (1, horizon_months, 6)
    assert result.constraint_violations.shape == (1, horizon_months, 6)
    assert result.constraint_annual_mask[0, 12]
    assert not result.constraint_annual_mask[0, -1]
    assert result.dividends is not None
    pre_dividend_equity = result.equity[0, 12] + result.dividends[0, 11]
    expected_eyr = (pre_dividend_equity - result.equity[0, 0] - 6.0) / result.equity[
        0, 0
    ]
    assert result.constraint_values[0, 12, 5].item() == pytest.approx(
        expected_eyr.item(), abs=1e-10
    )
    assert torch.isfinite(result.objective.total).all()
    assert torch.max(result.accounting_error.abs()).item() <= 1e-6
    assert torch.max(result.cash_reconciliation_error.abs()).item() <= 1e-6


def test_action_dependent_objective_gradient_matches_central_difference() -> None:
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SNB_SOURCE)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    discounts = torch.tensor(
        historical.initial_curve.discount_factors, dtype=torch.float64
    ).repeat(1, 61, 1)
    market = SimpleNamespace(
        discount_factors=discounts,
        initial_curve_identity=snapshot.initial_curve_identity,
        as_of_date=snapshot.as_of_date,
    )
    parameters = ObjectiveParameters(
        mu=torch.tensor([0.20], dtype=torch.float64),
        penalty_weight=torch.tensor([0.05], dtype=torch.float64),
    )
    direction = torch.zeros((1, 60, 29), dtype=torch.float64)
    direction[0, 0, 0] = 1.0

    amount = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    objective = (
        ALMSimulator()
        .rollout(
            snapshot,
            market,
            actions=amount * direction,
            objective_parameters=parameters,
        )
        .objective
    )
    assert objective is not None
    objective.total.sum().backward()

    def total_loss(value: float) -> float:
        result = (
            ALMSimulator()
            .rollout(
                snapshot,
                market,
                actions=value * direction,
                objective_parameters=parameters,
            )
            .objective
        )
        assert result is not None
        return result.total.item()

    epsilon = 1e-5
    finite_difference = (total_loss(1.0 + epsilon) - total_loss(1.0 - epsilon)) / (
        2.0 * epsilon
    )
    assert amount.grad is not None
    assert amount.grad.abs().item() > 1e-8
    assert abs(amount.grad.item() - finite_difference) <= 1e-6 + 1e-4 * abs(
        finite_difference
    )
