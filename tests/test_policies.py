from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepalm.policies import (
    BMConstantPolicy,
    BMDatePolicy,
    BMEqualPolicy,
    TreasuryPolicyState,
)
from deepalm.reference_bank import ReferenceBankProvider
from deepalm.runoff import ALMSimulator, RunoffSimulationError
from deepalm.term_structures import MarketScenarioModel

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
)


def test_bme_divides_each_maturity_scale_equally_and_shares_two_parameters() -> None:
    policy = BMEqualPolicy(dtype=torch.float64)
    state = TreasuryPolicyState(
        investments=torch.tensor([[13.0] + [0.0] * 179], dtype=torch.float64),
        funding=torch.tensor([[16.0] + [0.0] * 179], dtype=torch.float64),
        time=7,
        transitions=60,
    )

    action = policy(state)

    assert action.investments.shape == (1, 13)
    assert action.funding.shape == (1, 16)
    assert torch.allclose(action.investments, torch.ones((1, 13), dtype=torch.float64))
    assert torch.allclose(action.funding, torch.ones((1, 16), dtype=torch.float64))
    assert len(tuple(policy.parameters())) == 2


def test_constant_benchmark_learns_one_non_equal_distribution_per_ladder() -> None:
    policy = BMConstantPolicy(dtype=torch.float64)
    with torch.no_grad():
        policy.investment_distribution_logits.copy_(
            torch.tensor([0.0, 1.0] + [0.0] * 11, dtype=torch.float64)
        )
        policy.funding_distribution_logits.copy_(
            torch.tensor([0.0, 1.0] + [0.0] * 14, dtype=torch.float64)
        )
    first_state = TreasuryPolicyState(
        investments=torch.tensor([[13.0] + [0.0] * 179], dtype=torch.float64),
        funding=torch.tensor([[16.0] + [0.0] * 179], dtype=torch.float64),
        time=0,
        transitions=60,
    )
    later_state = TreasuryPolicyState(
        investments=first_state.investments,
        funding=first_state.funding,
        time=59,
        transitions=60,
    )

    first_action = policy(first_state)
    later_action = policy(later_state)
    equal_action = BMEqualPolicy(dtype=torch.float64)(first_state)

    assert torch.all(first_action.investments >= 0.0)
    assert torch.all(first_action.funding >= 0.0)
    assert first_action.investments.sum().item() == pytest.approx(13.0)
    assert first_action.funding.sum().item() == pytest.approx(16.0)
    assert torch.allclose(first_action.investments, later_action.investments)
    assert torch.allclose(first_action.funding, later_action.funding)
    assert not torch.allclose(first_action.investments, equal_action.investments)
    assert not torch.allclose(first_action.funding, equal_action.funding)


def test_date_benchmark_scales_actions_from_maturing_notional_plus_adjustment() -> None:
    """Equation 43 uses the first ladder bucket, not an absolute date scale."""

    policy = BMDatePolicy(transitions=60, dtype=torch.float64)
    parameters = policy.state_dict()
    parameters["investment_adjustments"] = torch.zeros(60, dtype=torch.float64)
    parameters["funding_adjustments"] = torch.zeros(60, dtype=torch.float64)
    parameters["investment_adjustments"][0] = 10.0
    parameters["funding_adjustments"][0] = -5.0
    policy.load_state_dict(parameters)

    def state(investment_maturity: float, funding_maturity: float) -> TreasuryPolicyState:
        return TreasuryPolicyState(
            investments=torch.tensor(
                [[investment_maturity] + [0.0] * 179], dtype=torch.float64
            ),
            funding=torch.tensor(
                [[funding_maturity] + [0.0] * 179], dtype=torch.float64
            ),
            time=0,
            transitions=60,
        )

    action = policy(state(100.0, 40.0))
    changed_maturity_action = policy(state(120.0, 20.0))
    unrelated_market_action = policy(
        replace(
            state(100.0, 40.0),
            cash=torch.tensor([999.0], dtype=torch.float64),
            curve=torch.linspace(0.01, 0.20, 180, dtype=torch.float64).unsqueeze(0),
            mortgages=torch.full((1, 180), 17.0, dtype=torch.float64),
            enterprise_loans=torch.full((1, 180), 19.0, dtype=torch.float64),
            non_maturity_deposits=torch.full((1, 180), 23.0, dtype=torch.float64),
            term_deposits=torch.full((1, 180), 29.0, dtype=torch.float64),
            discounts=torch.full((1, 180), 0.8, dtype=torch.float64),
            initial_assets=torch.tensor([1000.0], dtype=torch.float64),
            prior_constraint_values=torch.full((1, 6), 7.0, dtype=torch.float64),
            mu=torch.tensor([0.2], dtype=torch.float64),
            penalty_weight=torch.tensor([8.0], dtype=torch.float64),
        )
    )

    assert action.investments.sum().item() == pytest.approx(110.0)
    assert action.funding.sum().item() == pytest.approx(35.0)
    assert changed_maturity_action.investments.sum().item() == pytest.approx(130.0)
    assert changed_maturity_action.funding.sum().item() == pytest.approx(15.0)
    assert torch.allclose(action.concatenated, unrelated_market_action.concatenated)

    zero_adjustments = policy.state_dict()
    zero_adjustments["investment_adjustments"] = torch.zeros(60, dtype=torch.float64)
    zero_adjustments["funding_adjustments"] = torch.zeros(60, dtype=torch.float64)
    policy.load_state_dict(zero_adjustments)
    zero_action = policy(state(100.0, 40.0))
    nonnegative_action = policy(state(-120.0, 40.0))

    assert zero_action.investments.sum().item() == pytest.approx(100.0)
    assert zero_action.funding.sum().item() == pytest.approx(40.0)
    assert nonnegative_action.investments.sum().item() == pytest.approx(0.0)

    legacy_parameters = policy.state_dict()
    legacy_parameters["investment_scales"] = legacy_parameters.pop(
        "investment_adjustments"
    )
    legacy_parameters["funding_scales"] = legacy_parameters.pop("funding_adjustments")
    with pytest.raises(RuntimeError, match="Missing key"):
        policy.load_state_dict(legacy_parameters)


def test_date_benchmark_learns_independent_actions_for_each_decision_date() -> None:
    policy = BMDatePolicy(transitions=60, dtype=torch.float64)
    with torch.no_grad():
        policy.investment_adjustments[0] = 1.0
        policy.investment_adjustments[1] = 2.0
        policy.funding_adjustments[0] = 3.0
        policy.funding_adjustments[1] = 4.0
        policy.investment_distribution_logits[0, 0] = 2.0
        policy.investment_distribution_logits[1, 1] = 2.0
        policy.funding_distribution_logits[0, 0] = 2.0
        policy.funding_distribution_logits[1, 1] = 2.0
    ladders = torch.tensor([[13.0] + [0.0] * 179], dtype=torch.float64)
    first_action = policy(
        TreasuryPolicyState(
            investments=ladders,
            funding=torch.tensor([[16.0] + [0.0] * 179], dtype=torch.float64),
            time=0,
            transitions=60,
        )
    )
    second_action = policy(
        TreasuryPolicyState(
            investments=ladders,
            funding=torch.tensor([[16.0] + [0.0] * 179], dtype=torch.float64),
            time=1,
            transitions=60,
        )
    )
    repeated_second_action = policy(
        TreasuryPolicyState(
            investments=ladders.clone(),
            funding=torch.tensor([[16.0] + [0.0] * 179], dtype=torch.float64),
            time=1,
            transitions=60,
        )
    )

    assert first_action.investments.shape == (1, 13)
    assert first_action.funding.shape == (1, 16)
    assert torch.all(first_action.investments >= 0.0)
    assert torch.all(first_action.funding >= 0.0)
    assert not torch.allclose(first_action.investments, second_action.investments)
    assert not torch.allclose(first_action.funding, second_action.funding)
    assert torch.allclose(second_action.investments, repeated_second_action.investments)
    assert torch.allclose(second_action.funding, repeated_second_action.funding)
    assert policy.audit_metadata["decision_dates"] == 60
    assert policy.audit_metadata["parameter_count"] == 1_860
    assert policy.audit_metadata["investment_maturities_per_date"] == 13
    assert policy.audit_metadata["funding_maturities_per_date"] == 16

    final_policy = BMDatePolicy(transitions=180, dtype=torch.float64)
    final_action = final_policy(
        TreasuryPolicyState(
            investments=ladders,
            funding=torch.tensor([[16.0] + [0.0] * 179], dtype=torch.float64),
            time=179,
            transitions=180,
        )
    )
    assert final_action.investments.shape == (1, 13)
    assert final_action.funding.shape == (1, 16)
    assert final_policy.audit_metadata["parameter_count"] == 5_580


def test_date_benchmark_initial_action_ignores_future_market_scenario() -> None:
    """Only time zero shares a ladder when the future scenario paths diverge."""
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    baseline = np.broadcast_to(
        historical.initial_curve.discount_factors, (1, 61, 180)
    ).copy()
    shifted = baseline.copy()
    shifted[:, 1:] *= 0.95
    policy = BMDatePolicy(transitions=60, dtype=torch.float64)

    baseline_result = ALMSimulator().rollout(
        snapshot,
        SimpleNamespace(
            discount_factors=baseline,
            initial_curve_identity=snapshot.initial_curve_identity,
            as_of_date=snapshot.as_of_date,
        ),
        policy=policy,
    )
    shifted_result = ALMSimulator().rollout(
        snapshot,
        SimpleNamespace(
            discount_factors=shifted,
            initial_curve_identity=snapshot.initial_curve_identity,
            as_of_date=snapshot.as_of_date,
        ),
        policy=policy,
    )

    assert baseline_result.treasury_actions is not None
    assert shifted_result.treasury_actions is not None
    assert torch.allclose(
        baseline_result.treasury_actions[:, 0], shifted_result.treasury_actions[:, 0]
    )


@pytest.mark.parametrize("policy_type", [BMEqualPolicy, BMConstantPolicy])
def test_benchmark_actions_do_not_depend_on_market_scenario(
    policy_type: type[torch.nn.Module],
) -> None:
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    baseline = np.broadcast_to(
        historical.initial_curve.discount_factors, (1, 3, 180)
    ).copy()
    shifted = baseline.copy()
    shifted[:, 1:] *= 0.95
    policy = policy_type(dtype=torch.float64)

    baseline_result = ALMSimulator().rollout(
        snapshot,
        SimpleNamespace(
            discount_factors=baseline,
            initial_curve_identity=snapshot.initial_curve_identity,
            as_of_date=snapshot.as_of_date,
        ),
        policy=policy,
    )
    shifted_result = ALMSimulator().rollout(
        snapshot,
        SimpleNamespace(
            discount_factors=shifted,
            initial_curve_identity=snapshot.initial_curve_identity,
            as_of_date=snapshot.as_of_date,
        ),
        policy=policy,
    )

    assert baseline_result.treasury_actions is not None
    assert shifted_result.treasury_actions is not None
    assert torch.allclose(
        baseline_result.treasury_actions, shifted_result.treasury_actions
    )


def test_bme_policy_is_recomputed_inside_rollout_and_remains_differentiable() -> None:
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    discounts = np.broadcast_to(
        historical.initial_curve.discount_factors, (1, 3, 180)
    ).copy()
    policy = BMEqualPolicy(dtype=torch.float64)

    result = ALMSimulator().rollout(
        snapshot,
        SimpleNamespace(
            discount_factors=discounts,
            initial_curve_identity=snapshot.initial_curve_identity,
            as_of_date=snapshot.as_of_date,
        ),
        policy=policy,
    )
    result.equity[:, -1].sum().backward()

    assert result.treasury_actions is not None
    assert result.treasury_actions.shape == (1, 2, 29)
    assert policy.investment_scale_adjustment.grad is not None
    assert policy.investment_scale_adjustment.grad.abs().item() > 0.0
    with pytest.raises(RunoffSimulationError, match="either actions or policy"):
        ALMSimulator().rollout(
            snapshot,
            SimpleNamespace(
                discount_factors=discounts,
                initial_curve_identity=snapshot.initial_curve_identity,
                as_of_date=snapshot.as_of_date,
            ),
            actions=torch.zeros((1, 2, 29), dtype=torch.float64),
            policy=policy,
        )
