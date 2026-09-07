from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepalm.policies import BMConstantPolicy, BMEqualPolicy, TreasuryPolicyState
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
        snapshot, SimpleNamespace(discount_factors=baseline), policy=policy
    )
    shifted_result = ALMSimulator().rollout(
        snapshot, SimpleNamespace(discount_factors=shifted), policy=policy
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
        SimpleNamespace(discount_factors=discounts),
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
            SimpleNamespace(discount_factors=discounts),
            actions=torch.zeros((1, 2, 29), dtype=torch.float64),
            policy=policy,
        )
