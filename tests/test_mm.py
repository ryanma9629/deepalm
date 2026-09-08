from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deepalm import mm
from deepalm.baselines import FrozenDateBenchmarkReference
from deepalm.config import ArchitectureConfiguration
from deepalm.mm import (
    CurveFeaturePCA,
    MMObservationError,
    MMPolicy,
    RegisteredTrainingCurves,
)
from deepalm.objective import evaluation_objective_parameters
from deepalm.policies import BMDatePolicy, TreasuryPolicyState
from deepalm.reference_bank import ReferenceBankProvider
from deepalm.runoff import ALMSimulator
from deepalm.semantics import FINANCIAL_SEMANTICS_VERSION, artifact_semantics
from deepalm.term_structures import MarketScenarioModel
from deepalm.treasury import apply_treasury_action

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
)


def _training_curves(*, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    tenors = torch.linspace(0.0, 1.0, 180, dtype=dtype)
    paths = torch.arange(3, dtype=dtype).reshape(3, 1, 1)
    times = torch.arange(4, dtype=dtype).reshape(1, 4, 1)
    return (
        0.01
        + 0.003 * paths
        + 0.002 * times
        + 0.004 * tenors
        + 0.001 * paths * tenors.square()
    )


def _registered_training_curves(
    curves: torch.Tensor | None = None,
    *,
    data_identity: str = "training-data-v1",
    calibration_identity: str = "calibration-v1",
) -> RegisteredTrainingCurves:
    return RegisteredTrainingCurves(
        curves=curves if curves is not None else _training_curves(),
        data_identity=data_identity,
        calibration_identity=calibration_identity,
    )


def _state(*, requires_grad: bool = False) -> TreasuryPolicyState:
    options = {"dtype": torch.float64, "requires_grad": requires_grad}
    paths = 2
    ladder = lambda amount: torch.full((paths, 180), amount, **options)
    scalar = lambda amount: torch.full((paths,), amount, **options)
    return TreasuryPolicyState(
        investments=ladder(2.0),
        funding=ladder(1.0),
        mortgages=ladder(3.0),
        enterprise_loans=ladder(4.0),
        non_maturity_deposits=ladder(5.0),
        term_deposits=ladder(6.0),
        cash=scalar(7.0),
        curve=_training_curves()[:paths, 0].clone().requires_grad_(requires_grad),
        prior_constraint_values=torch.full((paths, 6), 1.0, **options),
        mu=scalar(0.04),
        penalty_weight=scalar(3.5),
        time=12,
        transitions=60,
    )


def _frozen_reference(tmp_path: Path) -> FrozenDateBenchmarkReference:
    tmp_path.mkdir(parents=True, exist_ok=True)
    baseline = BMDatePolicy(transitions=60, dtype=torch.float64)
    checkpoint_path = tmp_path / "BM_D_5y.pt"
    torch.save(
        {
            "policy": "BM^D",
            "code_identity": {
                "financial_semantics_version": FINANCIAL_SEMANTICS_VERSION,
                "artifact_semantics": artifact_semantics("training"),
            },
            "horizon_years": 5,
            "configuration_identity": "configuration-v1",
            "data_identities": {"market": "market-v1"},
            "policy_metadata": baseline.audit_metadata,
            "policy_state": baseline.state_dict(),
        },
        checkpoint_path,
    )
    return FrozenDateBenchmarkReference.freeze(checkpoint_path)


def test_curve_feature_pca_is_centered_persistable_and_identity_checked() -> None:
    curves = _training_curves()
    registered = _registered_training_curves(curves)
    transform = CurveFeaturePCA.fit(registered)

    factors = transform.transform(curves[:, 0])
    expected = (curves[:, 0] - transform.center) @ transform.projection
    restored = CurveFeaturePCA.from_dict(
        transform.to_dict(),
        expected_data_identity="training-data-v1",
        expected_calibration_identity="calibration-v1",
        expected_training_state_identity=registered.identity,
    )

    assert transform.center.shape == (180,)
    assert transform.projection.shape == (180, 3)
    assert len(transform.sample_indices) == 12
    assert torch.allclose(factors, expected)
    assert torch.allclose(restored.transform(curves[:, 0]), factors)
    with pytest.raises(MMObservationError, match="data identity"):
        CurveFeaturePCA.from_dict(
            transform.to_dict(),
            expected_data_identity="other-data",
            expected_calibration_identity="calibration-v1",
            expected_training_state_identity=registered.identity,
        )
    with pytest.raises(MMObservationError, match="registered training"):
        CurveFeaturePCA.fit(curves)  # type: ignore[arg-type]


def test_large_curve_pca_sample_stratifies_paths_and_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, times, cap = 100, 2, 50
    tenors = torch.linspace(0.0, 1.0, 180, dtype=torch.float64)
    curves = (
        0.01
        + torch.arange(paths, dtype=torch.float64).reshape(paths, 1, 1) / 10_000
        + torch.arange(times, dtype=torch.float64).reshape(1, times, 1) / 1_000
        + tenors.reshape(1, 1, 180) / 100
    )
    monkeypatch.setattr(mm, "_MAX_CURVE_TRAINING_STATES", cap)

    transform = CurveFeaturePCA.fit(_registered_training_curves(curves))

    assert len(transform.sample_indices) == cap
    assert {time for _path, time in transform.sample_indices} == {0, 1}
    assert [time for _path, time in transform.sample_indices] == [0] * 25 + [1] * 25


def test_mm_policy_uses_compact_or_paper_widths_to_create_legal_actions(
    tmp_path: Path,
) -> None:
    transform = CurveFeaturePCA.fit(_registered_training_curves())
    compact = MMPolicy(
        reference=_frozen_reference(tmp_path / "compact"),
        curve_features=transform,
        architecture=ArchitectureConfiguration(
            profile="compact", widths=(64, 64, 32, 32)
        ),
    )
    paper = MMPolicy(
        reference=_frozen_reference(tmp_path / "paper"),
        curve_features=transform,
        architecture=ArchitectureConfiguration(
            profile="paper", widths=(512, 512, 256, 128)
        ),
    )

    action = compact(_state())

    assert compact.build_observation(_state()).shape == (2, 145)
    assert action.concatenated.shape == (2, 29)
    assert torch.all(action.concatenated >= 0)
    assert compact.audit_metadata["widths"] == (64, 64, 32, 32)
    assert paper.audit_metadata["widths"] == (512, 512, 256, 128)
    assert (
        compact.final_encoding.out_features == paper.final_encoding.out_features == 64
    )


def test_mm_stop_gradient_preserves_forward_values_and_policy_gradient(
    tmp_path: Path,
) -> None:
    transform = CurveFeaturePCA.fit(_registered_training_curves())
    policy = MMPolicy(
        reference=_frozen_reference(tmp_path),
        curve_features=transform,
        architecture=ArchitectureConfiguration(
            profile="compact", widths=(64, 64, 32, 32)
        ),
    )
    differentiable_state = _state(requires_grad=True)
    detached_state = _state()

    future_loss = policy(differentiable_state).concatenated.square().sum()
    forward_action = policy(differentiable_state).concatenated.detach()
    detached_action = policy(detached_state).concatenated.detach()
    future_loss.backward()

    assert torch.allclose(forward_action, detached_action)
    assert differentiable_state.investments.grad is None
    assert differentiable_state.curve.grad is None
    assert differentiable_state.prior_constraint_values.grad is None
    assert policy.investment_encoder.weight.grad is not None
    assert torch.any(policy.investment_encoder.weight.grad != 0)
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in policy.parameters()
        if parameter.requires_grad
    )


def test_mm_action_keeps_gradient_through_a_future_financial_transition(
    tmp_path: Path,
) -> None:
    policy = MMPolicy(
        reference=_frozen_reference(tmp_path),
        curve_features=CurveFeaturePCA.fit(_registered_training_curves()),
        architecture=ArchitectureConfiguration(
            profile="compact", widths=(64, 64, 32, 32)
        ),
    )
    curve = _state().curve.clone().requires_grad_(True)
    prior_constraints = _state().prior_constraint_values.clone().requires_grad_(True)
    state = replace(_state(), curve=curve, prior_constraint_values=prior_constraints)
    discounts = torch.exp(
        -torch.arange(1, 181, dtype=torch.float64).unsqueeze(0) / 12_000.0
    ).expand(2, -1)
    future_discounts = discounts * torch.exp(
        -0.01 * torch.arange(1, 181, dtype=torch.float64).unsqueeze(0) / 12.0
    )

    action = policy(state)
    next_investments, next_funding, cash_settlement = apply_treasury_action(
        state.investments,
        state.funding,
        discounts,
        action,
    )
    terminal_loss = (
        (next_investments * future_discounts).sum()
        - (next_funding * future_discounts).sum()
        + cash_settlement.sum()
    )
    terminal_loss.backward()

    assert action.concatenated.requires_grad
    assert curve.grad is None
    assert prior_constraints.grad is None
    assert policy.investment_encoder.weight.grad is not None
    assert torch.any(policy.investment_encoder.weight.grad != 0)


def test_mm_requires_an_identity_checked_reference_and_complete_observations(
    tmp_path: Path,
) -> None:
    transform = CurveFeaturePCA.fit(_registered_training_curves())
    with pytest.raises(TypeError, match="baseline"):
        MMPolicy(
            baseline=BMDatePolicy(transitions=60, dtype=torch.float64),
            baseline_identity="frozen-baseline-v1",
            curve_features=transform,
            architecture=ArchitectureConfiguration(
                profile="compact", widths=(64, 64, 32, 32)
            ),
        )

    policy = MMPolicy(
        reference=_frozen_reference(tmp_path),
        curve_features=transform,
        architecture=ArchitectureConfiguration(
            profile="compact", widths=(64, 64, 32, 32)
        ),
    )
    incomplete = TreasuryPolicyState(
        investments=torch.ones((1, 180), dtype=torch.float64),
        funding=torch.ones((1, 180), dtype=torch.float64),
        time=0,
        transitions=60,
    )
    with pytest.raises(MMObservationError, match="complete MM observation"):
        policy(incomplete)
    incompatible_dtype = replace(_state(), funding=_state().funding.float())
    with pytest.raises(MMObservationError, match="device and dtype"):
        policy(incompatible_dtype)


def test_mm_loads_its_baseline_from_an_identity_checked_frozen_reference(
    tmp_path: Path,
) -> None:
    reference = _frozen_reference(tmp_path)
    transform = CurveFeaturePCA.fit(_registered_training_curves())

    policy = MMPolicy(
        reference=reference,
        curve_features=transform,
        architecture=ArchitectureConfiguration(
            profile="compact", widths=(64, 64, 32, 32)
        ),
        device="cpu",
        dtype=torch.float64,
    )

    assert policy.baseline_identity == reference.reference_identity
    assert not any(
        parameter.requires_grad for parameter in policy.baseline.parameters()
    )


def test_mm_policy_receives_complete_live_bank_state_during_rollout(
    tmp_path: Path,
) -> None:
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    transform = CurveFeaturePCA.fit(
        _registered_training_curves(
            torch.tensor(historical.spot_rates[-12:], dtype=torch.float64),
            data_identity=historical.source_hash,
            calibration_identity="registered-training-states-v1",
        )
    )
    torch.manual_seed(7)
    policy = MMPolicy(
        reference=_frozen_reference(tmp_path),
        curve_features=transform,
        architecture=ArchitectureConfiguration(
            profile="compact", widths=(64, 64, 32, 32)
        ),
    )
    discounts = np.broadcast_to(
        historical.initial_curve.discount_factors, (1, 61, 180)
    ).copy()
    spots = np.broadcast_to(historical.initial_curve.spot_rates, (1, 61, 180)).copy()

    result = ALMSimulator().rollout(
        snapshot,
        SimpleNamespace(
            discount_factors=discounts,
            spot_rates=spots,
            horizon_years=5,
            initial_curve_identity=snapshot.initial_curve_identity,
            as_of_date=snapshot.as_of_date,
            deposit_initial_history_identity=snapshot.deposit_initial_history.identity,
        ),
        policy=policy,
        include_loan_dynamics=True,
        include_deposit_dynamics=True,
        objective_parameters=evaluation_objective_parameters(1, horizon_years=5),
    )

    assert result.treasury_actions is not None
    assert result.treasury_actions.shape == (1, 60, 29)
    assert result.constraint_values is not None
    result.equity[:, -1].sum().backward()
    assert policy.investment_encoder.weight.grad is not None
    assert torch.any(policy.investment_encoder.weight.grad != 0)
