from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from test_mm_training import SOURCE, _configuration, _frozen_baseline

from deepalm.baselines import FrozenDateBenchmarkReference
from deepalm.config import (
    ExperimentConfiguration,
    PolicyConfiguration,
    RunScaleConfiguration,
)
from deepalm.policies import TreasuryPolicyState
from deepalm.reference_bank import ReferenceBankProvider
from deepalm.term_structures import MarketScenarioModel
from deepalm.training import BMDateTrainer, MMTrainer


def _fifteen_configuration(tmp_path: Path):
    configuration = _configuration(tmp_path)
    return replace(
        configuration,
        experiment=ExperimentConfiguration(horizons_years=(15,), include_swaps=False),
        policy=PolicyConfiguration(names=("BM^D", "MM")),
        run_scale=RunScaleConfiguration(
            profile="mm-15-fixture",
            epochs=2,
            training_paths_per_epoch=4,
            selection_paths=2,
            test_paths=2,
            batch_size=2,
        ),
        output=replace(configuration.output, run_name="mm-15-test"),
    )


def _state(*, time: int, transitions: int) -> TreasuryPolicyState:
    paths = 2
    ladder = lambda amount: torch.full((paths, 180), amount, dtype=torch.float64)
    return TreasuryPolicyState(
        investments=ladder(2.0),
        funding=ladder(1.0),
        mortgages=ladder(3.0),
        enterprise_loans=ladder(4.0),
        non_maturity_deposits=ladder(5.0),
        term_deposits=ladder(6.0),
        cash=torch.full((paths,), 7.0, dtype=torch.float64),
        curve=torch.linspace(0.01, 0.03, 180, dtype=torch.float64).repeat(paths, 1),
        discounts=torch.full((paths, 180), 0.95, dtype=torch.float64),
        initial_assets=torch.full((paths,), 100.0, dtype=torch.float64),
        prior_constraint_values=torch.ones((paths, 6), dtype=torch.float64),
        mu=torch.full((paths,), 0.04, dtype=torch.float64),
        penalty_weight=torch.full((paths,), 3.5, dtype=torch.float64),
        time=time,
        transitions=transitions,
    )


def _frozen_fifteen_baseline(configuration, model, tmp_path: Path):
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    baseline = BMDateTrainer(
        replace(
            configuration,
            policy=PolicyConfiguration(names=("BM^D",)),
            output=replace(configuration.output, run_name="bmd-15-source"),
        ),
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
    ).fit(horizon_years=15)
    assert baseline.baseline_reference_path is not None
    return (
        FrozenDateBenchmarkReference.load(baseline.baseline_reference_path),
        historical,
        calibration,
        snapshot,
    )


def test_fifteen_year_mm_trains_and_truncates_without_retraining(
    tmp_path: Path,
) -> None:
    configuration = _fifteen_configuration(tmp_path)
    model = MarketScenarioModel()
    reference, historical, calibration, snapshot = _frozen_fifteen_baseline(
        configuration, model, tmp_path
    )
    trainer = MMTrainer(
        configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
        baseline_reference=reference,
    )

    trained = trainer.fit(horizon_years=15)
    compact_bytes = trained.checkpoint_path.read_bytes()
    policy = trainer.load_selected_checkpoint(trained.checkpoint_path)
    standalone = MMTrainer.load_policy_from_checkpoint(trained.checkpoint_path)
    paper_check = trainer.run_paper_width_check(horizon_years=15)
    market = model.generate_hjm_scenarios(
        historical,
        calibration,
        horizon_years=15,
        paths=2,
        seed=91,
        split="truncation",
    )
    truncated = trainer.evaluate_five_year_truncation(
        checkpoint_path=trained.checkpoint_path, market=market
    )
    repeated = trainer.evaluate_five_year_truncation(
        checkpoint_path=trained.checkpoint_path, market=market
    )

    assert trained.optimizer_updates == 4
    assert paper_check.optimizer_updates == 1
    assert paper_check.paths == 2
    assert paper_check.finite and paper_check.updated and paper_check.clipped
    assert trained.checkpoint_path.read_bytes() == compact_bytes
    paper_artifact = json.loads(paper_check.artifact_path.read_text(encoding="utf-8"))
    assert paper_artifact["horizon_years"] == 15
    assert paper_artifact["optimizer_updates"] == 1
    assert paper_artifact["paths"] == 2
    assert paper_artifact["resource_profile"]["forward_backward_update_seconds"] >= 0
    assert policy.baseline.transitions == 180
    assert torch.allclose(
        policy.build_observation(_state(time=60, transitions=180))[:, -3],
        torch.full((2,), 1 / 3, dtype=torch.float64),
    )
    assert truncated.optimizer_updates == 0
    assert truncated.market.horizon_years == 5
    assert truncated.outcome.treasury_actions is not None
    assert truncated.outcome.treasury_actions.shape[1] == 60
    assert repeated.outcome.treasury_actions is not None
    assert torch.equal(
        truncated.outcome.treasury_actions, repeated.outcome.treasury_actions
    )
    assert all(
        not parameter.requires_grad for parameter in truncated.policy.parameters()
    )
    assert truncated.policy.audit_metadata["trained_horizon_years"] == 15
    assert truncated.policy.audit_metadata["truncated_decisions"] == 60
    assert torch.equal(
        truncated.policy(_state(time=59, transitions=60)).concatenated,
        policy(_state(time=59, transitions=180)).concatenated,
    )
    assert torch.equal(
        standalone(_state(time=59, transitions=180)).concatenated,
        policy(_state(time=59, transitions=180)).concatenated,
    )
    assert np.array_equal(market.innovations[:, :60], truncated.market.innovations)
    assert np.allclose(
        market.monthly_forwards[:, :61], truncated.market.monthly_forwards
    )

    five_configuration = _configuration(tmp_path / "five-year")
    five_reference = _frozen_baseline(five_configuration, model, tmp_path / "five-year")
    five_policy = (
        MMTrainer(
            five_configuration,
            snapshot=snapshot,
            historical=historical,
            calibration=calibration,
            baseline_reference=five_reference,
        )
        .fit(horizon_years=5)
        .policy
    )
    assert not torch.equal(
        truncated.policy(_state(time=59, transitions=60)).concatenated,
        five_policy(_state(time=59, transitions=60)).concatenated,
    )


def test_fifteen_year_market_prefix_reproduces_five_year_paths() -> None:
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    five = model.generate_hjm_scenarios(
        historical,
        calibration,
        horizon_years=5,
        paths=2,
        seed=77,
    )
    fifteen = model.generate_hjm_scenarios(
        historical,
        calibration,
        horizon_years=15,
        paths=2,
        seed=77,
    )

    prefix = fifteen.prefix(horizon_years=5)

    assert np.array_equal(prefix.innovations, five.innovations)
    assert np.array_equal(prefix.spot_rates, five.spot_rates)
    assert np.array_equal(prefix.discount_factors, five.discount_factors)


def test_fifteen_year_local_flow_executes_eight_compact_updates(tmp_path: Path) -> None:
    base_configuration = _configuration(tmp_path, fixture=False)
    configuration = replace(
        base_configuration,
        experiment=ExperimentConfiguration(horizons_years=(15,), include_swaps=False),
        policy=PolicyConfiguration(names=("BM^D", "MM")),
        output=replace(base_configuration.output, run_name="mm-15-local"),
    )
    model = MarketScenarioModel()
    reference, historical, calibration, snapshot = _frozen_fifteen_baseline(
        configuration, model, tmp_path
    )

    result = MMTrainer(
        configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
        baseline_reference=reference,
    ).fit(horizon_years=15)

    assert result.optimizer_updates == 8
    assert result.resource_profile.horizon_years == 15
    assert result.resource_profile.forward_backward_update_seconds > 0
    assert result.checkpoint_path.is_file()
