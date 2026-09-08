from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_mm_training import SOURCE, _configuration, _frozen_baseline

from deepalm.evaluation import (
    LockedEvaluationError,
    LockedEvaluator,
    PolicyCheckpoint,
    _constraint_report,
    paired_bootstrap,
    signed_tail_metrics,
)
from deepalm.policies import BMEqualPolicy
from deepalm.reference_bank import ReferenceBankProvider
from deepalm.term_structures import MarketScenarioModel
from deepalm.training import MMTrainer, _job_seed


def test_signed_tail_metrics_keep_loss_sign_and_paired_bootstrap_is_reproducible() -> (
    None
):
    returns = torch.tensor([-0.20, -0.10, 0.01, 0.05], dtype=torch.float64)

    tails = signed_tail_metrics(returns)
    first = paired_bootstrap(returns, returns + 0.01, seed=91, resamples=100)
    second = paired_bootstrap(returns, returns + 0.01, seed=91, resamples=100)

    assert tails["signed_var_95"] is not None and tails["signed_var_95"] < 0
    assert tails["signed_es_95"] is not None and tails["signed_es_95"] < 0
    assert first == second
    assert first[0] == pytest.approx(-0.01)


def test_locked_evaluation_uses_disjoint_common_test_paths_and_writes_manifest(
    tmp_path: Path,
) -> None:
    configuration = _configuration(tmp_path)
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    reference = _frozen_baseline(configuration, model, tmp_path)
    bme_checkpoint = torch.load(reference.checkpoint_path, weights_only=False)
    bme_policy = BMEqualPolicy(dtype=torch.float64)
    bme_checkpoint.update(
        policy="BM^E",
        policy_state=bme_policy.state_dict(),
        policy_metadata={},
    )
    for split in ("training", "selection"):
        bme_checkpoint["scenario_identities"][split]["job_seed"] = _job_seed(
            configuration.seeds["market_scenarios"], "BM^E", 5
        )
    bme_path = tmp_path / "BM_E_5y.pt"
    torch.save(bme_checkpoint, bme_path)
    evaluator = LockedEvaluator(
        configuration, snapshot=snapshot, historical=historical, calibration=calibration
    )

    result = evaluator.evaluate(
        (
            PolicyCheckpoint("BM^D", reference.checkpoint_path),
            PolicyCheckpoint("BM^E", bme_path),
        )
    )
    manifest_path = tmp_path / "locked-evaluation.json"
    evaluator.write_manifest(result, manifest_path)

    assert result.markets[5].split == "test"
    assert result.markets[5].global_path_indices == (0, 1)
    assert result.reports["BM^D"]["annualized_return"]["status"] == "unavailable"
    assert result.reports["BM^D"]["annualized_return"]["reason"]
    assert result.reports["BM^D"]["risk"]["status"] == "unavailable"
    assert result.manifest["bootstrap_resamples"] == 100
    assert result.paired_intervals
    assert all(
        interval.resamples == 100 and interval.paths == 2
        for interval in result.paired_intervals
    )
    annualized_pair = next(
        interval
        for interval in result.paired_intervals
        if interval.metric == "annualized_return"
    )
    assert annualized_pair.status == "unavailable"
    assert annualized_pair.reason
    assert manifest_path.is_file()

    with pytest.raises(LockedEvaluationError, match="at least 100"):
        evaluator.evaluate(
            (PolicyCheckpoint("BM^D", reference.checkpoint_path),),
            bootstrap_resamples=99,
        )

    checkpoint = torch.load(reference.checkpoint_path, weights_only=False)
    checkpoint["data_identities"]["market_source_hash"] = "wrong"
    incompatible = tmp_path / "incompatible.pt"
    torch.save(checkpoint, incompatible)
    with pytest.raises(LockedEvaluationError, match="data identities"):
        evaluator.evaluate((PolicyCheckpoint("bad", incompatible),))

    incompatible_seed = torch.load(reference.checkpoint_path, weights_only=False)
    incompatible_seed["configuration"]["seeds"]["bootstrap"] = 999
    incompatible_seed_path = tmp_path / "incompatible-seed.pt"
    torch.save(incompatible_seed, incompatible_seed_path)
    with pytest.raises(LockedEvaluationError, match="required seeds"):
        evaluator.evaluate((PolicyCheckpoint("bad-seed", incompatible_seed_path),))

    incompatible_paths = torch.load(reference.checkpoint_path, weights_only=False)
    incompatible_paths["configuration"]["run_scale"]["test_paths"] = 999
    incompatible_paths_path = tmp_path / "incompatible-paths.pt"
    torch.save(incompatible_paths, incompatible_paths_path)
    with pytest.raises(LockedEvaluationError, match="test path count"):
        evaluator.evaluate((PolicyCheckpoint("bad-paths", incompatible_paths_path),))


def test_locked_markets_share_five_year_brownian_prefix_with_fifteen_year_market(
    tmp_path: Path,
) -> None:
    configuration = _configuration(tmp_path)
    configuration = replace(
        configuration,
        experiment=replace(configuration.experiment, horizons_years=(5, 15)),
    )
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    evaluator = LockedEvaluator(
        configuration, snapshot=snapshot, historical=historical, calibration=calibration
    )

    five_year = evaluator._locked_market(5)
    fifteen_year = evaluator._locked_market(15)

    assert np.array_equal(five_year.innovations, fifteen_year.innovations[:, :60])
    assert five_year.global_path_indices == fifteen_year.global_path_indices


def test_locked_evaluation_loads_a_frozen_mm_checkpoint(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    reference = _frozen_baseline(configuration, model, tmp_path)
    mm_checkpoint = (
        MMTrainer(
            configuration,
            snapshot=snapshot,
            historical=historical,
            calibration=calibration,
            baseline_reference=reference,
        )
        .fit(horizon_years=5)
        .checkpoint_path
    )
    evaluator = LockedEvaluator(
        configuration, snapshot=snapshot, historical=historical, calibration=calibration
    )

    result = evaluator.evaluate((PolicyCheckpoint("MM", mm_checkpoint),))

    assert result.reports["MM"]["losses"]["total"] is not None
    assert result.manifest["checkpoints"]["MM"]["policy"] == "MM"


def test_constraint_summary_uses_annual_eyr_times_and_irs_upper_bound() -> None:
    values = torch.ones((2, 4, 6), dtype=torch.float64)
    values[:, :, 4] = torch.tensor(
        [[0.01, 0.09, 0.03, 0.08], [0.04, 0.02, 0.07, 0.06]],
        dtype=torch.float64,
    )
    values[:, :, 5] = torch.tensor(
        [[99.0, 0.02, 99.0, 0.04], [99.0, 0.06, 99.0, 0.08]],
        dtype=torch.float64,
    )
    violations = torch.zeros_like(values)
    annual_mask = torch.tensor([[False, True, False, True], [False, True, False, True]])

    report = _constraint_report(
        values, violations, annual_mask, torch.tensor([0.1, 0.3])
    )

    assert report["aggregate_penalty"]["mean"] == pytest.approx(0.2)
    assert report["by_constraint"]["irs"]["worst_value"] == pytest.approx(0.09)
    assert report["by_constraint"]["eyr"]["time_mean"] == pytest.approx([0.04, 0.06])
