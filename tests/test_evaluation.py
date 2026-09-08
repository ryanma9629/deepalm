from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_mm_training import SOURCE, _configuration, _frozen_baseline

from deepalm.evaluation import (
    LockedEvaluationError,
    LockedEvaluator,
    PolicyCheckpoint,
    _constraint_report,
    _report_outcome,
    equity_distribution_report,
    equity_risk_report,
    paired_bootstrap,
    penalty_tail_metrics,
    signed_tail_metrics,
)
from deepalm.policies import BMEqualPolicy
from deepalm.reference_bank import ReferenceBankProvider
from deepalm.term_structures import MarketScenarioModel
from deepalm.training import MMTrainer, _job_seed


def _outcome_for_report(
    *, horizon_years: int, dividends: torch.Tensor
) -> SimpleNamespace:
    months = horizon_years * 12
    return SimpleNamespace(
        objective=SimpleNamespace(
            total=torch.zeros(1, dtype=torch.float64),
            target=torch.zeros(1, dtype=torch.float64),
            penalty=torch.zeros(1, dtype=torch.float64),
            crra=torch.zeros(1, dtype=torch.float64),
        ),
        constraint_values=torch.ones((1, months, 6), dtype=torch.float64),
        constraint_violations=torch.zeros((1, months, 6), dtype=torch.float64),
        constraint_annual_mask=torch.arange(months).remainder(12).eq(11).unsqueeze(0),
        equity=torch.full((1, months + 1), 1000.0, dtype=torch.float64),
        dividends=dividends,
    )


def test_equity_tail_risk_is_centered_and_translation_invariant() -> None:
    values = torch.arange(100, dtype=torch.float64)
    tails = signed_tail_metrics(values)
    assert tails["signed_var_95"] == pytest.approx(-44.55)
    assert tails["signed_es_95"] == pytest.approx(-47.5)
    assert signed_tail_metrics(values + 1000) == pytest.approx(tails)


def test_penalty_tail_risk_selects_raw_upper_tail_before_centering() -> None:
    values = torch.arange(100, dtype=torch.float64)
    assert penalty_tail_metrics(values) == pytest.approx({"var95": 44.55, "es95": 47.5})
    assert penalty_tail_metrics(values + 1000) == pytest.approx(
        penalty_tail_metrics(values)
    )


@pytest.mark.parametrize(("horizon_years", "annual_payouts"), ((5, 4), (15, 14)))
def test_evaluation_dividend_yield_uses_nonterminal_dividend_years(
    horizon_years: int, annual_payouts: int
) -> None:
    months = horizon_years * 12
    dividends = torch.zeros((1, months), dtype=torch.float64)
    dividend_months = torch.arange(annual_payouts) * 12 + 11
    dividends[0, dividend_months] = 10.0
    outcome = _outcome_for_report(horizon_years=horizon_years, dividends=dividends)

    report, _ = _report_outcome(outcome, horizon_years)

    assert report["standardized_dividend_yield"] == {
        "value": pytest.approx(0.01),
        "status": "available",
    }


def test_evaluation_marks_dividend_yield_unavailable_without_nonterminal_years() -> (
    None
):
    months = 12
    outcome = _outcome_for_report(
        horizon_years=1, dividends=torch.zeros((1, months), dtype=torch.float64)
    )

    report, _ = _report_outcome(outcome, horizon_years=1)

    assert report["standardized_dividend_yield"] == {
        "value": None,
        "status": "unavailable",
        "reason": "evaluation horizon has no nonterminal dividend years",
    }


def test_equity_risk_includes_insolvent_paths_without_annualizing() -> None:
    report = equity_risk_report(
        torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    )
    assert report["status"] == "available"
    assert report["input"] == "terminal_equity_ratio"
    assert report["signed_var_95"] == pytest.approx(-1.35)
    assert report["signed_es_95"] == pytest.approx(-1.5)


def test_equity_distribution_uses_population_central_moments() -> None:
    report = equity_distribution_report(
        torch.tensor([1.0, 1.0, 1.0, 5.0], dtype=torch.float64)
    )

    assert report["mean"] == {"value": pytest.approx(2.0), "status": "available"}
    assert report["standard_deviation"] == {
        "value": pytest.approx(3.0**0.5),
        "status": "available",
    }
    assert report["skewness"] == {
        "value": pytest.approx(2.0 / 3.0**0.5),
        "status": "available",
    }
    assert report["excess_kurtosis"] == {
        "value": pytest.approx(-2.0 / 3.0),
        "status": "available",
    }
    assert report["moment_convention"] == "population_central"
    json.dumps(report, allow_nan=False)


def test_equity_distribution_marks_undefined_or_nonfinite_moments_unavailable() -> None:
    constant = equity_distribution_report(torch.full((4,), 3.0, dtype=torch.float64))
    nonfinite = equity_distribution_report(
        torch.tensor([1.0, float("nan")], dtype=torch.float64)
    )

    assert constant["standard_deviation"] == {
        "value": pytest.approx(0.0),
        "status": "available",
    }
    assert constant["skewness"] == {
        "value": None,
        "status": "unavailable",
        "reason": "equity ratio population variance is zero",
    }
    assert constant["excess_kurtosis"] == {
        "value": None,
        "status": "unavailable",
        "reason": "equity ratio population variance is zero",
    }
    assert nonfinite["mean"] == {
        "value": None,
        "status": "unavailable",
        "reason": "terminal equity ratios are empty or non-finite",
    }
    assert equity_distribution_report(
        torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    )["mean"] == {"value": pytest.approx(0.0), "status": "available"}
    json.dumps(constant, allow_nan=False)
    json.dumps(nonfinite, allow_nan=False)


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
    assert result.reports["BM^D"]["risk"]["status"] == "available"
    assert result.reports["BM^D"]["risk"]["input"] == "terminal_equity_ratio"
    assert result.reports["BM^D"]["risk"]["centered"] is True
    assert (
        result.manifest["risk_metric_convention"]
        == "population-moments-constraints-and-report-coverage-v4"
    )
    assert result.manifest["artifact_semantics"]["corrections"] == {
        "C-1": True,
        "C-2": True,
        "C-3": True,
        "C-5": True,
        "C-6": True,
        "C-7": True,
        "C-8": True,
        "R-1": True,
        "R-2": True,
    }
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
    if any(
        report["annualized_return"]["status"] == "unavailable"
        for report in result.reports.values()
    ):
        assert annualized_pair.status == "unavailable"
        assert annualized_pair.reason
    else:
        assert annualized_pair.status == "available"
    assert manifest_path.is_file()

    legacy = torch.load(reference.checkpoint_path, weights_only=False)
    legacy["code_identity"].pop("financial_semantics_version", None)
    legacy_path = tmp_path / "legacy-cash-bug.pt"
    torch.save(legacy, legacy_path)
    with pytest.raises(LockedEvaluationError, match="financial semantics"):
        evaluator.evaluate((PolicyCheckpoint("legacy", legacy_path),))

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
    assert report["by_constraint"]["irs"]["most_adverse_raw_value"] == {
        "value": pytest.approx(0.09),
        "status": "available",
    }
    assert report["by_constraint"]["eyr"]["time_mean"] == pytest.approx([0.04, 0.06])


def test_constraint_summary_separates_raw_values_and_conditional_counts() -> None:
    values = torch.ones((3, 2, 6), dtype=torch.float64)
    values[0, :, 2] = 0.94
    values[2, 0, 2] = 0.94
    violations = torch.zeros_like(values)
    violations[0, :, 2] = 0.1236
    violations[2, 0, 2] = 0.1236

    report = _constraint_report(
        values, violations, torch.ones((3, 2), dtype=torch.bool), torch.zeros(3)
    )
    cmr = report["by_constraint"]["cmr"]
    lcr = report["by_constraint"]["lcr"]

    assert cmr["raw_violating_mean"] == {
        "value": pytest.approx(0.94),
        "status": "available",
    }
    assert cmr["violation_penalty_mean"] == {
        "value": pytest.approx(0.1236),
        "status": "available",
    }
    assert cmr["violating_observation_count"] == 3
    assert cmr["ever_violating_path_share"] == pytest.approx(2.0 / 3.0)
    assert cmr["mean_violating_observations_per_violating_path"] == {
        "value": pytest.approx(1.5),
        "status": "available",
    }
    assert cmr["most_adverse_raw_value"] == {
        "value": pytest.approx(0.94),
        "status": "available",
    }
    assert cmr["last_applicable_state"] == {
        "state_index": 2,
        "month_index": 2,
        "raw_values": pytest.approx([0.94, 1.0, 1.0]),
    }
    assert cmr["eligible_observation_count"] == 6
    assert cmr["applicable_state_indices"] == [1, 2]
    assert lcr["violating_observation_count"] == 0
    assert lcr["raw_violating_mean"] == {
        "value": None,
        "status": "unavailable",
        "reason": "no violating observations",
    }
    assert lcr["mean_violating_observations_per_violating_path"] == {
        "value": None,
        "status": "unavailable",
        "reason": "no paths have violating observations",
    }
