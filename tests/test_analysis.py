from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from deepalm.analysis import (
    FrozenPolicyTrajectory,
    HorizonAnalysisError,
    HorizonScenarioAnalyzer,
)
from deepalm.term_structures import MarketScenarioBatch


def _five_year_market(paths: int = 5) -> MarketScenarioBatch:
    spot_rates = np.full((paths, 61, 180), 0.03, dtype=np.float64)
    spot_rates[:, -1, -1] = np.linspace(0.04, 0.08, paths)
    discounts = np.exp(-spot_rates * (np.arange(1, 181) / 12.0))
    forwards = np.zeros_like(spot_rates)
    return MarketScenarioBatch(
        spot_rates=spot_rates,
        discount_factors=discounts,
        monthly_forwards=forwards,
        innovations=np.zeros((paths, 60, 3), dtype=np.float64),
        horizon_years=5,
        seed=7,
        calibration_identity="test-calibration",
        round_trip_error=0.0,
        initial_curve_identity="test-curve",
        as_of_date="2022-07-15",
        global_path_indices=tuple(range(paths)),
        deposit_initial_history_identity="test-dated-deposit-history",
    )


def _fifteen_year_market(five_year: MarketScenarioBatch) -> MarketScenarioBatch:
    paths = len(five_year.spot_rates)
    spots = np.repeat(five_year.spot_rates[:, -1:, :], 181, axis=1)
    spots[:, :61, :] = five_year.spot_rates
    return MarketScenarioBatch(
        spot_rates=spots,
        discount_factors=np.exp(-spots * (np.arange(1, 181) / 12.0)),
        monthly_forwards=np.zeros_like(spots),
        innovations=np.zeros((paths, 180, 3), dtype=np.float64),
        horizon_years=15,
        seed=five_year.seed,
        calibration_identity=five_year.calibration_identity,
        round_trip_error=0.0,
        initial_curve_identity=five_year.initial_curve_identity,
        as_of_date=five_year.as_of_date,
        split=five_year.split,
        epoch=five_year.epoch,
        global_path_indices=five_year.global_path_indices,
        deposit_initial_history_identity=five_year.deposit_initial_history_identity,
    )


def _trajectory(
    label: str,
    actions: torch.Tensor,
    horizon_years: int,
    market: MarketScenarioBatch,
    *,
    checkpoint_identity: str,
    optimizer_updates: int = 1,
) -> FrozenPolicyTrajectory:
    return FrozenPolicyTrajectory(
        policy_label=label,
        actions=actions,
        source_horizon_years=horizon_years,
        evaluation_market=market,
        checkpoint_identity=checkpoint_identity,
        optimizer_updates=optimizer_updates,
        time_feature_horizon_years=horizon_years,
    )


def test_horizon_analysis_reports_normalized_turnover_and_terminal_concentration() -> None:
    five_year_actions = torch.zeros((5, 60, 1), dtype=torch.float64)
    five_year_actions[:, 1:36, 0] = 2.0
    five_year_actions[:, 36:, 0] = 4.0
    fifteen_year_actions = torch.cat(
        (five_year_actions, torch.full((5, 120, 1), 4.0, dtype=torch.float64)),
        dim=1,
    )
    five_year_market = _five_year_market()
    fifteen_year_market = _fifteen_year_market(five_year_market)
    analyzer = HorizonScenarioAnalyzer(five_year_market)

    result = analyzer.analyze(
        mm_five_year=_trajectory(
            "MM(5y)", five_year_actions, 5, five_year_market, checkpoint_identity="MM5"
        ),
        mm_fifteen_year=_trajectory(
            "MM(15y)", fifteen_year_actions, 15, fifteen_year_market, checkpoint_identity="MM15"
        ),
        mm_fifteen_year_truncated=_trajectory(
            "MM(15y|5y)", five_year_actions, 15, five_year_market,
            checkpoint_identity="MM15", optimizer_updates=0,
        ),
        bootstrap_seed=19,
    )

    five_year = result.metrics_by_policy["MM(5y)"]
    assert five_year.analysis_window_months == 60
    assert five_year.units == "mCHF"
    assert torch.allclose(
        five_year.normalized_action_turnover,
        torch.full((5,), 4.0 / 166.0, dtype=torch.float64),
    )
    assert torch.allclose(
        five_year.terminal_turnover_concentration,
        torch.full((5,), 0.5, dtype=torch.float64),
    )
    assert len(result.paired_intervals) == 6
    assert all(interval.resamples == 100 for interval in result.paired_intervals)


def test_horizon_analysis_preserves_paper_category_rules_and_tied_ordering() -> None:
    market = _five_year_market(paths=6)
    terminal_slopes = np.array([0.07, 0.07, 0.06, 0.05, 0.04, 0.03])
    market.spot_rates[:, -1, 0] = 0.03
    market.spot_rates[:, -1, -1] = 0.03 + terminal_slopes
    market = replace(
        market,
        discount_factors=np.exp(
            -market.spot_rates * (np.arange(1, 181) / 12.0)
        ),
        global_path_indices=(101, 102, 103, 104, 105, 106),
    )
    five_year_actions = torch.ones((6, 60, 1), dtype=torch.float64)
    fifteen_year_actions = torch.ones((6, 180, 1), dtype=torch.float64)
    fifteen_year_market = _fifteen_year_market(market)
    analyzer = HorizonScenarioAnalyzer(market)

    result = analyzer.analyze(
        mm_five_year=_trajectory(
            "MM(5y)", five_year_actions, 5, market, checkpoint_identity="MM5"
        ),
        mm_fifteen_year=_trajectory(
            "MM(15y)", fifteen_year_actions, 15, fifteen_year_market, checkpoint_identity="MM15"
        ),
        mm_fifteen_year_truncated=_trajectory(
            "MM(15y|5y)", five_year_actions, 15, market,
            checkpoint_identity="MM15", optimizer_updates=0,
        ),
        bootstrap_seed=19,
    )

    steep = next(
        item
        for item in result.category_outputs
        if item.category == "steep" and item.policy_label == "MM(5y)"
    )
    upward = next(
        item
        for item in result.category_outputs
        if item.category == "upward" and item.policy_label == "MM(5y)"
    )
    assert steep.path_indices == (101, 102, 103, 104, 105)
    assert upward.path_indices == steep.path_indices
    assert "15Y - 1M" in steep.path_rule
    assert "above 2%" in upward.path_rule
    assert steep.sample_size == 5
    assert steep.source_horizon_years == 5
    assert steep.analysis_window_months == 60
    assert steep.units == "mCHF"
    assert len(result.category_outputs) == 15
    assert {item.category for item in result.category_outputs} == {
        "steep",
        "upward",
        "downward",
        "inverted",
        "constant-steepness",
    }

    with pytest.raises(ValueError, match="exceeds available paths"):
        analyzer.analyze(
            mm_five_year=_trajectory(
                "MM(5y)", five_year_actions, 5, market, checkpoint_identity="MM5"
            ),
            mm_fifteen_year=_trajectory(
                "MM(15y)", fifteen_year_actions, 15, fifteen_year_market, checkpoint_identity="MM15"
            ),
            mm_fifteen_year_truncated=_trajectory(
                "MM(15y|5y)", five_year_actions, 15, market,
                checkpoint_identity="MM15", optimizer_updates=0,
            ),
            bootstrap_seed=19,
            category_size=7,
        )


def test_horizon_analysis_handles_zero_actions_and_ineligible_upward_paths() -> None:
    market = _five_year_market()
    zero_actions = torch.zeros((5, 60, 1), dtype=torch.float64)
    fifteen_year_market = _fifteen_year_market(market)
    analyzer = HorizonScenarioAnalyzer(market)

    result = analyzer.analyze(
        mm_five_year=_trajectory(
            "MM(5y)", zero_actions, 5, market, checkpoint_identity="MM5"
        ),
        mm_fifteen_year=_trajectory(
            "MM(15y)", torch.zeros((5, 180, 1), dtype=torch.float64), 15,
            fifteen_year_market, checkpoint_identity="MM15"
        ),
        mm_fifteen_year_truncated=_trajectory(
            "MM(15y|5y)", zero_actions, 15, market,
            checkpoint_identity="MM15", optimizer_updates=0,
        ),
        bootstrap_seed=19,
    )

    assert torch.equal(
        result.metrics_by_policy["MM(5y)"].normalized_action_turnover,
        torch.zeros(5, dtype=torch.float64),
    )
    assert torch.isnan(
        result.metrics_by_policy["MM(5y)"].terminal_turnover_concentration
    ).all()
    assert result.metrics_by_policy["MM(5y)"].terminal_concentration_status == "unavailable"

    ineligible_market = replace(market, spot_rates=market.spot_rates.copy())
    ineligible_market.spot_rates[0, -1, 0] = 0.02
    ineligible_market = replace(
        ineligible_market,
        discount_factors=np.exp(
            -ineligible_market.spot_rates * (np.arange(1, 181) / 12.0)
        ),
    )
    ineligible_analyzer = HorizonScenarioAnalyzer(ineligible_market)
    with pytest.raises(ValueError, match="above 2%"):
        ineligible_analyzer.analyze(
            mm_five_year=_trajectory(
                "MM(5y)", zero_actions, 5, ineligible_market, checkpoint_identity="MM5"
            ),
            mm_fifteen_year=_trajectory(
                "MM(15y)", torch.zeros((5, 180, 1), dtype=torch.float64), 15,
                _fifteen_year_market(ineligible_market), checkpoint_identity="MM15"
            ),
            mm_fifteen_year_truncated=_trajectory(
                "MM(15y|5y)", zero_actions, 15, ineligible_market,
                checkpoint_identity="MM15", optimizer_updates=0,
            ),
            bootstrap_seed=19,
        )


def test_horizon_analysis_rejects_misaligned_paths_and_retrained_truncation() -> None:
    five_year_market = _five_year_market()
    fifteen_year_market = _fifteen_year_market(five_year_market)
    five_year_actions = torch.ones((5, 60, 1), dtype=torch.float64)
    fifteen_year_actions = torch.ones((5, 180, 1), dtype=torch.float64)
    analyzer = HorizonScenarioAnalyzer(five_year_market)

    with pytest.raises(HorizonAnalysisError, match="scenario path identity"):
        analyzer.analyze(
            mm_five_year=_trajectory(
                "MM(5y)", five_year_actions, 5, five_year_market, checkpoint_identity="MM5"
            ),
            mm_fifteen_year=_trajectory(
                "MM(15y)", fifteen_year_actions, 15,
                replace(fifteen_year_market, global_path_indices=(4, 3, 2, 1, 0)),
                checkpoint_identity="MM15",
            ),
            mm_fifteen_year_truncated=_trajectory(
                "MM(15y|5y)", five_year_actions, 15, five_year_market,
                checkpoint_identity="MM15", optimizer_updates=0,
            ),
            bootstrap_seed=19,
        )

    with pytest.raises(HorizonAnalysisError, match="scenario path identity"):
        analyzer.analyze(
            mm_five_year=_trajectory(
                "MM(5y)", five_year_actions, 5, five_year_market, checkpoint_identity="MM5"
            ),
            mm_fifteen_year=_trajectory(
                "MM(15y)", fifteen_year_actions, 15,
                replace(
                    fifteen_year_market,
                    deposit_initial_history_identity="other-dated-deposit-history",
                ),
                checkpoint_identity="MM15",
            ),
            mm_fifteen_year_truncated=_trajectory(
                "MM(15y|5y)", five_year_actions, 15, five_year_market,
                checkpoint_identity="MM15", optimizer_updates=0,
            ),
            bootstrap_seed=19,
        )

    with pytest.raises(HorizonAnalysisError, match="zero optimizer updates"):
        analyzer.analyze(
            mm_five_year=_trajectory(
                "MM(5y)", five_year_actions, 5, five_year_market, checkpoint_identity="MM5"
            ),
            mm_fifteen_year=_trajectory(
                "MM(15y)", fifteen_year_actions, 15, fifteen_year_market, checkpoint_identity="MM15"
            ),
            mm_fifteen_year_truncated=_trajectory(
                "MM(15y|5y)", five_year_actions, 15, five_year_market,
                checkpoint_identity="MM15", optimizer_updates=1,
            ),
            bootstrap_seed=19,
        )

    mismatched_discounts = fifteen_year_market.discount_factors.copy()
    mismatched_discounts[0, 1, 0] *= 0.99
    with pytest.raises(HorizonAnalysisError, match="scenario path identity"):
        analyzer.analyze(
            mm_five_year=_trajectory(
                "MM(5y)", five_year_actions, 5, five_year_market, checkpoint_identity="MM5"
            ),
            mm_fifteen_year=_trajectory(
                "MM(15y)", fifteen_year_actions, 15,
                replace(fifteen_year_market, discount_factors=mismatched_discounts),
                checkpoint_identity="MM15",
            ),
            mm_fifteen_year_truncated=_trajectory(
                "MM(15y|5y)", five_year_actions, 15, five_year_market,
                checkpoint_identity="MM15", optimizer_updates=0,
            ),
            bootstrap_seed=19,
        )
