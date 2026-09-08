"""Post-training horizon comparisons for frozen Deep ALM trajectories.

This module deliberately accepts completed action trajectories only.  It has no
trainer, optimizer, or checkpoint-selection dependency, so analysis cannot
launch a new training branch.  Normalized turnover divides consecutive-action
L1 changes by total action L1 volume plus ``_TURNOVER_EPSILON``; all-zero
actions therefore report zero rather than an undefined ratio.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType

import numpy as np
import torch

from deepalm.evaluation import LockedEvaluationError, PairedInterval, paired_bootstrap
from deepalm.term_structures import MarketScenarioBatch

_FIVE_YEAR_MONTHS = 60
_TERMINAL_MONTHS = 24
_TURNOVER_EPSILON = 1e-12
_REQUIRED_POLICIES = ("MM(5y)", "MM(15y)", "MM(15y|5y)")


class HorizonAnalysisError(LockedEvaluationError):
    """Raised when immutable trajectories cannot support a horizon comparison."""


@dataclass(frozen=True)
class FrozenPolicyTrajectory:
    """A detached CPU snapshot from one completed, frozen policy evaluation."""

    policy_label: str
    actions: torch.Tensor
    source_horizon_years: int
    evaluation_market: InitVar[MarketScenarioBatch]
    checkpoint_identity: str
    optimizer_updates: int
    time_feature_horizon_years: int
    units: str = "mCHF"
    _action_identity: str = field(init=False, repr=False)
    _market_prefix_identity: _MarketPrefixIdentity = field(init=False, repr=False)
    evaluation_market_horizon_years: int = field(init=False)

    def __post_init__(self, evaluation_market: MarketScenarioBatch) -> None:
        """Detach the caller's tensor and retain a mutation-detecting identity."""

        if not isinstance(self.actions, torch.Tensor):
            raise HorizonAnalysisError("Frozen actions must be a torch tensor")
        snapshot = self.actions.detach().to(device="cpu").clone()
        snapshot.requires_grad_(False)
        object.__setattr__(self, "actions", snapshot)
        object.__setattr__(self, "_action_identity", _tensor_identity(snapshot))
        object.__setattr__(
            self, "_market_prefix_identity", _market_prefix_identity(evaluation_market)
        )
        object.__setattr__(
            self, "evaluation_market_horizon_years", evaluation_market.horizon_years
        )


@dataclass(frozen=True)
class PolicyHorizonMetrics:
    """Pathwise turnover metrics over one fixed five-year decision window."""

    policy_label: str
    source_horizon_years: int
    analysis_window_months: int
    paths: int
    units: str
    normalized_action_turnover: torch.Tensor
    terminal_turnover_concentration: torch.Tensor
    terminal_concentration_status: str
    terminal_concentration_reason: str | None


@dataclass(frozen=True)
class ScenarioCategoryOutput:
    """One policy's descriptive result on a paper-defined path subset."""

    category: str
    policy_label: str
    path_rule: str
    path_indices: tuple[int, ...]
    sample_size: int
    source_horizon_years: int
    analysis_window_months: int
    units: str
    mean_normalized_action_turnover: float
    mean_terminal_turnover_concentration: float | None
    terminal_concentration_status: str
    terminal_concentration_reason: str | None


@dataclass(frozen=True)
class _ScenarioPathCategory:
    """Private, named category selection retained between ranking and reporting."""

    category: str
    path_rule: str
    row_indices: tuple[int, ...]
    path_indices: tuple[int, ...]


@dataclass(frozen=True)
class _MarketPrefixIdentity:
    """Immutable identity for the first 60 monthly shocks and 61 curve dates."""

    convention: str
    seed: int
    calibration_identity: str
    initial_curve_identity: str
    as_of_date: str
    split: str
    epoch: int
    global_path_indices: tuple[int, ...]
    curve_content_hash: str
    discount_content_hash: str
    forward_content_hash: str
    innovation_content_hash: str


@dataclass(frozen=True)
class HorizonScenarioAnalysis:
    """Descriptive paired evidence from the three frozen MM trajectories."""

    metrics_by_policy: Mapping[str, PolicyHorizonMetrics]
    paired_intervals: tuple[PairedInterval, ...]
    category_outputs: tuple[ScenarioCategoryOutput, ...]


class HorizonScenarioAnalyzer:
    """Compare frozen five-year action windows on common scenario paths."""

    def __init__(self, market: MarketScenarioBatch) -> None:
        if market.horizon_years != 5:
            raise HorizonAnalysisError(
                "Horizon scenario analysis requires a five-year market window"
            )
        self._market = _five_year_market_snapshot(market)
        self._market_prefix_identity = _market_prefix_identity(self._market)

    def analyze(
        self,
        *,
        mm_five_year: FrozenPolicyTrajectory,
        mm_fifteen_year: FrozenPolicyTrajectory,
        mm_fifteen_year_truncated: FrozenPolicyTrajectory,
        bootstrap_seed: int,
        bootstrap_resamples: int = 100,
        category_size: int = 5,
    ) -> HorizonScenarioAnalysis:
        """Compare MM(5y), MM(15y)'s first 60 decisions, and MM(15y|5y).

        The common 60-month window is intentional: it keeps all paired
        measurements on the same market-path prefix.  The interval is
        descriptive only and uses the local minimum of 100 paired resamples.
        """

        if bootstrap_resamples < 100:
            raise HorizonAnalysisError(
                "Horizon comparisons require at least 100 bootstrap resamples"
            )
        trajectories = (
            mm_five_year,
            mm_fifteen_year,
            mm_fifteen_year_truncated,
        )
        if tuple(item.policy_label for item in trajectories) != _REQUIRED_POLICIES:
            raise HorizonAnalysisError(
                "Horizon comparison requires MM(5y), MM(15y), and MM(15y|5y)"
            )
        _validate_trajectory_provenance(
            trajectories, analysis_identity=self._market_prefix_identity
        )

        metrics = {
            trajectory.policy_label: self._metrics_for(trajectory)
            for trajectory in trajectories
        }
        intervals = _paired_horizon_intervals(
            metrics, seed=bootstrap_seed, resamples=bootstrap_resamples
        )
        categories = _scenario_categories(self._market, size=category_size)
        category_outputs = tuple(
            _category_output(category, metrics[policy_label])
            for category in categories
            for policy_label in _REQUIRED_POLICIES
        )
        return HorizonScenarioAnalysis(
            metrics_by_policy=MappingProxyType(metrics),
            paired_intervals=intervals,
            category_outputs=category_outputs,
        )

    def _metrics_for(self, trajectory: FrozenPolicyTrajectory) -> PolicyHorizonMetrics:
        actions = _five_year_actions(trajectory, paths=len(self._market.spot_rates))
        turnover_by_month = actions.diff(dim=1).abs().sum(dim=2)
        total_turnover = turnover_by_month.sum(dim=1)
        total_volume = actions.abs().sum(dim=(1, 2))
        normalized = total_turnover / (total_volume + _TURNOVER_EPSILON)
        zero_turnover = total_turnover <= _TURNOVER_EPSILON
        terminal = turnover_by_month[:, -_TERMINAL_MONTHS:].sum(dim=1) / (
            total_turnover + _TURNOVER_EPSILON
        )
        terminal = terminal.masked_fill(zero_turnover, float("nan"))
        unavailable_paths = int(zero_turnover.sum())
        return PolicyHorizonMetrics(
            policy_label=trajectory.policy_label,
            source_horizon_years=trajectory.source_horizon_years,
            analysis_window_months=_FIVE_YEAR_MONTHS,
            paths=int(actions.shape[0]),
            units=trajectory.units,
            normalized_action_turnover=normalized.detach().cpu(),
            terminal_turnover_concentration=terminal.detach().cpu(),
            terminal_concentration_status=(
                "unavailable" if unavailable_paths else "available"
            ),
            terminal_concentration_reason=(
                f"undefined for {unavailable_paths} paths with zero action turnover"
                if unavailable_paths
                else None
            ),
        )


def _five_year_actions(
    trajectory: FrozenPolicyTrajectory, *, paths: int
) -> torch.Tensor:
    actions = trajectory.actions
    if _tensor_identity(actions) != trajectory._action_identity:
        raise HorizonAnalysisError("Frozen action snapshot was mutated after capture")
    if actions.ndim != 3 or actions.shape[0] != paths or actions.shape[2] == 0:
        raise HorizonAnalysisError(
            "Frozen actions must have shape [common paths, decisions, action features]"
        )
    required_decisions = (
        180 if trajectory.policy_label == "MM(15y)" else _FIVE_YEAR_MONTHS
    )
    if actions.shape[1] < required_decisions:
        raise HorizonAnalysisError("Frozen actions require at least 60 decisions")
    if not torch.isfinite(actions).all():
        raise HorizonAnalysisError("Frozen actions must be finite")
    expected_horizon = 5 if trajectory.policy_label == "MM(5y)" else 15
    if trajectory.source_horizon_years != expected_horizon:
        raise HorizonAnalysisError(
            f"{trajectory.policy_label} has an incompatible source horizon"
        )
    return actions[:, :_FIVE_YEAR_MONTHS]


def _validate_trajectory_provenance(
    trajectories: tuple[FrozenPolicyTrajectory, ...],
    *,
    analysis_identity: _MarketPrefixIdentity,
) -> None:
    """Reject trajectories not evaluated on the common locked five-year prefix."""

    _mm_five_year, mm_fifteen_year, mm_fifteen_year_truncated = trajectories
    expected_market_horizons = (5, 15, 5)
    for trajectory, expected_market_horizon in zip(
        trajectories, expected_market_horizons, strict=True
    ):
        if trajectory.evaluation_market_horizon_years != expected_market_horizon:
            raise HorizonAnalysisError(
                f"{trajectory.policy_label} has an incompatible evaluation market horizon"
            )
        if trajectory.time_feature_horizon_years != trajectory.source_horizon_years:
            raise HorizonAnalysisError(
                f"{trajectory.policy_label} has incompatible time-feature provenance"
            )
        if trajectory._market_prefix_identity != analysis_identity:
            raise HorizonAnalysisError(
                "Horizon comparison requires a common scenario path identity and ordering"
            )
    if mm_fifteen_year_truncated.optimizer_updates != 0:
        raise HorizonAnalysisError("MM(15y|5y) requires zero optimizer updates")
    if mm_fifteen_year_truncated.checkpoint_identity != mm_fifteen_year.checkpoint_identity:
        raise HorizonAnalysisError(
            "MM(15y|5y) must reuse the selected MM(15y) checkpoint identity"
        )


def _market_prefix_identity(market: MarketScenarioBatch) -> _MarketPrefixIdentity:
    if market.horizon_years < 5:
        raise HorizonAnalysisError("Scenario path identity requires a five-year prefix")
    paths = len(market.spot_rates)
    global_indices = market.global_path_indices or tuple(range(paths))
    return _MarketPrefixIdentity(
        convention=market.convention,
        seed=market.seed,
        calibration_identity=market.calibration_identity,
        initial_curve_identity=market.initial_curve_identity,
        as_of_date=market.as_of_date,
        split=market.split,
        epoch=market.epoch,
        global_path_indices=global_indices,
        curve_content_hash=_array_identity(
            market.spot_rates[:, : _FIVE_YEAR_MONTHS + 1]
        ),
        discount_content_hash=_array_identity(
            market.discount_factors[:, : _FIVE_YEAR_MONTHS + 1]
        ),
        forward_content_hash=_array_identity(
            market.monthly_forwards[:, : _FIVE_YEAR_MONTHS + 1]
        ),
        innovation_content_hash=_array_identity(
            market.innovations[:, :_FIVE_YEAR_MONTHS]
        ),
    )


def _five_year_market_snapshot(market: MarketScenarioBatch) -> MarketScenarioBatch:
    """Copy the analysis prefix so later caller mutation cannot change categories."""

    if market.horizon_years != 5:
        raise HorizonAnalysisError("Analysis snapshot requires a five-year market")
    spot_rates = np.array(market.spot_rates[:, : _FIVE_YEAR_MONTHS + 1], copy=True)
    discount_factors = np.array(
        market.discount_factors[:, : _FIVE_YEAR_MONTHS + 1], copy=True
    )
    monthly_forwards = np.array(
        market.monthly_forwards[:, : _FIVE_YEAR_MONTHS + 1], copy=True
    )
    innovations = np.array(market.innovations[:, :_FIVE_YEAR_MONTHS], copy=True)
    for values in (spot_rates, discount_factors, monthly_forwards, innovations):
        values.setflags(write=False)
    return MarketScenarioBatch(
        spot_rates=spot_rates,
        discount_factors=discount_factors,
        monthly_forwards=monthly_forwards,
        innovations=innovations,
        convention=market.convention,
        horizon_years=5,
        seed=market.seed,
        calibration_identity=market.calibration_identity,
        round_trip_error=market.round_trip_error,
        initial_curve_identity=market.initial_curve_identity,
        as_of_date=market.as_of_date,
        split=market.split,
        epoch=market.epoch,
        global_path_indices=market.global_path_indices,
    )


def _tensor_identity(values: torch.Tensor) -> str:
    array = values.detach().cpu().contiguous().numpy()
    return _array_identity(array)


def _array_identity(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _paired_horizon_intervals(
    metrics: Mapping[str, PolicyHorizonMetrics], *, seed: int, resamples: int
) -> tuple[PairedInterval, ...]:
    intervals: list[PairedInterval] = []
    comparisons = (
        ("MM(5y)", "MM(15y)"),
        ("MM(5y)", "MM(15y|5y)"),
        ("MM(15y)", "MM(15y|5y)"),
    )
    for metric_name in (
        "normalized_action_turnover",
        "terminal_turnover_concentration",
    ):
        for offset, (left_label, right_label) in enumerate(comparisons):
            left = getattr(metrics[left_label], metric_name)
            right = getattr(metrics[right_label], metric_name)
            try:
                point, lower, upper = paired_bootstrap(
                    left, right, seed=seed + offset, resamples=resamples
                )
            except LockedEvaluationError as error:
                intervals.append(
                    PairedInterval(
                        left_label=left_label,
                        right_label=right_label,
                        metric=metric_name,
                        point_difference=None,
                        lower_95=None,
                        upper_95=None,
                        paths=int(left.numel()),
                        resamples=resamples,
                        status="unavailable",
                        reason=str(error),
                    )
                )
                continue
            intervals.append(
                PairedInterval(
                    left_label=left_label,
                    right_label=right_label,
                    metric=metric_name,
                    point_difference=point,
                    lower_95=lower,
                    upper_95=upper,
                    paths=int(left.numel()),
                    resamples=resamples,
                )
            )
    return tuple(intervals)


def _scenario_categories(
    market: MarketScenarioBatch, *, size: int
) -> tuple[_ScenarioPathCategory, ...]:
    """Select overlapping paper categories with deterministic global-index ties."""

    paths, periods, tenors = market.spot_rates.shape
    if size <= 0:
        raise ValueError("Scenario category size must be positive")
    if size > paths:
        raise ValueError(
            f"Requested category size {size} exceeds available paths {paths}"
        )
    if periods < _FIVE_YEAR_MONTHS + 1 or tenors < 180:
        raise HorizonAnalysisError(
            "Scenario categories require 61 curve dates and the 1M-to-15Y grid"
        )
    if not np.isfinite(market.spot_rates).all():
        raise HorizonAnalysisError("Scenario category curves must be finite")
    global_indices = market.global_path_indices or tuple(range(paths))
    if len(global_indices) != paths:
        raise HorizonAnalysisError("Scenario category paths require one global index each")

    final_curve = market.spot_rates[:, _FIVE_YEAR_MONTHS, :]
    final_steepness = final_curve[:, -1] - final_curve[:, 0]
    all_steepness = market.spot_rates[:, : _FIVE_YEAR_MONTHS + 1, -1] - market.spot_rates[
        :, : _FIVE_YEAR_MONTHS + 1, 0
    ]
    up_candidates = np.flatnonzero(final_curve[:, 0] > 0.02)
    if len(up_candidates) < size:
        raise ValueError(
            "Upward category request exceeds available paths with final 1M yield above 2%"
        )
    all_rows = np.arange(paths)
    return tuple(
        _ScenarioPathCategory(
            category=category,
            path_rule=path_rule,
            row_indices=rows,
            path_indices=tuple(global_indices[row] for row in rows),
        )
        for category, path_rule, rows in (
            (
                "steep",
                "maximum final 15Y - 1M yield-curve steepness",
                _rank_rows(
                    final_steepness,
                    all_rows,
                    global_indices,
                    size=size,
                    descending=True,
                ),
            ),
            (
                "upward",
                "maximum final 15Y - 1M steepness among paths with final 1M yield above 2%",
                _rank_rows(
                    final_steepness,
                    up_candidates,
                    global_indices,
                    size=size,
                    descending=True,
                ),
            ),
            (
                "downward",
                "minimum average yield across all final-curve maturities",
                _rank_rows(
                    final_curve.mean(axis=1),
                    all_rows,
                    global_indices,
                    size=size,
                    descending=False,
                ),
            ),
            (
                "inverted",
                "minimum final 15Y - 1M yield-curve steepness",
                _rank_rows(
                    final_steepness,
                    all_rows,
                    global_indices,
                    size=size,
                    descending=False,
                ),
            ),
            (
                "constant-steepness",
                "minimum standard deviation of 15Y - 1M steepness across the model periods",
                _rank_rows(
                    all_steepness.std(axis=1),
                    all_rows,
                    global_indices,
                    size=size,
                    descending=False,
                ),
            ),
        )
    )


def _rank_rows(
    scores: object,
    candidates: object,
    global_indices: tuple[int, ...],
    *,
    size: int,
    descending: bool,
) -> tuple[int, ...]:
    score_array = np.asarray(scores, dtype=np.float64)
    candidate_array = np.asarray(candidates, dtype=np.int64)
    ordered = sorted(
        candidate_array.tolist(),
        key=lambda row: (
            -float(score_array[row]) if descending else float(score_array[row]),
            global_indices[row],
        ),
    )
    return tuple(ordered[:size])


def _category_output(
    category: _ScenarioPathCategory,
    metrics: PolicyHorizonMetrics,
) -> ScenarioCategoryOutput:
    selection = torch.tensor(category.row_indices, dtype=torch.long)
    terminal = metrics.terminal_turnover_concentration[selection]
    unavailable_paths = int((~torch.isfinite(terminal)).sum())
    return ScenarioCategoryOutput(
        category=category.category,
        policy_label=metrics.policy_label,
        path_rule=category.path_rule,
        path_indices=category.path_indices,
        sample_size=len(category.row_indices),
        source_horizon_years=metrics.source_horizon_years,
        analysis_window_months=metrics.analysis_window_months,
        units=metrics.units,
        mean_normalized_action_turnover=float(
            metrics.normalized_action_turnover[selection].mean()
        ),
        mean_terminal_turnover_concentration=(
            None if unavailable_paths else float(terminal.mean())
        ),
        terminal_concentration_status=(
            "unavailable" if unavailable_paths else "available"
        ),
        terminal_concentration_reason=(
            f"undefined for {unavailable_paths} selected paths with zero action turnover"
            if unavailable_paths
            else None
        ),
    )
