"""Locked final-test evaluation, risk summaries, and paired small-sample statistics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from deepalm.config import ResolvedRunConfiguration
from deepalm.objective import evaluation_objective_parameters
from deepalm.policies import (
    BMConstantPolicy,
    BMDatePolicy,
    BMEqualPolicy,
    TreasuryPolicy,
)
from deepalm.reference_bank import (
    ReferenceBankError,
    ReferenceBankProvider,
    ReferenceBankSensitivity,
    ReferenceBankSnapshot,
)
from deepalm.runoff import ALMSimulator, PassiveRunoffResult
from deepalm.semantics import FINANCIAL_SEMANTICS_VERSION, current_financial_semantics
from deepalm.term_structures import (
    HistoricalTermStructures,
    HjmPcaCalibration,
    MarketScenarioBatch,
    MarketScenarioModel,
)
from deepalm.training import MMTrainer


class LockedEvaluationError(RuntimeError):
    """Raised when immutable checkpoint or test-manifest contracts are violated."""


@dataclass(frozen=True)
class PolicyCheckpoint:
    """A labelled frozen checkpoint to evaluate, never select or update."""

    label: str
    checkpoint_path: Path


@dataclass(frozen=True)
class PairedInterval:
    """Small-sample paired bootstrap evidence; intervals are descriptive only."""

    left_label: str
    right_label: str
    metric: str
    point_difference: float | None
    lower_95: float | None
    upper_95: float | None
    paths: int
    resamples: int
    status: str = "available"
    reason: str | None = None


@dataclass(frozen=True)
class LockedEvaluationResult:
    """Metrics and provenance from one immutable final-test evaluation."""

    reports: dict[str, dict[str, object]]
    markets: dict[int, MarketScenarioBatch]
    manifest: dict[str, object]
    paired_intervals: tuple[PairedInterval, ...]
    path_metrics: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)


@dataclass(frozen=True)
class SensitivityVariantStatus:
    """Visible construction outcome for one requested Reference Bank variant."""

    sensitivity: ReferenceBankSensitivity
    status: str
    snapshot_content_hash: str | None
    reason: str | None


@dataclass(frozen=True)
class FrozenPolicySensitivityResult:
    """A descriptive, no-retraining evaluation of a single bank sensitivity."""

    sensitivity: ReferenceBankSensitivity
    variant: ReferenceBankSnapshot
    variant_statuses: tuple[SensitivityVariantStatus, ...]
    evaluation: LockedEvaluationResult
    training_triggered: bool = False


def signed_tail_metrics(
    values: torch.Tensor, *, confidence: float = 0.95
) -> dict[str, float | None]:
    """Return centered lower-tail risk (equations 51f--g, E-10 corrected).

    Select the tail on the raw scale, then subtract the full-sample mean.
    Negative values measure downside deviation, not absolute negative returns.
    """

    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        return {"signed_var_95": None, "signed_es_95": None}
    quantile = torch.quantile(values, 1.0 - confidence)
    tail = values[values <= quantile]
    return {
        "signed_var_95": float(quantile - values.mean()),
        "signed_es_95": float(tail.mean() - values.mean()) if tail.numel() else None,
    }


def penalty_tail_metrics(values: torch.Tensor) -> dict[str, float | None]:
    """Centered upper-tail penalty risk, equations 50d--e with E-10 corrected."""
    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        return {"var95": None, "es95": None}
    threshold = torch.quantile(values, 0.95)
    return {
        "var95": float(threshold - values.mean()),
        "es95": float(values[values >= threshold].mean() - values.mean()),
    }


def paired_bootstrap(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    seed: int,
    resamples: int = 100,
) -> tuple[float, float, float]:
    """Return mean(left-right) and a deterministic paired percentile interval."""

    if left.ndim != 1 or right.shape != left.shape or left.numel() == 0:
        raise LockedEvaluationError(
            "Paired bootstrap requires matching non-empty path vectors"
        )
    if (
        not torch.isfinite(left).all()
        or not torch.isfinite(right).all()
        or resamples <= 0
    ):
        raise LockedEvaluationError(
            "Paired bootstrap requires finite values and positive resamples"
        )
    difference = (left - right).detach().cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        difference.numel(), (resamples, difference.numel()), generator=generator
    )
    samples = difference[indices].mean(dim=1)
    return (
        float(difference.mean()),
        float(torch.quantile(samples, 0.025)),
        float(torch.quantile(samples, 0.975)),
    )


class LockedEvaluator:
    """Evaluate only frozen checkpoints on a common, final-test market set."""

    def __init__(
        self,
        configuration: ResolvedRunConfiguration,
        *,
        snapshot: ReferenceBankSnapshot,
        historical: HistoricalTermStructures,
        calibration: HjmPcaCalibration,
        simulator: ALMSimulator | None = None,
        market_model: MarketScenarioModel | None = None,
        frozen_policy_snapshot: ReferenceBankSnapshot | None = None,
    ) -> None:
        self._configuration = configuration
        self._snapshot = snapshot
        self._frozen_policy_snapshot = frozen_policy_snapshot or snapshot
        if frozen_policy_snapshot is not None and (
            snapshot.profile != "sensitivity"
            or frozen_policy_snapshot.profile != "canonical"
        ):
            raise LockedEvaluationError(
                "A different frozen-policy snapshot requires a sensitivity variant and canonical source"
            )
        self._historical = historical
        self._calibration = calibration
        if frozen_policy_snapshot is not None:
            _validate_one_factor_sensitivity_snapshot(snapshot, historical)
        self._simulator = simulator or ALMSimulator()
        self._market_model = market_model or MarketScenarioModel()
        self._device = torch.device(configuration.optimization.device)
        self._dtype = getattr(torch, configuration.optimization.dtype)

    def evaluate(
        self,
        checkpoints: tuple[PolicyCheckpoint, ...],
        *,
        bootstrap_resamples: int = 100,
    ) -> LockedEvaluationResult:
        """Run every checkpoint once, grouped on common locked paths by horizon."""

        if not checkpoints or len({item.label for item in checkpoints}) != len(
            checkpoints
        ):
            raise LockedEvaluationError(
                "Locked evaluation requires uniquely labelled checkpoints"
            )
        if bootstrap_resamples < 100:
            raise LockedEvaluationError(
                "Locked local evaluation requires at least 100 bootstrap resamples"
            )
        metadata = {
            item.label: _checkpoint_metadata(item.checkpoint_path)
            for item in checkpoints
        }
        for label, checkpoint in metadata.items():
            _validate_checkpoint_compatibility(
                checkpoint,
                configuration=self._configuration,
                historical=self._historical,
                calibration=self._calibration,
                snapshot=self._frozen_policy_snapshot,
            )
            if (
                checkpoint["horizon_years"]
                not in self._configuration.experiment.horizons_years
            ):
                raise LockedEvaluationError(
                    f"{label} has a horizon outside the evaluation configuration"
                )
        loaded = {
            item.label: _load_frozen_policy(
                item.checkpoint_path, device=self._device, dtype=self._dtype
            )
            for item in checkpoints
        }
        markets = {
            horizon: self._locked_market(horizon)
            for horizon in sorted(
                {int(item["horizon_years"]) for item in metadata.values()}
            )
        }
        reports: dict[str, dict[str, object]] = {}
        path_metrics: dict[str, dict[str, torch.Tensor]] = {}
        for item in checkpoints:
            checkpoint = metadata[item.label]
            horizon = int(checkpoint["horizon_years"])
            with torch.no_grad():
                outcome = self._simulator.rollout(
                    self._snapshot,
                    markets[horizon],
                    policy=loaded[item.label],
                    device=self._device,
                    dtype=self._dtype,
                    include_loan_dynamics=True,
                    convention=self._configuration.convention.profile,
                    include_deposit_dynamics=True,
                    objective_parameters=evaluation_objective_parameters(
                        len(markets[horizon].spot_rates),
                        horizon_years=horizon,
                        device=self._device,
                        dtype=self._dtype,
                    ),
                )
            reports[item.label], path_metrics[item.label] = _report_outcome(
                outcome, horizon
            )
        intervals = _paired_intervals(
            path_metrics,
            metadata,
            seed=self._configuration.seeds["bootstrap"],
            resamples=bootstrap_resamples,
        )
        manifest = self._manifest(checkpoints, metadata, markets, bootstrap_resamples)
        return LockedEvaluationResult(
            reports=reports,
            markets=markets,
            manifest=manifest,
            paired_intervals=intervals,
            path_metrics=path_metrics,
        )

    def write_manifest(self, result: LockedEvaluationResult, path: Path) -> None:
        """Persist only provenance and descriptive statistics, never model updates."""

        contents = {
            **result.manifest,
            "reports": result.reports,
            "paired_intervals": [asdict(item) for item in result.paired_intervals],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(contents, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _locked_market(self, horizon_years: int) -> MarketScenarioBatch:
        return self._market_model.generate_hjm_scenarios(
            self._historical,
            self._calibration,
            convention=self._configuration.convention.profile,
            horizon_years=horizon_years,
            paths=self._configuration.run_scale.test_paths,
            seed=self._configuration.seeds["market_scenarios"],
            split="test",
            epoch=0,
            global_path_indices=tuple(range(self._configuration.run_scale.test_paths)),
        )

    def _manifest(
        self,
        checkpoints: tuple[PolicyCheckpoint, ...],
        metadata: dict[str, dict[str, object]],
        markets: dict[int, MarketScenarioBatch],
        resamples: int,
    ) -> dict[str, object]:
        return {
            "format_version": 1,
            "risk_metric_convention": "centered-equity-ratio-and-penalty-v2",
            "financial_semantics_version": FINANCIAL_SEMANTICS_VERSION,
            "kind": (
                "frozen-policy-reference-bank-sensitivity"
                if self._frozen_policy_snapshot.content_hash != self._snapshot.content_hash
                else "locked-final-test-evaluation"
            ),
            "convention": self._configuration.convention.profile,
            "reference_bank_content_hash": self._snapshot.content_hash,
            "data_identities": {
                "market_source_hash": self._historical.source_hash,
                "hjm_calibration_identity": self._calibration.calibration_identity,
                "reference_bank_content_hash": self._snapshot.content_hash,
            },
            "frozen_policy_reference_bank_content_hash": self._frozen_policy_snapshot.content_hash,
            "test_seed": self._configuration.seeds["market_scenarios"],
            "bootstrap_seed": self._configuration.seeds["bootstrap"],
            "bootstrap_resamples": resamples,
            "test_scenarios": {
                str(h): {
                    "split": market.split,
                    "epoch": market.epoch,
                    "global_path_indices": list(market.global_path_indices),
                    "paths": len(market.spot_rates),
                    "calibration_identity": market.calibration_identity,
                }
                for h, market in markets.items()
            },
            "checkpoints": {
                item.label: {
                    "sha256": _sha256(item.checkpoint_path),
                    "policy": metadata[item.label]["policy"],
                    "horizon_years": metadata[item.label]["horizon_years"],
                }
                for item in checkpoints
            },
            "statistical_status": "small-sample demonstration; intervals do not gate superiority",
        }


class FrozenPolicySensitivityEvaluator:
    """Evaluate approved bank variants with canonical checkpoints and no training.

    This explicitly separates a changed balance-sheet input from the immutable
    canonical policy source.  It is a descriptive local demonstration: neither
    an invalid variant nor an observed metric can select a checkpoint, adjust a
    hyperparameter, or start retraining.
    """

    def __init__(
        self,
        configuration: ResolvedRunConfiguration,
        *,
        canonical_snapshot: ReferenceBankSnapshot,
        historical: HistoricalTermStructures,
        calibration: HjmPcaCalibration,
        provider: ReferenceBankProvider | None = None,
        simulator: ALMSimulator | None = None,
        market_model: MarketScenarioModel | None = None,
    ) -> None:
        if configuration.convention.profile != "corrected":
            raise LockedEvaluationError(
                "Representative Reference Bank sensitivity requires the Corrected convention"
            )
        if canonical_snapshot.profile != "canonical":
            raise LockedEvaluationError(
                "Frozen-policy sensitivity requires a canonical policy source snapshot"
            )
        self._configuration = configuration
        self._canonical_snapshot = canonical_snapshot
        self._historical = historical
        self._calibration = calibration
        self._provider = provider or ReferenceBankProvider()
        self._simulator = simulator
        self._market_model = market_model
        self._provider.validate(canonical_snapshot)

    def evaluate(
        self,
        checkpoints: tuple[PolicyCheckpoint, ...],
        *,
        sensitivity: ReferenceBankSensitivity,
        validation_requests: tuple[ReferenceBankSensitivity, ...] = (),
        bootstrap_resamples: int = 100,
    ) -> FrozenPolicySensitivityResult:
        """Evaluate one valid variant and retain visible status for all requests."""

        requested = (sensitivity, *validation_requests)
        variants: dict[ReferenceBankSensitivity, ReferenceBankSnapshot] = {}
        statuses: list[SensitivityVariantStatus] = []
        for request in requested:
            try:
                variant = self._provider.build_sensitivity(self._historical, request)
            except ReferenceBankError as error:
                statuses.append(
                    SensitivityVariantStatus(
                        sensitivity=request,
                        status="invalid",
                        snapshot_content_hash=None,
                        reason=str(error),
                    )
                )
                continue
            variants[request] = variant
            statuses.append(
                SensitivityVariantStatus(
                    sensitivity=request,
                    status="available",
                    snapshot_content_hash=variant.content_hash,
                    reason=None,
                )
            )
        if sensitivity not in variants:
            raise LockedEvaluationError(
                "Representative sensitivity is invalid; inspect variant statuses before evaluation"
            )
        evaluator = LockedEvaluator(
            self._configuration,
            snapshot=variants[sensitivity],
            frozen_policy_snapshot=self._canonical_snapshot,
            historical=self._historical,
            calibration=self._calibration,
            simulator=self._simulator,
            market_model=self._market_model,
        )
        evaluation = evaluator.evaluate(
            checkpoints, bootstrap_resamples=bootstrap_resamples
        )
        return FrozenPolicySensitivityResult(
            sensitivity=sensitivity,
            variant=variants[sensitivity],
            variant_statuses=tuple(statuses),
            evaluation=evaluation,
        )

    def write_manifest(
        self, result: FrozenPolicySensitivityResult, path: Path
    ) -> None:
        """Persist the variant, invalid requests, and frozen-evaluation evidence."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                self.manifest_data(result), indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )

    def manifest_data(self, result: FrozenPolicySensitivityResult) -> dict[str, object]:
        """Return JSON-ready sensitivity evidence for an atomic run bundle."""

        return {
            "format_version": 1,
            "kind": "frozen-policy-reference-bank-sensitivity",
            "canonical_reference_bank_content_hash": self._canonical_snapshot.content_hash,
            "representative_sensitivity": asdict(result.sensitivity),
            "representative_variant_content_hash": result.variant.content_hash,
            "variant_statuses": [asdict(item) for item in result.variant_statuses],
            "training_triggered": result.training_triggered,
            "evaluation": {
                "manifest": result.evaluation.manifest,
                "reports": result.evaluation.reports,
                "paired_intervals": [
                    asdict(item) for item in result.evaluation.paired_intervals
                ],
            },
            "interpretation": (
                "small-sample frozen-policy demonstration; it does not establish "
                "generalization or structural conclusions"
            ),
        }


def _validate_one_factor_sensitivity_snapshot(
    snapshot: ReferenceBankSnapshot, historical: HistoricalTermStructures
) -> None:
    """Rebuild the declared variant and reject any undeclared financial change."""

    provider = ReferenceBankProvider()
    try:
        provider.validate(snapshot)
    except ReferenceBankError as error:
        raise LockedEvaluationError(
            "Sensitivity snapshot contains undeclared changes"
        ) from error
    try:
        sensitivity = ReferenceBankSensitivity(
            factor=str(snapshot.product_assumptions["sensitivity_factor"]),
            value=float(snapshot.product_assumptions["sensitivity_value"]),
        )
        expected = provider.build_sensitivity(historical, sensitivity)
    except (KeyError, TypeError, ValueError, ReferenceBankError) as error:
        raise LockedEvaluationError(
            "Sensitivity snapshot cannot be rebuilt from its declared one-factor input"
        ) from error
    if snapshot.content_hash != expected.content_hash:
        raise LockedEvaluationError(
            "Sensitivity snapshot differs from the declared canonical one-factor variant"
        )


def _report_outcome(
    outcome: PassiveRunoffResult, horizon_years: int
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    if (
        outcome.objective is None
        or outcome.constraint_values is None
        or outcome.constraint_violations is None
    ):
        raise LockedEvaluationError(
            "Locked evaluation rollout did not return objective and constraints"
        )
    initial = outcome.equity[:, 0]
    terminal = outcome.equity[:, -1]
    ratio = terminal / initial
    valid_return = ratio > 0
    annualized = torch.full_like(ratio, torch.nan)
    annualized[valid_return] = ratio[valid_return].pow(1.0 / horizon_years) - 1.0
    dividend_yield: torch.Tensor | None = None
    if outcome.dividends is not None:
        dividend_yield = outcome.dividends.sum(dim=1) / initial / horizon_years
    penalty = outcome.objective.penalty
    report: dict[str, object] = {
        "losses": {
            "total": _mean(outcome.objective.total),
            "target": _mean(outcome.objective.target),
            "penalty": _mean(penalty),
            "penalty_es95": _upper_es95(penalty),
        },
        "crra": _mean(outcome.objective.crra),
        "equity_ratio": {"mean": _mean(ratio), "standard_deviation": _std(ratio)},
        "annualized_return": _available(
            annualized, "terminal equity ratio is non-positive"
        ),
        "standardized_dividend_yield": _available(
            dividend_yield, "dividends unavailable"
        ),
        "risk": equity_risk_report(ratio),
        "constraints": _constraint_report(
            outcome.constraint_values,
            outcome.constraint_violations,
            outcome.constraint_annual_mask,
            penalty,
        ),
    }
    return report, {
        "equity_ratio": ratio,
        "annualized_return": annualized,
        "total_loss": outcome.objective.total,
        "penalty": penalty,
    }


def _constraint_report(
    values: torch.Tensor,
    violations: torch.Tensor,
    annual_mask: torch.Tensor | None,
    penalty: torch.Tensor,
) -> dict[str, object]:
    names = ("lcr", "nsfr", "cmr", "equity_rwa", "irs", "eyr")
    result: dict[str, object] = {
        "aggregate_penalty": {
            "mean": _mean(penalty),
            "es95": _upper_es95(penalty),
        },
        "by_constraint": {},
    }
    by_constraint = result["by_constraint"]
    assert isinstance(by_constraint, dict)
    for index, name in enumerate(names):
        current_values, current_violations = (
            values[:, :, index],
            violations[:, :, index],
        )
        if name == "eyr":
            if annual_mask is None:
                raise LockedEvaluationError(
                    "Locked evaluation requires an annual mask for EYR reporting"
                )
            eligible_times = annual_mask.all(dim=0)
            current_values = current_values[:, eligible_times]
            current_violations = current_violations[:, eligible_times]
        positive = current_violations > 0
        by_constraint[name] = {
            "time_mean": current_values.mean(dim=0).detach().cpu().tolist(),
            "time_median": current_values.median(dim=0).values.detach().cpu().tolist(),
            "ever_violation_share": float(positive.any(dim=1).float().mean()),
            "violating_count": int(positive.sum()),
            "violating_mean": float(current_violations[positive].mean())
            if positive.any()
            else None,
            "worst_value": float(
                current_values.max() if name == "irs" else current_values.min()
            ),
        }
    return result


def _paired_intervals(
    metrics: dict[str, dict[str, torch.Tensor]],
    metadata: dict[str, dict[str, object]],
    *,
    seed: int,
    resamples: int,
) -> tuple[PairedInterval, ...]:
    labels = sorted(metrics)
    intervals: list[PairedInterval] = []
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            if metadata[left]["horizon_years"] != metadata[right]["horizon_years"]:
                continue
            for metric in ("annualized_return", "total_loss", "penalty"):
                first, second = metrics[left][metric], metrics[right][metric]
                if not torch.isfinite(first).all() or not torch.isfinite(second).all():
                    intervals.append(
                        PairedInterval(
                            left,
                            right,
                            metric,
                            None,
                            None,
                            None,
                            int(first.numel()),
                            resamples,
                            status="unavailable",
                            reason="metric is undefined on one or more locked paths",
                        )
                    )
                    continue
                point, lower, upper = paired_bootstrap(
                    first,
                    second,
                    seed=_derived_seed(seed, left, right, metric),
                    resamples=resamples,
                )
                intervals.append(
                    PairedInterval(
                        left,
                        right,
                        metric,
                        point,
                        lower,
                        upper,
                        int(first.numel()),
                        resamples,
                    )
                )
    return tuple(intervals)


def _load_frozen_policy(
    path: Path, *, device: torch.device, dtype: torch.dtype
) -> TreasuryPolicy:
    checkpoint = _checkpoint_metadata(path)
    policy_name, horizon = checkpoint["policy"], int(checkpoint["horizon_years"])
    if policy_name == "MM":
        return MMTrainer.load_policy_from_checkpoint(path, device=device, dtype=dtype)
    types = {"BM^E": BMEqualPolicy, "BM^C": BMConstantPolicy, "BM^D": BMDatePolicy}
    if policy_name not in types:
        raise LockedEvaluationError(
            "Checkpoint policy is unsupported for locked evaluation"
        )
    policy = (
        BMDatePolicy(transitions=12 * horizon, device=device, dtype=dtype)
        if policy_name == "BM^D"
        else types[policy_name](device=device, dtype=dtype)
    )
    policy.load_state_dict(checkpoint["policy_state"])
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def _checkpoint_metadata(path: Path) -> dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise LockedEvaluationError(f"Could not load checkpoint: {path}") from error
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise LockedEvaluationError("Checkpoint has incompatible format")
    return checkpoint


def _validate_checkpoint_compatibility(
    checkpoint: dict[str, object],
    *,
    configuration: ResolvedRunConfiguration,
    historical: HistoricalTermStructures,
    calibration: HjmPcaCalibration,
    snapshot: ReferenceBankSnapshot,
) -> None:
    if not current_financial_semantics(checkpoint):
        raise LockedEvaluationError("Checkpoint financial semantics are incompatible; retraining required")
    expected = {
        "market_source_hash": historical.source_hash,
        "hjm_calibration_identity": calibration.calibration_identity,
        "reference_bank_content_hash": snapshot.content_hash,
    }
    if checkpoint.get("data_identities") != expected:
        raise LockedEvaluationError(
            "Checkpoint data identities are incompatible with locked evaluation"
        )
    try:
        convention = checkpoint["configuration"]["convention"]["profile"]
    except (KeyError, TypeError):
        raise LockedEvaluationError("Checkpoint lacks a convention manifest") from None
    if convention != configuration.convention.profile:
        raise LockedEvaluationError(
            "Checkpoint convention is incompatible with locked evaluation"
        )
    recorded_configuration = checkpoint.get("configuration")
    if not isinstance(recorded_configuration, dict):
        raise LockedEvaluationError("Checkpoint lacks a configuration manifest")
    recorded_scale = recorded_configuration.get("run_scale")
    expected_scale = configuration.run_scale
    if (
        recorded_configuration.get("seeds") != configuration.seeds
        or not isinstance(recorded_scale, dict)
        or any(
            recorded_scale.get(field) != getattr(expected_scale, field)
            for field in (
                "training_paths_per_epoch",
                "selection_paths",
                "test_paths",
            )
        )
    ):
        raise LockedEvaluationError(
            "Checkpoint required seeds or locked test path count are incompatible"
        )
    scenarios = checkpoint.get("scenario_identities")
    policy_name = checkpoint.get("policy")
    horizon_years = checkpoint.get("horizon_years")
    if not isinstance(scenarios, dict) or not isinstance(policy_name, str):
        raise LockedEvaluationError(
            "Checkpoint lacks disjoint training and selection scenario evidence"
        )
    try:
        horizon = int(horizon_years)
        training = scenarios["training"]
        selection = scenarios["selection"]
    except (KeyError, TypeError, ValueError):
        raise LockedEvaluationError(
            "Checkpoint lacks disjoint training and selection scenario evidence"
        ) from None
    expected_seed = configuration.seeds["market_scenarios"]
    expected_job_seed = _scenario_job_seed(expected_seed, policy_name, horizon)
    expected_training_paths = "0..training_paths_per_epoch-1"
    expected_selection_paths = "0..selection_paths-1"
    if (
        not isinstance(training, dict)
        or not isinstance(selection, dict)
        or training.get("split") != "training"
        or selection.get("split") != "selection"
        or training.get("base_seed") != expected_seed
        or selection.get("base_seed") != expected_seed
        or training.get("job_seed") != expected_job_seed
        or selection.get("job_seed") != expected_job_seed
        or training.get("epoch") != "1..epochs"
        or selection.get("epoch") != 0
        or training.get("global_path_indices") != expected_training_paths
        or selection.get("global_path_indices") != expected_selection_paths
    ):
        raise LockedEvaluationError(
            "Checkpoint training or selection scenario seeds or paths are incompatible"
        )


def _available(values: torch.Tensor | None, reason: str) -> dict[str, object]:
    if values is None or not torch.isfinite(values).all():
        return {"value": None, "status": "unavailable", "reason": reason}
    return {"value": _mean(values), "status": "available"}


def equity_risk_report(equity_ratios: torch.Tensor) -> dict[str, object]:
    """Paper equity-ratio risk, including finite zero/negative terminal equity."""

    if (
        equity_ratios.ndim != 1
        or equity_ratios.numel() == 0
        or not torch.isfinite(equity_ratios).all()
    ):
        return {
            "signed_var_95": None,
            "signed_es_95": None,
            "status": "unavailable",
            "reason": "terminal equity ratios are empty or non-finite",
            "input": "terminal_equity_ratio",
            "centered": True,
        }
    return {
        **signed_tail_metrics(equity_ratios),
        "status": "available",
        "input": "terminal_equity_ratio",
        "centered": True,
    }


def _mean(values: torch.Tensor) -> float:
    return float(values.mean())


def _std(values: torch.Tensor) -> float:
    return float(values.std(unbiased=False))


def _upper_es95(values: torch.Tensor) -> float | None:
    return penalty_tail_metrics(values)["es95"]


def _derived_seed(seed: int, *parts: str) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join((str(seed), *parts)).encode()).digest()[:8], "little"
    )


def _scenario_job_seed(seed: int, policy_name: str, horizon_years: int) -> int:
    return _derived_seed(seed, policy_name, str(horizon_years))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
