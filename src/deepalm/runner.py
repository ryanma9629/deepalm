"""The public execution seam for Deep ALM reproductions."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch

from deepalm.config import ConventionConfiguration, ResolvedRunConfiguration
from deepalm.planning import build_execution_plan
from deepalm.resources import BudgetExceeded, ResourceMonitor

if TYPE_CHECKING:
    from deepalm.evaluation import PolicyCheckpoint
    from deepalm.reference_bank import ReferenceBankSensitivity, ReferenceBankSnapshot
    from deepalm.term_structures import HistoricalTermStructures, HjmPcaCalibration
    from deepalm.training import DeviceValidationRecord


class RunStatus(StrEnum):
    COMPLETED = "completed"
    ACCEPTANCE_FAILED = "acceptance_failed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class AcceptanceStatus(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    PAIRED_CONVENTION_RESEARCH_PILOT = "paired-convention-research-pilot"


class OperationalRunError(RuntimeError):
    """A recoverable pipeline failure with serializable diagnostic context."""

    def __init__(
        self, message: str, *, diagnostics: Mapping[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


class MarketPreflightModel(Protocol):
    """Structural contract used by the bounded market preflight stage."""

    def load_historical_term_structures(
        self, source_path: Path, *, beta_unit: str
    ) -> Any: ...

    def calibrate_hjm_pca(self, historical: Any) -> Any: ...

    def generate_hjm_scenarios(
        self, historical: Any, calibration: Any, **kwargs: Any
    ) -> Any: ...


class DeviceValidationTrainer(Protocol):
    """A real trainer capable of a bounded update on each supported device."""

    def validate_devices(
        self, *, horizon_years: int
    ) -> tuple[DeviceValidationRecord, ...]: ...


@dataclass(frozen=True)
class RunBundle:
    """Auditable result of one reproduction execution."""

    status: RunStatus
    acceptance_status: AcceptanceStatus
    artifact_directory: Path | None
    error: str | None = None
    artifacts: tuple[Path, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    checkpoints: tuple[Path, ...] = ()
    acceptance_evidence: tuple[Path, ...] = ()


class ReproductionRunner:
    """Create an atomic, auditable bundle for one resolved run configuration."""

    def run(
        self,
        configuration: ResolvedRunConfiguration,
        *,
        acceptance_status: AcceptanceStatus = AcceptanceStatus.PENDING,
    ) -> RunBundle:
        """Write the minimum audit bundle, returning a failure bundle on I/O errors."""

        try:
            status = _status_for(acceptance_status)
            manifest = _build_manifest(configuration, status, acceptance_status)
            artifact_directory = _write_bundle_atomically(configuration, manifest)
        except (OperationalRunError, OSError, ValueError) as error:
            failure_directory = _write_failure_bundle(configuration, error)
            return RunBundle(
                status=RunStatus.FAILED,
                acceptance_status=AcceptanceStatus.PENDING,
                artifact_directory=failure_directory,
                error=str(error),
                artifacts=(failure_directory / "manifest.json",)
                if failure_directory is not None
                else (),
            )

        return RunBundle(
            status=status,
            acceptance_status=acceptance_status,
            artifact_directory=artifact_directory,
            artifacts=(artifact_directory / "manifest.json",),
        )

    def reuse_completed_workflow_stage(
        self,
        configuration: ResolvedRunConfiguration,
        *,
        source_run_directory: Path,
        stage: str,
    ) -> RunBundle:
        """Reuse a compatible completed workflow stage without rerunning its matrix.

        A local workflow is published atomically, so its checkpoints, evaluation,
        and acceptance evidence are one inseparable audit unit.  This adapter
        deliberately verifies that unit instead of silently retraining it when a
        user asks a later CLI stage to reuse it.
        """

        try:
            artifact_directory = source_run_directory.resolve()
            manifest_path = artifact_directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _validate_reusable_workflow_manifest(
                manifest, configuration=configuration, stage=stage
            )
            required = _workflow_stage_artifacts(stage)
            missing = [name for name in required if not (artifact_directory / name).is_file()]
            if missing:
                raise OperationalRunError(
                    f"Reusable workflow is missing {stage} evidence: {missing}"
                )
            _validate_reusable_stage_evidence(
                manifest,
                configuration=configuration,
                artifact_directory=artifact_directory,
                stage=stage,
            )
            acceptance = (
                AcceptanceStatus.PASSED
                if manifest.get("acceptance_status") == AcceptanceStatus.PASSED.value
                else AcceptanceStatus.PENDING
            )
            status = (
                RunStatus.COMPLETED
                if acceptance is AcceptanceStatus.PASSED
                else RunStatus.ACCEPTANCE_FAILED
            )
            return RunBundle(
                status=status,
                acceptance_status=acceptance,
                artifact_directory=artifact_directory,
                artifacts=tuple(artifact_directory / name for name in required),
                acceptance_evidence=(artifact_directory / "acceptance-report.json",)
                if stage == "accept"
                else (),
            )
        except (OSError, ValueError, OperationalRunError) as error:
            return RunBundle(
                status=RunStatus.FAILED,
                acceptance_status=AcceptanceStatus.PENDING,
                artifact_directory=None,
                error=str(error),
            )

    def preflight_market(
        self,
        configuration: ResolvedRunConfiguration,
        *,
        resource_monitor: ResourceMonitor | None = None,
        market_model: MarketPreflightModel | None = None,
    ) -> RunBundle:
        """Run bounded HJM input/calibration/scenario work without training a policy.

        This deliberately produces a pending acceptance bundle: completing a market
        preflight is evidence for resource planning, not a completed ALM workflow.
        """

        monitor = resource_monitor or ResourceMonitor(
            wall_clock_budget_seconds=configuration.resources.wall_clock_budget_seconds,
            process_rss_limit_bytes=configuration.resources.process_rss_limit_bytes,
            accelerator_memory_limit_bytes=(
                configuration.resources.accelerator_memory_limit_bytes
            ),
            device=configuration.optimization.device,
        )
        completed_stages: list[str] = []
        scenario_batches: list[dict[str, object]] = []
        evidence: dict[str, object] = {
            "completed_stages": completed_stages,
            "scenario_batches": scenario_batches,
        }
        try:
            if market_model is None:
                from deepalm.term_structures import MarketScenarioModel

                market_model = MarketScenarioModel()
            monitor.check("before-market-load")
            historical = market_model.load_historical_term_structures(
                configuration.source_data.snb_csv,
                beta_unit=configuration.source_data.nss_beta_unit,
            )
            evidence["historical"] = {
                "source_hash": historical.source_hash,
                "round_trip_error": historical.round_trip_error,
            }
            completed_stages.append("historical-term-structures")
            monitor.check("after-market-load")
            calibration = market_model.calibrate_hjm_pca(historical)
            evidence["calibration"] = {
                "identity": calibration.calibration_identity,
                "explained_variance": calibration.explained_variance.tolist(),
            }
            completed_stages.append("hjm-pca-calibration")
            monitor.check("after-market-calibration")

            for horizon_years in configuration.experiment.horizons_years:
                for start in range(
                    0,
                    configuration.run_scale.training_paths_per_epoch,
                    configuration.run_scale.batch_size,
                ):
                    monitor.check(f"before-hjm-{horizon_years}y-batch-{start}")
                    paths = min(
                        configuration.run_scale.batch_size,
                        configuration.run_scale.training_paths_per_epoch - start,
                    )
                    batch = market_model.generate_hjm_scenarios(
                        historical,
                        calibration,
                        convention=configuration.convention.profile,
                        horizon_years=horizon_years,
                        paths=paths,
                        seed=configuration.seeds["market_scenarios"],
                        split="preflight-training",
                        epoch=0,
                        global_path_indices=tuple(range(start, start + paths)),
                    )
                    scenario_batches.append(
                        {
                            "horizon_years": horizon_years,
                            "split": batch.split,
                            "epoch": batch.epoch,
                            "global_path_indices": list(batch.global_path_indices),
                            "round_trip_error": batch.round_trip_error,
                        }
                    )
                    monitor.check(f"after-hjm-{horizon_years}y-batch-{start}")
            completed_stages.append("bounded-hjm-scenarios")
            evidence["resource_snapshot"] = monitor.snapshot().to_dict()
            manifest = _build_manifest(
                configuration, RunStatus.COMPLETED, AcceptanceStatus.PENDING
            )
            manifest["market_preflight"] = evidence
            artifact_directory = _write_bundle_atomically(configuration, manifest)
            return RunBundle(
                status=RunStatus.COMPLETED,
                acceptance_status=AcceptanceStatus.PENDING,
                artifact_directory=artifact_directory,
                artifacts=(artifact_directory / "manifest.json",),
            )
        except BudgetExceeded as error:
            error.diagnostics["market_preflight"] = evidence
            error.diagnostics["resource_snapshot"] = monitor.snapshot().to_dict()
            failure_directory = _write_failure_bundle(
                configuration, error, status=RunStatus.INCOMPLETE
            )
            return RunBundle(
                status=RunStatus.INCOMPLETE,
                acceptance_status=AcceptanceStatus.PENDING,
                artifact_directory=failure_directory,
                error=str(error),
                artifacts=(failure_directory / "manifest.json",)
                if failure_directory is not None
                else (),
            )
        except (OperationalRunError, OSError, ValueError) as error:
            failure_directory = _write_failure_bundle(configuration, error)
            return RunBundle(
                status=RunStatus.FAILED,
                acceptance_status=AcceptanceStatus.PENDING,
                artifact_directory=failure_directory,
                error=str(error),
                artifacts=(failure_directory / "manifest.json",)
                if failure_directory is not None
                else (),
            )

    def build_reference_bank(
        self,
        configuration: ResolvedRunConfiguration,
        *,
        snapshot_path: Path | None = None,
    ) -> RunBundle:
        """Build or import a reviewable Reference Bank without policy training."""

        try:
            from deepalm.reference_bank import (
                ReferenceBankProvider,
                ReferenceBankSensitivity,
            )
            from deepalm.term_structures import MarketScenarioModel

            provider = ReferenceBankProvider()
            if snapshot_path is None:
                historical = MarketScenarioModel().load_historical_term_structures(
                    configuration.source_data.snb_csv,
                    beta_unit=configuration.source_data.nss_beta_unit,
                )
                snapshot = provider.build_canonical(historical)
                sensitivity = configuration.reference_bank.sensitivity
                if sensitivity is not None:
                    snapshot = provider.build_sensitivity(
                        historical,
                        ReferenceBankSensitivity(
                            factor=sensitivity.factor, value=sensitivity.value
                        ),
                    )
            else:
                snapshot = provider.load(snapshot_path)
            manifest = _build_manifest(
                configuration, RunStatus.COMPLETED, AcceptanceStatus.PENDING
            )
            manifest["reference_bank"] = {
                "content_hash": snapshot.content_hash,
                "schema_version": snapshot.schema_version,
                "profile": snapshot.profile,
                "as_of_date": snapshot.as_of_date,
                "initial_curve_identity": snapshot.initial_curve_identity,
                "total_assets": snapshot.total_assets,
                "total_liabilities": snapshot.total_liabilities,
                "equity": snapshot.equity,
                "currency": snapshot.currency,
                "unit": snapshot.unit,
                "loan_duration_years": snapshot.loan_duration_years,
                "deposit_duration_years": snapshot.deposit_duration_years,
                "sensitivity": dict(snapshot.provenance).get("sensitivity"),
            }
            artifact_directory = _write_bundle_atomically(
                configuration,
                manifest,
                extra_artifacts={
                    "reference-bank.json": provider.serialize(snapshot),
                    "table-1.json": provider.table_one(snapshot),
                },
            )
            return RunBundle(
                status=RunStatus.COMPLETED,
                acceptance_status=AcceptanceStatus.PENDING,
                artifact_directory=artifact_directory,
                artifacts=(
                    artifact_directory / "manifest.json",
                    artifact_directory / "reference-bank.json",
                    artifact_directory / "table-1.json",
                ),
            )
        except (OperationalRunError, OSError, ValueError) as error:
            failure_directory = _write_failure_bundle(configuration, error)
            return RunBundle(
                status=RunStatus.FAILED,
                acceptance_status=AcceptanceStatus.PENDING,
                artifact_directory=failure_directory,
                error=str(error),
                artifacts=(failure_directory / "manifest.json",)
                if failure_directory is not None
                else (),
            )

    def evaluate_reference_bank_sensitivity(
        self,
        configuration: ResolvedRunConfiguration,
        *,
        canonical_snapshot: ReferenceBankSnapshot,
        historical: HistoricalTermStructures,
        calibration: HjmPcaCalibration,
        checkpoints: tuple[PolicyCheckpoint, ...],
        validation_requests: tuple[ReferenceBankSensitivity, ...] = (),
    ) -> RunBundle:
        """Atomically record the bounded, frozen-policy 5,000 mCHF sensitivity.

        This stage deliberately has no trainer dependency.  The optional
        configuration request may explicitly repeat the required local 5,000
        mCHF scale demonstration; all other registered variants require the
        separately budgeted research plan.
        """

        try:
            from deepalm.evaluation import (
                FrozenPolicySensitivityEvaluator,
                LockedEvaluationError,
            )
            from deepalm.reference_bank import ReferenceBankSensitivity

            configured = configuration.reference_bank.sensitivity
            sensitivity = (
                ReferenceBankSensitivity(configured.factor, configured.value)
                if configured is not None
                else ReferenceBankSensitivity("total_assets_mchf", 5_000.0)
            )
            representative = ReferenceBankSensitivity("total_assets_mchf", 5_000.0)
            if sensitivity != representative:
                raise OperationalRunError(
                    "Local sensitivity evaluation is limited to the representative "
                    "total_assets_mchf=5000 variant; broader variants require the "
                    "explicitly budgeted research plan"
                )
            evaluator = FrozenPolicySensitivityEvaluator(
                configuration,
                canonical_snapshot=canonical_snapshot,
                historical=historical,
                calibration=calibration,
            )
            result = evaluator.evaluate(
                checkpoints,
                sensitivity=sensitivity,
                validation_requests=validation_requests,
            )
            sensitivity_manifest = evaluator.manifest_data(result)
            manifest = _build_manifest(
                configuration, RunStatus.COMPLETED, AcceptanceStatus.PENDING
            )
            manifest["reference_bank_sensitivity"] = {
                "representative_sensitivity": sensitivity_manifest[
                    "representative_sensitivity"
                ],
                "representative_variant_content_hash": sensitivity_manifest[
                    "representative_variant_content_hash"
                ],
                "training_triggered": False,
                "interpretation": sensitivity_manifest["interpretation"],
            }
            artifact_directory = _write_bundle_atomically(
                configuration,
                manifest,
                extra_artifacts={
                    "reference-bank-sensitivity.json": sensitivity_manifest
                },
            )
            return RunBundle(
                status=RunStatus.COMPLETED,
                acceptance_status=AcceptanceStatus.PENDING,
                artifact_directory=artifact_directory,
                artifacts=(
                    artifact_directory / "manifest.json",
                    artifact_directory / "reference-bank-sensitivity.json",
                ),
            )
        except (OperationalRunError, LockedEvaluationError, OSError, ValueError) as error:
            failure_directory = _write_failure_bundle(configuration, error)
            return RunBundle(
                status=RunStatus.FAILED,
                acceptance_status=AcceptanceStatus.PENDING,
                artifact_directory=failure_directory,
                error=str(error),
                artifacts=(failure_directory / "manifest.json",)
                if failure_directory is not None
                else (),
            )

    def validate_single_device_portability(
        self,
        configuration: ResolvedRunConfiguration,
        *,
        trainer: DeviceValidationTrainer,
        horizon_years: int,
        policy_name: str = "BM^E",
    ) -> RunBundle:
        """Record bounded, actual single-device update evidence atomically.

        CPU always runs. MPS and CUDA are attempted only when PyTorch reports
        their runtime as available; unavailable backends remain explicitly
        ``not-run`` rather than being simulated. This is a portability
        commissioning check, not a claim of cross-device bitwise equality or
        multi-GPU validation.
        """

        try:
            from deepalm.training import TrainingError

            records = trainer.validate_devices(horizon_years=horizon_years)
            serialized_records = [
                {
                    **asdict(record),
                    "checkpoint_path": (
                        str(record.checkpoint_path)
                        if record.checkpoint_path is not None
                        else None
                    ),
                    "recovery_path": (
                        str(record.recovery_path)
                        if record.recovery_path is not None
                        else None
                    ),
                }
                for record in records
            ]
            evidence = {
                "format_version": 1,
                "kind": "single-device-portability-validation",
                "policy": policy_name,
                "horizon_years": horizon_years,
                "records": serialized_records,
                "cross_device_bitwise_equality": "not-promised",
                "deferred_boundary": (
                    "real-bank mapping, multi-GPU/DDP, mixed precision tuning, "
                    "and cluster throughput commissioning"
                ),
            }
            failed = [
                record.device
                for record in records
                if record.status == "completed"
                and (
                    not record.finite
                    or not record.updated
                    or not record.clipped
                    or not record.checkpoint_loaded
                    or not record.resumed_from_cpu_recovery
                )
            ]
            unexpected_statuses = [
                f"{record.device}={record.status}"
                for record in records
                if record.status not in {"completed", "not-run"}
            ]
            if unexpected_statuses:
                raise OperationalRunError(
                    "Single-device validation has an unsupported status: "
                    + ", ".join(unexpected_statuses)
                )
            if failed:
                raise OperationalRunError(
                    "Single-device validation did not produce a finite clipped update: "
                    + ", ".join(failed)
                )
            manifest = _build_manifest(
                configuration, RunStatus.COMPLETED, AcceptanceStatus.PENDING
            )
            manifest["single_device_portability"] = evidence
            artifact_directory = _write_bundle_atomically(
                configuration,
                manifest,
                extra_artifacts={"single-device-validation.json": evidence},
            )
            return RunBundle(
                status=RunStatus.COMPLETED,
                acceptance_status=AcceptanceStatus.PENDING,
                artifact_directory=artifact_directory,
                artifacts=(
                    artifact_directory / "manifest.json",
                    artifact_directory / "single-device-validation.json",
                ),
            )
        except (OperationalRunError, TrainingError, OSError, ValueError) as error:
            failure_directory = _write_failure_bundle(configuration, error)
            return RunBundle(
                status=RunStatus.FAILED,
                acceptance_status=AcceptanceStatus.PENDING,
                artifact_directory=failure_directory,
                error=str(error),
                artifacts=(failure_directory / "manifest.json",)
                if failure_directory is not None
                else (),
            )

    def commission_single_device_portability(
        self,
        configuration: ResolvedRunConfiguration,
        *,
        horizon_years: int = 5,
        policy_name: str = "BM^E",
    ) -> RunBundle:
        """Construct synthetic inputs and run the bounded device-check stage.

        The bank-facing command deliberately commissions only a synthetic
        Reference Bank and one benchmark update. Real-bank input mapping and
        full policy training remain explicit later stages.
        """

        try:
            if horizon_years not in configuration.experiment.horizons_years:
                raise OperationalRunError(
                    "Single-device commissioning horizon is not declared in the configuration"
                )
            from deepalm.reference_bank import ReferenceBankProvider
            from deepalm.term_structures import MarketScenarioModel
            from deepalm.training import (
                BMDateTrainer,
                BMETrainer,
                MMTrainer,
                TrainingError,
            )

            market_model = MarketScenarioModel()
            historical = market_model.load_historical_term_structures(
                configuration.source_data.snb_csv,
                beta_unit=configuration.source_data.nss_beta_unit,
            )
            calibration = market_model.calibrate_hjm_pca(historical)
            snapshot = ReferenceBankProvider().build_canonical(historical)
            if policy_name == "BM^E":
                trainer: DeviceValidationTrainer = BMETrainer(
                    configuration,
                    snapshot=snapshot,
                    historical=historical,
                    calibration=calibration,
                    market_model=market_model,
                )
            elif policy_name == "MM":
                baseline_configuration = replace(
                    configuration,
                    output=replace(
                        configuration.output,
                        run_name=f"{configuration.output.run_name}-bmd-source",
                    ),
                )
                baseline = BMDateTrainer(
                    baseline_configuration,
                    snapshot=snapshot,
                    historical=historical,
                    calibration=calibration,
                    market_model=market_model,
                ).fit(horizon_years=horizon_years)
                if baseline.baseline_reference_path is None:
                    raise TrainingError(
                        "MM commissioning did not create a frozen BM^D reference"
                    )
                from deepalm.baselines import FrozenDateBenchmarkReference

                trainer = MMTrainer(
                    configuration,
                    snapshot=snapshot,
                    historical=historical,
                    calibration=calibration,
                    baseline_reference=FrozenDateBenchmarkReference.load(
                        baseline.baseline_reference_path
                    ),
                    market_model=market_model,
                )
            else:
                raise OperationalRunError(
                    "Single-device commissioning policy must be BM^E or MM"
                )
        except (OperationalRunError, TrainingError, OSError, ValueError) as error:
            failure_directory = _write_failure_bundle(configuration, error)
            return RunBundle(
                status=RunStatus.FAILED,
                acceptance_status=AcceptanceStatus.PENDING,
                artifact_directory=failure_directory,
                error=str(error),
                artifacts=(failure_directory / "manifest.json",)
                if failure_directory is not None
                else (),
            )
        return self.validate_single_device_portability(
            configuration,
            trainer=trainer,
            horizon_years=horizon_years,
            policy_name=policy_name,
        )

    def run_local_workflow(self, configuration: ResolvedRunConfiguration) -> RunBundle:
        """Execute the complete bounded no-swap workflow in one atomic bundle.

        The implementation deliberately reuses the existing stage contracts.  All
        mutable training files are first written to a sibling staging directory;
        they become a completed run only after evaluation, analysis, reporting,
        and acceptance evidence have all been produced.
        """

        staging_directory: Path | None = None
        staged_configuration: ResolvedRunConfiguration | None = None
        try:
            _validate_local_workflow_configuration(configuration)
            output_root = configuration.output.directory
            output_root.mkdir(parents=True, exist_ok=True)
            final_directory = output_root / configuration.output.run_name
            if final_directory.exists():
                raise OperationalRunError(
                    f"Run artifact directory already exists: {final_directory}"
                )
            staging_directory = Path(
                tempfile.mkdtemp(prefix=f".{configuration.output.run_name}-", dir=output_root)
            )
            staged_configuration = replace(
                configuration,
                output=replace(
                    configuration.output,
                    directory=staging_directory.parent,
                    run_name=staging_directory.name,
                ),
            )
            monitor = ResourceMonitor(
                wall_clock_budget_seconds=configuration.resources.wall_clock_budget_seconds,
                process_rss_limit_bytes=configuration.resources.process_rss_limit_bytes,
                accelerator_memory_limit_bytes=(
                    configuration.resources.accelerator_memory_limit_bytes
                ),
                device=configuration.optimization.device,
            )
            monitor.check("before-local-workflow")

            from deepalm.analysis import (
                FrozenPolicyTrajectory,
                HorizonScenarioAnalyzer,
            )
            from deepalm.baselines import FrozenDateBenchmarkReference
            from deepalm.evaluation import (
                FrozenPolicySensitivityEvaluator,
                LockedEvaluator,
                PolicyCheckpoint,
            )
            from deepalm.objective import evaluation_objective_parameters
            from deepalm.reference_bank import (
                ReferenceBankProvider,
                ReferenceBankSensitivity,
            )
            from deepalm.reporting import build_compact_no_swap_report
            from deepalm.runoff import ALMSimulator
            from deepalm.term_structures import MarketScenarioModel
            from deepalm.training import (
                BMConstantTrainer,
                BMDateTrainer,
                BMETrainer,
                MMTrainer,
                TrainingControl,
                TrainingInterrupted,
            )

            market_model = MarketScenarioModel()
            historical = market_model.load_historical_term_structures(
                staged_configuration.source_data.snb_csv,
                beta_unit=staged_configuration.source_data.nss_beta_unit,
            )
            calibration = market_model.calibrate_hjm_pca(historical)
            monitor.check("after-local-market-calibration")
            provider = ReferenceBankProvider()
            snapshot = provider.build_canonical(historical)
            bank_path = staging_directory / "reference-bank.json"
            provider.save(snapshot, bank_path)
            replayed_snapshot = provider.load(bank_path)
            if replayed_snapshot.content_hash != snapshot.content_hash:
                raise OperationalRunError("Reference Bank replay changed its content hash")
            monitor.check("after-local-reference-bank")

            replay = _workflow_replay_evidence(
                market_model,
                historical=historical,
                calibration=calibration,
                configuration=staged_configuration,
                snapshot_hash=snapshot.content_hash,
                replayed_snapshot_hash=replayed_snapshot.content_hash,
            )
            diagnostics = market_model.validate_hjm_one_step(
                calibration,
                convention=staged_configuration.convention.profile,
                paths=50_000,
                seed=staged_configuration.seeds["market_scenarios"],
            )
            factor_mean_limit = 3.0 / np.sqrt(50_000)
            one_step_passed = bool(
                np.all(np.abs(diagnostics.factor_means) <= factor_mean_limit)
                and diagnostics.relative_covariance_error <= 0.05
            )
            one_step_statistics = {
                "paths": 50_000,
                "factor_means": diagnostics.factor_means.tolist(),
                "factor_mean_abs_limit": factor_mean_limit,
                "relative_covariance_error": diagnostics.relative_covariance_error,
                "relative_covariance_error_limit": 0.05,
                "independent_oracle": "fitted PCA covariance",
                "status": "passed" if one_step_passed else "failed",
                "data_identities": {
                    "market_source_hash": historical.source_hash,
                    "hjm_calibration_identity": calibration.calibration_identity,
                },
                "runtime": _runtime_identity(staged_configuration),
                "git_revision": _git_revision(),
            }
            monitor.check("after-local-one-step-statistics")

            training_results: dict[tuple[str, int], Any] = {}
            bme = BMETrainer(
                staged_configuration,
                snapshot=snapshot,
                historical=historical,
                calibration=calibration,
                market_model=market_model,
                resource_monitor=monitor,
            )
            for horizon in staged_configuration.experiment.horizons_years:
                if horizon == 5:
                    try:
                        bme.fit(
                            horizon_years=horizon,
                            control=TrainingControl(stop_after_completed_epoch=1),
                        )
                    except TrainingInterrupted as interruption:
                        training_results[("BM^E", horizon)] = bme.fit(
                            horizon_years=horizon,
                            control=TrainingControl(
                                resume=True,
                                resume_from=interruption.recovery_path,
                            ),
                        )
                    else:
                        raise OperationalRunError(
                            "Short recovery verification did not interrupt BM^E 5y"
                        )
                else:
                    training_results[("BM^E", horizon)] = bme.fit(
                        horizon_years=horizon
                    )
            recovery_oracle_configuration = replace(
                staged_configuration,
                output=replace(
                    staged_configuration.output,
                    run_name=f"{staged_configuration.output.run_name}-recovery-oracle",
                ),
            )
            recovery_oracle_directory = (
                recovery_oracle_configuration.output.directory
                / recovery_oracle_configuration.output.run_name
            )
            try:
                uninterrupted = BMETrainer(
                    recovery_oracle_configuration,
                    snapshot=snapshot,
                    historical=historical,
                    calibration=calibration,
                    market_model=market_model,
                    resource_monitor=monitor,
                ).fit(horizon_years=5)
                recovery = torch.load(
                    staging_directory / "BM_E_5y.recovery.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                recovery_evidence = _recovery_equivalence_evidence(
                    resumed_checkpoint=training_results[("BM^E", 5)].checkpoint_path,
                    uninterrupted_checkpoint=uninterrupted.checkpoint_path,
                    recovery=recovery,
                )
            finally:
                shutil.rmtree(recovery_oracle_directory, ignore_errors=True)
            if recovery_evidence["status"] != "passed":
                raise OperationalRunError(
                    "Short recovery diverged from uninterrupted training: "
                    + json.dumps(recovery_evidence, sort_keys=True)
                )
            monitor.check("after-local-bme-training")

            bmc = BMConstantTrainer(
                staged_configuration,
                snapshot=snapshot,
                historical=historical,
                calibration=calibration,
                market_model=market_model,
                resource_monitor=monitor,
            )
            for horizon in staged_configuration.experiment.horizons_years:
                training_results[("BM^C", horizon)] = bmc.fit(horizon_years=horizon)
            monitor.check("after-local-bmc-training")

            bmd = BMDateTrainer(
                staged_configuration,
                snapshot=snapshot,
                historical=historical,
                calibration=calibration,
                market_model=market_model,
                resource_monitor=monitor,
            )
            baselines: dict[int, FrozenDateBenchmarkReference] = {}
            for horizon in staged_configuration.experiment.horizons_years:
                result = bmd.fit(horizon_years=horizon)
                if result.baseline_reference_path is None:
                    raise OperationalRunError("BM^D training did not freeze a baseline")
                training_results[("BM^D", horizon)] = result
                baselines[horizon] = FrozenDateBenchmarkReference.load(
                    result.baseline_reference_path
                )
            monitor.check("after-local-bmd-training")

            mm_trainers: dict[int, MMTrainer] = {}
            for horizon in staged_configuration.experiment.horizons_years:
                trainer = MMTrainer(
                    staged_configuration,
                    snapshot=snapshot,
                    historical=historical,
                    calibration=calibration,
                    baseline_reference=baselines[horizon],
                    market_model=market_model,
                    resource_monitor=monitor,
                )
                mm_trainers[horizon] = trainer
                training_results[("MM", horizon)] = trainer.fit(horizon_years=horizon)
            monitor.check("after-local-mm-training")

            width_checks = tuple(
                mm_trainers[horizon].run_paper_width_check(horizon_years=horizon)
                for horizon in staged_configuration.experiment.horizons_years
            )
            if not all(
                item.finite and item.updated and item.clipped
                for item in width_checks
            ):
                raise OperationalRunError("A paper-width MM check did not complete")
            monitor.check("after-local-paper-width-checks")

            checkpoints = tuple(
                PolicyCheckpoint(f"{policy}-{horizon}y", result.checkpoint_path)
                for (policy, horizon), result in training_results.items()
            )
            locked_evaluator = LockedEvaluator(
                staged_configuration,
                snapshot=snapshot,
                historical=historical,
                calibration=calibration,
                market_model=market_model,
            )
            locked = locked_evaluator.evaluate(checkpoints)
            locked_evaluator.write_manifest(locked, staging_directory / "locked-evaluation.json")
            monitor.check("after-local-locked-evaluation")

            fifteen_market = market_model.generate_hjm_scenarios(
                historical,
                calibration,
                convention=staged_configuration.convention.profile,
                horizon_years=15,
                paths=staged_configuration.run_scale.test_paths,
                seed=staged_configuration.seeds["market_scenarios"],
                split="horizon-analysis",
                epoch=0,
                global_path_indices=tuple(range(staged_configuration.run_scale.test_paths)),
            )
            five_market = fifteen_market.prefix(horizon_years=5)
            truncation = mm_trainers[15].evaluate_five_year_truncation(
                checkpoint_path=training_results[("MM", 15)].checkpoint_path,
                market=fifteen_market,
            )
            _write_json_artifact(
                staging_directory / "mm-truncation.json",
                {
                    "format_version": 1,
                    "kind": "mm-fifteen-year-truncation",
                    "source_horizon_years": 15,
                    "evaluation_horizon_years": 5,
                    "convention": staged_configuration.convention.profile,
                    "data_identities": {
                        "market_source_hash": historical.source_hash,
                        "hjm_calibration_identity": calibration.calibration_identity,
                        "reference_bank_content_hash": snapshot.content_hash,
                    },
                    "optimizer_updates": truncation.optimizer_updates,
                    "checkpoint": training_results[("MM", 15)].checkpoint_path.name,
                    "checkpoint_sha256": _sha256(
                        training_results[("MM", 15)].checkpoint_path
                    ),
                },
            )
            simulator = ALMSimulator()
            dtype = getattr(torch, staged_configuration.optimization.dtype)
            mm_five_outcome = simulator.rollout(
                snapshot,
                five_market,
                policy=training_results[("MM", 5)].policy,
                device=staged_configuration.optimization.device,
                dtype=dtype,
                include_loan_dynamics=True,
                convention=staged_configuration.convention.profile,
                include_deposit_dynamics=True,
                objective_parameters=evaluation_objective_parameters(
                    len(five_market.spot_rates),
                    horizon_years=5,
                    device=staged_configuration.optimization.device,
                    dtype=dtype,
                ),
            )
            mm_fifteen_outcome = simulator.rollout(
                snapshot,
                fifteen_market,
                policy=training_results[("MM", 15)].policy,
                device=staged_configuration.optimization.device,
                dtype=dtype,
                include_loan_dynamics=True,
                convention=staged_configuration.convention.profile,
                include_deposit_dynamics=True,
                objective_parameters=evaluation_objective_parameters(
                    len(fifteen_market.spot_rates),
                    horizon_years=15,
                    device=staged_configuration.optimization.device,
                    dtype=dtype,
                ),
            )
            if (
                mm_five_outcome.treasury_actions is None
                or mm_fifteen_outcome.treasury_actions is None
                or truncation.outcome.treasury_actions is None
            ):
                raise OperationalRunError("Horizon analysis requires complete MM actions")
            financial_rollout = _financial_rollout_evidence(
                five_year=mm_five_outcome,
                fifteen_year=mm_fifteen_outcome,
                truncation=truncation.outcome,
            )
            category_size = min(
                5,
                staged_configuration.run_scale.test_paths,
                int(np.sum(five_market.spot_rates[:, -1, 0] > 0.02)),
            )
            if category_size == 0:
                raise OperationalRunError(
                    "Horizon scenario analysis has no upward-rate category path"
                )
            try:
                horizon_analysis = HorizonScenarioAnalyzer(five_market).analyze(
                    mm_five_year=FrozenPolicyTrajectory(
                        policy_label="MM(5y)",
                        actions=mm_five_outcome.treasury_actions,
                        source_horizon_years=5,
                        evaluation_market=five_market,
                        checkpoint_identity=_sha256(training_results[("MM", 5)].checkpoint_path),
                        optimizer_updates=training_results[("MM", 5)].optimizer_updates,
                        time_feature_horizon_years=5,
                    ),
                    mm_fifteen_year=FrozenPolicyTrajectory(
                        policy_label="MM(15y)",
                        actions=mm_fifteen_outcome.treasury_actions,
                        source_horizon_years=15,
                        evaluation_market=fifteen_market,
                        checkpoint_identity=_sha256(training_results[("MM", 15)].checkpoint_path),
                        optimizer_updates=training_results[("MM", 15)].optimizer_updates,
                        time_feature_horizon_years=15,
                    ),
                    mm_fifteen_year_truncated=FrozenPolicyTrajectory(
                        policy_label="MM(15y|5y)",
                        actions=truncation.outcome.treasury_actions,
                        source_horizon_years=15,
                        evaluation_market=five_market,
                        checkpoint_identity=_sha256(training_results[("MM", 15)].checkpoint_path),
                        optimizer_updates=0,
                        time_feature_horizon_years=15,
                    ),
                    bootstrap_seed=staged_configuration.seeds["bootstrap"],
                    category_size=category_size,
                )
            except ValueError as error:
                horizon_analysis = None
                horizon_analysis_reason = str(error)
            else:
                horizon_analysis_reason = None
            _write_json_artifact(
                staging_directory / "horizon-scenario-analysis.json",
                _horizon_analysis_evidence(
                    horizon_analysis,
                    configuration=staged_configuration,
                    historical=historical,
                    calibration=calibration,
                    snapshot_hash=snapshot.content_hash,
                    unavailable_reason=horizon_analysis_reason,
                ),
            )
            monitor.check("after-local-horizon-analysis")

            sensitivity_evaluator = FrozenPolicySensitivityEvaluator(
                staged_configuration,
                canonical_snapshot=snapshot,
                historical=historical,
                calibration=calibration,
                market_model=market_model,
            )
            sensitivity = sensitivity_evaluator.evaluate(
                (PolicyCheckpoint("BM^E-5y", training_results[("BM^E", 5)].checkpoint_path),),
                sensitivity=ReferenceBankSensitivity("total_assets_mchf", 5_000.0),
            )
            sensitivity_evaluator.write_manifest(
                sensitivity, staging_directory / "reference-bank-sensitivity.json"
            )
            monitor.check("after-local-sensitivity")

            completed_jobs = _workflow_training_jobs(training_results)
            preliminary_acceptance = _local_workflow_acceptance(
                configuration=staged_configuration,
                completed_jobs=completed_jobs,
                width_checks=width_checks,
                replay=replay,
                one_step_statistics=one_step_statistics,
                locked=locked,
                sensitivity=sensitivity,
                horizon_analysis_available=horizon_analysis is not None,
                financial_rollout=financial_rollout,
                recovery=recovery_evidence,
            )
            workflow = {
                "purpose": "development-validation",
                "development_validation_status": (
                    "development-validated"
                    if preliminary_acceptance["status"] == "passed"
                    else "acceptance-failed"
                ),
                "completed_training_jobs": completed_jobs,
                "completed_primary_optimizer_updates": sum(
                    int(item["optimizer_updates"]) for item in completed_jobs
                ),
                "paper_width_checks": {
                    "completed_optimizer_updates": sum(
                        item.optimizer_updates for item in width_checks
                    ),
                    "artifacts": [item.artifact_path.name for item in width_checks],
                },
                "replay": replay,
                "one_step_market_statistics": one_step_statistics,
                "financial_rollout": financial_rollout,
                "recovery": recovery_evidence,
                "additional_recovery_optimizer_updates": uninterrupted.optimizer_updates,
                "unavailable_device_checks": _unavailable_device_checks(),
                "resource_measurements": monitor.snapshot().to_dict(),
                "overshoot_seconds": max(
                    0.0,
                    monitor.snapshot().elapsed_seconds
                    - configuration.resources.wall_clock_budget_seconds,
                ),
                "acceptance_evidence": [
                    "locked-evaluation.json",
                    "reference-bank-sensitivity.json",
                    "horizon-scenario-analysis.json",
                    "mm-truncation.json",
                    "acceptance-report.json",
                ],
            }
            source_manifest = _build_manifest(
                staged_configuration, RunStatus.COMPLETED, AcceptanceStatus.PASSED
            )
            source_manifest["market_preflight"] = {
                "calibration": {"identity": calibration.calibration_identity},
                "weekly_dates": [str(date) for date in calibration.weekly_dates],
            }
            source_manifest["reference_bank"] = {
                "content_hash": snapshot.content_hash,
                "snapshot": bank_path.name,
            }
            source_manifest["local_workflow"] = workflow
            _write_json_artifact(staging_directory / "manifest.json", source_manifest)
            monitor.check("before-local-report")
            report_artifacts = build_compact_no_swap_report(
                staged_configuration, source_run_directories=(staging_directory,)
            )
            _write_json_artifact(
                staging_directory / "compact-no-swap-report.json", report_artifacts.report
            )
            _write_json_artifact(
                staging_directory / "paper-coverage-inventory.json",
                report_artifacts.coverage_inventory,
            )
            _write_json_artifact(
                staging_directory / "calibration-evidence.json",
                report_artifacts.calibration_evidence,
            )
            _write_json_artifact(
                staging_directory / "hjm-hull-white-comparison.json",
                report_artifacts.market_comparison,
            )
            _write_json_artifact(
                staging_directory / "reference-bank-summary.json",
                report_artifacts.reference_bank_summary,
            )
            acceptance = _local_workflow_acceptance(
                configuration=staged_configuration,
                completed_jobs=completed_jobs,
                width_checks=width_checks,
                replay=replay,
                one_step_statistics=one_step_statistics,
                locked=locked,
                sensitivity=sensitivity,
                horizon_analysis_available=horizon_analysis is not None,
                financial_rollout=financial_rollout,
                recovery=recovery_evidence,
                report=report_artifacts.report,
            )
            _write_json_artifact(staging_directory / "acceptance-report.json", acceptance)
            accepted = acceptance["status"] == "passed"
            workflow["development_validation_status"] = (
                "development-validated" if accepted else "acceptance-failed"
            )
            reusable_artifacts = sorted(
                {
                    artifact
                    for stage in ("train", "resume", "evaluate", "accept")
                    for artifact in _workflow_stage_artifacts(stage)
                }
            )
            workflow["stage_artifact_sha256"] = {
                name: _sha256(staging_directory / name) for name in reusable_artifacts
            }
            workflow["generated_artifacts"] = sorted(
                path.name for path in staging_directory.iterdir() if path.is_file()
            )
            workflow["deferred_artifacts"] = report_artifacts.report["deferred_research"]
            workflow["resource_measurements"] = monitor.snapshot().to_dict()
            workflow["overshoot_seconds"] = max(
                0.0,
                monitor.snapshot().elapsed_seconds
                - configuration.resources.wall_clock_budget_seconds,
            )
            source_manifest["status"] = _status_for(
                AcceptanceStatus.PASSED if accepted else AcceptanceStatus.FAILED
            ).value
            source_manifest["acceptance_status"] = (
                AcceptanceStatus.PASSED if accepted else AcceptanceStatus.FAILED
            ).value
            source_manifest["requested_configuration"] = configuration.to_dict()
            source_manifest["local_workflow"] = workflow
            source_manifest["compact_report"] = {
                "kind": report_artifacts.report["kind"],
                "purpose": report_artifacts.report["purpose"],
                "coverage_inventory": "paper-coverage-inventory.json",
                "calibration_identity": calibration.calibration_identity,
                "reference_bank_content_hash": snapshot.content_hash,
                "weekly_dates": [str(date) for date in calibration.weekly_dates],
                "acceptance_evidence": workflow["acceptance_evidence"],
            }
            _write_json_artifact(staging_directory / "manifest.json", source_manifest)
            monitor.check("after-local-report")
            os.replace(staging_directory, final_directory)
            artifact_names = tuple(sorted(path for path in final_directory.iterdir() if path.is_file()))
            return RunBundle(
                status=_status_for(
                    AcceptanceStatus.PASSED if accepted else AcceptanceStatus.FAILED
                ),
                acceptance_status=(
                    AcceptanceStatus.PASSED if accepted else AcceptanceStatus.FAILED
                ),
                artifact_directory=final_directory,
                artifacts=artifact_names,
                checkpoints=tuple(
                    final_directory / item["checkpoint"] for item in completed_jobs
                ),
                acceptance_evidence=tuple(
                    final_directory / name for name in workflow["acceptance_evidence"]
                ),
            )
        except BudgetExceeded as error:
            return _workflow_failure_bundle(
                configuration,
                staging_directory=staging_directory,
                staged_configuration=staged_configuration,
                error=error,
                status=RunStatus.INCOMPLETE,
            )
        except (OperationalRunError, OSError, RuntimeError, ValueError) as error:
            return _workflow_failure_bundle(
                configuration,
                staging_directory=staging_directory,
                staged_configuration=staged_configuration,
                error=error,
                status=(
                    RunStatus.INCOMPLETE
                    if getattr(error, "reason", None)
                    in {"budget_exhausted", "cancelled", "out_of_memory"}
                    else RunStatus.FAILED
                ),
            )

    def run_paired_convention_pilot(
        self, configuration: ResolvedRunConfiguration
    ) -> RunBundle:
        """Run the bounded, opt-in Paper/Corrected M5 training pilot.

        The two conventions deliberately share only immutable inputs and named
        random streams.  Their checkpoint directories and frozen BM^D baselines
        remain convention-local, preventing accidental cross-convention reuse.
        """

        staging_directory: Path | None = None
        staged_configuration: ResolvedRunConfiguration | None = None
        try:
            _validate_paired_pilot_configuration(configuration)
            output_root = configuration.output.directory
            output_root.mkdir(parents=True, exist_ok=True)
            final_directory = output_root / configuration.output.run_name
            if final_directory.exists():
                raise OperationalRunError(
                    f"Run artifact directory already exists: {final_directory}"
                )
            staging_directory = Path(
                tempfile.mkdtemp(
                    prefix=f".{configuration.output.run_name}-", dir=output_root
                )
            )
            staged_configuration = replace(
                configuration,
                output=replace(
                    configuration.output,
                    directory=staging_directory.parent,
                    run_name=staging_directory.name,
                ),
            )
            monitor = ResourceMonitor(
                wall_clock_budget_seconds=configuration.resources.wall_clock_budget_seconds,
                process_rss_limit_bytes=configuration.resources.process_rss_limit_bytes,
                accelerator_memory_limit_bytes=(
                    configuration.resources.accelerator_memory_limit_bytes
                ),
                device=configuration.optimization.device,
            )
            from deepalm.baselines import FrozenDateBenchmarkReference
            from deepalm.reference_bank import ReferenceBankProvider
            from deepalm.term_structures import MarketScenarioModel
            from deepalm.training import (
                BMDateTrainer,
                MMTrainer,
                TrainingControl,
                TrainingInterrupted,
            )

            monitor.check("before-paired-pilot")
            market_model = MarketScenarioModel()
            historical = market_model.load_historical_term_structures(
                configuration.source_data.snb_csv,
                beta_unit=configuration.source_data.nss_beta_unit,
            )
            calibration = market_model.calibrate_hjm_pca(historical)
            snapshot = ReferenceBankProvider().build_canonical(historical)
            ReferenceBankProvider().save(snapshot, staging_directory / "reference-bank.json")
            monitor.check("after-paired-pilot-inputs")

            convention_configurations = _paired_pilot_convention_configurations(
                configuration, staging_directory=staging_directory
            )
            training_results: dict[tuple[str, str, int], Any] = {}
            baselines: dict[tuple[str, int], Any] = {}
            mm_trainers: dict[str, Any] = {}
            recoveries: dict[str, Path] = {}
            probe_started = monitor.snapshot().elapsed_seconds

            for label, convention_configuration in convention_configurations.items():
                bmd = BMDateTrainer(
                    convention_configuration,
                    snapshot=snapshot,
                    historical=historical,
                    calibration=calibration,
                    market_model=market_model,
                    resource_monitor=monitor,
                )
                result = bmd.fit(horizon_years=15)
                if result.baseline_reference_path is None:
                    raise OperationalRunError("BM^D 15y did not freeze a baseline")
                training_results[(label, "BM^D", 15)] = result
                baselines[(label, 15)] = FrozenDateBenchmarkReference.load(
                    result.baseline_reference_path
                )
                trainer = MMTrainer(
                    convention_configuration,
                    snapshot=snapshot,
                    historical=historical,
                    calibration=calibration,
                    baseline_reference=baselines[(label, 15)],
                    market_model=market_model,
                    resource_monitor=monitor,
                )
                mm_trainers[label] = trainer
                try:
                    trainer.fit(
                        horizon_years=15,
                        control=TrainingControl(stop_after_completed_epoch=1),
                    )
                except TrainingInterrupted as interruption:
                    if (
                        interruption.diagnostics.get("completed_epoch") != 1
                        or not interruption.recovery_path.is_file()
                    ):
                        raise
                    recoveries[label] = interruption.recovery_path
                else:
                    raise OperationalRunError(
                        "Paired pilot did not stop after MM 15y probe epoch"
                    )
                monitor.check(f"after-{label}-mm-15y-probe")

            probe_seconds = monitor.snapshot().elapsed_seconds - probe_started
            observed_primary_updates = 12
            estimated_training_seconds = probe_seconds * 32.0 / observed_primary_updates
            _write_json_artifact(
                staging_directory / "pilot-throughput-probe.json",
                {
                    "format_version": 1,
                    "kind": "paired-convention-pilot-throughput-probe",
                    "observed_mm_fifteen_year_first_epochs": 2,
                    "observed_optimizer_updates": observed_primary_updates,
                    "estimated_primary_training_seconds": estimated_training_seconds,
                    "maximum_primary_training_seconds": 420.0,
                    "status": (
                        "within-budget"
                        if estimated_training_seconds <= 420.0
                        else "over-budget"
                    ),
                },
            )
            if estimated_training_seconds > 420.0:
                raise BudgetExceeded(
                    "paired pilot projected training time exceeds 420 seconds",
                    diagnostics={
                        "reason": "projected_training_budget_exceeded",
                        "observed_probe_seconds": probe_seconds,
                        "estimated_primary_training_seconds": estimated_training_seconds,
                        "maximum_primary_training_seconds": 420.0,
                    },
                )

            for label, convention_configuration in convention_configurations.items():
                training_results[(label, "MM", 15)] = mm_trainers[label].fit(
                    horizon_years=15,
                    control=TrainingControl(resume=True, resume_from=recoveries[label]),
                )
                bmd = BMDateTrainer(
                    convention_configuration,
                    snapshot=snapshot,
                    historical=historical,
                    calibration=calibration,
                    market_model=market_model,
                    resource_monitor=monitor,
                )
                result = bmd.fit(horizon_years=5)
                if result.baseline_reference_path is None:
                    raise OperationalRunError("BM^D 5y did not freeze a baseline")
                training_results[(label, "BM^D", 5)] = result
                baseline = FrozenDateBenchmarkReference.load(result.baseline_reference_path)
                mm = MMTrainer(
                    convention_configuration,
                    snapshot=snapshot,
                    historical=historical,
                    calibration=calibration,
                    baseline_reference=baseline,
                    market_model=market_model,
                    resource_monitor=monitor,
                )
                training_results[(label, "MM", 5)] = mm.fit(horizon_years=5)
                monitor.check(f"after-{label}-paired-pilot-training")

            jobs = _paired_pilot_training_jobs(
                training_results, artifact_root=staging_directory
            )
            if len(jobs) != 8 or sum(int(job["optimizer_updates"]) for job in jobs) != 32:
                raise OperationalRunError("Paired pilot did not complete its 8-job matrix")
            manifest = _build_manifest(
                staged_configuration,
                RunStatus.COMPLETED,
                AcceptanceStatus.PAIRED_CONVENTION_RESEARCH_PILOT,
            )
            manifest["requested_configuration"] = configuration.to_dict()
            manifest["paired_convention_pilot"] = {
                "label": "paired-convention-research-pilot",
                "status": "completed",
                "conventions": {
                    label: convention_configuration.convention.profile
                    for label, convention_configuration in convention_configurations.items()
                },
                "shared_identities": {
                    "reference_bank_content_hash": snapshot.content_hash,
                    "market_source_hash": historical.source_hash,
                    "hjm_calibration_identity": calibration.calibration_identity,
                    "seed_registry": configuration.seeds,
                },
                "completed_training_jobs": jobs,
                "completed_primary_optimizer_updates": sum(
                    int(job["optimizer_updates"]) for job in jobs
                ),
                "mm_fifteen_year_truncation": {
                    label: {
                        "source_checkpoint": str(
                            training_results[(label, "MM", 15)].checkpoint_path.relative_to(
                                staging_directory
                            )
                        ),
                        "source_horizon_years": 15,
                        "evaluation_horizon_years": 5,
                        "optimizer_updates": 0,
                        "evaluation_stage": "ticket-29",
                    }
                    for label in convention_configurations
                },
                "throughput_probe": {
                    "observed_seconds": probe_seconds,
                    "estimated_primary_training_seconds": estimated_training_seconds,
                    "maximum_primary_training_seconds": 420.0,
                },
                "resource_measurements": monitor.snapshot().to_dict(),
                "deferred": [
                    "locked paired evaluation and bootstrap (ticket-29)",
                    "paired pilot report (ticket-30)",
                    "paper widths, multi-seed, sensitivity, and economic assessment",
                ],
            }
            _write_json_artifact(staging_directory / "manifest.json", manifest)
            monitor.check("after-paired-pilot")
            os.replace(staging_directory, final_directory)
            checkpoints = tuple(
                final_directory / result.checkpoint_path.relative_to(staging_directory)
                for result in training_results.values()
            )
            return RunBundle(
                status=RunStatus.COMPLETED,
                acceptance_status=AcceptanceStatus.PAIRED_CONVENTION_RESEARCH_PILOT,
                artifact_directory=final_directory,
                artifacts=tuple(sorted(final_directory.iterdir())),
                checkpoints=checkpoints,
            )
        except BudgetExceeded as error:
            return _paired_pilot_failure_bundle(
                configuration,
                staging_directory=staging_directory,
                staged_configuration=staged_configuration,
                error=error,
                status=RunStatus.INCOMPLETE,
            )
        except (OperationalRunError, OSError, RuntimeError, ValueError) as error:
            return _paired_pilot_failure_bundle(
                configuration,
                staging_directory=staging_directory,
                staged_configuration=staged_configuration,
                error=error,
                status=(
                    RunStatus.INCOMPLETE
                    if getattr(error, "reason", None)
                    in {"budget_exhausted", "cancelled", "out_of_memory"}
                    else RunStatus.FAILED
                ),
            )

    def generate_compact_report(
        self,
        configuration: ResolvedRunConfiguration,
        *,
        source_run_directories: tuple[Path, ...],
    ) -> RunBundle:
        """Publish one no-swap local report from completed evidence only.

        The source bundles are immutable evidence inputs.  A failed or
        incomplete source produces a diagnostic failure bundle rather than a
        report that could be mistaken for a completed local workflow.
        """

        try:
            from deepalm.reporting import ReportingError, build_compact_no_swap_report

            artifacts = build_compact_no_swap_report(
                configuration, source_run_directories=source_run_directories
            )
            manifest = _build_manifest(
                configuration, RunStatus.COMPLETED, AcceptanceStatus.PENDING
            )
            manifest["compact_report"] = {
                "kind": artifacts.report["kind"],
                "purpose": artifacts.report["purpose"],
                "source_runs": artifacts.report["source_runs"],
                "swaps_included": False,
                "coverage_inventory": "paper-coverage-inventory.json",
                "calibration_identity": artifacts.calibration_evidence[
                    "calibration_identity"
                ],
                "weekly_dates": artifacts.calibration_evidence["weekly_dates"],
                "reference_bank_content_hash": artifacts.reference_bank_summary[
                    "content_hash"
                ],
                "checkpoints": artifacts.report["presentation_metadata"][
                    "parameter_counts"
                ],
                "actual_work": artifacts.report["actual_work"],
                "resource_measurements": artifacts.report["actual_work"][
                    "resource_measurements"
                ],
                "metrics": artifacts.report["actual_work"]["numeric_metrics"],
                "acceptance_evidence": [
                    "calibration-evidence.json",
                    "hjm-hull-white-comparison.json",
                    "reference-bank-summary.json",
                    "paper-coverage-inventory.json",
                ],
            }
            artifact_directory = _write_bundle_atomically(
                configuration,
                manifest,
                extra_artifacts={
                    "compact-no-swap-report.json": artifacts.report,
                    "paper-coverage-inventory.json": artifacts.coverage_inventory,
                    "calibration-evidence.json": artifacts.calibration_evidence,
                    "hjm-hull-white-comparison.json": artifacts.market_comparison,
                    "reference-bank-summary.json": artifacts.reference_bank_summary,
                },
            )
            return RunBundle(
                status=RunStatus.COMPLETED,
                acceptance_status=AcceptanceStatus.PENDING,
                artifact_directory=artifact_directory,
                artifacts=(
                    artifact_directory / "manifest.json",
                    artifact_directory / "compact-no-swap-report.json",
                    artifact_directory / "paper-coverage-inventory.json",
                    artifact_directory / "calibration-evidence.json",
                    artifact_directory / "hjm-hull-white-comparison.json",
                    artifact_directory / "reference-bank-summary.json",
                ),
            )
        except (ReportingError, OSError, ValueError) as error:
            failure_directory = _write_failure_bundle(configuration, error)
            return RunBundle(
                status=RunStatus.FAILED,
                acceptance_status=AcceptanceStatus.PENDING,
                artifact_directory=failure_directory,
                error=str(error),
                artifacts=(failure_directory / "manifest.json",)
                if failure_directory is not None
                else (),
            )


def _validate_local_workflow_configuration(
    configuration: ResolvedRunConfiguration,
) -> None:
    """Keep the development-validation claim tied to its declared matrix."""

    if configuration.acceptance.purpose != "development-validation":
        raise OperationalRunError(
            "Local workflow requires acceptance purpose development-validation"
        )
    if configuration.acceptance.required_status != "development-validated":
        raise OperationalRunError(
            "Local workflow requires development-validated as its requested status"
        )
    if configuration.experiment.horizons_years != (5, 15):
        raise OperationalRunError("Local workflow requires both 5- and 15-year horizons")
    if set(configuration.policy.names) != {"BM^E", "BM^C", "BM^D", "MM"}:
        raise OperationalRunError(
            "Local workflow requires BM^E, BM^C, BM^D, and MM exactly once"
        )
    if configuration.experiment.include_swaps:
        raise OperationalRunError("Local workflow excludes swaps and MM^S")
    if configuration.run_scale.epochs < 2:
        raise OperationalRunError(
            "Local workflow requires two epochs to exercise epoch-boundary recovery"
        )


def _validate_paired_pilot_configuration(
    configuration: ResolvedRunConfiguration,
) -> None:
    """Defend the opt-in pilot contract for callers that bypass YAML resolution."""

    scale = configuration.run_scale
    if scale.profile != "paired_convention_pilot":
        raise OperationalRunError("Paired pilot requires paired_convention_pilot scale")
    if configuration.acceptance.purpose != "research":
        raise OperationalRunError("Paired pilot requires research purpose")
    if (
        configuration.acceptance.required_status
        != AcceptanceStatus.PAIRED_CONVENTION_RESEARCH_PILOT.value
    ):
        raise OperationalRunError("Paired pilot requires its research-pilot label")
    if (
        configuration.convention.profile != "corrected"
        or configuration.convention.is_custom
    ):
        raise OperationalRunError("Paired pilot requires locked Corrected input")
    if set(configuration.policy.names) != {"BM^D", "MM"}:
        raise OperationalRunError("Paired pilot requires BM^D and MM exactly once")
    if set(configuration.experiment.horizons_years) != {5, 15}:
        raise OperationalRunError("Paired pilot requires 5- and 15-year horizons")
    if configuration.experiment.include_swaps:
        raise OperationalRunError("Paired pilot excludes swaps")
    if (
        configuration.optimization.device != "mps"
        or configuration.optimization.dtype != "float32"
    ):
        raise OperationalRunError("Paired pilot requires MPS float32")
    if not torch.backends.mps.is_available():
        raise OperationalRunError("Paired pilot requires an available MPS runtime")
    if configuration.architecture.profile != "compact":
        raise OperationalRunError("Paired pilot requires compact architecture")
    if (
        scale.epochs,
        scale.training_paths_per_epoch,
        scale.selection_paths,
        scale.test_paths,
        scale.batch_size,
    ) != (2, 16, 16, 64, 8):
        raise OperationalRunError("Paired pilot scale differs from its locked M5 budget")
    if configuration.resources.wall_clock_budget_seconds != 600:
        raise OperationalRunError("Paired pilot requires a 600-second total budget")
    limit = 12 * 1024**3
    if (
        configuration.resources.process_rss_limit_bytes != limit
        or configuration.resources.accelerator_memory_limit_bytes != limit
    ):
        raise OperationalRunError("Paired pilot requires 12 GiB RSS and MPS guards")
    if configuration.reference_bank.sensitivity is not None:
        raise OperationalRunError("Paired pilot excludes sensitivity retraining")


def _paired_pilot_convention_configurations(
    configuration: ResolvedRunConfiguration, *, staging_directory: Path
) -> dict[str, ResolvedRunConfiguration]:
    """Derive isolated locked Paper and Corrected training configurations."""

    conventions = {
        "corrected": ConventionConfiguration(
            profile="corrected",
            pca_loading_scale="sqrt_eigenvalue",
            loan_interest_annualization="monthly",
            is_custom=False,
        ),
        "paper": ConventionConfiguration(
            profile="paper",
            pca_loading_scale="eigenvalue",
            loan_interest_annualization="unannualized",
            is_custom=False,
        ),
    }
    return {
        label: replace(
            configuration,
            convention=convention,
            output=replace(
                configuration.output,
                directory=staging_directory,
                run_name=label,
            ),
        )
        for label, convention in conventions.items()
    }


def _paired_pilot_training_jobs(
    results: Mapping[tuple[str, str, int], Any], *, artifact_root: Path
) -> list[dict[str, object]]:
    """Serialize completed jobs with their convention-local baseline evidence."""

    return [
        {
            "convention": convention,
            "policy": policy,
            "horizon_years": horizon,
            "checkpoint": str(result.checkpoint_path.relative_to(artifact_root)),
            "checkpoint_sha256": _sha256(result.checkpoint_path),
            "optimizer_updates": result.optimizer_updates,
            "selected_epoch": result.selected_epoch,
            "finite_clipped_gradients": bool(
                result.clipped_gradient_norms
                and all(
                    np.isfinite(value) and value <= 0.20001
                    for value in result.clipped_gradient_norms
                )
            ),
            "baseline_reference": (
                str(result.baseline_reference_path.relative_to(artifact_root))
                if result.baseline_reference_path is not None
                else None
            ),
            "baseline_reference_identity": result.baseline_reference_identity,
        }
        for (convention, policy, horizon), result in sorted(results.items())
    ]


def _workflow_stage_artifacts(stage: str) -> tuple[str, ...]:
    """Name the evidence a reusable CLI stage must find in an atomic workflow."""

    checkpoints = tuple(
        f"{policy}_{horizon}y.pt"
        for policy in ("BM_E", "BM_C", "BM_D", "MM")
        for horizon in (5, 15)
    )
    artifacts = {
        "train": checkpoints,
        "resume": (*checkpoints, "BM_E_5y.recovery.pt", "BM_E_5y.interruption.json"),
        "evaluate": ("locked-evaluation.json", "mm-truncation.json"),
        "accept": ("acceptance-report.json",),
    }
    try:
        return artifacts[stage]
    except KeyError as error:
        raise OperationalRunError(f"Unknown reusable workflow stage: {stage}") from error


def _validate_reusable_workflow_manifest(
    manifest: object,
    *,
    configuration: ResolvedRunConfiguration,
    stage: str,
) -> None:
    """Reject reuse when the completed evidence does not match this exact run."""

    if not isinstance(manifest, dict):
        raise OperationalRunError("Reusable workflow manifest is not an object")
    _workflow_stage_artifacts(stage)
    if manifest.get("status") != RunStatus.COMPLETED.value:
        raise OperationalRunError("Reusable workflow did not complete successfully")
    workflow = manifest.get("local_workflow")
    if not isinstance(workflow, dict):
        raise OperationalRunError("Source bundle is not a complete local workflow")
    source_configuration = manifest.get("requested_configuration")
    if not isinstance(source_configuration, dict):
        raise OperationalRunError("Reusable workflow lacks its requested configuration")
    if _configuration_hash(source_configuration) != _configuration_hash(configuration.to_dict()):
        raise OperationalRunError(
            "Reusable workflow configuration differs in data, convention, seed, or semantics"
        )
    if stage == "accept":
        if manifest.get("acceptance_status") != AcceptanceStatus.PASSED.value:
            raise OperationalRunError("Reusable workflow was not development-validated")
        if workflow.get("development_validation_status") != "development-validated":
            raise OperationalRunError("Reusable workflow has no passed development validation")


def _validate_reusable_stage_evidence(
    manifest: Mapping[str, object],
    *,
    configuration: ResolvedRunConfiguration,
    artifact_directory: Path,
    stage: str,
) -> None:
    """Verify provenance, bytes, and minimum semantics for reused stage evidence."""

    expected_input_hashes, input_errors = _available_input_hashes(configuration)
    if input_errors or manifest.get("input_hashes") != expected_input_hashes:
        raise OperationalRunError("Reusable workflow input hashes no longer match")
    if manifest.get("git_revision") != _git_revision():
        raise OperationalRunError("Reusable workflow was produced by different code")
    if manifest.get("runtime") != _runtime_identity(configuration):
        raise OperationalRunError("Reusable workflow runtime identity differs")

    workflow = manifest["local_workflow"]
    assert isinstance(workflow, Mapping)  # Validated by _validate_reusable_workflow_manifest.
    hashes = workflow.get("stage_artifact_sha256")
    if not isinstance(hashes, Mapping):
        raise OperationalRunError("Reusable workflow lacks artifact checksums")
    for name in _workflow_stage_artifacts(stage):
        if hashes.get(name) != _sha256(artifact_directory / name):
            raise OperationalRunError(f"Reusable workflow artifact hash differs: {name}")

    if stage in {"train", "resume"}:
        _validate_reusable_checkpoints(
            artifact_directory,
            manifest=manifest,
            required=_workflow_stage_artifacts(stage),
        )
    if stage == "evaluate":
        evaluation = _read_json_artifact(artifact_directory / "locked-evaluation.json")
        expected_data = _manifest_data_identities(manifest)
        if (
            evaluation.get("kind") != "locked-final-test-evaluation"
            or evaluation.get("convention") != configuration.convention.profile
            or evaluation.get("data_identities") != expected_data
        ):
            raise OperationalRunError("Reusable locked evaluation has incompatible semantics")
        truncation = _read_json_artifact(artifact_directory / "mm-truncation.json")
        if (
            truncation.get("kind") != "mm-fifteen-year-truncation"
            or truncation.get("source_horizon_years") != 15
            or truncation.get("evaluation_horizon_years") != 5
            or truncation.get("convention") != configuration.convention.profile
            or truncation.get("data_identities") != expected_data
        ):
            raise OperationalRunError("Reusable MM truncation has incompatible semantics")
    if stage == "accept":
        acceptance = _read_json_artifact(artifact_directory / "acceptance-report.json")
        checks = acceptance.get("checks")
        if (
            acceptance.get("kind") != "local-workflow-acceptance"
            or acceptance.get("purpose") != "development-validation"
            or acceptance.get("status") != "passed"
            or not isinstance(checks, list)
            or not checks
            or any(
                not isinstance(check, dict)
                or check.get("status") != "passed"
                or not {"purpose", "applicability", "observation", "threshold", "evidence"}
                <= check.keys()
                for check in checks
            )
        ):
            raise OperationalRunError("Reusable acceptance report has incompatible semantics")


def _validate_reusable_checkpoints(
    artifact_directory: Path,
    *,
    manifest: Mapping[str, object],
    required: tuple[str, ...],
) -> None:
    """Check policy, horizon, code, data, and resolved configuration identities."""

    expected_data = _manifest_data_identities(manifest)
    for name in required:
        if not name.endswith(".pt") or name.endswith(".recovery.pt"):
            continue
        checkpoint = torch.load(
            artifact_directory / name, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint, dict):
            raise OperationalRunError(f"Reusable checkpoint is not an object: {name}")
        policy_tag, horizon_tag = name.removesuffix(".pt").rsplit("_", 1)
        expected_policy = {"BM_E": "BM^E", "BM_C": "BM^C", "BM_D": "BM^D", "MM": "MM"}[policy_tag]
        expected_horizon = int(horizon_tag.removesuffix("y"))
        if (
            checkpoint.get("policy") != expected_policy
            or checkpoint.get("horizon_years") != expected_horizon
            or checkpoint.get("configuration_identity")
            != manifest.get("resolved_configuration_hash")
            or checkpoint.get("code_identity", {}).get("git_revision")
            != manifest.get("git_revision")
            or checkpoint.get("data_identities") != expected_data
        ):
            raise OperationalRunError(f"Reusable checkpoint has incompatible semantics: {name}")


def _read_json_artifact(path: Path) -> dict[str, object]:
    """Read one stage JSON file, rejecting non-object documents."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OperationalRunError(f"Reusable stage artifact is not an object: {path.name}")
    return value


def _manifest_data_identities(manifest: Mapping[str, object]) -> dict[str, object]:
    """Get the three input identities shared by all reusable financial evidence."""

    input_hashes = manifest.get("input_hashes")
    preflight = manifest.get("market_preflight")
    reference_bank = manifest.get("reference_bank")
    if not all(
        isinstance(value, Mapping)
        for value in (input_hashes, preflight, reference_bank)
    ):
        raise OperationalRunError("Reusable workflow lacks data identity metadata")
    calibration = preflight.get("calibration")
    if not isinstance(calibration, Mapping):
        raise OperationalRunError("Reusable workflow lacks calibration identity")
    identities = {
        "market_source_hash": input_hashes.get("snb_csv"),
        "hjm_calibration_identity": calibration.get("identity"),
        "reference_bank_content_hash": reference_bank.get("content_hash"),
    }
    if not all(isinstance(value, str) for value in identities.values()):
        raise OperationalRunError("Reusable workflow has invalid data identities")
    return identities


def _workflow_replay_evidence(
    market_model: Any,
    *,
    historical: Any,
    calibration: Any,
    configuration: ResolvedRunConfiguration,
    snapshot_hash: str,
    replayed_snapshot_hash: str,
) -> dict[str, object]:
    """Replay CPU market prefixes independently of the training scenario streams."""

    markets: dict[str, dict[str, object]] = {}
    for horizon in configuration.experiment.horizons_years:
        arguments = {
            "convention": configuration.convention.profile,
            "horizon_years": horizon,
            "paths": 2,
            "seed": configuration.seeds["market_scenarios"],
            "split": "workflow-replay",
            "epoch": 0,
            "global_path_indices": (0, 1),
        }
        first = market_model.generate_hjm_scenarios(historical, calibration, **arguments)
        second = market_model.generate_hjm_scenarios(historical, calibration, **arguments)
        first_hash = _array_sha256(first.spot_rates)
        second_hash = _array_sha256(second.spot_rates)
        if first_hash != second_hash:
            raise OperationalRunError(f"CPU replay changed the {horizon}-year market path")
        markets[str(horizon)] = {
            "spot_rates_sha256": first_hash,
            "replay_spot_rates_sha256": second_hash,
            "matches": True,
            "split": "workflow-replay",
            "paths": 2,
        }
    return {
        "reference_bank_content_hash": snapshot_hash,
        "replayed_reference_bank_content_hash": replayed_snapshot_hash,
        "reference_bank_content_hash_matches": snapshot_hash == replayed_snapshot_hash,
        "markets": markets,
    }


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _workflow_training_jobs(
    results: Mapping[tuple[str, int], Any],
) -> list[dict[str, object]]:
    return [
        {
            "policy": policy,
            "horizon_years": horizon,
            "checkpoint": result.checkpoint_path.name,
            "checkpoint_sha256": _sha256(result.checkpoint_path),
            "selected_epoch": result.selected_epoch,
            "optimizer_updates": result.optimizer_updates,
            "finite_clipped_gradients": bool(
                result.clipped_gradient_norms
                and all(
                    np.isfinite(norm) and norm <= 0.20001
                    for norm in result.clipped_gradient_norms
                )
            ),
            "resource_profile": result.resource_profile.to_dict(),
            "baseline_reference": (
                result.baseline_reference_path.name
                if result.baseline_reference_path is not None
                else None
            ),
        }
        for (policy, horizon), result in sorted(results.items())
    ]


def _recovery_equivalence_evidence(
    *,
    resumed_checkpoint: Path,
    uninterrupted_checkpoint: Path,
    recovery: object,
) -> dict[str, object]:
    """Compare a resumed CPU trajectory with an independently uninterrupted one."""

    resumed = torch.load(resumed_checkpoint, map_location="cpu", weights_only=False)
    uninterrupted = torch.load(
        uninterrupted_checkpoint, map_location="cpu", weights_only=False
    )
    if not all(isinstance(item, dict) for item in (resumed, uninterrupted, recovery)):
        return {"status": "failed", "reason": "recovery evidence was unreadable"}
    assert isinstance(resumed, dict)
    assert isinstance(uninterrupted, dict)
    assert isinstance(recovery, dict)
    resumed_state = resumed.get("policy_state")
    uninterrupted_state = uninterrupted.get("policy_state")
    same_state = isinstance(resumed_state, dict) and isinstance(
        uninterrupted_state, dict
    ) and set(resumed_state) == set(uninterrupted_state) and all(
        isinstance(resumed_state[name], torch.Tensor)
        and isinstance(uninterrupted_state[name], torch.Tensor)
        and torch.equal(resumed_state[name], uninterrupted_state[name])
        for name in resumed_state
    )
    same_schedule = (
        resumed.get("selection_history") == uninterrupted.get("selection_history")
        and resumed.get("learning_rates") == uninterrupted.get("learning_rates")
        and resumed.get("scheduler_state") == uninterrupted.get("scheduler_state")
    )
    lineage = recovery.get("resume_lineage")
    recovery_identity = recovery.get("recovery_identity")
    completed_epoch = recovery.get("completed_epoch")
    recovery_is_complete = (
        isinstance(recovery_identity, dict)
        and isinstance(completed_epoch, int)
        and completed_epoch >= 2
    )
    return {
        "status": "passed"
        if same_state and same_schedule and recovery_is_complete
        else "failed",
        "resumed_checkpoint_sha256": _sha256(resumed_checkpoint),
        "uninterrupted_checkpoint_sha256": _sha256(uninterrupted_checkpoint),
        "same_policy_state": same_state,
        "same_selection_and_scheduler": same_schedule,
        "recovery_identity_present": isinstance(recovery_identity, dict),
        "recovered_completed_epoch": completed_epoch,
        "resume_lineage_entries": len(lineage) if isinstance(lineage, list) else 0,
        "resume_execution_context": (
            "same-artifact-location" if isinstance(lineage, list) and not lineage else "changed"
        ),
    }


def _financial_rollout_evidence(
    *, five_year: Any, fifteen_year: Any, truncation: Any
) -> dict[str, object]:
    """Persist hard financial, timing, constraint, and action-contract observations."""

    outcomes = {"MM(5y)": five_year, "MM(15y)": fifteen_year, "MM(15y|5y)": truncation}
    observations: dict[str, dict[str, object]] = {}
    passed = True
    for label, outcome in outcomes.items():
        horizon = 15 if label == "MM(15y)" else 5
        actions = outcome.treasury_actions
        constraints = outcome.constraint_values
        annual_mask = outcome.constraint_annual_mask
        finite = all(
            torch.isfinite(values).all().item()
            for values in (outcome.accounting_error, outcome.cash_reconciliation_error)
        ) and all(
            value is not None and torch.isfinite(value).all().item()
            for value in (actions, constraints, annual_mask)
        )
        accounting_maximum = float(outcome.accounting_error.detach().abs().max().cpu())
        cash_maximum = float(
            outcome.cash_reconciliation_error.detach().abs().max().cpu()
        )
        tolerance = torch.maximum(
            torch.full_like(outcome.assets, 1e-6), outcome.assets.detach().abs() * 1e-10
        )
        accounting_ratio = float(
            (outcome.accounting_error.detach().abs() / tolerance).max().cpu()
        )
        cash_ratio = float(
            (outcome.cash_reconciliation_error.detach().abs() / tolerance).max().cpu()
        )
        action_contract = (
            actions is not None
            and actions.ndim == 3
            and actions.shape[1] == horizon * 12
            and actions.shape[2] == 29
            and bool((actions >= 0).all().item())
        )
        expected_annual_closes = horizon - 1
        timing_contract = (
            annual_mask is not None
            and annual_mask.ndim == 2
            and int(annual_mask.sum().item())
            == len(annual_mask) * expected_annual_closes
        )
        observation_passed = (
            finite
            and accounting_ratio <= 1.0
            and cash_ratio <= 1.0
            and action_contract
            and timing_contract
        )
        observations[label] = {
            "finite": finite,
            "max_accounting_error": accounting_maximum,
            "max_cash_reconciliation_error": cash_maximum,
            "max_accounting_error_to_model_tolerance": accounting_ratio,
            "max_cash_error_to_model_tolerance": cash_ratio,
            "action_contract": action_contract,
            "annual_event_timing_contract": timing_contract,
            "expected_annual_closes_per_path": expected_annual_closes,
            "passed": observation_passed,
        }
        passed = passed and observation_passed
    return {
        "status": "passed" if passed else "failed",
        "thresholds": {
            "accounting_and_cash": "same scale-aware 1e-6 absolute / 1e-10 relative tolerance as ALMSimulator",
            "action_features": 29,
        },
        "outcomes": observations,
    }


def _local_workflow_acceptance(
    *,
    configuration: ResolvedRunConfiguration,
    completed_jobs: list[dict[str, object]],
    width_checks: tuple[Any, ...],
    replay: Mapping[str, object],
    one_step_statistics: Mapping[str, object],
    locked: Any,
    sensitivity: Any,
    horizon_analysis_available: bool,
    financial_rollout: Mapping[str, object],
    recovery: Mapping[str, object],
    report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Derive development validation solely from observed stage evidence."""

    plan = build_execution_plan(configuration)
    expected_updates = plan.primary_optimizer_updates
    actual_updates = sum(int(job["optimizer_updates"]) for job in completed_jobs)
    checks: list[dict[str, object]] = []

    def record(
        *,
        purpose: str,
        applicability: str,
        observation: object,
        threshold: object,
        passed: bool,
        evidence: str,
    ) -> None:
        checks.append(
            {
                "purpose": purpose,
                "applicability": applicability,
                "observation": observation,
                "threshold": threshold,
                "status": "passed" if passed else "failed",
                "evidence": evidence,
            }
        )

    record(
        purpose="CPU replay",
        applicability="mandatory",
        observation=replay.get("reference_bank_content_hash_matches"),
        threshold=True,
        passed=replay.get("reference_bank_content_hash_matches") is True
        and all(
            item.get("matches") is True
            for item in dict(replay.get("markets", {})).values()
            if isinstance(item, dict)
        ),
        evidence="reference-bank.json and workflow-replay market hashes",
    )
    record(
        purpose="one-step HJM statistics",
        applicability="mandatory CPU validation",
        observation={
            "factor_means": one_step_statistics.get("factor_means"),
            "relative_covariance_error": one_step_statistics.get(
                "relative_covariance_error"
            ),
        },
        threshold={
            "factor_mean_abs_limit": one_step_statistics.get("factor_mean_abs_limit"),
            "relative_covariance_error_limit": one_step_statistics.get(
                "relative_covariance_error_limit"
            ),
        },
        passed=one_step_statistics.get("status") == "passed",
        evidence="50,000-path one-step fitted-PCA covariance diagnostic",
    )
    record(
        purpose="primary policy/horizon matrix",
        applicability="mandatory",
        observation={"jobs": len(completed_jobs), "optimizer_updates": actual_updates},
        threshold={
            "jobs": plan.primary_training_jobs,
            "optimizer_updates": expected_updates,
        },
        passed=(
            len(completed_jobs) == plan.primary_training_jobs
            and actual_updates == expected_updates
        ),
        evidence="selected checkpoints and resource profiles",
    )
    record(
        purpose="finite gradients and financial rollout",
        applicability="mandatory",
        observation={
            "finite_clipped_gradients": [
                job["finite_clipped_gradients"] for job in completed_jobs
            ],
            "financial_rollout": financial_rollout,
        },
        threshold=(
            "every completed training job has finite clipped gradients and each "
            "MM rollout passes accounting, cash, timing, constraint, and action checks"
        ),
        passed=bool(completed_jobs)
        and all(job["finite_clipped_gradients"] is True for job in completed_jobs)
        and financial_rollout.get("status") == "passed",
        evidence="training checkpoints, resource profiles, and frozen MM rollout diagnostics",
    )
    record(
        purpose="epoch-boundary recovery equivalence",
        applicability="mandatory",
        observation=recovery,
        threshold=(
            "resumed policy state, selection history, learning rates, and scheduler "
            "state equal the uninterrupted CPU oracle with a compatible epoch-two "
            "recovery artifact"
        ),
        passed=recovery.get("status") == "passed",
        evidence="BM_E_5y.recovery.pt and isolated uninterrupted CPU checkpoint",
    )
    record(
        purpose="paper-width MM checks",
        applicability="mandatory",
        observation={
            "checks": len(width_checks),
            "optimizer_updates": sum(item.optimizer_updates for item in width_checks),
        },
        threshold={"checks": 2, "optimizer_updates": 2},
        passed=(
            len(width_checks) == 2
            and all(
                item.optimizer_updates == 1
                and item.finite
                and item.updated
                and item.clipped
                for item in width_checks
            )
        ),
        evidence="MM_*y.paper-width-check.json",
    )
    record(
        purpose="locked evaluation and paired bootstrap",
        applicability="mandatory",
        observation={
            "reports": len(locked.reports),
            "paired_intervals": len(locked.paired_intervals),
        },
        threshold={"reports": plan.primary_training_jobs, "paired_intervals": 1},
        passed=(
            len(locked.reports) == plan.primary_training_jobs
            and bool(locked.paired_intervals)
        ),
        evidence="locked-evaluation.json",
    )
    record(
        purpose="representative frozen-policy sensitivity",
        applicability="mandatory",
        observation={
            "training_triggered": sensitivity.training_triggered,
            "variant_hash": sensitivity.variant.content_hash,
        },
        threshold={"training_triggered": False},
        passed=sensitivity.training_triggered is False,
        evidence="reference-bank-sensitivity.json",
    )
    record(
        purpose="horizon and scenario analysis",
        applicability="mandatory descriptive analysis",
        observation={"complete_category_and_bootstrap_analysis": horizon_analysis_available},
        threshold=True,
        passed=horizon_analysis_available,
        evidence="horizon-scenario-analysis.json",
    )
    if report is not None:
        record(
            purpose="compact local report",
            applicability="mandatory",
            observation=report.get("kind"),
            threshold="compact-no-swap-local-report",
            passed=report.get("kind") == "compact-no-swap-local-report",
            evidence="compact-no-swap-report.json",
        )
    return {
        "format_version": 1,
        "kind": "local-workflow-acceptance",
        "purpose": "development-validation",
        "status": "passed" if all(check["status"] == "passed" for check in checks) else "failed",
        "checks": checks,
        "non_gating_observations": (
            "loss reduction, economic return, constraint violation rates, strategy "
            "ordering, and plot similarity are reported but do not determine this status"
        ),
    }


def _horizon_analysis_evidence(
    analysis: Any,
    *,
    configuration: ResolvedRunConfiguration,
    historical: Any,
    calibration: Any,
    snapshot_hash: str,
    unavailable_reason: str | None,
) -> dict[str, object]:
    if analysis is None:
        return {
            "format_version": 1,
            "kind": "horizon-scenario-analysis",
            "convention": configuration.convention.profile,
            "data_identities": {
                "market_source_hash": historical.source_hash,
                "hjm_calibration_identity": calibration.calibration_identity,
                "reference_bank_content_hash": snapshot_hash,
            },
            "status": "unavailable",
            "reason": unavailable_reason,
            "metrics_by_policy": {},
            "paired_intervals": [],
            "category_outputs": [],
        }
    return {
        "format_version": 1,
        "kind": "horizon-scenario-analysis",
        "convention": configuration.convention.profile,
        "data_identities": {
            "market_source_hash": historical.source_hash,
            "hjm_calibration_identity": calibration.calibration_identity,
            "reference_bank_content_hash": snapshot_hash,
        },
        "metrics_by_policy": {
            label: {
                "source_horizon_years": metric.source_horizon_years,
                "analysis_window_months": metric.analysis_window_months,
                "paths": metric.paths,
                "units": metric.units,
                "normalized_action_turnover": metric.normalized_action_turnover.tolist(),
                "terminal_turnover_concentration": metric.terminal_turnover_concentration.tolist(),
                "terminal_concentration_status": metric.terminal_concentration_status,
                "terminal_concentration_reason": metric.terminal_concentration_reason,
            }
            for label, metric in analysis.metrics_by_policy.items()
        },
        "paired_intervals": [asdict(interval) for interval in analysis.paired_intervals],
        "category_outputs": [asdict(item) for item in analysis.category_outputs],
        "interpretation": "small-sample descriptive local workflow demonstration",
    }


def _unavailable_device_checks() -> dict[str, dict[str, object]]:
    """State non-executed accelerators explicitly; Ticket 27 owns their full checks."""

    return {
        "mps": {
            "status": "not-run",
            "reason": (
                "single-device portability commissioning is a separate bounded "
                "stage; this workflow uses its resolved execution device"
            ),
            "runtime_available": bool(torch.backends.mps.is_available()),
        },
        "cuda": {
            "status": "not-run",
            "reason": (
                "CUDA and multi-GPU validation require the bank-side commissioning "
                "environment and are not inferred locally"
            ),
            "runtime_available": bool(torch.cuda.is_available()),
        },
    }


def _write_json_artifact(path: Path, contents: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(contents, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _workflow_failure_bundle(
    configuration: ResolvedRunConfiguration,
    *,
    staging_directory: Path | None,
    staged_configuration: ResolvedRunConfiguration | None,
    error: Exception,
    status: RunStatus,
) -> RunBundle:
    """Preserve staged diagnostics while ensuring they never look completed."""

    if staging_directory is None or not staging_directory.is_dir():
        failure_directory = _write_failure_bundle(configuration, error, status=status)
    else:
        effective_configuration = staged_configuration or configuration
        manifest = _build_manifest(
            effective_configuration, status, AcceptanceStatus.PENDING
        )
        manifest.update(
            {
                "error": str(error),
                "diagnostics": _diagnostics_for(error),
                "local_workflow": {
                    "development_validation_status": "incomplete"
                    if status is RunStatus.INCOMPLETE
                    else "failed",
                    "generated_artifacts": sorted(
                        path.name for path in staging_directory.iterdir() if path.is_file()
                    ),
                },
            }
        )
        _write_json_artifact(staging_directory / "manifest.json", manifest)
        suffix = staging_directory.name.rsplit("-", 1)[-1]
        failure_directory = configuration.output.directory / (
            f"{configuration.output.run_name}.{status.value}-{suffix}"
        )
        try:
            os.replace(staging_directory, failure_directory)
        except OSError:
            failure_directory = None
    return RunBundle(
        status=status,
        acceptance_status=AcceptanceStatus.PENDING,
        artifact_directory=failure_directory,
        error=str(error),
        artifacts=(failure_directory / "manifest.json",)
        if failure_directory is not None
        else (),
    )


def _paired_pilot_failure_bundle(
    configuration: ResolvedRunConfiguration,
    *,
    staging_directory: Path | None,
    staged_configuration: ResolvedRunConfiguration | None,
    error: Exception,
    status: RunStatus,
) -> RunBundle:
    """Preserve pilot-stage diagnostics without labeling a partial matrix complete."""

    if staging_directory is None or not staging_directory.is_dir():
        failure_directory = _write_failure_bundle(configuration, error, status=status)
    else:
        effective_configuration = staged_configuration or configuration
        manifest = _build_manifest(
            effective_configuration, status, AcceptanceStatus.PENDING
        )
        manifest.update(
            {
                "requested_configuration": configuration.to_dict(),
                "error": str(error),
                "diagnostics": _diagnostics_for(error),
                "paired_convention_pilot": {
                    "label": "paired-convention-research-pilot",
                    "status": "incomplete"
                    if status is RunStatus.INCOMPLETE
                    else "failed",
                    "generated_artifacts": sorted(
                        str(path.relative_to(staging_directory))
                        for path in staging_directory.rglob("*")
                        if path.is_file()
                    ),
                },
            }
        )
        _write_json_artifact(staging_directory / "manifest.json", manifest)
        suffix = staging_directory.name.rsplit("-", 1)[-1]
        failure_directory = configuration.output.directory / (
            f"{configuration.output.run_name}.{status.value}-{suffix}"
        )
        try:
            os.replace(staging_directory, failure_directory)
        except OSError:
            failure_directory = None
    return RunBundle(
        status=status,
        acceptance_status=AcceptanceStatus.PENDING,
        artifact_directory=failure_directory,
        error=str(error),
        artifacts=(failure_directory / "manifest.json",)
        if failure_directory is not None
        else (),
    )


def _build_manifest(
    configuration: ResolvedRunConfiguration,
    status: RunStatus,
    acceptance_status: AcceptanceStatus,
) -> dict[str, object]:
    return {
        **_manifest_metadata(configuration),
        "status": status.value,
        "acceptance_status": acceptance_status.value,
        "input_hashes": {
            "snb_csv": _sha256(configuration.source_data.snb_csv),
            "paper_pdf": _sha256(configuration.source_data.paper_pdf),
        },
        "execution_plan": build_execution_plan(configuration).to_dict(),
    }


def _write_bundle_atomically(
    configuration: ResolvedRunConfiguration,
    manifest: dict[str, object],
    *,
    extra_artifacts: Mapping[str, Mapping[str, object]] | None = None,
) -> Path:
    output_root = configuration.output.directory
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"Output directory is not a directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    final_directory = output_root / configuration.output.run_name
    if final_directory.exists():
        raise ValueError(f"Run artifact directory already exists: {final_directory}")

    staging_directory = Path(
        tempfile.mkdtemp(prefix=f".{configuration.output.run_name}-", dir=output_root)
    )
    try:
        (staging_directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for name, contents in (extra_artifacts or {}).items():
            artifact_path = staging_directory / name
            if artifact_path.parent != staging_directory or artifact_path.name != name:
                raise ValueError(f"Artifact name must be a file name: {name}")
            artifact_path.write_text(
                json.dumps(contents, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        os.replace(staging_directory, final_directory)
    except OSError:
        shutil.rmtree(staging_directory, ignore_errors=True)
        raise
    return final_directory


def _write_failure_bundle(
    configuration: ResolvedRunConfiguration,
    error: Exception,
    *,
    status: RunStatus = RunStatus.FAILED,
) -> Path | None:
    """Best-effort persistence for failures that occur before a completed bundle exists."""

    output_root = configuration.output.directory
    staging_directory: Path | None = None
    try:
        if output_root.exists() and not output_root.is_dir():
            return None
        output_root.mkdir(parents=True, exist_ok=True)
        staging_directory = Path(
            tempfile.mkdtemp(
                prefix=f".{configuration.output.run_name}.failed-", dir=output_root
            )
        )
        input_hashes, input_hash_errors = _available_input_hashes(configuration)
        failure_manifest = {
            **_manifest_metadata(configuration),
            "status": status.value,
            "acceptance_status": AcceptanceStatus.PENDING.value,
            "error": str(error),
            "diagnostics": _diagnostics_for(error),
            "input_hashes": input_hashes,
            "input_hash_errors": input_hash_errors,
        }
        (staging_directory / "manifest.json").write_text(
            json.dumps(failure_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        failure_directory = output_root / (
            f"{configuration.output.run_name}.failed-{staging_directory.name.rsplit('-', 1)[-1]}"
        )
        os.replace(staging_directory, failure_directory)
        return failure_directory
    except OSError:
        if staging_directory is not None:
            shutil.rmtree(staging_directory, ignore_errors=True)
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _available_input_hashes(
    configuration: ResolvedRunConfiguration,
) -> tuple[dict[str, str | None], dict[str, str]]:
    hashes: dict[str, str | None] = {}
    errors: dict[str, str] = {}
    for name, path in {
        "snb_csv": configuration.source_data.snb_csv,
        "paper_pdf": configuration.source_data.paper_pdf,
    }.items():
        try:
            hashes[name] = _sha256(path)
        except OSError as error:
            hashes[name] = None
            errors[name] = str(error)
    return hashes, errors


def _runtime_identity(configuration: ResolvedRunConfiguration) -> dict[str, object]:
    dependencies = {
        package: importlib.metadata.version(distribution)
        for package, distribution in {
            "matplotlib": "matplotlib",
            "numpy": "numpy",
            "pandas": "pandas",
            "pyyaml": "PyYAML",
            "scikit-learn": "scikit-learn",
            "scipy": "scipy",
            "torch": "torch",
            "tqdm": "tqdm",
        }.items()
    }
    return {
        "python": sys.version,
        "dependencies": dependencies,
        "device": configuration.optimization.device,
        "dtype": configuration.optimization.dtype,
    }


def _git_revision() -> str:
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _manifest_metadata(configuration: ResolvedRunConfiguration) -> dict[str, object]:
    resolved_configuration = configuration.to_dict()
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "runtime": _runtime_identity(configuration),
        "seed_registry": configuration.seeds,
        "resolved_configuration": resolved_configuration,
        "resolved_configuration_hash": _configuration_hash(resolved_configuration),
    }


def _configuration_hash(resolved_configuration: dict[str, object]) -> str:
    serialized = json.dumps(
        resolved_configuration, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _status_for(acceptance_status: AcceptanceStatus) -> RunStatus:
    if acceptance_status is AcceptanceStatus.FAILED:
        return RunStatus.ACCEPTANCE_FAILED
    return RunStatus.COMPLETED


def _diagnostics_for(error: Exception) -> dict[str, object]:
    if isinstance(error, OperationalRunError):
        return error.diagnostics
    if isinstance(error, BudgetExceeded):
        return error.diagnostics
    diagnostics = getattr(error, "diagnostics", None)
    if isinstance(diagnostics, dict):
        return diagnostics
    return {}
