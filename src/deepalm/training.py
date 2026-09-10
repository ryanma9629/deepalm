"""Bounded local training for no-swap treasury benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from deepalm.baselines import FrozenDateBenchmarkReference
from deepalm.config import (
    ArchitectureConfiguration,
    OptimizationConfiguration,
    OutputConfiguration,
    ResolvedRunConfiguration,
    RunScaleConfiguration,
)
from deepalm.mm import (
    CurveFeaturePCA,
    MMObservationError,
    MMPolicy,
    RegisteredTrainingCurves,
    TruncatedMMPolicy,
)
from deepalm.objective import (
    ObjectiveParameters,
    evaluation_objective_parameters,
    sample_training_objective_parameters,
)
from deepalm.policies import (
    BMConstantPolicy,
    BMDatePolicy,
    BMEqualPolicy,
    TreasuryPolicy,
)
from deepalm.reference_bank import ReferenceBankSnapshot
from deepalm.resources import BudgetExceeded, ResourceMonitor, ResourceSnapshot
from deepalm.runoff import ALMSimulator, PassiveRunoffResult
from deepalm.semantics import (
    FINANCIAL_SEMANTICS_VERSION,
    artifact_semantics,
    artifact_semantics_error,
    current_financial_semantics,
)
from deepalm.term_structures import (
    HistoricalTermStructures,
    HjmPcaCalibration,
    MarketScenarioBatch,
    MarketScenarioModel,
)

_BASE_LEARNING_RATE = 5e-4
_MAX_LEARNING_RATE = 5e-3
_GRADIENT_CLIP_NORM = 0.2
_SELECTION_PROTOCOL_VERSION = 2
_POLICY_TYPES: dict[str, type[TreasuryPolicy]] = {
    "BM^E": BMEqualPolicy,
    "BM^C": BMConstantPolicy,
    "BM^D": BMDatePolicy,
}


class TrainingError(RuntimeError):
    """Raised when a bounded treasury benchmark job cannot complete safely."""

    def __init__(
        self, message: str, *, diagnostics: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class TrainingInterrupted(TrainingError):
    """A recoverable training interruption recorded at a safe epoch boundary."""

    def __init__(
        self,
        reason: str,
        *,
        recovery_path: Path,
        interruption_path: Path,
        diagnostics: dict[str, object],
    ) -> None:
        detail = f": {diagnostics['error']}" if diagnostics.get("error") else ""
        super().__init__(
            f"Training interrupted: {reason}{detail}",
            diagnostics={"reason": reason, **diagnostics},
        )
        self.reason = reason
        self.recovery_path = recovery_path
        self.interruption_path = interruption_path


@dataclass(frozen=True)
class TrainingControl:
    """Optional public controls for a recoverable benchmark-training invocation.

    A requested stop is honored only after a complete epoch has persisted its
    training state.  ``resume_from`` permits an output-directory override while
    retaining a separately stored recovery artifact.
    """

    resume: bool = False
    resume_from: Path | None = None
    stop_after_completed_epoch: int | None = None

    def __post_init__(self) -> None:
        if self.resume_from is not None and not self.resume:
            raise ValueError("resume_from requires resume=True")
        if (
            self.stop_after_completed_epoch is not None
            and self.stop_after_completed_epoch < 0
        ):
            raise ValueError("stop_after_completed_epoch must be non-negative")


@dataclass(frozen=True)
class SelectionSchedule:
    """Selection cadence and early-stopping settings for a run-scale profile."""

    first_selection_epoch: int
    patience: int | None
    minimum_relative_improvement: float

    @classmethod
    def for_profile(cls, profile: str) -> SelectionSchedule:
        if profile == "paper_scale":
            return cls(
                first_selection_epoch=20,
                patience=15,
                minimum_relative_improvement=0.001,
            )
        return cls(
            first_selection_epoch=1, patience=None, minimum_relative_improvement=0.0
        )

    @classmethod
    def from_run_scale(cls, scale: RunScaleConfiguration) -> SelectionSchedule:
        """Use resolved profile defaults or bank-training's explicit controls."""

        if scale.profile != "bank_training":
            return cls.for_profile(scale.profile)
        return cls(
            first_selection_epoch=scale.selection_start_epoch,
            patience=scale.early_stopping_patience,
            minimum_relative_improvement=scale.minimum_relative_improvement,
        )


@dataclass
class SelectionTracker:
    """Select the best checkpoint independently of significant loss improvement.

    A smaller validation total (or a lower penalty on an exact total tie) saves
    a checkpoint. Patience resets only when total loss beats its separate
    reference by more than the configured relative delta. Patience counts
    validation checks, not the distance between their epoch numbers.
    """

    schedule: SelectionSchedule
    best_selection: tuple[float, float] | None = None
    last_improvement_epoch: int | None = None
    stopping_reference_total: float | None = None
    checks_without_improvement: int = 0

    def should_select(self, epoch: int) -> bool:
        return epoch >= self.schedule.first_selection_epoch

    def record(self, epoch: int, candidate: tuple[float, float]) -> bool:
        if not self.should_select(epoch):
            return False
        if not all(math.isfinite(value) for value in candidate):
            raise TrainingError("Selection total and penalty loss must be finite")
        save_best = self.best_selection is None or candidate < self.best_selection
        if save_best:
            self.best_selection = candidate

        reference = self.stopping_reference_total
        improved = reference is None or candidate[0] < reference - (
            self.schedule.minimum_relative_improvement * max(abs(reference), 1e-12)
        )
        if improved:
            self.stopping_reference_total = candidate[0]
            self.last_improvement_epoch = epoch
            self.checks_without_improvement = 0
        else:
            self.checks_without_improvement += 1
        return save_best

    def should_stop(self, epoch: int) -> bool:
        return (
            self.schedule.patience is not None
            and self.last_improvement_epoch is not None
            and self.should_select(epoch)
            and self.checks_without_improvement >= self.schedule.patience
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "best_selection": list(self.best_selection)
            if self.best_selection is not None
            else None,
            "last_improvement_epoch": self.last_improvement_epoch,
            "stopping_reference_total": self.stopping_reference_total,
            "checks_without_improvement": self.checks_without_improvement,
        }


@dataclass(frozen=True)
class SelectionRecord:
    """One fixed-selection measurement made after an epoch."""

    epoch: int
    total_loss: float
    penalty_loss: float
    action_record: SelectionActionRecord


@dataclass(frozen=True)
class TrainingProgress:
    """One completed training epoch made available to an optional observer."""

    policy_name: str
    horizon_years: int
    epoch: int
    epochs: int
    optimizer_updates: int
    average_training_loss: float
    selection: SelectionRecord | None


TrainingProgressCallback = Callable[[TrainingProgress], None]


@dataclass(frozen=True)
class SelectionActionRecord:
    """Aggregated no-swap actions observed on one fixed selection evaluation."""

    investment_maturity_totals: tuple[float, ...]
    funding_maturity_totals: tuple[float, ...]
    minimum_action: float
    maximum_action: float


@dataclass(frozen=True)
class ResourceProfile:
    """Timing and memory evidence for one real policy/horizon training job."""

    horizon_years: int
    device: str
    warmup_seconds: float
    scenario_seconds: float
    forward_backward_update_seconds: float
    selection_seconds: float
    artifact_seconds: float
    peak_process_rss_bytes: int | None
    peak_accelerator_allocated_bytes: int | None
    accelerator_memory_status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkTrainingResult:
    """Selected benchmark policy and the evidence produced by local training."""

    policy: TreasuryPolicy
    checkpoint_path: Path
    selected_epoch: int
    selection_history: tuple[SelectionRecord, ...]
    optimizer_updates: int
    learning_rates: tuple[float, ...]
    clipped_gradient_norms: tuple[float, ...]
    resource_profile: ResourceProfile
    baseline_reference_path: Path | None = None
    baseline_reference_identity: str | None = None


@dataclass(frozen=True)
class DeviceValidationRecord:
    """A compact real-update validation result for one requested local device."""

    device: str
    status: str
    finite: bool
    updated: bool
    clipped: bool
    checkpoint_path: Path | None
    checkpoint_loaded: bool = False
    resumed_from_cpu_recovery: bool = False
    recovery_path: Path | None = None
    reason: str | None = None


@dataclass(frozen=True)
class MMWidthCheckResult:
    """Evidence from the isolated two-path paper-width MM optimizer update."""

    artifact_path: Path
    optimizer_updates: int
    paths: int
    finite: bool
    updated: bool
    clipped: bool
    baseline_reference_identity: str
    architecture: ArchitectureConfiguration
    resource_profile: ResourceProfile


@dataclass(frozen=True)
class MMTruncationResult:
    """Evaluation evidence for MM(15y|5y), with no optimizer invocation."""

    policy: TruncatedMMPolicy
    market: MarketScenarioBatch
    outcome: PassiveRunoffResult
    optimizer_updates: int


class BenchmarkTrainer:
    """Train, select, restore, and document one no-swap benchmark/horizon job."""

    def __init__(
        self,
        configuration: ResolvedRunConfiguration,
        *,
        snapshot: ReferenceBankSnapshot,
        historical: HistoricalTermStructures,
        calibration: HjmPcaCalibration,
        policy_name: str,
        simulator: ALMSimulator | None = None,
        market_model: MarketScenarioModel | None = None,
        resource_monitor: ResourceMonitor | None = None,
    ) -> None:
        if policy_name not in _POLICY_TYPES and policy_name != "MM":
            raise TrainingError(f"Unsupported benchmark policy: {policy_name}")
        self._configuration = configuration
        self._snapshot = snapshot
        self._historical = historical
        self._calibration = calibration
        self._simulator = simulator or ALMSimulator()
        self._market_model = market_model or MarketScenarioModel()
        self._device = torch.device(configuration.optimization.device)
        self._dtype = _torch_dtype(configuration.optimization.dtype)
        self._monitor = resource_monitor or ResourceMonitor(
            wall_clock_budget_seconds=configuration.resources.wall_clock_budget_seconds,
            process_rss_limit_bytes=configuration.resources.process_rss_limit_bytes,
            accelerator_memory_limit_bytes=(
                configuration.resources.accelerator_memory_limit_bytes
            ),
            device=configuration.optimization.device,
        )
        self._policy_name = policy_name

    def fit(
        self,
        *,
        horizon_years: int,
        control: TrainingControl | None = None,
        progress_callback: TrainingProgressCallback | None = None,
    ) -> BenchmarkTrainingResult:
        """Run, or safely resume, one benchmark-training job."""

        resolved_control = control or TrainingControl()
        recovery_path = resolved_control.resume_from or self._recovery_path(
            horizon_years
        )
        interruption_path = self._interruption_path(horizon_years)
        try:
            return self._fit(
                horizon_years=horizon_years,
                control=resolved_control,
                recovery_path=recovery_path,
                interruption_path=interruption_path,
                progress_callback=progress_callback,
            )
        except BudgetExceeded as error:
            self._record_interruption(
                interruption_path,
                reason="budget_exhausted",
                recovery_path=recovery_path,
                diagnostics=error.diagnostics,
            )
            raise TrainingInterrupted(
                "budget_exhausted",
                recovery_path=recovery_path,
                interruption_path=interruption_path,
                diagnostics=error.diagnostics,
            ) from error
        except MemoryError as error:
            diagnostics = {
                "error": str(error),
                "resource_snapshot": self._monitor.snapshot().to_dict(),
            }
            self._record_interruption(
                interruption_path,
                reason="out_of_memory",
                recovery_path=recovery_path,
                diagnostics=diagnostics,
            )
            raise TrainingInterrupted(
                "out_of_memory",
                recovery_path=recovery_path,
                interruption_path=interruption_path,
                diagnostics=diagnostics,
            ) from error
        except KeyboardInterrupt as error:
            diagnostics = {"resource_snapshot": self._monitor.snapshot().to_dict()}
            self._record_interruption(
                interruption_path,
                reason="cancelled",
                recovery_path=recovery_path,
                diagnostics=diagnostics,
            )
            raise TrainingInterrupted(
                "cancelled",
                recovery_path=recovery_path,
                interruption_path=interruption_path,
                diagnostics=diagnostics,
            ) from error
        except TrainingInterrupted:
            raise
        except TrainingError as error:
            diagnostics = {
                "error_type": type(error).__name__,
                "error": str(error),
                "training_diagnostics": error.diagnostics,
                "resource_snapshot": self._monitor.snapshot().to_dict(),
            }
            self._record_interruption(
                interruption_path,
                reason="operational_failure",
                recovery_path=recovery_path,
                diagnostics=diagnostics,
            )
            raise TrainingInterrupted(
                "operational_failure",
                recovery_path=recovery_path,
                interruption_path=interruption_path,
                diagnostics=diagnostics,
            ) from error
        except (OSError, RuntimeError) as error:
            reason = (
                "out_of_memory"
                if "out of memory" in str(error).lower()
                else "operational_failure"
            )
            diagnostics = {
                "error_type": type(error).__name__,
                "error": str(error),
                "training_diagnostics": getattr(error, "diagnostics", {}),
                "resource_snapshot": self._monitor.snapshot().to_dict(),
            }
            self._record_interruption(
                interruption_path,
                reason=reason,
                recovery_path=recovery_path,
                diagnostics=diagnostics,
            )
            raise TrainingInterrupted(
                reason,
                recovery_path=recovery_path,
                interruption_path=interruption_path,
                diagnostics=diagnostics,
            ) from error

    def _fit(
        self,
        *,
        horizon_years: int,
        control: TrainingControl,
        recovery_path: Path,
        interruption_path: Path,
        progress_callback: TrainingProgressCallback | None,
    ) -> BenchmarkTrainingResult:
        """Run all configured local updates and reload the best selection state."""

        if horizon_years not in self._configuration.experiment.horizons_years:
            raise TrainingError(
                "Requested horizon is not enabled by the resolved experiment"
            )
        if self._policy_name not in self._configuration.policy.names:
            raise TrainingError(
                f"Resolved policy configuration does not include {self._policy_name}"
            )

        scale = self._configuration.run_scale
        updates_per_epoch = _ceil_division(
            scale.training_paths_per_epoch, scale.batch_size
        )
        torch.manual_seed(
            _derived_seed(
                self._configuration.seeds["model_initialization"],
                self._policy_name,
                horizon_years,
            )
        )
        policy = self._create_policy(horizon_years=horizon_years)
        policy_dependencies = self._policy_dependencies(
            horizon_years=horizon_years, policy=policy
        )
        optimizer = torch.optim.RAdam(
            policy.parameters(), lr=_BASE_LEARNING_RATE, weight_decay=0.0
        )
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=_BASE_LEARNING_RATE,
            max_lr=_MAX_LEARNING_RATE,
            step_size_up=2 * updates_per_epoch,
            step_size_down=2 * updates_per_epoch,
            cycle_momentum=False,
        )
        timing = _TimingAccumulator()
        policy_tag = _policy_tag(self._policy_name)
        peaks: list[ResourceSnapshot] = [
            self._monitor.check(f"before-{policy_tag}-warmup")
        ]
        timing.warmup_seconds = _warmup_device(self._device)
        peaks.append(self._monitor.check(f"after-{policy_tag}-warmup"))

        best: tuple[float, float] | None = None
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        history: list[SelectionRecord] = []
        learning_rates: list[float] = []
        clipped_gradient_norms: list[float] = []
        updates = 0
        selection_tracker = SelectionTracker(SelectionSchedule.from_run_scale(scale))
        resume_lineage: list[dict[str, object]] = []
        start_epoch = 1
        if control.resume and recovery_path.is_file():
            recovered = _load_recovery(recovery_path)
            _validate_recovery_compatibility(
                recovered,
                expected=_recovery_identity(
                    configuration=self._configuration,
                    policy_name=self._policy_name,
                    horizon_years=horizon_years,
                    historical=self._historical,
                    calibration=self._calibration,
                    snapshot=self._snapshot,
                    policy_dependencies=policy_dependencies,
                ),
            )
            _validate_resource_override(recovered, configuration=self._configuration)
            policy.load_state_dict(recovered["policy_state"])
            optimizer.load_state_dict(recovered["optimizer_state"])
            scheduler.load_state_dict(recovered["scheduler_state"])
            recovered_best = recovered["best"]
            best = tuple(recovered_best) if recovered_best is not None else None
            best_epoch = int(recovered["best_epoch"])
            best_state = recovered["best_state"]
            history = [
                _selection_record_from_dict(record)
                for record in recovered["selection_history"]
            ]
            learning_rates = list(recovered["learning_rates"])
            clipped_gradient_norms = list(recovered["clipped_gradient_norms"])
            updates = int(recovered["optimizer_updates"])
            selection_tracker = _selection_tracker_from_dict(
                recovered["selection_tracker"], schedule=selection_tracker.schedule
            )
            resume_lineage = _resume_lineage(
                recovered,
                configuration=self._configuration,
                recovery_path=recovery_path,
            )
            start_epoch = int(recovered["completed_epoch"]) + 1
            if (
                selection_tracker.should_stop(start_epoch - 1)
                or start_epoch > scale.epochs
            ):
                # No later epoch will save the parent/override audit for this resume.
                _save_recovery(
                    recovery_path,
                    {
                        **recovered,
                        "execution_context": _execution_context(self._configuration),
                        "resume_lineage": resume_lineage,
                        "resource_snapshot": self._monitor.snapshot().to_dict(),
                    },
                )

        if control.stop_after_completed_epoch == 0:
            self._raise_controlled_interruption(
                recovery_path,
                interruption_path,
                completed_epoch=0,
            )

        completed_epoch = start_epoch - 1
        for epoch in range(start_epoch, scale.epochs + 1):
            # A recovery saved on the stopping epoch must not perform an extra update.
            if selection_tracker.should_stop(completed_epoch):
                break
            epoch_losses: list[float] = []
            for start in range(0, scale.training_paths_per_epoch, scale.batch_size):
                paths = min(scale.batch_size, scale.training_paths_per_epoch - start)
                market = self._training_market(
                    horizon_years=horizon_years,
                    epoch=epoch,
                    start=start,
                    paths=paths,
                    timing=timing,
                )
                parameters = _training_objective_parameters(
                    paths,
                    seed=_derived_seed(
                        _job_seed(
                            self._configuration.seeds["objective_parameters"],
                            self._policy_name,
                            horizon_years,
                        ),
                        epoch,
                        start,
                    ),
                    device=self._device,
                    dtype=self._dtype,
                )
                before = [
                    parameter.detach().clone() for parameter in policy.parameters()
                ]
                update_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                learning_rates.append(float(optimizer.param_groups[0]["lr"]))
                outcome = self._simulator.rollout(
                    self._snapshot,
                    market,
                    policy=policy,
                    device=self._device,
                    dtype=self._dtype,
                    include_loan_dynamics=True,
                    include_deposit_dynamics=True,
                    objective_parameters=parameters,
                )
                if outcome.objective is None:
                    raise TrainingError(
                        f"{self._policy_name} rollout did not return the training objective"
                    )
                loss = outcome.objective.total.mean()
                if not torch.isfinite(loss):
                    raise TrainingError(
                        f"{self._policy_name} training loss is non-finite",
                        diagnostics={
                            "horizon_years": horizon_years,
                            "epoch": epoch,
                            "batch_start": start,
                            "paths": paths,
                            "loss": float(loss.detach().cpu()),
                        },
                    )
                epoch_losses.append(float(loss.detach().cpu()))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), max_norm=_GRADIENT_CLIP_NORM
                )
                clipped_norm = _gradient_norm(policy)
                if (
                    not torch.isfinite(clipped_norm)
                    or clipped_norm.item() > _GRADIENT_CLIP_NORM + 1e-5
                ):
                    raise TrainingError(
                        f"{self._policy_name} gradient clipping did not produce a finite bound",
                        diagnostics={
                            "horizon_years": horizon_years,
                            "epoch": epoch,
                            "batch_start": start,
                            "paths": paths,
                            "clipped_gradient_norm": float(clipped_norm.detach().cpu()),
                        },
                    )
                clipped_gradient_norms.append(float(clipped_norm.detach().cpu()))
                optimizer.step()
                scheduler.step()
                _synchronize_device(self._device)
                timing.forward_backward_update_seconds += (
                    time.perf_counter() - update_started
                )
                updated = any(
                    not torch.equal(old, new.detach())
                    for old, new in zip(before, policy.parameters(), strict=True)
                )
                if clipped_norm.item() > 0.0 and not updated:
                    raise TrainingError(
                        f"{self._policy_name} optimizer step did not update policy parameters",
                        diagnostics={
                            "horizon_years": horizon_years,
                            "epoch": epoch,
                            "batch_start": start,
                            "paths": paths,
                            "clipped_gradient_norm": float(clipped_norm.detach().cpu()),
                            "learning_rate": learning_rates[-1],
                        },
                    )
                updates += 1
                peaks.append(
                    self._monitor.check(
                        f"after-{_policy_tag(self._policy_name)}-{horizon_years}y-update-{updates}"
                    )
                )

            selection: SelectionRecord | None = None
            if selection_tracker.should_select(epoch):
                selection = self._select(
                    policy,
                    epoch=epoch,
                    horizon_years=horizon_years,
                    timing=timing,
                )
                history.append(selection)
                candidate = (selection.total_loss, selection.penalty_loss)
                if selection_tracker.record(epoch, candidate):
                    best = candidate
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in policy.state_dict().items()
                    }
            completed_epoch = epoch
            _save_recovery(
                recovery_path,
                _recovery_contents(
                    configuration=self._configuration,
                    policy=policy,
                    policy_name=self._policy_name,
                    horizon_years=horizon_years,
                    completed_epoch=epoch,
                    best=best,
                    best_epoch=best_epoch,
                    best_state=best_state,
                    selection_history=history,
                    optimizer_updates=updates,
                    learning_rates=learning_rates,
                    clipped_gradient_norms=clipped_gradient_norms,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    historical=self._historical,
                    calibration=self._calibration,
                    snapshot=self._snapshot,
                    resource_snapshot=self._monitor.snapshot(),
                    selection_tracker=selection_tracker,
                    resume_lineage=resume_lineage,
                    policy_dependencies=policy_dependencies,
                ),
            )
            if progress_callback is not None:
                progress = TrainingProgress(
                    policy_name=self._policy_name,
                    horizon_years=horizon_years,
                    epoch=epoch,
                    epochs=scale.epochs,
                    optimizer_updates=updates,
                    average_training_loss=sum(epoch_losses) / len(epoch_losses),
                    selection=selection,
                )
                progress_callback(progress)
            if control.stop_after_completed_epoch == epoch:
                self._raise_controlled_interruption(
                    recovery_path,
                    interruption_path,
                    completed_epoch=epoch,
                )
            peaks.append(
                self._monitor.check(
                    f"after-{_policy_tag(self._policy_name)}-{horizon_years}y-selection-{epoch}"
                )
            )
            if selection_tracker.should_stop(epoch):
                break

        if best_state is None or best is None:
            raise TrainingError(
                f"{self._policy_name} training did not produce a selectable checkpoint"
            )
        policy.load_state_dict(best_state)
        profile = _resource_profile(
            horizon_years=horizon_years,
            device=str(self._device),
            timing=timing,
            snapshots=peaks,
        )
        checkpoint_path = self._checkpoint_path(horizon_years)
        artifact_started = time.perf_counter()
        _save_checkpoint(
            checkpoint_path,
            _checkpoint_contents(
                configuration=self._configuration,
                policy=policy,
                policy_name=self._policy_name,
                horizon_years=horizon_years,
                selected_epoch=best_epoch,
                selection_history=history,
                optimizer_updates=updates,
                learning_rates=learning_rates,
                clipped_gradient_norms=clipped_gradient_norms,
                optimizer=optimizer,
                scheduler=scheduler,
                resource_profile=profile,
                historical=self._historical,
                calibration=self._calibration,
                snapshot=self._snapshot,
                resource_profile_path=self._resource_profile_path(horizon_years),
                policy_dependencies=policy_dependencies,
                training_summary={
                    "selection_protocol_version": _SELECTION_PROTOCOL_VERSION,
                    "monitor": "selection_total_loss",
                    "mode": "min",
                    "completed_epoch": completed_epoch,
                    "configured_max_epochs": scale.epochs,
                    "selected_epoch": best_epoch,
                    "best_validation_loss": best[0],
                    "last_validation_loss": history[-1].total_loss,
                    "stop_reason": (
                        "patience_exhausted"
                        if selection_tracker.should_stop(completed_epoch)
                        else "max_epochs"
                    ),
                    "checks_without_improvement": selection_tracker.checks_without_improvement,
                    "early_stopping": asdict(selection_tracker.schedule),
                    "last_training_state_path": str(recovery_path),
                },
            ),
        )
        timing.artifact_seconds += time.perf_counter() - artifact_started
        profile = _resource_profile(
            horizon_years=horizon_years,
            device=str(self._device),
            timing=timing,
            snapshots=peaks,
        )
        profile_started = time.perf_counter()
        _save_resource_profile(self._resource_profile_path(horizon_years), profile)
        timing.artifact_seconds += time.perf_counter() - profile_started
        profile = _resource_profile(
            horizon_years=horizon_years,
            device=str(self._device),
            timing=timing,
            snapshots=peaks,
        )
        return BenchmarkTrainingResult(
            policy=policy,
            checkpoint_path=checkpoint_path,
            selected_epoch=best_epoch,
            selection_history=tuple(history),
            optimizer_updates=updates,
            learning_rates=tuple(learning_rates),
            clipped_gradient_norms=tuple(clipped_gradient_norms),
            resource_profile=profile,
        )

    def _create_policy(self, *, horizon_years: int) -> TreasuryPolicy:
        """Create the policy whose state is owned by this trainer.

        Subclasses with immutable dependencies (such as MM's frozen BM^D and
        curve transform) override this narrow seam without duplicating the
        training, selection, recovery, or artifact lifecycle.
        """

        return _build_policy(
            self._policy_name,
            horizon_years=horizon_years,
            device=self._device,
            dtype=self._dtype,
        )

    def _policy_dependencies(
        self, *, horizon_years: int, policy: TreasuryPolicy
    ) -> dict[str, object]:
        """Return immutable, checkpoint-bound dependencies for this policy."""

        del horizon_years, policy
        return {}

    def fit_all(self) -> tuple[BenchmarkTrainingResult, ...]:
        """Train the configured benchmark once for every resolved horizon."""

        return tuple(
            self.fit(horizon_years=horizon_years)
            for horizon_years in self._configuration.experiment.horizons_years
        )

    def validate_devices(
        self, *, horizon_years: int
    ) -> tuple[DeviceValidationRecord, ...]:
        """Resume a CPU recovery artifact on each available single device."""

        return _validate_device_portability(
            self._configuration,
            horizon_years=horizon_years,
            create_trainer=lambda configuration: BenchmarkTrainer(
                configuration,
                snapshot=self._snapshot,
                historical=self._historical,
                calibration=self._calibration,
                policy_name=self._policy_name,
                simulator=self._simulator,
                market_model=self._market_model,
            ),
        )

    def load_selected_checkpoint(self, checkpoint_path: Path) -> TreasuryPolicy:
        """Load a CPU-readable selected benchmark checkpoint onto this device."""

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise TrainingError(
                f"Could not load benchmark checkpoint: {checkpoint_path}"
            ) from error
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("policy") != self._policy_name
            or checkpoint.get("horizon_years") not in {5, 15}
        ):
            raise TrainingError("Benchmark checkpoint is incompatible")
        _validate_selected_checkpoint_contract(
            checkpoint,
            configuration=self._configuration,
            historical=self._historical,
            calibration=self._calibration,
            snapshot=self._snapshot,
            policy_label="Benchmark",
        )
        policy = self._create_policy(horizon_years=int(checkpoint["horizon_years"]))
        try:
            policy.load_state_dict(checkpoint["policy_state"])
        except (KeyError, RuntimeError) as error:
            raise TrainingError("Benchmark checkpoint policy state is incompatible") from error
        policy.eval()
        return policy

    def _training_market(
        self,
        *,
        horizon_years: int,
        epoch: int,
        start: int,
        paths: int,
        timing: _TimingAccumulator,
    ) -> MarketScenarioBatch:
        started = time.perf_counter()
        market = self._market_model.generate_hjm_scenarios(
            self._historical,
            self._calibration,
            horizon_years=horizon_years,
            paths=paths,
            seed=_job_seed(
                self._configuration.seeds["market_scenarios"],
                self._policy_name,
                horizon_years,
            ),
            split="training",
            epoch=epoch,
            global_path_indices=tuple(range(start, start + paths)),
        )
        timing.scenario_seconds += time.perf_counter() - started
        return market

    def _select(
        self,
        policy: TreasuryPolicy,
        *,
        epoch: int,
        horizon_years: int,
        timing: _TimingAccumulator,
    ) -> SelectionRecord:
        total_losses: list[torch.Tensor] = []
        penalty_losses: list[torch.Tensor] = []
        investment_maturity_totals = torch.zeros(13, dtype=torch.float64)
        funding_maturity_totals = torch.zeros(16, dtype=torch.float64)
        minimum_action = float("inf")
        maximum_action = float("-inf")
        with torch.no_grad():
            for start in range(
                0,
                self._configuration.run_scale.selection_paths,
                self._configuration.run_scale.batch_size,
            ):
                paths = min(
                    self._configuration.run_scale.batch_size,
                    self._configuration.run_scale.selection_paths - start,
                )
                scenario_started = time.perf_counter()
                market = self._market_model.generate_hjm_scenarios(
                    self._historical,
                    self._calibration,
                    horizon_years=horizon_years,
                    paths=paths,
                    seed=_job_seed(
                        self._configuration.seeds["market_scenarios"],
                        self._policy_name,
                        horizon_years,
                    ),
                    split="selection",
                    epoch=0,
                    global_path_indices=tuple(range(start, start + paths)),
                )
                timing.scenario_seconds += time.perf_counter() - scenario_started
                evaluation_started = time.perf_counter()
                outcome = self._simulator.rollout(
                    self._snapshot,
                    market,
                    policy=policy,
                    device=self._device,
                    dtype=self._dtype,
                    include_loan_dynamics=True,
                    include_deposit_dynamics=True,
                    objective_parameters=evaluation_objective_parameters(
                        paths,
                        horizon_years=horizon_years,
                        device=self._device,
                        dtype=self._dtype,
                    ),
                )
                if outcome.objective is None:
                    raise TrainingError(
                        f"{self._policy_name} selection rollout did not return an objective"
                    )
                if outcome.treasury_actions is None:
                    raise TrainingError(
                        f"{self._policy_name} selection rollout did not record actions"
                    )
                total_losses.append(outcome.objective.total.detach().cpu())
                penalty_losses.append(outcome.objective.penalty.detach().cpu())
                actions = outcome.treasury_actions.detach().cpu()
                investment_maturity_totals += actions[:, :, :13].sum(dim=(0, 1))
                funding_maturity_totals += actions[:, :, 13:].sum(dim=(0, 1))
                minimum_action = min(minimum_action, float(actions.min()))
                maximum_action = max(maximum_action, float(actions.max()))
                _synchronize_device(self._device)
                timing.selection_seconds += time.perf_counter() - evaluation_started
        return SelectionRecord(
            epoch=epoch,
            total_loss=float(torch.cat(total_losses).mean()),
            penalty_loss=float(torch.cat(penalty_losses).mean()),
            action_record=SelectionActionRecord(
                investment_maturity_totals=tuple(investment_maturity_totals.tolist()),
                funding_maturity_totals=tuple(funding_maturity_totals.tolist()),
                minimum_action=minimum_action,
                maximum_action=maximum_action,
            ),
        )

    def _checkpoint_path(self, horizon_years: int) -> Path:
        return (
            self._configuration.output.directory
            / self._configuration.output.run_name
            / f"{_policy_tag(self._policy_name)}_{horizon_years}y.pt"
        )

    def _resource_profile_path(self, horizon_years: int) -> Path:
        return self._checkpoint_path(horizon_years).with_suffix(".profile.json")

    def _recovery_path(self, horizon_years: int) -> Path:
        return self._checkpoint_path(horizon_years).with_suffix(".recovery.pt")

    def _interruption_path(self, horizon_years: int) -> Path:
        return self._checkpoint_path(horizon_years).with_suffix(".interruption.json")

    def _record_interruption(
        self,
        interruption_path: Path,
        *,
        reason: str,
        recovery_path: Path,
        diagnostics: dict[str, object],
    ) -> None:
        _save_json_artifact(
            interruption_path,
            {
                "format_version": 1,
                "status": "incomplete",
                "reason": reason,
                "recovery_path": recovery_path.name,
                "replay_from": (
                    "last_completed_epoch"
                    if recovery_path.is_file()
                    else "registered_initial_state"
                ),
                "resource_snapshot": self._monitor.snapshot().to_dict(),
                "diagnostics": diagnostics,
            },
        )

    def _raise_controlled_interruption(
        self,
        recovery_path: Path,
        interruption_path: Path,
        *,
        completed_epoch: int,
    ) -> None:
        diagnostics: dict[str, object] = {
            "completed_epoch": completed_epoch,
            "resource_snapshot": self._monitor.snapshot().to_dict(),
        }
        self._record_interruption(
            interruption_path,
            reason="cancelled",
            recovery_path=recovery_path,
            diagnostics=diagnostics,
        )
        raise TrainingInterrupted(
            "cancelled",
            recovery_path=recovery_path,
            interruption_path=interruption_path,
            diagnostics=diagnostics,
        )


class BMETrainer(BenchmarkTrainer):
    """BM^E adapter for the shared benchmark-training interface."""

    def __init__(
        self,
        configuration: ResolvedRunConfiguration,
        *,
        snapshot: ReferenceBankSnapshot,
        historical: HistoricalTermStructures,
        calibration: HjmPcaCalibration,
        simulator: ALMSimulator | None = None,
        market_model: MarketScenarioModel | None = None,
        resource_monitor: ResourceMonitor | None = None,
    ) -> None:
        super().__init__(
            configuration,
            snapshot=snapshot,
            historical=historical,
            calibration=calibration,
            policy_name="BM^E",
            simulator=simulator,
            market_model=market_model,
            resource_monitor=resource_monitor,
        )


class BMConstantTrainer(BenchmarkTrainer):
    """BM^C adapter for the shared benchmark-training interface."""

    def __init__(
        self,
        configuration: ResolvedRunConfiguration,
        *,
        snapshot: ReferenceBankSnapshot,
        historical: HistoricalTermStructures,
        calibration: HjmPcaCalibration,
        simulator: ALMSimulator | None = None,
        market_model: MarketScenarioModel | None = None,
        resource_monitor: ResourceMonitor | None = None,
    ) -> None:
        super().__init__(
            configuration,
            snapshot=snapshot,
            historical=historical,
            calibration=calibration,
            policy_name="BM^C",
            simulator=simulator,
            market_model=market_model,
            resource_monitor=resource_monitor,
        )


class BMDateTrainer(BenchmarkTrainer):
    """BM^D adapter that freezes each selected local baseline by content identity."""

    def __init__(
        self,
        configuration: ResolvedRunConfiguration,
        *,
        snapshot: ReferenceBankSnapshot,
        historical: HistoricalTermStructures,
        calibration: HjmPcaCalibration,
        simulator: ALMSimulator | None = None,
        market_model: MarketScenarioModel | None = None,
        resource_monitor: ResourceMonitor | None = None,
    ) -> None:
        super().__init__(
            configuration,
            snapshot=snapshot,
            historical=historical,
            calibration=calibration,
            policy_name="BM^D",
            simulator=simulator,
            market_model=market_model,
            resource_monitor=resource_monitor,
        )

    def fit(
        self,
        *,
        horizon_years: int,
        control: TrainingControl | None = None,
        progress_callback: TrainingProgressCallback | None = None,
    ) -> BenchmarkTrainingResult:
        result = super().fit(
            horizon_years=horizon_years,
            control=control,
            progress_callback=progress_callback,
        )
        reference = FrozenDateBenchmarkReference.freeze(result.checkpoint_path)
        return replace(
            result,
            baseline_reference_path=reference.reference_path,
            baseline_reference_identity=reference.reference_identity,
        )


class MMTrainer(BenchmarkTrainer):
    """Time-shared MM training around one frozen same-horizon BM^D reference.

    The shared :class:`BenchmarkTrainer` remains responsible for the actual
    multi-period optimization.  This adapter owns only the MM-specific
    immutable inputs: a selected BM^D reference and a PCA transform fitted on
    registered training curves.
    """

    def __init__(
        self,
        configuration: ResolvedRunConfiguration,
        *,
        snapshot: ReferenceBankSnapshot,
        historical: HistoricalTermStructures,
        calibration: HjmPcaCalibration,
        baseline_reference: FrozenDateBenchmarkReference,
        simulator: ALMSimulator | None = None,
        market_model: MarketScenarioModel | None = None,
        resource_monitor: ResourceMonitor | None = None,
    ) -> None:
        if baseline_reference.horizon_years not in {5, 15}:
            raise TrainingError(
                "MM requires a frozen five- or fifteen-year BM^D baseline"
            )
        expected_baseline_data = {
            "market_source_hash": historical.source_hash,
            "hjm_calibration_identity": calibration.calibration_identity,
            "reference_bank_content_hash": snapshot.content_hash,
        }
        if dict(baseline_reference.data_identities) != expected_baseline_data:
            raise TrainingError("Frozen MM baseline was not selected for these inputs")
        super().__init__(
            configuration,
            snapshot=snapshot,
            historical=historical,
            calibration=calibration,
            policy_name="MM",
            simulator=simulator,
            market_model=market_model,
            resource_monitor=resource_monitor,
        )
        self._baseline_reference = baseline_reference
        self._curve_features: CurveFeaturePCA | None = None

    def fit(
        self,
        *,
        horizon_years: int,
        control: TrainingControl | None = None,
        progress_callback: TrainingProgressCallback | None = None,
    ) -> BenchmarkTrainingResult:
        self._require_mm_horizon(horizon_years)
        self._materialize_baseline_reference(horizon_years)
        result = super().fit(
            horizon_years=horizon_years,
            control=control,
            progress_callback=progress_callback,
        )
        return replace(
            result,
            baseline_reference_path=self._baseline_reference.reference_path,
            baseline_reference_identity=self._baseline_reference.reference_identity,
        )

    def validate_devices(
        self, *, horizon_years: int
    ) -> tuple[DeviceValidationRecord, ...]:
        """Validate MM updates on available local devices with its frozen inputs."""

        self._require_mm_horizon(horizon_years)
        return _validate_device_portability(
            self._configuration,
            horizon_years=horizon_years,
            create_trainer=lambda configuration: MMTrainer(
                configuration,
                snapshot=self._snapshot,
                historical=self._historical,
                calibration=self._calibration,
                baseline_reference=self._baseline_reference,
                simulator=self._simulator,
                market_model=self._market_model,
            ),
        )

    def load_selected_checkpoint(self, checkpoint_path: Path) -> MMPolicy:
        """Reload a selected MM only when its frozen inputs still match exactly."""

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise TrainingError(
                f"Could not load MM checkpoint: {checkpoint_path}"
            ) from error
        if not isinstance(checkpoint, dict) or checkpoint.get("policy") != "MM":
            raise TrainingError("Checkpoint is not an MM policy")
        self._validate_selected_checkpoint_configuration(checkpoint)
        horizon_years = checkpoint.get("horizon_years")
        if horizon_years != self._baseline_reference.horizon_years:
            raise TrainingError("MM checkpoint horizon must match its frozen baseline")
        self._require_mm_horizon(int(horizon_years))
        dependencies = checkpoint.get("policy_dependencies")
        expected = self._policy_dependencies(
            horizon_years=int(horizon_years),
            policy=self._create_policy(horizon_years=int(horizon_years)),
        )
        if dependencies != expected:
            raise TrainingError("MM checkpoint immutable dependency identity mismatch")
        try:
            restored_features = CurveFeaturePCA.from_dict(
                dependencies["curve_feature_pca"],
                expected_data_identity=self._historical.source_hash,
                expected_calibration_identity=self._calibration.calibration_identity,
                expected_training_state_identity=self._curve_feature_pca(
                    horizon_years=int(horizon_years)
                ).training_state_identity,
            )
        except (KeyError, MMObservationError) as error:
            raise TrainingError(
                "MM checkpoint curve preprocessing is incompatible"
            ) from error
        policy = MMPolicy(
            reference=self._baseline_reference,
            curve_features=restored_features,
            architecture=self._configuration.architecture,
            device=self._device,
            dtype=self._dtype,
        )
        try:
            policy.load_state_dict(checkpoint["policy_state"])
        except (KeyError, RuntimeError) as error:
            raise TrainingError("MM checkpoint policy state is incompatible") from error
        policy.eval()
        return policy

    def _validate_selected_checkpoint_configuration(
        self, checkpoint: dict[str, object]
    ) -> None:
        """Reject semantic checkpoint changes before rebuilding MM dependencies."""

        _validate_selected_checkpoint_contract(
            checkpoint,
            configuration=self._configuration,
            historical=self._historical,
            calibration=self._calibration,
            snapshot=self._snapshot,
            policy_label="MM",
        )

    @classmethod
    def load_policy_from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> MMPolicy:
        """Restore an evaluable MM directly from its self-contained run folder."""

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise TrainingError(
                f"Could not load MM checkpoint: {checkpoint_path}"
            ) from error
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("policy") != "MM"
            or checkpoint.get("horizon_years") not in {5, 15}
        ):
            raise TrainingError("Checkpoint is not a supported MM policy")
        if not current_financial_semantics(checkpoint):
            raise TrainingError("MM checkpoint financial semantics are incompatible; retraining required")
        if error := artifact_semantics_error(checkpoint.get("code_identity"), "training"):
            raise TrainingError(error)
        dependencies = checkpoint.get("policy_dependencies")
        if not isinstance(dependencies, dict):
            raise TrainingError("MM checkpoint has no immutable dependency record")
        try:
            reference_filename = dependencies["baseline_reference_filename"]
            reference = FrozenDateBenchmarkReference.load(
                checkpoint_path.parent / str(reference_filename),
                expected_reference_identity=str(
                    dependencies["baseline_reference_identity"]
                ),
            )
            if (
                reference.checkpoint_sha256
                != dependencies["baseline_checkpoint_sha256"]
            ):
                raise TrainingError(
                    "MM checkpoint baseline checkpoint identity mismatch"
                )
            data_identities = checkpoint["data_identities"]
            pca = CurveFeaturePCA.from_dict(
                dependencies["curve_feature_pca"],
                expected_data_identity=str(data_identities["market_source_hash"]),
                expected_calibration_identity=str(
                    data_identities["hjm_calibration_identity"]
                ),
                expected_training_state_identity=str(
                    dependencies["curve_feature_pca"]["training_state_identity"]
                ),
            )
            architecture_data = checkpoint["configuration"]["architecture"]
            architecture = ArchitectureConfiguration(
                profile=str(architecture_data["profile"]),
                widths=tuple(int(width) for width in architecture_data["widths"]),
                encoder_features=int(architecture_data["encoder_features"]),
                observation_features=int(architecture_data["observation_features"]),
                final_encoding_features=int(
                    architecture_data["final_encoding_features"]
                ),
                action_features=int(architecture_data["action_features"]),
            )
        except (KeyError, TypeError, ValueError, MMObservationError) as error:
            raise TrainingError(
                "MM checkpoint dependency record is incompatible"
            ) from error
        policy = MMPolicy(
            reference=reference,
            curve_features=pca,
            architecture=architecture,
            device=device,
            dtype=dtype,
        )
        try:
            policy.load_state_dict(checkpoint["policy_state"])
        except (KeyError, RuntimeError) as error:
            raise TrainingError("MM checkpoint policy state is incompatible") from error
        policy.eval()
        return policy

    def evaluate_five_year_truncation(
        self,
        *,
        checkpoint_path: Path,
        market: MarketScenarioBatch,
    ) -> MMTruncationResult:
        """Evaluate MM(15y|5y) without optimizing or renormalizing time.

        The market prefix determines the 60-step balance-sheet rollout and its
        five-year terminal objective.  ``TruncatedMMPolicy`` separately keeps
        the trained policy's 180-step state horizon, so every decision uses
        ``t / 15 years`` and the frozen 15-year BM^D baseline.
        """

        self._require_mm_horizon(15)
        if market.horizon_years != 15:
            raise TrainingError("MM(15y|5y) requires a fifteen-year source market")
        policy = self.load_selected_checkpoint(checkpoint_path)
        if policy.baseline.transitions != 180:
            raise TrainingError("MM(15y|5y) requires a selected fifteen-year policy")
        prefix = market.prefix(horizon_years=5)
        truncated_policy = TruncatedMMPolicy(policy)
        outcome = self._simulator.rollout(
            self._snapshot,
            prefix,
            policy=truncated_policy,
            device=self._device,
            dtype=self._dtype,
            include_loan_dynamics=True,
            include_deposit_dynamics=True,
            objective_parameters=evaluation_objective_parameters(
                len(prefix.spot_rates),
                horizon_years=5,
                device=self._device,
                dtype=self._dtype,
            ),
        )
        if (
            outcome.objective is None
            or outcome.treasury_actions is None
            or outcome.treasury_actions.shape[1] != 60
        ):
            raise TrainingError("MM(15y|5y) did not produce a complete 60-step rollout")
        if not torch.isfinite(outcome.treasury_actions).all():
            raise TrainingError("MM(15y|5y) produced non-finite actions")
        return MMTruncationResult(
            policy=truncated_policy,
            market=prefix,
            outcome=outcome,
            optimizer_updates=0,
        )

    def run_paper_width_check(self, *, horizon_years: int) -> MMWidthCheckResult:
        """Run one isolated two-path full-horizon update at paper width.

        This is deliberately an engineering feasibility check, not a competing
        model-selection run.  It writes a separate JSON evidence artifact and
        never mutates the compact selected checkpoint.
        """

        self._require_mm_horizon(horizon_years)
        self._materialize_baseline_reference(horizon_years)
        paper_architecture = ArchitectureConfiguration(
            profile="paper", widths=(512, 512, 256, 128)
        )
        torch.manual_seed(
            _derived_seed(
                self._configuration.seeds["model_initialization"],
                "MM",
                horizon_years,
                "paper-width-check",
            )
        )
        policy = MMPolicy(
            reference=self._baseline_reference,
            curve_features=self._curve_feature_pca(horizon_years),
            architecture=paper_architecture,
            device=self._device,
            dtype=self._dtype,
        )
        trainable_parameters = [
            parameter for parameter in policy.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.RAdam(
            trainable_parameters, lr=_BASE_LEARNING_RATE, weight_decay=0.0
        )
        timing = _TimingAccumulator()
        peaks = [self._monitor.check("before-MM-paper-width-warmup")]
        timing.warmup_seconds = _warmup_device(self._device)
        peaks.append(self._monitor.check("after-MM-paper-width-warmup"))
        paths = 2
        market = self._training_market(
            horizon_years=horizon_years,
            epoch=1,
            start=0,
            paths=paths,
            timing=timing,
        )
        # This is a feasibility probe, not a training draw. Use a fixed upper-range
        # target so the probe exercises backward/update even when low sampled targets
        # would make the corrected balance-sheet path produce exactly zero loss.
        parameters = ObjectiveParameters(
            mu=torch.full((paths,), 0.07, device=self._device, dtype=self._dtype),
            penalty_weight=torch.full(
                (paths,), 3.5, device=self._device, dtype=self._dtype
            ),
        )
        before = [parameter.detach().clone() for parameter in trainable_parameters]
        update_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outcome = self._simulator.rollout(
            self._snapshot,
            market,
            policy=policy,
            device=self._device,
            dtype=self._dtype,
            include_loan_dynamics=True,
            include_deposit_dynamics=True,
            objective_parameters=parameters,
        )
        if outcome.objective is None:
            raise TrainingError("MM paper-width rollout did not return an objective")
        loss = outcome.objective.total.mean()
        finite = bool(torch.isfinite(loss).item())
        if not finite:
            raise TrainingError("MM paper-width loss is non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters, max_norm=_GRADIENT_CLIP_NORM
        )
        clipped_norm = _gradient_norm(policy)
        clipped = bool(
            torch.isfinite(clipped_norm).item()
            and clipped_norm.item() <= _GRADIENT_CLIP_NORM + 1e-5
        )
        if not clipped:
            raise TrainingError(
                "MM paper-width gradient clipping did not produce a finite bound"
            )
        optimizer.step()
        _synchronize_device(self._device)
        timing.forward_backward_update_seconds += time.perf_counter() - update_started
        updated = any(
            not torch.equal(old, new.detach())
            for old, new in zip(before, trainable_parameters, strict=True)
        )
        if not updated:
            raise TrainingError(
                "MM paper-width optimizer step did not update parameters"
            )
        peaks.append(self._monitor.check("after-MM-paper-width-update-1"))
        profile = _resource_profile(
            horizon_years=horizon_years,
            device=str(self._device),
            timing=timing,
            snapshots=peaks,
        )
        artifact_path = self._checkpoint_path(horizon_years).with_suffix(
            ".paper-width-check.json"
        )
        artifact_contents = {
            "format_version": 1,
            "kind": "MM-paper-width-check",
            "policy": "MM",
            "horizon_years": horizon_years,
            "optimizer_updates": 1,
            "paths": paths,
            "finite": finite,
            "updated": updated,
            "clipped": clipped,
            "gradient_clip_norm": _GRADIENT_CLIP_NORM,
            "clipped_gradient_norm": float(clipped_norm.detach().cpu()),
            "architecture": asdict(paper_architecture),
            "policy_dependencies": self._policy_dependencies(
                horizon_years=horizon_years, policy=policy
            ),
        }
        artifact_started = time.perf_counter()
        _save_json_artifact(
            artifact_path,
            {**artifact_contents, "resource_profile": profile.to_dict()},
        )
        timing.artifact_seconds += time.perf_counter() - artifact_started
        profile = _resource_profile(
            horizon_years=horizon_years,
            device=str(self._device),
            timing=timing,
            snapshots=peaks,
        )
        _save_json_artifact(
            artifact_path, {**artifact_contents, "resource_profile": profile.to_dict()}
        )
        return MMWidthCheckResult(
            artifact_path=artifact_path,
            optimizer_updates=1,
            paths=paths,
            finite=finite,
            updated=updated,
            clipped=clipped,
            baseline_reference_identity=self._baseline_reference.reference_identity,
            architecture=paper_architecture,
            resource_profile=profile,
        )

    def _create_policy(self, *, horizon_years: int) -> MMPolicy:
        self._require_mm_horizon(horizon_years)
        return MMPolicy(
            reference=self._baseline_reference,
            curve_features=self._curve_feature_pca(horizon_years),
            architecture=self._configuration.architecture,
            device=self._device,
            dtype=self._dtype,
        )

    def _policy_dependencies(
        self, *, horizon_years: int, policy: TreasuryPolicy
    ) -> dict[str, object]:
        del policy
        features = self._curve_feature_pca(horizon_years)
        return {
            "baseline_reference_identity": self._baseline_reference.reference_identity,
            "baseline_checkpoint_sha256": self._baseline_reference.checkpoint_sha256,
            "baseline_horizon_years": self._baseline_reference.horizon_years,
            "baseline_reference_filename": self._baseline_reference.reference_path.name,
            "curve_feature_pca": features.to_dict(),
            "preprocessing_scenario": {
                "split": "training",
                "epoch": 0,
                "global_path_indices": (
                    f"0..{self._configuration.run_scale.training_paths_per_epoch - 1}"
                ),
                "job_seed": _job_seed(
                    self._configuration.seeds["market_scenarios"], "MM", horizon_years
                ),
            },
        }

    def _curve_feature_pca(self, horizon_years: int) -> CurveFeaturePCA:
        self._require_mm_horizon(horizon_years)
        if self._curve_features is None:
            paths = self._configuration.run_scale.training_paths_per_epoch
            market = self._market_model.generate_hjm_scenarios(
                self._historical,
                self._calibration,
                horizon_years=horizon_years,
                paths=paths,
                seed=_job_seed(
                    self._configuration.seeds["market_scenarios"], "MM", horizon_years
                ),
                split="training",
                epoch=0,
                global_path_indices=tuple(range(paths)),
            )
            registered = RegisteredTrainingCurves(
                curves=torch.tensor(market.spot_rates, dtype=torch.float64),
                data_identity=self._historical.source_hash,
                calibration_identity=self._calibration.calibration_identity,
            )
            self._curve_features = CurveFeaturePCA.fit(registered)
        return self._curve_features

    def _require_mm_horizon(self, horizon_years: int) -> None:
        if horizon_years not in {5, 15}:
            raise TrainingError(
                "MM implementation is limited to five- and fifteen-year horizons"
            )
        if horizon_years != self._baseline_reference.horizon_years:
            raise TrainingError("MM horizon must match its frozen BM^D baseline")
        if self._configuration.architecture.profile != "compact":
            raise TrainingError(
                "MM selection training must use the compact architecture"
            )

    def _materialize_baseline_reference(self, horizon_years: int) -> None:
        """Copy the content-addressed BM^D pair beside an MM run checkpoint."""

        destination_directory = self._checkpoint_path(horizon_years).parent
        destination_reference = (
            destination_directory / self._baseline_reference.reference_path.name
        )
        destination_checkpoint = (
            destination_directory / self._baseline_reference.checkpoint_path.name
        )
        _copy_identity_checked_artifact(
            self._baseline_reference.checkpoint_path, destination_checkpoint
        )
        _copy_identity_checked_artifact(
            self._baseline_reference.reference_path, destination_reference
        )
        try:
            materialized = FrozenDateBenchmarkReference.load(
                destination_reference,
                expected_reference_identity=self._baseline_reference.reference_identity,
            )
        except Exception as error:
            raise TrainingError(
                "Could not materialize frozen MM baseline reference"
            ) from error
        if materialized.checkpoint_sha256 != self._baseline_reference.checkpoint_sha256:
            raise TrainingError("Materialized MM baseline checkpoint identity mismatch")


@dataclass
class _TimingAccumulator:
    warmup_seconds: float = 0.0
    scenario_seconds: float = 0.0
    forward_backward_update_seconds: float = 0.0
    selection_seconds: float = 0.0
    artifact_seconds: float = 0.0


def _training_objective_parameters(
    paths: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ObjectiveParameters:
    sampled = sample_training_objective_parameters(
        paths,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=dtype,
    )
    return ObjectiveParameters(
        mu=sampled.mu.to(device=device),
        penalty_weight=sampled.penalty_weight.to(device=device),
    )


def _checkpoint_contents(
    *,
    configuration: ResolvedRunConfiguration,
    policy: TreasuryPolicy,
    policy_name: str,
    horizon_years: int,
    selected_epoch: int,
    selection_history: list[SelectionRecord],
    optimizer_updates: int,
    learning_rates: list[float],
    clipped_gradient_norms: list[float],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    resource_profile: ResourceProfile,
    historical: HistoricalTermStructures,
    calibration: HjmPcaCalibration,
    snapshot: ReferenceBankSnapshot,
    resource_profile_path: Path,
    policy_dependencies: dict[str, object],
    training_summary: dict[str, object],
) -> dict[str, object]:
    configuration_data = configuration.to_dict()
    return {
        "format_version": 1,
        "policy": policy_name,
        "horizon_years": horizon_years,
        "selected_epoch": selected_epoch,
        "selection_history": [asdict(record) for record in selection_history],
        "training_summary": training_summary,
        "optimizer_updates": optimizer_updates,
        "learning_rates": learning_rates,
        "clipped_gradient_norms": clipped_gradient_norms,
        "policy_state": {
            name: value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
        },
        "policy_metadata": _policy_metadata(policy),
        "policy_dependencies": policy_dependencies,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "optimizer": {
            "name": "RAdam",
            "weight_decay": 0.0,
            "gradient_clip_norm": _GRADIENT_CLIP_NORM,
            "base_learning_rate": _BASE_LEARNING_RATE,
            "max_learning_rate": _MAX_LEARNING_RATE,
            "cycle_updates": 4
            * _ceil_division(
                configuration.run_scale.training_paths_per_epoch,
                configuration.run_scale.batch_size,
            ),
        },
        "configuration": configuration_data,
        "configuration_identity": _json_identity(configuration_data),
        "code_identity": {
            "artifact_semantics": artifact_semantics("training"),
            "financial_semantics_version": FINANCIAL_SEMANTICS_VERSION,
            "git_revision": _training_git_revision(),
            "checkpoint_schema_version": 1,
        },
        "data_identities": {
            "market_source_hash": historical.source_hash,
            "hjm_calibration_identity": calibration.calibration_identity,
            "reference_bank_content_hash": snapshot.content_hash,
        },
        "scenario_identities": {
            "training": {
                "split": "training",
                "base_seed": configuration.seeds["market_scenarios"],
                "job_seed": _job_seed(
                    configuration.seeds["market_scenarios"], policy_name, horizon_years
                ),
                "epoch": "1..epochs",
                "global_path_indices": "0..training_paths_per_epoch-1",
            },
            "selection": {
                "split": "selection",
                "base_seed": configuration.seeds["market_scenarios"],
                "job_seed": _job_seed(
                    configuration.seeds["market_scenarios"], policy_name, horizon_years
                ),
                "epoch": 0,
                "global_path_indices": "0..selection_paths-1",
            },
            "objective_parameters": {
                "base_seed": configuration.seeds["objective_parameters"],
                "job_seed": _job_seed(
                    configuration.seeds["objective_parameters"],
                    policy_name,
                    horizon_years,
                ),
                "derivation": "policy|horizon|epoch|batch_start",
            },
            "model_initialization": {
                "base_seed": configuration.seeds["model_initialization"],
                "job_seed": _job_seed(
                    configuration.seeds["model_initialization"],
                    policy_name,
                    horizon_years,
                ),
                "policy": f"{policy_name} deterministic-initial-parameters",
            },
            "data_loader_order": {
                "base_seed": configuration.seeds["data_loader_order"],
                "job_seed": _job_seed(
                    configuration.seeds["data_loader_order"], policy_name, horizon_years
                ),
                "ordering": "ascending global path index",
            },
        },
        "resource_profile": resource_profile.to_dict(),
        "resource_profile_path": resource_profile_path.name,
    }


def _resource_profile(
    *,
    horizon_years: int,
    device: str,
    timing: _TimingAccumulator,
    snapshots: list[ResourceSnapshot],
) -> ResourceProfile:
    rss_values = [
        snapshot.process_rss_bytes
        for snapshot in snapshots
        if snapshot.process_rss_bytes is not None
    ]
    accelerator_values = [
        snapshot.accelerator_allocated_bytes
        for snapshot in snapshots
        if snapshot.accelerator_allocated_bytes is not None
    ]
    return ResourceProfile(
        horizon_years=horizon_years,
        device=device,
        warmup_seconds=timing.warmup_seconds,
        scenario_seconds=timing.scenario_seconds,
        forward_backward_update_seconds=timing.forward_backward_update_seconds,
        selection_seconds=timing.selection_seconds,
        artifact_seconds=timing.artifact_seconds,
        peak_process_rss_bytes=max(rss_values) if rss_values else None,
        peak_accelerator_allocated_bytes=(
            max(accelerator_values) if accelerator_values else None
        ),
        accelerator_memory_status=("measured" if accelerator_values else "unavailable"),
    )


def _build_policy(
    policy_name: str,
    *,
    horizon_years: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TreasuryPolicy:
    policy_type = _POLICY_TYPES[policy_name]
    if policy_type is BMDatePolicy:
        return BMDatePolicy(transitions=12 * horizon_years, device=device, dtype=dtype)
    return policy_type(device=device, dtype=dtype)


def _policy_metadata(policy: TreasuryPolicy) -> dict[str, object]:
    metadata = getattr(policy, "audit_metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _gradient_norm(policy: TreasuryPolicy) -> torch.Tensor:
    gradients = [
        parameter.grad.reshape(-1)
        for parameter in policy.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        raise TrainingError("Benchmark policy has no gradients to clip")
    return torch.linalg.vector_norm(torch.cat(gradients))


def _device_validation_configuration(
    configuration: ResolvedRunConfiguration, device: str, *, label: str
) -> ResolvedRunConfiguration:
    dtype = "float32"
    return replace(
        configuration,
        run_scale=RunScaleConfiguration(
            profile="device-validation",
            epochs=2,
            training_paths_per_epoch=2,
            selection_paths=2,
            test_paths=2,
            batch_size=2,
        ),
        optimization=OptimizationConfiguration(device=device, dtype=dtype),
        output=OutputConfiguration(
            directory=configuration.output.directory,
            run_name=f"{configuration.output.run_name}-device-validation-{label}",
        ),
    )


def _validate_device_portability(
    configuration: ResolvedRunConfiguration,
    *,
    horizon_years: int,
    create_trainer: Callable[[ResolvedRunConfiguration], BenchmarkTrainer],
) -> tuple[DeviceValidationRecord, ...]:
    """Resume a CPU-written artifact and reload its checkpoint on each device."""

    source = create_trainer(
        _device_validation_configuration(configuration, "cpu", label="source-cpu")
    )
    try:
        source.fit(
            horizon_years=horizon_years,
            control=TrainingControl(stop_after_completed_epoch=1),
        )
    except TrainingInterrupted as interruption:
        recovery_path = interruption.recovery_path
    else:  # pragma: no cover - the fixed two-epoch fixture must interrupt at epoch one.
        raise TrainingError("CPU portability fixture did not stop at its recovery boundary")
    if not recovery_path.is_file():
        raise TrainingError("CPU portability fixture did not write a recovery artifact")

    records: list[DeviceValidationRecord] = []
    for device in ("cpu", "mps", "cuda"):
        if not _device_is_available(device):
            records.append(
                DeviceValidationRecord(
                    device=device,
                    status="not-run",
                    finite=False,
                    updated=False,
                    clipped=False,
                    checkpoint_path=None,
                    reason=_unavailable_device_reason(device),
                )
            )
            continue
        target = create_trainer(
            _device_validation_configuration(configuration, device, label=device)
        )
        result = target.fit(
            horizon_years=horizon_years,
            control=TrainingControl(resume=True, resume_from=recovery_path),
        )
        loaded = target.load_selected_checkpoint(result.checkpoint_path)
        checkpoint_loaded = all(
            parameter.device.type == device for parameter in loaded.parameters()
        )
        records.append(
            DeviceValidationRecord(
                device=device,
                status="completed",
                finite=all(
                    torch.isfinite(torch.tensor(record.total_loss))
                    for record in result.selection_history
                ),
                updated=result.optimizer_updates > 1,
                clipped=all(
                    norm <= _GRADIENT_CLIP_NORM + 1e-5
                    for norm in result.clipped_gradient_norms
                ),
                checkpoint_path=result.checkpoint_path,
                checkpoint_loaded=checkpoint_loaded,
                resumed_from_cpu_recovery=True,
                recovery_path=recovery_path,
            )
        )
    return tuple(records)


def _device_is_available(device: str) -> bool:
    """Return availability without creating tensors on an unavailable backend."""

    if device == "cpu":
        return True
    if device == "mps":
        return torch.backends.mps.is_available()
    if device == "cuda":
        return torch.cuda.is_available()
    raise ValueError(f"Unsupported device validation target: {device}")


def _unavailable_device_reason(device: str) -> str:
    """Explain a true runtime omission without pretending it was validated."""

    names = {"mps": "MPS", "cuda": "CUDA"}
    try:
        return f"PyTorch {names[device]} runtime is not available"
    except KeyError as error:
        raise ValueError(f"Unsupported unavailable device target: {device}") from error


def _recovery_contents(
    *,
    configuration: ResolvedRunConfiguration,
    policy: TreasuryPolicy,
    policy_name: str,
    horizon_years: int,
    completed_epoch: int,
    best: tuple[float, float] | None,
    best_epoch: int,
    best_state: dict[str, torch.Tensor] | None,
    selection_history: list[SelectionRecord],
    optimizer_updates: int,
    learning_rates: list[float],
    clipped_gradient_norms: list[float],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    historical: HistoricalTermStructures,
    calibration: HjmPcaCalibration,
    snapshot: ReferenceBankSnapshot,
    resource_snapshot: ResourceSnapshot,
    selection_tracker: SelectionTracker,
    resume_lineage: list[dict[str, object]],
    policy_dependencies: dict[str, object],
) -> dict[str, object]:
    return {
        "format_version": 1,
        "kind": "epoch-recovery",
        "completed_epoch": completed_epoch,
        "best": list(best) if best is not None else None,
        "best_epoch": best_epoch,
        "best_state": (
            {name: value.detach().cpu().clone() for name, value in best_state.items()}
            if best_state is not None
            else None
        ),
        "selection_history": [asdict(record) for record in selection_history],
        "optimizer_updates": optimizer_updates,
        "learning_rates": learning_rates,
        "clipped_gradient_norms": clipped_gradient_norms,
        "policy_state": {
            name: value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "recovery_identity": _recovery_identity(
            configuration=configuration,
            policy_name=policy_name,
            horizon_years=horizon_years,
            historical=historical,
            calibration=calibration,
            snapshot=snapshot,
            policy_dependencies=policy_dependencies,
        ),
        "execution_context": _execution_context(configuration),
        "resume_lineage": resume_lineage,
        "resource_snapshot": resource_snapshot.to_dict(),
        "selection_tracker": selection_tracker.to_dict(),
    }


def _training_git_revision() -> str:
    """Return the source revision that created a portable checkpoint."""

    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unavailable"


def _load_recovery(path: Path) -> dict[str, object]:
    try:
        recovered = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingError(f"Could not load recovery artifact: {path}") from error
    if (
        not isinstance(recovered, dict)
        or recovered.get("format_version") != 1
        or recovered.get("kind") != "epoch-recovery"
    ):
        raise TrainingError("Recovery artifact has an incompatible format")
    if error := artifact_semantics_error(recovered.get("recovery_identity"), "training"):
        raise TrainingError(error)
    return recovered


def _validate_selected_checkpoint_contract(
    checkpoint: dict[str, object],
    *,
    configuration: ResolvedRunConfiguration,
    historical: HistoricalTermStructures,
    calibration: HjmPcaCalibration,
    snapshot: ReferenceBankSnapshot,
    policy_label: str,
) -> None:
    """Reject a selected checkpoint whose semantic inputs no longer match.

    Output location, resource limits, execution device, and tensor dtype are
    deliberately absent from this contract: a selected CPU checkpoint is
    expected to be remapped onto a different single device, potentially with
    PyTorch's audited state-dict dtype conversion. Financial semantics,
    declared model architecture, actual horizon, source-unit preprocessing,
    and the three immutable input identities must instead agree exactly.
    """

    if not current_financial_semantics(checkpoint):
        raise TrainingError(f"{policy_label} checkpoint financial semantics are incompatible; retraining required")
    if error := artifact_semantics_error(checkpoint.get("code_identity"), "training"):
        raise TrainingError(error)
    recorded = checkpoint.get("configuration")
    if not isinstance(recorded, dict):
        raise TrainingError(f"{policy_label} checkpoint lacks a configuration manifest")
    expected = configuration.to_dict()
    if recorded.get("architecture") != expected["architecture"]:
        raise TrainingError(f"{policy_label} checkpoint architecture is incompatible")

    recorded_experiment = recorded.get("experiment")
    expected_experiment = expected["experiment"]
    horizon_years = checkpoint.get("horizon_years")
    if (
        not isinstance(recorded_experiment, dict)
        or recorded_experiment.get("include_swaps")
        != expected_experiment["include_swaps"]
        or not isinstance(horizon_years, int)
        or horizon_years not in expected_experiment["horizons_years"]
    ):
        raise TrainingError(f"{policy_label} checkpoint horizon is incompatible")

    recorded_source = recorded.get("source_data")
    if (
        not isinstance(recorded_source, dict)
        or recorded_source.get("nss_beta_unit")
        != expected["source_data"]["nss_beta_unit"]
    ):
        raise TrainingError(
            f"{policy_label} checkpoint preprocessing is incompatible"
        )

    expected_data = {
        "market_source_hash": historical.source_hash,
        "hjm_calibration_identity": calibration.calibration_identity,
        "reference_bank_content_hash": snapshot.content_hash,
    }
    if checkpoint.get("data_identities") != expected_data:
        raise TrainingError(f"{policy_label} checkpoint data identities are incompatible")


def _recovery_identity(
    *,
    configuration: ResolvedRunConfiguration,
    policy_name: str,
    horizon_years: int,
    historical: HistoricalTermStructures,
    calibration: HjmPcaCalibration,
    snapshot: ReferenceBankSnapshot,
    policy_dependencies: dict[str, object],
) -> dict[str, object]:
    semantic_configuration = configuration.to_dict()
    semantic_configuration.pop("output")
    semantic_configuration.pop("resources")
    semantic_configuration["optimization"].pop("device")
    return {
        "policy": policy_name,
        "horizon_years": horizon_years,
        "selection_protocol_version": _SELECTION_PROTOCOL_VERSION,
        "artifact_semantics": artifact_semantics("training"),
        "financial_semantics_version": FINANCIAL_SEMANTICS_VERSION,
        "semantic_configuration_identity": _json_identity(semantic_configuration),
        "data_identities": {
            "market_source_hash": historical.source_hash,
            "hjm_calibration_identity": calibration.calibration_identity,
            "reference_bank_content_hash": snapshot.content_hash,
        },
        "policy_dependencies": policy_dependencies,
    }


def _resource_limits(configuration: ResolvedRunConfiguration) -> dict[str, float | int]:
    return {
        "wall_clock_budget_seconds": configuration.resources.wall_clock_budget_seconds,
        "process_rss_limit_bytes": configuration.resources.process_rss_limit_bytes,
        "accelerator_memory_limit_bytes": configuration.resources.accelerator_memory_limit_bytes,
    }


def _execution_context(configuration: ResolvedRunConfiguration) -> dict[str, object]:
    """Capture allowed operational overrides separately from model semantics."""

    return {
        "device": configuration.optimization.device,
        "output_directory": str(configuration.output.directory),
        "run_name": configuration.output.run_name,
        "resource_limits": _resource_limits(configuration),
    }


def _resume_lineage(
    recovered: dict[str, object],
    *,
    configuration: ResolvedRunConfiguration,
    recovery_path: Path,
) -> list[dict[str, object]]:
    """Append the operational override that led to this resumed execution."""

    stored = recovered.get("resume_lineage", [])
    if not isinstance(stored, list) or not all(
        isinstance(item, dict) for item in stored
    ):
        raise TrainingError("Recovery artifact has an invalid resume lineage")
    lineage = [dict(item) for item in stored]
    previous = _recovered_execution_context(recovered)
    current = _execution_context(configuration)
    if previous != current:
        lineage.append(
            {
                "source_recovery_path": str(recovery_path),
                "source_completed_epoch": int(recovered["completed_epoch"]),
                "resumed_at_unix_seconds": time.time(),
                "source_recovery_identity": recovered["recovery_identity"],
                "previous_execution": previous,
                "current_execution": current,
            }
        )
    return lineage


def _recovered_execution_context(recovered: dict[str, object]) -> dict[str, object]:
    """Read current-format execution data, with a narrow compatibility fallback."""

    context = recovered.get("execution_context", recovered.get("execution_overrides"))
    if not isinstance(context, dict):
        raise TrainingError(
            "Recovery artifact is incompatible: missing execution context"
        )
    resource_limits = context.get("resource_limits", recovered.get("resource_limits"))
    if not isinstance(resource_limits, dict):
        raise TrainingError(
            "Recovery artifact is incompatible: missing resource limits"
        )
    required = {"device", "output_directory", "run_name"}
    if not required.issubset(context):
        raise TrainingError(
            "Recovery artifact is incompatible: invalid execution context"
        )
    return {
        "device": context["device"],
        "output_directory": context["output_directory"],
        "run_name": context["run_name"],
        "resource_limits": dict(resource_limits),
    }


def _validate_recovery_compatibility(
    recovered: dict[str, object], *, expected: dict[str, object]
) -> None:
    identity = recovered.get("recovery_identity")
    if (
        not isinstance(identity, dict)
        or identity.get("selection_protocol_version") != _SELECTION_PROTOCOL_VERSION
    ):
        raise TrainingError(
            "Recovery selection protocol is incompatible; start a new training run"
        )
    if recovered.get("recovery_identity") != expected:
        raise TrainingError("Recovery artifact is incompatible with this training job")


def _validate_resource_override(
    recovered: dict[str, object], *, configuration: ResolvedRunConfiguration
) -> None:
    saved = _recovered_execution_context(recovered)["resource_limits"]
    if not isinstance(saved, dict):  # pragma: no cover - checked by helper above.
        raise TrainingError(
            "Recovery artifact is incompatible: missing resource limits"
        )
    current = _resource_limits(configuration)
    if any(current[name] < saved[name] for name in current):
        raise TrainingError(
            "Recovery artifact is incompatible: resource budgets may only increase"
        )


def _selection_record_from_dict(record: object) -> SelectionRecord:
    if not isinstance(record, dict):
        raise TrainingError("Recovery artifact has an invalid selection history")
    action = record.get("action_record")
    if not isinstance(action, dict):
        raise TrainingError("Recovery artifact has an invalid action record")
    return SelectionRecord(
        epoch=int(record["epoch"]),
        total_loss=float(record["total_loss"]),
        penalty_loss=float(record["penalty_loss"]),
        action_record=SelectionActionRecord(
            investment_maturity_totals=tuple(action["investment_maturity_totals"]),
            funding_maturity_totals=tuple(action["funding_maturity_totals"]),
            minimum_action=float(action["minimum_action"]),
            maximum_action=float(action["maximum_action"]),
        ),
    )


def _selection_tracker_from_dict(
    value: object, *, schedule: SelectionSchedule
) -> SelectionTracker:
    if not isinstance(value, dict):
        raise TrainingError("Recovery artifact has an invalid selection tracker")
    raw_best = value.get("best_selection")
    if raw_best is not None and (not isinstance(raw_best, list) or len(raw_best) != 2):
        raise TrainingError("Recovery artifact has an invalid selection tracker")
    reference = value.get("stopping_reference_total")
    wait_count = value.get("checks_without_improvement")
    if (
        "stopping_reference_total" not in value
        or type(wait_count) is not int
        or wait_count < 0
        or (reference is not None and (
            type(reference) not in (int, float) or not math.isfinite(reference)
        ))
        or (raw_best is not None and reference is None)
    ):
        raise TrainingError("Recovery artifact has an invalid early-stopping state")
    return SelectionTracker(
        schedule=schedule,
        best_selection=(float(raw_best[0]), float(raw_best[1]))
        if raw_best is not None
        else None,
        last_improvement_epoch=(
            int(value["last_improvement_epoch"])
            if value.get("last_improvement_epoch") is not None
            else None
        ),
        stopping_reference_total=reference,
        checks_without_improvement=wait_count,
    )


def _save_recovery(path: Path, contents: dict[str, object]) -> None:
    _save_checkpoint(path, contents)


def _copy_identity_checked_artifact(source: Path, destination: Path) -> None:
    """Atomically copy one immutable dependency, refusing a conflicting target."""

    if not source.is_file():
        raise TrainingError(f"Required immutable artifact is missing: {source}")
    source_hash = _file_sha256(source)
    if destination.is_file():
        if _file_sha256(destination) != source_hash:
            raise TrainingError(
                f"Immutable artifact destination conflicts with source: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    if _file_sha256(destination) != source_hash:
        raise TrainingError(
            f"Immutable artifact copy failed identity check: {destination}"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_json_artifact(path: Path, contents: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", delete=False
    ) as temporary:
        temporary.write(json.dumps(contents, indent=2, sort_keys=True) + "\n")
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _save_checkpoint(path: Path, contents: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(contents, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _save_resource_profile(path: Path, profile: ResourceProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", delete=False
    ) as temporary:
        temporary.write(json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _warmup_device(device: torch.device) -> float:
    started = time.perf_counter()
    torch.zeros((), device=device).add_(1.0)
    _synchronize_device(device)
    return time.perf_counter() - started


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _derived_seed(base_seed: int, *parts: object) -> int:
    encoded = "|".join((str(base_seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


def _job_seed(base_seed: int, policy_name: str, horizon_years: int) -> int:
    return _derived_seed(base_seed, policy_name, horizon_years)


def _policy_tag(policy_name: str) -> str:
    return policy_name.replace("^", "_")


def _json_identity(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _ceil_division(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator
