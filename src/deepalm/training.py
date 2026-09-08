"""Bounded local training for no-swap treasury benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from deepalm.baselines import FrozenDateBenchmarkReference
from deepalm.config import (
    OptimizationConfiguration,
    OutputConfiguration,
    ResolvedRunConfiguration,
    RunScaleConfiguration,
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
from deepalm.runoff import ALMSimulator
from deepalm.term_structures import (
    HistoricalTermStructures,
    HjmPcaCalibration,
    MarketScenarioBatch,
    MarketScenarioModel,
)

_BASE_LEARNING_RATE = 5e-4
_MAX_LEARNING_RATE = 5e-3
_GRADIENT_CLIP_NORM = 0.2
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
        return cls(first_selection_epoch=1, patience=None, minimum_relative_improvement=0.0)

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
    """Persistable paper-scale early-stopping state with penalty-loss tie breaks."""

    schedule: SelectionSchedule
    best_selection: tuple[float, float] | None = None
    last_improvement_epoch: int | None = None

    def should_select(self, epoch: int) -> bool:
        return epoch >= self.schedule.first_selection_epoch

    def record(self, epoch: int, candidate: tuple[float, float]) -> bool:
        if not self.should_select(epoch):
            return False
        if self.best_selection is None:
            self.best_selection = candidate
            self.last_improvement_epoch = epoch
            return True
        total_before, penalty_before = self.best_selection
        total_after, penalty_after = candidate
        relative_improvement = (total_before - total_after) / max(
            abs(total_before), 1e-12
        )
        improved = (
            relative_improvement + 1e-12 >= self.schedule.minimum_relative_improvement
            or (total_after == total_before and penalty_after < penalty_before)
        )
        if improved:
            self.best_selection = candidate
            self.last_improvement_epoch = epoch
        return improved

    def should_stop(self, epoch: int) -> bool:
        return (
            self.schedule.patience is not None
            and self.last_improvement_epoch is not None
            and epoch - self.last_improvement_epoch >= self.schedule.patience
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "best_selection": list(self.best_selection)
            if self.best_selection is not None
            else None,
            "last_improvement_epoch": self.last_improvement_epoch,
        }


@dataclass(frozen=True)
class SelectionRecord:
    """One fixed-selection measurement made after an epoch."""

    epoch: int
    total_loss: float
    penalty_loss: float
    action_record: SelectionActionRecord


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
        if policy_name not in _POLICY_TYPES:
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
    ) -> BenchmarkTrainingResult:
        """Run, or safely resume, one benchmark-training job."""

        resolved_control = control or TrainingControl()
        recovery_path = resolved_control.resume_from or self._recovery_path(horizon_years)
        interruption_path = self._interruption_path(horizon_years)
        try:
            return self._fit(
                horizon_years=horizon_years,
                control=resolved_control,
                recovery_path=recovery_path,
                interruption_path=interruption_path,
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
            diagnostics = {"error": str(error), "resource_snapshot": self._monitor.snapshot().to_dict()}
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
        policy = _build_policy(
            self._policy_name,
            horizon_years=horizon_years,
            device=self._device,
            dtype=self._dtype,
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
                ),
            )
            _validate_resource_override(
                recovered, configuration=self._configuration
            )
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

        if control.stop_after_completed_epoch == 0:
            self._raise_controlled_interruption(
                recovery_path,
                interruption_path,
                completed_epoch=0,
            )

        for epoch in range(start_epoch, scale.epochs + 1):
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
                    convention=self._configuration.convention.profile,
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
                if not any(
                    not torch.equal(old, new.detach())
                    for old, new in zip(before, policy.parameters(), strict=True)
                ):
                    raise TrainingError(
                        f"{self._policy_name} optimizer step did not update policy parameters",
                        diagnostics={
                            "horizon_years": horizon_years,
                            "epoch": epoch,
                            "batch_start": start,
                            "paths": paths,
                        },
                    )
                updates += 1
                peaks.append(
                    self._monitor.check(
                        f"after-{_policy_tag(self._policy_name)}-{horizon_years}y-update-{updates}"
                    )
                )

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
                ),
            )
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

    def fit_all(self) -> tuple[BenchmarkTrainingResult, ...]:
        """Train the configured benchmark once for every resolved horizon."""

        return tuple(
            self.fit(horizon_years=horizon_years)
            for horizon_years in self._configuration.experiment.horizons_years
        )

    def validate_devices(
        self, *, horizon_years: int
    ) -> tuple[DeviceValidationRecord, ...]:
        """Run the smallest real-update fixture on CPU and available MPS."""

        records: list[DeviceValidationRecord] = []
        for device in ("cpu", "mps"):
            if device == "mps" and not torch.backends.mps.is_available():
                records.append(
                    DeviceValidationRecord(
                        device=device,
                        status="not-run",
                        finite=False,
                        updated=False,
                        clipped=False,
                        checkpoint_path=None,
                    )
                )
                continue
            result = BenchmarkTrainer(
                _device_validation_configuration(self._configuration, device),
                snapshot=self._snapshot,
                historical=self._historical,
                calibration=self._calibration,
                policy_name=self._policy_name,
                simulator=self._simulator,
                market_model=self._market_model,
            ).fit(horizon_years=horizon_years)
            records.append(
                DeviceValidationRecord(
                    device=device,
                    status="completed",
                    finite=all(
                        torch.isfinite(torch.tensor(record.total_loss))
                        for record in result.selection_history
                    ),
                    updated=result.optimizer_updates > 0,
                    clipped=all(
                        norm <= _GRADIENT_CLIP_NORM + 1e-5
                        for norm in result.clipped_gradient_norms
                    ),
                    checkpoint_path=result.checkpoint_path,
                )
            )
        return tuple(records)

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
            convention=self._configuration.convention.profile,
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
                    convention=self._configuration.convention.profile,
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
                    convention=self._configuration.convention.profile,
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
    ) -> BenchmarkTrainingResult:
        result = super().fit(horizon_years=horizon_years, control=control)
        reference = FrozenDateBenchmarkReference.freeze(result.checkpoint_path)
        return replace(
            result,
            baseline_reference_path=reference.reference_path,
            baseline_reference_identity=reference.reference_identity,
        )


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
) -> dict[str, object]:
    configuration_data = configuration.to_dict()
    return {
        "format_version": 1,
        "policy": policy_name,
        "horizon_years": horizon_years,
        "selected_epoch": selected_epoch,
        "selection_history": [asdict(record) for record in selection_history],
        "optimizer_updates": optimizer_updates,
        "learning_rates": learning_rates,
        "clipped_gradient_norms": clipped_gradient_norms,
        "policy_state": {
            name: value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
        },
        "policy_metadata": _policy_metadata(policy),
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
    configuration: ResolvedRunConfiguration, device: str
) -> ResolvedRunConfiguration:
    dtype = "float32" if device == "mps" else "float64"
    return replace(
        configuration,
        run_scale=RunScaleConfiguration(
            profile="device-validation",
            epochs=1,
            training_paths_per_epoch=2,
            selection_paths=2,
            test_paths=2,
            batch_size=2,
        ),
        optimization=OptimizationConfiguration(device=device, dtype=dtype),
        output=OutputConfiguration(
            directory=configuration.output.directory,
            run_name=f"{configuration.output.run_name}-device-validation-{device}",
        ),
    )


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
        ),
        "execution_context": _execution_context(configuration),
        "resume_lineage": resume_lineage,
        "resource_snapshot": resource_snapshot.to_dict(),
        "selection_tracker": selection_tracker.to_dict(),
    }


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
    return recovered


def _recovery_identity(
    *,
    configuration: ResolvedRunConfiguration,
    policy_name: str,
    horizon_years: int,
    historical: HistoricalTermStructures,
    calibration: HjmPcaCalibration,
    snapshot: ReferenceBankSnapshot,
) -> dict[str, object]:
    semantic_configuration = configuration.to_dict()
    semantic_configuration.pop("output")
    semantic_configuration.pop("resources")
    semantic_configuration["optimization"].pop("device")
    return {
        "policy": policy_name,
        "horizon_years": horizon_years,
        "semantic_configuration_identity": _json_identity(semantic_configuration),
        "data_identities": {
            "market_source_hash": historical.source_hash,
            "hjm_calibration_identity": calibration.calibration_identity,
            "reference_bank_content_hash": snapshot.content_hash,
        },
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
    if not isinstance(stored, list) or not all(isinstance(item, dict) for item in stored):
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
        raise TrainingError("Recovery artifact is incompatible: missing execution context")
    resource_limits = context.get("resource_limits", recovered.get("resource_limits"))
    if not isinstance(resource_limits, dict):
        raise TrainingError("Recovery artifact is incompatible: missing resource limits")
    required = {"device", "output_directory", "run_name"}
    if not required.issubset(context):
        raise TrainingError("Recovery artifact is incompatible: invalid execution context")
    return {
        "device": context["device"],
        "output_directory": context["output_directory"],
        "run_name": context["run_name"],
        "resource_limits": dict(resource_limits),
    }


def _validate_recovery_compatibility(
    recovered: dict[str, object], *, expected: dict[str, object]
) -> None:
    if recovered.get("recovery_identity") != expected:
        raise TrainingError("Recovery artifact is incompatible with this training job")


def _validate_resource_override(
    recovered: dict[str, object], *, configuration: ResolvedRunConfiguration
) -> None:
    saved = _recovered_execution_context(recovered)["resource_limits"]
    if not isinstance(saved, dict):  # pragma: no cover - checked by helper above.
        raise TrainingError("Recovery artifact is incompatible: missing resource limits")
    current = _resource_limits(configuration)
    if any(current[name] < saved[name] for name in current):
        raise TrainingError("Recovery artifact is incompatible: resource budgets may only increase")


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
    if raw_best is not None and (
        not isinstance(raw_best, list) or len(raw_best) != 2
    ):
        raise TrainingError("Recovery artifact has an invalid selection tracker")
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
    )


def _save_recovery(path: Path, contents: dict[str, object]) -> None:
    _save_checkpoint(path, contents)


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
