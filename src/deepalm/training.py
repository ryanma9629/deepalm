"""Bounded local training for the BM^E treasury benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

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
from deepalm.policies import BMEqualPolicy
from deepalm.reference_bank import ReferenceBankSnapshot
from deepalm.resources import ResourceMonitor, ResourceSnapshot
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


class TrainingError(RuntimeError):
    """Raised when a bounded BM^E training job cannot be completed safely."""

    def __init__(
        self, message: str, *, diagnostics: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class SelectionRecord:
    """One fixed-selection measurement made after an epoch."""

    epoch: int
    total_loss: float
    penalty_loss: float


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
class BMETrainingResult:
    """Selected BM^E policy and the evidence produced by its local training."""

    policy: BMEqualPolicy
    checkpoint_path: Path
    selected_epoch: int
    selection_history: tuple[SelectionRecord, ...]
    optimizer_updates: int
    learning_rates: tuple[float, ...]
    clipped_gradient_norms: tuple[float, ...]
    resource_profile: ResourceProfile


@dataclass(frozen=True)
class DeviceValidationRecord:
    """A compact real-update validation result for one requested local device."""

    device: str
    status: str
    finite: bool
    updated: bool
    clipped: bool
    checkpoint_path: Path | None


class BMETrainer:
    """Train, select, restore, and document one BM^E policy/horizon job."""

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

    def fit(self, *, horizon_years: int) -> BMETrainingResult:
        """Run all configured local BM^E updates and reload the best selection state."""

        if horizon_years not in self._configuration.experiment.horizons_years:
            raise TrainingError(
                "Requested horizon is not enabled by the resolved experiment"
            )
        if "BM^E" not in self._configuration.policy.names:
            raise TrainingError("Resolved policy configuration does not include BM^E")

        scale = self._configuration.run_scale
        updates_per_epoch = _ceil_division(
            scale.training_paths_per_epoch, scale.batch_size
        )
        torch.manual_seed(
            _derived_seed(
                self._configuration.seeds["model_initialization"], "BM^E", horizon_years
            )
        )
        policy = BMEqualPolicy(device=self._device, dtype=self._dtype)
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
        peaks: list[ResourceSnapshot] = [self._monitor.check("before-bme-warmup")]
        timing.warmup_seconds = _warmup_device(self._device)
        peaks.append(self._monitor.check("after-bme-warmup"))

        best: tuple[float, float] | None = None
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        history: list[SelectionRecord] = []
        learning_rates: list[float] = []
        clipped_gradient_norms: list[float] = []
        updates = 0
        for epoch in range(1, scale.epochs + 1):
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
                        "BM^E rollout did not return the training objective"
                    )
                loss = outcome.objective.total.mean()
                if not torch.isfinite(loss):
                    raise TrainingError(
                        "BM^E training loss is non-finite",
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
                        "BM^E gradient clipping did not produce a finite bound",
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
                        "BM^E optimizer step did not update either scale",
                        diagnostics={
                            "horizon_years": horizon_years,
                            "epoch": epoch,
                            "batch_start": start,
                            "paths": paths,
                        },
                    )
                updates += 1
                peaks.append(
                    self._monitor.check(f"after-bme-{horizon_years}y-update-{updates}")
                )

            selection = self._select(
                policy,
                epoch=epoch,
                horizon_years=horizon_years,
                timing=timing,
            )
            history.append(selection)
            candidate = (selection.total_loss, selection.penalty_loss)
            if best is None or candidate < best:
                best = candidate
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in policy.state_dict().items()
                }
            peaks.append(
                self._monitor.check(f"after-bme-{horizon_years}y-selection-{epoch}")
            )

        if best_state is None or best is None:
            raise TrainingError("BM^E training did not produce a selectable checkpoint")
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
        return BMETrainingResult(
            policy=policy,
            checkpoint_path=checkpoint_path,
            selected_epoch=best_epoch,
            selection_history=tuple(history),
            optimizer_updates=updates,
            learning_rates=tuple(learning_rates),
            clipped_gradient_norms=tuple(clipped_gradient_norms),
            resource_profile=profile,
        )

    def fit_all(self) -> tuple[BMETrainingResult, ...]:
        """Train the configured BM^E policy once for every resolved horizon."""

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
            result = BMETrainer(
                _device_validation_configuration(self._configuration, device),
                snapshot=self._snapshot,
                historical=self._historical,
                calibration=self._calibration,
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
                self._configuration.seeds["market_scenarios"], horizon_years
            ),
            split="training",
            epoch=epoch,
            global_path_indices=tuple(range(start, start + paths)),
        )
        timing.scenario_seconds += time.perf_counter() - started
        return market

    def _select(
        self,
        policy: BMEqualPolicy,
        *,
        epoch: int,
        horizon_years: int,
        timing: _TimingAccumulator,
    ) -> SelectionRecord:
        total_losses: list[torch.Tensor] = []
        penalty_losses: list[torch.Tensor] = []
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
                        self._configuration.seeds["market_scenarios"], horizon_years
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
                        "BM^E selection rollout did not return an objective"
                    )
                total_losses.append(outcome.objective.total.detach().cpu())
                penalty_losses.append(outcome.objective.penalty.detach().cpu())
                _synchronize_device(self._device)
                timing.selection_seconds += time.perf_counter() - evaluation_started
        return SelectionRecord(
            epoch=epoch,
            total_loss=float(torch.cat(total_losses).mean()),
            penalty_loss=float(torch.cat(penalty_losses).mean()),
        )

    def _checkpoint_path(self, horizon_years: int) -> Path:
        return (
            self._configuration.output.directory
            / self._configuration.output.run_name
            / f"BM_E_{horizon_years}y.pt"
        )

    def _resource_profile_path(self, horizon_years: int) -> Path:
        return self._checkpoint_path(horizon_years).with_suffix(".profile.json")


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
    policy: BMEqualPolicy,
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
        "policy": "BM^E",
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
                    configuration.seeds["market_scenarios"], horizon_years
                ),
                "epoch": "1..epochs",
                "global_path_indices": "0..training_paths_per_epoch-1",
            },
            "selection": {
                "split": "selection",
                "base_seed": configuration.seeds["market_scenarios"],
                "job_seed": _job_seed(
                    configuration.seeds["market_scenarios"], horizon_years
                ),
                "epoch": 0,
                "global_path_indices": "0..selection_paths-1",
            },
            "objective_parameters": {
                "base_seed": configuration.seeds["objective_parameters"],
                "job_seed": _job_seed(
                    configuration.seeds["objective_parameters"], horizon_years
                ),
                "derivation": "policy|horizon|epoch|batch_start",
            },
            "model_initialization": {
                "base_seed": configuration.seeds["model_initialization"],
                "job_seed": _job_seed(
                    configuration.seeds["model_initialization"], horizon_years
                ),
                "policy": "BM^E deterministic-zero-scales",
            },
            "data_loader_order": {
                "base_seed": configuration.seeds["data_loader_order"],
                "job_seed": _job_seed(
                    configuration.seeds["data_loader_order"], horizon_years
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


def _gradient_norm(policy: BMEqualPolicy) -> torch.Tensor:
    gradients = [
        parameter.grad.reshape(-1)
        for parameter in policy.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        raise TrainingError("BM^E policy has no gradients to clip")
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


def _job_seed(base_seed: int, horizon_years: int) -> int:
    return _derived_seed(base_seed, "BM^E", horizon_years)


def _json_identity(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _ceil_division(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator
