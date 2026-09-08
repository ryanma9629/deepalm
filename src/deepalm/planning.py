"""Resolved, bounded execution plans for Deep ALM workflow runs."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from deepalm.config import ResolvedRunConfiguration


@dataclass(frozen=True)
class PlannedTrainingJob:
    """One policy/horizon training job in a resolved workflow plan."""

    policy: str
    convention: str
    horizon_years: int
    seed: int
    epochs: int
    training_paths_per_epoch: int
    selection_paths: int
    test_paths: int
    batch_size: int
    optimizer_updates: int
    architecture_profile: str
    architecture_widths: tuple[int, int, int, int]


@dataclass(frozen=True)
class ExecutionPlan:
    """Auditable work inventory before any training is started."""

    run_scale: str
    purpose: str
    jobs: tuple[PlannedTrainingJob, ...]
    primary_training_jobs: int
    primary_optimizer_updates: int
    paper_width_optimizer_updates: int
    mm_truncation_requires_training: bool
    evaluation_paths_per_horizon: int
    resource_budget: dict[str, int | float]
    estimated_cost_status: str
    optional_stages: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON-ready plan data for CLI output and manifests."""

        result = asdict(self)
        result["jobs"] = [asdict(job) for job in self.jobs]
        result["optional_stages"] = list(self.optional_stages)
        return result


def build_execution_plan(configuration: ResolvedRunConfiguration) -> ExecutionPlan:
    """Declare all requested work without allocating scenarios or model tensors."""

    scale = configuration.run_scale
    updates_per_epoch = _ceil_division(scale.training_paths_per_epoch, scale.batch_size)
    updates = updates_per_epoch * scale.epochs
    conventions = (
        ("corrected", "paper")
        if scale.profile == "paired_convention_pilot"
        else (configuration.convention.profile,)
    )
    jobs = tuple(
        PlannedTrainingJob(
            policy=policy,
            convention=convention,
            horizon_years=horizon,
            seed=_job_seed(
                configuration.seeds["model_initialization"], policy, horizon
            ),
            epochs=scale.epochs,
            training_paths_per_epoch=scale.training_paths_per_epoch,
            selection_paths=scale.selection_paths,
            test_paths=scale.test_paths,
            batch_size=scale.batch_size,
            optimizer_updates=updates,
            architecture_profile=configuration.architecture.profile,
            architecture_widths=configuration.architecture.widths,
        )
        for convention in conventions
        for policy in configuration.policy.names
        for horizon in configuration.experiment.horizons_years
    )
    both_horizons = set(configuration.experiment.horizons_years) == {5, 15}
    has_mm = "MM" in configuration.policy.names
    paper_width_checks = (
        2
        if has_mm
        and both_horizons
        and scale.profile != "paired_convention_pilot"
        else 0
    )
    return ExecutionPlan(
        run_scale=scale.profile,
        purpose=configuration.acceptance.purpose,
        jobs=jobs,
        primary_training_jobs=len(jobs),
        primary_optimizer_updates=sum(job.optimizer_updates for job in jobs),
        paper_width_optimizer_updates=paper_width_checks,
        mm_truncation_requires_training=False,
        evaluation_paths_per_horizon=scale.test_paths,
        resource_budget={
            "wall_clock_budget_seconds": configuration.resources.wall_clock_budget_seconds,
            "process_rss_limit_bytes": configuration.resources.process_rss_limit_bytes,
            "accelerator_memory_limit_bytes": (
                configuration.resources.accelerator_memory_limit_bytes
            ),
        },
        estimated_cost_status="unmeasured",
        optional_stages=(
            "first-training-slice profiling",
            "recovery fixture",
            "representative sensitivity",
            "paired Paper training",
            "extended research matrix",
        ),
    )


def _ceil_division(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _job_seed(base_seed: int, policy: str, horizon_years: int) -> int:
    """Name a policy/horizon initialization stream without positional coupling."""

    encoded = f"{base_seed}|{policy}|{horizon_years}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")
