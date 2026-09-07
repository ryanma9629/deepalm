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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from deepalm.config import ResolvedRunConfiguration
from deepalm.planning import build_execution_plan
from deepalm.resources import BudgetExceeded, ResourceMonitor


class RunStatus(StrEnum):
    COMPLETED = "completed"
    ACCEPTANCE_FAILED = "acceptance_failed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class AcceptanceStatus(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


class OperationalRunError(RuntimeError):
    """A recoverable pipeline failure with serializable diagnostic context."""

    def __init__(
        self, message: str, *, diagnostics: Mapping[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


class MarketPreflightModel(Protocol):
    """Structural contract used by the bounded market preflight stage."""

    def load_historical_term_structures(self, source_path: Path, *, beta_unit: str) -> Any: ...

    def calibrate_hjm_pca(self, historical: Any) -> Any: ...

    def generate_hjm_scenarios(self, historical: Any, calibration: Any, **kwargs: Any) -> Any: ...


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
        self, configuration: ResolvedRunConfiguration
    ) -> RunBundle:
        """Build the canonical, reviewable Reference Bank without policy training."""

        try:
            from deepalm.reference_bank import ReferenceBankProvider
            from deepalm.term_structures import MarketScenarioModel

            historical = MarketScenarioModel().load_historical_term_structures(
                configuration.source_data.snb_csv,
                beta_unit=configuration.source_data.nss_beta_unit,
            )
            provider = ReferenceBankProvider()
            snapshot = provider.build_canonical(historical)
            manifest = _build_manifest(
                configuration, RunStatus.COMPLETED, AcceptanceStatus.PENDING
            )
            manifest["reference_bank"] = {
                "content_hash": snapshot.content_hash,
                "as_of_date": snapshot.as_of_date,
                "initial_curve_identity": snapshot.initial_curve_identity,
                "total_assets_mchf": snapshot.total_assets,
                "total_liabilities_mchf": snapshot.total_liabilities,
                "equity_mchf": snapshot.equity,
                "loan_duration_years": snapshot.loan_duration_years,
                "deposit_duration_years": snapshot.deposit_duration_years,
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
    return {}
