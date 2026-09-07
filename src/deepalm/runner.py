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
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from deepalm.config import ResolvedRunConfiguration


class RunStatus(StrEnum):
    COMPLETED = "completed"
    ACCEPTANCE_FAILED = "acceptance_failed"
    FAILED = "failed"


class AcceptanceStatus(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class RunBundle:
    """Auditable result of one reproduction execution."""

    status: RunStatus
    acceptance_status: AcceptanceStatus
    artifact_directory: Path | None
    error: str | None = None


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
        except (OSError, ValueError) as error:
            return RunBundle(
                status=RunStatus.FAILED,
                acceptance_status=AcceptanceStatus.PENDING,
                artifact_directory=None,
                error=str(error),
            )

        return RunBundle(
            status=status,
            acceptance_status=acceptance_status,
            artifact_directory=artifact_directory,
        )


def _build_manifest(
    configuration: ResolvedRunConfiguration,
    status: RunStatus,
    acceptance_status: AcceptanceStatus,
) -> dict[str, object]:
    resolved_configuration = configuration.to_dict()
    return {
        "status": status.value,
        "acceptance_status": acceptance_status.value,
        "created_at": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "runtime": _runtime_identity(configuration),
        "input_hashes": {
            "snb_csv": _sha256(configuration.source_data.snb_csv),
            "paper_pdf": _sha256(configuration.source_data.paper_pdf),
        },
        "seed_registry": configuration.seeds,
        "resolved_configuration": resolved_configuration,
        "resolved_configuration_hash": _configuration_hash(resolved_configuration),
    }


def _write_bundle_atomically(
    configuration: ResolvedRunConfiguration, manifest: dict[str, object]
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
        os.replace(staging_directory, final_directory)
    except OSError:
        shutil.rmtree(staging_directory, ignore_errors=True)
        raise
    return final_directory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _configuration_hash(resolved_configuration: dict[str, object]) -> str:
    serialized = json.dumps(
        resolved_configuration, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _status_for(acceptance_status: AcceptanceStatus) -> RunStatus:
    if acceptance_status is AcceptanceStatus.FAILED:
        return RunStatus.ACCEPTANCE_FAILED
    return RunStatus.COMPLETED
