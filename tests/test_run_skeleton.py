from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deepalm.cli import main
from deepalm.config import ConfigurationError, resolve_configuration
from deepalm.runner import ReproductionRunner, RunStatus


def configuration_data(tmp_path: Path) -> dict[str, object]:
    return {
        "source_data": {
            "snb_csv": str(tmp_path / "snb.csv"),
            "paper_pdf": str(tmp_path / "paper.pdf"),
        },
        "convention": {"profile": "corrected"},
        "run_scale": {"profile": "quick"},
        "reference_bank": {
            "profile": "canonical",
            "initial_assets": {"value": 10_000, "unit": "mCHF"},
        },
        "experiment": {"horizons_years": [5, 15], "include_swaps": False},
        "policy": {"names": ["bme"]},
        "optimization": {"device": "cpu", "dtype": "float64"},
        "seeds": {"market": 11, "training": 12, "bootstrap": 13},
        "output": {"directory": str(tmp_path / "runs"), "run_name": "smoke"},
        "acceptance": {"required_status": "development-validated"},
    }


def write_source_inputs(configuration: dict[str, object]) -> None:
    source_data = configuration["source_data"]
    assert isinstance(source_data, dict)
    Path(str(source_data["snb_csv"])).write_text("source-data", encoding="utf-8")
    Path(str(source_data["paper_pdf"])).write_bytes(b"paper-data")


def test_corrected_profile_resolves_and_override_is_custom(tmp_path: Path) -> None:
    resolved = resolve_configuration(configuration_data(tmp_path))

    assert resolved.convention.pca_loading_scale == "sqrt_eigenvalue"
    assert resolved.convention.loan_interest_annualization == "monthly"
    assert resolved.convention.is_custom is False
    assert resolved.run_scale.epochs == 5
    assert resolved.run_scale.training_paths_per_epoch == 256

    overridden = configuration_data(tmp_path)
    overridden["convention"] = {
        "profile": "corrected",
        "pca_loading_scale": "eigenvalue",
    }

    custom = resolve_configuration(overridden)

    assert custom.convention.is_custom is True
    assert custom.convention.pca_loading_scale == "eigenvalue"
    assert custom.convention.loan_interest_annualization == "monthly"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda data: data.update({"unexpected": True}),
            "Unknown configuration sections",
        ),
        (
            lambda data: data["experiment"].update({"horizons_years": [7]}),
            "horizons_years",
        ),
        (
            lambda data: data["reference_bank"].update(
                {"initial_assets": {"value": 10_000, "unit": "CHF"}}
            ),
            "mCHF",
        ),
        (lambda data: data["seeds"].update({"market": -1}), "non-negative integer"),
    ],
)
def test_configuration_rejects_invalid_inputs(
    tmp_path: Path, change: object, message: str
) -> None:
    invalid = configuration_data(tmp_path)
    assert callable(change)
    change(invalid)

    with pytest.raises(ConfigurationError, match=message):
        resolve_configuration(invalid)


def test_runner_writes_an_atomic_auditable_bundle(tmp_path: Path) -> None:
    configuration = configuration_data(tmp_path)
    write_source_inputs(configuration)
    resolved = resolve_configuration(configuration)

    bundle = ReproductionRunner().run(resolved)

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.acceptance_status == "pending"
    assert bundle.artifact_directory is not None
    assert bundle.artifact_directory.is_dir()
    assert not list(bundle.artifact_directory.parent.glob(".smoke-*"))

    manifest = json.loads((bundle.artifact_directory / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["acceptance_status"] == "pending"
    assert manifest["resolved_configuration"]["convention"]["profile"] == "corrected"
    assert manifest["seed_registry"] == {"bootstrap": 13, "market": 11, "training": 12}
    assert (
        manifest["input_hashes"]["snb_csv"]
        == hashlib.sha256(b"source-data").hexdigest()
    )
    assert (
        manifest["input_hashes"]["paper_pdf"]
        == hashlib.sha256(b"paper-data").hexdigest()
    )
    assert manifest["runtime"]["python"]
    assert manifest["runtime"]["dependencies"]["pyyaml"]
    assert manifest["runtime"]["device"] == "cpu"
    assert manifest["runtime"]["dtype"] == "float64"
    assert manifest["git_revision"]


def test_runner_reports_operational_failure_without_a_completed_bundle(
    tmp_path: Path,
) -> None:
    configuration = configuration_data(tmp_path)
    write_source_inputs(configuration)
    configuration["output"] = {
        "directory": str(tmp_path / "not-a-directory"),
        "run_name": "smoke",
    }
    Path(str(configuration["output"]["directory"])).write_text("file", encoding="utf-8")

    bundle = ReproductionRunner().run(resolve_configuration(configuration))

    assert bundle.status is RunStatus.FAILED
    assert bundle.artifact_directory is None
    assert bundle.error is not None
    assert "not a directory" in bundle.error


def test_cli_returns_a_configuration_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invalid = configuration_data(tmp_path)
    invalid["unexpected"] = True
    configuration_path = tmp_path / "invalid.yaml"
    import yaml

    configuration_path.write_text(yaml.safe_dump(invalid), encoding="utf-8")

    exit_code = main(["run", "--config", str(configuration_path)])

    assert exit_code == 2
    assert "Unknown configuration sections" in capsys.readouterr().err
