from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deepalm.cli import main
from deepalm.config import ConfigurationError, resolve_configuration
from deepalm.planning import build_execution_plan
from deepalm.resources import BudgetExceeded, ResourceMonitor
from deepalm.runner import AcceptanceStatus, ReproductionRunner, RunStatus


def configuration_data(tmp_path: Path) -> dict[str, object]:
    return {
        "source_data": {
            "snb_csv": str(tmp_path / "snb.csv"),
            "paper_pdf": str(tmp_path / "paper.pdf"),
            "nss_beta_unit": "percentage_points",
        },
        "convention": {"profile": "corrected"},
        "run_scale": {"profile": "local_flow"},
        "architecture": {"profile": "compact"},
        "reference_bank": {
            "profile": "canonical",
            "initial_assets": {"value": 10_000, "unit": "mCHF"},
        },
        "experiment": {"horizons_years": [5, 15], "include_swaps": False},
        "policy": {"names": ["BM^E"]},
        "optimization": {"device": "cpu", "dtype": "float64"},
        "resources": {
            "wall_clock_budget_seconds": 1800,
            "process_rss_limit_bytes": 4 * 1024**3,
            "accelerator_memory_limit_bytes": 4 * 1024**3,
        },
        "seeds": {
            "market_scenarios": 11,
            "objective_parameters": 12,
            "model_initialization": 13,
            "data_loader_order": 14,
            "bootstrap": 15,
            "sensitivity": 16,
        },
        "output": {"directory": str(tmp_path / "runs"), "run_name": "smoke"},
        "acceptance": {
            "purpose": "development-validation",
            "required_status": "development-validated",
        },
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
    assert resolved.run_scale.epochs == 2
    assert resolved.run_scale.training_paths_per_epoch == 32
    assert resolved.architecture.widths == (64, 64, 32, 32)
    assert resolved.resources.wall_clock_budget_seconds == 1800

    overridden = configuration_data(tmp_path)
    overridden["convention"] = {
        "profile": "corrected",
        "pca_loading_scale": "eigenvalue",
    }

    custom = resolve_configuration(overridden)

    assert custom.convention.is_custom is True
    assert custom.convention.pca_loading_scale == "eigenvalue"
    assert custom.convention.loan_interest_annualization == "monthly"


def test_profiles_are_independent_and_local_plan_is_bounded(tmp_path: Path) -> None:
    configuration = configuration_data(tmp_path)
    configuration["policy"] = {"names": ["BM^E", "BM^C", "BM^D", "MM"]}
    resolved = resolve_configuration(configuration)

    plan = build_execution_plan(resolved)

    assert resolved.convention.is_custom is False
    assert resolved.architecture.profile == "compact"
    assert plan.primary_training_jobs == 8
    assert plan.primary_optimizer_updates == 64
    assert plan.paper_width_optimizer_updates == 2
    assert plan.mm_truncation_requires_training is False
    assert {job.horizon_years for job in plan.jobs} == {5, 15}
    assert all(job.training_paths_per_epoch == 32 for job in plan.jobs)
    assert all(job.optimizer_updates == 8 for job in plan.jobs)


def test_bank_training_requires_explicit_scale_and_bank_purpose(tmp_path: Path) -> None:
    configuration = configuration_data(tmp_path)
    configuration["run_scale"] = {"profile": "bank_training"}
    with pytest.raises(ConfigurationError, match="bank_training"):
        resolve_configuration(configuration)

    configuration["run_scale"] = {
        "profile": "bank_training",
        "epochs": 8,
        "training_paths_per_epoch": 64,
        "selection_paths": 32,
        "test_paths": 32,
        "batch_size": 8,
    }
    with pytest.raises(ConfigurationError, match="early stopping"):
        resolve_configuration(configuration)

    configuration["run_scale"].update(
        {
            "selection_start_epoch": 3,
            "early_stopping_patience": 4,
            "minimum_relative_improvement": 0.002,
        }
    )
    with pytest.raises(ConfigurationError, match="bank-training"):
        resolve_configuration(configuration)

    configuration["acceptance"] = {
        "purpose": "bank-training",
        "required_status": "development-validated",
    }
    resolved = resolve_configuration(configuration)
    assert resolved.run_scale.profile == "bank_training"
    assert resolved.run_scale.selection_start_epoch == 3
    assert resolved.run_scale.early_stopping_patience == 4
    assert resolved.run_scale.minimum_relative_improvement == 0.002
    assert resolved.acceptance.purpose == "bank-training"


def test_auto_device_is_resolved_once_before_manifest_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = configuration_data(tmp_path)
    configuration["optimization"] = {"device": "auto", "dtype": "float32"}
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    resolved = resolve_configuration(configuration)

    assert resolved.optimization.device == "cuda"
    assert resolved.to_dict()["optimization"]["device"] == "cuda"


def test_resource_monitor_raises_typed_budget_failure() -> None:
    clock_values = iter((10.0, 10.5))
    monitor = ResourceMonitor(
        wall_clock_budget_seconds=0.25,
        process_rss_limit_bytes=100,
        accelerator_memory_limit_bytes=100,
        clock=lambda: next(clock_values),
        rss_reader=lambda: None,
        accelerator_reader=lambda: None,
    )

    with pytest.raises(BudgetExceeded, match="wall-clock"):
        monitor.check("after-market-calibration")


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
        (
            lambda data: data["reference_bank"].update(
                {"initial_assets": {"value": 5_000, "unit": "mCHF"}}
            ),
            "10,000 mCHF",
        ),
        (
            lambda data: data["source_data"].update({"nss_beta_unit": "decimal"}),
            "percentage_points",
        ),
        (
            lambda data: data["seeds"].update({"market_scenarios": -1}),
            "non-negative integer",
        ),
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


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data["acceptance"].update(
            {"required_status": "methodologically-reproduced"}
        ),
        lambda data: (
            data["run_scale"].update({"profile": "paper_scale"}),
            data["acceptance"].update(
                {"required_status": "methodologically-reproduced"}
            ),
        ),
    ],
)
def test_configuration_rejects_incompatible_run_combinations(
    tmp_path: Path, change: object
) -> None:
    invalid = configuration_data(tmp_path)
    assert callable(change)
    change(invalid)

    with pytest.raises(ConfigurationError, match="methodologically-reproduced"):
        resolve_configuration(invalid)


@pytest.mark.parametrize(
    "convention",
    [
        {"profile": "paper"},
        {"profile": "corrected", "pca_loading_scale": "sqrt_eigenvalue"},
    ],
)
def test_methodological_reproduction_requires_the_locked_corrected_convention(
    tmp_path: Path, convention: dict[str, str]
) -> None:
    invalid = configuration_data(tmp_path)
    invalid["convention"] = convention
    invalid["run_scale"] = {"profile": "paper_scale"}
    invalid["policy"] = {"names": ["BM^E", "BM^C", "BM^D", "MM"]}
    invalid["acceptance"] = {
        "purpose": "research",
        "required_status": "methodologically-reproduced",
    }

    with pytest.raises(ConfigurationError, match="Corrected convention"):
        resolve_configuration(invalid)


def test_runner_writes_an_atomic_auditable_bundle(tmp_path: Path) -> None:
    configuration = configuration_data(tmp_path)
    write_source_inputs(configuration)
    resolved = resolve_configuration(configuration)

    bundle = ReproductionRunner().run(resolved)

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.acceptance_status is AcceptanceStatus.PENDING
    assert bundle.artifact_directory is not None
    assert bundle.artifact_directory.is_dir()
    assert not list(bundle.artifact_directory.parent.glob(".smoke-*"))

    manifest = json.loads((bundle.artifact_directory / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["acceptance_status"] == "pending"
    assert manifest["resolved_configuration"]["convention"]["profile"] == "corrected"
    assert manifest["seed_registry"] == {
        "bootstrap": 15,
        "data_loader_order": 14,
        "market_scenarios": 11,
        "model_initialization": 13,
        "objective_parameters": 12,
        "sensitivity": 16,
    }
    expected_configuration_hash = hashlib.sha256(
        json.dumps(
            manifest["resolved_configuration"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    assert manifest["resolved_configuration_hash"] == expected_configuration_hash
    assert (
        manifest["input_hashes"]["snb_csv"]
        == hashlib.sha256(b"source-data").hexdigest()
    )
    assert (
        manifest["input_hashes"]["paper_pdf"]
        == hashlib.sha256(b"paper-data").hexdigest()
    )
    assert manifest["runtime"]["python"]
    assert set(manifest["runtime"]["dependencies"]) == {
        "matplotlib",
        "numpy",
        "pandas",
        "pyyaml",
        "scikit-learn",
        "scipy",
        "torch",
        "tqdm",
    }
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


def test_runner_preserves_diagnostics_for_a_failed_run(tmp_path: Path) -> None:
    configuration = configuration_data(tmp_path)

    bundle = ReproductionRunner().run(resolve_configuration(configuration))

    assert bundle.status is RunStatus.FAILED
    assert bundle.artifact_directory is not None
    assert bundle.artifact_directory.is_dir()
    manifest = json.loads((bundle.artifact_directory / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]
    assert manifest["input_hashes"] == {"paper_pdf": None, "snb_csv": None}
    assert set(manifest["input_hash_errors"]) == {"paper_pdf", "snb_csv"}
    assert bundle.metrics == {}
    assert bundle.checkpoints == ()
    assert bundle.acceptance_evidence == ()


def test_runner_writes_an_acceptance_failed_bundle(tmp_path: Path) -> None:
    configuration = configuration_data(tmp_path)
    write_source_inputs(configuration)

    bundle = ReproductionRunner().run(
        resolve_configuration(configuration), acceptance_status=AcceptanceStatus.FAILED
    )

    assert bundle.status is RunStatus.ACCEPTANCE_FAILED
    assert bundle.acceptance_status is AcceptanceStatus.FAILED
    assert bundle.artifact_directory is not None
    manifest = json.loads((bundle.artifact_directory / "manifest.json").read_text())
    assert manifest["status"] == "acceptance_failed"
    assert manifest["acceptance_status"] == "failed"


def test_market_preflight_generates_bounded_hjm_batches_and_keeps_acceptance_pending(
    tmp_path: Path,
) -> None:
    configuration = configuration_data(tmp_path)
    write_source_inputs(configuration)
    calls: list[dict[str, object]] = []

    class FakeMarketScenarioModel:
        def load_historical_term_structures(self, *args: object, **kwargs: object) -> object:
            return SimpleNamespace(source_hash="market-source", round_trip_error=0.0)

        def calibrate_hjm_pca(self, historical: object) -> object:
            assert historical
            return SimpleNamespace(
                calibration_identity="calibration", explained_variance=np.array([0.9, 0.05, 0.03])
            )

        def generate_hjm_scenarios(self, *args: object, **kwargs: object) -> object:
            calls.append(dict(kwargs))
            return SimpleNamespace(
                split=kwargs["split"],
                epoch=kwargs["epoch"],
                global_path_indices=kwargs["global_path_indices"],
                round_trip_error=0.0,
            )

    bundle = ReproductionRunner().preflight_market(
        resolve_configuration(configuration), market_model=FakeMarketScenarioModel()
    )

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.acceptance_status is AcceptanceStatus.PENDING
    assert len(calls) == 8
    assert all(call["paths"] == 8 for call in calls)
    manifest = json.loads((bundle.artifact_directory / "manifest.json").read_text())
    assert manifest["market_preflight"]["completed_stages"] == [
        "historical-term-structures",
        "hjm-pca-calibration",
        "bounded-hjm-scenarios",
    ]
    assert manifest["execution_plan"]["estimated_cost_status"] == "unmeasured"


def test_market_preflight_preserves_evidence_when_budget_is_exhausted(
    tmp_path: Path,
) -> None:
    configuration = configuration_data(tmp_path)
    write_source_inputs(configuration)
    clock_values = iter((0.0, 0.0, 2.0, 2.0))
    monitor = ResourceMonitor(
        wall_clock_budget_seconds=1.0,
        process_rss_limit_bytes=4 * 1024**3,
        accelerator_memory_limit_bytes=4 * 1024**3,
        clock=lambda: next(clock_values),
        rss_reader=lambda: None,
        accelerator_reader=lambda: None,
    )

    class FakeMarketScenarioModel:
        def load_historical_term_structures(self, *args: object, **kwargs: object) -> object:
            return SimpleNamespace(source_hash="market-source", round_trip_error=0.0)

        def calibrate_hjm_pca(self, historical: object) -> object:
            raise AssertionError("budget should be checked before calibration")

    bundle = ReproductionRunner().preflight_market(
        resolve_configuration(configuration),
        resource_monitor=monitor,
        market_model=FakeMarketScenarioModel(),
    )

    assert bundle.status is RunStatus.INCOMPLETE
    assert bundle.artifact_directory is not None
    manifest = json.loads((bundle.artifact_directory / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["diagnostics"]["reason"] == "budget_exhausted"
    assert manifest["diagnostics"]["market_preflight"]["completed_stages"] == [
        "historical-term-structures"
    ]


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


def test_cli_dispatches_a_successful_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configuration = configuration_data(tmp_path)
    write_source_inputs(configuration)
    configuration_path = tmp_path / "valid.yaml"
    import yaml

    configuration_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")

    exit_code = main(["run", "--config", str(configuration_path)])

    artifact_directory = Path(capsys.readouterr().out.strip())
    assert exit_code == 0
    assert artifact_directory.is_dir()


def test_cli_prints_a_plan_without_starting_a_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configuration_path = tmp_path / "plan.yaml"
    import yaml

    configuration_path.write_text(
        yaml.safe_dump(configuration_data(tmp_path)), encoding="utf-8"
    )

    exit_code = main(["plan", "--config", str(configuration_path)])

    plan = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert plan["run_scale"] == "local_flow"
    assert plan["primary_optimizer_updates"] == 16


def test_cli_returns_an_operational_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configuration = configuration_data(tmp_path)
    write_source_inputs(configuration)
    output_path = tmp_path / "not-a-directory"
    output_path.write_text("file", encoding="utf-8")
    configuration["output"] = {"directory": str(output_path), "run_name": "smoke"}
    configuration_path = tmp_path / "failure.yaml"
    import yaml

    configuration_path.write_text(yaml.safe_dump(configuration), encoding="utf-8")

    exit_code = main(["run", "--config", str(configuration_path)])

    assert exit_code == 1
    assert "Operational failure" in capsys.readouterr().err
