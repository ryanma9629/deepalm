"""Public corrected-pilot configuration and CLI seam coverage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_run_skeleton import configuration_data

from deepalm.cli import main
from deepalm.config import ConfigurationError, resolve_configuration
from deepalm.planning import build_execution_plan
from deepalm.runner import AcceptanceStatus, ReproductionRunner, RunStatus


def _corrected_pilot_data(tmp_path: Path) -> dict[str, object]:
    data = configuration_data(tmp_path)
    data["workflow_contract"] = {"name": "local-two-policy-validation"}
    data["execution_profile"] = {"name": "m5-compact"}
    data["run_scale"] = {"profile": "corrected_pilot"}
    data["policy"] = {"names": ["BM^D", "MM"]}
    data["optimization"] = {"device": "mps", "dtype": "float32"}
    data["resources"] = {
        "wall_clock_budget_seconds": 600,
        "process_rss_limit_bytes": 12 * 1024**3,
        "accelerator_memory_limit_bytes": 12 * 1024**3,
    }
    data["acceptance"] = {
        "purpose": "development-validation",
        "required_status": "development-validated",
    }
    return data


def test_corrected_pilot_has_the_locked_four_member_local_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    configuration = resolve_configuration(_corrected_pilot_data(tmp_path))

    plan = build_execution_plan(configuration)
    assert configuration.run_scale.profile == "corrected_pilot"
    assert configuration.run_scale.epochs == 2
    assert configuration.run_scale.training_paths_per_epoch == 16
    assert configuration.run_scale.selection_paths == 16
    assert configuration.run_scale.batch_size == 8
    assert plan.primary_training_jobs == 4
    assert plan.primary_optimizer_updates == 16


def test_versioned_corrected_pilot_config_exposes_no_paired_profile() -> None:
    repository = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load(
        (repository / "configs/local-two-policy-m5.yaml").read_text(encoding="utf-8")
    )

    assert raw["run_scale"] == {"profile": "corrected_pilot"}
    assert raw["output"]["run_name"] == "corrected-local-validation-pilot"
    assert not (repository / "configs/paired-convention-pilot.yaml").exists()


def test_configuration_rejects_the_retired_paired_pilot_profile(tmp_path: Path) -> None:
    retired = _corrected_pilot_data(tmp_path)
    retired["run_scale"] = {"profile": "paired_convention_pilot"}

    with pytest.raises(ConfigurationError, match="corrected_pilot"):
        resolve_configuration(retired)


def test_paired_pilot_is_not_a_public_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["paired-pilot"])

    assert stopped.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert not hasattr(ReproductionRunner, "run_paired_convention_pilot")


def test_corrected_pilot_publishes_four_updated_members_and_matching_baselines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deepalm.reference_bank as reference_bank_module
    import deepalm.runoff as runoff_module
    import deepalm.training as training_module

    assert runoff_module.ReferenceBankProvider is reference_bank_module.ReferenceBankProvider
    assert training_module.FrozenDateBenchmarkReference is not None

    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    configuration = resolve_configuration(_corrected_pilot_data(tmp_path))
    configuration.source_data.snb_csv.write_text("snb", encoding="utf-8")
    configuration.source_data.paper_pdf.write_bytes(b"paper")
    mm_baselines: dict[int, Path] = {}

    class FakeSnapshot:
        content_hash = "reference-bank"

    class FakeMonitor:
        def __init__(self, **_: object) -> None:
            pass

        def check(self, _: str) -> None:
            pass

        def snapshot(self) -> object:
            return SimpleNamespace(to_dict=lambda: {"elapsed_seconds": 1.0})

    class FakeMarketScenarioModel:
        def load_historical_term_structures(self, *_: object, **__: object) -> object:
            return SimpleNamespace(source_hash="market-source")

        def calibrate_hjm_pca(self, _: object) -> object:
            return SimpleNamespace(calibration_identity="calibration")

    class FakeReferenceBankProvider:
        def build_canonical(self, _: object) -> FakeSnapshot:
            return FakeSnapshot()

        def save(self, _: object, path: Path) -> None:
            path.write_text("reference-bank", encoding="utf-8")

    class FakeFrozenBaseline:
        @staticmethod
        def load(path: Path) -> object:
            return SimpleNamespace(reference_path=path)

    def training_result(configuration: object, policy: str, horizon_years: int) -> object:
        output = configuration.output.directory / configuration.output.run_name
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / f"{policy}_{horizon_years}y.pt"
        checkpoint.write_text(f"{policy}-{horizon_years}", encoding="utf-8")
        baseline = output / f"BM_D_{horizon_years}y.baseline.json"
        if policy == "BM^D":
            baseline.write_text("baseline", encoding="utf-8")
        return SimpleNamespace(
            checkpoint_path=checkpoint,
            baseline_reference_path=baseline,
            baseline_reference_identity=f"baseline-{horizon_years}",
            optimizer_updates=4,
            selected_epoch=2,
            clipped_gradient_norms=(0.1, 0.1, 0.1, 0.1),
        )

    class FakeBMDateTrainer:
        def __init__(self, configuration: object, **_: object) -> None:
            self.configuration = configuration

        def fit(
            self, *, horizon_years: int, progress_callback: object = None
        ) -> object:
            del progress_callback
            return training_result(self.configuration, "BM^D", horizon_years)

    class FakeMMTrainer:
        def __init__(
            self, configuration: object, *, baseline_reference: object, **_: object
        ) -> None:
            self.configuration = configuration
            self.baseline_reference = baseline_reference

        def fit(
            self, *, horizon_years: int, progress_callback: object = None
        ) -> object:
            del progress_callback
            mm_baselines[horizon_years] = self.baseline_reference.reference_path
            return training_result(self.configuration, "MM", horizon_years)

    monkeypatch.setattr("deepalm.runner.ResourceMonitor", FakeMonitor)
    monkeypatch.setattr(
        "deepalm.term_structures.MarketScenarioModel", FakeMarketScenarioModel
    )
    monkeypatch.setattr(
        reference_bank_module, "ReferenceBankProvider", FakeReferenceBankProvider
    )
    monkeypatch.setattr(
        "deepalm.baselines.FrozenDateBenchmarkReference", FakeFrozenBaseline
    )
    monkeypatch.setattr(training_module, "BMDateTrainer", FakeBMDateTrainer)
    monkeypatch.setattr(training_module, "MMTrainer", FakeMMTrainer)

    bundle = ReproductionRunner().run_configured_workflow(configuration)

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.acceptance_status is AcceptanceStatus.PENDING
    assert bundle.artifact_directory is not None
    manifest = json.loads((bundle.artifact_directory / "manifest.json").read_text())
    pilot = manifest["corrected_local_validation_pilot"]
    assert pilot["completed_primary_optimizer_updates"] == 16
    assert {
        (job["policy"], job["horizon_years"])
        for job in pilot["completed_training_jobs"]
    } == {("BM^D", 5), ("BM^D", 15), ("MM", 5), ("MM", 15)}
    assert all(job["finite_nonzero_optimization_signal"] for job in pilot["completed_training_jobs"])
    jobs = {
        (job["policy"], job["horizon_years"]): job
        for job in pilot["completed_training_jobs"]
    }
    for horizon_years in (5, 15):
        assert mm_baselines[horizon_years].name == f"BM_D_{horizon_years}y.baseline.json"
        assert jobs[("MM", horizon_years)]["baseline_reference"] == jobs[
            ("BM^D", horizon_years)
        ]["baseline_reference"]

    sentinel = bundle.artifact_directory / "old-run-sentinel.txt"
    sentinel.write_text("preserve until explicit overwrite", encoding="utf-8")
    rejected = ReproductionRunner().run_configured_workflow(configuration)
    assert rejected.status is RunStatus.FAILED
    assert sentinel.exists()

    replaced = ReproductionRunner().run_configured_workflow(
        configuration, overwrite=True
    )
    assert replaced.status is RunStatus.COMPLETED
    assert replaced.artifact_directory == bundle.artifact_directory
    assert not sentinel.exists()
