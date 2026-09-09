"""Public configuration and runner coverage for the opt-in paired pilot."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from test_run_skeleton import configuration_data

from deepalm.cli import main
from deepalm.config import ConfigurationError, resolve_configuration
from deepalm.planning import build_execution_plan
from deepalm.runner import (
    AcceptanceStatus,
    ReproductionRunner,
    RunBundle,
    RunStatus,
    _paired_convention_intervals,
)


def _paired_pilot_data(tmp_path: Path) -> dict[str, object]:
    data = configuration_data(tmp_path)
    data["run_scale"] = {"profile": "paired_convention_pilot"}
    data["policy"] = {"names": ["BM^D", "MM"]}
    data["optimization"] = {"device": "mps", "dtype": "float32"}
    data["resources"] = {
        "wall_clock_budget_seconds": 600,
        "process_rss_limit_bytes": 12 * 1024**3,
        "accelerator_memory_limit_bytes": 12 * 1024**3,
    }
    data["acceptance"] = {
        "purpose": "research",
        "required_status": "paired-convention-research-pilot",
    }
    return data


def test_paired_pilot_config_writes_distinct_repair_evidence() -> None:
    repository = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load(
        (repository / "configs/paired-convention-pilot.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert raw["output"]["run_name"] == "paired-convention-pilot-financial-corrections"


def test_paired_convention_pilot_resolves_the_locked_m5_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    configuration = resolve_configuration(_paired_pilot_data(tmp_path))

    assert configuration.run_scale.profile == "paired_convention_pilot"
    assert configuration.run_scale.epochs == 2
    assert configuration.run_scale.training_paths_per_epoch == 16
    assert configuration.run_scale.selection_paths == 16
    assert configuration.run_scale.test_paths == 64
    assert configuration.run_scale.batch_size == 8
    assert configuration.optimization.device == "mps"
    assert configuration.optimization.dtype == "float32"
    plan = build_execution_plan(configuration)
    assert plan.primary_training_jobs == 8
    assert plan.primary_optimizer_updates == 32
    assert plan.paper_width_optimizer_updates == 0


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda data: data["policy"].update({"names": ["BM^D"]}),
            r"BM\^D and MM",
        ),
        (
            lambda data: data["optimization"].update({"device": "cpu", "dtype": "float64"}),
            "MPS float32",
        ),
        (
            lambda data: data["resources"].update({"wall_clock_budget_seconds": 601}),
            "600-second",
        ),
        (
            lambda data: data["experiment"].update({"include_swaps": True}),
            "include_swaps",
        ),
    ),
)
def test_paired_convention_pilot_rejects_unpaired_or_unbounded_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: object,
    message: str,
) -> None:
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    data = _paired_pilot_data(tmp_path)
    assert callable(mutate)
    mutate(data)

    with pytest.raises(ConfigurationError, match=message):
        resolve_configuration(data)


def test_paired_pilot_command_dispatches_to_the_public_runner_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    data = _paired_pilot_data(tmp_path)
    configuration = resolve_configuration(data)
    config_path = tmp_path / "paired-pilot.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    artifact_directory = tmp_path / "pilot"

    def fake_pilot(self: ReproductionRunner, received: object) -> RunBundle:
        assert received == configuration
        return RunBundle(
            status=RunStatus.COMPLETED,
            acceptance_status=AcceptanceStatus.PENDING,
            artifact_directory=artifact_directory,
        )

    monkeypatch.setattr(ReproductionRunner, "run_paired_convention_pilot", fake_pilot)

    assert main(["paired-pilot", "--config", str(config_path)]) == 0
    assert capsys.readouterr().out.strip() == str(artifact_directory)


def test_paired_evaluation_command_dispatches_and_keeps_nonfinite_pairs_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    data = _paired_pilot_data(tmp_path)
    config_path = tmp_path / "paired-pilot.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    source = tmp_path / "pilot"
    artifact_directory = tmp_path / "evaluation"

    def fake_evaluation(
        self: ReproductionRunner, received: object, *, source_run_directory: Path
    ) -> RunBundle:
        assert received == resolve_configuration(data)
        assert source_run_directory == source
        return RunBundle(
            status=RunStatus.COMPLETED,
            acceptance_status=AcceptanceStatus.PAIRED_CONVENTION_RESEARCH_PILOT,
            artifact_directory=artifact_directory,
        )

    monkeypatch.setattr(
        ReproductionRunner, "evaluate_paired_convention_pilot", fake_evaluation
    )
    assert main(
        ["paired-evaluate", "--config", str(config_path), "--source-run", str(source)]
    ) == 0
    assert capsys.readouterr().out.strip() == str(artifact_directory)

    intervals = _paired_convention_intervals(
        {"MM-5y": {"total_loss": torch.tensor([1.0, 2.0])}},
        {"MM-5y": {"total_loss": torch.tensor([0.0, 1.0])}},
        seed=8,
    )
    total_loss = next(item for item in intervals if item["metric"] == "total_loss")
    annualized = next(item for item in intervals if item["metric"] == "annualized_return")
    assert total_loss["status"] == "available"
    assert total_loss["paths"] == 2
    assert annualized["status"] == "not-applicable"


@pytest.mark.parametrize(
    ("probe_seconds", "expected_status"),
    ((120.0, RunStatus.COMPLETED), (200.0, RunStatus.INCOMPLETE)),
    ids=("within-budget", "projected-over-budget"),
)
def test_paired_pilot_runs_or_bounds_the_isolated_training_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe_seconds: float,
    expected_status: RunStatus,
) -> None:
    """The public runner preserves the paired matrix without real MPS work."""

    import deepalm.baselines as baselines_module
    import deepalm.reference_bank as reference_bank_module
    import deepalm.term_structures as term_structures_module
    import deepalm.training as training_module
    from deepalm.training import TrainingInterrupted

    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    configuration = resolve_configuration(_paired_pilot_data(tmp_path))
    configuration.source_data.snb_csv.write_text("snb", encoding="utf-8")
    configuration.source_data.paper_pdf.write_bytes(b"paper")
    mm_baselines: list[tuple[str, Path]] = []

    class FakeSnapshot:
        def __init__(self, elapsed_seconds: float) -> None:
            self.elapsed_seconds = elapsed_seconds

        def to_dict(self) -> dict[str, object]:
            return {"elapsed_seconds": self.elapsed_seconds, "peak_rss_bytes": 0}

    class FakeMonitor:
        def __init__(self, **_: object) -> None:
            self.snapshot_count = 0

        def check(self, _: str) -> None:
            pass

        def snapshot(self) -> FakeSnapshot:
            elapsed_seconds = 0.0 if self.snapshot_count == 0 else probe_seconds
            self.snapshot_count += 1
            return FakeSnapshot(elapsed_seconds)

    class FakeMarketScenarioModel:
        def load_historical_term_structures(self, *_: object, **__: object) -> object:
            return SimpleNamespace(source_hash="market-source")

        def calibrate_hjm_pca(self, _: object) -> object:
            return SimpleNamespace(calibration_identity="calibration")

    class FakeReferenceBankProvider:
        def build_canonical(self, _: object) -> object:
            return SimpleNamespace(content_hash="reference-bank")

        def save(self, _: object, path: Path) -> None:
            path.write_text("reference-bank", encoding="utf-8")

    class FakeFrozenBaseline:
        @staticmethod
        def load(path: Path) -> dict[str, Path]:
            return {"path": path}

    def result(configuration: object, policy: str, horizon_years: int) -> object:
        output = configuration.output.directory / configuration.output.run_name
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / f"{policy}_{horizon_years}y.pt"
        checkpoint.write_text(f"{policy}-{horizon_years}", encoding="utf-8")
        baseline = output / f"BM_D_{horizon_years}y.baseline.json"
        if policy == "BM_D":
            baseline.write_text("baseline", encoding="utf-8")
        return SimpleNamespace(
            checkpoint_path=checkpoint,
            baseline_reference_path=baseline if policy == "BM_D" else None,
            baseline_reference_identity=f"corrected-financial-semantics-{horizon_years}",
            optimizer_updates=4,
            selected_epoch=2,
            clipped_gradient_norms=(0.2,),
        )

    class FakeBMDateTrainer:
        def __init__(self, configuration: object, **_: object) -> None:
            self.configuration = configuration

        def fit(self, *, horizon_years: int, **_: object) -> object:
            return result(self.configuration, "BM_D", horizon_years)

    class FakeMMTrainer:
        def __init__(
            self, configuration: object, *, baseline_reference: dict[str, Path], **_: object
        ) -> None:
            self.configuration = configuration
            self.baseline_reference = baseline_reference
            mm_baselines.append(
                ("corrected", baseline_reference["path"])
            )

        def fit(self, *, horizon_years: int, control: object | None = None) -> object:
            if horizon_years == 15 and control is not None and not control.resume:
                recovery = (
                    self.configuration.output.directory
                    / self.configuration.output.run_name
                    / "MM_15y.recovery.pt"
                )
                recovery.write_text("recovery", encoding="utf-8")
                interruption = recovery.with_suffix(".interruption.json")
                interruption.write_text("interruption", encoding="utf-8")
                raise TrainingInterrupted(
                    "requested_stop",
                    recovery_path=recovery,
                    interruption_path=interruption,
                    diagnostics={"completed_epoch": 1},
                )
            return result(self.configuration, "MM", horizon_years)

    monkeypatch.setattr("deepalm.runner.ResourceMonitor", FakeMonitor)
    monkeypatch.setattr(term_structures_module, "MarketScenarioModel", FakeMarketScenarioModel)
    monkeypatch.setattr(reference_bank_module, "ReferenceBankProvider", FakeReferenceBankProvider)
    monkeypatch.setattr(baselines_module, "FrozenDateBenchmarkReference", FakeFrozenBaseline)
    monkeypatch.setattr(training_module, "BMDateTrainer", FakeBMDateTrainer)
    monkeypatch.setattr(training_module, "MMTrainer", FakeMMTrainer)

    bundle = ReproductionRunner().run_paired_convention_pilot(configuration)

    assert bundle.status is expected_status
    assert bundle.artifact_directory is not None
    if expected_status is RunStatus.INCOMPLETE:
        manifest = json.loads((bundle.artifact_directory / "manifest.json").read_text())
        assert manifest["diagnostics"]["reason"] == "projected_training_budget_exceeded"
        assert (bundle.artifact_directory / "pilot-throughput-probe.json").is_file()
        return

    assert bundle.acceptance_status is AcceptanceStatus.PAIRED_CONVENTION_RESEARCH_PILOT
    manifest = json.loads((bundle.artifact_directory / "manifest.json").read_text())
    pilot = manifest["paired_convention_pilot"]
    assert pilot["completed_primary_optimizer_updates"] == 32
    assert len(pilot["completed_training_jobs"]) == 8
    assert {job["convention"] for job in pilot["completed_training_jobs"]} == {
        "corrected",
        "paper",
    }
    assert all(not Path(job["checkpoint"]).is_absolute() for job in pilot["completed_training_jobs"])
    assert pilot["mm_fifteen_year_truncation"]["corrected"]["optimizer_updates"] == 0
    assert {path.parent.name for _, path in mm_baselines} == {"corrected", "paper"}
    assert len({path for _, path in mm_baselines}) == 4


@pytest.mark.parametrize(
    ("configuration_change", "message"),
    (
        (
            lambda configuration: replace(
                configuration,
                experiment=replace(configuration.experiment, include_swaps=True),
            ),
            "excludes swaps",
        ),
        (lambda configuration: configuration, "available MPS runtime"),
    ),
)
def test_paired_pilot_preserves_configuration_or_mps_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configuration_change: object,
    message: str,
) -> None:
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    configuration = resolve_configuration(_paired_pilot_data(tmp_path))
    configuration.source_data.snb_csv.write_text("snb", encoding="utf-8")
    configuration.source_data.paper_pdf.write_bytes(b"paper")
    assert callable(configuration_change)
    changed = configuration_change(configuration)
    if changed is configuration:
        monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)

    bundle = ReproductionRunner().run_paired_convention_pilot(changed)

    assert bundle.status is RunStatus.FAILED
    assert message in (bundle.error or "")
    assert bundle.artifact_directory is not None
    manifest = json.loads((bundle.artifact_directory / "manifest.json").read_text())
    assert manifest["status"] == RunStatus.FAILED.value
