from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from test_run_skeleton import configuration_data

from deepalm.baselines import BaselineReferenceError, FrozenDateBenchmarkReference
from deepalm.cli import main
from deepalm.config import (
    ArchitectureConfiguration,
    ExperimentConfiguration,
    PolicyConfiguration,
    RunScaleConfiguration,
    resolve_configuration,
)
from deepalm.reference_bank import ReferenceBankProvider
from deepalm.resources import BudgetExceeded, ResourceSnapshot
from deepalm.runner import ReproductionRunner, RunStatus
from deepalm.term_structures import MarketScenarioModel
from deepalm.training import (
    BMConstantTrainer,
    BMDateTrainer,
    BMETrainer,
    SelectionSchedule,
    SelectionTracker,
    TrainingControl,
    TrainingError,
    TrainingInterrupted,
)

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
)


def _configuration(tmp_path: Path):
    resolved = resolve_configuration(
        {
            "source_data": {
                "snb_csv": str(SOURCE),
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
            "experiment": {"horizons_years": [5], "include_swaps": False},
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
            "output": {"directory": str(tmp_path / "runs"), "run_name": "bme-test"},
            "acceptance": {
                "purpose": "development-validation",
                "required_status": "development-validated",
            },
        }
    )
    return replace(
        resolved,
        run_scale=RunScaleConfiguration(
            profile="test",
            epochs=1,
            training_paths_per_epoch=2,
            selection_paths=2,
            test_paths=2,
            batch_size=2,
        ),
    )


def test_bme_trainer_updates_policy_selects_checkpoint_and_records_resources(
    tmp_path: Path,
) -> None:
    configuration = _configuration(tmp_path)
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)

    configuration = replace(
        configuration,
        experiment=ExperimentConfiguration(horizons_years=(5, 15), include_swaps=False),
    )
    results = BMETrainer(
        configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
    ).fit_all()
    result = results[0]
    checkpoint = torch.load(result.checkpoint_path, weights_only=False)

    assert result.optimizer_updates == 1
    assert result.selected_epoch == 1
    assert len(result.selection_history) == 1
    assert result.checkpoint_path.is_file()
    assert [item.resource_profile.horizon_years for item in results] == [5, 15]
    assert checkpoint["policy"] == "BM^E"
    assert checkpoint["horizon_years"] == 5
    assert checkpoint["code_identity"]["git_revision"]
    assert checkpoint["code_identity"]["checkpoint_schema_version"] == 1
    assert checkpoint["scenario_identities"]["training"]["split"] == "training"
    assert checkpoint["scenario_identities"]["selection"]["split"] == "selection"
    assert checkpoint["data_identities"]["market_source_hash"] == historical.source_hash
    assert (
        checkpoint["data_identities"]["hjm_calibration_identity"]
        == calibration.calibration_identity
    )
    assert (
        checkpoint["data_identities"]["reference_bank_content_hash"]
        == snapshot.content_hash
    )
    assert checkpoint["resource_profile"]["peak_process_rss_bytes"] is not None
    assert (
        result.checkpoint_path.parent / checkpoint["resource_profile_path"]
    ).is_file()
    assert all(norm <= 0.20001 for norm in result.clipped_gradient_norms)
    assert result.resource_profile.forward_backward_update_seconds >= 0.0
    assert result.resource_profile.scenario_seconds >= 0.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda checkpoint: checkpoint["configuration"]["convention"].update(
                {"profile": "paper"}
            ),
            "convention",
        ),
        (
            lambda checkpoint: checkpoint["configuration"]["architecture"].update(
                {"widths": [1, 1, 1, 1]}
            ),
            "architecture",
        ),
        (
            lambda checkpoint: checkpoint.update({"horizon_years": 15}),
            "horizon",
        ),
        (
            lambda checkpoint: checkpoint["configuration"]["source_data"].update(
                {"nss_beta_unit": "decimal"}
            ),
            "preprocessing",
        ),
        (
            lambda checkpoint: checkpoint["data_identities"].update(
                {"market_source_hash": "wrong-market"}
            ),
            "data identities",
        ),
    ),
)
def test_bme_checkpoint_loader_rejects_incompatible_semantic_contract(
    tmp_path: Path, mutate, message: str
) -> None:
    configuration = _configuration(tmp_path)
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    trainer = BMETrainer(
        configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
    )
    result = trainer.fit(horizon_years=5)
    checkpoint = torch.load(result.checkpoint_path, weights_only=False)
    mutate(checkpoint)
    incompatible_path = tmp_path / f"incompatible-{message}.pt"
    torch.save(checkpoint, incompatible_path)

    with pytest.raises(TrainingError, match=message):
        trainer.load_selected_checkpoint(incompatible_path)


@pytest.mark.parametrize("horizon_years", [5, 15])
def test_bme_device_validation_records_cpu_mps_and_cuda_truthfully(
    tmp_path: Path, horizon_years: int
) -> None:
    configuration = replace(
        _configuration(tmp_path),
        experiment=ExperimentConfiguration(horizons_years=(5, 15), include_swaps=False),
    )
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)

    records = BMETrainer(
        configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
    ).validate_devices(horizon_years=horizon_years)

    records_by_device = {record.device: record for record in records}
    assert set(records_by_device) == {"cpu", "mps", "cuda"}
    cpu = records_by_device["cpu"]
    assert cpu.status == "completed"
    assert cpu.finite and cpu.updated and cpu.clipped
    assert cpu.checkpoint_path is not None
    assert cpu.checkpoint_loaded and cpu.resumed_from_cpu_recovery
    assert cpu.recovery_path is not None and cpu.recovery_path.is_file()
    cpu_checkpoint = torch.load(cpu.checkpoint_path, weights_only=False)
    assert cpu_checkpoint["configuration"]["optimization"]["dtype"] == "float32"

    mps = records_by_device["mps"]
    if torch.backends.mps.is_available():
        assert mps.status == "completed"
        assert mps.finite and mps.updated and mps.clipped
        assert mps.checkpoint_path is not None
        assert mps.checkpoint_loaded and mps.resumed_from_cpu_recovery
    else:
        assert mps.status == "not-run"
        assert mps.checkpoint_path is None
        assert mps.reason == "PyTorch MPS runtime is not available"

    cuda = records_by_device["cuda"]
    if torch.cuda.is_available():
        assert cuda.status == "completed"
        assert cuda.finite and cuda.updated and cuda.clipped
        assert cuda.checkpoint_path is not None
        assert cuda.checkpoint_loaded and cuda.resumed_from_cpu_recovery
    else:
        assert cuda.status == "not-run"
        assert cuda.checkpoint_path is None
        assert cuda.reason == "PyTorch CUDA runtime is not available"


def test_runner_persists_actual_single_device_validation_evidence(
    tmp_path: Path,
) -> None:
    configuration = _configuration(tmp_path)
    configuration.source_data.paper_pdf.write_bytes(b"paper fixture")
    configuration = replace(
        configuration,
        output=replace(configuration.output, run_name="single-device-portability"),
    )
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    trainer = BMETrainer(
        configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
    )

    bundle = ReproductionRunner().validate_single_device_portability(
        configuration, trainer=trainer, horizon_years=5
    )

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.artifact_directory is not None
    evidence = json.loads(
        (bundle.artifact_directory / "single-device-validation.json").read_text()
    )
    records = {record["device"]: record for record in evidence["records"]}
    assert records["cpu"]["status"] == "completed"
    assert records["cpu"]["checkpoint_path"] is not None
    assert records["cpu"]["checkpoint_loaded"] is True
    assert records["cpu"]["resumed_from_cpu_recovery"] is True
    assert records["cuda"]["status"] == (
        "completed" if torch.cuda.is_available() else "not-run"
    )
    if not torch.cuda.is_available():
        assert records["cuda"]["reason"] == "PyTorch CUDA runtime is not available"
    assert evidence["cross_device_bitwise_equality"] == "not-promised"


@pytest.mark.parametrize(
    ("policy_name", "horizon_years"),
    (("BM^E", 5), ("BM^E", 15), ("MM", 5), ("MM", 15)),
)
def test_cli_runs_the_bounded_single_device_commissioning_check(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    policy_name: str,
    horizon_years: int,
) -> None:
    configuration = configuration_data(tmp_path)
    source_data = configuration["source_data"]
    assert isinstance(source_data, dict)
    source_data["snb_csv"] = str(SOURCE)
    Path(str(source_data["paper_pdf"])).write_bytes(b"paper fixture")
    if policy_name == "MM":
        configuration["policy"] = {"names": ["BM^D", "MM"]}
    configuration["output"] = {
        "directory": str(tmp_path / "runs"),
        "run_name": f"cli-device-check-{policy_name}-{horizon_years}",
    }
    configuration_path = tmp_path / "device-check.yaml"
    configuration_path.write_text(
        yaml.safe_dump(configuration), encoding="utf-8"
    )

    exit_code = main(
        [
            "device-check",
            "--config",
            str(configuration_path),
            "--horizon",
            str(horizon_years),
            "--policy",
            policy_name,
        ]
    )

    captured = capsys.readouterr()
    artifact_directory = Path(captured.out.strip())
    assert exit_code == 0, captured.err
    evidence = json.loads(
        (artifact_directory / "single-device-validation.json").read_text()
    )
    assert evidence["horizon_years"] == horizon_years
    assert evidence["policy"] == policy_name


def test_constant_benchmark_uses_the_shared_trainer_for_both_horizons(
    tmp_path: Path,
) -> None:
    configuration = replace(
        _configuration(tmp_path),
        experiment=ExperimentConfiguration(horizons_years=(5, 15), include_swaps=False),
        policy=PolicyConfiguration(names=("BM^C",)),
    )
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)

    results = BMConstantTrainer(
        configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
    ).fit_all()
    checkpoint = torch.load(results[0].checkpoint_path, weights_only=False)

    assert [result.resource_profile.horizon_years for result in results] == [5, 15]
    assert all(result.optimizer_updates == 1 for result in results)
    assert checkpoint["policy"] == "BM^C"
    action_record = checkpoint["selection_history"][0]["action_record"]
    assert len(action_record["investment_maturity_totals"]) == 13
    assert len(action_record["funding_maturity_totals"]) == 16
    assert action_record["minimum_action"] >= 0.0


def test_date_benchmark_freezes_each_selected_horizon_with_verified_identity(
    tmp_path: Path,
) -> None:
    configuration = replace(
        _configuration(tmp_path),
        experiment=ExperimentConfiguration(horizons_years=(5, 15), include_swaps=False),
        policy=PolicyConfiguration(names=("BM^D",)),
    )
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)

    results = BMDateTrainer(
        configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
    ).fit_all()
    first = results[0]
    checkpoint = torch.load(first.checkpoint_path, weights_only=False)

    assert [result.resource_profile.horizon_years for result in results] == [5, 15]
    assert all(result.optimizer_updates == 1 for result in results)
    assert all(result.baseline_reference_path is not None for result in results)
    assert checkpoint["policy"] == "BM^D"
    assert checkpoint["policy_metadata"]["decision_dates"] == 60
    assert checkpoint["policy_metadata"]["parameter_count"] == 1_860
    assert "T action dates" in checkpoint["policy_metadata"]["indexing_note"]

    assert first.baseline_reference_identity is not None
    reference = FrozenDateBenchmarkReference.load(first.baseline_reference_path)
    assert reference.reference_identity == first.baseline_reference_identity
    frozen_policy = reference.load_policy(device="cpu", dtype=torch.float64)
    assert all(not parameter.requires_grad for parameter in frozen_policy.parameters())

    with pytest.raises(BaselineReferenceError, match="does not match its filename"):
        FrozenDateBenchmarkReference.load(
            first.baseline_reference_path,
            expected_reference_identity="0" * 64,
        )

    original_reference = first.baseline_reference_path.read_text(encoding="utf-8")
    first.baseline_reference_path.write_text(
        original_reference.replace("BM_D_5y.pt", "../BM_D_15y.pt"), encoding="utf-8"
    )
    with pytest.raises(BaselineReferenceError, match="reference identity"):
        FrozenDateBenchmarkReference.load(
            first.baseline_reference_path,
            expected_reference_identity=first.baseline_reference_identity,
        )
    first.baseline_reference_path.write_text(original_reference, encoding="utf-8")
    first.checkpoint_path.write_bytes(first.checkpoint_path.read_bytes() + b"changed")
    with pytest.raises(BaselineReferenceError, match="identity"):
        reference.load_policy(device="cpu", dtype=torch.float64)


def test_epoch_boundary_recovery_matches_an_uninterrupted_cpu_training_run(
    tmp_path: Path,
) -> None:
    configuration = replace(
        _configuration(tmp_path),
        run_scale=RunScaleConfiguration(
            profile="recovery-fixture",
            epochs=4,
            training_paths_per_epoch=2,
            selection_paths=2,
            test_paths=2,
            batch_size=2,
        ),
    )
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)

    uninterrupted = BMETrainer(
        replace(
            configuration,
            output=replace(configuration.output, run_name="uninterrupted"),
        ),
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
    ).fit(horizon_years=5)
    interrupted_trainer = BMETrainer(
        replace(
            configuration,
            output=replace(configuration.output, run_name="interrupted"),
        ),
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
    )
    with pytest.raises(TrainingInterrupted, match="cancelled") as captured:
        interrupted_trainer.fit(
            horizon_years=5,
            control=TrainingControl(stop_after_completed_epoch=2),
        )

    interruption = captured.value
    assert interruption.recovery_path.is_file()
    resumed_configuration = replace(
        configuration,
        output=replace(configuration.output, run_name="resumed-with-overrides"),
        resources=replace(
            configuration.resources,
            wall_clock_budget_seconds=configuration.resources.wall_clock_budget_seconds
            * 2,
            process_rss_limit_bytes=configuration.resources.process_rss_limit_bytes * 2,
            accelerator_memory_limit_bytes=(
                configuration.resources.accelerator_memory_limit_bytes * 2
            ),
        ),
    )
    recovered = BMETrainer(
        resumed_configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
    ).fit(
        horizon_years=5,
        control=TrainingControl(resume=True, resume_from=interruption.recovery_path),
    )
    expected = torch.load(uninterrupted.checkpoint_path, weights_only=False)
    actual = torch.load(recovered.checkpoint_path, weights_only=False)

    assert recovered.optimizer_updates == uninterrupted.optimizer_updates == 4
    assert actual["selection_history"] == expected["selection_history"]
    assert actual["scheduler_state"] == expected["scheduler_state"]
    assert actual["learning_rates"] == expected["learning_rates"]
    for name, value in expected["policy_state"].items():
        torch.testing.assert_close(actual["policy_state"][name], value)
    recovery = torch.load(interruption.recovery_path, weights_only=False)
    assert len(recovery["resume_lineage"]) == 1
    override = recovery["resume_lineage"][0]
    assert override["source_recovery_path"] == str(interruption.recovery_path)
    assert override["source_completed_epoch"] == 2
    assert override["resumed_at_unix_seconds"] > 0
    assert override["previous_execution"]["run_name"] == "interrupted"
    assert override["current_execution"]["run_name"] == "resumed-with-overrides"
    assert (
        override["current_execution"]["resource_limits"]["wall_clock_budget_seconds"]
        == override["previous_execution"]["resource_limits"]["wall_clock_budget_seconds"]
        * 2
    )


def test_recovery_replays_initial_state_and_rejects_semantic_mismatches(
    tmp_path: Path,
) -> None:
    configuration = replace(
        _configuration(tmp_path),
        run_scale=RunScaleConfiguration(
            profile="recovery-fixture",
            epochs=2,
            training_paths_per_epoch=2,
            selection_paths=2,
            test_paths=2,
            batch_size=2,
        ),
    )
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    trainer = BMETrainer(
        configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
    )

    with pytest.raises(TrainingInterrupted, match="cancelled") as captured:
        trainer.fit(
            horizon_years=5,
            control=TrainingControl(stop_after_completed_epoch=0),
        )
    assert not captured.value.recovery_path.exists()
    assert captured.value.interruption_path.is_file()

    with pytest.raises(TrainingInterrupted):
        trainer.fit(
            horizon_years=5,
            control=TrainingControl(stop_after_completed_epoch=1),
        )
    incompatible = replace(
        configuration,
        architecture=ArchitectureConfiguration(
            profile="paper", widths=(512, 512, 256, 128)
        ),
    )
    with pytest.raises(TrainingError, match="incompatible"):
        BMETrainer(
            incompatible,
            snapshot=snapshot,
            historical=historical,
            calibration=calibration,
        ).fit(horizon_years=5, control=TrainingControl(resume=True))


class _ExhaustedMonitor:
    def check(self, stage: str) -> ResourceSnapshot:
        raise BudgetExceeded(
            f"wall-clock budget exceeded at {stage}",
            diagnostics={"reason": "budget_exhausted", "stage": stage},
        )

    def snapshot(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            elapsed_seconds=1.0,
            process_rss_bytes=1,
            accelerator_allocated_bytes=None,
        )


def test_budget_exhaustion_is_recorded_as_recoverable_interruption(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)

    with pytest.raises(TrainingInterrupted, match="budget_exhausted") as captured:
        BMETrainer(
            configuration,
            snapshot=snapshot,
            historical=historical,
            calibration=calibration,
            resource_monitor=_ExhaustedMonitor(),
        ).fit(horizon_years=5)

    assert not captured.value.recovery_path.exists()
    assert captured.value.interruption_path.is_file()


def test_training_errors_are_recorded_as_operational_failures(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)

    with pytest.raises(TrainingInterrupted, match="operational_failure") as captured:
        BMETrainer(
            configuration,
            snapshot=snapshot,
            historical=historical,
            calibration=calibration,
        ).fit(horizon_years=7)

    interruption = json.loads(captured.value.interruption_path.read_text())
    assert interruption["reason"] == "operational_failure"
    assert interruption["diagnostics"]["error_type"] == "TrainingError"


def test_paper_scale_selection_starts_at_epoch_twenty_and_honors_ties_and_patience() -> None:
    tracker = SelectionTracker(SelectionSchedule.for_profile("paper_scale"))

    assert not tracker.should_select(19)
    assert tracker.record(20, (100.0, 10.0))
    assert not tracker.record(21, (99.95, 1.0))
    assert tracker.best_selection == (100.0, 10.0)
    assert tracker.record(22, (99.9, 10.0))
    assert tracker.record(23, (99.9, 9.0))
    for epoch in range(24, 38):
        assert not tracker.record(epoch, (99.9, 9.0))
        assert not tracker.should_stop(epoch)
    assert not tracker.record(38, (99.9, 9.0))
    assert tracker.should_stop(38)
