from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
import torch
from test_mm import _frozen_reference
from test_reporting import _report_configuration
from test_training import SOURCE, _configuration

from deepalm.baselines import BaselineReferenceError, FrozenDateBenchmarkReference
from deepalm.evaluation import LockedEvaluationError, LockedEvaluator, PolicyCheckpoint
from deepalm.reference_bank import ReferenceBankError, ReferenceBankProvider
from deepalm.reporting import ReportingError, build_compact_no_swap_report
from deepalm.runner import ReproductionRunner, RunStatus
from deepalm.term_structures import MarketScenarioModel
from deepalm.training import (
    BMETrainer,
    TrainingControl,
    TrainingError,
    TrainingInterrupted,
)


@pytest.mark.parametrize("mutation", ["missing", "unknown", "claimed-complete", "claimed-correction", "boolean-version"])
def test_freezing_checkpoint_rejects_unknown_artifact_semantics(tmp_path: Path, mutation: str) -> None:
    reference = _frozen_reference(tmp_path)
    checkpoint = torch.load(reference.checkpoint_path, weights_only=False)
    identity = checkpoint["code_identity"]
    if mutation == "missing":
        identity.pop("artifact_semantics")
    elif mutation == "unknown":
        identity["artifact_semantics"] = {"contract_version": 999}
    elif mutation == "claimed-complete":
        identity["artifact_semantics"]["repair_status"] = "complete"
    elif mutation == "claimed-correction":
        identity["artifact_semantics"]["corrections"]["C-1"] = True
    else:
        identity["artifact_semantics"]["contract_version"] = True
    torch.save(checkpoint, reference.checkpoint_path)

    with pytest.raises(BaselineReferenceError, match="artifact semantics"):
        FrozenDateBenchmarkReference.freeze(reference.checkpoint_path)


def test_training_records_actual_pending_repairs_without_metric_coupling(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    trainer = BMETrainer(
        configuration,
        snapshot=ReferenceBankProvider().build_canonical(historical),
        historical=historical,
        calibration=model.calibrate_hjm_pca(historical),
    )
    result = trainer.fit(horizon_years=5)
    checkpoint = torch.load(result.checkpoint_path, weights_only=False)
    semantics = checkpoint["code_identity"]["artifact_semantics"]
    assert semantics["repair_status"] == "pending"
    assert semantics["corrections"] == {
        "C-1": False, "C-2": False, "C-3": False, "C-5": True, "C-6": False,
    }
    assert "metric_version" not in semantics
    assert "R-1" not in semantics["corrections"]
    trainer.load_selected_checkpoint(result.checkpoint_path)
    checkpoint["code_identity"]["artifact_semantics"]["policy_version"] = "future-policy"
    incompatible = tmp_path / "incompatible.pt"
    torch.save(checkpoint, incompatible)
    with pytest.raises(TrainingError, match="artifact semantics"):
        trainer.load_selected_checkpoint(incompatible)
    evaluator = LockedEvaluator(
        configuration,
        snapshot=ReferenceBankProvider().build_canonical(historical),
        historical=historical,
        calibration=model.calibrate_hjm_pca(historical),
    )
    evaluation = evaluator.evaluate((PolicyCheckpoint("BM^E", result.checkpoint_path),))
    assert evaluation.manifest["artifact_semantics"]["repair_status"] == "pending"
    assert evaluation.manifest["artifact_semantics"]["corrections"]["R-1"] is False
    with pytest.raises(LockedEvaluationError, match="artifact semantics"):
        evaluator.evaluate((PolicyCheckpoint("incompatible", incompatible),))


def test_recovery_rejects_missing_semantics_before_resuming(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    trainer = BMETrainer(
        configuration,
        snapshot=ReferenceBankProvider().build_canonical(historical),
        historical=historical,
        calibration=model.calibrate_hjm_pca(historical),
    )
    with pytest.raises(TrainingInterrupted) as stopped:
        trainer.fit(horizon_years=5, control=TrainingControl(stop_after_completed_epoch=1))
    path = stopped.value.recovery_path
    recovery = torch.load(path, weights_only=False)
    assert recovery["recovery_identity"]["artifact_semantics"]["repair_status"] == "pending"
    recovery["recovery_identity"].pop("artifact_semantics")
    torch.save(recovery, path)
    with pytest.raises(TrainingError, match="artifact semantics"):
        trainer.fit(horizon_years=5, control=TrainingControl(resume=True, resume_from=path))


def test_snapshot_persists_its_actual_schema_and_rejects_claimed_future_repair(tmp_path: Path) -> None:
    provider = ReferenceBankProvider()
    historical = MarketScenarioModel().load_historical_term_structures(SOURCE)
    bank = provider.build_canonical(historical)
    path = tmp_path / "bank.json"
    provider.save(bank, path)
    saved = json.loads(path.read_text())
    assert saved["schema_version"] == 3
    assert provider.schema()["artifact_semantics"]["repair_status"] == "pending"
    assert saved["artifact_semantics"]["corrections"] == {"C-5": True, "C-6": False}
    assert provider.load(path).content_hash == bank.content_hash
    saved["artifact_semantics"]["repair_status"] = "complete"
    path.write_text(json.dumps(saved))
    with pytest.raises(ReferenceBankError, match="artifact semantics"):
        provider.load(path)


def test_frozen_reference_rejects_semantically_wrong_but_hash_valid_sidecar(tmp_path: Path) -> None:
    reference = _frozen_reference(tmp_path)
    payload = json.loads(reference.reference_path.read_text())
    payload["artifact_semantics"] = {"contract_version": 999}
    contents = json.dumps(payload)
    identity = sha256(contents.encode()).hexdigest()
    path = tmp_path / f"changed.baseline.{identity}.json"
    path.write_text(contents)
    with pytest.raises(BaselineReferenceError, match="artifact semantics"):
        FrozenDateBenchmarkReference.load(path)


def test_report_rejects_stale_source_semantics_and_preserves_pending_status(tmp_path: Path) -> None:
    configuration = _report_configuration(tmp_path)
    runner = ReproductionRunner()
    source = runner.run(replace(configuration, output=replace(configuration.output, run_name="source")))
    assert source.artifact_directory is not None
    source_path = source.artifact_directory / "manifest.json"
    manifest = json.loads(source_path.read_text())
    assert manifest["artifact_semantics"]["repair_status"] == "pending"
    report = runner.generate_compact_report(
        replace(configuration, output=replace(configuration.output, run_name="compatible")),
        source_run_directories=(source.artifact_directory,),
    )
    assert report.status is RunStatus.COMPLETED
    assert report.artifact_directory is not None
    contents = json.loads((report.artifact_directory / "compact-no-swap-report.json").read_text())
    assert contents["artifact_semantics"]["repair_status"] == "pending"
    stale_child = source.artifact_directory / "locked-evaluation.json"
    stale_child.write_text(json.dumps({
        "format_version": 1,
        "kind": "locked-final-test-evaluation",
        "artifact_semantics": {"contract_version": 999},
    }))
    with pytest.raises(ReportingError, match="artifact semantics"):
        build_compact_no_swap_report(configuration, source_run_directories=(source.artifact_directory,))
    stale_child.unlink()
    manifest["artifact_semantics"]["metric_version"] = "stale-metric"
    source_path.write_text(json.dumps(manifest))
    rejected = runner.generate_compact_report(
        replace(configuration, output=replace(configuration.output, run_name="incompatible")),
        source_run_directories=(source.artifact_directory,),
    )
    assert rejected.status is RunStatus.FAILED
    assert "artifact semantics" in str(rejected.error)
