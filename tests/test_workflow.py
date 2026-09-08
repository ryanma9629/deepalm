"""End-to-end public-seam coverage for the bounded local workflow."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import yaml
from test_run_skeleton import configuration_data

from deepalm.cli import main
from deepalm.config import (
    PolicyConfiguration,
    RunScaleConfiguration,
    resolve_configuration,
)
from deepalm.runner import AcceptanceStatus, ReproductionRunner, RunBundle, RunStatus


def _workflow_configuration(tmp_path: Path):
    raw = configuration_data(tmp_path)
    repository = Path(__file__).resolve().parents[1]
    source_data = raw["source_data"]
    assert isinstance(source_data, dict)
    source_data.update(
        {
            "snb_csv": str(
                repository / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
            ),
            "paper_pdf": str(repository / "docs/Deep treasury management for banks.pdf"),
        }
    )
    configuration = resolve_configuration(raw)
    return replace(
        configuration,
        policy=PolicyConfiguration(names=("BM^E", "BM^C", "BM^D", "MM")),
        run_scale=RunScaleConfiguration(
            profile="workflow-fixture",
            epochs=2,
            training_paths_per_epoch=2,
            selection_paths=2,
            test_paths=2,
            batch_size=2,
        ),
        output=replace(configuration.output, run_name="local-workflow"),
        seeds={**configuration.seeds, "market_scenarios": 0},
    )


def test_local_workflow_runs_every_required_stage_and_publishes_one_bundle(
    tmp_path: Path,
) -> None:
    configuration = _workflow_configuration(tmp_path)

    runner = ReproductionRunner()
    bundle = runner.run_local_workflow(configuration)

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.acceptance_status is AcceptanceStatus.PASSED
    assert bundle.artifact_directory is not None
    assert not list(bundle.artifact_directory.parent.glob(".local-workflow-*"))
    manifest = json.loads((bundle.artifact_directory / "manifest.json").read_text())
    workflow = manifest["local_workflow"]
    assert workflow["development_validation_status"] == "development-validated"
    assert len(workflow["completed_training_jobs"]) == 8
    assert workflow["completed_primary_optimizer_updates"] == 16
    assert workflow["paper_width_checks"]["completed_optimizer_updates"] == 2
    assert workflow["replay"]["reference_bank_content_hash_matches"] is True
    assert workflow["one_step_market_statistics"]["paths"] == 50_000
    assert workflow["unavailable_device_checks"]["cuda"]["status"] == "not-run"
    acceptance = json.loads(
        (bundle.artifact_directory / "acceptance-report.json").read_text()
    )
    assert acceptance["status"] == "passed"
    assert all(
        {"purpose", "applicability", "observation", "threshold", "status", "evidence"}
        <= check.keys()
        for check in acceptance["checks"]
    )
    checks = {check["purpose"]: check for check in acceptance["checks"]}
    assert checks["finite gradients and financial rollout"]["status"] == "passed"
    assert checks["epoch-boundary recovery equivalence"]["status"] == "passed"
    assert (bundle.artifact_directory / "locked-evaluation.json").is_file()
    assert (bundle.artifact_directory / "reference-bank-sensitivity.json").is_file()
    assert (bundle.artifact_directory / "horizon-scenario-analysis.json").is_file()
    assert (bundle.artifact_directory / "compact-no-swap-report.json").is_file()
    for stage in ("train", "resume", "evaluate", "accept"):
        reused = runner.reuse_completed_workflow_stage(
            configuration,
            source_run_directory=bundle.artifact_directory,
            stage=stage,
        )
        assert reused.status is RunStatus.COMPLETED
        assert reused.artifact_directory == bundle.artifact_directory
    manifest_path = bundle.artifact_directory / "manifest.json"
    original_manifest = manifest_path.read_text()
    manifest["training_identity"]["artifact_semantics"]["policy_version"] = "old-policy"
    manifest_path.write_text(json.dumps(manifest))
    semantic_reuse = runner.reuse_completed_workflow_stage(
        configuration, source_run_directory=bundle.artifact_directory, stage="train"
    )
    assert semantic_reuse.status is RunStatus.FAILED
    assert "artifact semantics" in str(semantic_reuse.error)
    manifest_path.write_text(original_manifest)
    metric_changed = json.loads(original_manifest)
    metric_changed["artifact_semantics"]["metric_version"] = "different-report-metrics"
    manifest_path.write_text(json.dumps(metric_changed))
    weights_reused = runner.reuse_completed_workflow_stage(
        configuration, source_run_directory=bundle.artifact_directory, stage="train"
    )
    assert weights_reused.status is RunStatus.COMPLETED
    metrics_rejected = runner.reuse_completed_workflow_stage(
        configuration, source_run_directory=bundle.artifact_directory, stage="evaluate"
    )
    assert metrics_rejected.status is RunStatus.FAILED
    assert "artifact semantics" in str(metrics_rejected.error)
    manifest_path.write_text(original_manifest)
    incompatible = runner.reuse_completed_workflow_stage(
        replace(configuration, seeds={**configuration.seeds, "bootstrap": 99}),
        source_run_directory=bundle.artifact_directory,
        stage="train",
    )
    assert incompatible.status is RunStatus.FAILED
    assert "configuration differs" in (incompatible.error or "")
    acceptance_path = bundle.artifact_directory / "acceptance-report.json"
    acceptance_path.write_text("{}\n", encoding="utf-8")
    tampered = runner.reuse_completed_workflow_stage(
        configuration,
        source_run_directory=bundle.artifact_directory,
        stage="accept",
    )
    assert tampered.status is RunStatus.FAILED
    assert "artifact hash differs" in (tampered.error or "")


def test_workflow_command_dispatches_to_the_public_workflow_seam(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    raw = configuration_data(tmp_path)
    repository = Path(__file__).resolve().parents[1]
    source_data = raw["source_data"]
    assert isinstance(source_data, dict)
    source_data.update(
        {
            "snb_csv": str(
                repository / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
            ),
            "paper_pdf": str(repository / "docs/Deep treasury management for banks.pdf"),
        }
    )
    raw["policy"] = {"names": ["BM^E", "BM^C", "BM^D", "MM"]}
    configuration = resolve_configuration(raw)
    config_path = tmp_path / "workflow.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    artifact_directory = tmp_path / "completed-workflow"

    def fake_workflow(
        self: ReproductionRunner, received: object
    ) -> RunBundle:
        assert received == configuration
        return RunBundle(
            status=RunStatus.COMPLETED,
            acceptance_status=AcceptanceStatus.PASSED,
            artifact_directory=artifact_directory,
        )

    monkeypatch.setattr(ReproductionRunner, "run_local_workflow", fake_workflow)

    assert main(["workflow", "--config", str(config_path)]) == 0
    assert capsys.readouterr().out.strip() == str(artifact_directory)


def test_local_workflow_preserves_incomplete_diagnostics_without_success(
    tmp_path: Path,
) -> None:
    configuration = _workflow_configuration(tmp_path)
    configuration = replace(
        configuration,
        resources=replace(configuration.resources, wall_clock_budget_seconds=0.0),
    )

    bundle = ReproductionRunner().run_local_workflow(configuration)

    assert bundle.status is RunStatus.INCOMPLETE
    assert bundle.acceptance_status is AcceptanceStatus.PENDING
    assert bundle.artifact_directory is not None
    manifest = json.loads((bundle.artifact_directory / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["local_workflow"]["development_validation_status"] == "incomplete"
