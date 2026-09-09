from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
import torch
import yaml
from test_run_skeleton import configuration_data

from deepalm import reporting
from deepalm.cli import main
from deepalm.config import resolve_configuration
from deepalm.reporting import ReportingError
from deepalm.runner import ReproductionRunner, RunStatus
from deepalm.semantics import artifact_semantics


def _report_configuration(tmp_path: Path):
    raw = configuration_data(tmp_path)
    repository = Path(__file__).resolve().parents[1]
    source_data = raw["source_data"]
    assert isinstance(source_data, dict)
    source_data.update(
        {
            "snb_csv": str(
                repository / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
            ),
            "paper_pdf": str(
                repository / "docs/Deep treasury management for banks.pdf"
            ),
        }
    )
    return resolve_configuration(raw)


def test_runner_generates_an_auditable_no_swap_report_and_coverage_inventory(
    tmp_path: Path,
) -> None:
    configuration = _report_configuration(tmp_path)
    runner = ReproductionRunner()
    source = runner.run(
        replace(configuration, output=replace(configuration.output, run_name="source"))
    )
    assert source.artifact_directory is not None

    bundle = runner.generate_compact_report(
        replace(configuration, output=replace(configuration.output, run_name="report")),
        source_run_directories=(source.artifact_directory,),
    )

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.artifact_directory is not None
    report = json.loads(
        (bundle.artifact_directory / "compact-no-swap-report.json").read_text()
    )
    inventory = json.loads(
        (bundle.artifact_directory / "paper-coverage-inventory.json").read_text()
    )
    calibration = json.loads(
        (bundle.artifact_directory / "calibration-evidence.json").read_text()
    )
    comparison = json.loads(
        (bundle.artifact_directory / "hjm-hull-white-comparison.json").read_text()
    )
    manifest = json.loads((bundle.artifact_directory / "manifest.json").read_text())

    assert report["kind"] == "compact-no-swap-local-report"
    assert report["purpose"] == "local workflow demonstration"
    assert report["swaps_included"] is False
    assert report["source_runs"][0]["status"] == "completed"
    assert report["deferred_research"][0]["reason"]
    assert set(report["local_evidence"]) == {
        "policy_metrics",
        "equity_and_constraints",
        "durations_and_sensitivities",
        "recovery",
        "mm_truncation",
        "scenario_categories_and_bootstrap",
    }
    assert all(
        evidence["status"] in {"included", "deferred"}
        for evidence in report["local_evidence"].values()
    )
    assert len(inventory["entries"]) == 20
    assert {entry["paper_item"] for entry in inventory["entries"]} == {
        *(f"Table {number}" for number in range(1, 6)),
        *(f"Figure {number}" for number in range(3, 18)),
    }
    assert {
        (entry["paper_item"], entry["paper_page"], entry["subject"])
        for entry in inventory["entries"]
    } == {
        ("Table 1", 7, "Economic balance sheet"),
        ("Table 2", 16, "Hyperparameters"),
        (
            "Table 3",
            21,
            "Main results: losses, equity distribution, returns and dividends",
        ),
        ("Table 4", 26, "Constraint statistics"),
        ("Table 5", 34, "Category statistics; disclose covered policy columns"),
        ("Figure 3", 17, "Simulated one-month yields under HJM-PCA"),
        (
            "Figure 4",
            17,
            "Terminal five-year yield curves, HJM-PCA versus Hull-White",
        ),
        ("Figure 5", 20, "Decision-network architecture"),
        ("Figure 6", 24, "Equity-ratio histograms"),
        ("Figure 7", 24, "Constant benchmark strategies"),
        ("Figure 8", 25, "Investment and financing volume"),
        ("Figure 9", 26, "LCR and CMR"),
        ("Figure 10", 27, "Equity/RWA"),
        ("Figure 11", 28, "Interest-rate sensitivity and portfolio durations"),
        ("Figure 12", 29, "Five-year yield-curve scenarios"),
        ("Figure 13", 30, "Five-year decisions"),
        ("Figure 14", 30, "Five-year sensitivity gaps"),
        ("Figure 15", 31, "Fifteen-year yield-curve scenarios"),
        ("Figure 16", 32, "Fifteen-year decisions"),
        ("Figure 17", 32, "Fifteen-year sensitivity gaps"),
    }
    assert {entry["status"] for entry in inventory["entries"]} == {"missing"}
    assert {entry["source_pdf_sha256"] for entry in inventory["entries"]} == {
        sha256(configuration.source_data.paper_pdf.read_bytes()).hexdigest()
    }
    assert all(
        entry["matching_output_evidence"]["status"] == "missing"
        for entry in inventory["entries"]
    )
    table_five = next(
        entry for entry in inventory["entries"] if entry["paper_item"] == "Table 5"
    )
    assert table_five["supported_policy_scope"]["excluded_policy_columns"] == [
        "swap strategies"
    ]
    assert all(
        {
            "policy",
            "horizon_years",
            "sample_size",
            "units",
            "source_runs",
            "architecture",
            "parameter_counts",
            "seed_registry",
        }
        <= entry["metadata"].keys()
        for entry in inventory["entries"]
    )
    assert calibration["three_component_explained_variance"]["threshold"] == 0.9
    assert calibration["cubic_fit_reference_levels"]["waiver_status"] == "waived"
    assert comparison["sample_size"] <= 32
    assert comparison["models"] == ["hjm-pca", "project-hull-white"]
    assert comparison["horizon_years"] == 5
    assert comparison["hull_white_configuration"] == {
        "label": "project_choice",
        "mean_reversion": 0.2,
        "volatility": 0.01,
    }
    assert comparison["presentation_metadata"]["seed_registry"]
    assert (
        manifest["compact_report"]["calibration_identity"]
        == calibration["calibration_identity"]
    )
    assert manifest["compact_report"]["reference_bank_content_hash"]
    assert manifest["compact_report"]["weekly_dates"] == calibration["weekly_dates"]
    assert manifest["compact_report"]["acceptance_evidence"]
    assert manifest["compact_report"]["actual_work"]["completed_training_jobs"] == []
    assert (
        manifest["compact_report"]["resource_measurements"]["status"] == "unavailable"
    )
    assert report["actual_work"]["numeric_metrics"]["status"] == "unavailable"



def test_report_rejects_nonfinite_json_and_cleans_failed_atomic_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"metric": NaN}', encoding="utf-8")
    with pytest.raises(ReportingError, match="unreadable"):
        reporting._read_report_object(nonfinite, "nonfinite evidence")

    configuration = _report_configuration(tmp_path)
    artifacts = reporting.CompactReportArtifacts(
        report={
            "kind": "compact-no-swap-local-report",
            "purpose": "local workflow demonstration",
            "source_runs": [],
            "presentation_metadata": {"parameter_counts": []},
            "actual_work": {"resource_measurements": {}, "numeric_metrics": {}},
            "nonfinite": float("nan"),
        },
        coverage_inventory={},
        calibration_evidence={
            "calibration_identity": "calibration",
            "weekly_dates": [],
        },
        market_comparison={},
        reference_bank_summary={"content_hash": "reference-bank"},
    )
    monkeypatch.setattr(
        reporting, "build_compact_no_swap_report", lambda *_args, **_kwargs: artifacts
    )
    output = replace(configuration.output, run_name="nonfinite-report")

    bundle = ReproductionRunner().generate_compact_report(
        replace(configuration, output=output), source_run_directories=()
    )

    assert bundle.status is RunStatus.FAILED
    assert not (output.directory / output.run_name).exists()
    assert not list(output.directory.glob(f".{output.run_name}-*"))


def test_report_rejects_incomplete_source_bundles(tmp_path: Path) -> None:
    configuration = _report_configuration(tmp_path)
    source_directory = tmp_path / "incomplete-source"
    source_directory.mkdir()
    (source_directory / "manifest.json").write_text(
        json.dumps({"status": "incomplete"}), encoding="utf-8"
    )

    bundle = ReproductionRunner().generate_compact_report(
        configuration, source_run_directories=(source_directory,)
    )

    assert bundle.status is RunStatus.FAILED
    assert bundle.error is not None
    assert "not completed" in bundle.error


def test_report_requires_at_least_one_completed_source_bundle(tmp_path: Path) -> None:
    configuration = _report_configuration(tmp_path)

    bundle = ReproductionRunner().generate_compact_report(
        configuration, source_run_directories=()
    )

    assert bundle.status is RunStatus.FAILED
    assert bundle.error is not None
    assert "at least one" in bundle.error


def test_report_rejects_source_evidence_with_a_different_market_identity(
    tmp_path: Path,
) -> None:
    configuration = _report_configuration(tmp_path)
    source = ReproductionRunner().run(
        replace(configuration, output=replace(configuration.output, run_name="source"))
    )
    assert source.artifact_directory is not None
    manifest_path = source.artifact_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_hashes"]["snb_csv"] = "another-market"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    bundle = ReproductionRunner().generate_compact_report(
        replace(configuration, output=replace(configuration.output, run_name="report")),
        source_run_directories=(source.artifact_directory,),
    )

    assert bundle.status is RunStatus.FAILED
    assert bundle.error is not None
    assert "input identities" in bundle.error


def test_report_does_not_treat_an_unvalidated_same_named_json_as_evidence(
    tmp_path: Path,
) -> None:
    configuration = _report_configuration(tmp_path)
    source = ReproductionRunner().run(
        replace(configuration, output=replace(configuration.output, run_name="source"))
    )
    assert source.artifact_directory is not None
    (source.artifact_directory / "locked-evaluation.json").write_text(
        json.dumps({"kind": "forged"}), encoding="utf-8"
    )

    bundle = ReproductionRunner().generate_compact_report(
        replace(configuration, output=replace(configuration.output, run_name="report")),
        source_run_directories=(source.artifact_directory,),
    )

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.artifact_directory is not None
    report = json.loads(
        (bundle.artifact_directory / "compact-no-swap-report.json").read_text()
    )
    assert report["local_evidence"]["policy_metrics"]["status"] == "deferred"


def test_report_requires_the_recovery_artifact_named_by_an_interruption(
    tmp_path: Path,
) -> None:
    configuration = _report_configuration(tmp_path)
    source = ReproductionRunner().run(
        replace(configuration, output=replace(configuration.output, run_name="source"))
    )
    assert source.artifact_directory is not None
    (source.artifact_directory / "BM_E_5y.interruption.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "status": "incomplete",
                "reason": "cancelled",
                "recovery_path": "BM_E_5y.recovery.pt",
                "diagnostics": {},
            }
        ),
        encoding="utf-8",
    )

    bundle = ReproductionRunner().generate_compact_report(
        replace(configuration, output=replace(configuration.output, run_name="report")),
        source_run_directories=(source.artifact_directory,),
    )

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.artifact_directory is not None
    report = json.loads(
        (bundle.artifact_directory / "compact-no-swap-report.json").read_text()
    )
    assert report["local_evidence"]["recovery"]["status"] == "deferred"


def test_report_rejects_malformed_recovery_identity_without_crashing(
    tmp_path: Path,
) -> None:
    configuration = _report_configuration(tmp_path)
    source = ReproductionRunner().run(
        replace(configuration, output=replace(configuration.output, run_name="source"))
    )
    assert source.artifact_directory is not None
    (source.artifact_directory / "BM_E_5y.interruption.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "status": "incomplete",
                "reason": "cancelled",
                "recovery_path": "BM_E_5y.recovery.pt",
                "diagnostics": {},
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "format_version": 1,
            "kind": "epoch-recovery",
            "recovery_identity": {"data_identities": []},
        },
        source.artifact_directory / "BM_E_5y.recovery.pt",
    )

    bundle = ReproductionRunner().generate_compact_report(
        replace(configuration, output=replace(configuration.output, run_name="report")),
        source_run_directories=(source.artifact_directory,),
    )

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.artifact_directory is not None
    report = json.loads(
        (bundle.artifact_directory / "compact-no-swap-report.json").read_text()
    )
    assert report["local_evidence"]["recovery"]["status"] == "deferred"


def test_report_uses_selected_checkpoint_epoch_after_identity_audit(
    tmp_path: Path,
) -> None:
    configuration = _report_configuration(tmp_path)
    source = ReproductionRunner().run(
        replace(configuration, output=replace(configuration.output, run_name="source"))
    )
    assert source.artifact_directory is not None
    manifest_path = source.artifact_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["market_preflight"] = {"calibration": {"identity": "calibration-1"}}
    manifest["reference_bank"] = {"content_hash": "reference-bank-1"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    torch.save(
        {
            "format_version": 1,
            "policy": "BM^E",
            "horizon_years": 5,
            "policy_state": {"weight": torch.tensor([1.0])},
            "optimizer_updates": 2,
            "selected_epoch": 1,
            "selection_history": [
                {"epoch": 1, "total_loss": 3.0, "penalty_loss": 0.2},
                {"epoch": 2, "total_loss": 1.0, "penalty_loss": 0.1},
            ],
            "configuration": manifest["resolved_configuration"],
            "configuration_identity": manifest["resolved_configuration_hash"],
            "data_identities": {
                "market_source_hash": manifest["input_hashes"]["snb_csv"],
                "hjm_calibration_identity": "calibration-1",
                "reference_bank_content_hash": "reference-bank-1",
            },
            "code_identity": {
                "artifact_semantics": artifact_semantics("training"),
            },
        },
        source.artifact_directory / "BM_E_5y.pt",
    )

    bundle = ReproductionRunner().generate_compact_report(
        replace(configuration, output=replace(configuration.output, run_name="report")),
        source_run_directories=(source.artifact_directory,),
    )

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.artifact_directory is not None
    report = json.loads(
        (bundle.artifact_directory / "compact-no-swap-report.json").read_text()
    )
    completed = report["actual_work"]["completed_training_jobs"]
    assert len(completed) == 1
    assert completed[0]["checkpoint_sha256"]
    assert completed[0]["selection_metrics"] == {
        "total_loss": 3.0,
        "penalty_loss": 0.2,
    }
    assert report["actual_work"]["completed_optimizer_updates"] == 2


def test_report_excludes_checkpoint_with_stale_training_semantics(
    tmp_path: Path,
) -> None:
    configuration = _report_configuration(tmp_path)
    source = ReproductionRunner().run(
        replace(configuration, output=replace(configuration.output, run_name="source"))
    )
    assert source.artifact_directory is not None
    manifest_path = source.artifact_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["market_preflight"] = {"calibration": {"identity": "calibration-1"}}
    manifest["reference_bank"] = {"content_hash": "reference-bank-1"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    torch.save(
        {
            "format_version": 1,
            "policy": "BM^E",
            "horizon_years": 5,
            "policy_state": {"weight": torch.tensor([1.0])},
            "optimizer_updates": 2,
            "selected_epoch": 1,
            "selection_history": [{"epoch": 1, "total_loss": 3.0, "penalty_loss": 0.2}],
            "configuration": manifest["resolved_configuration"],
            "configuration_identity": manifest["resolved_configuration_hash"],
            "data_identities": {
                "market_source_hash": manifest["input_hashes"]["snb_csv"],
                "hjm_calibration_identity": "calibration-1",
                "reference_bank_content_hash": "reference-bank-1",
            },
            "code_identity": {"artifact_semantics": {"contract_version": 999}},
        },
        source.artifact_directory / "BM_E_5y.pt",
    )

    bundle = ReproductionRunner().generate_compact_report(
        replace(configuration, output=replace(configuration.output, run_name="report")),
        source_run_directories=(source.artifact_directory,),
    )

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.artifact_directory is not None
    report = json.loads(
        (bundle.artifact_directory / "compact-no-swap-report.json").read_text()
    )
    assert report["actual_work"]["completed_training_jobs"] == []
    audit = report["actual_work"]["checkpoint_audit"]
    assert audit[0]["status"] == "unavailable"


def test_cli_writes_a_report_from_a_completed_source_bundle(
    tmp_path: Path, capsys
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
            "paper_pdf": str(
                repository / "docs/Deep treasury management for banks.pdf"
            ),
        }
    )
    configuration = resolve_configuration(raw)
    source = ReproductionRunner().run(
        replace(configuration, output=replace(configuration.output, run_name="source"))
    )
    assert source.artifact_directory is not None
    config_path = tmp_path / "report.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    exit_code = main(
        [
            "report",
            "--config",
            str(config_path),
            "--source-run",
            str(source.artifact_directory),
        ]
    )

    artifact_directory = Path(capsys.readouterr().out.strip())
    assert exit_code == 0
    assert (artifact_directory / "compact-no-swap-report.json").is_file()
