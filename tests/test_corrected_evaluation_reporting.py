"""Public corrected evaluation and reporting lifecycle coverage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_corrected_pilot import _corrected_pilot_data

from deepalm import reporting
from deepalm.config import resolve_configuration
from deepalm.runner import ReproductionRunner, RunStatus
from deepalm.semantics import artifact_semantics


def _write_completed_corrected_pilot(source: Path) -> dict[str, object]:
    source.mkdir()
    jobs: list[dict[str, object]] = []
    for policy in ("BM^D", "MM"):
        for horizon_years in (5, 15):
            checkpoint = source / f"{policy}-{horizon_years}y.pt"
            checkpoint.write_bytes(f"{policy}-{horizon_years}".encode())
            jobs.append(
                {
                    "policy": policy,
                    "horizon_years": horizon_years,
                    "checkpoint": checkpoint.name,
                    "checkpoint_sha256": hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                    "optimizer_updates": 4,
                    "selected_epoch": 2,
                    "finite_nonzero_optimization_signal": True,
                    "baseline_reference": f"BM_D-{horizon_years}y.json",
                    "baseline_reference_identity": f"baseline-{horizon_years}",
                }
            )
    (source / "reference-bank.json").write_text("reference-bank", encoding="utf-8")
    manifest: dict[str, object] = {
        "artifact_semantics": artifact_semantics("evaluation"),
        "training_identity": {"artifact_semantics": artifact_semantics("training")},
        "status": "completed",
        "git_revision": "source-revision",
        "corrected_local_validation_pilot": {
            "status": "completed",
            "completed_primary_optimizer_updates": 16,
            "shared_identities": {
                "reference_bank_content_hash": "reference-bank",
                "market_source_hash": "market-source",
                "hjm_calibration_identity": "calibration",
                "seed_registry": {"bootstrap": 15},
            },
            "completed_training_jobs": jobs,
        },
    }
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_corrected_evaluation_and_report_accept_only_one_complete_current_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deepalm.reference_bank as reference_bank_module
    import deepalm.term_structures as term_structures_module

    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    configuration = resolve_configuration(_corrected_pilot_data(tmp_path))
    configuration.source_data.snb_csv.write_text("snb", encoding="utf-8")
    configuration.source_data.paper_pdf.write_bytes(b"paper")
    source = tmp_path / "completed-pilot"
    manifest = _write_completed_corrected_pilot(source)

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
        def load(self, _: Path) -> object:
            return SimpleNamespace(content_hash="reference-bank")

    class FakeLockedEvaluator:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def evaluate(
            self,
            checkpoints: tuple[object, ...],
            *,
            include_paired_bootstrap: bool,
            **_: object,
        ) -> object:
            assert include_paired_bootstrap is False
            expected = {
                f"{policy}-{horizon_years}y"
                for policy in ("BM^D", "MM")
                for horizon_years in (5, 15)
            }
            assert {checkpoint.label for checkpoint in checkpoints} == expected
            checkpoint_records = {
                checkpoint.label: {
                    "policy": checkpoint.label.rsplit("-", 1)[0],
                    "horizon_years": int(checkpoint.label.rsplit("-", 1)[1][:-1]),
                    "sha256": next(
                        job["checkpoint_sha256"]
                        for job in manifest["corrected_local_validation_pilot"][
                            "completed_training_jobs"
                        ]
                        if f"{job['policy']}-{job['horizon_years']}y"
                        == checkpoint.label
                    ),
                }
                for checkpoint in checkpoints
            }
            return SimpleNamespace(
                reports={label: {"losses": {"total": 1.0}} for label in expected},
                manifest={
                    "kind": "locked-final-test-evaluation",
                    "artifact_semantics": artifact_semantics("evaluation"),
                    "data_identities": {
                        "reference_bank_content_hash": "reference-bank",
                        "market_source_hash": "market-source",
                        "hjm_calibration_identity": "calibration",
                    },
                    "checkpoints": checkpoint_records,
                },
            )

    monkeypatch.setattr("deepalm.runner.ResourceMonitor", FakeMonitor)
    monkeypatch.setattr(
        term_structures_module, "MarketScenarioModel", FakeMarketScenarioModel
    )
    monkeypatch.setattr(
        reference_bank_module, "ReferenceBankProvider", FakeReferenceBankProvider
    )
    monkeypatch.setattr("deepalm.evaluation.LockedEvaluator", FakeLockedEvaluator)

    runner = ReproductionRunner()
    evaluation = runner.evaluate_corrected_pilot(
        configuration, source_run_directory=source
    )

    assert evaluation.status is RunStatus.COMPLETED
    assert evaluation.artifact_directory is not None
    evidence = json.loads(
        (evaluation.artifact_directory / "corrected-evaluation.json").read_text()
    )
    assert evidence["source_manifest_sha256"] == hashlib.sha256(
        (source / "manifest.json").read_bytes()
    ).hexdigest()
    assert set(evidence["reports"]) == {
        "BM^D-5y",
        "BM^D-15y",
        "MM-5y",
        "MM-15y",
    }
    assert "paired_intervals" not in evidence

    report = runner.generate_corrected_pilot_report(
        configuration,
        pilot_run_directory=source,
        evaluation_directory=evaluation.artifact_directory,
    )

    assert report.status is RunStatus.COMPLETED
    assert report.artifact_directory is not None
    report_data = json.loads(
        (report.artifact_directory / "corrected-pilot-report.json").read_text()
    )
    assert report_data["status"] == "completed"
    assert len(report_data["members"]) == 4
    assert "paired_intervals" not in report_data
    assert "conventions" not in report_data

    source_manifest = json.loads((source / "manifest.json").read_text())
    jobs = source_manifest["corrected_local_validation_pilot"][
        "completed_training_jobs"
    ]
    jobs.append(dict(jobs[0]))
    (source / "manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")
    evidence["source_manifest_sha256"] = hashlib.sha256(
        (source / "manifest.json").read_bytes()
    ).hexdigest()
    (evaluation.artifact_directory / "corrected-evaluation.json").write_text(
        json.dumps(evidence), encoding="utf-8"
    )
    with pytest.raises(reporting.ReportingError, match="incomplete"):
        reporting.build_corrected_pilot_report(
            pilot_run_directory=source,
            evaluation_directory=evaluation.artifact_directory,
        )


@pytest.mark.parametrize("command", ["paired-evaluate", "paired-report"])
def test_paired_evaluation_commands_are_not_public(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stopped:
        from deepalm.cli import main

        main([command])

    assert stopped.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert not hasattr(ReproductionRunner, "evaluate_paired_convention_pilot")
    assert not hasattr(ReproductionRunner, "generate_paired_pilot_report")
