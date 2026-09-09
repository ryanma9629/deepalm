"""Four-policy corrected local-validation pilot contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_run_skeleton import configuration_data

from deepalm import reporting
from deepalm.cli import main
from deepalm.config import resolve_configuration
from deepalm.planning import build_execution_plan
from deepalm.runner import AcceptanceStatus, ReproductionRunner, RunBundle, RunStatus
from deepalm.semantics import artifact_semantics


def _four_policy_pilot_data(tmp_path: Path) -> dict[str, object]:
    data = configuration_data(tmp_path)
    data["workflow_contract"] = {"name": "local-four-policy-comparison"}
    data["execution_profile"] = {"name": "m5-compact"}
    data["run_scale"] = {"profile": "four_policy_pilot"}
    data["policy"] = {"names": ["BM^E", "BM^C", "BM^D", "MM"]}
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


def test_four_policy_pilot_locks_eight_member_m5_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    configuration = resolve_configuration(_four_policy_pilot_data(tmp_path))

    plan = build_execution_plan(configuration)

    assert configuration.run_scale.profile == "four_policy_pilot"
    assert plan.primary_training_jobs == 8
    assert plan.primary_optimizer_updates == 32
    assert configuration.run_scale.test_paths == 64


def test_four_policy_pilot_command_dispatches_to_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    data = _four_policy_pilot_data(tmp_path)
    configuration = resolve_configuration(data)
    config_path = tmp_path / "four-policy-pilot.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    artifact_directory = tmp_path / "pilot"

    def fake_pilot(self: ReproductionRunner, received: object) -> RunBundle:
        assert received == configuration
        return RunBundle(
            status=RunStatus.COMPLETED,
            acceptance_status=configuration.acceptance.required_status,
            artifact_directory=artifact_directory,
        )

    monkeypatch.setattr(ReproductionRunner, "run_four_policy_pilot", fake_pilot)

    assert main(["four-policy-pilot", "--config", str(config_path)]) == 0
    assert capsys.readouterr().out.strip() == str(artifact_directory)


def test_four_policy_pilot_publishes_all_members_before_locked_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import deepalm.reference_bank as reference_bank_module
    import deepalm.training as training_module

    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    configuration = resolve_configuration(_four_policy_pilot_data(tmp_path))
    configuration.source_data.snb_csv.write_text("snb", encoding="utf-8")
    configuration.source_data.paper_pdf.write_bytes(b"paper")
    order: list[tuple[str, int]] = []
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

    def result(configuration: object, policy: str, horizon: int) -> object:
        output = configuration.output.directory / configuration.output.run_name
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / f"{policy}_{horizon}y.pt"
        checkpoint.write_text(f"{policy}-{horizon}", encoding="utf-8")
        baseline = output / f"BM_D_{horizon}y.baseline.json"
        if policy == "BM^D":
            baseline.write_text("baseline", encoding="utf-8")
        return SimpleNamespace(
            checkpoint_path=checkpoint,
            baseline_reference_path=baseline if policy in {"BM^D", "MM"} else None,
            baseline_reference_identity=f"baseline-{horizon}" if policy in {"BM^D", "MM"} else None,
            optimizer_updates=4,
            selected_epoch=2,
            clipped_gradient_norms=(0.1, 0.1, 0.1, 0.1),
        )

    def trainer(policy: str):
        class FakeTrainer:
            def __init__(self, configuration: object, **kwargs: object) -> None:
                self.configuration = configuration
                self.baseline_reference = kwargs.get("baseline_reference")

            def fit(self, *, horizon_years: int) -> object:
                order.append((policy, horizon_years))
                if policy == "MM":
                    assert self.baseline_reference is not None
                    mm_baselines[horizon_years] = self.baseline_reference.reference_path
                return result(self.configuration, policy, horizon_years)

        return FakeTrainer

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
    monkeypatch.setattr(training_module, "BMETrainer", trainer("BM^E"))
    monkeypatch.setattr(training_module, "BMConstantTrainer", trainer("BM^C"))
    monkeypatch.setattr(training_module, "BMDateTrainer", trainer("BM^D"))
    monkeypatch.setattr(training_module, "MMTrainer", trainer("MM"))

    bundle = ReproductionRunner().run_four_policy_pilot(configuration)

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.acceptance_status is AcceptanceStatus.PENDING
    assert order == [
        ("BM^E", 5), ("BM^C", 5), ("BM^D", 5), ("MM", 5),
        ("BM^E", 15), ("BM^C", 15), ("BM^D", 15), ("MM", 15),
    ]
    manifest = json.loads((bundle.artifact_directory / "manifest.json").read_text())
    pilot = manifest["four_policy_corrected_local_validation_pilot"]
    assert pilot["completed_primary_optimizer_updates"] == 32
    jobs = {
        (job["policy"], job["horizon_years"]): job
        for job in pilot["completed_training_jobs"]
    }
    assert set(jobs) == {
        (policy, horizon)
        for policy in ("BM^E", "BM^C", "BM^D", "MM")
        for horizon in (5, 15)
    }
    for horizon in (5, 15):
        assert mm_baselines[horizon].name == f"BM_D_{horizon}y.baseline.json"
        assert jobs[("MM", horizon)]["baseline_reference"] == jobs[
            ("BM^D", horizon)
        ]["baseline_reference"]


def test_four_policy_report_requires_all_eight_identity_linked_members(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pilot"
    evaluation = tmp_path / "evaluation"
    source.mkdir()
    evaluation.mkdir()
    members = tuple(
        (policy, horizon)
        for policy in ("BM^E", "BM^C", "BM^D", "MM")
        for horizon in (5, 15)
    )
    jobs = []
    checkpoints = {}
    reports = {}
    for policy, horizon in members:
        checkpoint = source / f"{policy}-{horizon}y.pt"
        checkpoint.write_text(f"{policy}-{horizon}", encoding="utf-8")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        jobs.append(
            {
                "policy": policy,
                "horizon_years": horizon,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": digest,
                "optimizer_updates": 4,
                "selected_epoch": 2,
                "finite_nonzero_optimization_signal": True,
                "baseline_reference": f"BM_D-{horizon}y.json"
                if policy in {"BM^D", "MM"}
                else None,
                "baseline_reference_identity": f"baseline-{horizon}"
                if policy in {"BM^D", "MM"}
                else None,
            }
        )
        label = f"{policy}-{horizon}y"
        checkpoints[label] = {
            "policy": policy,
            "horizon_years": horizon,
            "sha256": digest,
        }
        reports[label] = {"annualized_return": {"value": 0.03}}
    shared = {
        "reference_bank_content_hash": "bank",
        "market_source_hash": "market",
        "hjm_calibration_identity": "calibration",
    }
    manifest = {
        "artifact_semantics": artifact_semantics("evaluation"),
        "training_identity": {"artifact_semantics": artifact_semantics("training")},
        "status": "completed",
        "git_revision": "test",
        "four_policy_corrected_local_validation_pilot": {
            "status": "completed",
            "completed_primary_optimizer_updates": 32,
            "shared_identities": {**shared, "seed_registry": {}},
            "completed_training_jobs": jobs,
        },
    }
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    evidence = {
        "format_version": 1,
        "kind": "four-policy-corrected-local-validation-pilot-evaluation",
        "artifact_semantics": artifact_semantics("evaluation"),
        "status": "completed",
        "source_run": str(source.resolve()),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "reports": reports,
        "locked_evaluation_manifest": {
            "kind": "locked-final-test-evaluation",
            "artifact_semantics": artifact_semantics("evaluation"),
            "data_identities": shared,
            "checkpoints": checkpoints,
        },
    }
    (evaluation / "corrected-evaluation.json").write_text(
        json.dumps(evidence), encoding="utf-8"
    )

    report = reporting.build_local_validation_pilot_report(
        pilot_run_directory=source,
        evaluation_directory=evaluation,
        pilot_manifest_key="four_policy_corrected_local_validation_pilot",
        evaluation_kind="four-policy-corrected-local-validation-pilot-evaluation",
        report_kind="four-policy-corrected-local-validation-pilot-report",
        expected_members=members,
        expected_updates=32,
        label="Four-policy corrected pilot",
    )

    assert report.report["kind"] == "four-policy-corrected-local-validation-pilot-report"
    assert len(report.report["members"]) == 8
