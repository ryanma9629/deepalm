"""Four-policy corrected local-validation pilot contracts."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_run_skeleton import configuration_data

from deepalm.cli import main
from deepalm.config import resolve_configuration
from deepalm.local_validation import is_valid_local_validation_selection_epoch
from deepalm.planning import build_execution_plan
from deepalm.runner import ReproductionRunner, RunStatus
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


@pytest.mark.parametrize("selected_epoch", (1, 2))
def test_local_validation_selection_epoch_accepts_the_two_training_epochs(
    selected_epoch: object,
) -> None:
    assert is_valid_local_validation_selection_epoch(selected_epoch)


@pytest.mark.parametrize("selected_epoch", (0, 3, 1.5, "1", True))
def test_local_validation_selection_epoch_requires_a_json_integer_in_range(
    selected_epoch: object,
) -> None:
    assert not is_valid_local_validation_selection_epoch(selected_epoch)


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


def test_four_policy_pilot_publishes_all_members_before_locked_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import deepalm.reference_bank as reference_bank_module
    import deepalm.training as training_module

    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    data = _four_policy_pilot_data(tmp_path)
    configuration = resolve_configuration(data)
    configuration.source_data.snb_csv.write_text("snb", encoding="utf-8")
    configuration.source_data.paper_pdf.write_bytes(b"paper")
    config_path = tmp_path / "local-four-policy.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
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

            def fit(
                self, *, horizon_years: int, progress_callback: object = None
            ) -> object:
                order.append((policy, horizon_years))
                if policy == "MM":
                    assert self.baseline_reference is not None
                    mm_baselines[horizon_years] = self.baseline_reference.reference_path
                if progress_callback is not None:
                    progress_callback(
                        SimpleNamespace(
                            policy_name=policy,
                            horizon_years=horizon_years,
                            epoch=1,
                            epochs=2,
                            optimizer_updates=2,
                            average_training_loss=1.25,
                            selection=SimpleNamespace(
                                total_loss=1.5, penalty_loss=0.25
                            ),
                        )
                    )
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

    assert main(["plan", "--config", str(config_path)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["workflow_contract"] == "local-four-policy-comparison"
    assert plan["execution_profile"] == "m5-compact"
    assert plan["primary_training_jobs"] == 8
    assert plan["primary_optimizer_updates"] == 32

    assert main(["run", "--config", str(config_path)]) == 0
    run_output = capsys.readouterr().out.splitlines()
    bundle_directory = Path(run_output[-1])

    assert "[train] BM^E 5y: starting" in run_output
    assert (
        "[train] BM^E 5y | epoch 1/2 | updates=2 | "
        "train_loss=1.250000 | selection_loss=1.500000 | penalty_loss=0.250000"
        in run_output
    )
    assert "[train] MM 15y: completed | selected_epoch=2 | updates=4" in run_output

    assert bundle_directory.is_dir()
    assert order == [
        ("BM^E", 5), ("BM^C", 5), ("BM^D", 5), ("MM", 5),
        ("BM^E", 15), ("BM^C", 15), ("BM^D", 15), ("MM", 15),
    ]
    manifest = json.loads((bundle_directory / "manifest.json").read_text())
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

    quiet_data = _four_policy_pilot_data(tmp_path)
    quiet_data["output"]["run_name"] = "quiet-four-policy"
    quiet_config_path = tmp_path / "quiet-four-policy.yaml"
    quiet_config_path.write_text(yaml.safe_dump(quiet_data), encoding="utf-8")

    assert main(["run", "--config", str(quiet_config_path), "--no-verbose"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        str(tmp_path / "runs" / "quiet-four-policy")
    ]


def test_four_policy_report_requires_all_eight_identity_linked_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import deepalm.reference_bank as reference_bank_module
    import deepalm.term_structures as term_structures_module

    # The evaluator imports the shared bank provider at module load time.
    importlib.import_module("deepalm.evaluation")
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    configuration = resolve_configuration(_four_policy_pilot_data(tmp_path))
    configuration.source_data.snb_csv.write_text("snb", encoding="utf-8")
    configuration.source_data.paper_pdf.write_bytes(b"paper")
    config_path = tmp_path / "local-four-policy.yaml"
    config_path.write_text(
        yaml.safe_dump(_four_policy_pilot_data(tmp_path)), encoding="utf-8"
    )
    source = tmp_path / "pilot"
    source.mkdir()
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
                "selected_epoch": 1 if (policy, horizon) == ("BM^E", 15) else 2,
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
        "resolved_configuration": configuration.to_dict(),
    }
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (source / "reference-bank.json").write_text("reference-bank", encoding="utf-8")

    class FakeMonitor:
        def __init__(self, **_: object) -> None:
            pass

        def check(self, _: str) -> None:
            pass

        def snapshot(self) -> object:
            return SimpleNamespace(to_dict=lambda: {"elapsed_seconds": 1.0})

    class FakeMarketScenarioModel:
        def load_historical_term_structures(self, *_: object, **__: object) -> object:
            return SimpleNamespace(source_hash="market")

        def calibrate_hjm_pca(self, _: object) -> object:
            return SimpleNamespace(calibration_identity="calibration")

    class FakeReferenceBankProvider:
        def load(self, _: Path) -> object:
            return SimpleNamespace(content_hash="bank")

    class FakeLockedEvaluator:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def evaluate(self, received: tuple[object, ...], **_: object) -> object:
            assert {checkpoint.label for checkpoint in received} == set(checkpoints)
            return SimpleNamespace(
                reports=reports,
                manifest={
                    "kind": "locked-final-test-evaluation",
                    "artifact_semantics": artifact_semantics("evaluation"),
                    "data_identities": shared,
                    "checkpoints": checkpoints,
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

    assert main(
        ["evaluate", "--config", str(config_path), "--source-run", str(source)]
    ) == 0
    evaluation_directory = Path(capsys.readouterr().out.strip())
    evidence = json.loads(
        (evaluation_directory / "corrected-evaluation.json").read_text()
    )
    assert evidence["workflow_contract"] == "local-four-policy-comparison"
    assert evidence["execution_profile"] == "m5-compact"
    assert evidence["locked_evaluation_manifest"]["data_identities"] == shared

    assert main(
        [
            "report",
            "--config",
            str(config_path),
            "--source-run",
            str(source),
            "--evaluation-run",
            str(evaluation_directory),
        ]
    ) == 0
    report_directory = Path(capsys.readouterr().out.strip())
    report_data = json.loads(
        (report_directory / "four-policy-pilot-report.json").read_text()
    )
    assert report_data["kind"] == "four-policy-corrected-local-validation-pilot-report"
    assert len(report_data["members"]) == 8
    assert report_data["locked_evaluation_identity"]["data_identities"] == shared
    assert "not convergence evidence" in report_data["disclosure"]


def test_generic_four_policy_report_names_its_contract_on_source_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract diagnostics must not fall back to the two-policy pilot label."""

    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    configuration = resolve_configuration(_four_policy_pilot_data(tmp_path))
    source = tmp_path / "two-policy-source"
    source.mkdir()
    incompatible = configuration.to_dict()
    incompatible["workflow_contract"] = {"name": "local-two-policy-validation"}
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "training_identity": {
                    "artifact_semantics": artifact_semantics("training")
                },
                "resolved_configuration": incompatible,
            }
        ),
        encoding="utf-8",
    )

    report = ReproductionRunner().report_configured_workflow(
        configuration,
        source_run_directories=(source,),
        evaluation_directory=tmp_path / "evaluation",
    )

    assert report.status is RunStatus.FAILED
    assert report.error is not None
    assert "Four-policy corrected pilot source differs" in report.error
