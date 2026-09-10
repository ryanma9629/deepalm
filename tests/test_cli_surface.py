"""Public action-oriented CLI surface contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from deepalm.cli import main
from deepalm.runner import OperationalRunError, ReproductionRunner

_RETIRED_COMMANDS = (
    "profile",
    "calibrate",
    "workflow",
    "train",
    "accept",
    "corrected-pilot",
    "corrected-evaluate",
    "corrected-report",
    "four-policy-pilot",
    "four-policy-evaluate",
    "four-policy-report",
    "resume",
)


@pytest.mark.parametrize("command", _RETIRED_COMMANDS)
def test_retired_commands_are_not_public(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main([command])

    assert stopped.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_help_exposes_generic_lifecycle_and_unambiguous_independent_actions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])

    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    commands = set(help_text.split("{", 1)[1].split("}", 1)[0].split(","))
    for command in (
        "plan",
        "run",
        "evaluate",
        "report",
        "preflight",
        "reference-bank",
        "device-check",
        "verify-recovery",
    ):
        assert command in commands
    for command in (*_RETIRED_COMMANDS, "bank"):
        assert command not in commands


def test_legacy_bank_alias_is_hidden_but_always_warns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["bank", "--help"])

    assert stopped.value.code == 0
    captured = capsys.readouterr()
    assert "reference-bank" in captured.out
    assert "Deprecated command 'bank'" in captured.err


@pytest.mark.parametrize(
    ("command", "expected_detail"),
    (
        ("run", "only stdout result"),
        ("plan", "without allocating scenarios or models"),
        ("preflight", "regulatory approval"),
        ("reference-bank", "regulatory approval"),
        ("device-check", "regulatory approval"),
        ("evaluate", "locked paths"),
        ("report", "locked evaluation run"),
        ("verify-recovery", "without continuing training"),
    ),
)
def test_action_help_explains_its_evidence_boundary(
    command: str,
    expected_detail: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main([command, "--help"])

    assert stopped.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert expected_detail in help_text
    assert f"Example: deepalm {command}" in help_text


@pytest.mark.parametrize(
    ("filename", "workflow_contract", "execution_profile"),
    (
        ("local-two-policy-m5.yaml", "local-two-policy-validation", "m5-compact"),
        ("local-four-policy-m5.yaml", "local-four-policy-comparison", "m5-compact"),
        (
            "paper-oriented-research-plan.yaml",
            "paper-oriented-research-plan",
            "paper-research",
        ),
        (
            "bank-single-gpu-commissioning.yaml",
            "bank-single-gpu-commissioning",
            "cuda-single-gpu",
        ),
    ),
)
def test_plan_text_summarizes_each_shipped_workflow_contract(
    filename: str,
    workflow_contract: str,
    execution_profile: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    assert main(["plan", "--config", str(repository / "configs" / filename)]) == 0

    output = capsys.readouterr().out
    assert f"Workflow Contract: {workflow_contract}" in output
    assert f"Execution Profile: {execution_profile}" in output
    assert "Supported actions:" in output


def test_plan_json_retains_complete_machine_readable_execution_plan(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    repository = Path(__file__).resolve().parents[1]
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)

    assert (
        main(
            [
                "plan",
                "--config",
                str(repository / "configs" / "local-two-policy-m5.yaml"),
                "--format",
                "json",
            ]
        )
        == 0
    )

    plan = json.loads(capsys.readouterr().out)
    assert plan["workflow_contract"] == "local-two-policy-validation"
    assert plan["execution_profile"] == "m5-compact"
    assert {item["action"] for item in plan["action_capabilities"]} == {
        "plan",
        "run",
        "evaluate",
        "report",
        "preflight",
        "reference-bank",
        "device-check",
        "verify-recovery",
    }
    assert all(
        item["status"] in {"executable", "diagnostic-only"}
        for item in plan["action_capabilities"]
    )


@pytest.mark.parametrize(
    ("filename", "expected_guidance"),
    (
        ("paper-oriented-research-plan.yaml", "formal paper-scale training"),
        ("bank-single-gpu-commissioning.yaml", "full policy training"),
    ),
)
def test_run_rejects_non_executable_workflow_contract_before_publishing_an_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    filename: str,
    expected_guidance: str,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    data = yaml.safe_load((repository / "configs" / filename).read_text("utf-8"))
    data["output"] = {"directory": str(tmp_path / "artifacts"), "run_name": "blocked"}
    config_path = tmp_path / filename
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    assert main(["run", "--config", str(config_path)]) == 2

    captured = capsys.readouterr()
    assert "Action unavailable" in captured.err
    assert expected_guidance in captured.err
    assert "Available actions: plan, preflight, reference-bank, device-check" in captured.err
    assert not (tmp_path / "artifacts" / "blocked").exists()


@pytest.mark.parametrize("command", ("evaluate", "report"))
def test_unavailable_stage_explains_capability_before_requiring_evidence_paths(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    data = yaml.safe_load(
        (repository / "configs" / "bank-single-gpu-commissioning.yaml").read_text(
            "utf-8"
        )
    )
    data["output"] = {"directory": str(tmp_path / "artifacts"), "run_name": "blocked"}
    config_path = tmp_path / "bank.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    exit_code = main([command, "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Action unavailable" in captured.err
    assert "Available actions:" in captured.err


@pytest.mark.parametrize(
    ("command", "extra_arguments", "expected_error"),
    (
        (
            "evaluate",
            ("--source-run", "missing-source"),
            "source run directory does not exist",
        ),
        (
            "report",
            ("--source-run", "missing-source"),
            "requires exactly one --source-run and --evaluation-run",
        ),
    ),
)
def test_local_validation_stage_inputs_fail_before_runner_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    extra_arguments: tuple[str, ...],
    expected_error: str,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    data = yaml.safe_load(
        (repository / "configs" / "local-two-policy-m5.yaml").read_text("utf-8")
    )
    data["output"] = {"directory": str(tmp_path / "artifacts"), "run_name": "pilot"}
    config_path = tmp_path / "local-validation.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    assert main([command, "--config", str(config_path), *extra_arguments]) == 2

    assert expected_error in capsys.readouterr().err
    assert not (tmp_path / "artifacts").exists()


def test_evaluate_rejects_an_existing_but_incomplete_source_as_input_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    data = yaml.safe_load(
        (repository / "configs" / "local-two-policy-m5.yaml").read_text("utf-8")
    )
    data["output"] = {"directory": str(tmp_path / "artifacts"), "run_name": "pilot"}
    config_path = tmp_path / "local-validation.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    source = tmp_path / "incomplete-source"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")

    exit_code = main(
        ["evaluate", "--config", str(config_path), "--source-run", str(source)]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Input error:" in captured.err
    assert "training artifact semantics" in captured.err
    assert not (tmp_path / "artifacts").exists()


def test_report_maps_incompatible_existing_evidence_to_an_input_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    data = yaml.safe_load(
        (repository / "configs" / "local-two-policy-m5.yaml").read_text("utf-8")
    )
    data["output"] = {"directory": str(tmp_path / "artifacts"), "run_name": "pilot"}
    config_path = tmp_path / "local-validation.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    source = tmp_path / "source"
    evaluation = tmp_path / "evaluation"
    source.mkdir()
    evaluation.mkdir()

    def reject_evidence(*_args: object, **_kwargs: object) -> None:
        raise OperationalRunError("locked evaluation belongs to another workflow")

    monkeypatch.setattr(
        "deepalm.cli.validate_configured_workflow_stage_inputs", reject_evidence
    )
    monkeypatch.setattr(
        ReproductionRunner,
        "report_configured_workflow",
        lambda *_args, **_kwargs: pytest.fail("invalid input must not reach the runner"),
    )

    exit_code = main(
        [
            "report",
            "--config",
            str(config_path),
            "--source-run",
            str(source),
            "--evaluation-run",
            str(evaluation),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Input error: locked evaluation belongs to another workflow" in captured.err
    assert not (tmp_path / "artifacts").exists()
