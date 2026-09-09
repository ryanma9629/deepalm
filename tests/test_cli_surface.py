"""Public action-oriented CLI surface contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepalm.cli import main

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
)


@pytest.mark.parametrize("command", _RETIRED_COMMANDS)
def test_retired_commands_are_not_public(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main([command])

    assert stopped.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_help_exposes_only_generic_lifecycle_and_independent_actions(
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
        "bank",
        "device-check",
        "resume",
    ):
        assert command in commands
    for command in _RETIRED_COMMANDS:
        assert command not in commands


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
            "bounded-local-cpu-development.yaml",
            "bounded-local-workflow",
            "local-cpu-compact",
        ),
        (
            "bank-single-gpu-commissioning.yaml",
            "bank-single-gpu-commissioning",
            "cuda-single-gpu",
        ),
    ),
)
def test_plan_smoke_covers_each_shipped_workflow_contract(
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

    plan = json.loads(capsys.readouterr().out)
    assert plan["workflow_contract"] == workflow_contract
    assert plan["execution_profile"] == execution_profile
