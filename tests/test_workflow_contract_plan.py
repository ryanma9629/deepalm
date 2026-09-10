"""Public CLI planning contract for configuration-defined workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from test_run_skeleton import configuration_data

from deepalm.cli import main
from deepalm.config import (
    ConfigurationError,
    resolve_configuration,
)
from deepalm.config import (
    _resolve_internal_test_configuration as resolve_internal_configuration,
)
from deepalm.planning import build_execution_plan


def test_plan_resolves_the_local_two_policy_contract_and_m5_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A complete config, not the command name, declares the planned workflow."""

    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    configuration = configuration_data(tmp_path)
    configuration.update(
        {
            "workflow_contract": {"name": "local-two-policy-validation"},
            "execution_profile": {"name": "m5-compact"},
            "run_scale": {"profile": "corrected_pilot"},
            "policy": {"names": ["BM^D", "MM"]},
            "optimization": {"device": "mps", "dtype": "float32"},
            "resources": {
                "wall_clock_budget_seconds": 600,
                "process_rss_limit_bytes": 12 * 1024**3,
                "accelerator_memory_limit_bytes": 12 * 1024**3,
            },
        }
    )
    path = tmp_path / "local-two-policy.yaml"
    path.write_text(yaml.safe_dump(configuration), encoding="utf-8")

    assert main(["plan", "--config", str(path), "--format", "json"]) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["workflow_contract"] == "local-two-policy-validation"
    assert plan["execution_profile"] == "m5-compact"
    assert plan["corrected_financial_semantics"] == "corrected-financial-semantics-v2"
    assert [(job["policy"], job["horizon_years"]) for job in plan["jobs"]] == [
        ("BM^D", 5),
        ("BM^D", 15),
        ("MM", 5),
        ("MM", 15),
    ]
    assert plan["resource_budget"]["wall_clock_budget_seconds"] == 600
    assert plan["resolved_configuration"]["optimization"] == {
        "device": "mps",
        "dtype": "float32",
    }


def test_declared_contract_rejects_a_conflicting_policy_matrix(tmp_path: Path) -> None:
    configuration = configuration_data(tmp_path)
    configuration.update(
        {
            "workflow_contract": {"name": "internal-integration-validation"},
            "execution_profile": {"name": "local-cpu-compact"},
            "policy": {"names": ["BM^D", "MM"]},
        }
    )
    configuration["run_scale"] = {"profile": "quick"}

    with pytest.raises(ConfigurationError, match="internal-integration-validation requires run_scale"):
        resolve_internal_configuration(configuration)


def test_configuration_rejects_the_retired_bounded_local_workflow(
    tmp_path: Path,
) -> None:
    configuration = configuration_data(tmp_path)
    configuration["workflow_contract"] = {"name": "bounded-local-workflow"}

    with pytest.raises(ConfigurationError, match="workflow_contract.name is unsupported"):
        resolve_configuration(configuration)


def test_public_configuration_rejects_the_internal_integration_harness(
    tmp_path: Path,
) -> None:
    configuration = configuration_data(tmp_path)

    with pytest.raises(ConfigurationError, match="workflow_contract.name is unsupported"):
        resolve_configuration(configuration)


@pytest.mark.parametrize(
    ("command", "extra_arguments"),
    (
        ("plan", ()),
        ("run", ()),
        ("evaluate", ("--source-run", "missing-source")),
        ("report", ("--source-run", "missing-source")),
        ("preflight", ()),
        ("reference-bank", ()),
        ("device-check", ()),
        ("verify-recovery", ("--source-run", "missing-source")),
    ),
)
def test_internal_integration_harness_has_no_public_cli_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    extra_arguments: tuple[str, ...],
) -> None:
    path = tmp_path / "internal.yaml"
    path.write_text(yaml.safe_dump(configuration_data(tmp_path)), encoding="utf-8")

    exit_code = main([command, "--config", str(path), *extra_arguments])

    assert exit_code == 2
    assert "workflow_contract.name is unsupported" in capsys.readouterr().err


def test_configuration_can_plan_the_declared_research_contract(tmp_path: Path) -> None:
    configuration = configuration_data(tmp_path)
    configuration.update(
        {
            "workflow_contract": {"name": "paper-oriented-research-plan"},
            "execution_profile": {"name": "paper-research"},
            "run_scale": {"profile": "paper_scale"},
            "architecture": {"profile": "paper"},
            "policy": {"names": ["BM^E", "BM^C", "BM^D", "MM"]},
            "acceptance": {
                "purpose": "research",
                "required_status": "methodologically-reproduced",
            },
            "resources": {
                "wall_clock_budget_seconds": 86_400,
                "process_rss_limit_bytes": 32 * 1024**3,
                "accelerator_memory_limit_bytes": 32 * 1024**3,
            },
        }
    )

    plan = build_execution_plan(resolve_configuration(configuration))

    assert plan.workflow_contract == "paper-oriented-research-plan"
    assert plan.execution_profile == "paper-research"
    assert plan.primary_training_jobs == 8
