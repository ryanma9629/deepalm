"""Developer-facing documentation contracts for corrected financial semantics."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from deepalm.cli import main


def test_corrected_semantics_docs_expose_one_current_execution_contract() -> None:
    repository = Path(__file__).resolve().parents[1]
    readme = (repository / "README.md").read_text(encoding="utf-8")
    readme_zh = (repository / "README.zh-CN.md").read_text(encoding="utf-8")
    errata_path = repository / "docs/implementation-errata.md"

    for document in (readme, readme_zh):
        for action in ("plan", "run", "evaluate", "report"):
            assert f"uv run deepalm {action}" in document
        assert "Workflow Contract" in document
        assert "Execution Profile" in document
        assert "corrected-pilot" not in document
        assert "corrected-evaluate" not in document
        assert "corrected-report" not in document
        assert "four-policy-pilot" not in document
        assert "four-policy-evaluate" not in document
        assert "four-policy-report" not in document
        assert "workflow --config" not in document
    assert not (repository / "docs/financial-corrections.md").exists()

    errata = errata_path.read_text(encoding="utf-8")
    for required_section in ("已采纳修正", "明确简化", "能力缺口"):
        assert required_section in errata
    assert "docs/errata.pdf" in errata
    assert "影响边界" in errata
    assert "公开验证" in errata
    assert "Corrected Financial Semantics" in errata
    for action in ("plan", "run", "evaluate", "report"):
        assert f"uv run deepalm {action}" in errata
    assert "configs/local-two-policy-m5.yaml" in errata
    assert "corrected-pilot" not in errata
    assert "corrected-evaluate" not in errata
    assert "corrected-report" not in errata


def test_shipped_configurations_name_scenario_and_environment() -> None:
    repository = Path(__file__).resolve().parents[1]
    configurations = {
        "local-two-policy-m5.yaml": "local-two-policy-validation",
        "local-four-policy-m5.yaml": "local-four-policy-comparison",
        "paper-oriented-research-plan.yaml": "paper-oriented-research-plan",
        "bounded-local-cpu-development.yaml": "bounded-local-workflow",
        "bank-single-gpu-commissioning.yaml": "bank-single-gpu-commissioning",
    }

    for filename, contract in configurations.items():
        raw = yaml.safe_load((repository / "configs" / filename).read_text("utf-8"))
        assert raw["workflow_contract"] == {"name": contract}
        assert raw["execution_profile"]["name"]

    for retired_name in (
        "corrected-pilot.yaml",
        "four-policy-corrected-pilot.yaml",
        "quick-skeleton.yaml",
    ):
        assert not (repository / "configs" / retired_name).exists()

    bank = yaml.safe_load(
        (repository / "configs/bank-single-gpu-commissioning.yaml").read_text("utf-8")
    )
    assert bank["reference_bank"] == {
        "profile": "imported",
        "snapshot_path": "/secure/path/reference-bank.json",
    }


def test_bank_action_help_is_configuration_neutral(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])

    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "configured Reference Bank" in help_text
    assert "canonical Reference Bank" not in help_text
