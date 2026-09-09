"""Generic public CLI lifecycle for the two-policy Workflow Contract."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from test_corrected_pilot import _corrected_pilot_data

from deepalm.cli import main
from deepalm.runner import AcceptanceStatus, ReproductionRunner, RunBundle, RunStatus


def test_generic_actions_dispatch_the_two_policy_workflow_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Actions choose work; the configuration is the only scenario selector."""

    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    data = _corrected_pilot_data(tmp_path)
    config_path = tmp_path / "local-two-policy.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    source = tmp_path / "source"
    evaluation = tmp_path / "evaluation"
    artifact = tmp_path / "artifact"
    calls: list[tuple[str, tuple[Path, ...]]] = []

    def completed(_: ReproductionRunner, *paths: Path) -> RunBundle:
        calls.append(("called", paths))
        return RunBundle(
            status=RunStatus.COMPLETED,
            acceptance_status=AcceptanceStatus.PENDING,
            artifact_directory=artifact,
        )

    monkeypatch.setattr(
        ReproductionRunner, "run_configured_workflow", lambda self, _: completed(self)
    )
    monkeypatch.setattr(
        ReproductionRunner,
        "evaluate_configured_workflow",
        lambda self, _, *, source_run_directory: completed(self, source_run_directory),
    )
    monkeypatch.setattr(
        ReproductionRunner,
        "report_configured_workflow",
        lambda self, _, *, source_run_directories, evaluation_directory: completed(
            self, *source_run_directories, evaluation_directory
        ),
    )

    assert main(["run", "--config", str(config_path)]) == 0
    assert main(
        ["evaluate", "--config", str(config_path), "--source-run", str(source)]
    ) == 0
    assert main(
        [
            "report",
            "--config",
            str(config_path),
            "--source-run",
            str(source),
            "--evaluation-run",
            str(evaluation),
        ]
    ) == 0

    assert calls == [
        ("called", ()),
        ("called", (source,)),
        ("called", (source, evaluation)),
    ]
    assert capsys.readouterr().out.splitlines() == [str(artifact)] * 3
