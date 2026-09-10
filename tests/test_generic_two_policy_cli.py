"""Generic public CLI lifecycle for the two-policy Workflow Contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from test_corrected_pilot import _corrected_pilot_data
from test_four_policy_pilot import _four_policy_pilot_data

from deepalm.cli import main
from deepalm.runner import AcceptanceStatus, ReproductionRunner, RunBundle, RunStatus


@pytest.mark.parametrize(
    "configuration_factory",
    [_corrected_pilot_data, _four_policy_pilot_data],
)
def test_generic_actions_dispatch_local_validation_workflow_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    configuration_factory: object,
) -> None:
    """Actions choose work; the configuration is the only scenario selector."""

    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    assert callable(configuration_factory)
    data = configuration_factory(tmp_path)
    config_path = tmp_path / "local-validation.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    source = tmp_path / "source"
    evaluation = tmp_path / "evaluation"
    artifact = tmp_path / "artifact"
    calls: list[tuple[str, tuple[Path, ...]]] = []

    assert main(["plan", "--config", str(config_path)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["workflow_contract"] == data["workflow_contract"]["name"]
    assert plan["execution_profile"] == data["execution_profile"]["name"]
    assert len(plan["jobs"]) == len(data["policy"]["names"]) * 2
    assert plan["primary_optimizer_updates"] == len(plan["jobs"]) * 4

    def completed(_: ReproductionRunner, *paths: Path) -> RunBundle:
        calls.append(("called", paths))
        return RunBundle(
            status=RunStatus.COMPLETED,
            acceptance_status=AcceptanceStatus.PENDING,
            artifact_directory=artifact,
        )

    monkeypatch.setattr(
        ReproductionRunner,
        "run_configured_workflow",
        lambda self, _, *, verbose: completed(self),
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


@pytest.mark.parametrize(
    ("verbosity_argument", "expected_verbose"),
    (((), True), (("--verbose",), True), (("--no-verbose",), False)),
)
def test_run_passes_an_explicit_or_default_verbose_preference_to_the_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verbosity_argument: tuple[str, ...],
    expected_verbose: bool,
) -> None:
    """The generic run action owns the human-facing progress preference."""

    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    config_path = tmp_path / "local-validation.yaml"
    config_path.write_text(
        yaml.safe_dump(_corrected_pilot_data(tmp_path)), encoding="utf-8"
    )
    observed: list[bool] = []
    artifact = tmp_path / "artifact"

    def completed(
        _: ReproductionRunner, _configuration: object, *, verbose: bool
    ) -> RunBundle:
        observed.append(verbose)
        return RunBundle(
            status=RunStatus.COMPLETED,
            acceptance_status=AcceptanceStatus.PENDING,
            artifact_directory=artifact,
        )

    monkeypatch.setattr(ReproductionRunner, "run_configured_workflow", completed)

    assert main(["run", "--config", str(config_path), *verbosity_argument]) == 0
    assert observed == [expected_verbose]
