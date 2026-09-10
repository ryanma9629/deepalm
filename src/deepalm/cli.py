"""Command-line adapter for the Deep ALM reproduction runner."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from deepalm.config import ConfigurationError, load_configuration
from deepalm.planning import build_execution_plan
from deepalm.runner import ReproductionRunner, RunStatus


def main(argv: Sequence[str] | None = None) -> int:
    """Run a configured reproduction and return a process exit status."""

    parser = argparse.ArgumentParser(prog="deepalm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="create an auditable run bundle")
    run_parser.add_argument(
        "--config", type=Path, required=True, help="YAML run configuration"
    )
    run_parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show per-epoch training progress (default: enabled)",
    )
    plan_parser = subparsers.add_parser(
        "plan", help="show bounded work before any market or training stage runs"
    )
    plan_parser.add_argument(
        "--config", type=Path, required=True, help="YAML run configuration"
    )
    preflight_parser = subparsers.add_parser(
        "preflight", help="run bounded HJM calibration and scenario generation only"
    )
    preflight_parser.add_argument(
        "--config", type=Path, required=True, help="YAML run configuration"
    )
    bank_parser = subparsers.add_parser(
        "bank", help="build or import the configured Reference Bank and reviewable Table 1"
    )
    bank_parser.add_argument(
        "--config", type=Path, required=True, help="YAML run configuration"
    )
    device_check_parser = subparsers.add_parser(
        "device-check",
        help="run bounded actual CPU/MPS/CUDA commissioning updates",
    )
    device_check_parser.add_argument(
        "--config", type=Path, required=True, help="YAML run configuration"
    )
    device_check_parser.add_argument(
        "--policy",
        choices=("BM^E", "MM"),
        default="BM^E",
        help="benchmark or MM path to commission; MM includes a frozen BM^D dependency",
    )
    device_check_parser.add_argument(
        "--horizon",
        type=int,
        choices=(5, 15),
        default=5,
        help="full 5- or 15-year rollout used by the commissioning update",
    )
    report_parser = subparsers.add_parser(
        "report",
        help="generate a compact no-swap local report from completed run bundles",
    )
    report_parser.add_argument(
        "--config", type=Path, required=True, help="YAML report configuration"
    )
    report_parser.add_argument(
        "--source-run",
        type=Path,
        action="append",
        required=True,
        help="completed source run directory; repeat to combine independent evidence",
    )
    report_parser.add_argument(
        "--evaluation-run",
        type=Path,
        help="completed locked evaluation artifact required by the selected contract",
    )
    for command, help_text in (
        ("resume", "validate compatible recovery evidence without rerunning the matrix"),
        ("evaluate", "reuse compatible locked evaluation evidence"),
    ):
        stage_parser = subparsers.add_parser(command, help=help_text)
        stage_parser.add_argument(
            "--config", type=Path, required=True, help="YAML local workflow configuration"
        )
        stage_parser.add_argument(
            "--source-run", type=Path, required=True, help="completed compatible workflow"
        )
    arguments = parser.parse_args(argv)

    try:
        configuration = load_configuration(arguments.config)
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    if arguments.command == "plan":
        print(json.dumps(build_execution_plan(configuration).to_dict(), indent=2))
        return 0

    runner = ReproductionRunner()
    if arguments.command == "preflight":
        bundle = runner.preflight_market(configuration)
    elif arguments.command == "bank":
        bundle = runner.build_reference_bank(configuration)
    elif arguments.command == "device-check":
        bundle = runner.commission_single_device_portability(
            configuration,
            horizon_years=arguments.horizon,
            policy_name=arguments.policy,
        )
    elif arguments.command == "report":
        bundle = runner.report_configured_workflow(
            configuration,
            source_run_directories=tuple(arguments.source_run),
            evaluation_directory=arguments.evaluation_run,
        )
    elif arguments.command == "evaluate":
        bundle = runner.evaluate_configured_workflow(
            configuration, source_run_directory=arguments.source_run
        )
    elif arguments.command == "resume":
        bundle = runner.reuse_completed_workflow_stage(
            configuration,
            source_run_directory=arguments.source_run,
            stage=arguments.command,
        )
    else:
        bundle = runner.run_configured_workflow(
            configuration, verbose=arguments.verbose
        )
    if bundle.status is RunStatus.COMPLETED:
        assert bundle.artifact_directory is not None
        print(bundle.artifact_directory)
        return 0
    if bundle.status is RunStatus.ACCEPTANCE_FAILED:
        print(
            "Run completed but did not meet acceptance requirements.", file=sys.stderr
        )
        return 3
    if bundle.status is RunStatus.INCOMPLETE:
        print(f"Run incomplete: {bundle.error}", file=sys.stderr)
        return 4

    print(f"Operational failure: {bundle.error}", file=sys.stderr)
    return 1
