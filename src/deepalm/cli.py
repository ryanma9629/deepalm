"""Command-line adapter for the Deep ALM reproduction runner."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from deepalm.capabilities import (
    ActionCapabilityStatus,
    WorkflowActionCapability,
    public_action_help,
    render_execution_plan,
    workflow_action_capabilities,
    workflow_action_capability,
)
from deepalm.config import (
    ConfigurationError,
    ResolvedRunConfiguration,
    load_configuration,
)
from deepalm.planning import build_execution_plan
from deepalm.runner import (
    OperationalRunError,
    ReproductionRunner,
    RunStatus,
    validate_configured_workflow_stage_inputs,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a configured reproduction and return a process exit status."""

    supplied_argv = list(sys.argv[1:] if argv is None else argv)
    uses_legacy_bank_alias = bool(supplied_argv and supplied_argv[0] == "bank")
    if uses_legacy_bank_alias:
        supplied_argv[0] = "reference-bank"
        print("Deprecated command 'bank'; use 'reference-bank' instead.", file=sys.stderr)

    parser = argparse.ArgumentParser(
        prog="deepalm",
        description="Configuration-defined Deep ALM workflow actions.",
        epilog=(
            "Typical local flow:\n"
            "  deepalm plan --config configs/local-two-policy-m5.yaml\n"
            "  deepalm run --config configs/local-two-policy-m5.yaml\n"
            "  deepalm evaluate --config configs/local-two-policy-m5.yaml "
            "--source-run artifacts/corrected-local-validation-pilot\n"
            "See README for four-policy, paper-oriented, and bank paths."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_help = public_action_help("run")
    run_parser = subparsers.add_parser(
        "run",
        help=run_help.summary,
        description=run_help.description,
        epilog=f"Example:\n  {run_help.example}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="experiment YAML: Workflow Contract, Execution Profile, inputs, and output",
    )
    run_parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show per-epoch training progress (default: enabled)",
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing artifact only after a successful run",
    )
    plan_help = public_action_help("plan")
    plan_parser = subparsers.add_parser(
        "plan",
        help=plan_help.summary,
        description=plan_help.description,
        epilog=f"Example:\n  {plan_help.example}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    plan_parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="experiment YAML: Workflow Contract, Execution Profile, inputs, and output",
    )
    plan_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="plan output format (default: text; json retains the complete resolved plan)",
    )
    preflight_help = public_action_help("preflight")
    preflight_parser = subparsers.add_parser(
        "preflight",
        help=preflight_help.summary,
        description=preflight_help.description,
        epilog=f"Example:\n  {preflight_help.example}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    preflight_parser.add_argument(
        "--config", type=Path, required=True, help="YAML run configuration"
    )
    bank_help = public_action_help("reference-bank")
    bank_parser = subparsers.add_parser(
        "reference-bank",
        help=bank_help.summary,
        description=bank_help.description,
        epilog=f"Example:\n  {bank_help.example}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bank_parser.add_argument(
        "--config", type=Path, required=True, help="YAML experiment configuration"
    )
    device_help = public_action_help("device-check")
    device_check_parser = subparsers.add_parser(
        "device-check",
        help=device_help.summary,
        description=device_help.description,
        epilog=f"Example:\n  {device_help.example}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    report_help = public_action_help("report")
    report_parser = subparsers.add_parser(
        "report",
        help=report_help.summary,
        description=report_help.description,
        epilog=f"Example:\n  {report_help.example}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    report_parser.add_argument(
        "--config", type=Path, required=True, help="YAML experiment configuration"
    )
    report_parser.add_argument(
        "--source-run",
        type=Path,
        action="append",
        help="compatible completed source run; repeat only where the Workflow Contract permits it",
    )
    report_parser.add_argument(
        "--evaluation-run",
        type=Path,
        help="compatible locked evaluation artifact; required by local validation contracts",
    )
    for command in ("verify-recovery", "evaluate"):
        action_help = public_action_help(command)
        stage_parser = subparsers.add_parser(
            command,
            help=action_help.summary,
            description=action_help.description,
            epilog=f"Example:\n  {action_help.example}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        stage_parser.add_argument(
            "--config", type=Path, required=True, help="YAML experiment configuration"
        )
        stage_parser.add_argument(
            "--source-run", type=Path, help="compatible completed workflow artifact"
        )
        if command == "evaluate":
            stage_parser.add_argument(
                "--verbose",
                action=argparse.BooleanOptionalAction,
                default=True,
                help="show evaluation-stage progress (default: enabled)",
            )
    arguments = parser.parse_args(supplied_argv)

    try:
        configuration = load_configuration(arguments.config)
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    if arguments.command == "plan":
        plan = build_execution_plan(configuration)
        capability = workflow_action_capability(configuration, "plan")
        if capability.status is ActionCapabilityStatus.UNAVAILABLE:
            print(_unavailable_action_message(configuration, capability), file=sys.stderr)
            return 2
        if arguments.format == "json":
            output = plan.to_dict()
            output["action_capabilities"] = [
                item.to_dict()
                for item in workflow_action_capabilities(configuration)
            ]
            print(json.dumps(output, indent=2))
        else:
            print(render_execution_plan(plan, configuration))
        return 0

    capability = workflow_action_capability(configuration, arguments.command)
    if not capability.can_execute:
        print(_unavailable_action_message(configuration, capability), file=sys.stderr)
        return 2

    if input_error := _stage_input_error(arguments, capability, configuration):
        print(f"Input error: {input_error}", file=sys.stderr)
        return 2

    runner = ReproductionRunner()
    if arguments.command == "preflight":
        bundle = runner.preflight_market(configuration)
    elif arguments.command == "reference-bank":
        bundle = runner.build_reference_bank(configuration)
    elif arguments.command == "device-check":
        bundle = runner.commission_single_device_portability(
            configuration,
            horizon_years=arguments.horizon,
            policy_name=arguments.policy,
        )
    elif arguments.command == "report":
        assert arguments.source_run is not None
        bundle = runner.report_configured_workflow(
            configuration,
            source_run_directories=tuple(arguments.source_run),
            evaluation_directory=arguments.evaluation_run,
        )
    elif arguments.command == "evaluate":
        assert arguments.source_run is not None
        bundle = runner.evaluate_configured_workflow(
            configuration,
            source_run_directory=arguments.source_run,
            verbose=arguments.verbose,
        )
    elif arguments.command == "verify-recovery":
        assert arguments.source_run is not None
        bundle = runner.reuse_completed_workflow_stage(
            configuration,
            source_run_directory=arguments.source_run,
            stage="resume",
        )
    else:
        bundle = runner.run_configured_workflow(
            configuration, verbose=arguments.verbose, overwrite=arguments.overwrite
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


def _unavailable_action_message(
    configuration: ResolvedRunConfiguration,
    capability: WorkflowActionCapability,
) -> str:
    """Name an unavailable action and the meaningful alternatives."""

    available = ", ".join(
        item.action
        for item in workflow_action_capabilities(configuration)
        if item.status is not ActionCapabilityStatus.UNAVAILABLE
    )
    suffix = f" Available actions: {available}." if available else ""
    return (
        f"Action unavailable for {configuration.workflow_contract.name}: "
        f"{capability.action} — {capability.guidance}.{suffix}"
    )


def _stage_input_error(
    arguments: argparse.Namespace,
    capability: WorkflowActionCapability,
    configuration: ResolvedRunConfiguration,
) -> str | None:
    """Reject missing local-validation evidence before a runner can publish failure data."""

    if (
        arguments.command in {"evaluate", "verify-recovery"}
        and arguments.source_run is None
    ):
        return f"{arguments.command} requires --source-run"
    if arguments.command == "verify-recovery":
        return None
    if arguments.command == "evaluate":
        if not arguments.source_run.is_dir():
            return f"source run directory does not exist: {arguments.source_run}"
        source_run_directory = arguments.source_run
        evaluation_directory = None
    elif arguments.command == "report" and capability.requires_locked_evaluation:
        if (
            arguments.source_run is None
            or len(arguments.source_run) != 1
            or arguments.evaluation_run is None
        ):
            return "local validation report requires exactly one --source-run and --evaluation-run"
        if not arguments.source_run[0].is_dir():
            return f"source run directory does not exist: {arguments.source_run[0]}"
        if not arguments.evaluation_run.is_dir():
            return f"evaluation run directory does not exist: {arguments.evaluation_run}"
        source_run_directory = arguments.source_run[0]
        evaluation_directory = arguments.evaluation_run
    else:
        return None
    try:
        validate_configured_workflow_stage_inputs(
            configuration,
            action=arguments.command,
            source_run_directory=source_run_directory,
            evaluation_directory=evaluation_directory,
        )
    except (OperationalRunError, OSError, ValueError, KeyError) as error:
        return str(error)
    return None
