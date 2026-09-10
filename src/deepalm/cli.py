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
    run_parser = subparsers.add_parser(
        "run",
        help="execute a configured workflow when its Workflow Contract supports training",
        description=(
            "Execute the complete policy/horizon matrix declared by an executable "
            "Workflow Contract. Progress goes to stderr and the completed artifact "
            "directory is the only stdout result."
        ),
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
    plan_parser = subparsers.add_parser(
        "plan",
        help="show capability, bounded work, and resources before execution",
        description=(
            "Resolve the experiment without allocating scenarios or models. Text is "
            "human-readable; JSON includes the complete plan and action capabilities."
        ),
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
    preflight_parser = subparsers.add_parser(
        "preflight",
        help="run bounded HJM calibration and scenario generation only",
        description=(
            "Create bounded market-calibration and scenario evidence only. This does "
            "not train a policy matrix or establish economic acceptance."
        ),
    )
    preflight_parser.add_argument(
        "--config", type=Path, required=True, help="YAML run configuration"
    )
    bank_parser = subparsers.add_parser(
        "reference-bank",
        help="build or validate the configured Reference Bank and reviewable Table 1",
        description=(
            "Build or validate the Reference Bank selected by configuration and write "
            "reviewable balance-sheet evidence. This does not train policies."
        ),
    )
    bank_parser.add_argument(
        "--config", type=Path, required=True, help="YAML experiment configuration"
    )
    device_check_parser = subparsers.add_parser(
        "device-check",
        help="create bounded CPU/MPS/CUDA commissioning evidence; does not train a policy matrix",
        description=(
            "Run one bounded update selected from the configuration's declared policy "
            "and horizon matrix. This is device evidence, not production acceptance."
        ),
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
        help="generate a report from compatible completed run and locked-evaluation evidence",
        description=(
            "Generate a report from compatible completed evidence. Local validation "
            "contracts require exactly one source run and its locked evaluation run."
        ),
    )
    report_parser.add_argument(
        "--config", type=Path, required=True, help="YAML experiment configuration"
    )
    report_parser.add_argument(
        "--source-run",
        type=Path,
        action="append",
        required=True,
        help="compatible completed source run; repeat only where the Workflow Contract permits it",
    )
    report_parser.add_argument(
        "--evaluation-run",
        type=Path,
        help="compatible locked evaluation artifact; required by local validation contracts",
    )
    for command, help_text, description in (
        (
            "verify-recovery",
            "validate compatible recovery evidence; does not continue training",
            (
                "Validate compatible recovery evidence without continuing training or "
                "publishing a replacement policy matrix."
            ),
        ),
        (
            "evaluate",
            "evaluate a compatible completed workflow on locked paths",
            (
                "Evaluate a compatible completed workflow on locked paths. Progress goes "
                "to stderr and the completed evaluation directory is the stdout result."
            ),
        ),
    ):
        stage_parser = subparsers.add_parser(
            command, help=help_text, description=description
        )
        stage_parser.add_argument(
            "--config", type=Path, required=True, help="YAML experiment configuration"
        )
        stage_parser.add_argument(
            "--source-run", type=Path, required=True, help="compatible completed workflow artifact"
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
        bundle = runner.report_configured_workflow(
            configuration,
            source_run_directories=tuple(arguments.source_run),
            evaluation_directory=arguments.evaluation_run,
        )
    elif arguments.command == "evaluate":
        bundle = runner.evaluate_configured_workflow(
            configuration,
            source_run_directory=arguments.source_run,
            verbose=arguments.verbose,
        )
    elif arguments.command == "verify-recovery":
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

    if arguments.command == "evaluate":
        if not arguments.source_run.is_dir():
            return f"source run directory does not exist: {arguments.source_run}"
        source_run_directory = arguments.source_run
        evaluation_directory = None
    elif arguments.command == "report" and capability.requires_locked_evaluation:
        if len(arguments.source_run) != 1 or arguments.evaluation_run is None:
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
