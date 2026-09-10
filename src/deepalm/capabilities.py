"""Workflow Contract capabilities presented by the public CLI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from deepalm.config import ResolvedRunConfiguration
from deepalm.planning import ExecutionPlan


class ActionCapabilityStatus(StrEnum):
    """Whether a public action has a meaningful implementation today."""

    EXECUTABLE = "executable"
    DIAGNOSTIC_ONLY = "diagnostic-only"
    PLANNING_ONLY = "planning-only"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class WorkflowActionCapability:
    """One public action's capability for a resolved Workflow Contract."""

    action: str
    status: ActionCapabilityStatus
    guidance: str
    requires_locked_evaluation: bool = False

    @property
    def can_execute(self) -> bool:
        """Whether the action may enter its implementation."""

        return self.status in {
            ActionCapabilityStatus.EXECUTABLE,
            ActionCapabilityStatus.DIAGNOSTIC_ONLY,
        }

    def to_dict(self) -> dict[str, str | bool]:
        """Return the stable machine-readable capability representation."""

        return {
            "action": self.action,
            "status": self.status.value,
            "guidance": self.guidance,
            "requires_locked_evaluation": self.requires_locked_evaluation,
        }


@dataclass(frozen=True)
class PublicActionHelp:
    """Configuration-neutral help for one visible CLI action."""

    summary: str
    description: str
    example: str


_DIAGNOSTIC_DISCLAIMER = (
    "It does not train a full policy matrix or establish economic validity, "
    "production readiness, regulatory approval, or multi-GPU support."
)

_PUBLIC_ACTION_HELP = {
    "run": PublicActionHelp(
        summary="execute a configured workflow when its Workflow Contract supports training",
        description=(
            "Execute the complete policy/horizon matrix declared by an executable "
            "Workflow Contract. Progress goes to stderr and the completed artifact "
            "directory is the only stdout result."
        ),
        example="deepalm run --config configs/local-two-policy-m5.yaml",
    ),
    "plan": PublicActionHelp(
        summary="show capability, bounded work, and resources before execution",
        description=(
            "Resolve the experiment without allocating scenarios or models. Text is "
            "human-readable; JSON includes the complete plan and action capabilities."
        ),
        example="deepalm plan --config configs/local-two-policy-m5.yaml",
    ),
    "preflight": PublicActionHelp(
        summary="run bounded HJM calibration and scenario generation only",
        description=f"Create bounded market-calibration and scenario evidence. {_DIAGNOSTIC_DISCLAIMER}",
        example="deepalm preflight --config configs/local-two-policy-m5.yaml",
    ),
    "reference-bank": PublicActionHelp(
        summary="build or validate the configured Reference Bank and reviewable Table 1",
        description=f"Create bounded Reference Bank evidence. {_DIAGNOSTIC_DISCLAIMER}",
        example="deepalm reference-bank --config configs/local-two-policy-m5.yaml",
    ),
    "device-check": PublicActionHelp(
        summary="create bounded CPU/MPS/CUDA commissioning evidence",
        description=(
            "Select one update from the configuration's declared policy/horizon matrix. "
            f"{_DIAGNOSTIC_DISCLAIMER}"
        ),
        example=(
            "deepalm device-check --config configs/bank-single-gpu-commissioning.yaml "
            "--policy MM --horizon 5"
        ),
    ),
    "evaluate": PublicActionHelp(
        summary="evaluate a compatible completed workflow on locked paths",
        description=(
            "Evaluate a compatible completed workflow on locked paths. Progress goes "
            "to stderr and the completed evaluation directory is the stdout result."
        ),
        example=(
            "deepalm evaluate --config configs/local-two-policy-m5.yaml "
            "--source-run artifacts/corrected-local-validation-pilot"
        ),
    ),
    "report": PublicActionHelp(
        summary="generate a report from completed run and locked-evaluation evidence",
        description=(
            "Generate a report from compatible completed evidence. Local validation "
            "contracts require exactly one source run and its locked evaluation run."
        ),
        example=(
            "deepalm report --config configs/local-two-policy-m5.yaml "
            "--source-run artifacts/corrected-local-validation-pilot "
            "--evaluation-run artifacts/corrected-local-validation-pilot-evaluation"
        ),
    ),
    "verify-recovery": PublicActionHelp(
        summary="validate compatible recovery evidence; does not continue training",
        description=(
            "Validate compatible recovery evidence without continuing training or "
            "publishing a replacement policy matrix."
        ),
        example=(
            "deepalm verify-recovery --config configs/local-two-policy-m5.yaml "
            "--source-run artifacts/corrected-local-validation-pilot"
        ),
    ),
}


def public_action_help(action: str) -> PublicActionHelp:
    """Return the single source of configuration-neutral action help."""

    try:
        return _PUBLIC_ACTION_HELP[action]
    except KeyError as error:
        raise ValueError(f"Unknown Deep ALM action: {action}") from error


_COMMON_DIAGNOSTICS = {
    "preflight": WorkflowActionCapability(
        action="preflight",
        status=ActionCapabilityStatus.DIAGNOSTIC_ONLY,
        guidance="creates bounded market evidence; it does not train a policy matrix",
    ),
    "reference-bank": WorkflowActionCapability(
        action="reference-bank",
        status=ActionCapabilityStatus.DIAGNOSTIC_ONLY,
        guidance="builds or validates Reference Bank evidence; it does not train a policy matrix",
    ),
    "device-check": WorkflowActionCapability(
        action="device-check",
        status=ActionCapabilityStatus.DIAGNOSTIC_ONLY,
        guidance="commissions bounded device updates; it is not production acceptance",
    ),
    "verify-recovery": WorkflowActionCapability(
        action="verify-recovery",
        status=ActionCapabilityStatus.DIAGNOSTIC_ONLY,
        guidance="validates compatible recovery evidence without continuing training",
    ),
}


def workflow_action_capabilities(
    configuration: ResolvedRunConfiguration,
) -> tuple[WorkflowActionCapability, ...]:
    """Return the complete public action inventory for one Workflow Contract."""

    return _lifecycle_capabilities(configuration.workflow_contract.name) + tuple(
        _COMMON_DIAGNOSTICS.values()
    )


def workflow_action_capability(
    configuration: ResolvedRunConfiguration, action: str
) -> WorkflowActionCapability:
    """Return one public action capability or reject an unknown CLI action."""

    for capability in workflow_action_capabilities(configuration):
        if capability.action == action:
            return capability
    raise ValueError(f"Unknown Deep ALM action: {action}")


def render_execution_plan(
    plan: ExecutionPlan, configuration: ResolvedRunConfiguration
) -> str:
    """Render a compact human plan without hiding the machine-readable form."""

    budget = plan.resource_budget
    capability_lines = "\n".join(
        f"  {item.action}: {item.status} — {item.guidance}"
        for item in workflow_action_capabilities(configuration)
    )
    policies = ", ".join(configuration.policy.names)
    horizons = ", ".join(f"{years}y" for years in configuration.experiment.horizons_years)
    return "\n".join(
        (
            "Deep ALM execution plan",
            f"Workflow Contract: {plan.workflow_contract}",
            f"Execution Profile: {plan.execution_profile}",
            f"Purpose: {plan.purpose}",
            f"TreasuryPolicy matrix: {policies} × {horizons}",
            f"Training jobs: {plan.primary_training_jobs}",
            f"Optimizer updates: {plan.primary_optimizer_updates}",
            f"Artifact directory: {configuration.output.directory / configuration.output.run_name}",
            (
                "Resource guard: "
                f"{budget['wall_clock_budget_seconds']}s wall clock; "
                f"{budget['process_rss_limit_bytes']} B RSS; "
                f"{budget['accelerator_memory_limit_bytes']} B accelerator memory"
            ),
            "Supported actions:",
            capability_lines,
        )
    )


def _lifecycle_capabilities(
    contract: str,
) -> tuple[WorkflowActionCapability, ...]:
    if contract in {
        "local-two-policy-validation",
        "local-four-policy-comparison",
    }:
        return _executable_lifecycle()
    if contract == "paper-oriented-research-plan":
        return _limited_lifecycle(
            plan_guidance="declares the paper-scale target; formal research training is not delivered",
            run_guidance="formal paper-scale training is not delivered",
            evaluation_guidance="run is unavailable until formal paper-scale training exists",
        )
    if contract == "bank-single-gpu-commissioning":
        return _limited_lifecycle(
            plan_guidance="declares bank commissioning inputs and resource bounds",
            run_guidance="imported Reference Bank full policy training is not delivered",
            evaluation_guidance="full policy training is unavailable for this commissioning contract",
        )
    raise ValueError(f"Unsupported Workflow Contract: {contract}")


def _executable_lifecycle() -> tuple[WorkflowActionCapability, ...]:
    return (
        WorkflowActionCapability(
            "plan",
            ActionCapabilityStatus.EXECUTABLE,
            "declares work and resource bounds without allocating scenarios or models",
        ),
        WorkflowActionCapability(
            "run",
            ActionCapabilityStatus.EXECUTABLE,
            "executes the configured workflow and publishes an auditable run bundle",
        ),
        WorkflowActionCapability(
            "evaluate",
            ActionCapabilityStatus.EXECUTABLE,
            "requires a compatible completed source run and produces locked evaluation evidence",
        ),
        WorkflowActionCapability(
            "report",
            ActionCapabilityStatus.EXECUTABLE,
            "requires compatible completed evidence and produces a report bundle",
            requires_locked_evaluation=True,
        ),
    )


def _limited_lifecycle(
    *, plan_guidance: str, run_guidance: str, evaluation_guidance: str
) -> tuple[WorkflowActionCapability, ...]:
    return (
        WorkflowActionCapability("plan", ActionCapabilityStatus.PLANNING_ONLY, plan_guidance),
        WorkflowActionCapability("run", ActionCapabilityStatus.UNAVAILABLE, run_guidance),
        WorkflowActionCapability(
            "evaluate", ActionCapabilityStatus.UNAVAILABLE, evaluation_guidance
        ),
        WorkflowActionCapability(
            "report", ActionCapabilityStatus.UNAVAILABLE, evaluation_guidance
        ),
    )
