"""Compact, auditable no-swap reporting from completed local evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepalm.config import ResolvedRunConfiguration
from deepalm.reference_bank import ReferenceBankProvider
from deepalm.term_structures import (
    HullWhiteConfiguration,
    MarketScenarioModel,
)


class ReportingError(ValueError):
    """Raised when completed evidence cannot support an honest local report."""


@dataclass(frozen=True)
class CompactReportArtifacts:
    """JSON-ready report artifacts written together by the runner."""

    report: dict[str, object]
    coverage_inventory: dict[str, object]
    calibration_evidence: dict[str, object]
    market_comparison: dict[str, object]
    reference_bank_summary: dict[str, object]


@dataclass(frozen=True)
class PairedPilotReportArtifacts:
    """The immutable evidence assembled into the bounded paired-pilot report."""

    report: dict[str, object]


def build_paired_convention_pilot_report(
    *, pilot_run_directory: Path, evaluation_directory: Path
) -> PairedPilotReportArtifacts:
    """Build a descriptive report from completed, identity-linked pilot evidence."""

    pilot_manifest_path = pilot_run_directory / "manifest.json"
    evaluation_path = evaluation_directory / "paired-evaluation.json"
    pilot_manifest = _read_report_object(pilot_manifest_path, "pilot manifest")
    evaluation = _read_report_object(evaluation_path, "paired evaluation")
    pilot = pilot_manifest.get("paired_convention_pilot")
    if (
        pilot_manifest.get("status") != "completed"
        or not isinstance(pilot, dict)
        or pilot.get("status") != "completed"
        or pilot.get("label") != "paired-convention-research-pilot"
    ):
        raise ReportingError("Paired-pilot report requires a completed pilot bundle")
    if (
        evaluation.get("format_version") != 1
        or evaluation.get("kind") != "paired-convention-pilot-evaluation"
        or evaluation.get("status") != "completed"
        or evaluation.get("label") != "paired-convention-research-pilot"
        or evaluation.get("source_run") != str(pilot_run_directory.resolve())
        or evaluation.get("source_manifest_sha256") != _sha256(pilot_manifest_path)
    ):
        raise ReportingError("Paired evaluation is not identity-linked to the completed pilot")
    jobs = pilot.get("completed_training_jobs")
    reports = evaluation.get("reports")
    locked_manifests = evaluation.get("locked_evaluation_manifests")
    truncations = evaluation.get("mm_fifteen_year_truncation")
    intervals = evaluation.get("paired_intervals")
    if not all(
        isinstance(value, dict) for value in (reports, locked_manifests, truncations)
    ) or not isinstance(jobs, list) or not isinstance(intervals, list):
        raise ReportingError("Paired evaluation lacks report, identity, or interval evidence")
    conventions: dict[str, dict[str, object]] = {}
    for convention in ("paper", "corrected"):
        convention_jobs = [
            job for job in jobs if isinstance(job, dict) and job.get("convention") == convention
        ]
        convention_reports = reports.get(convention)
        locked = locked_manifests.get(convention)
        truncation = truncations.get(convention)
        expected_labels = {f"{policy}-{horizon}y" for policy in ("BM^D", "MM") for horizon in (5, 15)}
        if (
            len(convention_jobs) != 4
            or not isinstance(convention_reports, dict)
            or set(convention_reports) != expected_labels
            or not isinstance(locked, dict)
            or not isinstance(truncation, dict)
        ):
            raise ReportingError(f"Paired report lacks complete {convention} evidence")
        formula_choices = (
            {
                "pca_loading_scale": "eigenvalue",
                "loan_interest_annualization": "unannualized",
            }
            if convention == "paper"
            else {
                "pca_loading_scale": "sqrt_eigenvalue",
                "loan_interest_annualization": "monthly",
            }
        )
        conventions[convention] = {
            "formula_choices": formula_choices,
            "jobs": convention_jobs,
            "locked_reports": convention_reports,
            "locked_evaluation_identity": locked,
            "mm_fifteen_year_truncation": truncation,
            "resource_use": evaluation.get("resource_measurements"),
        }
    report = {
        "format_version": 1,
        "kind": "paired-convention-pilot-report",
        "label": "paired-convention-research-pilot",
        "pilot_source": {
            "directory": str(pilot_run_directory.resolve()),
            "manifest_sha256": _sha256(pilot_manifest_path),
            "git_revision": pilot_manifest.get("git_revision"),
            "financial_semantics_version": pilot_manifest.get("financial_semantics_version"),
            "shared_identities": pilot.get("shared_identities"),
            "resource_use": pilot.get("resource_measurements"),
        },
        "evaluation_source": {
            "directory": str(evaluation_directory.resolve()),
            "resource_use": evaluation.get("resource_measurements"),
        },
        "conventions": conventions,
        "paired_intervals": intervals,
        "numerical_behavior": _paired_pilot_numerical_behavior(pilot_run_directory),
        "deferred_work": [
            "paper widths",
            "three MM seeds",
            "sensitivity retraining",
            "10,000 bootstrap",
            "full paper figures",
            "Ticket 24 economic acceptance gates",
        ],
        "disclosure": (
            "This is a paired-convention-research-pilot. It is neither convergence "
            "evidence, paper-result replication, methodologically-reproduced, nor "
            "bank-model approval."
        ),
    }
    return PairedPilotReportArtifacts(report=report)


def _read_report_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportingError(f"Paired-pilot {label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ReportingError(f"Paired-pilot {label} must be a JSON object")
    return value


def _paired_pilot_numerical_behavior(directory: Path) -> list[dict[str, object]]:
    """Expose recovery probes and failures without treating them as current failures."""

    evidence: list[dict[str, object]] = []
    for path in sorted(directory.glob("*/*.interruption.json")):
        contents = _read_report_object(path, "interruption evidence")
        evidence.append({"path": str(path.relative_to(directory)), "evidence": contents})
    return evidence


def build_compact_no_swap_report(
    configuration: ResolvedRunConfiguration,
    *,
    source_run_directories: tuple[Path, ...],
) -> CompactReportArtifacts:
    """Build report content from completed bundles plus bounded market evidence.

    This function never trains a policy or produces a paper-style economic
    claim.  It makes absent local evidence explicit in the coverage inventory
    rather than creating a visually plausible substitute.
    """

    if not source_run_directories:
        raise ReportingError("Compact report requires at least one completed source bundle")
    expected_input_hashes = {
        "snb_csv": _sha256(configuration.source_data.snb_csv),
        "paper_pdf": _sha256(configuration.source_data.paper_pdf),
    }
    expected_configuration = configuration.to_dict()
    expected_convention = expected_configuration["convention"]
    assert isinstance(expected_convention, dict)
    sources = tuple(
        _load_completed_source(
            path,
            expected_input_hashes=expected_input_hashes,
            expected_convention=expected_convention,
        )
        for path in source_run_directories
    )
    historical = MarketScenarioModel().load_historical_term_structures(
        configuration.source_data.snb_csv,
        beta_unit=configuration.source_data.nss_beta_unit,
    )
    market_model = MarketScenarioModel()
    calibration = market_model.calibrate_hjm_pca(historical)
    calibration_evidence = _calibration_evidence(configuration, historical, calibration)
    reference_bank_summary = _reference_bank_summary(historical)
    market_comparison = _hjm_hull_white_comparison(
        configuration, market_model, historical, calibration
    )
    metadata = _presentation_metadata(sources)
    calibration_evidence["presentation_metadata"] = metadata
    calibration_evidence["long_end_extrapolation"] = _long_end_extrapolation(sources)
    reference_bank_summary["presentation_metadata"] = metadata
    market_comparison["presentation_metadata"] = metadata
    local_evidence = _local_evidence(sources)
    actual_work = _actual_work_ledger(sources)
    coverage_inventory = _coverage_inventory(
        sources,
        metadata=metadata,
        local_evidence=local_evidence,
        long_end_status=calibration_evidence["long_end_extrapolation"]["status"],
    )
    report = {
        "format_version": 1,
        "kind": "compact-no-swap-local-report",
        "purpose": "local workflow demonstration",
        "swaps_included": False,
        "source_runs": [source.summary for source in sources],
        "presentation_metadata": metadata,
        "included_evidence": {
            "reference_bank": "reference-bank-summary.json",
            "calibration_and_pca": "calibration-evidence.json",
            "hjm_hull_white_comparison": "hjm-hull-white-comparison.json",
            "coverage_inventory": "paper-coverage-inventory.json",
            "source_artifacts": [
                artifact.summary for source in sources for artifact in source.artifacts
            ],
        },
        "local_evidence": local_evidence,
        "actual_work": actual_work,
        "deferred_research": [
            {
                "scope": "full no-swap Tables 1-5 and Figures 3-17",
                "reason": (
                    "The compact report preserves coverage and evidence only; "
                    "full paper-style reporting remains deferred research work."
                ),
            },
            {
                "scope": "policy/equity/constraint trajectories and representative paths",
                "reason": (
                    "They require completed locked policy-evaluation evidence and "
                    "are not inferred from a small report-only market run."
                ),
            },
            {
                "scope": "real-bank mapping, multi-GPU, and economic approval",
                "reason": "They remain separate bank-side commissioning and acceptance work.",
            },
        ],
        "disclosures": {
            "caption_number_mapping": (
                "Coverage uses printed paper item numbers; any ambiguous caption or "
                "cross-reference is disclosed through the inventory rather than guessed."
            ),
            "visual_similarity_is_acceptance_test": False,
            "cubic_fit_reference_levels": "waived diagnostic; see calibration evidence",
            "local_result_is_not": "a reproduction of trained paper economic results",
        },
    }
    return CompactReportArtifacts(
        report=report,
        coverage_inventory=coverage_inventory,
        calibration_evidence=calibration_evidence,
        market_comparison=market_comparison,
        reference_bank_summary=reference_bank_summary,
    )


def _local_evidence(sources: tuple[_CompletedSource, ...]) -> dict[str, dict[str, object]]:
    """State whether supplied bundles actually contain each local report topic."""

    required = {
        "policy_metrics": (
            _is_locked_evaluation,
            "Requires a completed locked final-test evaluation artifact.",
        ),
        "equity_and_constraints": (
            _is_locked_evaluation,
            "Requires policy rollout evidence with equity and constraint summaries.",
        ),
        "durations_and_sensitivities": (
            _is_sensitivity_evaluation,
            "Requires the representative frozen-policy sensitivity artifact.",
        ),
        "recovery": (
            _is_recovery_diagnostic,
            "Requires a completed epoch-boundary recovery diagnostic artifact.",
        ),
        "mm_truncation": (
            _is_mm_truncation,
            "Requires the selected MM(15y|5y) truncation evaluation artifact.",
        ),
        "scenario_categories_and_bootstrap": (
            _is_horizon_scenario_analysis,
            "Requires common-path scenario-category and paired-bootstrap evidence.",
        ),
    }
    return {
        name: {
            "status": "included" if (included := _validated_artifacts(sources, validator)) else "deferred",
            "source_artifacts": [artifact.summary for artifact in included],
            "observations": [artifact.contents for artifact in included],
            "reason": reason,
        }
        for name, (validator, reason) in required.items()
    }


def _validated_artifacts(
    sources: tuple[_CompletedSource, ...],
    validator: Callable[[_CompletedSource, _EvidenceArtifact], bool],
) -> tuple[_EvidenceArtifact, ...]:
    return tuple(
        artifact
        for source in sources
        for artifact in source.artifacts
        if validator(source, artifact)
    )


def _is_locked_evaluation(
    source: _CompletedSource, artifact: _EvidenceArtifact
) -> bool:
    contents = artifact.contents
    return (
        artifact.name in {"locked-evaluation.json", "final-test-evaluation.json"}
        and isinstance(contents, dict)
        and contents.get("format_version") == 1
        and contents.get("kind") == "locked-final-test-evaluation"
        and isinstance(contents.get("reports"), dict)
        and _matches_source_market_and_convention(source, contents)
    )


def _is_sensitivity_evaluation(
    source: _CompletedSource, artifact: _EvidenceArtifact
) -> bool:
    contents = artifact.contents
    evaluation = contents.get("evaluation") if isinstance(contents, dict) else None
    evaluation_manifest = evaluation.get("manifest") if isinstance(evaluation, dict) else None
    return (
        artifact.name == "reference-bank-sensitivity.json"
        and isinstance(contents, dict)
        and contents.get("format_version") == 1
        and contents.get("kind") == "frozen-policy-reference-bank-sensitivity"
        and isinstance(evaluation, dict)
        and isinstance(evaluation.get("reports"), dict)
        and isinstance(evaluation_manifest, dict)
        and _matches_source_market_and_convention(source, evaluation_manifest)
    )


def _is_recovery_diagnostic(
    source: _CompletedSource, artifact: _EvidenceArtifact
) -> bool:
    contents = artifact.contents
    if not (
        artifact.name.endswith(".interruption.json")
        and isinstance(contents, dict)
        and contents.get("format_version") == 1
        and contents.get("status") == "incomplete"
        and isinstance(contents.get("reason"), str)
        and isinstance(contents.get("recovery_path"), str)
        and isinstance(contents.get("diagnostics"), dict)
    ):
        return False
    recovery_name = Path(str(contents["recovery_path"]))
    if recovery_name.name != str(contents["recovery_path"]):
        return False
    recovery_path = source.directory / recovery_name
    try:
        recovered = torch.load(recovery_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError):
        return False
    if (
        not isinstance(recovered, dict)
        or recovered.get("format_version") != 1
        or recovered.get("kind") != "epoch-recovery"
        or not isinstance(recovered.get("recovery_identity"), dict)
    ):
        return False
    identity = recovered["recovery_identity"]
    data_identities = identity.get("data_identities")
    if not isinstance(data_identities, dict):
        return False
    return (
        data_identities.get("market_source_hash")
        == source.manifest["input_hashes"]["snb_csv"]
        and identity.get("semantic_configuration_identity")
        == _semantic_configuration_identity(source.manifest["resolved_configuration"])
    )


def _is_mm_truncation(
    source: _CompletedSource, artifact: _EvidenceArtifact
) -> bool:
    contents = artifact.contents
    return (
        artifact.name in {"mm-truncation.json", "mm-fifteen-truncation.json"}
        and isinstance(contents, dict)
        and contents.get("format_version") == 1
        and contents.get("kind") == "mm-fifteen-year-truncation"
        and contents.get("source_horizon_years") == 15
        and contents.get("evaluation_horizon_years") == 5
        and _matches_source_market_and_convention(source, contents)
    )


def _is_horizon_scenario_analysis(
    source: _CompletedSource, artifact: _EvidenceArtifact
) -> bool:
    contents = artifact.contents
    return (
        artifact.name in {"horizon-scenario-analysis.json", "scenario-bootstrap-analysis.json"}
        and isinstance(contents, dict)
        and contents.get("format_version") == 1
        and contents.get("kind") == "horizon-scenario-analysis"
        and isinstance(contents.get("metrics_by_policy"), dict)
        and isinstance(contents.get("paired_intervals"), list)
        and isinstance(contents.get("category_outputs"), list)
        and _matches_source_market_and_convention(source, contents)
    )


def _matches_source_market_and_convention(
    source: _CompletedSource, contents: dict[str, object]
) -> bool:
    configuration = source.manifest.get("resolved_configuration")
    identities = contents.get("data_identities")
    if not isinstance(configuration, dict) or not isinstance(identities, dict):
        return False
    convention = configuration.get("convention")
    input_hashes = source.manifest.get("input_hashes")
    return (
        isinstance(convention, dict)
        and contents.get("convention") == convention.get("profile")
        and isinstance(input_hashes, dict)
        and identities.get("market_source_hash") == input_hashes.get("snb_csv")
    )


def _semantic_configuration_identity(configuration: object) -> str | None:
    if not isinstance(configuration, dict):
        return None
    semantic = json.loads(json.dumps(configuration))
    semantic.pop("output", None)
    semantic.pop("resources", None)
    optimization = semantic.get("optimization")
    if isinstance(optimization, dict):
        optimization.pop("device", None)
    serialized = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def _long_end_extrapolation(
    sources: tuple[_CompletedSource, ...]
) -> dict[str, object]:
    evidence = _validated_artifacts(sources, _is_long_end_extrapolation)
    if evidence:
        return {
            "status": "included",
            "source_artifacts": [artifact.summary for artifact in evidence],
            "observations": [artifact.contents for artifact in evidence],
        }
    return {
        "status": "deferred",
        "reason": (
            "No completed long-end extrapolation diagnostic with matching market "
            "identity and convention was supplied by the source bundles."
        ),
    }


def _is_long_end_extrapolation(
    source: _CompletedSource, artifact: _EvidenceArtifact
) -> bool:
    contents = artifact.contents
    return (
        artifact.name == "long-end-extrapolation.json"
        and isinstance(contents, dict)
        and contents.get("format_version") == 1
        and contents.get("kind") == "long-end-extrapolation-diagnostics"
        and _matches_source_market_and_convention(source, contents)
    )


@dataclass(frozen=True)
class _EvidenceArtifact:
    name: str
    sha256: str
    contents: dict[str, object] | None

    @property
    def summary(self) -> dict[str, str]:
        return {"name": self.name, "sha256": self.sha256}


@dataclass(frozen=True)
class _CompletedSource:
    directory: Path
    manifest: dict[str, object]
    artifacts: tuple[_EvidenceArtifact, ...]

    @property
    def summary(self) -> dict[str, object]:
        return {
            "directory": str(self.directory),
            "status": self.manifest["status"],
            "git_revision": self.manifest.get("git_revision", "unavailable"),
            "resolved_configuration_hash": self.manifest.get(
                "resolved_configuration_hash", "unavailable"
            ),
            "input_hashes": self.manifest.get("input_hashes", {}),
            "artifacts": [artifact.summary for artifact in self.artifacts],
        }


def _load_completed_source(
    directory: Path,
    *,
    expected_input_hashes: dict[str, str],
    expected_convention: dict[str, object],
) -> _CompletedSource:
    if not directory.is_dir():
        raise ReportingError(f"Report source directory does not exist: {directory}")
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportingError(f"Report source manifest is unreadable: {manifest_path}") from error
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise ReportingError(f"Report source bundle is not completed: {directory}")
    if manifest.get("input_hashes") != expected_input_hashes:
        raise ReportingError(
            f"Report source input identities do not match: {directory}"
        )
    source_configuration = manifest.get("resolved_configuration")
    if (
        not isinstance(source_configuration, dict)
        or source_configuration.get("convention") != expected_convention
    ):
        raise ReportingError(
            f"Report source convention does not match: {directory}"
        )
    return _CompletedSource(
        directory=directory,
        manifest=manifest,
        artifacts=tuple(
            _load_evidence_artifact(path)
            for path in sorted(directory.glob("*.json"))
            if path.name != "manifest.json"
        ),
    )


def _load_evidence_artifact(path: Path) -> _EvidenceArtifact:
    try:
        contents = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        contents = None
    return _EvidenceArtifact(
        name=path.name,
        sha256=_sha256(path),
        contents=contents if isinstance(contents, dict) else None,
    )


def _calibration_evidence(
    configuration: ResolvedRunConfiguration, historical: Any, calibration: Any
) -> dict[str, object]:
    convention = configuration.convention.profile
    component_errors = calibration.polynomial_fit_errors[convention]
    loading = calibration.scaled_loadings[convention]
    fitted = calibration.fitted_loadings[convention]
    aggregate_error = float(np.linalg.norm(fitted - loading) / np.linalg.norm(loading))
    return {
        "format_version": 1,
        "kind": "local-calibration-and-pca-evidence",
        "market_source_hash": historical.source_hash,
        "calibration_identity": calibration.calibration_identity,
        "weekly_dates": [str(date) for date in calibration.weekly_dates],
        "three_component_explained_variance": {
            "observed": float(np.sum(calibration.explained_variance[:3])),
            "threshold": 0.9,
            "status": (
                "passed" if float(np.sum(calibration.explained_variance[:3])) >= 0.9 else "failed"
            ),
        },
        "cubic_fit_reference_levels": {
            "component_relative_error_reference": 0.15,
            "aggregate_relative_error_reference": 0.10,
            "observed_component_relative_errors": [
                float(error) for error in component_errors
            ],
            "observed_aggregate_relative_error": aggregate_error,
            "waiver_status": "waived",
            "waiver_provenance": (
                "Specification revision 2 retains the original reference levels as "
                "visible diagnostics and records the accepted fit-error deviation."
            ),
        },
    }


def _reference_bank_summary(historical: Any) -> dict[str, object]:
    snapshot = ReferenceBankProvider().build_canonical(historical)
    return {
        "format_version": 1,
        "kind": "canonical-reference-bank-summary",
        "content_hash": snapshot.content_hash,
        "profile": snapshot.profile,
        "as_of_date": snapshot.as_of_date,
        "currency": snapshot.currency,
        "unit": snapshot.unit,
        "total_assets": snapshot.total_assets,
        "total_liabilities": snapshot.total_liabilities,
        "equity": snapshot.equity,
        "loan_duration_years": snapshot.loan_duration_years,
        "deposit_duration_years": snapshot.deposit_duration_years,
        "assumptions": dict(snapshot.product_assumptions),
    }


def _hjm_hull_white_comparison(
    configuration: ResolvedRunConfiguration,
    market_model: MarketScenarioModel,
    historical: Any,
    calibration: Any,
) -> dict[str, object]:
    paths = min(32, configuration.run_scale.test_paths)
    if paths <= 0:
        raise ReportingError("Report HJM/Hull-White comparison requires at least one path")
    horizon_years = 5
    hjm = market_model.generate_hjm_scenarios(
        historical,
        calibration,
        convention=configuration.convention.profile,
        horizon_years=horizon_years,
        paths=paths,
        seed=configuration.seeds["market_scenarios"],
        split="report-hjm-hull-white",
        epoch=0,
        global_path_indices=tuple(range(paths)),
    )
    hull_white_configuration = HullWhiteConfiguration(mean_reversion=0.2, volatility=0.01)
    hull_white = market_model.generate_hull_white_scenarios(
        historical,
        horizon_years=horizon_years,
        paths=paths,
        seed=configuration.seeds["market_scenarios"],
        configuration=hull_white_configuration,
    )
    diversity = market_model.summarize_terminal_curve_diversity(hjm, hull_white)
    result = asdict(diversity)
    result["hjm_terminal_standard_deviation"] = diversity.hjm_terminal_standard_deviation.tolist()
    result["hull_white_terminal_standard_deviation"] = (
        diversity.hull_white_terminal_standard_deviation.tolist()
    )
    result["models"] = ["hjm-pca", "project-hull-white"]
    result["format_version"] = 1
    result["kind"] = "local-hjm-hull-white-comparison"
    result["role"] = "diagnostic-only comparison; not a policy-training model"
    result["hull_white_configuration"] = asdict(hull_white_configuration)
    return result


@dataclass(frozen=True)
class _CoverageItem:
    paper_item: str
    subject: str
    report_artifact: str | None = None
    evidence_topic: str | None = None


_COVERAGE_ITEMS = (
    _CoverageItem("Table 1", "Reference Bank assumptions and balance-sheet summary", "reference-bank-summary.json"),
    _CoverageItem("Table 2", "No-swap policy outcome summary", evidence_topic="policy_metrics"),
    _CoverageItem("Table 3", "No-swap policy risk and constraint summary", evidence_topic="equity_and_constraints"),
    _CoverageItem("Table 4", "Horizon and truncation comparison", evidence_topic="mm_truncation"),
    _CoverageItem("Table 5", "Reference Bank sensitivity summary", evidence_topic="durations_and_sensitivities"),
    _CoverageItem("Figure 3", "Term-structure calibration and PCA evidence", "calibration-evidence.json"),
    _CoverageItem("Figure 4", "HJM and Hull-White scenario diversity comparison", "hjm-hull-white-comparison.json"),
    _CoverageItem("Figure 5", "Long-end extrapolation diagnostics", evidence_topic="long_end_extrapolation"),
    _CoverageItem("Figure 6", "PCA loading diagnostics", "calibration-evidence.json"),
    _CoverageItem("Figure 7", "Policy actions", evidence_topic="policy_metrics"),
    _CoverageItem("Figure 8", "Equity outcomes", evidence_topic="equity_and_constraints"),
    _CoverageItem("Figure 9", "Constraint trajectories", evidence_topic="equity_and_constraints"),
    _CoverageItem("Figure 10", "Duration evidence", "reference-bank-summary.json"),
    _CoverageItem("Figure 11", "Interest-rate sensitivity evidence", evidence_topic="durations_and_sensitivities"),
    _CoverageItem("Figure 12", "Representative path evidence", evidence_topic="policy_metrics"),
    _CoverageItem("Figure 13", "Scenario-category policy actions", evidence_topic="scenario_categories_and_bootstrap"),
    _CoverageItem("Figure 14", "Scenario-category constraint evidence", evidence_topic="scenario_categories_and_bootstrap"),
    _CoverageItem("Figure 15", "Horizon action-turnover evidence", evidence_topic="scenario_categories_and_bootstrap"),
    _CoverageItem("Figure 16", "Terminal-concentration evidence", evidence_topic="scenario_categories_and_bootstrap"),
    _CoverageItem("Figure 17", "Bootstrap and sensitivity evidence", evidence_topic="scenario_categories_and_bootstrap"),
)


def _coverage_inventory(
    _sources: tuple[_CompletedSource, ...],
    *,
    metadata: dict[str, object],
    local_evidence: dict[str, dict[str, object]],
    long_end_status: object,
) -> dict[str, object]:
    entries = []
    for item in _COVERAGE_ITEMS:
        evidence = (
            {"status": long_end_status, "reason": "See calibration evidence."}
            if item.evidence_topic == "long_end_extrapolation"
            else local_evidence.get(item.evidence_topic or "", {})
        )
        generated = item.report_artifact is not None or evidence.get("status") == "included"
        entries.append(
            {
                "paper_item": item.paper_item,
                "subject": item.subject,
                "status": "generated" if generated else "deferred",
                "reason": (
                    "Generated as a compact local workflow evidence artifact."
                    if generated
                    else evidence.get(
                        "reason",
                        "Requires completed policy-evaluation or extended-research evidence.",
                    )
                ),
                "report_artifacts": [item.report_artifact]
                if item.report_artifact is not None
                else [],
                "source_artifacts": evidence.get("source_artifacts", []),
                "metadata": metadata,
                "caption_number_disclosure": {
                    "printed_item": item.paper_item,
                    "caption_status": "not-transcribed",
                    "cross_reference_status": "not-verified",
                    "handling": (
                        "The compact inventory uses the printed item number only; "
                        "it does not infer a caption-number mapping from visual similarity."
                    ),
                },
            }
        )
    return {
        "format_version": 1,
        "kind": "no-swap-paper-coverage-inventory",
        "entries": entries,
        "presentation_metadata": metadata,
        "swaps_included": False,
        "visual_similarity_is_acceptance_test": False,
    }


def _presentation_metadata(sources: tuple[_CompletedSource, ...]) -> dict[str, object]:
    configurations = [
        source.manifest.get("resolved_configuration", {}) for source in sources
    ]
    policies: set[str] = set()
    horizons: set[int] = set()
    conventions: set[str] = set()
    units: set[str] = set()
    sample_sizes: list[dict[str, object]] = []
    architectures: list[dict[str, object]] = []
    seed_registries: list[dict[str, object]] = []
    for configuration in configurations:
        if not isinstance(configuration, dict):
            continue
        policy = configuration.get("policy", {})
        experiment = configuration.get("experiment", {})
        convention = configuration.get("convention", {})
        reference_bank = configuration.get("reference_bank", {})
        run_scale = configuration.get("run_scale", {})
        architecture = configuration.get("architecture", {})
        if isinstance(policy, dict):
            policies.update(str(name) for name in policy.get("names", []))
        if isinstance(experiment, dict):
            horizons.update(int(value) for value in experiment.get("horizons_years", []))
        if isinstance(convention, dict) and convention.get("profile") is not None:
            conventions.add(str(convention["profile"]))
        if isinstance(reference_bank, dict):
            initial_assets = reference_bank.get("initial_assets", {})
            if isinstance(initial_assets, dict) and initial_assets.get("unit") is not None:
                units.add(str(initial_assets["unit"]))
        if isinstance(run_scale, dict):
            sample_sizes.append(
                {
                    name: run_scale.get(name)
                    for name in (
                        "training_paths_per_epoch",
                        "selection_paths",
                        "test_paths",
                    )
                }
            )
        if isinstance(architecture, dict):
            architectures.append(
                {
                    "profile": architecture.get("profile"),
                    "widths": architecture.get("widths"),
                }
            )
    for source in sources:
        seeds = source.manifest.get("seed_registry")
        if isinstance(seeds, dict):
            seed_registries.append(dict(seeds))
    return {
        "policy": sorted(policies),
        "horizon_years": sorted(horizons),
        "convention": sorted(conventions),
        "sample_size": sample_sizes,
        "units": {"rates": "decimal annual rates", "monetary": sorted(units)},
        "source_runs": [str(source.directory) for source in sources],
        "architecture": architectures,
        "parameter_counts": _checkpoint_parameter_counts(sources),
        "seed_registry": seed_registries,
    }


def _checkpoint_parameter_counts(
    sources: tuple[_CompletedSource, ...]
) -> list[dict[str, object]]:
    return [
        {
            key: value
            for key, value in audit.items()
            if key
            in {
                "source_run",
                "checkpoint",
                "checkpoint_sha256",
                "status",
                "reason",
                "policy",
                "horizon_years",
                "parameter_count",
            }
        }
        for audit in _checkpoint_audit(sources)
    ]


def _actual_work_ledger(sources: tuple[_CompletedSource, ...]) -> dict[str, object]:
    audits = _checkpoint_audit(sources)
    completed = [audit for audit in audits if audit.get("status") == "included"]
    selection_metrics = [
        audit["selection_metrics"]
        for audit in completed
        if audit.get("selection_metrics") is not None
    ]
    profiles = _resource_profiles(sources)
    planned_jobs = [
        {
            "source_run": str(source.directory),
            "jobs": source.manifest.get("execution_plan", {}).get("jobs", []),
        }
        for source in sources
    ]
    return {
        "planned_jobs": planned_jobs,
        "completed_training_jobs": completed,
        "checkpoint_audit": audits,
        "completed_optimizer_updates": sum(
            int(audit["optimizer_updates"])
            for audit in completed
            if isinstance(audit.get("optimizer_updates"), int)
        ),
        "resource_measurements": {
            "status": "included" if profiles else "unavailable",
            "observations": profiles,
            "reason": None if profiles else "No valid resource-profile artifact was supplied.",
        },
        "numeric_metrics": {
            "status": "included" if selection_metrics else "unavailable",
            "observations": selection_metrics,
            "reason": (
                None
                if selection_metrics
                else "No selected-checkpoint loss and constraint metrics were supplied."
            ),
        },
    }


def _checkpoint_audit(sources: tuple[_CompletedSource, ...]) -> list[dict[str, object]]:
    audits: list[dict[str, object]] = []
    for source in sources:
        for checkpoint_path in sorted(source.directory.glob("*.pt")):
            if checkpoint_path.name.endswith(".recovery.pt"):
                continue
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                if not isinstance(checkpoint, dict):
                    raise TypeError("checkpoint is not a mapping")
                _validate_checkpoint_for_source(source, checkpoint)
                state = checkpoint.get("policy_state")
                if not isinstance(state, dict):
                    raise TypeError("missing policy state")
                count = sum(
                    value.numel() for value in state.values() if isinstance(value, torch.Tensor)
                )
                history = checkpoint.get("selection_history")
                selected_epoch = checkpoint.get("selected_epoch")
                if not isinstance(history, list) or not isinstance(selected_epoch, int):
                    raise TypeError("missing selected epoch or selection history")
                selected = next(
                    (
                        record
                        for record in history
                        if isinstance(record, dict) and record.get("epoch") == selected_epoch
                    ),
                    None,
                )
                if not isinstance(selected, dict):
                    raise TypeError("selected epoch has no selection record")
                selection_metrics = {
                    key: selected[key]
                    for key in ("total_loss", "penalty_loss")
                    if key in selected
                }
                audits.append(
                    {
                        "source_run": str(source.directory),
                        "checkpoint": checkpoint_path.name,
                        "checkpoint_sha256": _sha256(checkpoint_path),
                        "status": "included",
                        "policy": checkpoint["policy"],
                        "horizon_years": checkpoint["horizon_years"],
                        "parameter_count": count,
                        "optimizer_updates": checkpoint.get("optimizer_updates"),
                        "selected_epoch": selected_epoch,
                        "selection_metrics": selection_metrics,
                    }
                )
            except (OSError, RuntimeError, ValueError, TypeError, AttributeError):
                audits.append(
                    {
                        "source_run": str(source.directory),
                        "checkpoint": checkpoint_path.name,
                        "checkpoint_sha256": _sha256(checkpoint_path),
                        "status": "unavailable",
                        "parameter_count": "unavailable",
                        "reason": "Checkpoint failed the completed-work identity and schema audit.",
                    }
                )
    return audits


def _validate_checkpoint_for_source(
    source: _CompletedSource, checkpoint: dict[str, object]
) -> None:
    """Reject files that look like checkpoints but cannot evidence this source run."""

    source_configuration = source.manifest.get("resolved_configuration")
    source_configuration_identity = source.manifest.get("resolved_configuration_hash")
    expected_data_identities = _source_checkpoint_data_identities(source)
    if (
        checkpoint.get("format_version") != 1
        or not isinstance(checkpoint.get("policy"), str)
        or not isinstance(checkpoint.get("horizon_years"), int)
        or checkpoint.get("configuration") != source_configuration
        or checkpoint.get("configuration_identity") != source_configuration_identity
        or checkpoint.get("data_identities") != expected_data_identities
    ):
        raise ValueError("checkpoint does not match completed source identities")


def _source_checkpoint_data_identities(
    source: _CompletedSource,
) -> dict[str, object] | None:
    """Read the three input identities a completed training checkpoint must carry."""

    input_hashes = source.manifest.get("input_hashes")
    preflight = source.manifest.get("market_preflight")
    calibration = preflight.get("calibration") if isinstance(preflight, dict) else None
    reference_bank = source.manifest.get("reference_bank")
    market_hash = input_hashes.get("snb_csv") if isinstance(input_hashes, dict) else None
    calibration_identity = (
        calibration.get("identity") if isinstance(calibration, dict) else None
    )
    reference_bank_hash = (
        reference_bank.get("content_hash") if isinstance(reference_bank, dict) else None
    )
    if not all(
        isinstance(value, str)
        for value in (market_hash, calibration_identity, reference_bank_hash)
    ):
        return None
    return {
        "market_source_hash": market_hash,
        "hjm_calibration_identity": calibration_identity,
        "reference_bank_content_hash": reference_bank_hash,
    }


def _resource_profiles(sources: tuple[_CompletedSource, ...]) -> list[dict[str, object]]:
    required = {
        "horizon_years",
        "device",
        "forward_backward_update_seconds",
        "scenario_seconds",
        "selection_seconds",
        "peak_process_rss_bytes",
    }
    return [
        {"source_run": str(source.directory), "artifact": artifact.name, **artifact.contents}
        for source in sources
        for artifact in source.artifacts
        if artifact.name.endswith(".profile.json")
        and isinstance(artifact.contents, dict)
        and required.issubset(artifact.contents)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
