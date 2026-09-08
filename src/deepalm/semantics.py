"""Version actual artifact meaning, independently of runtime and repair plans."""

import json
from collections.abc import Mapping
from typing import Literal

FINANCIAL_SEMANTICS_VERSION = "dated-deposit-history-v6"
POLICY_SEMANTICS_VERSION = "maturity-relative-bmd-v3"
METRIC_SEMANTICS_VERSION = "centered-equity-ratio-and-penalty-v2"
SNAPSHOT_SCHEMA_VERSION = 5

ArtifactScope = Literal["snapshot", "training", "evaluation"]
_CORRECTIONS = {
    "snapshot": ("C-5", "C-6"),
    "training": ("C-1", "C-2", "C-3", "C-5", "C-6"),
    "evaluation": ("C-1", "C-2", "C-3", "C-5", "C-6", "C-7", "C-8", "R-1", "R-2"),
}
# Activate only alongside a completed formula repair and its regression evidence.
_IMPLEMENTED_CORRECTIONS: frozenset[str] = frozenset(
    {"C-1", "C-2", "C-3", "C-5", "C-6"}
)


def artifact_semantics(scope: ArtifactScope) -> dict[str, object]:
    """Describe implemented behavior; a new envelope is not a formula repair.

    Training deliberately excludes metric/report versions and their correction
    status. A report-only repair must not invalidate otherwise valid weights.
    """
    corrections = {
        correction: correction in _IMPLEMENTED_CORRECTIONS
        for correction in _CORRECTIONS[scope]
    }
    identity: dict[str, object] = {
        "contract_version": 1,
        "scope": scope,
        "financial_version": FINANCIAL_SEMANTICS_VERSION,
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "corrections": corrections,
        "repair_status": "complete" if all(corrections.values()) else "pending",
    }
    if scope != "snapshot":
        identity["policy_version"] = POLICY_SEMANTICS_VERSION
    if scope == "evaluation":
        identity["metric_version"] = METRIC_SEMANTICS_VERSION
    return identity


def artifact_semantics_error(
    container: object, scope: ArtifactScope
) -> str | None:
    """Return a fail-closed compatibility diagnostic for a persisted envelope."""
    record = container.get("artifact_semantics") if isinstance(container, Mapping) else None
    expected = artifact_semantics(scope)
    if not isinstance(record, dict):
        return f"{scope} artifact semantics missing; regenerate compatible evidence"
    try:
        matches = json.dumps(record, sort_keys=True, allow_nan=False) == json.dumps(
            expected, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError):
        matches = False
    if not matches:
        return f"{scope} artifact semantics incompatible; regenerate compatible evidence"
    legacy_fields: dict[str, object] = {"financial_semantics_version": FINANCIAL_SEMANTICS_VERSION}
    if scope == "evaluation":
        legacy_fields["risk_metric_convention"] = METRIC_SEMANTICS_VERSION
    if scope == "snapshot":
        legacy_fields["schema_version"] = SNAPSHOT_SCHEMA_VERSION
    for name, value in legacy_fields.items():
        if name in container and (
            type(container[name]) is not type(value) or container[name] != value
        ):
            return f"{scope} artifact semantics contradict {name}; regenerate compatible evidence"
    return None


def current_financial_semantics(checkpoint: Mapping[str, object]) -> bool:
    identity = checkpoint.get("code_identity")
    return (
        isinstance(identity, dict)
        and identity.get("financial_semantics_version") == FINANCIAL_SEMANTICS_VERSION
    )
