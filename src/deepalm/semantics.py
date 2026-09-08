"""Version the financial rules independently of architecture and run budget."""

from collections.abc import Mapping

FINANCIAL_SEMANTICS_VERSION = "fixed-rate-cohorts-cash-rollover-v3"


def current_financial_semantics(checkpoint: Mapping[str, object]) -> bool:
    identity = checkpoint.get("code_identity")
    return (
        isinstance(identity, dict)
        and identity.get("financial_semantics_version") == FINANCIAL_SEMANTICS_VERSION
    )
