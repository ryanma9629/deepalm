"""Approved one-factor Reference Bank sensitivity definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensitivityDefinition:
    """Canonical value and explicitly approved alternatives for one factor."""

    canonical_value: float
    approved_values: tuple[float, ...]


REFERENCE_BANK_SENSITIVITIES = {
    "total_assets_mchf": SensitivityDefinition(
        canonical_value=10_000.0,
        approved_values=(5_000.0, 10_000.0, 20_000.0),
    ),
    "mortgage_ten_year_weight": SensitivityDefinition(
        canonical_value=0.40,
        approved_values=(0.20, 0.40, 0.60),
    ),
    "non_maturity_ten_year_weight": SensitivityDefinition(
        canonical_value=0.05,
        approved_values=(0.00, 0.05, 0.15),
    ),
    "term_deposit_ten_year_weight": SensitivityDefinition(
        canonical_value=0.30,
        approved_values=(0.15, 0.30, 0.45),
    ),
    "loan_spread_decimal": SensitivityDefinition(
        canonical_value=0.015,
        approved_values=(0.01, 0.015, 0.02),
    ),
    "operating_cost_multiplier": SensitivityDefinition(
        canonical_value=1.0,
        approved_values=(0.75, 1.0, 1.25),
    ),
}


def approved_sensitivity_value(factor: str, value: float) -> float:
    """Return an approved value or raise a value error with no fallback."""

    definition = REFERENCE_BANK_SENSITIVITIES.get(factor)
    if definition is None or value not in definition.approved_values:
        raise ValueError(
            f"Sensitivity {factor!r}={value!r} is not an approved one-factor value"
        )
    return value


def canonical_sensitivity_value(factor: str) -> float:
    """Return the registered canonical value for a named sensitivity factor."""

    try:
        return REFERENCE_BANK_SENSITIVITIES[factor].canonical_value
    except KeyError as error:
        raise ValueError(f"Sensitivity factor {factor!r} is not approved") from error
