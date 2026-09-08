"""Synthetic, auditable Reference Bank construction for Deep ALM."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType

import numpy as np

from deepalm.runner import OperationalRunError
from deepalm.semantics import (
    SNAPSHOT_SCHEMA_VERSION,
    artifact_semantics,
    artifact_semantics_error,
)
from deepalm.sensitivities import (
    REFERENCE_BANK_SENSITIVITIES,
    approved_sensitivity_value,
    canonical_sensitivity_value,
)
from deepalm.term_structures import (
    DepositInitialHistory,
    HistoricalTermStructures,
    MarketScenarioModel,
    deposit_initial_history_identity,
    deposit_initial_history_target_dates,
)

_TARGETS = {
    "cash": 2_000.0,
    "investments": 500.0,
    "mortgages": 5_500.0,
    "enterprise_loans": 2_000.0,
    "non_maturity_deposits": 1_000.0,
    "term_deposits": 4_000.0,
    "funding": 4_000.0,
}
_LADDER_NAMES = tuple(name for name in _TARGETS if name != "cash")
_SCHEMA_VERSION = SNAPSHOT_SCHEMA_VERSION
_PROFILES = {"canonical", "imported", "sensitivity"}
_CANONICAL_AS_OF_DATE = "2022-07-15"
_CANONICAL_CURVE_IDENTITY = (
    "f05edd68a673955908c0bf04e5771a64ff70a10f987f27046511bc41f36a2d8f"
)
_PROVENANCE_FIELDS = {
    "source_system",
    "product_mapping",
    "assumptions",
    "valuation",
    "market",
}


class ReferenceBankError(OperationalRunError):
    """Raised when a Reference Bank snapshot is economically inconsistent."""


@dataclass(frozen=True)
class ReferenceBankSensitivity:
    """One approved, explicit deviation from the canonical Reference Bank."""

    factor: str
    value: float


@dataclass(frozen=True)
class LoanCohort:
    """One initial fixed-rate loan cohort's remaining principal schedule."""

    principal_cash_flows: np.ndarray
    monthly_coupon_rate: float


@dataclass(frozen=True)
class DepositReferenceSchedule:
    """One product's cash-flow schedules, separated by original reference term."""

    terms_months: tuple[int, ...]
    weights: tuple[float, ...]
    cash_flows: np.ndarray
    provenance: str


@dataclass(frozen=True)
class ReferenceBankSnapshot:
    """Immutable initial bank state shared by the two model horizons."""

    schema_version: int
    profile: str
    as_of_date: str
    initial_curve_identity: str
    cash: float
    equity: float
    ladders: Mapping[str, np.ndarray]
    loan_cohorts: Mapping[str, tuple[LoanCohort, ...]]
    deposit_reference_schedules: Mapping[str, DepositReferenceSchedule]
    deposit_initial_history: DepositInitialHistory
    target_economic_values: Mapping[str, float]
    target_value_errors: Mapping[str, float]
    product_assumptions: Mapping[str, object]
    provenance: Mapping[str, str]
    personnel_cost: float
    material_cost: float
    currency: str
    unit: str
    content_hash: str

    @property
    def total_assets(self) -> float:
        return self.cash + sum(
            self.target_economic_values[name]
            for name in ("investments", "mortgages", "enterprise_loans")
        )

    @property
    def total_liabilities(self) -> float:
        return sum(
            self.target_economic_values[name]
            for name in ("non_maturity_deposits", "term_deposits", "funding")
        )

    @property
    def loan_duration_years(self) -> float:
        return float(self.product_assumptions["loan_duration_years"])

    @property
    def deposit_duration_years(self) -> float:
        return float(self.product_assumptions["deposit_duration_years"])


class ReferenceBankProvider:
    """Build the canonical bank or import a validated single-currency snapshot."""

    def build_canonical(
        self, historical: HistoricalTermStructures
    ) -> ReferenceBankSnapshot:
        discounts = np.asarray(
            historical.initial_curve.discount_factors, dtype=np.float64
        )
        if discounts.shape != (180,) or not np.all(np.isfinite(discounts)):
            raise ReferenceBankError(
                "Canonical initial curve must contain 180 finite discounts"
            )
        deposit_initial_history = _initial_deposit_history(historical)
        mortgage_weights = [0.06] * 11
        mortgage_weights[8] = 0.40
        loan_cohorts = {
            "mortgages": _scaled_loan_cohorts(
                _seasoned_loan_cohorts(
                    tuple(range(24, 145, 12)),
                    mortgage_weights,
                    historical.initial_curve.spot_rates,
                    0.015,
                ),
                target=_TARGETS["mortgages"],
                discounts=discounts,
            ),
            "enterprise_loans": _scaled_loan_cohorts(
                _seasoned_loan_cohorts(
                    (1, 2, 3),
                    (1 / 3,) * 3,
                    historical.initial_curve.spot_rates,
                    0.015,
                ),
                target=_TARGETS["enterprise_loans"],
                discounts=discounts,
            ),
        }
        deposit_reference_schedules = {
            "non_maturity_deposits": _seasoned_deposit_reference_schedule(
                (1, 2, 12, 120), (0.40, 0.30, 0.25, 0.05), 0.005,
                "synthetic seasoned NMD reference-term allocation",
            ),
            "term_deposits": _seasoned_deposit_reference_schedule(
                (1, 2, 12, 120), (0.10, 0.10, 0.50, 0.30), 0.01,
                "synthetic seasoned term-deposit reference-term allocation",
            ),
        }
        raw = {
            "mortgages": _loan_cohort_cash_flows(loan_cohorts["mortgages"]),
            "enterprise_loans": _loan_cohort_cash_flows(
                loan_cohorts["enterprise_loans"]
            ),
            "investments": _seasoned_ladder(
                tuple(range(36, 181, 12)), (1 / 13,) * 13, 0.01
            ),
            "funding": _seasoned_ladder((3, *range(12, 181, 12)), (1 / 16,) * 16, 0.01),
            **{
                name: schedule.cash_flows.sum(axis=0)
                for name, schedule in deposit_reference_schedules.items()
            },
        }
        deposit_scales = {
            name: _TARGETS[name] / float(raw[name] @ discounts)
            for name in deposit_reference_schedules
        }
        ladders = {
            name: _readonly(
                raw[name]
                if name in loan_cohorts
                else raw[name]
                * (
                    deposit_scales[name]
                    if name in deposit_scales
                    else _TARGETS[name] / float(raw[name] @ discounts)
                )
            )
            for name in _LADDER_NAMES
        }
        durations = {
            name: _duration(ladder, discounts) for name, ladder in ladders.items()
        }
        assumptions: dict[str, object] = {
            "mortgage_terms_months": list(range(24, 145, 12)),
            "mortgage_weights": mortgage_weights,
            "enterprise_terms_months": [1, 2, 3],
            "deposit_reference_terms_months": [1, 2, 12, 120],
            "non_maturity_weights": [0.40, 0.30, 0.25, 0.05],
            "term_deposit_weights": [0.10, 0.10, 0.50, 0.30],
            "loan_spread_decimal": 0.015,
            "personnel_growth_annual": 0.02,
            "loan_duration_years": _weighted_duration(
                durations, ("mortgages", "enterprise_loans")
            ),
            "deposit_duration_years": _weighted_duration(
                durations, ("non_maturity_deposits", "term_deposits")
            ),
            "legacy_coupon_pricing": "project assumption: fixed 1% synthetic legacy bond coupons",
        }
        snapshot = _make_snapshot(
            profile="canonical",
            as_of_date=str(historical.initial_curve.as_of_date.astype("datetime64[D]")),
            initial_curve_identity=historical.source_hash,
            cash=_TARGETS["cash"],
            ladders=ladders,
            loan_cohorts=loan_cohorts,
            deposit_reference_schedules={
                name: DepositReferenceSchedule(
                    terms_months=schedule.terms_months,
                    weights=schedule.weights,
                    cash_flows=schedule.cash_flows * deposit_scales[name],
                    provenance=schedule.provenance,
                )
                for name, schedule in deposit_reference_schedules.items()
            },
            deposit_initial_history=deposit_initial_history,
            target_economic_values=_TARGETS,
            equity=1_000.0,
            assumptions=assumptions,
            target_value_errors={
                name: abs(float(ladder @ discounts) - _TARGETS[name]) / _TARGETS[name]
                for name, ladder in ladders.items()
            },
            provenance={
                "source_system": "synthetic Reference Bank",
                "product_mapping": "seasoned six-ladder project mapping",
                "assumptions": "paper-anchored synthetic product assumptions",
                "valuation": "initial SNB curve discounted cash-flow values",
                "market": _market_provenance(
                    historical.source_hash,
                    str(historical.initial_curve.as_of_date.astype("datetime64[D]")),
                ),
            },
            currency="CHF",
            unit="mCHF",
        )
        self.validate(snapshot, discounts=discounts)
        return snapshot

    def build_sensitivity(
        self,
        historical: HistoricalTermStructures,
        sensitivity: ReferenceBankSensitivity,
    ) -> ReferenceBankSnapshot:
        """Build one validated one-factor variant without modifying the canonical bank.

        The approved values mirror the explicit scale, duration-weight, spread,
        and operating-cost alternatives in the specification.  A variant is a
        separately content-addressed ``sensitivity`` snapshot, never a relaxed
        canonical snapshot.
        """

        discounts = np.asarray(
            historical.initial_curve.discount_factors, dtype=np.float64
        )
        if discounts.shape != (180,) or not np.all(np.isfinite(discounts)):
            raise ReferenceBankError(
                "Sensitivity initial curve must contain 180 finite discounts"
            )
        deposit_initial_history = _initial_deposit_history(historical)
        factor, value = sensitivity.factor, float(sensitivity.value)
        try:
            approved = approved_sensitivity_value(factor, value)
            baseline_value = canonical_sensitivity_value(factor)
        except ValueError as error:
            raise ReferenceBankError(str(error)) from error
        scale = value / 10_000 if factor == "total_assets_mchf" else 1.0
        targets = {name: target * scale for name, target in _TARGETS.items()}
        mortgage_weights = [0.06] * 11
        mortgage_weights[8] = 0.40
        non_maturity_weights = [0.40, 0.30, 0.25, 0.05]
        term_deposit_weights = [0.10, 0.10, 0.50, 0.30]
        loan_spread = 0.015
        cost_multiplier = 1.0
        if factor == "mortgage_ten_year_weight":
            mortgage_weights = [(1.0 - value) / 10] * 11
            mortgage_weights[8] = value
        elif factor == "non_maturity_ten_year_weight":
            non_maturity_weights[2] += non_maturity_weights[3] - value
            non_maturity_weights[3] = value
        elif factor == "term_deposit_ten_year_weight":
            term_deposit_weights[2] += term_deposit_weights[3] - value
            term_deposit_weights[3] = value
        elif factor == "loan_spread_decimal":
            loan_spread = value
        elif factor == "operating_cost_multiplier":
            cost_multiplier = value
        loan_cohorts = {
            "mortgages": _scaled_loan_cohorts(
                _seasoned_loan_cohorts(
                    tuple(range(24, 145, 12)),
                    mortgage_weights,
                    historical.initial_curve.spot_rates,
                    loan_spread,
                ),
                target=targets["mortgages"],
                discounts=discounts,
            ),
            "enterprise_loans": _scaled_loan_cohorts(
                _seasoned_loan_cohorts(
                    (1, 2, 3),
                    (1 / 3,) * 3,
                    historical.initial_curve.spot_rates,
                    loan_spread,
                ),
                target=targets["enterprise_loans"],
                discounts=discounts,
            ),
        }
        deposit_reference_schedules = {
            "non_maturity_deposits": _seasoned_deposit_reference_schedule(
                (1, 2, 12, 120), tuple(non_maturity_weights), 0.005,
                "synthetic sensitivity NMD reference-term allocation",
            ),
            "term_deposits": _seasoned_deposit_reference_schedule(
                (1, 2, 12, 120), tuple(term_deposit_weights), 0.01,
                "synthetic sensitivity term-deposit reference-term allocation",
            ),
        }
        raw = {
            "mortgages": _loan_cohort_cash_flows(loan_cohorts["mortgages"]),
            "enterprise_loans": _loan_cohort_cash_flows(
                loan_cohorts["enterprise_loans"]
            ),
            "investments": _seasoned_ladder(
                tuple(range(36, 181, 12)), (1 / 13,) * 13, 0.01
            ),
            "funding": _seasoned_ladder(
                (3, *range(12, 181, 12)), (1 / 16,) * 16, 0.01
            ),
            **{
                name: schedule.cash_flows.sum(axis=0)
                for name, schedule in deposit_reference_schedules.items()
            },
        }
        deposit_scales = {
            name: targets[name] / float(raw[name] @ discounts)
            for name in deposit_reference_schedules
        }
        ladders = {
            name: _readonly(
                raw[name]
                if name in loan_cohorts
                else raw[name]
                * (
                    deposit_scales[name]
                    if name in deposit_scales
                    else targets[name] / float(raw[name] @ discounts)
                )
            )
            for name in _LADDER_NAMES
        }
        durations = {
            name: _duration(ladder, discounts) for name, ladder in ladders.items()
        }
        assumptions: dict[str, object] = {
            "mortgage_terms_months": list(range(24, 145, 12)),
            "mortgage_weights": mortgage_weights,
            "enterprise_terms_months": [1, 2, 3],
            "deposit_reference_terms_months": [1, 2, 12, 120],
            "non_maturity_weights": non_maturity_weights,
            "term_deposit_weights": term_deposit_weights,
            "loan_spread_decimal": loan_spread,
            "personnel_growth_annual": 0.02,
            "loan_duration_years": _weighted_duration(
                durations, ("mortgages", "enterprise_loans")
            ),
            "deposit_duration_years": _weighted_duration(
                durations, ("non_maturity_deposits", "term_deposits")
            ),
            "legacy_coupon_pricing": "project assumption: fixed 1% synthetic legacy bond coupons",
            "sensitivity_factor": factor,
            "sensitivity_baseline_value": baseline_value,
            "sensitivity_value": approved,
        }
        snapshot = _make_snapshot(
            profile="sensitivity",
            as_of_date=str(historical.initial_curve.as_of_date.astype("datetime64[D]")),
            initial_curve_identity=historical.source_hash,
            cash=targets["cash"],
            ladders=ladders,
            loan_cohorts=loan_cohorts,
            deposit_reference_schedules={
                name: DepositReferenceSchedule(
                    terms_months=schedule.terms_months,
                    weights=schedule.weights,
                    cash_flows=schedule.cash_flows * deposit_scales[name],
                    provenance=schedule.provenance,
                )
                for name, schedule in deposit_reference_schedules.items()
            },
            deposit_initial_history=deposit_initial_history,
            target_economic_values=targets,
            equity=targets["cash"]
            + targets["investments"]
            + targets["mortgages"]
            + targets["enterprise_loans"]
            - targets["non_maturity_deposits"]
            - targets["term_deposits"]
            - targets["funding"],
            assumptions=assumptions,
            target_value_errors={
                name: abs(float(ladder @ discounts) - targets[name]) / targets[name]
                for name, ladder in ladders.items()
            },
            provenance={
                "source_system": "synthetic Reference Bank sensitivity",
                "product_mapping": "seasoned six-ladder project mapping",
                "assumptions": "one-factor canonical sensitivity",
                "valuation": "initial SNB curve discounted cash-flow values",
                "market": _market_provenance(
                    historical.source_hash,
                    str(historical.initial_curve.as_of_date.astype("datetime64[D]")),
                ),
                "sensitivity": f"{factor}: {baseline_value} -> {approved}",
            },
            personnel_cost=3.0 * cost_multiplier,
            material_cost=1.0 * cost_multiplier,
            currency="CHF",
            unit="mCHF",
        )
        self.validate(snapshot, discounts=discounts)
        return snapshot

    def validate(
        self,
        snapshot: ReferenceBankSnapshot,
        *,
        discounts: np.ndarray | None = None,
        market_curve_identity: str | None = None,
        market_as_of_date: str | None = None,
    ) -> None:
        """Validate generic ALM invariants, plus canonical targets when declared."""

        if snapshot.schema_version != _SCHEMA_VERSION:
            raise ReferenceBankError("Reference Bank schema version is unsupported")
        if snapshot.profile not in _PROFILES:
            raise ReferenceBankError("Reference Bank profile is unsupported")
        if not _is_iso_date(snapshot.as_of_date) or not snapshot.initial_curve_identity:
            raise ReferenceBankError(
                "Reference Bank requires as-of date, curve identity and provenance"
            )
        if (
            len(snapshot.currency) != 3
            or snapshot.currency != snapshot.currency.upper()
            or snapshot.unit != f"m{snapshot.currency}"
        ):
            raise ReferenceBankError(
                "Reference Bank currency and unit must be one currency in millions"
            )
        if (
            market_curve_identity
            and market_curve_identity != snapshot.initial_curve_identity
        ):
            raise ReferenceBankError(
                "Reference Bank curve identity does not match market"
            )
        if market_as_of_date and market_as_of_date != snapshot.as_of_date:
            raise ReferenceBankError(
                "Reference Bank valuation date does not match market"
            )
        if set(snapshot.ladders) != set(_LADDER_NAMES):
            raise ReferenceBankError("Reference Bank ladders have an invalid identity")
        if set(snapshot.loan_cohorts) != {"mortgages", "enterprise_loans"}:
            raise ReferenceBankError("Reference Bank fixed-rate loan cohorts are incomplete")
        if set(snapshot.deposit_reference_schedules) != {
            "non_maturity_deposits",
            "term_deposits",
        }:
            raise ReferenceBankError(
                "Reference Bank deposit reference-term schedules are incomplete"
            )
        history = snapshot.deposit_initial_history
        available_dates = history.available_observation_dates
        available_yields = history.available_six_month_yields
        if (
            len(history.target_dates) != 2
            or len(history.observation_dates) != 2
            or len(history.six_month_yields) != 2
            or not available_dates
            or len(available_dates) != len(available_yields)
            or not all(_is_iso_date(date) for date in history.target_dates)
            or not all(_is_iso_date(date) for date in history.observation_dates)
            or not all(_is_iso_date(date) for date in available_dates)
            or any(
                earlier >= later
                for earlier, later in pairwise(available_dates)
            )
            or any(
                observation > target
                for observation, target in zip(
                    history.observation_dates, history.target_dates, strict=True
                )
            )
            or not np.all(np.isfinite(history.six_month_yields))
            or not np.all(np.isfinite(available_yields))
            or not isinstance(history.source_identity, str)
            or not isinstance(history.identity, str)
            or not history.source_identity
            or not history.identity
            or history.target_dates
            != deposit_initial_history_target_dates(snapshot.as_of_date)
            or history.identity
            != deposit_initial_history_identity(
                history.target_dates,
                history.observation_dates,
                history.six_month_yields,
                available_dates,
                available_yields,
                history.source_identity,
            )
        ):
            raise ReferenceBankError("Reference Bank deposit initial history is invalid")
        for target, observation, yield_ in zip(
            history.target_dates,
            history.observation_dates,
            history.six_month_yields,
            strict=True,
        ):
            eligible = [
                index for index, date in enumerate(available_dates) if date <= target
            ]
            if not eligible:
                raise ReferenceBankError("Reference Bank deposit initial history is invalid")
            selected = eligible[-1]
            if (
                observation != available_dates[selected]
                or not np.isclose(yield_, available_yields[selected], rtol=0.0, atol=0.0)
            ):
                raise ReferenceBankError("Reference Bank deposit initial history is invalid")
        for name, schedule in snapshot.deposit_reference_schedules.items():
            if (
                schedule.terms_months != (1, 2, 12, 120)
                or any(type(term) is not int for term in schedule.terms_months)
                or len(schedule.weights) != 4
                or len(schedule.terms_months) != len(schedule.weights)
                or not np.all(np.isfinite(schedule.weights))
                or any(weight < 0 for weight in schedule.weights)
                or not np.isclose(sum(schedule.weights), 1.0)
                or schedule.cash_flows.shape != (4, 180)
                or not np.all(np.isfinite(schedule.cash_flows))
                or np.any(schedule.cash_flows < 0)
                or schedule.cash_flows.flags.writeable
                or not isinstance(schedule.provenance, str)
                or not schedule.provenance
            ):
                raise ReferenceBankError(
                    f"Reference Bank {name} reference-term schedule is invalid"
                )
            if not np.allclose(
                schedule.cash_flows.sum(axis=0),
                snapshot.ladders[name],
                rtol=0.0,
                atol=1e-8,
            ):
                raise ReferenceBankError(
                    f"Reference Bank {name} reference-term schedule does not match its ladder"
                )
            weight_key = (
                "non_maturity_weights"
                if name == "non_maturity_deposits"
                else "term_deposit_weights"
            )
            try:
                declared_terms = tuple(
                    snapshot.product_assumptions["deposit_reference_terms_months"]
                )
                declared_weights = tuple(snapshot.product_assumptions[weight_key])
            except (KeyError, TypeError) as error:
                raise ReferenceBankError(
                    "Reference Bank deposit reference-term assumptions are incomplete"
                ) from error
            if (
                declared_terms != schedule.terms_months
                or len(declared_weights) != len(schedule.weights)
                or not np.allclose(declared_weights, schedule.weights)
            ):
                raise ReferenceBankError(
                    f"Reference Bank {name} reference-term schedule does not match declared assumptions"
                )
        for name, cohorts in snapshot.loan_cohorts.items():
            if not cohorts:
                raise ReferenceBankError(
                    f"Reference Bank {name} requires at least one fixed-rate cohort"
                )
            for cohort in cohorts:
                if (
                    cohort.principal_cash_flows.shape != (180,)
                    or not np.all(np.isfinite(cohort.principal_cash_flows))
                    or np.any(cohort.principal_cash_flows < 0)
                    or not np.isfinite(cohort.monthly_coupon_rate)
                    or cohort.monthly_coupon_rate < 0
                    or cohort.principal_cash_flows.flags.writeable
                ):
                    raise ReferenceBankError(
                        f"Reference Bank {name} fixed-rate cohort is invalid"
                    )
            if snapshot.ladders[name].shape == (180,) and not np.allclose(
                _loan_cohort_cash_flows(cohorts), snapshot.ladders[name], atol=1e-8
            ):
                raise ReferenceBankError(
                    f"Reference Bank {name} ladder does not match its fixed-rate cohorts"
                )
        if set(snapshot.target_economic_values) != set(_TARGETS):
            raise ReferenceBankError(
                "Reference Bank value targets have an invalid identity"
            )
        if set(snapshot.target_value_errors) != set(_LADDER_NAMES):
            raise ReferenceBankError(
                "Reference Bank value-error targets have an invalid identity"
            )
        for name, ladder in snapshot.ladders.items():
            if ladder.shape != (180,):
                raise ReferenceBankError(
                    f"Reference Bank ladder {name} must contain 180 entries"
                )
            if not np.all(np.isfinite(ladder)) or np.any(ladder < -1e-8):
                raise ReferenceBankError(
                    f"Reference Bank ladder {name} has invalid nominal cash flows"
                )
            if ladder.flags.writeable:
                raise ReferenceBankError(
                    f"Reference Bank ladder {name} must be immutable"
                )
        if any(
            not np.isfinite(value) or value < 0
            for value in snapshot.target_value_errors.values()
        ):
            raise ReferenceBankError("Reference Bank value-error targets are invalid")
        if not _PROVENANCE_FIELDS.issubset(snapshot.provenance) or any(
            not isinstance(snapshot.provenance[field], str)
            or not snapshot.provenance[field]
            for field in _PROVENANCE_FIELDS
        ):
            raise ReferenceBankError("Reference Bank provenance is incomplete")
        if snapshot.provenance["market"] != _market_provenance(
            snapshot.initial_curve_identity, snapshot.as_of_date
        ):
            raise ReferenceBankError(
                "Reference Bank market provenance does not match curve identity and valuation date"
            )
        if (
            any(
                not np.isfinite(value) or value < 0
                for value in snapshot.target_economic_values.values()
            )
            or not np.isfinite(snapshot.cash)
            or snapshot.cash < 0
        ):
            raise ReferenceBankError("Reference Bank economic values are invalid")
        if abs(snapshot.target_economic_values["cash"] - snapshot.cash) > 1e-8:
            raise ReferenceBankError(
                "Reference Bank cash does not match its valuation target"
            )
        if (
            not np.isfinite(snapshot.equity)
            or abs(snapshot.total_assets - snapshot.total_liabilities - snapshot.equity)
            > 1e-8
        ):
            raise ReferenceBankError("Reference Bank balance-sheet identity is invalid")
        if any(
            not np.isfinite(value) or value < 0
            for value in (
                snapshot.personnel_cost,
                snapshot.material_cost,
                snapshot.loan_duration_years,
                snapshot.deposit_duration_years,
            )
        ):
            raise ReferenceBankError(
                "Reference Bank costs or product assumptions are invalid"
            )
        if discounts is not None:
            discounts = np.asarray(discounts, dtype=np.float64)
            if (
                discounts.shape != (180,)
                or not np.all(np.isfinite(discounts))
                or np.any(discounts <= 0)
            ):
                raise ReferenceBankError(
                    "Reference Bank valuation discounts must be 180 positive finite values"
                )
            for name in _LADDER_NAMES:
                target = snapshot.target_economic_values[name]
                actual = float(snapshot.ladders[name] @ discounts)
                error = abs(actual - target) / target if target else abs(actual)
                if error > max(snapshot.target_value_errors[name], 1e-6):
                    raise ReferenceBankError(
                        f"Reference Bank value target failed for {name}"
                    )
        if snapshot.content_hash != _content_hash(snapshot):
            raise ReferenceBankError(
                "Reference Bank content hash does not match its contents"
            )
        if snapshot.profile == "canonical":
            self._validate_canonical(snapshot)
        if snapshot.profile == "sensitivity":
            self._validate_sensitivity(snapshot)

    @staticmethod
    def schema() -> dict[str, object]:
        """Return the versioned persisted-input contract for bank adapters."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "artifact_semantics": artifact_semantics("snapshot"),
            "profiles": sorted(_PROFILES),
            "ladder_names": list(_LADDER_NAMES),
            "loan_cohort_products": ["mortgages", "enterprise_loans"],
            "deposit_reference_schedule_products": [
                "non_maturity_deposits",
                "term_deposits",
            ],
            "deposit_initial_history": {
                "six_month_yield_tenor_months": 6,
                "required_fields": [
                    "target_dates",
                    "observation_dates",
                    "six_month_yields",
                    "available_observation_dates",
                    "available_six_month_yields",
                    "source_identity",
                    "identity",
                ],
            },
            "ladder_length_months": 180,
            "required_provenance": sorted(_PROVENANCE_FIELDS),
            "unit_rule": "unit must be m followed by the declared ISO currency",
            "sensitivity_factors": {
                factor: {
                    "canonical_value": definition.canonical_value,
                    "approved_values": list(definition.approved_values),
                }
                for factor, definition in REFERENCE_BANK_SENSITIVITIES.items()
            },
        }

    @staticmethod
    def _validate_canonical(snapshot: ReferenceBankSnapshot) -> None:
        if snapshot.currency != "CHF" or snapshot.unit != "mCHF":
            raise ReferenceBankError("Canonical Reference Bank requires CHF and mCHF")
        if (
            snapshot.as_of_date != _CANONICAL_AS_OF_DATE
            or snapshot.initial_curve_identity != _CANONICAL_CURVE_IDENTITY
        ):
            raise ReferenceBankError(
                "Canonical Reference Bank requires the locked valuation date and curve identity"
            )
        if dict(snapshot.target_economic_values) != _TARGETS:
            raise ReferenceBankError(
                "Canonical Reference Bank economic-value targets are invalid"
            )
        if (
            abs(snapshot.total_assets - 10_000) > 1e-8
            or abs(snapshot.equity - 1_000) > 1e-8
        ):
            raise ReferenceBankError(
                "Canonical Reference Bank balance-sheet identity is invalid"
            )
        if snapshot.loan_duration_years >= 5 or snapshot.deposit_duration_years >= 3:
            raise ReferenceBankError(
                "Canonical Reference Bank duration targets are invalid"
            )

    @staticmethod
    def _validate_sensitivity(snapshot: ReferenceBankSnapshot) -> None:
        """Require an auditable, registered declaration for every variant."""

        try:
            factor = str(snapshot.product_assumptions["sensitivity_factor"])
            baseline = float(snapshot.product_assumptions["sensitivity_baseline_value"])
            value = float(snapshot.product_assumptions["sensitivity_value"])
            approved = approved_sensitivity_value(factor, value)
            expected_baseline = canonical_sensitivity_value(factor)
        except (KeyError, TypeError, ValueError) as error:
            raise ReferenceBankError(
                "Sensitivity Reference Bank lacks an approved one-factor declaration"
            ) from error
        if baseline != expected_baseline or snapshot.provenance.get("sensitivity") != (
            f"{factor}: {baseline} -> {approved}"
        ):
            raise ReferenceBankError(
                "Sensitivity Reference Bank declaration does not match its approved factor"
            )

    def save(self, snapshot: ReferenceBankSnapshot, path: Path) -> None:
        self.validate(snapshot)
        path.write_text(
            json.dumps(self.serialize(snapshot), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def load(self, path: Path) -> ReferenceBankSnapshot:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["schema_version"] != _SCHEMA_VERSION:
                raise ReferenceBankError("Reference Bank schema version is unsupported")
            if error := artifact_semantics_error(data, "snapshot"):
                raise ReferenceBankError(error)
            snapshot = _make_snapshot(
                profile=data["profile"],
                as_of_date=data["as_of_date"],
                initial_curve_identity=data["initial_curve_identity"],
                cash=data["cash"],
                ladders={
                    name: np.asarray(values, dtype=np.float64)
                    for name, values in data["ladders"].items()
                },
                loan_cohorts={
                    name: tuple(
                        LoanCohort(
                            principal_cash_flows=np.asarray(
                                cohort["principal_cash_flows"], dtype=np.float64
                            ),
                            monthly_coupon_rate=float(cohort["monthly_coupon_rate"]),
                        )
                        for cohort in cohorts
                    )
                    for name, cohorts in data["loan_cohorts"].items()
                },
                deposit_reference_schedules={
                    name: DepositReferenceSchedule(
                        terms_months=tuple(schedule["terms_months"]),
                        weights=tuple(schedule["weights"]),
                        cash_flows=np.asarray(schedule["cash_flows"], dtype=np.float64),
                        provenance=schedule["provenance"],
                    )
                    for name, schedule in data["deposit_reference_schedules"].items()
                },
                deposit_initial_history=DepositInitialHistory(
                    target_dates=tuple(data["deposit_initial_history"]["target_dates"]),
                    observation_dates=tuple(
                        data["deposit_initial_history"]["observation_dates"]
                    ),
                    six_month_yields=tuple(
                        data["deposit_initial_history"]["six_month_yields"]
                    ),
                    available_observation_dates=tuple(
                        data["deposit_initial_history"]["available_observation_dates"]
                    ),
                    available_six_month_yields=tuple(
                        data["deposit_initial_history"]["available_six_month_yields"]
                    ),
                    source_identity=data["deposit_initial_history"]["source_identity"],
                    identity=data["deposit_initial_history"]["identity"],
                ),
                target_economic_values=data["target_economic_values"],
                equity=data["equity"],
                assumptions=data["product_assumptions"],
                target_value_errors=data["target_value_errors"],
                provenance=data["provenance"],
                personnel_cost=data["personnel_cost"],
                material_cost=data["material_cost"],
                currency=data["currency"],
                unit=data["unit"],
            )
            if data != _to_data(snapshot):
                raise ReferenceBankError(
                    "Reference Bank saved fields do not match the canonical contract or snapshot schema"
                )
            if data["content_hash"] != snapshot.content_hash:
                raise ReferenceBankError(
                    "Reference Bank content hash does not match saved contents"
                )
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise ReferenceBankError(
                f"Could not load Reference Bank snapshot: {error}"
            ) from error
        try:
            self.validate(snapshot)
        except (KeyError, TypeError, ValueError) as error:
            raise ReferenceBankError(
                f"Reference Bank product assumptions are invalid: {error}"
            ) from error
        return snapshot

    def write_table_one(self, snapshot: ReferenceBankSnapshot, path: Path) -> None:
        self.validate(snapshot)
        document = self.table_one(snapshot)
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def serialize(self, snapshot: ReferenceBankSnapshot) -> dict[str, object]:
        """Return the complete, hashable persisted representation."""

        self.validate(snapshot)
        return _to_data(snapshot)

    def table_one(self, snapshot: ReferenceBankSnapshot) -> dict[str, object]:
        """Return an assumption-labeled review table for any valid snapshot."""

        self.validate(snapshot)
        return {
            "table": {name: snapshot.target_economic_values[name] for name in _TARGETS},
            "equity": snapshot.equity,
            "assumptions": dict(snapshot.product_assumptions),
            "provenance": dict(snapshot.provenance),
            "currency": snapshot.currency,
            "unit": snapshot.unit,
            "schema_version": snapshot.schema_version,
            "profile": snapshot.profile,
            "content_hash": snapshot.content_hash,
        }


def _seasoned_loan_cohorts(
    terms: tuple[int, ...],
    weights: tuple[float, ...] | list[float],
    spot_rates: np.ndarray,
    spread: float,
) -> tuple[LoanCohort, ...]:
    cohorts: list[LoanCohort] = []
    for term, weight in zip(terms, weights, strict=True):
        coupon = float(np.expm1((spot_rates[term - 1] + spread) / 12.0))
        schedule = np.zeros(180, dtype=np.float64)
        for age in range(term):
            schedule[term - age - 1] = weight / term
        cohorts.append(
            LoanCohort(
                principal_cash_flows=schedule, monthly_coupon_rate=coupon
            )
        )
    return tuple(cohorts)


def _loan_cohort_cash_flows(cohorts: tuple[LoanCohort, ...]) -> np.ndarray:
    result = np.zeros(180, dtype=np.float64)
    for cohort in cohorts:
        outstanding = np.cumsum(cohort.principal_cash_flows[::-1])[::-1]
        result += cohort.principal_cash_flows + cohort.monthly_coupon_rate * outstanding
    return result


def _scaled_loan_cohorts(
    cohorts: tuple[LoanCohort, ...], *, target: float, discounts: np.ndarray
) -> tuple[LoanCohort, ...]:
    current_value = float(_loan_cohort_cash_flows(cohorts) @ discounts)
    if current_value <= 0:
        raise ReferenceBankError("Fixed-rate loan cohorts have no positive value")
    scale = target / current_value
    return tuple(
        LoanCohort(
            principal_cash_flows=cohort.principal_cash_flows * scale,
            monthly_coupon_rate=cohort.monthly_coupon_rate,
        )
        for cohort in cohorts
    )


def _seasoned_ladder(
    terms: tuple[int, ...], weights: tuple[float, ...] | list[float], coupon: float
) -> np.ndarray:
    ladder = np.zeros(180, dtype=np.float64)
    for term, weight in zip(terms, weights, strict=True):
        for age in range(term):
            remaining = term - age
            notional = weight / term
            ladder[:remaining] += notional * coupon / 12
            ladder[remaining - 1] += notional
    return ladder


def _seasoned_deposit_reference_schedule(
    terms: tuple[int, ...], weights: tuple[float, ...], coupon: float, provenance: str
) -> DepositReferenceSchedule:
    return DepositReferenceSchedule(
        terms_months=terms,
        weights=weights,
        cash_flows=np.stack(
            [
                _seasoned_ladder((term,), (weight,), coupon)
                for term, weight in zip(terms, weights, strict=True)
            ]
        ),
        provenance=provenance,
    )


def _initial_deposit_history(
    historical: HistoricalTermStructures,
) -> DepositInitialHistory:
    return MarketScenarioModel().deposit_initial_history(
        historical,
        as_of_date=str(historical.initial_curve.as_of_date.astype("datetime64[D]")),
    )


def _duration(ladder: np.ndarray, discounts: np.ndarray) -> float:
    value = float(ladder @ discounts)
    return float((ladder * discounts @ (np.arange(1, 181) / 12)) / value)


def _weighted_duration(durations: Mapping[str, float], names: tuple[str, str]) -> float:
    return sum(durations[name] * _TARGETS[name] for name in names) / sum(
        _TARGETS[name] for name in names
    )


def _make_snapshot(
    *,
    profile: str,
    as_of_date: str,
    initial_curve_identity: str,
    cash: float,
    ladders: Mapping[str, np.ndarray],
    loan_cohorts: Mapping[str, tuple[LoanCohort, ...]],
    deposit_reference_schedules: Mapping[str, DepositReferenceSchedule],
    deposit_initial_history: DepositInitialHistory,
    target_economic_values: Mapping[str, float],
    equity: float,
    assumptions: Mapping[str, object],
    target_value_errors: Mapping[str, float] | None = None,
    provenance: Mapping[str, str],
    personnel_cost: float = 3.0,
    material_cost: float = 1.0,
    currency: str = "CHF",
    unit: str = "mCHF",
) -> ReferenceBankSnapshot:
    errors = dict(target_value_errors or {name: 0.0 for name in _LADDER_NAMES})
    fields = {
        "schema_version": _SCHEMA_VERSION,
        "profile": profile,
        "as_of_date": as_of_date,
        "initial_curve_identity": initial_curve_identity,
        "cash": float(cash),
        "equity": float(equity),
        "ladders": MappingProxyType(
            {name: _readonly(value) for name, value in ladders.items()}
        ),
        "loan_cohorts": MappingProxyType(
            {
                name: tuple(
                    LoanCohort(
                        principal_cash_flows=_readonly(cohort.principal_cash_flows),
                        monthly_coupon_rate=float(cohort.monthly_coupon_rate),
                    )
                    for cohort in cohorts
                )
                for name, cohorts in loan_cohorts.items()
            }
        ),
        "deposit_reference_schedules": MappingProxyType(
            {
                name: DepositReferenceSchedule(
                    terms_months=tuple(schedule.terms_months),
                    weights=tuple(float(weight) for weight in schedule.weights),
                    cash_flows=_readonly(schedule.cash_flows),
                    provenance=schedule.provenance,
                )
                for name, schedule in deposit_reference_schedules.items()
            }
        ),
        "deposit_initial_history": DepositInitialHistory(
            target_dates=tuple(deposit_initial_history.target_dates),
            observation_dates=tuple(deposit_initial_history.observation_dates),
            six_month_yields=tuple(
                float(value) for value in deposit_initial_history.six_month_yields
            ),
            available_observation_dates=tuple(
                deposit_initial_history.available_observation_dates
            ),
            available_six_month_yields=tuple(
                float(value)
                for value in deposit_initial_history.available_six_month_yields
            ),
            source_identity=deposit_initial_history.source_identity,
            identity=deposit_initial_history.identity,
        ),
        "target_economic_values": MappingProxyType(
            {name: float(value) for name, value in target_economic_values.items()}
        ),
        "target_value_errors": MappingProxyType(errors),
        "product_assumptions": MappingProxyType(dict(assumptions)),
        "provenance": MappingProxyType(dict(provenance)),
        "personnel_cost": float(personnel_cost),
        "material_cost": float(material_cost),
        "currency": currency,
        "unit": unit,
    }
    provisional = ReferenceBankSnapshot(**fields, content_hash="")
    return ReferenceBankSnapshot(**fields, content_hash=_content_hash(provisional))


def _to_data(snapshot: ReferenceBankSnapshot) -> dict[str, object]:
    return {
        "artifact_semantics": artifact_semantics("snapshot"),
        "schema_version": snapshot.schema_version,
        "profile": snapshot.profile,
        "as_of_date": snapshot.as_of_date,
        "initial_curve_identity": snapshot.initial_curve_identity,
        "cash": snapshot.cash,
        "equity": snapshot.equity,
        "ladders": {name: ladder.tolist() for name, ladder in snapshot.ladders.items()},
        "loan_cohorts": {
            name: [
                {
                    "principal_cash_flows": cohort.principal_cash_flows.tolist(),
                    "monthly_coupon_rate": cohort.monthly_coupon_rate,
                }
                for cohort in cohorts
            ]
            for name, cohorts in snapshot.loan_cohorts.items()
        },
        "deposit_reference_schedules": {
            name: {
                "terms_months": list(schedule.terms_months),
                "weights": list(schedule.weights),
                "cash_flows": schedule.cash_flows.tolist(),
                "provenance": schedule.provenance,
            }
            for name, schedule in snapshot.deposit_reference_schedules.items()
        },
        "deposit_initial_history": {
            "target_dates": list(snapshot.deposit_initial_history.target_dates),
            "observation_dates": list(
                snapshot.deposit_initial_history.observation_dates
            ),
            "six_month_yields": list(snapshot.deposit_initial_history.six_month_yields),
            "available_observation_dates": list(
                snapshot.deposit_initial_history.available_observation_dates
            ),
            "available_six_month_yields": list(
                snapshot.deposit_initial_history.available_six_month_yields
            ),
            "source_identity": snapshot.deposit_initial_history.source_identity,
            "identity": snapshot.deposit_initial_history.identity,
        },
        "target_economic_values": dict(snapshot.target_economic_values),
        "target_value_errors": dict(snapshot.target_value_errors),
        "product_assumptions": dict(snapshot.product_assumptions),
        "provenance": dict(snapshot.provenance),
        "personnel_cost": snapshot.personnel_cost,
        "material_cost": snapshot.material_cost,
        "currency": snapshot.currency,
        "unit": snapshot.unit,
        "content_hash": snapshot.content_hash,
    }


def _content_hash(snapshot: ReferenceBankSnapshot) -> str:
    data = _to_data(snapshot)
    data.pop("content_hash", None)
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _is_iso_date(value: str) -> bool:
    try:
        parsed = np.datetime64(value, "D")
    except (TypeError, ValueError):
        return False
    return not np.isnat(parsed)


def _market_provenance(curve_identity: str, as_of_date: str) -> str:
    return f"{curve_identity}@{as_of_date}"
