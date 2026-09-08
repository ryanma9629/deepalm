"""Synthetic, auditable Reference Bank construction for Deep ALM."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from deepalm.runner import OperationalRunError
from deepalm.term_structures import HistoricalTermStructures

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
_SCHEMA_VERSION = 1
_PROFILES = {"canonical", "imported"}
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
class ReferenceBankSnapshot:
    """Immutable initial bank state shared by the two model horizons."""

    schema_version: int
    profile: str
    as_of_date: str
    initial_curve_identity: str
    cash: float
    equity: float
    ladders: Mapping[str, np.ndarray]
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
        mortgage_weights = [0.06] * 11
        mortgage_weights[8] = 0.40
        raw = {
            "mortgages": _seasoned_ladder(
                tuple(range(24, 145, 12)), mortgage_weights, 0.015
            ),
            "enterprise_loans": _seasoned_ladder((1, 2, 3), (1 / 3,) * 3, 0.015),
            "investments": _seasoned_ladder(
                tuple(range(36, 181, 12)), (1 / 13,) * 13, 0.01
            ),
            "funding": _seasoned_ladder((3, *range(12, 181, 12)), (1 / 16,) * 16, 0.01),
            "non_maturity_deposits": _seasoned_ladder(
                (1, 2, 12, 120), (0.40, 0.30, 0.25, 0.05), 0.005
            ),
            "term_deposits": _seasoned_ladder(
                (1, 2, 12, 120), (0.10, 0.10, 0.50, 0.30), 0.01
            ),
        }
        ladders = {
            name: _readonly(raw[name] * (_TARGETS[name] / float(raw[name] @ discounts)))
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

    @staticmethod
    def schema() -> dict[str, object]:
        """Return the versioned persisted-input contract for bank adapters."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "profiles": sorted(_PROFILES),
            "ladder_names": list(_LADDER_NAMES),
            "ladder_length_months": 180,
            "required_provenance": sorted(_PROVENANCE_FIELDS),
            "unit_rule": "unit must be m followed by the declared ISO currency",
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
            snapshot = _make_snapshot(
                profile=data["profile"],
                as_of_date=data["as_of_date"],
                initial_curve_identity=data["initial_curve_identity"],
                cash=data["cash"],
                ladders={
                    name: np.asarray(values, dtype=np.float64)
                    for name, values in data["ladders"].items()
                },
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
        "schema_version": snapshot.schema_version,
        "profile": snapshot.profile,
        "as_of_date": snapshot.as_of_date,
        "initial_curve_identity": snapshot.initial_curve_identity,
        "cash": snapshot.cash,
        "equity": snapshot.equity,
        "ladders": {name: ladder.tolist() for name, ladder in snapshot.ladders.items()},
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
