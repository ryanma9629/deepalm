"""Historical SNB NSS term structures for Deep ALM market calibration."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from deepalm.runner import OperationalRunError

_PARAMETERS = ("b0", "b1", "b2", "b3", "t1", "t2")
_CALIBRATION_START = pd.Timestamp("2005-01-01")
_CALIBRATION_END = pd.Timestamp("2022-07-15")
_INITIAL_CURVE_DATE = pd.Timestamp("2022-07-15")
_MONTHLY_TENORS_YEARS = np.arange(1, 181, dtype=np.float64) / 12.0


class TermStructureError(OperationalRunError):
    """Raised when SNB inputs cannot produce a valid historical curve set."""


@dataclass(frozen=True)
class TermStructureSnapshot:
    """One continuously compounded spot curve and its discrete forward form."""

    as_of_date: np.datetime64
    tenors_years: np.ndarray
    spot_rates: np.ndarray
    discount_factors: np.ndarray
    monthly_forwards: np.ndarray


@dataclass(frozen=True)
class HistoricalTermStructures:
    """Validated calibration curves and the paper-window initial curve."""

    dates: np.ndarray
    tenors_years: np.ndarray
    spot_rates: np.ndarray
    discount_factors: np.ndarray
    monthly_forwards: np.ndarray
    initial_curve: TermStructureSnapshot
    source_path: Path
    source_hash: str
    round_trip_error: float


class MarketScenarioModel:
    """Load the historical term structures used by the later HJM scenario model."""

    def load_historical_term_structures(
        self, source_path: Path
    ) -> HistoricalTermStructures:
        """Parse the SNB NSS export and reconstruct its calibration-window curves."""

        parameter_table = _load_parameter_table(source_path)
        calibration_table = _select_calibration_window(parameter_table)
        spot_rates = _reconstruct_spot_rates(calibration_table)
        discount_factors = np.exp(-spot_rates * _MONTHLY_TENORS_YEARS)
        monthly_forwards = _monthly_forwards(discount_factors)
        round_trip_error = _round_trip_error(
            spot_rates, discount_factors, monthly_forwards
        )
        if round_trip_error > 1e-10:
            raise TermStructureError(
                "Spot, discount, and forward curves do not round-trip within 1e-10",
                diagnostics={"round_trip_error": round_trip_error},
            )

        dates = calibration_table.index.to_numpy(dtype="datetime64[ns]")
        initial_curve_index = np.flatnonzero(
            dates == _INITIAL_CURVE_DATE.to_datetime64()
        )
        if len(initial_curve_index) != 1:
            raise TermStructureError(
                "The canonical 2022-07-15 initial curve is absent from calibration data"
            )
        index = int(initial_curve_index[0])
        initial_curve = TermStructureSnapshot(
            as_of_date=dates[index],
            tenors_years=_readonly(_MONTHLY_TENORS_YEARS),
            spot_rates=_readonly(spot_rates[index]),
            discount_factors=_readonly(discount_factors[index]),
            monthly_forwards=_readonly(monthly_forwards[index]),
        )
        return HistoricalTermStructures(
            dates=_readonly(dates),
            tenors_years=_readonly(_MONTHLY_TENORS_YEARS),
            spot_rates=_readonly(spot_rates),
            discount_factors=_readonly(discount_factors),
            monthly_forwards=_readonly(monthly_forwards),
            initial_curve=initial_curve,
            source_path=source_path,
            source_hash=_sha256(source_path),
            round_trip_error=round_trip_error,
        )


def _load_parameter_table(source_path: Path) -> pd.DataFrame:
    header_row = _validate_metadata_and_find_header(source_path)
    try:
        raw = pd.read_csv(
            source_path,
            sep=";",
            skiprows=header_row,
            encoding="utf-8-sig",
            dtype={"Date": "string", "d0": "string"},
        )
    except (OSError, pd.errors.ParserError) as error:
        raise TermStructureError(f"Could not parse SNB source: {error}") from error

    expected_columns = {"Date", "d0", "Value"}
    if set(raw.columns) != expected_columns:
        raise TermStructureError(
            "SNB source must contain exactly Date, d0, and Value columns",
            diagnostics={"columns": list(raw.columns)},
        )
    raw["Date"] = pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="coerce")
    raw["d0"] = raw["d0"].str.strip()
    if raw["Date"].isna().any() or raw["d0"].isna().any():
        raise TermStructureError("SNB source contains invalid dates or parameter codes")
    unexpected_parameters = sorted(set(raw["d0"]) - set(_PARAMETERS))
    missing_parameters = sorted(set(_PARAMETERS) - set(raw["d0"]))
    if unexpected_parameters or missing_parameters:
        raise TermStructureError(
            "SNB source parameter codes must be b0, b1, b2, b3, t1, and t2",
            diagnostics={
                "unexpected_parameters": unexpected_parameters,
                "missing_parameters": missing_parameters,
            },
        )
    duplicates = raw.duplicated(["Date", "d0"], keep=False)
    if duplicates.any():
        duplicate_rows = raw.loc[duplicates, ["Date", "d0"]].head(4)
        raise TermStructureError(
            "SNB source contains duplicate date/parameter keys",
            diagnostics={"duplicates": duplicate_rows.astype(str).to_dict("records")},
        )

    raw["Value"] = pd.to_numeric(raw["Value"], errors="coerce")
    table = raw.pivot(index="Date", columns="d0", values="Value").reindex(
        columns=_PARAMETERS
    )
    blank_dates = table.isna().all(axis=1)
    table = table.loc[~blank_dates]
    partial_dates = table.isna().any(axis=1)
    if partial_dates.any():
        dates = ", ".join(
            timestamp.strftime("%Y-%m-%d")
            for timestamp in table.index[partial_dates][:5]
        )
        raise TermStructureError(f"SNB source contains partial parameters for {dates}")
    if table.empty:
        raise TermStructureError("SNB source contains no complete parameter dates")
    if not np.isfinite(table.to_numpy(dtype=np.float64)).all():
        raise TermStructureError(
            "SNB source contains non-finite complete parameter values"
        )
    if (table[["t1", "t2"]] <= 0).any(axis=None):
        raise TermStructureError("SNB source requires positive t1 and t2 parameters")
    return table.sort_index()


def _validate_metadata_and_find_header(source_path: Path) -> int:
    try:
        with source_path.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.reader(source, delimiter=";"))
    except OSError as error:
        raise TermStructureError(f"Could not read SNB source: {error}") from error
    if not rows or rows[0] != ["CubeId", "rendopar"]:
        raise TermStructureError("SNB source must declare CubeId rendopar")
    for index, row in enumerate(rows):
        if row == ["Date", "d0", "Value"]:
            return index
    raise TermStructureError("SNB source is missing the Date/d0/Value header")


def _select_calibration_window(parameter_table: pd.DataFrame) -> pd.DataFrame:
    window = parameter_table.loc[_CALIBRATION_START:_CALIBRATION_END]
    if window.empty:
        raise TermStructureError(
            "SNB source has no complete data in the calibration window"
        )
    if _INITIAL_CURVE_DATE not in window.index:
        raise TermStructureError(
            "SNB source is missing the canonical initial-curve date 2022-07-15"
        )
    return window


def _reconstruct_spot_rates(parameter_table: pd.DataFrame) -> np.ndarray:
    parameters = parameter_table.to_numpy(dtype=np.float64)
    beta0, beta1, beta2, beta3 = (parameters[:, index] / 100.0 for index in range(4))
    tau1, tau2 = parameters[:, 4], parameters[:, 5]
    x1 = _MONTHLY_TENORS_YEARS[None, :] / tau1[:, None]
    x2 = _MONTHLY_TENORS_YEARS[None, :] / tau2[:, None]
    loading1 = -np.expm1(-x1) / x1
    loading2 = -np.expm1(-x2) / x2
    spots = (
        beta0[:, None]
        + beta1[:, None] * loading1
        + beta2[:, None] * (loading1 - np.exp(-x1))
        + beta3[:, None] * (loading2 - np.exp(-x2))
    )
    if not np.isfinite(spots).all():
        raise TermStructureError("NSS reconstruction produced non-finite spot rates")
    return spots


def _monthly_forwards(discount_factors: np.ndarray) -> np.ndarray:
    log_discounts = np.log(discount_factors)
    initial_log_discount = np.zeros((len(discount_factors), 1), dtype=np.float64)
    return -np.diff(
        np.concatenate((initial_log_discount, log_discounts), axis=1), axis=1
    ) / (1.0 / 12.0)


def _round_trip_error(
    spot_rates: np.ndarray, discount_factors: np.ndarray, monthly_forwards: np.ndarray
) -> float:
    monthly_step = 1.0 / 12.0
    rebuilt_discounts = np.exp(-np.cumsum(monthly_forwards * monthly_step, axis=1))
    rebuilt_spots = -np.log(rebuilt_discounts) / _MONTHLY_TENORS_YEARS
    return float(
        max(
            np.max(np.abs(discount_factors - rebuilt_discounts)),
            np.max(np.abs(spot_rates - rebuilt_spots)),
        )
    )


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
