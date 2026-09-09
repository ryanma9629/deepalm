"""Historical SNB NSS term structures for Deep ALM market calibration."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from deepalm.runner import OperationalRunError
from deepalm.scenario_identity import ScenarioIdentity

_PARAMETERS = ("b0", "b1", "b2", "b3", "t1", "t2")
_CALIBRATION_START = pd.Timestamp("2005-01-01")
_CALIBRATION_END = pd.Timestamp("2022-07-15")
_INITIAL_CURVE_DATE = pd.Timestamp("2022-07-15")
_MONTHLY_TENORS_YEARS = np.arange(1, 181, dtype=np.float64) / 12.0
_BETA_UNIT = "percentage_points"


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
    instantaneous_forwards: np.ndarray


@dataclass(frozen=True)
class TermStructureHistory:
    """A dated curve history used for one downstream purpose."""

    dates: np.ndarray
    tenors_years: np.ndarray
    spot_rates: np.ndarray
    discount_factors: np.ndarray
    monthly_forwards: np.ndarray
    instantaneous_forwards: np.ndarray


@dataclass(frozen=True)
class DepositInitialHistory:
    """The two dated yields required before the simulated deposit path begins."""

    target_dates: tuple[str, str]
    observation_dates: tuple[str, str]
    six_month_yields: tuple[float, float]
    available_observation_dates: tuple[str, ...]
    available_six_month_yields: tuple[float, ...]
    source_identity: str
    identity: str


@dataclass(frozen=True)
class HistoricalTermStructures:
    """Validated calibration curves and the paper-window initial curve."""

    dates: np.ndarray
    tenors_years: np.ndarray
    spot_rates: np.ndarray
    discount_factors: np.ndarray
    monthly_forwards: np.ndarray
    instantaneous_forwards: np.ndarray
    deposit_reference_history: TermStructureHistory
    initial_curve: TermStructureSnapshot
    initial_nss_parameters: np.ndarray
    source_path: Path
    source_hash: str
    source_beta_unit: str
    rate_unit: str
    round_trip_error: float


@dataclass(frozen=True)
class HjmPcaCalibration:
    """Auditable weekly-PCA calibration for the monthly HJM model."""

    weekly_dates: np.ndarray
    weekly_forwards: np.ndarray
    forward_differences: np.ndarray
    covariance: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    explained_variance: np.ndarray
    scaled_loadings: np.ndarray
    cubic_coefficients: np.ndarray
    fitted_loadings: np.ndarray
    implied_covariances: np.ndarray
    pca_truncation_error: float
    polynomial_fit_errors: np.ndarray
    calibration_identity: str


@dataclass(frozen=True)
class MarketScenarioBatch:
    """Observable monthly market scenarios on the 180-node ALM grid."""

    spot_rates: np.ndarray
    discount_factors: np.ndarray
    monthly_forwards: np.ndarray
    innovations: np.ndarray
    horizon_years: int
    seed: int
    calibration_identity: str
    round_trip_error: float
    initial_curve_identity: str
    as_of_date: str
    split: str = "default"
    epoch: int = 0
    global_path_indices: tuple[int, ...] = ()
    deposit_initial_history_identity: str | None = None

    def prefix(self, *, horizon_years: int) -> MarketScenarioBatch:
        """Return an identity-preserving prefix of a longer HJM path batch.

        HJM innovations are generated path-by-path, so the first sixty shocks
        of a fifteen-year batch are exactly the paired five-year shocks when
        seed, split, epoch, and global path indices agree.  The returned batch
        deliberately advertises the shorter horizon for the ALM terminal roll,
        while retaining the original provenance fields.
        """

        if horizon_years not in {5, 15} or horizon_years > self.horizon_years:
            raise TermStructureError(
                "Market prefix horizon must be an enabled shorter horizon"
            )
        steps = 12 * horizon_years
        spot_rates = _readonly(self.spot_rates[:, : steps + 1].copy())
        discount_factors = _readonly(self.discount_factors[:, : steps + 1].copy())
        monthly_forwards = _readonly(self.monthly_forwards[:, : steps + 1].copy())
        innovations = _readonly(self.innovations[:, :steps].copy())
        return MarketScenarioBatch(
            spot_rates=spot_rates,
            discount_factors=discount_factors,
            monthly_forwards=monthly_forwards,
            innovations=innovations,
            horizon_years=horizon_years,
            seed=self.seed,
            calibration_identity=self.calibration_identity,
            round_trip_error=_scenario_round_trip_error(
                spot_rates, discount_factors, monthly_forwards
            ),
            split=self.split,
            epoch=self.epoch,
            global_path_indices=self.global_path_indices,
            initial_curve_identity=self.initial_curve_identity,
            as_of_date=self.as_of_date,
            deposit_initial_history_identity=self.deposit_initial_history_identity,
        )


@dataclass(frozen=True)
class HjmOneStepDiagnostics:
    """Fixed-seed statistical evidence for the HJM one-step shock."""

    factor_means: np.ndarray
    shock_covariance: np.ndarray
    target_covariance: np.ndarray
    relative_covariance_error: float
    paths: int
    seed: int


@dataclass(frozen=True)
class HullWhiteConfiguration:
    """Explicit, non-paper assumptions for the diagnostic-only comparator."""

    mean_reversion: float
    volatility: float
    label: str = "project_choice"


@dataclass(frozen=True)
class TerminalCurveDiversity:
    """Metadata and dispersion summaries for terminal-curve comparison."""

    horizon_years: int
    seed: int
    sample_size: int
    units: str
    hjm_terminal_standard_deviation: np.ndarray
    hull_white_terminal_standard_deviation: np.ndarray


class MarketScenarioModel:
    """Load the historical term structures used by the later HJM scenario model."""

    def load_historical_term_structures(
        self, source_path: Path, *, beta_unit: str = _BETA_UNIT
    ) -> HistoricalTermStructures:
        """Parse the SNB NSS export and reconstruct its calibration-window curves."""

        if beta_unit != _BETA_UNIT:
            raise TermStructureError(
                "SNB NSS beta unit must be declared as percentage_points",
                diagnostics={"beta_unit": beta_unit},
            )
        parameter_table = _load_parameter_table(source_path, beta_unit=beta_unit)
        calibration_table = _select_calibration_window(parameter_table)
        spot_rates, discount_factors, monthly_forwards, instantaneous_forwards = (
            _reconstruct_curve_representations(calibration_table)
        )
        round_trip_error = _round_trip_error(
            spot_rates, discount_factors, monthly_forwards
        )
        if round_trip_error > 1e-10:
            raise TermStructureError(
                "Spot, discount, and forward curves do not round-trip within 1e-10",
                diagnostics={"round_trip_error": round_trip_error},
            )

        deposit_reference_table = parameter_table.loc[:_CALIBRATION_END]
        (
            deposit_spots,
            deposit_discounts,
            deposit_forwards,
            deposit_instantaneous_forwards,
        ) = _reconstruct_curve_representations(deposit_reference_table)
        deposit_reference_history = TermStructureHistory(
            dates=_readonly(
                deposit_reference_table.index.to_numpy(dtype="datetime64[ns]")
            ),
            tenors_years=_readonly(_MONTHLY_TENORS_YEARS),
            spot_rates=_readonly(deposit_spots),
            discount_factors=_readonly(deposit_discounts),
            monthly_forwards=_readonly(deposit_forwards),
            instantaneous_forwards=_readonly(deposit_instantaneous_forwards),
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
            instantaneous_forwards=_readonly(instantaneous_forwards[index]),
        )
        return HistoricalTermStructures(
            dates=_readonly(dates),
            tenors_years=_readonly(_MONTHLY_TENORS_YEARS),
            spot_rates=_readonly(spot_rates),
            discount_factors=_readonly(discount_factors),
            monthly_forwards=_readonly(monthly_forwards),
            instantaneous_forwards=_readonly(instantaneous_forwards),
            deposit_reference_history=deposit_reference_history,
            initial_curve=initial_curve,
            initial_nss_parameters=_readonly(
                calibration_table.iloc[index].to_numpy(dtype=np.float64)
            ),
            source_path=source_path,
            source_hash=_sha256(source_path),
            source_beta_unit=beta_unit,
            rate_unit="decimal",
            round_trip_error=round_trip_error,
        )

    def deposit_initial_history(
        self,
        historical: HistoricalTermStructures,
        *,
        as_of_date: str,
    ) -> DepositInitialHistory:
        """Select the two completed calendar-month curves before ``as_of_date``."""

        try:
            valuation_date = pd.Timestamp(as_of_date)
        except (TypeError, ValueError) as error:
            raise TermStructureError(
                "Deposit history requires a valid valuation date"
            ) from error
        targets = _deposit_initial_history_targets(valuation_date)
        dates = historical.deposit_reference_history.dates.astype("datetime64[D]")
        observed: list[np.datetime64] = []
        yields: list[float] = []
        for target in targets:
            target_day = target.to_datetime64().astype("datetime64[D]")
            index = int(np.searchsorted(dates, target_day, side="right") - 1)
            if index < 0:
                raise TermStructureError(
                    "Deposit history is missing a completed curve on or before "
                    f"{target_day}"
                )
            observed.append(dates[index])
            yields.append(
                float(historical.deposit_reference_history.spot_rates[index, 5])
            )
        target_dates = tuple(str(target.date()) for target in targets)
        observation_dates = tuple(str(date) for date in observed)
        six_month_yields = tuple(yields)
        available_observation_dates = tuple(
            str(date.astype("datetime64[D]"))
            for date in historical.deposit_reference_history.dates
        )
        available_six_month_yields = tuple(
            float(value)
            for value in historical.deposit_reference_history.spot_rates[:, 5]
        )
        identity = deposit_initial_history_identity(
            target_dates,
            observation_dates,
            six_month_yields,
            available_observation_dates,
            available_six_month_yields,
            historical.source_hash,
        )
        return DepositInitialHistory(
            target_dates=target_dates,
            observation_dates=observation_dates,
            six_month_yields=six_month_yields,
            available_observation_dates=available_observation_dates,
            available_six_month_yields=available_six_month_yields,
            source_identity=historical.source_hash,
            identity=identity,
        )

    def _initial_deposit_history_identity(
        self, historical: HistoricalTermStructures
    ) -> str:
        return self.deposit_initial_history(
            historical,
            as_of_date=str(historical.initial_curve.as_of_date.astype("datetime64[D]")),
        ).identity

    def calibrate_hjm_pca(
        self, historical: HistoricalTermStructures
    ) -> HjmPcaCalibration:
        """Calibrate three signed weekly PCA factors and their cubic loadings."""

        weekly_dates, weekly_forwards = _friday_ending_weekly_observations(
            historical.dates, historical.monthly_forwards
        )
        differences = np.diff(weekly_forwards, axis=0)
        if len(differences) < 4:
            raise TermStructureError("At least five weekly observations are required")
        covariance = np.cov(differences, rowvar=False, ddof=1) * 52.0
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        for column in range(eigenvectors.shape[1]):
            pivot = np.argmax(np.abs(eigenvectors[:, column]))
            if eigenvectors[pivot, column] < 0:
                eigenvectors[:, column] *= -1.0
        retained_values = eigenvalues[:3]
        retained_vectors = eigenvectors[:, :3]
        explained_variance = retained_values / eigenvalues.sum()
        design = np.vander(historical.tenors_years / 15.0, N=4, increasing=True)
        loadings = retained_vectors * np.sqrt(np.clip(retained_values, 0.0, None))
        cubic_coefficients, _, _, _ = np.linalg.lstsq(design, loadings, rcond=None)
        fitted_loadings = design @ cubic_coefficients
        polynomial_fit_errors = np.linalg.norm(
            fitted_loadings - loadings, axis=0
        ) / np.linalg.norm(loadings, axis=0)
        implied_covariances = fitted_loadings @ fitted_loadings.T
        retained_covariance = (
            retained_vectors @ np.diag(retained_values) @ retained_vectors.T
        )
        pca_truncation_error = float(
            np.linalg.norm(covariance - retained_covariance)
            / np.linalg.norm(covariance)
        )
        if explained_variance.sum() < 0.9:
            raise TermStructureError(
                "Three PCA components explain less than 90% of variance"
            )
        identity = _calibration_identity(weekly_dates, covariance, eigenvalues)
        return HjmPcaCalibration(
            weekly_dates=_readonly(weekly_dates),
            weekly_forwards=_readonly(weekly_forwards),
            forward_differences=_readonly(differences),
            covariance=_readonly(covariance),
            eigenvalues=_readonly(eigenvalues),
            eigenvectors=_readonly(eigenvectors),
            explained_variance=_readonly(explained_variance),
            scaled_loadings=_readonly(loadings),
            cubic_coefficients=_readonly(cubic_coefficients),
            fitted_loadings=_readonly(fitted_loadings),
            implied_covariances=_readonly(implied_covariances),
            pca_truncation_error=pca_truncation_error,
            polynomial_fit_errors=_readonly(polynomial_fit_errors),
            calibration_identity=identity,
        )

    def generate_hjm_scenarios(
        self,
        historical: HistoricalTermStructures,
        calibration: HjmPcaCalibration,
        *,
        horizon_years: int,
        paths: int,
        seed: int,
        split: str = "default",
        epoch: int = 0,
        global_path_indices: tuple[int, ...] | None = None,
    ) -> MarketScenarioBatch:
        """Generate bitwise-reproducible, unclipped monthly HJM scenarios."""

        if horizon_years not in {5, 15}:
            raise TermStructureError("HJM horizon must be 5 or 15 years")
        if paths <= 0:
            raise TermStructureError("HJM paths must be positive")
        if not split:
            raise TermStructureError("HJM scenario split must be non-empty")
        if epoch < 0:
            raise TermStructureError("HJM scenario epoch must be non-negative")
        indices = global_path_indices or tuple(range(paths))
        if (
            len(indices) != paths
            or len(set(indices)) != paths
            or any(index < 0 for index in indices)
        ):
            raise TermStructureError(
                "HJM global path indices must be unique non-negative values matching paths"
            )
        steps = horizon_years * 12
        extended_tenors = np.arange(1, 181 + steps, dtype=np.float64) / 12.0
        initial_forwards = _nss_monthly_forwards(
            historical.initial_nss_parameters, extended_tenors
        )
        coefficients = calibration.cubic_coefficients
        volatility = _evaluate_cubics(coefficients, extended_tenors / 15.0)
        drift = _hjm_drift(
            volatility, _evaluate_cubics(coefficients, np.array([0.0]))[0]
        )
        _require_finite("initial HJM values", initial_forwards, volatility, drift)
        innovations = np.stack(
            [
                np.random.default_rng(
                    ScenarioIdentity(
                        split=split,
                        epoch=epoch,
                        seed=seed,
                        global_path_index=index,
                    ).derived_seed()
                ).standard_normal((steps, 3), dtype=np.float64)
                for index in indices
            ]
        )
        forwards = np.empty((paths, steps + 1, 180), dtype=np.float64)
        forwards[:, 0] = initial_forwards[:180]
        curve = np.broadcast_to(initial_forwards, (paths, len(initial_forwards))).copy()
        dt = 1.0 / 12.0
        for step in range(steps):
            length = curve.shape[1]
            shock = innovations[:, step] @ volatility[: length - 1].T * np.sqrt(dt)
            curve = (
                curve[:, :-1]
                + (curve[:, 1:] - curve[:, :-1])
                + drift[: length - 1] * dt
                + shock
            )
            _require_finite("HJM forward path", curve)
            forwards[:, step + 1] = curve[:, :180]
        discount_factors, spot_rates = _curve_representations_from_forwards(forwards)
        _require_finite("HJM output curves", discount_factors, spot_rates)
        round_trip_error = _scenario_round_trip_error(
            spot_rates, discount_factors, forwards
        )
        return MarketScenarioBatch(
            spot_rates=_readonly(spot_rates),
            discount_factors=_readonly(discount_factors),
            monthly_forwards=_readonly(forwards),
            innovations=_readonly(innovations),
            horizon_years=horizon_years,
            seed=seed,
            calibration_identity=calibration.calibration_identity,
            round_trip_error=round_trip_error,
            split=split,
            epoch=epoch,
            global_path_indices=indices,
            initial_curve_identity=historical.source_hash,
            as_of_date=str(historical.initial_curve.as_of_date.astype("datetime64[D]")),
            deposit_initial_history_identity=self._initial_deposit_history_identity(
                historical
            ),
        )

    def validate_hjm_one_step(
        self,
        calibration: HjmPcaCalibration,
        *,
        paths: int = 50_000,
        seed: int,
    ) -> HjmOneStepDiagnostics:
        """Measure one monthly HJM shock against its exact fitted-covariance target."""

        if paths <= 1:
            raise TermStructureError("At least two one-step paths are required")
        volatility = calibration.fitted_loadings
        factors = np.random.default_rng(seed).standard_normal((paths, 3))
        shocks = factors @ volatility.T * np.sqrt(1.0 / 12.0)
        shock_covariance = np.cov(shocks, rowvar=False, ddof=1)
        target_covariance = volatility @ volatility.T / 12.0
        relative_error = float(
            np.linalg.norm(shock_covariance - target_covariance)
            / np.linalg.norm(target_covariance)
        )
        if not np.isfinite(relative_error):
            raise TermStructureError("HJM one-step diagnostics are non-finite")
        return HjmOneStepDiagnostics(
            factor_means=_readonly(factors.mean(axis=0)),
            shock_covariance=_readonly(shock_covariance),
            target_covariance=_readonly(target_covariance),
            relative_covariance_error=relative_error,
            paths=paths,
            seed=seed,
        )

    def generate_hull_white_scenarios(
        self,
        historical: HistoricalTermStructures,
        *,
        horizon_years: int,
        paths: int,
        seed: int,
        configuration: HullWhiteConfiguration,
    ) -> MarketScenarioBatch:
        """Generate diagnostic-only extended Vasicek curves from the canonical curve."""

        if horizon_years not in {5, 15}:
            raise TermStructureError("Hull-White horizon must be 5 or 15 years")
        if (
            paths <= 0
            or configuration.mean_reversion <= 0
            or configuration.volatility < 0
        ):
            raise TermStructureError("Hull-White paths and parameters must be positive")
        steps = horizon_years * 12
        dt = 1.0 / 12.0
        a = configuration.mean_reversion
        sigma = configuration.volatility
        innovation = np.random.default_rng(seed).standard_normal((paths, 180, 1))[
            :, :steps
        ]
        factor = np.zeros((paths, steps + 1), dtype=np.float64)
        standard_deviation = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a))
        for step in range(steps):
            factor[:, step + 1] = (
                np.exp(-a * dt) * factor[:, step]
                + standard_deviation * innovation[:, step, 0]
            )
        tenors = historical.tenors_years
        b_over_t = -np.expm1(-a * tenors) / (a * tenors)
        spot_rates = historical.initial_curve.spot_rates[None, None, :] + (
            factor[:, :, None] * b_over_t[None, None, :]
        )
        discount_factors = np.exp(-spot_rates * tenors[None, None, :])
        monthly_forwards = (
            -np.diff(
                np.concatenate(
                    (
                        np.zeros((paths, steps + 1, 1), dtype=np.float64),
                        np.log(discount_factors),
                    ),
                    axis=2,
                ),
                axis=2,
            )
            / dt
        )
        _require_finite(
            "Hull-White output curves", spot_rates, discount_factors, monthly_forwards
        )
        return MarketScenarioBatch(
            spot_rates=_readonly(spot_rates),
            discount_factors=_readonly(discount_factors),
            monthly_forwards=_readonly(monthly_forwards),
            innovations=_readonly(innovation),
            horizon_years=horizon_years,
            seed=seed,
            calibration_identity=_hull_white_identity(configuration),
            round_trip_error=_scenario_round_trip_error(
                spot_rates, discount_factors, monthly_forwards
            ),
            initial_curve_identity=historical.source_hash,
            as_of_date=str(historical.initial_curve.as_of_date.astype("datetime64[D]")),
            deposit_initial_history_identity=self._initial_deposit_history_identity(
                historical
            ),
        )

    def summarize_terminal_curve_diversity(
        self, hjm: MarketScenarioBatch, hull_white: MarketScenarioBatch
    ) -> TerminalCurveDiversity:
        """Describe terminal curve spread without treating Hull-White as a policy model."""

        if hjm.horizon_years != hull_white.horizon_years or len(hjm.spot_rates) != len(
            hull_white.spot_rates
        ):
            raise TermStructureError(
                "Terminal curve batches must share horizon and sample size"
            )
        return TerminalCurveDiversity(
            horizon_years=hjm.horizon_years,
            seed=hjm.seed,
            sample_size=len(hjm.spot_rates),
            units="decimal annual rates",
            hjm_terminal_standard_deviation=_readonly(
                hjm.spot_rates[:, -1].std(axis=0)
            ),
            hull_white_terminal_standard_deviation=_readonly(
                hull_white.spot_rates[:, -1].std(axis=0)
            ),
        )

    def write_terminal_curve_diversity_artifacts(
        self,
        directory: Path,
        diversity: TerminalCurveDiversity,
        hjm: MarketScenarioBatch,
        hull_white: MarketScenarioBatch,
    ) -> tuple[Path, Path]:
        """Write labeled terminal-curve diversity plot and machine-readable metadata."""

        import matplotlib.pyplot as plt

        directory.mkdir(parents=True, exist_ok=True)
        plot_path = directory / "terminal-curve-diversity.png"
        metadata_path = directory / "terminal-curve-diversity.json"
        tenors = np.arange(1, 181, dtype=np.float64) / 12.0
        figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for axis, batch, title in (
            (axes[0], hjm, "HJM-PCA"),
            (axes[1], hull_white, "Hull-White extended Vasicek"),
        ):
            axis.plot(tenors, batch.spot_rates[:, -1].T, alpha=0.25)
            axis.set_title(title)
            axis.set_xlabel("Tenor (years)")
        axes[0].set_ylabel("Terminal spot rate (decimal annual)")
        figure.suptitle(
            f"Terminal curve diversity | {diversity.horizon_years}y | "
            f"n={diversity.sample_size} | seed={diversity.seed}"
        )
        figure.tight_layout()
        figure.savefig(plot_path, dpi=150)
        plt.close(figure)
        metadata_path.write_text(
            json.dumps(
                {
                    "models": ["hjm-pca", "project-hull-white"],
                    "horizon_years": diversity.horizon_years,
                    "units": diversity.units,
                    "seed": diversity.seed,
                    "sample_size": diversity.sample_size,
                    "hull_white_calibration_identity": hull_white.calibration_identity,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return plot_path, metadata_path


def _load_parameter_table(source_path: Path, *, beta_unit: str) -> pd.DataFrame:
    header_row = _validate_metadata_and_find_header(source_path)
    try:
        raw = pd.read_csv(
            source_path,
            sep=";",
            skiprows=header_row,
            encoding="utf-8-sig",
            dtype={"Date": "string", "d0": "string", "Value": "string"},
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

    value_text = raw["Value"].str.strip()
    blank_values = value_text.isna() | value_text.eq("")
    raw["Value"] = pd.to_numeric(value_text, errors="coerce")
    invalid_values = ~blank_values & raw["Value"].isna()
    if invalid_values.any():
        invalid_rows = raw.loc[invalid_values, ["Date", "d0"]].head(4)
        raise TermStructureError(
            "SNB source contains invalid parameter values",
            diagnostics={"invalid_values": invalid_rows.astype(str).to_dict("records")},
        )
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
    table.loc[:, ["b0", "b1", "b2", "b3"]] = table[["b0", "b1", "b2", "b3"]] / 100.0
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
    beta0, beta1, beta2, beta3 = (parameters[:, index] for index in range(4))
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


def _reconstruct_curve_representations(
    parameter_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spot_rates = _reconstruct_spot_rates(parameter_table)
    with np.errstate(over="ignore", invalid="ignore"):
        discount_factors = np.exp(-spot_rates * _MONTHLY_TENORS_YEARS)
    if not np.isfinite(discount_factors).all():
        raise TermStructureError(
            "NSS reconstruction produced non-finite discount factors"
        )
    monthly_forwards = _monthly_forwards(discount_factors)
    if not np.isfinite(monthly_forwards).all():
        raise TermStructureError(
            "NSS reconstruction produced non-finite monthly forwards"
        )
    instantaneous_forwards = _instantaneous_forwards(parameter_table)
    if not np.isfinite(instantaneous_forwards).all():
        raise TermStructureError(
            "NSS reconstruction produced non-finite instantaneous forwards"
        )
    return spot_rates, discount_factors, monthly_forwards, instantaneous_forwards


def _instantaneous_forwards(parameter_table: pd.DataFrame) -> np.ndarray:
    parameters = parameter_table.to_numpy(dtype=np.float64)
    beta0, beta1, beta2, beta3 = (parameters[:, index] for index in range(4))
    tau1, tau2 = parameters[:, 4], parameters[:, 5]
    x1 = _MONTHLY_TENORS_YEARS[None, :] / tau1[:, None]
    x2 = _MONTHLY_TENORS_YEARS[None, :] / tau2[:, None]
    return (
        beta0[:, None]
        + beta1[:, None] * np.exp(-x1)
        + beta2[:, None] * x1 * np.exp(-x1)
        + beta3[:, None] * x2 * np.exp(-x2)
    )


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


def _friday_ending_weekly_observations(
    dates: np.ndarray, forwards: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame(forwards, index=pd.DatetimeIndex(dates))
    selected = frame.resample("W-FRI").last().dropna(how="any")
    selected_dates = np.array(
        [frame.loc[:week_end].index[-1] for week_end in selected.index],
        dtype="datetime64[ns]",
    )
    return selected_dates, selected.to_numpy(dtype=np.float64)


def _nss_monthly_forwards(parameters: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    beta0, beta1, beta2, beta3, tau1, tau2 = parameters
    x1 = tenors / tau1
    x2 = tenors / tau2
    loading1 = -np.expm1(-x1) / x1
    loading2 = -np.expm1(-x2) / x2
    spots = (
        beta0
        + beta1 * loading1
        + beta2 * (loading1 - np.exp(-x1))
        + beta3 * (loading2 - np.exp(-x2))
    )
    discounts = np.exp(-spots * tenors)
    return -np.diff(np.concatenate(([0.0], np.log(discounts)))) / (1.0 / 12.0)


def _evaluate_cubics(
    coefficients: np.ndarray, normalized_tenors: np.ndarray
) -> np.ndarray:
    design = np.vander(normalized_tenors, N=4, increasing=True)
    return design @ coefficients


def _hjm_drift(volatility: np.ndarray, value_at_zero: np.ndarray) -> np.ndarray:
    step = 1.0 / 12.0
    integral = np.cumsum(
        (np.vstack((value_at_zero, volatility[:-1])) + volatility) * step / 2.0,
        axis=0,
    )
    return np.sum(volatility * integral, axis=1)


def _curve_representations_from_forwards(
    forwards: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    discounts = np.exp(-np.cumsum(forwards / 12.0, axis=2))
    tenors = _MONTHLY_TENORS_YEARS[None, None, :]
    spots = -np.log(discounts) / tenors
    return discounts, spots


def _scenario_round_trip_error(
    spot_rates: np.ndarray, discount_factors: np.ndarray, forwards: np.ndarray
) -> float:
    rebuilt_discounts = np.exp(-np.cumsum(forwards / 12.0, axis=2))
    rebuilt_spots = -np.log(rebuilt_discounts) / _MONTHLY_TENORS_YEARS[None, None, :]
    return float(
        max(
            np.max(np.abs(discount_factors - rebuilt_discounts)),
            np.max(np.abs(spot_rates - rebuilt_spots)),
        )
    )


def _require_finite(label: str, *values: np.ndarray) -> None:
    if not all(np.isfinite(value).all() for value in values):
        raise TermStructureError(f"{label} contains non-finite values")


def _calibration_identity(
    weekly_dates: np.ndarray, covariance: np.ndarray, eigenvalues: np.ndarray
) -> str:
    digest = hashlib.sha256()
    for value in (weekly_dates, covariance, eigenvalues):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def deposit_initial_history_target_dates(as_of_date: str) -> tuple[str, str]:
    """Return the prior one- and two-calendar-month target dates."""

    try:
        valuation_date = pd.Timestamp(as_of_date)
    except (TypeError, ValueError) as error:
        raise TermStructureError("Deposit history requires a valid valuation date") from error
    return tuple(str(target.date()) for target in _deposit_initial_history_targets(valuation_date))


def _deposit_initial_history_targets(valuation_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    return tuple(
        valuation_date - pd.DateOffset(months=offset) for offset in (1, 2)
    )


def deposit_initial_history_identity(
    target_dates: tuple[str, str],
    observation_dates: tuple[str, str],
    six_month_yields: tuple[float, float],
    available_observation_dates: tuple[str, ...],
    available_six_month_yields: tuple[float, ...],
    source_identity: str,
) -> str:
    payload = json.dumps(
        {
            "target_dates": target_dates,
            "observation_dates": observation_dates,
            "six_month_yields": six_month_yields,
            "available_observation_dates": available_observation_dates,
            "available_six_month_yields": available_six_month_yields,
            "source_identity": source_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _hull_white_identity(configuration: HullWhiteConfiguration) -> str:
    payload = (
        f"{configuration.label}:{configuration.mean_reversion:.17g}:"
        f"{configuration.volatility:.17g}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()
