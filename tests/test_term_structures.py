from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from deepalm.term_structures import (
    HullWhiteConfiguration,
    MarketScenarioModel,
    TermStructureError,
)

PARAMETERS = ("b0", "b1", "b2", "b3", "t1", "t2")
SNB_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
)


def write_snb_export(
    path: Path,
    observations: dict[str, dict[str, float | None]],
    *,
    cube_id: str = "rendopar",
) -> None:
    rows = [
        f'"CubeId";"{cube_id}"',
        '"PublishingDate";"2025-09-01 14:30"',
        "",
        '"Date";"d0";"Value"',
    ]
    for date, values in observations.items():
        for parameter in PARAMETERS:
            value = values.get(parameter)
            encoded_value = "" if value is None else str(value)
            rows.append(f'"{date}";"{parameter}";"{encoded_value}"')
    path.write_text("\n".join(rows) + "\n", encoding="utf-8-sig")


def flat_curve_parameters(rate_in_percentage_points: float) -> dict[str, float]:
    return {
        "b0": rate_in_percentage_points,
        "b1": 0.0,
        "b2": 0.0,
        "b3": 0.0,
        "t1": 1.0,
        "t2": 2.0,
    }


def test_market_scenario_model_reconstructs_calibration_and_initial_curves(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rendopar.csv"
    observations = {
        "2004-12-31": flat_curve_parameters(1.0),
        "2005-01-03": flat_curve_parameters(2.0),
        "2005-01-04": {parameter: None for parameter in PARAMETERS},
        "2022-07-15": flat_curve_parameters(3.0),
        "2022-07-18": flat_curve_parameters(4.0),
    }
    write_snb_export(source, observations)

    curves = MarketScenarioModel().load_historical_term_structures(source)

    assert curves.dates.astype("datetime64[D]").tolist() == [
        np.datetime64("2005-01-03"),
        np.datetime64("2022-07-15"),
    ]
    assert curves.tenors_years.shape == (180,)
    assert curves.spot_rates.shape == (2, 180)
    assert curves.discount_factors.shape == (2, 180)
    assert curves.monthly_forwards.shape == (2, 180)
    assert np.allclose(curves.spot_rates[0], 0.02, atol=1e-12)
    assert np.allclose(curves.monthly_forwards[0], 0.02, atol=1e-12)
    assert curves.discount_factors[0, 0] == pytest.approx(np.exp(-0.02 / 12))
    assert curves.initial_curve.as_of_date == np.datetime64("2022-07-15")
    assert np.allclose(curves.initial_curve.spot_rates, 0.03, atol=1e-12)
    assert curves.round_trip_error < 1e-10
    assert curves.source_hash == hashlib.sha256(source.read_bytes()).hexdigest()
    assert curves.source_beta_unit == "percentage_points"
    assert curves.rate_unit == "decimal"
    assert np.array_equal(
        curves.initial_nss_parameters,
        np.array([0.03, 0.0, 0.0, 0.0, 1.0, 2.0]),
    )
    assert curves.deposit_reference_history.dates.astype("datetime64[D]").tolist() == [
        np.datetime64("2004-12-31"),
        np.datetime64("2005-01-03"),
        np.datetime64("2022-07-15"),
    ]
    assert np.array_equal(
        curves.deposit_reference_history.tenors_years, curves.tenors_years
    )


def test_market_scenario_model_rejects_partial_parameter_dates(tmp_path: Path) -> None:
    source = tmp_path / "partial.csv"
    partial = flat_curve_parameters(2.0)
    partial["b3"] = None
    write_snb_export(
        source, {"2005-01-03": partial, "2022-07-15": flat_curve_parameters(3.0)}
    )

    with pytest.raises(TermStructureError, match="partial parameters.*2005-01-03"):
        MarketScenarioModel().load_historical_term_structures(source)


def test_deposit_initial_history_uses_calendar_month_end_and_prior_observation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "deposit-history.csv"
    write_snb_export(
        source,
        {
            "2005-01-03": flat_curve_parameters(1.0),
            "2022-01-31": flat_curve_parameters(1.0),
            "2022-02-25": flat_curve_parameters(2.0),
            "2022-07-15": flat_curve_parameters(3.0),
        },
    )
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(source)

    history = model.deposit_initial_history(historical, as_of_date="2022-03-31")

    assert history.target_dates == ("2022-02-28", "2022-01-31")
    assert history.observation_dates == ("2022-02-25", "2022-01-31")
    assert history.six_month_yields == pytest.approx((0.02, 0.01))


def test_deposit_initial_history_rejects_missing_pre_valuation_observation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "missing-deposit-history.csv"
    write_snb_export(
        source,
        {
            "2005-01-03": flat_curve_parameters(1.0),
            "2022-07-15": flat_curve_parameters(3.0),
        },
    )
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(source)

    with pytest.raises(TermStructureError, match="missing a completed curve"):
        model.deposit_initial_history(historical, as_of_date="2005-02-01")


def test_market_scenario_model_rejects_duplicate_parameter_keys(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.csv"
    values = flat_curve_parameters(2.0)
    write_snb_export(
        source, {"2005-01-03": values, "2022-07-15": flat_curve_parameters(3.0)}
    )
    with source.open("a", encoding="utf-8") as output:
        output.write('"2005-01-03";"b0";"2.0"\n')

    with pytest.raises(TermStructureError, match="duplicate date/parameter"):
        MarketScenarioModel().load_historical_term_structures(source)


def test_market_scenario_model_rejects_an_unrecognized_snb_cube(tmp_path: Path) -> None:
    source = tmp_path / "wrong-cube.csv"
    write_snb_export(
        source,
        {
            "2005-01-03": flat_curve_parameters(2.0),
            "2022-07-15": flat_curve_parameters(3.0),
        },
        cube_id="other",
    )

    with pytest.raises(TermStructureError, match="rendopar"):
        MarketScenarioModel().load_historical_term_structures(source)


def test_market_scenario_model_reconstructs_nss_shape_and_diagnostic_forward(
    tmp_path: Path,
) -> None:
    source = tmp_path / "shaped.csv"
    shaped_curve = {
        "b0": 2.0,
        "b1": -1.0,
        "b2": 0.5,
        "b3": 0.25,
        "t1": 1.0,
        "t2": 3.0,
    }
    write_snb_export(source, {"2005-01-03": shaped_curve, "2022-07-15": shaped_curve})

    curves = MarketScenarioModel().load_historical_term_structures(source)

    tenor = 1.0 / 12.0
    first_loading = (1.0 - np.exp(-tenor)) / tenor
    expected_spot = (
        0.02
        - 0.01 * first_loading
        + 0.005 * (first_loading - np.exp(-tenor))
        + 0.0025 * ((1.0 - np.exp(-tenor / 3.0)) / (tenor / 3.0) - np.exp(-tenor / 3.0))
    )
    expected_instantaneous_forward = (
        0.02
        - 0.01 * np.exp(-tenor)
        + 0.005 * tenor * np.exp(-tenor)
        + 0.0025 * (tenor / 3.0) * np.exp(-tenor / 3.0)
    )
    assert curves.spot_rates[0, 0] == pytest.approx(expected_spot)
    assert curves.instantaneous_forwards[0, 0] == pytest.approx(
        expected_instantaneous_forward
    )


def test_market_scenario_model_rejects_invalid_unit_and_date(tmp_path: Path) -> None:
    source = tmp_path / "invalid.csv"
    write_snb_export(
        source,
        {
            "2005-01-03": flat_curve_parameters(2.0),
            "2022-07-15": flat_curve_parameters(3.0),
        },
    )

    with pytest.raises(TermStructureError, match="percentage_points"):
        MarketScenarioModel().load_historical_term_structures(
            source, beta_unit="decimal"
        )

    invalid_date = source.read_text(encoding="utf-8-sig").replace(
        "2005-01-03", "not-a-date", 1
    )
    source.write_text(invalid_date, encoding="utf-8-sig")
    with pytest.raises(TermStructureError, match="invalid dates"):
        MarketScenarioModel().load_historical_term_structures(source)


def test_market_scenario_model_rejects_non_finite_reconstructed_curves(
    tmp_path: Path,
) -> None:
    source = tmp_path / "overflow.csv"
    write_snb_export(
        source,
        {
            "2005-01-03": flat_curve_parameters(-100_000.0),
            "2022-07-15": flat_curve_parameters(3.0),
        },
    )

    with pytest.raises(TermStructureError, match="discount factors"):
        MarketScenarioModel().load_historical_term_structures(source)


def test_market_scenario_model_rejects_nonblank_invalid_parameter_values(
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid-values.csv"
    write_snb_export(
        source,
        {
            "2005-01-03": {parameter: "not-a-number" for parameter in PARAMETERS},
            "2022-07-15": flat_curve_parameters(3.0),
        },
    )

    with pytest.raises(TermStructureError, match="invalid parameter values"):
        MarketScenarioModel().load_historical_term_structures(source)


def test_market_scenario_model_calibrates_and_generates_single_semantic_hjm_paths() -> None:
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SNB_SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    five_year = model.generate_hjm_scenarios(
        historical,
        calibration,
        horizon_years=5,
        paths=2,
        seed=73,
    )
    fifteen_year = model.generate_hjm_scenarios(
        historical,
        calibration,
        horizon_years=15,
        paths=2,
        seed=73,
    )

    assert calibration.weekly_dates.shape[0] == calibration.weekly_forwards.shape[0]
    assert np.all(np.diff(calibration.weekly_dates) > np.timedelta64(0, "D"))
    assert np.allclose(
        calibration.eigenvectors.T @ calibration.eigenvectors, np.eye(180)
    )
    assert calibration.explained_variance.sum() >= 0.9
    assert np.allclose(
        calibration.scaled_loadings,
        calibration.eigenvectors[:, :3]
        * np.sqrt(np.clip(calibration.eigenvalues[:3], 0.0, None)),
    )
    assert calibration.fitted_loadings.shape == (180, 3)
    assert five_year.spot_rates.shape == (2, 61, 180)
    assert fifteen_year.spot_rates.shape == (2, 181, 180)
    assert five_year.round_trip_error <= 1e-10
    assert np.array_equal(five_year.innovations, fifteen_year.innovations[:, :60])
    assert np.allclose(
        five_year.monthly_forwards,
        fifteen_year.monthly_forwards[:, :61],
        rtol=0.0,
        atol=1e-15,
    )
    assert np.array_equal(
        five_year.spot_rates,
        model.generate_hjm_scenarios(
            historical,
            calibration,
            horizon_years=5,
            paths=2,
            seed=73,
        ).spot_rates,
    )
    diagnostics = model.validate_hjm_one_step(calibration, paths=50_000, seed=91)
    assert np.all(np.abs(diagnostics.factor_means) <= 3.0 / np.sqrt(50_000))
    assert diagnostics.relative_covariance_error <= 0.05

    invalid_calibration = replace(
        calibration,
        cubic_coefficients=np.full((4, 3), np.inf),
    )
    with pytest.raises(TermStructureError, match="non-finite"):
        model.generate_hjm_scenarios(
            historical,
            invalid_calibration,
            horizon_years=5,
            paths=1,
            seed=73,
        )


def test_hjm_paths_are_identical_across_batch_partitions() -> None:
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SNB_SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    combined = model.generate_hjm_scenarios(
        historical,
        calibration,
        horizon_years=5,
        paths=4,
        seed=125,
        split="training",
        epoch=2,
        global_path_indices=(20, 21, 22, 23),
    )
    first = model.generate_hjm_scenarios(
        historical,
        calibration,
        horizon_years=5,
        paths=2,
        seed=125,
        split="training",
        epoch=2,
        global_path_indices=(20, 21),
    )
    second = model.generate_hjm_scenarios(
        historical,
        calibration,
        horizon_years=5,
        paths=2,
        seed=125,
        split="training",
        epoch=2,
        global_path_indices=(22, 23),
    )

    assert np.array_equal(combined.innovations[:2], first.innovations)
    assert np.array_equal(combined.innovations[2:], second.innovations)
    assert np.array_equal(combined.spot_rates[:2], first.spot_rates)
    assert np.array_equal(combined.spot_rates[2:], second.spot_rates)


def test_market_scenario_model_compares_hull_white_terminal_diversity(
    tmp_path: Path,
) -> None:
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SNB_SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    hjm = model.generate_hjm_scenarios(
        historical,
        calibration,
        horizon_years=5,
        paths=3,
        seed=103,
    )
    hull_white = model.generate_hull_white_scenarios(
        historical,
        horizon_years=5,
        paths=3,
        seed=103,
        configuration=HullWhiteConfiguration(mean_reversion=0.2, volatility=0.01),
    )
    diversity = model.summarize_terminal_curve_diversity(hjm, hull_white)
    plot_path, metadata_path = model.write_terminal_curve_diversity_artifacts(
        tmp_path, diversity, hjm, hull_white
    )

    assert np.allclose(
        hull_white.spot_rates[:, 0], historical.initial_curve.spot_rates, atol=1e-12
    )
    assert hull_white.spot_rates.shape == (3, 61, 180)
    assert hull_white.round_trip_error <= 1e-10
    assert diversity.sample_size == 3
    assert diversity.units == "decimal annual rates"
    assert plot_path.is_file()
    assert json.loads(metadata_path.read_text()) == {
        "horizon_years": 5,
        "hull_white_calibration_identity": hull_white.calibration_identity,
        "models": ["hjm-pca", "project-hull-white"],
        "sample_size": 3,
        "seed": 103,
        "units": "decimal annual rates",
    }
