from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from deepalm.term_structures import MarketScenarioModel, TermStructureError

PARAMETERS = ("b0", "b1", "b2", "b3", "t1", "t2")


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


def test_market_scenario_model_rejects_partial_parameter_dates(tmp_path: Path) -> None:
    source = tmp_path / "partial.csv"
    partial = flat_curve_parameters(2.0)
    partial["b3"] = None
    write_snb_export(
        source, {"2005-01-03": partial, "2022-07-15": flat_curve_parameters(3.0)}
    )

    with pytest.raises(TermStructureError, match="partial parameters.*2005-01-03"):
        MarketScenarioModel().load_historical_term_structures(source)


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
