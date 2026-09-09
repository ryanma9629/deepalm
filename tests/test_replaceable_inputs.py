from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deepalm.config import resolve_configuration
from deepalm.reference_bank import ReferenceBankError, ReferenceBankProvider
from deepalm.runner import ReproductionRunner, RunStatus
from deepalm.runoff import ALMSimulator
from deepalm.semantics import artifact_semantics
from deepalm.term_structures import (
    MarketScenarioModel,
    deposit_initial_history_identity,
    deposit_initial_history_target_dates,
)


def _imported_snapshot_data() -> dict[str, object]:
    targets = {
        "cash": 400.0,
        "investments": 200.0,
        "mortgages": 500.0,
        "enterprise_loans": 250.0,
        "non_maturity_deposits": 300.0,
        "term_deposits": 350.0,
        "funding": 400.0,
    }
    ladders = {
        name: [amount] + [0.0] * 179
        for name, amount in targets.items()
        if name != "cash"
    }
    deposit_reference_schedules = {
        name: {
            "terms_months": [1, 2, 12, 120],
            "weights": [1.0, 0.0, 0.0, 0.0],
            "provenance": "synthetic imported one-month deposit mapping",
            "cash_flows": [
                [amount] + [0.0] * 179,
                *([[0.0] * 180] * 3),
            ],
        }
        for name, amount in (
            ("non_maturity_deposits", targets["non_maturity_deposits"]),
            ("term_deposits", targets["term_deposits"]),
        )
    }
    history_targets = ("2023-12-31", "2023-11-30")
    history_observations = ("2023-12-29", "2023-11-30")
    history_yields = (0.02, 0.01)
    available_history_dates = ("2023-11-30", "2023-12-29")
    available_history_yields = (0.01, 0.02)
    history_source = "synthetic-usd-deposit-history"
    data: dict[str, object] = {
        "artifact_semantics": artifact_semantics("snapshot"),
        "schema_version": 5,
        "profile": "imported",
        "as_of_date": "2024-01-31",
        "initial_curve_identity": "synthetic-usd-2024-01-31",
        "cash": targets["cash"],
        "equity": 300.0,
        "ladders": ladders,
        "loan_cohorts": {
            name: [
                {
                    "principal_cash_flows": [amount] + [0.0] * 179,
                    "monthly_coupon_rate": 0.0,
                }
            ]
            for name, amount in (
                ("mortgages", targets["mortgages"]),
                ("enterprise_loans", targets["enterprise_loans"]),
            )
        },
        "deposit_reference_schedules": deposit_reference_schedules,
        "deposit_initial_history": {
            "target_dates": list(history_targets),
            "observation_dates": list(history_observations),
            "six_month_yields": list(history_yields),
            "available_observation_dates": list(available_history_dates),
            "available_six_month_yields": list(available_history_yields),
            "source_identity": history_source,
            "identity": deposit_initial_history_identity(
                history_targets,
                history_observations,
                history_yields,
                available_history_dates,
                available_history_yields,
                history_source,
            ),
        },
        "target_economic_values": targets,
        "target_value_errors": {name: 0.0 for name in ladders},
        "product_assumptions": {
            "loan_duration_years": 7.0,
            "deposit_duration_years": 4.0,
            "deposit_reference_terms_months": [1, 2, 12, 120],
            "non_maturity_weights": [1.0, 0.0, 0.0, 0.0],
            "term_deposit_weights": [1.0, 0.0, 0.0, 0.0],
            "loan_spread_decimal": 0.02,
            "mapping_version": "synthetic-usd-v1",
        },
        "provenance": {
            "source_system": "synthetic test fixture",
            "product_mapping": "six ALM ladder mapping",
            "assumptions": "declared noncanonical durations and spread",
            "valuation": "nominal one-month matched synthetic market",
            "market": "synthetic-usd-2024-01-31@2024-01-31",
        },
        "personnel_cost": 1.5,
        "material_cost": 0.7,
        "currency": "USD",
        "unit": "mUSD",
    }
    data["content_hash"] = _content_hash(data)
    return data


def _content_hash(data: dict[str, object]) -> str:
    hashed = dict(data)
    hashed.pop("content_hash", None)
    return hashlib.sha256(
        json.dumps(hashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_imported_snapshot_round_trips_and_rolls_on_its_matched_market(
    tmp_path: Path,
) -> None:
    provider = ReferenceBankProvider()
    source = tmp_path / "synthetic-usd-bank.json"
    source.write_text(json.dumps(_imported_snapshot_data()), encoding="utf-8")

    imported = provider.load(source)
    restored = tmp_path / "restored.json"
    provider.save(imported, restored)
    reloaded = provider.load(restored)
    market = SimpleNamespace(
        discount_factors=np.ones((1, 2, 180)),
        initial_curve_identity=imported.initial_curve_identity,
        as_of_date=imported.as_of_date,
    )
    outcome = ALMSimulator().rollout(imported, market)

    assert imported.profile == "imported"
    assert imported.currency == "USD"
    assert imported.unit == "mUSD"
    assert imported.total_assets == pytest.approx(1_350.0)
    assert imported.equity == pytest.approx(300.0)
    assert reloaded.content_hash == imported.content_hash
    assert outcome.cash[0, -1].item() == pytest.approx(300.0)
    assert outcome.accounting_error.abs().max().item() == pytest.approx(0.0)


def test_imported_snapshot_rejects_incomplete_or_incompatible_contracts(
    tmp_path: Path,
) -> None:
    provider = ReferenceBankProvider()
    data = _imported_snapshot_data()
    data["provenance"] = {"source_system": "incomplete"}
    data["content_hash"] = _content_hash(data)
    source = tmp_path / "incomplete.json"
    source.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ReferenceBankError, match="provenance"):
        provider.load(source)

    imported_data = _imported_snapshot_data()
    source.write_text(json.dumps(imported_data), encoding="utf-8")
    imported = provider.load(source)
    mismatch_market = SimpleNamespace(
        discount_factors=np.ones((1, 2, 180)),
        initial_curve_identity="another-curve",
        as_of_date=imported.as_of_date,
    )
    with pytest.raises(ReferenceBankError, match="curve identity"):
        ALMSimulator().rollout(imported, mismatch_market)

    with pytest.raises(ReferenceBankError, match="requires market curve identity"):
        ALMSimulator().rollout(
            imported, SimpleNamespace(discount_factors=np.ones((1, 2, 180)))
        )


def test_imported_snapshot_rejects_legacy_schema_without_complete_contract(
    tmp_path: Path,
) -> None:
    data = _imported_snapshot_data()
    data["schema_version"] = 2
    data.pop("loan_cohorts")
    data.pop("deposit_reference_schedules")
    data["content_hash"] = _content_hash(data)
    source = tmp_path / "legacy-v1.json"
    source.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ReferenceBankError, match="schema version"):
        ReferenceBankProvider().load(source)


def test_imported_snapshot_rejects_missing_dated_deposit_history(tmp_path: Path) -> None:
    data = _imported_snapshot_data()
    data.pop("deposit_initial_history")
    data["content_hash"] = _content_hash(data)
    source = tmp_path / "missing-deposit-history.json"
    source.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ReferenceBankError, match="deposit_initial_history"):
        ReferenceBankProvider().load(source)


def test_imported_snapshot_rejects_nonlatest_deposit_history_observation(
    tmp_path: Path,
) -> None:
    data = _imported_snapshot_data()
    history = data["deposit_initial_history"]
    history["observation_dates"][0] = "2023-11-30"
    history["six_month_yields"][0] = 0.01
    history["identity"] = deposit_initial_history_identity(
        tuple(history["target_dates"]),
        tuple(history["observation_dates"]),
        tuple(history["six_month_yields"]),
        tuple(history["available_observation_dates"]),
        tuple(history["available_six_month_yields"]),
        history["source_identity"],
    )
    data["content_hash"] = _content_hash(data)
    source = tmp_path / "nonlatest-deposit-history.json"
    source.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ReferenceBankError, match="deposit initial history"):
        ReferenceBankProvider().load(source)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda data: data["ladders"]["mortgages"].pop(),
            "180",
        ),
        (
            lambda data: data["ladders"]["funding"].__setitem__(0, -0.1),
            "cash flows",
        ),
        (
            lambda data: data.__setitem__("unit", "mCHF"),
            "currency and unit",
        ),
        (
            lambda data: data["loan_cohorts"]["mortgages"][0].update(
                {"monthly_coupon_rate": 0.01}
            ),
            "does not match",
        ),
        (
            lambda data: data["deposit_reference_schedules"][
                "non_maturity_deposits"
            ]["cash_flows"][0].__setitem__(0, 299.998),
            "reference-term schedule does not match",
        ),
    ),
)
def test_imported_snapshot_rejects_unsupported_ladders_and_mixed_units(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    data = _imported_snapshot_data()
    mutate(data)
    data["content_hash"] = _content_hash(data)
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ReferenceBankError, match=message):
        ReferenceBankProvider().load(source)


def test_runner_bank_stage_imports_a_versioned_noncanonical_snapshot(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = tmp_path / "synthetic-usd-bank.json"
    source.write_text(json.dumps(_imported_snapshot_data()), encoding="utf-8")
    configuration = resolve_configuration(
        {
            "source_data": {
                "snb_csv": str(
                    project_root / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
                ),
                "paper_pdf": str(
                    project_root / "docs/Deep treasury management for banks.pdf"
                ),
                "nss_beta_unit": "percentage_points",
            },
            "workflow_contract": {"name": "bounded-local-workflow"},
            "execution_profile": {"name": "local-cpu-compact"},
            "run_scale": {"profile": "local_flow"},
            "architecture": {"profile": "compact"},
            "reference_bank": {
                "profile": "imported",
                "snapshot_path": str(source),
            },
            "experiment": {"horizons_years": [5], "include_swaps": False},
            "policy": {"names": ["BM^E"]},
            "optimization": {"device": "cpu", "dtype": "float64"},
            "resources": {
                "wall_clock_budget_seconds": 1800,
                "process_rss_limit_bytes": 4 * 1024**3,
                "accelerator_memory_limit_bytes": 4 * 1024**3,
            },
            "seeds": {
                "market_scenarios": 11,
                "objective_parameters": 12,
                "model_initialization": 13,
                "data_loader_order": 14,
                "bootstrap": 15,
                "sensitivity": 16,
            },
            "output": {"directory": str(tmp_path / "runs"), "run_name": "imported"},
            "acceptance": {
                "purpose": "development-validation",
                "required_status": "development-validated",
            },
        }
    )

    bundle = ReproductionRunner().build_reference_bank(configuration)

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.artifact_directory is not None
    manifest = json.loads((bundle.artifact_directory / "manifest.json").read_text())
    assert manifest["reference_bank"]["profile"] == "imported"
    assert manifest["reference_bank"]["unit"] == "mUSD"
    assert manifest["resolved_configuration"]["reference_bank"]["snapshot_path"] == str(
        source
    )


def test_canonical_profile_rejects_an_unlocked_valuation_date(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    historical = MarketScenarioModel().load_historical_term_structures(
        project_root / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
    )
    data = ReferenceBankProvider().serialize(
        ReferenceBankProvider().build_canonical(historical)
    )
    data["as_of_date"] = "2024-01-31"
    data["provenance"]["market"] = (
        f"{data['initial_curve_identity']}@{data['as_of_date']}"
    )
    targets = deposit_initial_history_target_dates(data["as_of_date"])
    observations = ("2023-12-29", "2023-11-30")
    yields = (0.02, 0.01)
    available_dates = ("2023-11-30", "2023-12-29")
    available_yields = (0.01, 0.02)
    source_identity = "forged-history-fixture"
    data["deposit_initial_history"] = {
        "target_dates": list(targets),
        "observation_dates": list(observations),
        "six_month_yields": list(yields),
        "available_observation_dates": list(available_dates),
        "available_six_month_yields": list(available_yields),
        "source_identity": source_identity,
        "identity": deposit_initial_history_identity(
            targets,
            observations,
            yields,
            available_dates,
            available_yields,
            source_identity,
        ),
    }
    data["content_hash"] = _content_hash(data)
    source = tmp_path / "forged-canonical.json"
    source.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ReferenceBankError, match="locked valuation date"):
        ReferenceBankProvider().load(source)


def test_canonical_rollout_rejects_a_market_without_lineage(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    historical = MarketScenarioModel().load_historical_term_structures(
        project_root / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
    )
    snapshot = ReferenceBankProvider().build_canonical(historical)

    with pytest.raises(ReferenceBankError, match="requires market curve identity"):
        ALMSimulator().rollout(
            snapshot,
            SimpleNamespace(
                discount_factors=np.broadcast_to(
                    historical.initial_curve.discount_factors, (1, 2, 180)
                )
            ),
        )
