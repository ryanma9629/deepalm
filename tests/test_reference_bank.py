from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from deepalm.config import resolve_configuration
from deepalm.reference_bank import ReferenceBankError, ReferenceBankProvider
from deepalm.runner import ReproductionRunner, RunStatus
from deepalm.term_structures import MarketScenarioModel

SOURCE = Path(__file__).resolve().parents[1] / "data/snb-data-rendopar-en-all_19880401-20250731.csv"


@pytest.fixture(scope="module")
def canonical_bank() -> object:
    history = MarketScenarioModel().load_historical_term_structures(SOURCE)
    return ReferenceBankProvider().build_canonical(history)


def test_canonical_reference_bank_matches_balance_sheet_and_curve_values(canonical_bank: object) -> None:
    bank = canonical_bank
    assert bank.currency == "CHF"
    assert bank.unit == "mCHF"
    assert bank.cash == pytest.approx(2_000)
    assert bank.equity == pytest.approx(1_000)
    assert bank.total_assets == pytest.approx(10_000)
    assert bank.total_liabilities == pytest.approx(9_000)
    assert set(bank.ladders) == {"mortgages", "enterprise_loans", "investments", "funding", "non_maturity_deposits", "term_deposits"}
    assert all(ladder.shape == (180,) for ladder in bank.ladders.values())
    assert all(np.all(ladder >= 0) for ladder in bank.ladders.values())
    assert all(error <= 1e-8 for error in bank.target_value_errors.values())
    assert bank.schema_version == 5
    assert tuple(bank.loan_cohorts) == ("mortgages", "enterprise_loans")
    assert len(bank.loan_cohorts["mortgages"]) == 11
    assert len(bank.loan_cohorts["enterprise_loans"]) == 3
    assert set(bank.deposit_reference_schedules) == {
        "non_maturity_deposits",
        "term_deposits",
    }
    for name, schedule in bank.deposit_reference_schedules.items():
        assert schedule.terms_months == (1, 2, 12, 120)
        assert schedule.cash_flows.shape == (4, 180)
        assert np.allclose(schedule.cash_flows.sum(axis=0), bank.ladders[name])
        assert not schedule.cash_flows.flags.writeable
        assert schedule.provenance
    assert all(
        not cohort.principal_cash_flows.flags.writeable
        for cohorts in bank.loan_cohorts.values()
        for cohort in cohorts
    )
    assert bank.loan_duration_years < 5
    assert bank.deposit_duration_years < 3
    assert len(bank.content_hash) == 64


def test_canonical_bank_records_dated_deposit_rate_history(canonical_bank: object) -> None:
    bank = canonical_bank
    history = bank.deposit_initial_history

    assert history.target_dates == ("2022-06-15", "2022-05-15")
    assert history.observation_dates == ("2022-06-15", "2022-05-13")
    assert history.source_identity == bank.initial_curve_identity
    assert len(history.six_month_yields) == 2


def test_saved_snapshot_round_trips_and_table_one_is_assumption_labeled(
    canonical_bank: object, tmp_path: Path
) -> None:
    provider = ReferenceBankProvider()
    snapshot_path = tmp_path / "reference-bank.json"
    table_path = tmp_path / "table-1.json"
    provider.save(canonical_bank, snapshot_path)
    loaded = provider.load(snapshot_path)
    provider.write_table_one(canonical_bank, table_path)

    assert loaded.content_hash == canonical_bank.content_hash
    assert np.array_equal(loaded.ladders["mortgages"], canonical_bank.ladders["mortgages"])
    assert '"assumptions"' in table_path.read_text(encoding="utf-8")


def test_reference_bank_validation_rejects_invalid_ladder(canonical_bank: object) -> None:
    bad_ladders = dict(canonical_bank.ladders)
    bad_ladders["mortgages"] = bad_ladders["mortgages"][:-1]
    invalid = replace(canonical_bank, ladders=bad_ladders)

    with pytest.raises(ReferenceBankError, match="180"):
        ReferenceBankProvider().validate(invalid)


def test_loading_rejects_tampered_persisted_fields(
    canonical_bank: object, tmp_path: Path
) -> None:
    provider = ReferenceBankProvider()
    snapshot_path = tmp_path / "reference-bank.json"
    provider.save(canonical_bank, snapshot_path)
    contents = json.loads(snapshot_path.read_text(encoding="utf-8"))
    contents["unit"] = "CHF"
    snapshot_path.write_text(json.dumps(contents), encoding="utf-8")

    with pytest.raises(ReferenceBankError, match="canonical contract"):
        provider.load(snapshot_path)


def test_standalone_reference_bank_stage_writes_snapshot_and_table_one(
    tmp_path: Path,
) -> None:
    project_root = SOURCE.parents[1]
    configuration = resolve_configuration(
        {
            "source_data": {
                "snb_csv": str(SOURCE),
                "paper_pdf": str(project_root / "docs/Deep treasury management for banks.pdf"),
                "nss_beta_unit": "percentage_points",
            },
            "workflow_contract": {"name": "bounded-local-workflow"},
            "execution_profile": {"name": "local-cpu-compact"},
            "run_scale": {"profile": "local_flow"},
            "architecture": {"profile": "compact"},
            "reference_bank": {
                "profile": "canonical",
                "initial_assets": {"value": 10_000, "unit": "mCHF"},
            },
            "experiment": {"horizons_years": [5, 15], "include_swaps": False},
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
            "output": {"directory": str(tmp_path), "run_name": "reference-bank"},
            "acceptance": {
                "purpose": "development-validation",
                "required_status": "development-validated",
            },
        }
    )

    bundle = ReproductionRunner().build_reference_bank(configuration)

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.artifact_directory is not None
    assert {path.name for path in bundle.artifacts} == {
        "manifest.json",
        "reference-bank.json",
        "table-1.json",
    }
    table_one = json.loads((bundle.artifact_directory / "table-1.json").read_text())
    assert table_one["table"]["mortgages"] == 5_500
    assert table_one["assumptions"]["loan_spread_decimal"] == 0.015
