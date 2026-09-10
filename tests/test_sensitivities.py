from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from test_mm_training import _configuration, _frozen_baseline
from test_run_skeleton import configuration_data

from deepalm.config import (
    ConfigurationError,
    ReferenceBankSensitivityConfiguration,
)
from deepalm.config import (
    _resolve_internal_test_configuration as resolve_configuration,
)
from deepalm.evaluation import (
    LockedEvaluationError,
    LockedEvaluator,
    PolicyCheckpoint,
)
from deepalm.reference_bank import (
    ReferenceBankError,
    ReferenceBankProvider,
    ReferenceBankSensitivity,
)
from deepalm.runner import ReproductionRunner, RunStatus
from deepalm.term_structures import MarketScenarioModel

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
)


@pytest.fixture(scope="module")
def historical() -> object:
    return MarketScenarioModel().load_historical_term_structures(SOURCE)


def test_reference_bank_scale_sensitivity_is_a_valid_single_factor_snapshot(
    historical: object,
) -> None:
    provider = ReferenceBankProvider()
    canonical = provider.build_canonical(historical)

    variant = provider.build_sensitivity(
        historical,
        ReferenceBankSensitivity("total_assets_mchf", 5_000),
    )

    assert variant.profile == "sensitivity"
    assert variant.total_assets == pytest.approx(5_000)
    assert variant.equity == pytest.approx(500)
    assert variant.provenance["sensitivity"] == "total_assets_mchf: 10000.0 -> 5000.0"
    assert variant.product_assumptions["sensitivity_factor"] == "total_assets_mchf"
    assert variant.product_assumptions["loan_spread_decimal"] == canonical.product_assumptions[
        "loan_spread_decimal"
    ]
    assert variant.deposit_initial_history == canonical.deposit_initial_history
    provider.validate(variant, discounts=historical.initial_curve.discount_factors)


@pytest.mark.parametrize(
    "sensitivity",
    (
        ReferenceBankSensitivity("total_assets_mchf", 10_000),
        ReferenceBankSensitivity("total_assets_mchf", 20_000),
        ReferenceBankSensitivity("mortgage_ten_year_weight", 0.40),
        ReferenceBankSensitivity("mortgage_ten_year_weight", 0.20),
        ReferenceBankSensitivity("mortgage_ten_year_weight", 0.60),
        ReferenceBankSensitivity("non_maturity_ten_year_weight", 0.05),
        ReferenceBankSensitivity("non_maturity_ten_year_weight", 0.00),
        ReferenceBankSensitivity("non_maturity_ten_year_weight", 0.15),
        ReferenceBankSensitivity("term_deposit_ten_year_weight", 0.30),
        ReferenceBankSensitivity("term_deposit_ten_year_weight", 0.15),
        ReferenceBankSensitivity("term_deposit_ten_year_weight", 0.45),
        ReferenceBankSensitivity("loan_spread_decimal", 0.015),
        ReferenceBankSensitivity("loan_spread_decimal", 0.01),
        ReferenceBankSensitivity("loan_spread_decimal", 0.02),
        ReferenceBankSensitivity("operating_cost_multiplier", 1.0),
        ReferenceBankSensitivity("operating_cost_multiplier", 0.75),
        ReferenceBankSensitivity("operating_cost_multiplier", 1.25),
    ),
)
def test_approved_reference_bank_sensitivity_values_build_and_validate(
    historical: object, sensitivity: ReferenceBankSensitivity
) -> None:
    provider = ReferenceBankProvider()
    canonical = provider.build_canonical(historical)
    variant = provider.build_sensitivity(historical, sensitivity)

    assert variant.product_assumptions["sensitivity_factor"] == sensitivity.factor
    if sensitivity.factor == "total_assets_mchf":
        scale = sensitivity.value / 10_000
        assert variant.total_assets == pytest.approx(sensitivity.value)
        assert variant.equity == pytest.approx(canonical.equity * scale)
    elif sensitivity.factor == "mortgage_ten_year_weight":
        weights = variant.product_assumptions["mortgage_weights"]
        assert weights[8] == pytest.approx(sensitivity.value)
        assert all(weight == pytest.approx((1 - sensitivity.value) / 10) for index, weight in enumerate(weights) if index != 8)
    elif sensitivity.factor == "non_maturity_ten_year_weight":
        assert variant.product_assumptions["non_maturity_weights"][3] == pytest.approx(sensitivity.value)
    elif sensitivity.factor == "term_deposit_ten_year_weight":
        assert variant.product_assumptions["term_deposit_weights"][3] == pytest.approx(sensitivity.value)
    elif sensitivity.factor == "loan_spread_decimal":
        assert variant.product_assumptions["loan_spread_decimal"] == pytest.approx(sensitivity.value)
    else:
        assert variant.personnel_cost == pytest.approx(3.0 * sensitivity.value)
        assert variant.material_cost == pytest.approx(sensitivity.value)
    provider.validate(variant, discounts=historical.initial_curve.discount_factors)


def test_invalid_sensitivity_is_rejected_before_it_can_be_evaluated(
    historical: object,
) -> None:
    with pytest.raises(ReferenceBankError, match="approved"):
        ReferenceBankProvider().build_sensitivity(
            historical,
            ReferenceBankSensitivity("loan_spread_decimal", 0.0125),
        )


def test_economically_inconsistent_sensitivity_snapshot_fails_validation(
    historical: object,
) -> None:
    provider = ReferenceBankProvider()
    variant = provider.build_sensitivity(
        historical, ReferenceBankSensitivity("total_assets_mchf", 5_000)
    )

    with pytest.raises(ReferenceBankError, match="balance-sheet identity"):
        provider.validate(replace(variant, equity=variant.equity + 1.0))


def test_sensitivity_configuration_accepts_only_approved_one_factor_values(
    tmp_path: Path,
) -> None:
    configured = configuration_data(tmp_path)
    reference_bank = configured["reference_bank"]
    assert isinstance(reference_bank, dict)
    reference_bank["sensitivity"] = {
        "factor": "total_assets_mchf",
        "value": 5_000,
    }

    resolved = resolve_configuration(configured)

    assert resolved.reference_bank.sensitivity is not None
    assert resolved.reference_bank.sensitivity.factor == "total_assets_mchf"
    assert resolved.reference_bank.sensitivity.value == 5_000

    reference_bank["sensitivity"] = {
        "factor": "total_assets_mchf",
        "value": 7_500,
    }
    with pytest.raises(ConfigurationError, match="approved"):
        resolve_configuration(configured)


def test_reference_bank_stage_builds_the_configured_sensitivity(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)
    configuration.source_data.paper_pdf.write_bytes(b"paper fixture")
    configuration = replace(
        configuration,
        reference_bank=replace(
            configuration.reference_bank,
            sensitivity=ReferenceBankSensitivityConfiguration(
                factor="total_assets_mchf", value=5_000
            ),
        ),
        output=replace(configuration.output, run_name="scale-sensitivity"),
    )

    bundle = ReproductionRunner().build_reference_bank(configuration)

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.artifact_directory is not None
    table = (bundle.artifact_directory / "table-1.json").read_text(encoding="utf-8")
    assert '"profile": "sensitivity"' in table
    assert '"total_assets_mchf: 10000.0 -> 5000.0"' in table


def test_scale_sensitivity_uses_frozen_canonical_policy_on_common_test_paths(
    historical: object, tmp_path: Path
) -> None:
    configuration = _configuration(tmp_path)
    model = MarketScenarioModel()
    calibration = model.calibrate_hjm_pca(historical)
    canonical = ReferenceBankProvider().build_canonical(historical)
    frozen_baseline = _frozen_baseline(configuration, model, tmp_path)
    configuration.source_data.paper_pdf.write_bytes(b"paper fixture")
    configuration = replace(
        configuration,
        output=replace(configuration.output, run_name="frozen-scale-sensitivity"),
    )
    bundle = ReproductionRunner().evaluate_reference_bank_sensitivity(
        configuration,
        canonical_snapshot=canonical,
        historical=historical,
        calibration=calibration,
        checkpoints=(PolicyCheckpoint("BM^D", frozen_baseline.checkpoint_path),),
        validation_requests=(ReferenceBankSensitivity("loan_spread_decimal", 0.0125),),
    )

    assert bundle.status is RunStatus.COMPLETED
    assert bundle.artifact_directory is not None
    sensitivity_manifest = (bundle.artifact_directory / "reference-bank-sensitivity.json").read_text(encoding="utf-8")
    manifest = (bundle.artifact_directory / "manifest.json").read_text(encoding="utf-8")
    assert '"training_triggered": false' in sensitivity_manifest
    assert '"status": "invalid"' in sensitivity_manifest
    assert canonical.content_hash in sensitivity_manifest
    assert '"total_assets_mchf"' in sensitivity_manifest
    assert "does not establish generalization" in sensitivity_manifest
    assert '"reference_bank_sensitivity"' in manifest


def test_frozen_policy_evaluator_rejects_undeclared_changes_to_a_sensitivity_snapshot(
    historical: object, tmp_path: Path
) -> None:
    configuration = _configuration(tmp_path)
    model = MarketScenarioModel()
    canonical = ReferenceBankProvider().build_canonical(historical)
    variant = ReferenceBankProvider().build_sensitivity(
        historical, ReferenceBankSensitivity("total_assets_mchf", 5_000)
    )
    forged = replace(variant, personnel_cost=variant.personnel_cost + 1.0)

    with pytest.raises(LockedEvaluationError, match="undeclared changes"):
        LockedEvaluator(
            configuration,
            snapshot=forged,
            frozen_policy_snapshot=canonical,
            historical=historical,
            calibration=model.calibrate_hjm_pca(historical),
        )


def test_sensitivity_runner_writes_a_failure_bundle_for_invalid_evaluation_input(
    historical: object, tmp_path: Path
) -> None:
    configuration = _configuration(tmp_path)
    configuration.source_data.paper_pdf.write_bytes(b"paper fixture")
    configuration = replace(
        configuration,
        output=replace(configuration.output, run_name="failed-scale-sensitivity"),
    )
    model = MarketScenarioModel()

    bundle = ReproductionRunner().evaluate_reference_bank_sensitivity(
        configuration,
        canonical_snapshot=ReferenceBankProvider().build_canonical(historical),
        historical=historical,
        calibration=model.calibrate_hjm_pca(historical),
        checkpoints=(),
    )

    assert bundle.status is RunStatus.FAILED
    assert bundle.artifact_directory is not None
    assert (bundle.artifact_directory / "manifest.json").is_file()


def test_sensitivity_runner_rejects_nonrepresentative_local_evaluation(
    historical: object, tmp_path: Path
) -> None:
    configuration = _configuration(tmp_path)
    configuration.source_data.paper_pdf.write_bytes(b"paper fixture")
    configuration = replace(
        configuration,
        reference_bank=replace(
            configuration.reference_bank,
            sensitivity=ReferenceBankSensitivityConfiguration(
                factor="loan_spread_decimal", value=0.01
            ),
        ),
        output=replace(configuration.output, run_name="nonrepresentative-sensitivity"),
    )
    model = MarketScenarioModel()

    bundle = ReproductionRunner().evaluate_reference_bank_sensitivity(
        configuration,
        canonical_snapshot=ReferenceBankProvider().build_canonical(historical),
        historical=historical,
        calibration=model.calibrate_hjm_pca(historical),
        checkpoints=(),
    )

    assert bundle.status is RunStatus.FAILED
    assert bundle.artifact_directory is not None
    failure_manifest = (bundle.artifact_directory / "manifest.json").read_text(
        encoding="utf-8"
    )
    assert "broader variants require" in failure_manifest
