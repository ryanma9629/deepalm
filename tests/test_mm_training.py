from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from deepalm.baselines import FrozenDateBenchmarkReference
from deepalm.config import (
    ArchitectureConfiguration,
    ConventionConfiguration,
    ExperimentConfiguration,
    PolicyConfiguration,
    RunScaleConfiguration,
    resolve_configuration,
)
from deepalm.mm import CurveFeaturePCA, MMPolicy
from deepalm.policies import TreasuryPolicyState
from deepalm.reference_bank import ReferenceBankProvider
from deepalm.term_structures import MarketScenarioModel
from deepalm.training import (
    BMDateTrainer,
    MMTrainer,
    TrainingControl,
    TrainingError,
    TrainingInterrupted,
    _derived_seed,
)

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "data/snb-data-rendopar-en-all_19880401-20250731.csv"
)


def _configuration(tmp_path: Path, *, fixture: bool = True):
    resolved = resolve_configuration(
        {
            "source_data": {
                "snb_csv": str(SOURCE),
                "paper_pdf": str(tmp_path / "paper.pdf"),
                "nss_beta_unit": "percentage_points",
            },
            "convention": {"profile": "corrected"},
            "run_scale": {"profile": "local_flow"},
            "architecture": {"profile": "compact"},
            "reference_bank": {
                "profile": "canonical",
                "initial_assets": {"value": 10_000, "unit": "mCHF"},
            },
            "experiment": {"horizons_years": [5], "include_swaps": False},
            "policy": {"names": ["BM^D", "MM"]},
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
            "output": {"directory": str(tmp_path / "runs"), "run_name": "mm-test"},
            "acceptance": {
                "purpose": "development-validation",
                "required_status": "development-validated",
            },
        }
    )
    if not fixture:
        return resolved
    return replace(
        resolved,
        run_scale=RunScaleConfiguration(
            profile="mm-fixture",
            epochs=2,
            training_paths_per_epoch=4,
            selection_paths=2,
            test_paths=2,
            batch_size=2,
        ),
    )


def _frozen_baseline(
    configuration,
    model: MarketScenarioModel,
    tmp_path: Path,
    *,
    horizon_years: int = 5,
) -> FrozenDateBenchmarkReference:
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    baseline_configuration = replace(
        configuration,
        policy=PolicyConfiguration(names=("BM^D",)),
        output=replace(configuration.output, run_name="bmd-source"),
    )
    result = BMDateTrainer(
        baseline_configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
    ).fit(horizon_years=horizon_years)
    assert result.baseline_reference_path is not None
    return FrozenDateBenchmarkReference.load(result.baseline_reference_path)


def _deterministic_smoke_state() -> TreasuryPolicyState:
    """A complete, fixed month-12 MM state for post-training action checks."""

    paths = 2
    ladder = lambda amount: torch.full((paths, 180), amount, dtype=torch.float64)
    return TreasuryPolicyState(
        investments=ladder(2.0),
        funding=ladder(1.0),
        mortgages=ladder(3.0),
        enterprise_loans=ladder(4.0),
        non_maturity_deposits=ladder(5.0),
        term_deposits=ladder(6.0),
        cash=torch.full((paths,), 7.0, dtype=torch.float64),
        curve=torch.linspace(0.01, 0.03, 180, dtype=torch.float64).repeat(paths, 1),
        discounts=torch.full((paths, 180), 0.95, dtype=torch.float64),
        initial_assets=torch.full((paths,), 100.0, dtype=torch.float64),
        prior_constraint_values=torch.ones((paths, 6), dtype=torch.float64),
        mu=torch.full((paths,), 0.04, dtype=torch.float64),
        penalty_weight=torch.full((paths,), 3.5, dtype=torch.float64),
        time=12,
        transitions=60,
    )


def test_mm_trainer_selects_and_reloads_a_compact_five_year_policy(
    tmp_path: Path,
) -> None:
    configuration = _configuration(tmp_path)
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    reference = _frozen_baseline(configuration, model, tmp_path)
    trainer = MMTrainer(
        configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
        baseline_reference=reference,
    )

    result = trainer.fit(horizon_years=5)
    checkpoint = torch.load(result.checkpoint_path, weights_only=False)
    reloaded = trainer.load_selected_checkpoint(result.checkpoint_path)
    standalone = MMTrainer.load_policy_from_checkpoint(result.checkpoint_path)
    dependencies = checkpoint["policy_dependencies"]
    pca = CurveFeaturePCA.from_dict(
        dependencies["curve_feature_pca"],
        expected_data_identity=historical.source_hash,
        expected_calibration_identity=calibration.calibration_identity,
        expected_training_state_identity=dependencies["curve_feature_pca"][
            "training_state_identity"
        ],
    )
    torch.manual_seed(
        _derived_seed(configuration.seeds["model_initialization"], "MM", 5)
    )
    initial = MMPolicy(
        reference=reference,
        curve_features=pca,
        architecture=configuration.architecture,
    )
    state = _deterministic_smoke_state()

    assert result.optimizer_updates == 4
    assert result.baseline_reference_identity == reference.reference_identity
    assert checkpoint["policy"] == "MM"
    assert (
        checkpoint["policy_dependencies"]["baseline_reference_identity"]
        == reference.reference_identity
    )
    assert checkpoint["policy_dependencies"]["curve_feature_pca"][
        "training_state_identity"
    ]
    assert (
        result.checkpoint_path.parent
        / checkpoint["policy_dependencies"]["baseline_reference_filename"]
    ).is_file()
    assert checkpoint["scenario_identities"]["training"]["split"] == "training"
    assert checkpoint["scenario_identities"]["selection"]["split"] == "selection"
    assert reloaded.audit_metadata["baseline_identity"] == reference.reference_identity
    assert (
        standalone.audit_metadata["baseline_identity"] == reference.reference_identity
    )
    assert not torch.equal(
        initial(state).concatenated.detach(), reloaded(state).concatenated.detach()
    )
    assert torch.equal(
        reloaded(state).concatenated.detach(), standalone(state).concatenated.detach()
    )
    device_records = trainer.validate_devices(horizon_years=5)
    assert device_records[0].status == "completed"
    assert (
        device_records[0].finite
        and device_records[0].updated
        and device_records[0].clipped
    )


@pytest.mark.parametrize("horizon_years", [5, 15])
def test_mm_device_validation_covers_both_full_horizons_truthfully(
    tmp_path: Path, horizon_years: int
) -> None:
    configuration = replace(
        _configuration(tmp_path),
        experiment=ExperimentConfiguration(horizons_years=(5, 15), include_swaps=False),
        run_scale=RunScaleConfiguration(
            profile="mm-device-validation-fixture",
            epochs=2,
            training_paths_per_epoch=2,
            selection_paths=2,
            test_paths=2,
            batch_size=2,
        ),
    )
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    reference = _frozen_baseline(
        configuration, model, tmp_path, horizon_years=horizon_years
    )
    records = MMTrainer(
        configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
        baseline_reference=reference,
    ).validate_devices(horizon_years=horizon_years)

    records_by_device = {record.device: record for record in records}
    assert set(records_by_device) == {"cpu", "mps", "cuda"}
    cpu = records_by_device["cpu"]
    assert cpu.status == "completed"
    assert cpu.finite and cpu.updated and cpu.clipped
    assert cpu.checkpoint_loaded and cpu.resumed_from_cpu_recovery
    assert cpu.recovery_path is not None and cpu.recovery_path.is_file()

    mps = records_by_device["mps"]
    if torch.backends.mps.is_available():
        assert mps.status == "completed"
        assert mps.finite and mps.updated and mps.clipped
        assert mps.checkpoint_loaded and mps.resumed_from_cpu_recovery
    else:
        assert mps.status == "not-run"
        assert mps.reason == "PyTorch MPS runtime is not available"

    cuda = records_by_device["cuda"]
    if torch.cuda.is_available():
        assert cuda.status == "completed"
        assert cuda.finite and cuda.updated and cuda.clipped
        assert cuda.checkpoint_loaded and cuda.resumed_from_cpu_recovery
    else:
        assert cuda.status == "not-run"
        assert cuda.reason == "PyTorch CUDA runtime is not available"


def test_mm_paper_width_check_is_one_finite_update_and_does_not_replace_compact_checkpoint(
    tmp_path: Path,
) -> None:
    configuration = _configuration(tmp_path)
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    reference = _frozen_baseline(configuration, model, tmp_path)
    trainer = MMTrainer(
        configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
        baseline_reference=reference,
    )
    compact = trainer.fit(horizon_years=5)
    compact_bytes = compact.checkpoint_path.read_bytes()

    paper_check = trainer.run_paper_width_check(horizon_years=5)

    assert paper_check.optimizer_updates == 1
    assert paper_check.paths == 2
    assert paper_check.finite and paper_check.updated and paper_check.clipped
    assert paper_check.artifact_path.is_file()
    assert compact.checkpoint_path.read_bytes() == compact_bytes
    assert paper_check.baseline_reference_identity == reference.reference_identity
    assert paper_check.architecture == ArchitectureConfiguration(
        profile="paper", widths=(512, 512, 256, 128)
    )
    artifact = json.loads(paper_check.artifact_path.read_text(encoding="utf-8"))
    assert artifact["optimizer_updates"] == 1
    assert artifact["paths"] == 2
    assert artifact["horizon_years"] == 5
    assert (
        artifact["policy_dependencies"]["baseline_reference_identity"]
        == reference.reference_identity
    )
    assert artifact["resource_profile"]["forward_backward_update_seconds"] >= 0
    assert artifact["resource_profile"]["artifact_seconds"] >= 0


def test_mm_local_flow_default_has_exactly_eight_compact_updates(
    tmp_path: Path,
) -> None:
    configuration = _configuration(tmp_path, fixture=False)

    assert configuration.run_scale.epochs == 2
    assert configuration.run_scale.training_paths_per_epoch == 32
    assert configuration.run_scale.batch_size == 8
    assert (
        configuration.run_scale.epochs
        * (
            configuration.run_scale.training_paths_per_epoch
            // configuration.run_scale.batch_size
        )
        == 8
    )
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    reference = _frozen_baseline(configuration, model, tmp_path)

    result = MMTrainer(
        configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
        baseline_reference=reference,
    ).fit(horizon_years=5)

    assert result.optimizer_updates == 8
    assert len(result.selection_history) == 2
    assert result.checkpoint_path.is_file()


def test_mm_recovery_keeps_its_frozen_dependency_contract(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    reference = _frozen_baseline(configuration, model, tmp_path)
    trainer = MMTrainer(
        configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
        baseline_reference=reference,
    )

    with pytest.raises(TrainingInterrupted):
        trainer.fit(
            horizon_years=5,
            control=TrainingControl(stop_after_completed_epoch=1),
        )
    recovery = torch.load(trainer._recovery_path(5), weights_only=False)
    assert (
        recovery["recovery_identity"]["policy_dependencies"][
            "baseline_reference_identity"
        ]
        == reference.reference_identity
    )

    result = trainer.fit(horizon_years=5, control=TrainingControl(resume=True))

    assert result.optimizer_updates == 4


def test_mm_checkpoint_loader_rejects_a_different_convention_explicitly(
    tmp_path: Path,
) -> None:
    configuration = _configuration(tmp_path)
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    reference = _frozen_baseline(configuration, model, tmp_path)
    checkpoint = MMTrainer(
        configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
        baseline_reference=reference,
    ).fit(horizon_years=5).checkpoint_path
    paper_configuration = replace(
        configuration,
        convention=ConventionConfiguration(
            profile="paper",
            pca_loading_scale="eigenvalue",
            loan_interest_annualization="unannualized",
            is_custom=False,
        ),
    )

    with pytest.raises(TrainingError, match="convention"):
        MMTrainer(
            paper_configuration,
            snapshot=snapshot,
            historical=historical,
            calibration=calibration,
            baseline_reference=reference,
        ).load_selected_checkpoint(checkpoint)


def test_mm_checkpoint_loader_rejects_different_custom_convention_fields(
    tmp_path: Path,
) -> None:
    configuration = _configuration(tmp_path)
    model = MarketScenarioModel()
    historical = model.load_historical_term_structures(SOURCE)
    calibration = model.calibrate_hjm_pca(historical)
    snapshot = ReferenceBankProvider().build_canonical(historical)
    reference = _frozen_baseline(configuration, model, tmp_path)
    checkpoint = MMTrainer(
        configuration,
        snapshot=snapshot,
        historical=historical,
        calibration=calibration,
        baseline_reference=reference,
    ).fit(horizon_years=5).checkpoint_path
    custom_configuration = replace(
        configuration,
        convention=ConventionConfiguration(
            profile="corrected",
            pca_loading_scale="eigenvalue",
            loan_interest_annualization="unannualized",
            is_custom=True,
        ),
    )

    with pytest.raises(TrainingError, match="convention"):
        MMTrainer(
            custom_configuration,
            snapshot=snapshot,
            historical=historical,
            calibration=calibration,
            baseline_reference=reference,
        ).load_selected_checkpoint(checkpoint)
