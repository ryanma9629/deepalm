"""Configuration resolution for auditable Deep ALM runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import yaml


class ConfigurationError(ValueError):
    """Raised when a run configuration is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class ConventionConfiguration:
    profile: str
    pca_loading_scale: str
    loan_interest_annualization: str
    is_custom: bool


@dataclass(frozen=True)
class RunScaleConfiguration:
    profile: str
    epochs: int
    training_paths_per_epoch: int
    selection_paths: int
    test_paths: int
    batch_size: int


@dataclass(frozen=True)
class ArchitectureConfiguration:
    profile: str
    widths: tuple[int, int, int, int]
    encoder_features: int = 32
    observation_features: int = 145
    final_encoding_features: int = 64
    action_features: int = 29


@dataclass(frozen=True)
class SourceDataConfiguration:
    snb_csv: Path
    paper_pdf: Path
    nss_beta_unit: str


@dataclass(frozen=True)
class ReferenceBankConfiguration:
    profile: str
    initial_assets_mchf: float


@dataclass(frozen=True)
class ExperimentConfiguration:
    horizons_years: tuple[int, ...]
    include_swaps: bool


@dataclass(frozen=True)
class PolicyConfiguration:
    names: tuple[str, ...]


@dataclass(frozen=True)
class OptimizationConfiguration:
    device: str
    dtype: str


@dataclass(frozen=True)
class ResourceConfiguration:
    wall_clock_budget_seconds: float
    process_rss_limit_bytes: int
    accelerator_memory_limit_bytes: int


@dataclass(frozen=True)
class OutputConfiguration:
    directory: Path
    run_name: str


@dataclass(frozen=True)
class AcceptanceConfiguration:
    purpose: str
    required_status: str


@dataclass(frozen=True)
class ResolvedRunConfiguration:
    source_data: SourceDataConfiguration
    convention: ConventionConfiguration
    run_scale: RunScaleConfiguration
    architecture: ArchitectureConfiguration
    reference_bank: ReferenceBankConfiguration
    experiment: ExperimentConfiguration
    policy: PolicyConfiguration
    optimization: OptimizationConfiguration
    resources: ResourceConfiguration
    seeds: dict[str, int]
    output: OutputConfiguration
    acceptance: AcceptanceConfiguration

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation of the resolved configuration."""

        result = asdict(self)
        result["configuration_schema_version"] = _CONFIGURATION_SCHEMA_VERSION
        result["source_data"] = {
            "snb_csv": str(self.source_data.snb_csv),
            "paper_pdf": str(self.source_data.paper_pdf),
            "nss_beta_unit": self.source_data.nss_beta_unit,
        }
        result["output"] = {
            "directory": str(self.output.directory),
            "run_name": self.output.run_name,
        }
        result["experiment"]["horizons_years"] = list(self.experiment.horizons_years)
        result["policy"]["names"] = list(self.policy.names)
        result["architecture"]["widths"] = list(self.architecture.widths)
        return result


_REQUIRED_SECTIONS = {
    "source_data",
    "convention",
    "run_scale",
    "architecture",
    "reference_bank",
    "experiment",
    "policy",
    "optimization",
    "resources",
    "seeds",
    "output",
    "acceptance",
}

_CONVENTION_PROFILES = {
    "paper": {
        "pca_loading_scale": "eigenvalue",
        "loan_interest_annualization": "unannualized",
    },
    "corrected": {
        "pca_loading_scale": "sqrt_eigenvalue",
        "loan_interest_annualization": "monthly",
    },
}

_RUN_SCALE_PROFILES = {
    "local_flow": {
        "epochs": 2,
        "training_paths_per_epoch": 32,
        "selection_paths": 32,
        "test_paths": 32,
        "batch_size": 8,
    },
    "quick": {
        "epochs": 5,
        "training_paths_per_epoch": 256,
        "selection_paths": 128,
        "test_paths": 128,
        "batch_size": 32,
    },
    "paper_scale": {
        "epochs": 100,
        "training_paths_per_epoch": 40_000,
        "selection_paths": 1_600,
        "test_paths": 1_600,
        "batch_size": 32,
    },
}

_ARCHITECTURE_PROFILES = {
    "compact": (64, 64, 32, 32),
    "paper": (512, 512, 256, 128),
}

_CONFIGURATION_SCHEMA_VERSION = 2


def load_configuration(path: Path) -> ResolvedRunConfiguration:
    """Load and resolve one YAML run configuration."""

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"Could not load configuration: {error}") from error

    return resolve_configuration(loaded)


def resolve_configuration(raw: object) -> ResolvedRunConfiguration:
    """Validate a raw configuration mapping and apply locked profile defaults."""

    root = _mapping(raw, "configuration")
    unknown_sections = sorted(set(root) - _REQUIRED_SECTIONS)
    missing_sections = sorted(_REQUIRED_SECTIONS - set(root))
    if unknown_sections:
        raise ConfigurationError(
            f"Unknown configuration sections: {', '.join(unknown_sections)}"
        )
    if missing_sections:
        raise ConfigurationError(
            f"Missing configuration sections: {', '.join(missing_sections)}"
        )

    resolved = ResolvedRunConfiguration(
        source_data=_resolve_source_data(root["source_data"]),
        convention=_resolve_convention(root["convention"]),
        run_scale=_resolve_run_scale(root["run_scale"]),
        architecture=_resolve_architecture(root["architecture"]),
        reference_bank=_resolve_reference_bank(root["reference_bank"]),
        experiment=_resolve_experiment(root["experiment"]),
        policy=_resolve_policy(root["policy"]),
        optimization=_resolve_optimization(root["optimization"]),
        resources=_resolve_resources(root["resources"]),
        seeds=_resolve_seeds(root["seeds"]),
        output=_resolve_output(root["output"]),
        acceptance=_resolve_acceptance(root["acceptance"]),
    )
    _validate_combinations(resolved)
    return resolved


def _resolve_source_data(raw: object) -> SourceDataConfiguration:
    section = _section(raw, "source_data", {"snb_csv", "paper_pdf", "nss_beta_unit"})
    nss_beta_unit = _string(section["nss_beta_unit"], "source_data.nss_beta_unit")
    if nss_beta_unit != "percentage_points":
        raise ConfigurationError(
            "source_data.nss_beta_unit must be 'percentage_points'"
        )
    return SourceDataConfiguration(
        snb_csv=Path(_string(section["snb_csv"], "source_data.snb_csv")),
        paper_pdf=Path(_string(section["paper_pdf"], "source_data.paper_pdf")),
        nss_beta_unit=nss_beta_unit,
    )


def _resolve_convention(raw: object) -> ConventionConfiguration:
    section = _section(
        raw,
        "convention",
        {"profile", "pca_loading_scale", "loan_interest_annualization"},
        required_keys={"profile"},
    )
    profile = _string(section["profile"], "convention.profile")
    if profile not in _CONVENTION_PROFILES:
        raise ConfigurationError("convention.profile must be 'paper' or 'corrected'")

    resolved = _CONVENTION_PROFILES[profile].copy()
    overrides = {
        key: value
        for key, value in section.items()
        if key in {"pca_loading_scale", "loan_interest_annualization"}
    }
    for key, value in overrides.items():
        resolved[key] = _string(value, f"convention.{key}")

    if resolved["pca_loading_scale"] not in {"eigenvalue", "sqrt_eigenvalue"}:
        raise ConfigurationError(
            "convention.pca_loading_scale must be 'eigenvalue' or 'sqrt_eigenvalue'"
        )
    if resolved["loan_interest_annualization"] not in {"unannualized", "monthly"}:
        raise ConfigurationError(
            "convention.loan_interest_annualization must be 'unannualized' or 'monthly'"
        )

    return ConventionConfiguration(
        profile=profile,
        pca_loading_scale=resolved["pca_loading_scale"],
        loan_interest_annualization=resolved["loan_interest_annualization"],
        is_custom=bool(overrides),
    )


def _resolve_run_scale(raw: object) -> RunScaleConfiguration:
    section = _mapping(raw, "run_scale")
    allowed = {
        "profile",
        "epochs",
        "training_paths_per_epoch",
        "selection_paths",
        "test_paths",
        "batch_size",
    }
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigurationError(
            f"Unknown configuration keys in run_scale: {', '.join(unknown)}"
        )
    if "profile" not in section:
        raise ConfigurationError("Missing configuration keys in run_scale: profile")
    profile = _string(section["profile"], "run_scale.profile")
    if profile == "bank_training":
        required = allowed - {"profile"}
        missing = sorted(required - set(section))
        if missing:
            raise ConfigurationError(
                "bank_training requires explicit values for: " + ", ".join(missing)
            )
        return RunScaleConfiguration(
            profile=profile,
            **{
                name: _positive_integer(section[name], f"run_scale.{name}")
                for name in required
            },
        )
    values = _RUN_SCALE_PROFILES.get(profile)
    if values is None:
        raise ConfigurationError(
            "run_scale.profile must be 'local_flow', 'quick', 'paper_scale', or "
            "'bank_training'"
        )
    overrides = sorted(set(section) - {"profile"})
    if overrides:
        raise ConfigurationError(
            f"run_scale {profile} does not accept overrides: {', '.join(overrides)}"
        )
    return RunScaleConfiguration(profile=profile, **values)


def _resolve_architecture(raw: object) -> ArchitectureConfiguration:
    section = _mapping(raw, "architecture")
    allowed = {"profile", "widths"}
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigurationError(
            f"Unknown configuration keys in architecture: {', '.join(unknown)}"
        )
    profile = _string(section.get("profile"), "architecture.profile")
    if profile in _ARCHITECTURE_PROFILES:
        if set(section) != {"profile"}:
            raise ConfigurationError(
                f"architecture {profile} does not accept custom widths"
            )
        widths = _ARCHITECTURE_PROFILES[profile]
    elif profile == "custom":
        if set(section) != {"profile", "widths"}:
            raise ConfigurationError("architecture custom requires widths")
        raw_widths = section["widths"]
        if not isinstance(raw_widths, list) or len(raw_widths) != 4:
            raise ConfigurationError("architecture.widths must contain four positive integers")
        widths = cast(
            tuple[int, int, int, int],
            tuple(
                _positive_integer(value, f"architecture.widths[{index}]")
                for index, value in enumerate(raw_widths)
            ),
        )
    else:
        raise ConfigurationError(
            "architecture.profile must be 'compact', 'paper', or 'custom'"
        )
    return ArchitectureConfiguration(profile=profile, widths=widths)


def _resolve_reference_bank(raw: object) -> ReferenceBankConfiguration:
    section = _section(raw, "reference_bank", {"profile", "initial_assets"})
    profile = _string(section["profile"], "reference_bank.profile")
    if profile != "canonical":
        raise ConfigurationError("reference_bank.profile must be 'canonical'")

    quantity = _section(
        section["initial_assets"],
        "reference_bank.initial_assets",
        {"value", "unit"},
    )
    if quantity["unit"] != "mCHF":
        raise ConfigurationError("reference_bank.initial_assets.unit must be 'mCHF'")
    value = quantity["value"]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError("reference_bank.initial_assets.value must be positive")
    if value != 10_000:
        raise ConfigurationError(
            "reference_bank.initial_assets.value must be 10,000 mCHF for the canonical profile"
        )

    return ReferenceBankConfiguration(profile=profile, initial_assets_mchf=float(value))


def _resolve_experiment(raw: object) -> ExperimentConfiguration:
    section = _section(raw, "experiment", {"horizons_years", "include_swaps"})
    horizons = section["horizons_years"]
    if not isinstance(horizons, list) or not horizons:
        raise ConfigurationError("experiment.horizons_years must be a non-empty list")
    if any(not isinstance(years, int) or isinstance(years, bool) for years in horizons):
        raise ConfigurationError("experiment.horizons_years must contain integers")
    if set(horizons) - {5, 15} or len(set(horizons)) != len(horizons):
        raise ConfigurationError(
            "experiment.horizons_years may contain each of 5 and 15 once"
        )

    include_swaps = section["include_swaps"]
    if include_swaps is not False:
        raise ConfigurationError(
            "experiment.include_swaps must be false for this reproduction"
        )
    return ExperimentConfiguration(tuple(horizons), include_swaps=False)


def _resolve_policy(raw: object) -> PolicyConfiguration:
    section = _section(raw, "policy", {"names"})
    names = section["names"]
    allowed = {"BM^E", "BM^C", "BM^D", "MM"}
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) for name in names)
    ):
        raise ConfigurationError(
            "policy.names must be a non-empty list of policy names"
        )
    if set(names) - allowed or len(set(names)) != len(names):
        raise ConfigurationError(
            "policy.names must contain unique supported policy names"
        )
    return PolicyConfiguration(tuple(names))


def _resolve_optimization(raw: object) -> OptimizationConfiguration:
    section = _section(raw, "optimization", {"device", "dtype"})
    requested_device = _string(section["device"], "optimization.device")
    if requested_device not in {"auto", "cpu", "mps", "cuda"}:
        raise ConfigurationError(
            "optimization.device must be 'auto', 'cpu', 'mps', or 'cuda'"
        )
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise ConfigurationError(
            "optimization.device is 'mps', but MPS is not available"
        )
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError(
            "optimization.device is 'cuda', but CUDA is not available"
        )
    device = requested_device
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dtype = _string(section["dtype"], "optimization.dtype")
    if dtype not in {"float32", "float64"}:
        raise ConfigurationError("optimization.dtype must be 'float32' or 'float64'")
    if device == "mps" and dtype != "float32":
        raise ConfigurationError("optimization.dtype must be float32 on MPS")
    return OptimizationConfiguration(device=device, dtype=dtype)


def _resolve_resources(raw: object) -> ResourceConfiguration:
    section = _section(
        raw,
        "resources",
        {
            "wall_clock_budget_seconds",
            "process_rss_limit_bytes",
            "accelerator_memory_limit_bytes",
        },
    )
    wall_clock = section["wall_clock_budget_seconds"]
    if (
        not isinstance(wall_clock, (int, float))
        or isinstance(wall_clock, bool)
        or wall_clock <= 0
    ):
        raise ConfigurationError(
            "resources.wall_clock_budget_seconds must be a positive number"
        )
    return ResourceConfiguration(
        wall_clock_budget_seconds=float(wall_clock),
        process_rss_limit_bytes=_positive_integer(
            section["process_rss_limit_bytes"], "resources.process_rss_limit_bytes"
        ),
        accelerator_memory_limit_bytes=_positive_integer(
            section["accelerator_memory_limit_bytes"],
            "resources.accelerator_memory_limit_bytes",
        ),
    )


def _resolve_seeds(raw: object) -> dict[str, int]:
    section = _mapping(raw, "seeds")
    required = {
        "market_scenarios",
        "objective_parameters",
        "model_initialization",
        "data_loader_order",
        "bootstrap",
        "sensitivity",
    }
    missing = sorted(required - set(section))
    if missing:
        raise ConfigurationError(f"Missing named seeds: {', '.join(missing)}")
    resolved: dict[str, int] = {}
    for name, value in section.items():
        if not isinstance(name, str) or not name:
            raise ConfigurationError("Seed names must be non-empty strings")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ConfigurationError("Every seed must be a non-negative integer")
        resolved[name] = value
    return dict(sorted(resolved.items()))


def _resolve_output(raw: object) -> OutputConfiguration:
    section = _section(raw, "output", {"directory", "run_name"})
    run_name = _string(section["run_name"], "output.run_name")
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ConfigurationError("output.run_name must be a simple directory name")
    return OutputConfiguration(
        directory=Path(_string(section["directory"], "output.directory")),
        run_name=run_name,
    )


def _resolve_acceptance(raw: object) -> AcceptanceConfiguration:
    section = _section(raw, "acceptance", {"purpose", "required_status"})
    purpose = _string(section["purpose"], "acceptance.purpose")
    if purpose not in {"development-validation", "research", "bank-training"}:
        raise ConfigurationError(
            "acceptance.purpose must be 'development-validation', 'research', or "
            "'bank-training'"
        )
    required_status = _string(section["required_status"], "acceptance.required_status")
    if required_status not in {"development-validated", "methodologically-reproduced"}:
        raise ConfigurationError(
            "acceptance.required_status must be 'development-validated' or "
            "'methodologically-reproduced'"
        )
    return AcceptanceConfiguration(purpose=purpose, required_status=required_status)


def _validate_combinations(configuration: ResolvedRunConfiguration) -> None:
    if (
        configuration.run_scale.profile == "bank_training"
        and configuration.acceptance.purpose != "bank-training"
    ):
        raise ConfigurationError("bank_training requires acceptance purpose bank-training")
    if configuration.acceptance.required_status != "methodologically-reproduced":
        return
    if configuration.acceptance.purpose != "research":
        raise ConfigurationError("methodologically-reproduced requires research purpose")
    if (
        configuration.convention.profile != "corrected"
        or configuration.convention.is_custom
    ):
        raise ConfigurationError(
            "methodologically-reproduced requires the locked Corrected convention"
        )
    if configuration.run_scale.profile != "paper_scale":
        raise ConfigurationError(
            "methodologically-reproduced requires the paper_scale run profile"
        )
    if set(configuration.experiment.horizons_years) != {5, 15}:
        raise ConfigurationError(
            "methodologically-reproduced requires both 5-year and 15-year horizons"
        )
    if set(configuration.policy.names) != {"BM^E", "BM^C", "BM^D", "MM"}:
        raise ConfigurationError(
            "methodologically-reproduced requires BM^E, BM^C, BM^D, and MM"
        )


def _section(
    raw: object,
    name: str,
    allowed_keys: set[str],
    *,
    required_keys: set[str] | None = None,
) -> Mapping[str, Any]:
    section = _mapping(raw, name)
    unknown = sorted(set(section) - allowed_keys)
    missing = sorted((required_keys or allowed_keys) - set(section))
    if unknown:
        raise ConfigurationError(
            f"Unknown configuration keys in {name}: {', '.join(unknown)}"
        )
    if missing:
        raise ConfigurationError(
            f"Missing configuration keys in {name}: {', '.join(missing)}"
        )
    return section


def _mapping(raw: object, name: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return raw


def _string(raw: object, name: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ConfigurationError(f"{name} must be a non-empty string")
    return raw


def _positive_integer(raw: object, name: str) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return raw
