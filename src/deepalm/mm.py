"""Shared-parameter multi-period treasury policy and its observation contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from itertools import pairwise

import torch
from torch import nn

from deepalm.baselines import FrozenDateBenchmarkReference
from deepalm.config import ArchitectureConfiguration
from deepalm.constraints import (
    CONSTRAINT_FEATURE_LOWER_BOUNDS,
    ConstraintState,
    portfolio_values,
)
from deepalm.policies import TreasuryPolicy, TreasuryPolicyState
from deepalm.treasury import TreasuryAction, TreasuryActionError

_CURVE_FEATURES = 3
_ENCODER_FEATURES = 32
_OBSERVATION_FEATURES = 145
_FINAL_ENCODING_FEATURES = 64
_INVESTMENT_ACTIONS = 13
_FUNDING_ACTIONS = 16
_MAX_CURVE_TRAINING_STATES = 50_000
_APPROVED_WIDTHS = {
    "compact": (64, 64, 32, 32),
    "paper": (512, 512, 256, 128),
}


class MMObservationError(TreasuryActionError):
    """Raised when MM preprocessing, state, or action contracts are incompatible."""


@dataclass(frozen=True)
class RegisteredTrainingCurves:
    """A named, immutable input set permitted to fit MM curve preprocessing."""

    curves: torch.Tensor
    data_identity: str
    calibration_identity: str

    def __post_init__(self) -> None:
        _require_identity("data identity", self.data_identity)
        _require_identity("calibration identity", self.calibration_identity)
        _curve_training_matrix(self.curves)

    @property
    def identity(self) -> str:
        values = self.curves.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(tuple(values.shape)).encode())
        digest.update(str(values.dtype).encode())
        digest.update(values.numpy().tobytes())
        digest.update(self.data_identity.encode())
        digest.update(self.calibration_identity.encode())
        return digest.hexdigest()


@dataclass(frozen=True)
class CurveFeaturePCA:
    """Centered, unscaled three-factor yield-curve preprocessing for MM."""

    center: torch.Tensor
    projection: torch.Tensor
    sample_indices: tuple[tuple[int, int], ...]
    data_identity: str
    calibration_identity: str
    training_state_identity: str
    identity: str

    @classmethod
    def fit(
        cls,
        training_states: RegisteredTrainingCurves,
    ) -> CurveFeaturePCA:
        """Fit only registered training curves, without variance standardization."""

        if not isinstance(training_states, RegisteredTrainingCurves):
            raise MMObservationError(
                "Curve PCA may fit only registered training states"
            )
        curves, paths, times = _curve_training_matrix(training_states.curves)
        selected = _stratified_curve_indices(paths, times, _MAX_CURVE_TRAINING_STATES)
        flattened = curves.reshape(paths * times, 180)
        samples = flattened[_flatten_indices(selected, times)]
        if samples.shape[0] < _CURVE_FEATURES:
            raise MMObservationError(
                "Curve PCA requires at least three training states"
            )
        center = samples.mean(dim=0)
        centered = samples - center
        try:
            _unused_u, _singular_values, right_vectors = torch.linalg.svd(
                centered, full_matrices=False
            )
        except RuntimeError as error:
            raise MMObservationError("Curve PCA fitting failed") from error
        if right_vectors.shape[0] < _CURVE_FEATURES:
            raise MMObservationError("Curve PCA did not produce three components")
        projection = _orient_components(right_vectors[:_CURVE_FEATURES].transpose(0, 1))
        identity = _curve_feature_identity(
            center=center,
            projection=projection,
            sample_indices=selected,
            data_identity=training_states.data_identity,
            calibration_identity=training_states.calibration_identity,
            training_state_identity=training_states.identity,
        )
        return cls(
            center=center.detach().cpu().clone(),
            projection=projection.detach().cpu().clone(),
            sample_indices=selected,
            data_identity=training_states.data_identity,
            calibration_identity=training_states.calibration_identity,
            training_state_identity=training_states.identity,
            identity=identity,
        )

    def transform(self, curves: torch.Tensor) -> torch.Tensor:
        """Project one current yield curve per path with the registered transform."""

        _require_curve_matrix(curves, "Curve features")
        center = self.center.to(device=curves.device, dtype=curves.dtype)
        projection = self.projection.to(device=curves.device, dtype=curves.dtype)
        return (curves - center) @ projection

    def to_dict(self) -> dict[str, object]:
        """Return checkpoint-ready preprocessing evidence and registered identities."""

        return {
            "format_version": 1,
            "center": self.center.tolist(),
            "projection": self.projection.tolist(),
            "sample_indices": [list(index) for index in self.sample_indices],
            "data_identity": self.data_identity,
            "calibration_identity": self.calibration_identity,
            "training_state_identity": self.training_state_identity,
            "identity": self.identity,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        expected_data_identity: str,
        expected_calibration_identity: str,
        expected_training_state_identity: str,
    ) -> CurveFeaturePCA:
        """Restore only preprocessing bound to the expected data and calibration."""

        if not isinstance(value, dict) or value.get("format_version") != 1:
            raise MMObservationError("Curve feature preprocessing has invalid format")
        data_identity = value.get("data_identity")
        calibration_identity = value.get("calibration_identity")
        if data_identity != expected_data_identity:
            raise MMObservationError(
                "Curve feature preprocessing data identity mismatch"
            )
        if calibration_identity != expected_calibration_identity:
            raise MMObservationError(
                "Curve feature preprocessing calibration identity mismatch"
            )
        training_state_identity = value.get("training_state_identity")
        if training_state_identity != expected_training_state_identity:
            raise MMObservationError(
                "Curve feature preprocessing training-state identity mismatch"
            )
        try:
            center = torch.tensor(value["center"], dtype=torch.float64)
            projection = torch.tensor(value["projection"], dtype=torch.float64)
            sample_indices = tuple(
                (int(index[0]), int(index[1])) for index in value["sample_indices"]
            )
            identity = str(value["identity"])
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise MMObservationError(
                "Curve feature preprocessing is invalid"
            ) from error
        _require_curve_vector(center, "Curve PCA center")
        if (
            projection.shape != (180, _CURVE_FEATURES)
            or not torch.isfinite(projection).all()
        ):
            raise MMObservationError("Curve PCA projection must have shape [180, 3]")
        if not sample_indices:
            raise MMObservationError("Curve PCA requires registered sample indices")
        expected_identity = _curve_feature_identity(
            center=center,
            projection=projection,
            sample_indices=sample_indices,
            data_identity=data_identity,
            calibration_identity=calibration_identity,
            training_state_identity=training_state_identity,
        )
        if identity != expected_identity:
            raise MMObservationError("Curve feature preprocessing identity mismatch")
        return cls(
            center=center,
            projection=projection,
            sample_indices=sample_indices,
            data_identity=data_identity,
            calibration_identity=calibration_identity,
            training_state_identity=training_state_identity,
            identity=identity,
        )


class MMPolicy(TreasuryPolicy):
    """One time-shared ELU residual policy around a frozen BM^D baseline."""

    def __init__(
        self,
        *,
        reference: FrozenDateBenchmarkReference,
        curve_features: CurveFeaturePCA,
        architecture: ArchitectureConfiguration,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        baseline = reference.load_policy(device=device, dtype=dtype)
        baseline_identity = reference.reference_identity
        if architecture.encoder_features != _ENCODER_FEATURES:
            raise MMObservationError("MM requires four independent 32-feature encoders")
        if architecture.observation_features != _OBSERVATION_FEATURES:
            raise MMObservationError("MM requires a 145-feature observation")
        if architecture.final_encoding_features != _FINAL_ENCODING_FEATURES:
            raise MMObservationError("MM requires a 64-feature final encoding")
        if architecture.action_features != _INVESTMENT_ACTIONS + _FUNDING_ACTIONS:
            raise MMObservationError("MM requires 29 no-swap action features")
        if (
            architecture.profile not in _APPROVED_WIDTHS
            or architecture.widths != _APPROVED_WIDTHS[architecture.profile]
        ):
            raise MMObservationError(
                "MM architecture must use approved compact or paper widths"
            )

        reference_parameter = next(baseline.parameters(), None)
        if (
            reference_parameter is None
        ):  # pragma: no cover - BM^D always has parameters.
            raise MMObservationError("MM baseline must contain BM^D parameters")
        options = {
            "device": reference_parameter.device,
            "dtype": reference_parameter.dtype,
        }
        self.baseline = baseline
        self.baseline_identity = baseline_identity
        self.curve_feature_identity = curve_features.identity
        self.architecture = architecture
        self.register_buffer("curve_center", curve_features.center.to(**options))
        self.register_buffer(
            "curve_projection", curve_features.projection.to(**options)
        )
        self.investment_encoder = nn.Linear(180, _ENCODER_FEATURES, **options)
        self.funding_encoder = nn.Linear(180, _ENCODER_FEATURES, **options)
        self.loan_encoder = nn.Linear(180, _ENCODER_FEATURES, **options)
        self.deposit_encoder = nn.Linear(180, _ENCODER_FEATURES, **options)
        dimensions = (architecture.observation_features, *architecture.widths)
        self.residual_blocks = nn.ModuleList(
            _ELUResidualBlock(in_features, out_features, **options)
            for in_features, out_features in pairwise(dimensions)
        )
        self.final_encoding = nn.Linear(
            architecture.widths[-1], architecture.final_encoding_features, **options
        )
        self.investment_scale_head = nn.Linear(_FINAL_ENCODING_FEATURES, 1, **options)
        self.investment_distribution_head = nn.Linear(
            _FINAL_ENCODING_FEATURES, _INVESTMENT_ACTIONS, **options
        )
        self.funding_scale_head = nn.Linear(_FINAL_ENCODING_FEATURES, 1, **options)
        self.funding_distribution_head = nn.Linear(
            _FINAL_ENCODING_FEATURES, _FUNDING_ACTIONS, **options
        )

    @property
    def requires_full_state(self) -> bool:
        """Signal that the simulator must provide curve and prior constraints."""

        return True

    @property
    def audit_metadata(self) -> dict[str, object]:
        """Expose the fixed shared architecture and frozen dependencies."""

        return {
            "architecture_profile": self.architecture.profile,
            "widths": self.architecture.widths,
            "encoder_features": _ENCODER_FEATURES,
            "observation_features": _OBSERVATION_FEATURES,
            "final_encoding_features": _FINAL_ENCODING_FEATURES,
            "action_features": _INVESTMENT_ACTIONS + _FUNDING_ACTIONS,
            "curve_feature_identity": self.curve_feature_identity,
            "baseline_identity": self.baseline_identity,
            "batch_normalization": False,
            "time_parameters": "shared",
        }

    def build_observation(self, state: TreasuryPolicyState) -> torch.Tensor:
        """Build the complete stop-gradient MM observation at one decision date."""

        _validate_mm_state(state, baseline_transitions=self.baseline.transitions)
        if (
            state.investments.device != self.investment_encoder.weight.device
            or state.investments.dtype != self.investment_encoder.weight.dtype
        ):
            raise MMObservationError(
                "MM observation tensors must share device and dtype with the policy"
            )
        return self._observation_from_stopped_state(state)

    def forward(self, state: TreasuryPolicyState) -> TreasuryAction:
        observation = self.build_observation(state)
        encoding = observation
        for block in self.residual_blocks:
            encoding = block(encoding)
        encoding = torch.nn.functional.elu(self.final_encoding(encoding))
        investment_adjustment = self.investment_scale_head(encoding) * torch.softmax(
            self.investment_distribution_head(encoding), dim=1
        )
        funding_adjustment = self.funding_scale_head(encoding) * torch.softmax(
            self.funding_distribution_head(encoding), dim=1
        )
        baseline = self.baseline(_detached_baseline_state(state))
        return TreasuryAction(
            investments=torch.relu(baseline.investments + investment_adjustment),
            funding=torch.relu(baseline.funding + funding_adjustment),
        )

    def _observation_from_stopped_state(
        self, state: TreasuryPolicyState
    ) -> torch.Tensor:
        assert state.mortgages is not None
        assert state.enterprise_loans is not None
        assert state.non_maturity_deposits is not None
        assert state.term_deposits is not None
        assert state.cash is not None
        assert state.curve is not None
        assert state.discounts is not None
        assert state.initial_assets is not None
        assert state.prior_constraint_values is not None
        assert state.mu is not None
        assert state.penalty_weight is not None
        investments = state.investments.detach()
        funding = state.funding.detach()
        mortgages = state.mortgages.detach()
        enterprise_loans = state.enterprise_loans.detach()
        non_maturity_deposits = state.non_maturity_deposits.detach()
        term_deposits = state.term_deposits.detach()
        cash = state.cash.detach()
        curve = state.curve.detach()
        discounts = state.discounts.detach()
        initial_assets = state.initial_assets.detach()
        prior_constraints = state.prior_constraint_values.detach()
        mu = state.mu.detach()
        penalty_weight = state.penalty_weight.detach()
        encodings = (
            torch.nn.functional.elu(self.investment_encoder(investments / 100.0)),
            torch.nn.functional.elu(self.funding_encoder(funding / 100.0)),
            torch.nn.functional.elu(
                self.loan_encoder((mortgages + enterprise_loans) / 100.0)
            ),
            torch.nn.functional.elu(
                self.deposit_encoder((non_maturity_deposits + term_deposits) / 100.0)
            ),
        )
        curve_factors = (curve - self.curve_center) @ self.curve_projection
        (
            investment_value,
            mortgage_value,
            enterprise_loan_value,
            non_maturity_deposit_value,
            term_deposit_value,
            funding_value,
        ) = portfolio_values(
            ConstraintState(
                cash=cash,
                investments=investments,
                mortgages=mortgages,
                enterprise_loans=enterprise_loans,
                non_maturity_deposits=non_maturity_deposits,
                term_deposits=term_deposits,
                funding=funding,
                discounts=discounts,
            )
        )
        total_assets = cash + investment_value + mortgage_value + enterprise_loan_value
        _require_positive_ratio_denominator(
            total_assets, name="economic assets", time=state.time
        )
        _require_positive_ratio_denominator(
            initial_assets, name="initial economic assets", time=state.time
        )
        equity = (
            total_assets
            - non_maturity_deposit_value
            - term_deposit_value
            - funding_value
        )
        relative_balance_sheet = torch.stack(
            (
                total_assets / initial_assets,
                equity / total_assets,
                cash / total_assets,
                investment_value / total_assets,
                funding_value / total_assets,
            ),
            dim=1,
        )
        centered_constraints = prior_constraints - prior_constraints.new_tensor(
            CONSTRAINT_FEATURE_LOWER_BOUNDS
        )
        normalized_time = torch.full_like(mu, state.time / state.transitions)
        return torch.cat(
            (
                *encodings,
                curve_factors,
                relative_balance_sheet,
                centered_constraints,
                normalized_time.unsqueeze(1),
                mu.unsqueeze(1),
                penalty_weight.unsqueeze(1),
            ),
            dim=1,
        )


class TruncatedMMPolicy(TreasuryPolicy):
    """Read-only first-five-year view of a selected fifteen-year MM policy.

    The ALM rollout owns a 60-step terminal horizon, but the wrapped policy is
    deliberately given its original 180-step state horizon.  Consequently its
    time feature is ``t / 180`` and its frozen 15-year BM^D baseline remains in
    force; this object is evaluation-only and creates no new trainable policy.
    """

    def __init__(self, policy: MMPolicy, *, decisions: int = 60) -> None:
        super().__init__()
        if policy.baseline.transitions != 180 or decisions != 60:
            raise MMObservationError(
                "MM truncation requires the first 60 decisions of a 15-year policy"
            )
        self.policy = policy
        self.decisions = decisions
        self.policy.eval()
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)

    @property
    def requires_full_state(self) -> bool:
        return self.policy.requires_full_state

    @property
    def audit_metadata(self) -> dict[str, object]:
        return {
            **self.policy.audit_metadata,
            "trained_horizon_years": self.policy.baseline.transitions // 12,
            "truncated_decisions": self.decisions,
            "time_normalization": "original-trained-horizon",
            "optimizer_updates": 0,
        }

    def forward(self, state: TreasuryPolicyState) -> TreasuryAction:
        if state.time >= self.decisions:
            raise MMObservationError(
                "Truncated MM may only execute its first 60 decisions"
            )
        return self.policy(replace(state, transitions=180))


class _ELUResidualBlock(nn.Module):
    """Residual ELU block with a learned projection only when widths differ."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.expand = nn.Linear(in_features, out_features, device=device, dtype=dtype)
        self.refine = nn.Linear(out_features, out_features, device=device, dtype=dtype)
        self.skip: nn.Module = (
            nn.Identity()
            if in_features == out_features
            else nn.Linear(in_features, out_features, device=device, dtype=dtype)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.skip(values)
        transformed = torch.nn.functional.elu(self.expand(values))
        return torch.nn.functional.elu(self.refine(transformed) + residual)


def _validate_mm_state(
    state: TreasuryPolicyState, *, baseline_transitions: int
) -> None:
    required = (
        state.mortgages,
        state.enterprise_loans,
        state.non_maturity_deposits,
        state.term_deposits,
        state.cash,
        state.curve,
        state.discounts,
        state.initial_assets,
        state.prior_constraint_values,
        state.mu,
        state.penalty_weight,
    )
    if any(value is None for value in required):
        raise MMObservationError("MM requires a complete MM observation state")
    if state.transitions != baseline_transitions:
        raise MMObservationError("MM state horizon must match its frozen BM^D baseline")
    reference = state.investments
    for value in (state.funding, *required):
        assert value is not None
        if value.device != reference.device or value.dtype != reference.dtype:
            raise MMObservationError(
                "MM observation tensors must share device and dtype"
            )
    if state.curve is not None and state.curve.shape != reference.shape:
        raise MMObservationError("MM curve must have shape [paths, 180]")


def _detached_baseline_state(state: TreasuryPolicyState) -> TreasuryPolicyState:
    return TreasuryPolicyState(
        investments=state.investments.detach(),
        funding=state.funding.detach(),
        time=state.time,
        transitions=state.transitions,
    )


def _require_positive_ratio_denominator(
    values: torch.Tensor, *, name: str, time: int
) -> None:
    invalid_paths = torch.nonzero(values <= 0, as_tuple=False).flatten().tolist()
    if invalid_paths:
        raise MMObservationError(
            f"MM {name} must be positive at decision {time}; invalid paths {invalid_paths}"
        )


def _curve_training_matrix(
    training_curves: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    if training_curves.ndim == 2:
        _require_curve_matrix(training_curves, "Curve PCA training states")
        return training_curves.unsqueeze(1), training_curves.shape[0], 1
    if training_curves.ndim != 3 or training_curves.shape[2] != 180:
        raise MMObservationError(
            "Curve PCA training states must have shape [paths, states, 180]"
        )
    if training_curves.shape[0] == 0 or training_curves.shape[1] == 0:
        raise MMObservationError("Curve PCA requires positive paths and states")
    if not torch.isfinite(training_curves).all():
        raise MMObservationError("Curve PCA training states must be finite")
    return training_curves, training_curves.shape[0], training_curves.shape[1]


def _stratified_curve_indices(
    paths: int, times: int, max_samples: int
) -> tuple[tuple[int, int], ...]:
    total = paths * times
    if total <= max_samples:
        return tuple((path, time) for path in range(paths) for time in range(times))
    selected_times = _evenly_spaced_indices(times, min(times, max_samples))
    base_quota, remainder = divmod(max_samples, len(selected_times))
    indices: list[tuple[int, int]] = []
    for position, time in enumerate(selected_times):
        quota = base_quota + int(position < remainder)
        indices.extend((path, time) for path in _evenly_spaced_indices(paths, quota))
    return tuple(indices)


def _evenly_spaced_indices(upper_bound: int, count: int) -> tuple[int, ...]:
    if count <= 0 or count > upper_bound:
        raise MMObservationError("Curve PCA stratification has an invalid sample quota")
    selected = (
        torch.linspace(0, upper_bound - 1, steps=count, dtype=torch.float64)
        .round()
        .to(torch.long)
    )
    return tuple(int(index) for index in selected.tolist())


def _flatten_indices(indices: tuple[tuple[int, int], ...], times: int) -> torch.Tensor:
    return torch.tensor(
        [path * times + time for path, time in indices], dtype=torch.long
    )


def _orient_components(projection: torch.Tensor) -> torch.Tensor:
    oriented = projection.clone()
    for component in range(oriented.shape[1]):
        pivot = oriented[:, component].abs().argmax()
        if oriented[pivot, component] < 0:
            oriented[:, component] *= -1
    return oriented


def _curve_feature_identity(
    *,
    center: torch.Tensor,
    projection: torch.Tensor,
    sample_indices: tuple[tuple[int, int], ...],
    data_identity: str,
    calibration_identity: str,
    training_state_identity: str,
) -> str:
    payload = {
        "center": center.detach().cpu().tolist(),
        "projection": projection.detach().cpu().tolist(),
        "sample_indices": sample_indices,
        "data_identity": data_identity,
        "calibration_identity": calibration_identity,
        "training_state_identity": training_state_identity,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _require_identity(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise MMObservationError(f"MM {name} must be a non-empty string")


def _require_curve_matrix(curves: torch.Tensor, name: str) -> None:
    if curves.ndim != 2 or curves.shape[0] == 0 or curves.shape[1] != 180:
        raise MMObservationError(f"{name} must have shape [positive paths, 180]")
    if not torch.isfinite(curves).all():
        raise MMObservationError(f"{name} must be finite")


def _require_curve_vector(curve: torch.Tensor, name: str) -> None:
    if curve.shape != (180,) or not torch.isfinite(curve).all():
        raise MMObservationError(f"{name} must have shape [180] and be finite")
