"""Treasury policy modules used by the differentiable ALM simulator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from deepalm.treasury import TreasuryAction, TreasuryActionError


@dataclass(frozen=True)
class TreasuryPolicyState:
    """Minimal decision-time state exposed to a no-swap treasury policy.

    The simulator owns the broader balance-sheet transition.  A policy only sees
    the two ladders it may trade and the point in the finite decision horizon.
    """

    investments: torch.Tensor
    funding: torch.Tensor
    time: int
    transitions: int
    mortgages: torch.Tensor | None = None
    enterprise_loans: torch.Tensor | None = None
    non_maturity_deposits: torch.Tensor | None = None
    term_deposits: torch.Tensor | None = None
    cash: torch.Tensor | None = None
    curve: torch.Tensor | None = None
    discounts: torch.Tensor | None = None
    initial_assets: torch.Tensor | None = None
    prior_constraint_values: torch.Tensor | None = None
    mu: torch.Tensor | None = None
    penalty_weight: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.investments.ndim != 2 or self.investments.shape[1] != 180:
            raise TreasuryActionError("Policy investments must have shape [paths, 180]")
        if self.funding.shape != self.investments.shape:
            raise TreasuryActionError(
                "Policy funding must match investment ladder shape"
            )
        if not 0 <= self.time < self.transitions:
            raise TreasuryActionError("Policy time must be within the decision horizon")
        if (
            not torch.isfinite(self.investments).all()
            or not torch.isfinite(self.funding).all()
        ):
            raise TreasuryActionError("Policy state must be finite")
        paths = self.investments.shape[0]
        ladders = {
            "mortgages": self.mortgages,
            "enterprise_loans": self.enterprise_loans,
            "non_maturity_deposits": self.non_maturity_deposits,
            "term_deposits": self.term_deposits,
            "curve": self.curve,
            "discounts": self.discounts,
        }
        for name, value in ladders.items():
            if value is not None and value.shape != self.investments.shape:
                raise TreasuryActionError(
                    f"Policy {name} must match investment ladder shape"
                )
        scalar_features = {
            "cash": self.cash,
            "initial_assets": self.initial_assets,
            "mu": self.mu,
            "penalty_weight": self.penalty_weight,
        }
        for name, value in scalar_features.items():
            if value is not None and value.shape != (paths,):
                raise TreasuryActionError(f"Policy {name} must have shape [paths]")
        if (
            self.prior_constraint_values is not None
            and self.prior_constraint_values.shape != (paths, 6)
        ):
            raise TreasuryActionError(
                "Policy prior_constraint_values must have shape [paths, 6]"
            )
        additional = (*ladders.values(), *scalar_features.values(), self.prior_constraint_values)
        if any(value is not None and not torch.isfinite(value).all() for value in additional):
            raise TreasuryActionError("Policy state must be finite")


class TreasuryPolicy(nn.Module):
    """A trainable policy mapping one decision-time state to a TreasuryAction."""

    def forward(self, state: TreasuryPolicyState) -> TreasuryAction:  # pragma: no cover
        raise NotImplementedError


class BMEqualPolicy(TreasuryPolicy):
    """BM^E benchmark: equally distribute each current maturity scale.

    The two scalar adjustments are shared across every decision month.  They
    preserve a non-negative no-swap action through a ReLU at the policy edge.
    """

    def __init__(
        self,
        *,
        investment_adjustment: float = 0.0,
        funding_adjustment: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        options: dict[str, torch.device | str | torch.dtype] = {}
        if device is not None:
            options["device"] = device
        if dtype is not None:
            options["dtype"] = dtype
        self.investment_scale_adjustment = nn.Parameter(
            torch.tensor(investment_adjustment, **options)
        )
        self.funding_scale_adjustment = nn.Parameter(
            torch.tensor(funding_adjustment, **options)
        )

    def forward(self, state: TreasuryPolicyState) -> TreasuryAction:
        investment_scale = torch.relu(
            state.investments[:, 0] + self.investment_scale_adjustment
        )
        funding_scale = torch.relu(state.funding[:, 0] + self.funding_scale_adjustment)
        return TreasuryAction(
            investments=investment_scale.unsqueeze(1).expand(-1, 13) / 13.0,
            funding=funding_scale.unsqueeze(1).expand(-1, 16) / 16.0,
        )


class BMConstantPolicy(TreasuryPolicy):
    """BM^C benchmark: learned fixed maturity distributions and scales."""

    def __init__(
        self,
        *,
        investment_adjustment: float = 0.0,
        funding_adjustment: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        options: dict[str, torch.device | str | torch.dtype] = {}
        if device is not None:
            options["device"] = device
        if dtype is not None:
            options["dtype"] = dtype
        self.investment_scale_adjustment = nn.Parameter(
            torch.tensor(investment_adjustment, **options)
        )
        self.funding_scale_adjustment = nn.Parameter(
            torch.tensor(funding_adjustment, **options)
        )
        self.investment_distribution_logits = nn.Parameter(torch.zeros(13, **options))
        self.funding_distribution_logits = nn.Parameter(torch.zeros(16, **options))

    def forward(self, state: TreasuryPolicyState) -> TreasuryAction:
        investment_scale = torch.relu(
            state.investments[:, 0] + self.investment_scale_adjustment
        )
        funding_scale = torch.relu(state.funding[:, 0] + self.funding_scale_adjustment)
        return TreasuryAction(
            investments=investment_scale.unsqueeze(1)
            * torch.softmax(self.investment_distribution_logits, dim=0),
            funding=funding_scale.unsqueeze(1)
            * torch.softmax(self.funding_distribution_logits, dim=0),
        )


class BMDatePolicy(TreasuryPolicy):
    """BM^D benchmark: independent scale and maturity choices by decision date.

    The simulator has one actionable decision for each transition, followed by a
    terminal passive transition.  BM^D therefore owns exactly ``transitions``
    rows of parameters, not an extra terminal row.
    """

    def __init__(
        self,
        *,
        transitions: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if transitions not in (60, 180):
            raise TreasuryActionError(
                "BM^D supports the resolved 60- or 180-transition horizons"
            )
        options: dict[str, torch.device | str | torch.dtype] = {}
        if device is not None:
            options["device"] = device
        if dtype is not None:
            options["dtype"] = dtype
        self.transitions = transitions
        self.investment_scales = nn.Parameter(torch.ones(transitions, **options))
        self.funding_scales = nn.Parameter(torch.ones(transitions, **options))
        self.investment_distribution_logits = nn.Parameter(
            torch.zeros((transitions, 13), **options)
        )
        self.funding_distribution_logits = nn.Parameter(
            torch.zeros((transitions, 16), **options)
        )

    @property
    def audit_metadata(self) -> dict[str, int | str]:
        """Expose the model's actual parameterization for checkpoint audit."""

        return {
            "decision_dates": self.transitions,
            "investment_maturities_per_date": 13,
            "funding_maturities_per_date": 16,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "indexing_note": (
                "The simulator uses T action dates [0, T-1] and then one passive "
                "terminal transition; BM^D therefore has 60/180 parameter rows, "
                "rather than adding a terminal T+1 action row."
            ),
        }

    def forward(self, state: TreasuryPolicyState) -> TreasuryAction:
        if state.transitions != self.transitions:
            raise TreasuryActionError(
                "BM^D policy horizon must match the simulator decision horizon"
            )
        time = state.time
        investment_scale = torch.relu(self.investment_scales[time])
        funding_scale = torch.relu(self.funding_scales[time])
        return TreasuryAction(
            investments=investment_scale.expand(state.investments.shape[0], 1)
            * torch.softmax(self.investment_distribution_logits[time], dim=0),
            funding=funding_scale.expand(state.funding.shape[0], 1)
            * torch.softmax(self.funding_distribution_logits[time], dim=0),
        )
