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
