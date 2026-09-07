"""No-swap treasury actions and par-bond cash-flow construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from deepalm.runner import OperationalRunError

INVESTMENT_MATURITIES = tuple(range(36, 181, 12))
FUNDING_MATURITIES = (3, *tuple(range(12, 181, 12)))


class TreasuryActionError(OperationalRunError):
    """Raised when an action violates the no-swap treasury contract."""


@dataclass(frozen=True)
class TreasuryAction:
    investments: torch.Tensor
    funding: torch.Tensor

    def __post_init__(self) -> None:
        if self.investments.ndim != 2 or self.investments.shape[1] != 13:
            raise TreasuryActionError("Investment action must have shape [paths, 13]")
        if self.funding.ndim != 2 or self.funding.shape != (self.investments.shape[0], 16):
            raise TreasuryActionError("Funding action must have shape [paths, 16]")
        if not torch.isfinite(self.investments).all() or not torch.isfinite(self.funding).all():
            raise TreasuryActionError("Treasury action must be finite")
        if torch.any(self.investments < 0) or torch.any(self.funding < 0):
            raise TreasuryActionError("Treasury action cannot be negative")

    @property
    def concatenated(self) -> torch.Tensor:
        return torch.cat((self.investments, self.funding), dim=1)


def unit_bond_cash_flows(
    discounts: torch.Tensor, *, maturity_months: int, annual_spread: float
) -> torch.Tensor:
    """Create spread-adjusted semiannual cash flows for a bond issued at par."""

    if discounts.ndim != 2 or discounts.shape[1] != 180 or maturity_months not in range(1, 181):
        raise TreasuryActionError("Bond pricing requires discounts[paths, 180] and a valid maturity")
    if not torch.isfinite(discounts).all() or torch.any(discounts <= 0):
        raise TreasuryActionError("Bond discounts must be finite and positive")
    schedule = torch.zeros((180,), device=discounts.device, dtype=discounts.dtype)
    accruals = torch.zeros_like(schedule)
    dates = list(range(6, maturity_months + 1, 6))
    if maturity_months not in dates:
        dates.append(maturity_months)
    previous_date = 0
    for date in dates:
        schedule[date - 1] = 1.0
        accruals[date - 1] = (date - previous_date) / 12.0
        previous_date = date
    redemption = torch.zeros_like(schedule)
    redemption[maturity_months - 1] = 1.0
    coupon = (1.0 - discounts[:, maturity_months - 1]) / (discounts @ schedule)
    return coupon.unsqueeze(1) * schedule + redemption + annual_spread * accruals


def apply_treasury_action(
    investment_ladder: torch.Tensor,
    funding_ladder: torch.Tensor,
    discounts: torch.Tensor,
    action: TreasuryAction,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add all purchased/issued bond cash flows and return the cash settlement."""

    if investment_ladder.shape != funding_ladder.shape or investment_ladder.shape != discounts.shape:
        raise TreasuryActionError("Ladders and discounts must share shape [paths, 180]")
    if action.investments.shape[0] != discounts.shape[0]:
        raise TreasuryActionError("Action path count must match the market")
    investment_cashflows = torch.stack(
        [
            unit_bond_cash_flows(
                discounts, maturity_months=maturity, annual_spread=-0.0015
            )
            for maturity in INVESTMENT_MATURITIES
        ],
        dim=1,
    )
    funding_cashflows = torch.stack(
        [
            unit_bond_cash_flows(
                discounts, maturity_months=maturity, annual_spread=0.0015
            )
            for maturity in FUNDING_MATURITIES
        ],
        dim=1,
    )
    new_investments = investment_ladder + (action.investments.unsqueeze(2) * investment_cashflows).sum(dim=1)
    new_funding = funding_ladder + (action.funding.unsqueeze(2) * funding_cashflows).sum(dim=1)
    investment_values = (investment_cashflows * discounts.unsqueeze(1)).sum(dim=2)
    funding_values = (funding_cashflows * discounts.unsqueeze(1)).sum(dim=2)
    cash_settlement = (
        (action.funding * funding_values).sum(dim=1)
        - (action.investments * investment_values).sum(dim=1)
    )
    return new_investments, new_funding, cash_settlement
