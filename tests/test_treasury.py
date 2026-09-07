from __future__ import annotations

import pytest
import torch

from deepalm.treasury import (
    TreasuryAction,
    TreasuryActionError,
    apply_treasury_action,
    unit_bond_cash_flows,
)


def test_par_bond_cashflows_and_quantity_weighted_cash_settlement_are_differentiable() -> None:
    discounts = torch.exp(-0.03 * torch.arange(1, 181, dtype=torch.float64).unsqueeze(0) / 12)
    action_values = torch.zeros((1, 13), dtype=torch.float64)
    action_values[0, 0] = 2.0
    action_values.requires_grad_()
    funding_values = torch.zeros((1, 16), dtype=torch.float64)
    funding_values[0, 0] = 3.0
    funding_values.requires_grad_()
    action = TreasuryAction(action_values, funding_values)
    unit = unit_bond_cash_flows(discounts, maturity_months=36, annual_spread=0.0)
    investments, funding, cash = apply_treasury_action(
        torch.zeros_like(discounts), torch.zeros_like(discounts), discounts, action
    )
    loss = investments.sum() + funding.sum() + cash.sum()
    loss.backward()
    assert (unit * discounts).sum().item() == pytest.approx(1.0)
    expected_cash = (
        3.0
        * (unit_bond_cash_flows(discounts, maturity_months=3, annual_spread=0.0015) * discounts).sum()
        - 2.0
        * (unit_bond_cash_flows(discounts, maturity_months=36, annual_spread=-0.0015) * discounts).sum()
    )
    assert cash.item() == pytest.approx(expected_cash.item())
    assert cash.item() != pytest.approx(1.0)
    assert action_values.grad is not None and funding_values.grad is not None


def test_treasury_action_rejects_negative_or_malformed_actions() -> None:
    with pytest.raises(TreasuryActionError, match="negative"):
        TreasuryAction(torch.full((1, 13), -1.0), torch.zeros((1, 16)))
    with pytest.raises(TreasuryActionError, match="shape"):
        TreasuryAction(torch.zeros((1, 12)), torch.zeros((1, 16)))


def test_short_funding_stub_uses_a_quarter_year_of_annualized_spread() -> None:
    discounts = torch.exp(-0.03 * torch.arange(1, 181, dtype=torch.float64).unsqueeze(0) / 12)
    without_spread = unit_bond_cash_flows(
        discounts, maturity_months=3, annual_spread=0.0
    )
    with_spread = unit_bond_cash_flows(
        discounts, maturity_months=3, annual_spread=0.0015
    )

    assert (with_spread - without_spread)[0, 2].item() == pytest.approx(
        0.0015 / 4.0
    )
    assert torch.count_nonzero(with_spread - without_spread).item() == 1
