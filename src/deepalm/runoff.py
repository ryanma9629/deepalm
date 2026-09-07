"""Differentiable passive balance-sheet roll-forward for the Reference Bank."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from deepalm.deposits import (
    DEFAULT_DEPOSIT_CONFIGURATION,
    DepositConfiguration,
    allocate_deposit_tranches,
    cash_penalty,
    deposit_rates,
    monthly_deposit_interest,
    operating_cost,
)
from deepalm.loans import (
    DEFAULT_LOAN_CONFIGURATION,
    LoanConfiguration,
    apply_loan_transition,
)
from deepalm.reference_bank import ReferenceBankProvider, ReferenceBankSnapshot
from deepalm.runner import OperationalRunError
from deepalm.term_structures import MarketScenarioBatch
from deepalm.treasury import TreasuryAction, apply_treasury_action

_ASSET_LADDERS = ("investments", "mortgages", "enterprise_loans")
_LIABILITY_LADDERS = ("non_maturity_deposits", "term_deposits", "funding")
_ALL_LADDERS = _ASSET_LADDERS + _LIABILITY_LADDERS


class RunoffSimulationError(OperationalRunError):
    """Raised when a passive roll-forward cannot produce auditable states."""


@dataclass(frozen=True)
class PassiveRunoffResult:
    """Batch-first economic states for an action-free monthly roll-forward.

    Every state includes the initial state at index zero.  ``settled_cash_flows``
    contains one row per transition: the amount settled from each outstanding
    nominal ladder before that ladder is shifted left by one month.
    """

    cash: torch.Tensor
    assets: torch.Tensor
    liabilities: torch.Tensor
    equity: torch.Tensor
    portfolio_values: Mapping[str, torch.Tensor]
    settled_cash_flows: Mapping[str, torch.Tensor]
    accounting_error: torch.Tensor
    cash_reconciliation_error: torch.Tensor
    loan_originations: torch.Tensor | None = None
    loan_interest_cash_flows: torch.Tensor | None = None
    enterprise_impairment_factors: torch.Tensor | None = None
    deposit_growth: torch.Tensor | None = None
    operating_costs: torch.Tensor | None = None
    cash_penalties: torch.Tensor | None = None
    dividends: torch.Tensor | None = None
    treasury_cash_settlements: torch.Tensor | None = None

    @property
    def paths(self) -> int:
        return int(self.cash.shape[0])

    @property
    def transitions(self) -> int:
        return int(self.cash.shape[1] - 1)


class ALMSimulator:
    """Settle, shift, and revalue the six initial Reference Bank ladders.

    The implementation is intentionally expressed with Torch operations.  It is
    action-free in this ticket, but later treasury actions can use the same state
    update without changing the balance-sheet accounting convention.
    """

    def rollout(
        self,
        snapshot: ReferenceBankSnapshot,
        market: MarketScenarioBatch | object,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
        include_loan_dynamics: bool = False,
        convention: str = "corrected",
        loan_configuration: LoanConfiguration = DEFAULT_LOAN_CONFIGURATION,
        include_deposit_dynamics: bool = False,
        deposit_configuration: DepositConfiguration = DEFAULT_DEPOSIT_CONFIGURATION,
        actions: torch.Tensor | None = None,
    ) -> PassiveRunoffResult:
        """Return all action-free states implied by a market discount-path batch."""

        ReferenceBankProvider().validate(snapshot)
        discounts = _discount_tensor(market, device=device, dtype=dtype)
        paths, states, _tenors = discounts.shape
        transitions = states - 1
        _validate_horizon(market, transitions)
        _require_positive_finite("discount_factors", discounts)
        action_schedule = _action_schedule(actions, paths, transitions, device, dtype)
        spots = (
            _spot_tensor(market, device=device, dtype=dtype, expected_shape=discounts.shape)
            if include_loan_dynamics or include_deposit_dynamics
            else None
        )

        ladders = {
            name: torch.tensor(snapshot.ladders[name], device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(paths, -1)
            .clone()
            for name in _ALL_LADDERS
        }
        cash = torch.empty((paths, states), device=device, dtype=dtype)
        assets = torch.empty_like(cash)
        liabilities = torch.empty_like(cash)
        equity = torch.empty_like(cash)
        accounting_error = torch.zeros_like(cash)
        cash_reconciliation_error = torch.zeros_like(cash)
        values = {
            name: torch.empty((paths, states), device=device, dtype=dtype)
            for name in _ALL_LADDERS
        }
        settlements = {
            name: torch.empty((paths, transitions), device=device, dtype=dtype)
            for name in _ALL_LADDERS
        }
        loan_originations = (
            torch.empty((paths, transitions), device=device, dtype=dtype)
            if include_loan_dynamics
            else None
        )
        loan_interest = (
            torch.empty((paths, transitions), device=device, dtype=dtype)
            if include_loan_dynamics
            else None
        )
        impairments = (
            torch.empty((paths, transitions), device=device, dtype=dtype)
            if include_loan_dynamics
            else None
        )
        deposit_growth = torch.empty((paths, transitions), device=device, dtype=dtype) if include_deposit_dynamics else None
        costs = torch.empty((paths, transitions), device=device, dtype=dtype) if include_deposit_dynamics else None
        penalties = torch.empty((paths, transitions), device=device, dtype=dtype) if include_deposit_dynamics else None
        dividends = torch.zeros((paths, transitions), device=device, dtype=dtype) if include_deposit_dynamics else None
        treasury_cash = torch.zeros((paths, transitions), device=device, dtype=dtype) if action_schedule is not None else None

        cash[:, 0] = snapshot.cash
        if action_schedule is not None:
            ladders["investments"], ladders["funding"], treasury_cash[:, 0] = apply_treasury_action(ladders["investments"], ladders["funding"], discounts[:, 0], _treasury_action_at(action_schedule, 0))
            cash[:, 0] += treasury_cash[:, 0]
        self._revalue_state(ladders, discounts[:, 0], values, 0)
        self._record_balance_sheet(cash, assets, liabilities, equity, values, 0)
        self._validate_state(cash, assets, liabilities, equity, accounting_error, 0)

        for transition in range(transitions):
            asset_cash_movement = sum(
                ladders[name][:, 0] for name in _ASSET_LADDERS
            )
            liability_cash_movement = sum(
                ladders[name][:, 0] for name in _LIABILITY_LADDERS
            )
            cash_movement = asset_cash_movement - liability_cash_movement
            for name, ladder in ladders.items():
                settlements[name][:, transition] = ladder[:, 0]

            reconstructed_asset_settlements = sum(
                settlements[name][:, transition] for name in _ASSET_LADDERS
            )
            reconstructed_liability_settlements = sum(
                settlements[name][:, transition] for name in _LIABILITY_LADDERS
            )
            reconstructed_cash_movement = (
                reconstructed_asset_settlements - reconstructed_liability_settlements
            )
            if include_loan_dynamics:
                assert spots is not None
                loan_event = apply_loan_transition(
                    ladders["mortgages"],
                    ladders["enterprise_loans"],
                    six_month_yield=spots[:, transition + 1, 5],
                    six_month_yield_one_year_ago=(
                        spots[:, transition + 1 - 12, 5]
                        if transition + 1 >= 12 and transition + 1 < transitions
                        else None
                    ),
                    initial_total_loan_value=(
                        snapshot.target_economic_values["mortgages"]
                        + snapshot.target_economic_values["enterprise_loans"]
                    ),
                    convention=convention,
                    annual_close=(transition + 1) % 12 == 0
                    and transition + 1 < transitions,
                    configuration=loan_configuration,
                )
                ladders["mortgages"] = loan_event.mortgages
                ladders["enterprise_loans"] = loan_event.enterprise_loans
                loan_originations[:, transition] = loan_event.originations
                loan_interest[:, transition] = loan_event.interest_cash_flow
                impairments[:, transition] = loan_event.impairment_factor
                cash_movement = (
                    cash_movement
                    - loan_event.originations
                    + loan_event.interest_cash_flow
                )
                reconstructed_cash_movement = (
                    reconstructed_cash_movement
                    - loan_event.originations
                    + loan_event.interest_cash_flow
                )
            if include_deposit_dynamics:
                assert spots is not None
                history = torch.stack(
                    [spots[:, max(0, transition - offset), 5] for offset in (0, 1, 2)], dim=1
                )
                rates = deposit_rates(history, spots[:, transition + 1, 5])
                nmd_growth = ladders["non_maturity_deposits"].sum(dim=1) * deposit_configuration.non_maturity_growth / 12.0
                td_growth = ladders["term_deposits"].sum(dim=1) * deposit_configuration.term_growth / 12.0
                nmd_matured, td_matured = ladders["non_maturity_deposits"][:, 0], ladders["term_deposits"][:, 0]
                nmd_interest = monthly_deposit_interest(rates.non_maturity) * ladders["non_maturity_deposits"].sum(dim=1)
                td_interest = monthly_deposit_interest(rates.term) * ladders["term_deposits"].sum(dim=1)
                ladders["non_maturity_deposits"] = torch.nn.functional.pad(ladders["non_maturity_deposits"][:, 1:], (0, 1)) + allocate_deposit_tranches(nmd_matured + nmd_growth + nmd_interest, deposit_configuration.reference_terms_months, deposit_configuration.non_maturity_weights)
                ladders["term_deposits"] = torch.nn.functional.pad(ladders["term_deposits"][:, 1:], (0, 1)) + allocate_deposit_tranches(td_matured + td_growth + td_interest, deposit_configuration.reference_terms_months, deposit_configuration.term_weights)
                deposit_growth[:, transition] = nmd_growth + td_growth
                costs[:, transition] = operating_cost(personnel_cost=snapshot.personnel_cost_mchf, material_cost=snapshot.material_cost_mchf, completed_years=transition // 12, configuration=deposit_configuration)
                reserves = deposit_configuration.reserve_ratio * (values["non_maturity_deposits"][:, transition] + values["term_deposits"][:, transition])
                penalties[:, transition] = cash_penalty(cash[:, transition], reserves, discounts[:, transition, 0])
                cash_movement = cash_movement + deposit_growth[:, transition] - costs[:, transition] - penalties[:, transition]
                reconstructed_cash_movement = reconstructed_cash_movement + deposit_growth[:, transition] - costs[:, transition] - penalties[:, transition]
            cash[:, transition + 1] = (
                cash[:, transition] + cash_movement
            )
            ladders = {
                name: (
                    ladder
                    if (include_loan_dynamics and name in {"mortgages", "enterprise_loans"}) or (include_deposit_dynamics and name in {"non_maturity_deposits", "term_deposits"})
                    else torch.nn.functional.pad(ladder[:, 1:], (0, 1))
                )
                for name, ladder in ladders.items()
            }
            self._revalue_state(
                ladders, discounts[:, transition + 1], values, transition + 1
            )
            self._record_balance_sheet(
                cash, assets, liabilities, equity, values, transition + 1
            )
            action_settlement = torch.zeros_like(cash[:, transition])
            if action_schedule is not None and transition + 1 < transitions:
                (
                    ladders["investments"],
                    ladders["funding"],
                    treasury_cash[:, transition + 1],
                ) = apply_treasury_action(
                    ladders["investments"],
                    ladders["funding"],
                    discounts[:, transition + 1],
                    _treasury_action_at(action_schedule, transition + 1),
                )
                action_settlement = treasury_cash[:, transition + 1]
                cash[:, transition + 1] += action_settlement
                self._revalue_state(
                    ladders, discounts[:, transition + 1], values, transition + 1
                )
                self._record_balance_sheet(
                    cash, assets, liabilities, equity, values, transition + 1
                )
            dividend = torch.zeros_like(cash[:, transition])
            if include_deposit_dynamics and (transition + 1) % 12 == 0 and transition + 1 < transitions:
                dividends[:, transition] = torch.clamp_min(equity[:, transition + 1] - equity[:, transition + 1 - 12], 0.0) * deposit_configuration.dividend_share
                dividend = dividends[:, transition]
                cash[:, transition + 1] -= dividend
                self._record_balance_sheet(cash, assets, liabilities, equity, values, transition + 1)
            cash_reconciliation_error[:, transition + 1] = (
                cash[:, transition + 1]
                - cash[:, transition]
                - reconstructed_cash_movement
                - action_settlement
                + dividend
            )
            self._validate_state(
                cash,
                assets,
                liabilities,
                equity,
                accounting_error,
                transition + 1,
            )
            _validate_cash_reconciliation(
                cash_reconciliation_error[:, transition + 1],
                assets[:, transition + 1],
                transition + 1,
            )

        return PassiveRunoffResult(
            cash=cash,
            assets=assets,
            liabilities=liabilities,
            equity=equity,
            portfolio_values=MappingProxyType(values),
            settled_cash_flows=MappingProxyType(settlements),
            accounting_error=accounting_error,
            cash_reconciliation_error=cash_reconciliation_error,
            loan_originations=loan_originations,
            loan_interest_cash_flows=loan_interest,
            enterprise_impairment_factors=impairments,
            deposit_growth=deposit_growth,
            operating_costs=costs,
            cash_penalties=penalties,
            dividends=dividends,
            treasury_cash_settlements=treasury_cash,
        )

    @staticmethod
    def _revalue_state(
        ladders: Mapping[str, torch.Tensor],
        discounts: torch.Tensor,
        values: Mapping[str, torch.Tensor],
        time: int,
    ) -> None:
        for name, ladder in ladders.items():
            values[name][:, time] = (ladder * discounts).sum(dim=1)

    @staticmethod
    def _record_balance_sheet(
        cash: torch.Tensor,
        assets: torch.Tensor,
        liabilities: torch.Tensor,
        equity: torch.Tensor,
        values: Mapping[str, torch.Tensor],
        time: int,
    ) -> None:
        assets[:, time] = cash[:, time] + sum(
            values[name][:, time] for name in _ASSET_LADDERS
        )
        liabilities[:, time] = sum(
            values[name][:, time] for name in _LIABILITY_LADDERS
        )
        equity[:, time] = assets[:, time] - liabilities[:, time]

    @staticmethod
    def _validate_state(
        cash: torch.Tensor,
        assets: torch.Tensor,
        liabilities: torch.Tensor,
        equity: torch.Tensor,
        accounting_error: torch.Tensor,
        time: int,
    ) -> None:
        for name, component in {
            "cash": cash[:, time],
            "assets": assets[:, time],
            "liabilities": liabilities[:, time],
            "equity": equity[:, time],
        }.items():
            _require_finite(name, component, time=time)
        accounting_error[:, time] = (
            assets[:, time] - liabilities[:, time] - equity[:, time]
        )
        _validate_accounting(accounting_error[:, time], assets[:, time], time)


def _discount_tensor(
    market: MarketScenarioBatch | object,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    try:
        raw = market.discount_factors
    except AttributeError as error:
        raise RunoffSimulationError(
            "Market paths must expose discount_factors[paths, states, 180]"
        ) from error
    discounts = (
        raw.to(device=device, dtype=dtype)
        if isinstance(raw, torch.Tensor)
        else torch.tensor(raw, device=device, dtype=dtype)
    )
    if discounts.ndim != 3 or discounts.shape[0] == 0 or discounts.shape[2] != 180:
        raise RunoffSimulationError(
            "Market discount_factors must have shape [positive paths, states, 180]",
            diagnostics={"shape": tuple(discounts.shape)},
        )
    if discounts.shape[1] < 2:
        raise RunoffSimulationError(
            "Market discount_factors must provide an initial state and one transition",
            diagnostics={"states": int(discounts.shape[1])},
        )
    return discounts


def _action_schedule(
    actions: torch.Tensor | None,
    paths: int,
    transitions: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if actions is None:
        return None
    schedule = actions.to(device=device, dtype=dtype)
    if schedule.shape != (paths, transitions, 29):
        raise RunoffSimulationError(
            "Treasury actions must have shape [paths, transitions, 29]",
            diagnostics={"shape": tuple(schedule.shape)},
        )
    if not torch.isfinite(schedule).all() or torch.any(schedule < 0):
        raise RunoffSimulationError("Treasury actions must be finite and non-negative")
    return schedule


def _treasury_action_at(schedule: torch.Tensor, time: int) -> TreasuryAction:
    return TreasuryAction(schedule[:, time, :13], schedule[:, time, 13:])


def _spot_tensor(
    market: MarketScenarioBatch | object,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    expected_shape: torch.Size,
) -> torch.Tensor:
    try:
        raw = market.spot_rates
    except AttributeError as error:
        raise RunoffSimulationError(
            "Loan dynamics require market spot_rates[paths, states, 180]"
        ) from error
    spots = raw.to(device=device, dtype=dtype) if isinstance(raw, torch.Tensor) else torch.tensor(raw, device=device, dtype=dtype)
    if spots.shape != expected_shape:
        raise RunoffSimulationError(
            "Market spot_rates must match discount_factors shape for loan dynamics",
            diagnostics={"spot_shape": tuple(spots.shape), "discount_shape": tuple(expected_shape)},
        )
    _require_finite("spot_rates", spots, time=0)
    return spots


def _validate_horizon(market: object, transitions: int) -> None:
    horizon_years = getattr(market, "horizon_years", None)
    if horizon_years is not None and transitions != int(horizon_years) * 12:
        raise RunoffSimulationError(
            "Market horizon does not match its number of monthly transitions",
            diagnostics={
                "horizon_years": horizon_years,
                "transitions": transitions,
            },
        )


def _require_positive_finite(name: str, values: torch.Tensor) -> None:
    invalid = ~torch.isfinite(values) | (values <= 0)
    if invalid.any():
        path, time, tenor = (int(value) for value in invalid.nonzero()[0].tolist())
        raise RunoffSimulationError(
            f"{name} contains a non-finite or non-positive value",
            diagnostics={"path": path, "time": time, "component": f"{name}[{tenor}]"},
        )


def _require_finite(name: str, values: torch.Tensor, *, time: int) -> None:
    invalid = ~torch.isfinite(values)
    if invalid.any():
        path = int(invalid.nonzero()[0].item())
        raise RunoffSimulationError(
            f"{name} becomes non-finite during passive runoff",
            diagnostics={"path": path, "time": time, "component": name},
        )


def _validate_accounting(
    error: torch.Tensor, assets: torch.Tensor, time: int
) -> None:
    _validate_tolerance("accounting", error, assets, time)


def _validate_cash_reconciliation(
    error: torch.Tensor, assets: torch.Tensor, time: int
) -> None:
    _validate_tolerance("cash_reconciliation", error, assets, time)


def _validate_tolerance(
    component: str, error: torch.Tensor, assets: torch.Tensor, time: int
) -> None:
    tolerance = torch.maximum(torch.full_like(assets, 1e-6), assets.abs() * 1e-10)
    failed = error.abs() > tolerance
    if failed.any():
        path = int(failed.nonzero()[0].item())
        raise RunoffSimulationError(
            f"{component} exceeds the accounting tolerance",
            diagnostics={
                "path": path,
                "time": time,
                "component": component,
                "observed_error": float(error[path].detach().cpu()),
                "tolerance": float(tolerance[path].detach().cpu()),
            },
        )


# The public simulator seam starts with passive runoff.  Later tickets add the
# decision-independent customer flows and treasury actions to this same class.
PassiveRunoffSimulator = ALMSimulator
