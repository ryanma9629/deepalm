"""Differentiable passive balance-sheet roll-forward for the Reference Bank."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from deepalm.constraints import ConstraintState, ConstraintValues, evaluate_constraints
from deepalm.deposits import (
    DEFAULT_DEPOSIT_CONFIGURATION,
    DepositConfiguration,
    allocate_reference_term_tranches,
    cash_penalty,
    deposit_rates,
    monthly_deposit_interest,
    operating_cost,
)
from deepalm.loans import (
    DEFAULT_LOAN_CONFIGURATION,
    LoanConfiguration,
    apply_loan_transition,
    initial_loan_cohort_state,
)
from deepalm.objective import ObjectiveParameters, ObjectiveResult, evaluate_objective
from deepalm.policies import TreasuryPolicy, TreasuryPolicyState
from deepalm.reference_bank import (
    ReferenceBankError,
    ReferenceBankProvider,
    ReferenceBankSnapshot,
)
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
    deposit_reference_schedules: Mapping[str, torch.Tensor] | None = None
    deposit_growth: torch.Tensor | None = None
    operating_costs: torch.Tensor | None = None
    cash_penalties: torch.Tensor | None = None
    dividends: torch.Tensor | None = None
    treasury_cash_settlements: torch.Tensor | None = None
    treasury_actions: torch.Tensor | None = None
    initial_constraint_values: torch.Tensor | None = None
    initial_constraint_violations: torch.Tensor | None = None
    constraint_values: torch.Tensor | None = None
    constraint_violations: torch.Tensor | None = None
    constraint_annual_mask: torch.Tensor | None = None
    objective: ObjectiveResult | None = None

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
        loan_configuration: LoanConfiguration = DEFAULT_LOAN_CONFIGURATION,
        include_deposit_dynamics: bool = False,
        deposit_configuration: DepositConfiguration = DEFAULT_DEPOSIT_CONFIGURATION,
        actions: torch.Tensor | None = None,
        policy: TreasuryPolicy | None = None,
        include_constraints: bool = False,
        objective_parameters: ObjectiveParameters | None = None,
    ) -> PassiveRunoffResult:
        """Return all action-free states implied by a market discount-path batch."""

        provider = ReferenceBankProvider()
        provider.validate(snapshot)
        discounts = _discount_tensor(market, device=device, dtype=dtype)
        paths, states, _tenors = discounts.shape
        transitions = states - 1
        _validate_horizon(market, transitions)
        _require_positive_finite("discount_factors", discounts)
        market_curve_identity = getattr(market, "initial_curve_identity", None)
        market_as_of_date = getattr(market, "as_of_date", None)
        if not market_curve_identity or not market_as_of_date:
            raise ReferenceBankError(
                "Reference Bank requires market curve identity and valuation date"
            )
        provider.validate(
            snapshot,
            market_curve_identity=market_curve_identity,
            market_as_of_date=market_as_of_date,
        )
        policy_requires_full_state = policy is not None and bool(
            getattr(policy, "requires_full_state", False)
        )
        if objective_parameters is not None:
            include_constraints = True
        if policy_requires_full_state:
            if objective_parameters is None:
                raise RunoffSimulationError(
                    "A full-state treasury policy requires objective parameters"
                )
            include_constraints = True
        if actions is not None and policy is not None:
            raise RunoffSimulationError("Provide either actions or policy, not both")
        action_schedule = _action_schedule(actions, paths, transitions, device, dtype)
        has_treasury_actions = action_schedule is not None or policy is not None
        spots = (
            _spot_tensor(
                market, device=device, dtype=dtype, expected_shape=discounts.shape
            )
            if (
                include_loan_dynamics
                or include_deposit_dynamics
                or policy_requires_full_state
            )
            else None
        )
        if include_deposit_dynamics and getattr(
            market, "deposit_initial_history_identity", None
        ) != snapshot.deposit_initial_history.identity:
            raise ReferenceBankError(
                "Reference Bank deposit initial-history identity does not match market"
            )

        ladders = {
            name: torch.tensor(snapshot.ladders[name], device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(paths, -1)
            .clone()
            for name in _ALL_LADDERS
        }
        loan_cohorts = (
            {
                name: initial_loan_cohort_state(
                    snapshot.loan_cohorts[name],
                    paths=paths,
                    device=device,
                    dtype=dtype,
                )
                for name in ("mortgages", "enterprise_loans")
            }
            if include_loan_dynamics
            else None
        )
        deposit_reference_schedules = (
            {
                name: torch.tensor(
                    schedule.cash_flows, device=device, dtype=dtype
                )
                .unsqueeze(0)
                .expand(paths, -1, -1)
                .clone()
                for name, schedule in snapshot.deposit_reference_schedules.items()
            }
            if include_deposit_dynamics
            else None
        )
        deposit_initial_history = (
            torch.tensor(
                snapshot.deposit_initial_history.six_month_yields,
                device=device,
                dtype=dtype,
            )
            .unsqueeze(0)
            .expand(paths, -1)
            if include_deposit_dynamics
            else None
        )
        if loan_cohorts is not None:
            for name, cohorts in loan_cohorts.items():
                ladders[name] = cohorts.cash_flows
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
        deposit_growth = (
            torch.empty((paths, transitions), device=device, dtype=dtype)
            if include_deposit_dynamics
            else None
        )
        costs = (
            torch.empty((paths, transitions), device=device, dtype=dtype)
            if include_deposit_dynamics
            else None
        )
        penalties = (
            torch.empty((paths, transitions), device=device, dtype=dtype)
            if include_deposit_dynamics
            else None
        )
        dividends = (
            torch.zeros((paths, transitions), device=device, dtype=dtype)
            if include_deposit_dynamics
            else None
        )
        treasury_cash = (
            torch.zeros((paths, transitions), device=device, dtype=dtype)
            if has_treasury_actions
            else None
        )
        executed_actions: list[torch.Tensor] | None = (
            [] if has_treasury_actions else None
        )
        constraint_steps: list[ConstraintValues] | None = (
            [] if include_constraints else None
        )

        cash[:, 0] = snapshot.cash
        initial_assets = cash[:, 0] + sum(
            (ladders[name] * discounts[:, 0]).sum(dim=1)
            for name in _ASSET_LADDERS
        )
        initial_constraints = (
            evaluate_constraints(
                _constraint_state(cash[:, 0], ladders, discounts[:, 0])
            )
            if include_constraints
            else None
        )
        initial_action = _policy_action_at(
            action_schedule,
            policy,
            ladders,
            time=0,
            transitions=transitions,
            cash=cash[:, 0],
            curve=spots[:, 0]
            if policy_requires_full_state and spots is not None
            else None,
            discounts=discounts[:, 0] if policy_requires_full_state else None,
            initial_assets=initial_assets if policy_requires_full_state else None,
            prior_constraint_values=(
                initial_constraints.values if policy_requires_full_state else None
            ),
            objective_parameters=objective_parameters
            if policy_requires_full_state
            else None,
        )
        if initial_action is not None:
            assert treasury_cash is not None
            assert executed_actions is not None
            ladders["investments"], ladders["funding"], treasury_cash[:, 0] = (
                apply_treasury_action(
                    ladders["investments"],
                    ladders["funding"],
                    discounts[:, 0],
                    initial_action,
                )
            )
            cash[:, 0] += treasury_cash[:, 0]
            executed_actions.append(initial_action.concatenated)
        self._revalue_state(ladders, discounts[:, 0], values, 0)
        self._record_balance_sheet(cash, assets, liabilities, equity, values, 0)
        self._validate_state(cash, assets, liabilities, equity, accounting_error, 0)
        if include_constraints:
            assert constraint_steps is not None
            current_constraints = evaluate_constraints(
                _constraint_state(cash[:, 0], ladders, discounts[:, 0])
            )
            constraint_steps.append(current_constraints)

        for transition in range(transitions):
            asset_cash_movement = sum(ladders[name][:, 0] for name in _ASSET_LADDERS)
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
                assert loan_cohorts is not None
                loan_event = apply_loan_transition(
                    loan_cohorts["mortgages"],
                    loan_cohorts["enterprise_loans"],
                    spot_curve=spots[:, transition + 1],
                    six_month_yield_one_year_ago=(
                        spots[:, transition + 1 - 12, 5]
                        if transition + 1 >= 12
                        else None
                    ),
                    annual_impairment=(transition + 1) % 12 == 0,
                    configuration=loan_configuration,
                )
                loan_cohorts["mortgages"] = loan_event.mortgage_cohorts
                loan_cohorts["enterprise_loans"] = loan_event.enterprise_cohorts
                ladders["mortgages"] = loan_event.mortgages
                ladders["enterprise_loans"] = loan_event.enterprise_loans
                loan_originations[:, transition] = loan_event.originations
                loan_interest[:, transition] = loan_event.interest_cash_flow
                impairments[:, transition] = loan_event.impairment_factor
                cash_movement = (
                    cash_movement
                    - loan_event.originations
                )
                reconstructed_cash_movement = (
                    reconstructed_cash_movement
                    - loan_event.originations
                )
            if include_deposit_dynamics:
                assert spots is not None
                assert deposit_reference_schedules is not None
                assert deposit_initial_history is not None
                if transition == 0:
                    history = torch.cat(
                        (spots[:, :1, 5], deposit_initial_history), dim=1
                    )
                elif transition == 1:
                    history = torch.cat(
                        (
                            spots[:, 1:2, 5],
                            spots[:, :1, 5],
                            deposit_initial_history[:, :1],
                        ),
                        dim=1,
                    )
                else:
                    history = torch.stack(
                        [
                            spots[:, transition - offset, 5]
                            for offset in (0, 1, 2)
                        ],
                        dim=1,
                    )
                rates = deposit_rates(history, spots[:, transition + 1, 5])
                product_growth: dict[str, torch.Tensor] = {}
                product_matured: dict[str, torch.Tensor] = {}
                for name, annual_growth, rate in (
                    (
                        "non_maturity_deposits",
                        deposit_configuration.non_maturity_growth,
                        rates.non_maturity,
                    ),
                    ("term_deposits", deposit_configuration.term_growth, rates.term),
                ):
                    schedule = snapshot.deposit_reference_schedules[name]
                    class_ladders = deposit_reference_schedules[name]
                    class_balances = class_ladders.sum(dim=2)
                    total_balance = class_balances.sum(dim=1)
                    product_growth[name] = total_balance * annual_growth / 12.0
                    growth_by_class = product_growth[name].unsqueeze(1) * torch.tensor(
                        schedule.weights, device=device, dtype=dtype
                    ).unsqueeze(0)
                    matured_by_class = class_ladders[:, :, 0]
                    product_matured[name] = matured_by_class.sum(dim=1)
                    interest_by_class = (
                        monthly_deposit_interest(rate).unsqueeze(1) * class_balances
                    )
                    deposit_reference_schedules[name] = torch.nn.functional.pad(
                        class_ladders[:, :, 1:], (0, 1)
                    ) + allocate_reference_term_tranches(
                        matured_by_class + growth_by_class + interest_by_class,
                        schedule.terms_months,
                    )
                    ladders[name] = deposit_reference_schedules[name].sum(dim=1)
                nmd_growth = product_growth["non_maturity_deposits"]
                td_growth = product_growth["term_deposits"]
                nmd_matured = product_matured["non_maturity_deposits"]
                td_matured = product_matured["term_deposits"]
                deposit_growth[:, transition] = nmd_growth + td_growth
                costs[:, transition] = operating_cost(
                    personnel_cost=snapshot.personnel_cost,
                    material_cost=snapshot.material_cost,
                    completed_years=transition // 12,
                    configuration=deposit_configuration,
                )
                reserves = deposit_configuration.reserve_ratio * (
                    values["non_maturity_deposits"][:, transition]
                    + values["term_deposits"][:, transition]
                )
                penalties[:, transition] = cash_penalty(
                    cash[:, transition], reserves, discounts[:, transition, 0]
                )
                cash_movement = (
                    cash_movement
                    # Maturing deposits are paid out above and immediately renewed.
                    + nmd_matured
                    + td_matured
                    + deposit_growth[:, transition]
                    - costs[:, transition]
                    - penalties[:, transition]
                )
                reconstructed_cash_movement = (
                    reconstructed_cash_movement
                    + settlements["non_maturity_deposits"][:, transition]
                    + settlements["term_deposits"][:, transition]
                    + deposit_growth[:, transition]
                    - costs[:, transition]
                    - penalties[:, transition]
                )
            cash[:, transition + 1] = cash[:, transition] + cash_movement
            ladders = {
                name: (
                    ladder
                    if (
                        include_loan_dynamics
                        and name in {"mortgages", "enterprise_loans"}
                    )
                    or (
                        include_deposit_dynamics
                        and name in {"non_maturity_deposits", "term_deposits"}
                    )
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
            next_action = (
                _policy_action_at(
                    action_schedule,
                    policy,
                    ladders,
                    time=transition + 1,
                    transitions=transitions,
                    cash=cash[:, transition + 1],
                    curve=(
                        spots[:, transition + 1]
                        if policy_requires_full_state and spots is not None
                        else None
                    ),
                    discounts=(
                        discounts[:, transition + 1]
                        if policy_requires_full_state
                        else None
                    ),
                    initial_assets=(initial_assets if policy_requires_full_state else None),
                    prior_constraint_values=(
                        current_constraints.values
                        if policy_requires_full_state
                        else None
                    ),
                    objective_parameters=(
                        objective_parameters if policy_requires_full_state else None
                    ),
                )
                if transition + 1 < transitions
                else None
            )
            if next_action is not None:
                assert treasury_cash is not None
                assert executed_actions is not None
                (
                    ladders["investments"],
                    ladders["funding"],
                    treasury_cash[:, transition + 1],
                ) = apply_treasury_action(
                    ladders["investments"],
                    ladders["funding"],
                    discounts[:, transition + 1],
                    next_action,
                )
                action_settlement = treasury_cash[:, transition + 1]
                cash[:, transition + 1] += action_settlement
                executed_actions.append(next_action.concatenated)
                self._revalue_state(
                    ladders, discounts[:, transition + 1], values, transition + 1
                )
                self._record_balance_sheet(
                    cash, assets, liabilities, equity, values, transition + 1
                )
            if include_constraints and transition + 1 < transitions:
                assert constraint_steps is not None
                annual_close = (transition + 1) % 12 == 0
                current_constraints = evaluate_constraints(
                    _constraint_state(
                        cash[:, transition + 1],
                        ladders,
                        discounts[:, transition + 1],
                    ),
                    previous_annual_equity=(
                        equity[:, transition + 1 - 12].clone() if annual_close else None
                    ),
                )
                constraint_steps.append(current_constraints)
            dividend = torch.zeros_like(cash[:, transition])
            if (
                include_deposit_dynamics
                and (transition + 1) % 12 == 0
                and transition + 1 < transitions
            ):
                dividends[:, transition] = (
                    torch.clamp_min(
                        equity[:, transition + 1] - equity[:, transition + 1 - 12], 0.0
                    )
                    * deposit_configuration.dividend_share
                )
                dividend = dividends[:, transition]
                cash[:, transition + 1] -= dividend
                self._record_balance_sheet(
                    cash, assets, liabilities, equity, values, transition + 1
                )
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

        constraint_values = (
            torch.stack([step.values for step in constraint_steps], dim=1)
            if constraint_steps is not None
            else None
        )
        constraint_violations = (
            torch.stack([step.violations for step in constraint_steps], dim=1)
            if constraint_steps is not None
            else None
        )
        constraint_annual_mask = (
            torch.stack([step.annual_mask for step in constraint_steps], dim=1)
            if constraint_steps is not None
            else None
        )
        if objective_parameters is not None:
            assert constraint_violations is not None
        objective = (
            evaluate_objective(
                torch.full_like(equity[:, 0], snapshot.equity),
                equity[:, -1],
                horizon_years=transitions // 12,
                violations=constraint_violations,
                parameters=objective_parameters,
            )
            if objective_parameters is not None
            else None
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
            deposit_reference_schedules=(
                MappingProxyType(deposit_reference_schedules)
                if deposit_reference_schedules is not None
                else None
            ),
            deposit_growth=deposit_growth,
            operating_costs=costs,
            cash_penalties=penalties,
            dividends=dividends,
            treasury_cash_settlements=treasury_cash,
            treasury_actions=(
                torch.stack(executed_actions, dim=1)
                if executed_actions is not None
                else None
            ),
            initial_constraint_values=(
                initial_constraints.values if initial_constraints is not None else None
            ),
            initial_constraint_violations=(
                initial_constraints.violations
                if initial_constraints is not None
                else None
            ),
            constraint_values=constraint_values,
            constraint_violations=constraint_violations,
            constraint_annual_mask=constraint_annual_mask,
            objective=objective,
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
        liabilities[:, time] = sum(values[name][:, time] for name in _LIABILITY_LADDERS)
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


def _policy_action_at(
    schedule: torch.Tensor | None,
    policy: TreasuryPolicy | None,
    ladders: Mapping[str, torch.Tensor],
    *,
    time: int,
    transitions: int,
    cash: torch.Tensor | None = None,
    curve: torch.Tensor | None = None,
    discounts: torch.Tensor | None = None,
    initial_assets: torch.Tensor | None = None,
    prior_constraint_values: torch.Tensor | None = None,
    objective_parameters: ObjectiveParameters | None = None,
) -> TreasuryAction | None:
    if schedule is not None:
        return _treasury_action_at(schedule, time)
    if policy is None:
        return None
    full_state = _policy_requires_full_state(policy)
    return policy(
        TreasuryPolicyState(
            investments=ladders["investments"],
            funding=ladders["funding"],
            time=time,
            transitions=transitions,
            mortgages=ladders["mortgages"] if full_state else None,
            enterprise_loans=(ladders["enterprise_loans"] if full_state else None),
            non_maturity_deposits=(
                ladders["non_maturity_deposits"] if full_state else None
            ),
            term_deposits=ladders["term_deposits"] if full_state else None,
            cash=cash if full_state else None,
            curve=curve if full_state else None,
            discounts=discounts if full_state else None,
            initial_assets=initial_assets if full_state else None,
            prior_constraint_values=prior_constraint_values if full_state else None,
            mu=(
                objective_parameters.mu if full_state and objective_parameters else None
            ),
            penalty_weight=(
                objective_parameters.penalty_weight
                if full_state and objective_parameters
                else None
            ),
        )
    )


def _policy_requires_full_state(policy: TreasuryPolicy) -> bool:
    """Keep benchmark policy calls on their compact historical state contract."""

    return bool(getattr(policy, "requires_full_state", False))


def _constraint_state(
    cash: torch.Tensor,
    ladders: Mapping[str, torch.Tensor],
    discounts: torch.Tensor,
) -> ConstraintState:
    return ConstraintState(
        cash=cash.clone(),
        investments=ladders["investments"],
        mortgages=ladders["mortgages"],
        enterprise_loans=ladders["enterprise_loans"],
        non_maturity_deposits=ladders["non_maturity_deposits"],
        term_deposits=ladders["term_deposits"],
        funding=ladders["funding"],
        discounts=discounts,
    )


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
    spots = (
        raw.to(device=device, dtype=dtype)
        if isinstance(raw, torch.Tensor)
        else torch.tensor(raw, device=device, dtype=dtype)
    )
    if spots.shape != expected_shape:
        raise RunoffSimulationError(
            "Market spot_rates must match discount_factors shape for loan dynamics",
            diagnostics={
                "spot_shape": tuple(spots.shape),
                "discount_shape": tuple(expected_shape),
            },
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


def _validate_accounting(error: torch.Tensor, assets: torch.Tensor, time: int) -> None:
    _validate_tolerance("accounting", error, assets, time)


def _validate_cash_reconciliation(
    error: torch.Tensor, assets: torch.Tensor, time: int
) -> None:
    _validate_tolerance("cash_reconciliation", error, assets, time)


def _validate_tolerance(
    component: str, error: torch.Tensor, assets: torch.Tensor, time: int
) -> None:
    if assets.dtype in {torch.float16, torch.bfloat16, torch.float32}:
        absolute_tolerance, relative_tolerance = 1e-3, 1e-5
    else:
        absolute_tolerance, relative_tolerance = 1e-6, 1e-10
    tolerance = torch.maximum(
        torch.full_like(assets, absolute_tolerance), assets.abs() * relative_tolerance
    )
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
