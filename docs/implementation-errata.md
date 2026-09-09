# Implementation Errata

## Authority and scope

This is the single current mapping from *Deep treasury management for banks*,
the [original errata](errata.pdf) (`docs/errata.pdf`), and the project audit to the executable
**Corrected Financial Semantics**. The runtime identity is
`corrected-financial-semantics-v1`.

The paper and its errata remain methodological and audit sources. They do not
create an executable alternative formula profile. Historic paired artifacts and
diagnostic records stay where they were written, but no command loads, upgrades,
or promotes them.

Each item below records its source, adopted behaviour, classification, impact
boundary, and public verification. “Public verification” means a repository
test or generated artifact; it is not a claim of convergence, paper-result
replication, or bank-model approval.

## 已采纳修正

| 依据 | 已采纳行为 | 影响边界 | 公开验证 |
| --- | --- | --- | --- |
| Paper PCA construction; errata E-08 | PCA scenario loadings use square-root eigenvalue scaling. | Calibration and every generated interest-rate scenario. | `tests/test_term_structures.py`; calibration evidence. |
| errata E-02 | Loan coupon interest is converted to a monthly rate before monthly cash settlement. | Existing and newly originated loan cash flows. | `tests/test_loans.py`; training and checkpoint semantics. |
| C-1/C-2; Paper Eq. 45 | MM consumes current pre-action economic ratios in the stated order; non-IRS constraint features are shifted by their bounds. | MM observations and actions only. | `tests/test_mm.py`, `tests/test_policies.py`. |
| C-3; Paper Eq. 43 | BM^D action total is the live first maturing bucket plus a learned date adjustment, floored at zero. | BM^D and frozen BM^D used by MM at 5/15 years. | `tests/test_policies.py`, `tests/test_mm_training.py`; frozen baseline references. |
| C-5/C-6; Paper Eq. 11c | Deposit rollover retains each reference-term class; the first two rate windows use dated pre-valuation six-month yields rather than repeated Y0. | Deposit ladder, liability interest, cash and duration path. | `tests/test_deposits.py`, `tests/test_reference_bank.py`; `reference-bank.json`. |
| errata E-03; Paper Eq. 8 | Negative-rate cash charge is a non-negative cost; loan growth begins from the current pre-roll nominal balance. | Monthly cash and loan transition. | `tests/test_deposits.py`, `tests/test_loans.py`. |
| errata E-10 | Select raw-scale tails before centering by the full-sample mean; equity risk uses terminal equity ratios and penalty risk uses the upper tail. | Objective risk terms and evaluation statistics. | `tests/test_evaluation.py`. |
| C-7/C-8/R-1 | Dividend yield uses nonterminal dividend years; constraint reports retain raw violating values and population moments expose unavailable values safely. | Locked evaluation and reports. | `tests/test_evaluation.py`, `locked-evaluation.json`. |
| R-2 | Coverage names every printed Table 1–5 and Figure 3–17 by page and subject; missing evidence remains missing. | Reporting only. | `tests/test_reporting.py`, `paper-coverage-inventory.json`. |

## 明确简化

| 依据 | 保留的简化 | 影响边界 | 公开验证 |
| --- | --- | --- | --- |
| Paper Eq. 9; available public inputs | New loans use one shared current six-month yield across maturities, rather than a maturity-specific origination curve. | New loan cohorts and future interest cash flows. | `tests/test_loans.py`; this document. |
| Data availability | The canonical Reference Bank is a transparent substitute for the paper's private bank inputs. | Starting balance sheet and calibration handoff. | `docs/reference-bank-inputs.md`; `tests/test_reference_bank.py`. |
| Product scope | Swaps and MM^S are out of scope. | Policies, state transitions, constraints, and reporting. | Configuration rejects swaps; no-swap workflow tests. |

## 能力缺口

| 依据 | 尚未交付的能力 | 影响边界 | 公开验证 / 下一步 |
| --- | --- | --- | --- |
| Paper network scale | The local compact 64/64/32/32 network is a technical-flow validation, not paper-width convergence evidence. | Local pilot results and claims. | `configs/corrected-pilot.yaml`; `tests/test_corrected_pilot.py`. |
| Bank training handoff | Formal paper-width training, realistic sample sizes, and hyperparameter optimisation require real bank data and approved GPU resources. | Any performance or production claim. | [bank commissioning guide](bank-single-gpu-commissioning.md). |
| Deployment | CUDA cluster, multi-GPU/DDP, mixed precision tuning, and bank/regulatory acceptance are not implemented by this repository. | Production readiness. | Bank-owned commissioning and acceptance process. |

## Current executable lifecycle

The bounded local validation flow has exactly three dedicated commands:

```bash
uv run deepalm corrected-pilot --config configs/corrected-pilot.yaml
uv run deepalm corrected-evaluate --config configs/corrected-pilot.yaml --source-run artifacts/corrected-local-validation-pilot
uv run deepalm corrected-report --config configs/corrected-pilot.yaml --source-run artifacts/corrected-local-validation-pilot --evaluation-run artifacts/corrected-local-validation-pilot-evaluation
```

It trains BM^D and MM at 5 and 15 years, requiring content-linked frozen BM^D
references for MM. Evaluation and reporting accept only the complete,
current-semantic four-member evidence chain. The pilot is bounded to a MacBook
workflow check and remains distinct from economic acceptance.

## Provenance terminology

The source PDF and its hash remain required input provenance. “Paper-width”
describes a network-scale and commissioning target, while the paper coverage
inventory describes disclosed Tables/Figures. Neither phrase denotes a runnable
financial convention.
