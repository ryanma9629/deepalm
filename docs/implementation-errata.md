# 实现版勘误与差异说明

## 定位与阅读方式

本文件是项目当前唯一的“论文 - 实现”映射说明。它把
*Deep treasury management for banks*、[原始勘误审阅](errata.pdf)
（`docs/errata.pdf`）以及实现审计，映射到唯一可执行的
**Corrected Financial Semantics**：`corrected-financial-semantics-v2`。

“差异”必须分清来源：

1. **原文错误**：论文的写法本身存在问题，按原始 errata 修正。
2. **实现错误**：论文（经 errata 修正后）的意图没有问题，但项目早期代码没有正确落实。
3. **明确简化**：论文没有错；为了公开数据可得性、产品范围或交付边界，项目有意采用较小的模型。
4. **能力缺口**：论文没有错；当前本机实现为了跑通技术流程而限制了资源，正式银行训练仍需另行完成。

前两类改变或校正金融/计量行为；后两类不是“发现论文错了”。测试或生成 artifact 的通过，只证明当前代码路径可运行，不代表收敛、论文数值复现或银行模型批准。论文和 errata 是方法与审计来源，不构成可执行的替代公式 profile。

## 一、原文错误：按 errata 采用的修正

### 已采纳修正

| 依据 | 已采纳行为 | 影响边界 | 公开验证 |
| --- | --- | --- | --- |
| Paper PCA construction; errata E-08 | PCA scenario loadings use square-root eigenvalue scaling. | Calibration and every generated interest-rate scenario. | `tests/test_term_structures.py`; calibration evidence. |
| errata E-02; Paper Eq. 9 | 每笔新贷款以发行时同期限收益率加 spread 锁定非负月度 coupon；合成 legacy cohort 采用同一零下限。 | Existing and newly originated loan cash flows. | `tests/test_loans.py`; training and checkpoint semantics. |
| C-1/C-2; Paper Eq. 45 | MM consumes current pre-action economic ratios in the stated order; non-IRS constraint features are shifted by their bounds. | MM observations and actions only. | `tests/test_mm.py`, `tests/test_policies.py`. |
| C-3; Paper Eq. 43 | BM^D action total is the live first maturing bucket plus a learned date adjustment, floored at zero. | BM^D and frozen BM^D used by MM at 5/15 years. | `tests/test_policies.py`, `tests/test_mm_training.py`; frozen baseline references. |
| C-5/C-6; Paper Eq. 11c | Deposit rollover retains each reference-term class; the first two rate windows use dated pre-valuation six-month yields rather than repeated Y0. | Deposit ladder, liability interest, cash and duration path. | `tests/test_deposits.py`, `tests/test_reference_bank.py`; `reference-bank.json`. |
| errata E-03 | Negative-rate cash charge is a non-negative cost. | Monthly cash transition. | `tests/test_deposits.py`. |
| errata E-10 | Select raw-scale tails before centering by the full-sample mean; equity risk uses terminal equity ratios and penalty risk uses the upper tail. | Objective risk terms and evaluation statistics. | `tests/test_evaluation.py`. |
| C-7/C-8/R-1 | Dividend yield uses nonterminal dividend years; constraint reports retain raw violating values and population moments expose unavailable values safely. | Locked evaluation and reports. | `tests/test_evaluation.py`, `locked-evaluation.json`. |
| R-2 | Coverage names every printed Table 1–5 and Figure 3–17 by page and subject; missing evidence remains missing. | Reporting only. | `tests/test_reporting.py`, `paper-coverage-inventory.json`. |

### 原始 errata 中的其他边界

| Errata | 当前处理 | 原因与边界 |
| --- | --- | --- |
| E-01 | 连续复利贴现不额外乘未定义的时间步长。 | 期限按“年”统一；影响债券、贷款和负债估值。 |
| E-04 | Treasury 交易现金按各期限“名义数量 × 当期价值”求和入账。 | 保持现金守恒；`tests/test_runoff.py`。 |
| E-05 | 下限约束惩罚 shortfall，IRS 上限惩罚 excess。 | 影响目标函数和梯度；`tests/test_constraints.py`。 |
| E-06 / E-09 | 初始时点不触发年度关闭；5/15 年分别采用 60/180 个决策区间与 61/181 个状态节点。终点不分红、不计算 EYR，但 terminal roll 仍结算并更新贷款，包括适用的年度企业贷款减值。 | 影响分红、EYR、终值、贷款价值和时间网格；`tests/test_runoff.py`、`tests/test_loans.py`。 |
| E-07 | 不适用。 | 它是互换限额问题；当前范围不含 swaps 或 `MM^S`，不把“未实现”说成“已修正”。 |

## 二、原文正确、但早期代码实现错误：已采纳修正

这些项目来自论文一致性审计，不是新的论文勘误。论文（或经上表 errata 修正后的论文）已经给出应实现的经济逻辑，但早期代码曾有遗漏、错位或报告层错误。

| 审计项 | 当前采用行为 | 为什么不是论文错误 | 影响边界与公开验证 |
| --- | --- | --- | --- |
| C-1 / C-2，Paper Eq. 45 | MM 在动作前读取当期经济比率；非 IRS 约束特征按监管阈值平移，特征顺序固定。 | 是对论文观测定义的代码落实。 | MM 观测和动作；`tests/test_mm.py`、`tests/test_policies.py`。 |
| C-3，Paper Eq. 43 | BM^D 总动作量为当前最先到期 bucket 加学习到的日期调整，并以零为下限。 | 论文所指的是当前到期量，不是初始 bucket。 | BM^D 与 MM 依赖的冻结 baseline；`tests/test_policies.py`、`tests/test_mm_training.py`。 |
| C-5 / C-6，Paper Eq. 11c | 存款 rollover 保留 1/2/12/120 月类别；前两个窗口使用估值日前对应日期的六个月收益率。 | 论文没有要求重复使用初始 `Y0`。 | 存款、现金和久期；`tests/test_deposits.py`、`tests/test_reference_bank.py`。 |
| 现金／贷款转移审计 | 贷款增长从当期、roll 前的名义本金开始计算。 | 这是对状态转移时点的实现修正，不属于 errata E-03 或 Paper Eq. 8。 | 贷款路径；`tests/test_loans.py`。 |
| Paper §2.2.2 | 企业贷款每个年度检查日均按六个月利率年度升幅进行减值，包括终点年度；该贷款动态不继承分红/EYR 的终点豁免。 | 终值权益、目标损失和贷款风险。 | `tests/test_loans.py`。 |
| C-7 / C-8 / R-1 | 股息收益率只平均非终端年度；报告保留原始违规值，并显式标记不可用总体矩。 | 是对定义与可审计报告的落实。 | 锁定评估和报告；`tests/test_evaluation.py`。 |
| R-2 | 覆盖清单逐项标出 Table 1–5、Figure 3–17 的主题和缺失证据。 | 防止把未生成的图表或统计说成已复现。 | 报告层；`tests/test_reporting.py`、`paper-coverage-inventory.json`。 |

## 三、原文未错、但项目有意不同：明确简化

### 明确简化

| 依据 | 保留的简化 | 影响边界 | 公开验证 |
| --- | --- | --- | --- |
| Data availability | The canonical Reference Bank is a transparent substitute for the paper's private bank inputs. | Starting balance sheet and calibration handoff. | `docs/reference-bank-inputs.md`; `tests/test_reference_bank.py`. |
| Product scope | Swaps and MM^S are out of scope. | Policies, state transitions, constraints, and reporting. | Configuration rejects swaps; no-swap workflow tests. |

### 训练与选模规则（项目补充）

论文第 21 页 §3.2.3.3 提到开发时使用 early stopping，并对正式结果模型
重新生成每轮训练情景，但未公开监控指标、patience、最小改善幅度或停止后
恢复哪一轮权重。以下是项目的可复现工程规则，不是论文公式修正。

- 每轮 selection 使用固定的独立情景和固定的 `mu`、`lambda`，监控平均
  total loss；锁定 test 集不参与选模或早停。
- 最佳 checkpoint：total loss 严格下降就保存；total 完全相同时，保留
  penalty 更低的一轮；两者均相同则保留较早一轮。
- 早停：单独维护上次显著改善的 total loss `reference`。只有
  `current < reference - minimum_relative_improvement * max(abs(reference), 1e-12)`
  才清零等待计数；否则增加一次。细微改善可以更新最佳 checkpoint，却不重置
  早停计数。patience 按实际验证次数计数，当前每个 eligible epoch 验证一次。
- `selection_start_epoch` 是首个参与验证选模和早停的 epoch；此前只训练。
  到达 patience 或配置的最大 epoch 就结束。非有限验证 loss 是操作失败，
  不视作正常早停；沿用现有诊断和恢复机制。
- 本机两轮 pilot 的 patience 为 `null`，按完整预算运行。可选 `paper_scale`
  的第 20 轮开始验证、patience 15、相对阈值 0.001（0.1%）均为项目默认值，
  不是作者披露的参数。`bank_training` 在现有配置中显式给出这三个值。
- 最终 `POLICY_Hy.pt` 保存最佳权重；`POLICY_Hy.recovery.pt` 保存最后完成轮、
  与该轮匹配的优化器/scheduler、最佳权重副本及完整早停状态。续训使用后者。
  新 checkpoint 的 `training_summary` 标记最佳轮、完成轮、loss、阈值、等待次数
  和 `max_epochs` / `patience_exhausted` 停止原因。

这遵循 [Lightning EarlyStopping](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html)
与 [ModelCheckpoint](https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/callbacks/model_checkpoint.html)
分开处理停止与保存的思路。Lightning 的 `min_delta` 使用绝对差值；本项目沿用
已有的相对阈值配置，以适应不同期限和参考银行规模的 loss 数量级。

选模协议版本 2 修正了旧逻辑中“相同 loss 在零阈值下被计为改善”及
“微小的最佳 loss 改善被早停阈值挡住”的行为。旧恢复文件缺少新的早停状态，
拒绝在新协议下续训，需建立新 run；已有冻结 checkpoint 和报告仍可读取，
不重新标记为新协议结果。金融语义版本与本机资源预算不变。
回归证据包括恒定 loss、阈值边界、非有限值、间隔验证、真实小样本训练
及早停触发前/触发当轮中断恢复的一致性（`tests/test_training.py`）。

## 四、原文未错、但尚未做到：能力缺口

### 能力缺口

| 依据 | 尚未交付的能力 | 影响边界 | 公开验证 / 下一步 |
| --- | --- | --- | --- |
| Paper network scale | The local compact 64/64/32/32 network is a technical-flow validation, not paper-width convergence evidence. | Local pilot results and claims. | `configs/local-two-policy-m5.yaml`; `tests/test_corrected_pilot.py`. |
| Bank training handoff | Formal paper-width training, realistic sample sizes, and hyperparameter optimisation require real bank data and approved GPU resources. | Any performance or production claim. | [bank commissioning guide](bank-single-gpu-commissioning.md). |
| Deployment | CUDA cluster, multi-GPU/DDP, mixed precision tuning, and bank/regulatory acceptance are not implemented by this repository. | Production readiness. | Bank-owned commissioning and acceptance process. |

### 固定的 MacBook 本机预算

| 参数 | 当前值 | 为什么与正式训练不同 |
| --- | --- | --- |
| 设备与类型 | Apple MPS，`float32`。 | 验证 Apple Silicon 路径，不是 CUDA 集群吞吐或跨设备一致性验证。 |
| 网络 | compact `64/64/32/32`。 | 非 paper-width 网络，不能据此主张网络规模或收敛表现一致。 |
| 训练规模 | 2 epochs；每 epoch 16 条训练路径、16 条 selection 路径；batch size 8；64 条锁定 test 路径。 | 样本和更新次数只够验证训练、冻结 baseline、评估和报告闭环。 |
| 优化矩阵 | BM^D/MM × 5/15 年，共 4 个任务；每任务 4 次、合计 16 次更新。 | 不是多 seed、敏感性或经济优越性实验。 |
| 资源 guard | 总计 600 秒；RSS 与 MPS 各 12 GiB。 | 防止本机技术验证演变为长期训练。 |

本机实际验收中，pilot 用时 151.30 秒，峰值 RSS 约 0.96 GiB、MPS 已分配内存约 0.22 GiB；锁定评估用时 21.04 秒。这证明资源预算内链路可运行，不提供经济收敛证据。

在银行环境扩大实验时，**不能改变**金融修正、`corrected-financial-semantics-v2` 身份、同期限冻结 BM^D baseline 依赖，以及数据/市场身份校验。可以在重新训练后**扩大**网络宽度、epoch、路径数、batch size、随机种子、真实 Reference Bank 输入、CUDA 或多 GPU 执行和超参数搜索。扩大后必须重新生成 checkpoint、冻结 baseline、锁定评估和报告；小型 MacBook checkpoint 不能被解释为正式银行模型权重。

## Current executable lifecycle

The bounded local validation flow uses generic actions. Its complete YAML
configuration is the authority for the Workflow Contract and Execution Profile:

```bash
uv run deepalm plan --config configs/local-two-policy-m5.yaml
uv run deepalm run --config configs/local-two-policy-m5.yaml
uv run deepalm evaluate --config configs/local-two-policy-m5.yaml --source-run artifacts/corrected-local-validation-pilot
uv run deepalm report --config configs/local-two-policy-m5.yaml --source-run artifacts/corrected-local-validation-pilot --evaluation-run artifacts/corrected-local-validation-pilot-evaluation
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
