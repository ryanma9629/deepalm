# Deep ALM

[English version](README.md)

Deep ALM 是对论文 *Deep treasury management for banks* 的 PyTorch
实现与方法论复现项目。项目以瑞士国家银行的公开收益率曲线数据和一个透明的
Reference Bank，替代论文不可获得的私有银行输入。

## 先选择目标路径

| 目标 | 起点 | 跑通后能证明什么 | 不能证明什么 |
| --- | --- | --- | --- |
| 1. 在 MacBook 上用小资源跑通全流程 | `configs/local-two-policy-m5.yaml` | 校准、情景生成、Reference Bank、5/15 年 BM^D/MM 训练、冻结 baseline、评估与报告能完整衔接。 | 收敛、经济效果优越、论文数值复现或银行模型批准。 |
| 2. 尽量贴近论文设定 | `configs/paper-oriented-research-plan.yaml`；先用公开数据，再迁移银行数据 | 配置层可以表达并审计论文尺度的资源与网络目标。 | 今天就能完成端到端论文尺度复现；正式研究训练能力尚未交付。 |
| 3. 接入银行真实数据与 GPU | `configs/bank-single-gpu-commissioning.yaml` 和导入的 Reference Bank 快照 | 可以验证同一套代码在单张 CUDA 卡上运行，并校验银行输入合同。 | 生产就绪、多 GPU、或直接授权在银行数据上训练。 |

三条路径应逐层推进。不要把 MacBook 的 compact checkpoint 当成论文尺度或银行模型的初始权重：网络宽度、数据身份、市场身份或金融口径任一发生变化，都必须重新训练。

## 准备环境

需要 Python 3.11+ 与 [uv](https://docs.astral.sh/uv/)。仓库已经包含各示例配置引用的公开市场 CSV 和论文 PDF。

```bash
uv sync --locked
uv run deepalm --help
```

所有命令都会在 `artifacts/` 下输出可审计 run bundle。其 `manifest.json`
记录解析后的配置、输入哈希、Git 版本、设备/运行时、随机种子和资源测量。

命令表示要执行的动作；YAML 配置才是实验的唯一完整定义。配置中的
**Workflow Contract** 决定策略矩阵、期限和证据规则，**Execution Profile**
决定设备与资源边界。文件名只用于说明场景，不能决定运行行为。通用生命周期为
`plan`、`run`、`evaluate`、`report`。

## 1. MacBook：以小资源跑通完整技术流程

当你的目标只是确认技术链路正确接通、而不是得到有经济意义的策略时，使用这条路径。对于本项目约定的 Apple Silicon MacBook，这是推荐的第一步。

锁定的 `corrected_pilot` 配置采用 MPS/`float32`、compact
`64/64/32/32` 网络、2 个 epoch、每个 epoch 各 16 条训练和 selection 路径、64 条锁定测试路径、batch size 8、600 秒总时限，以及 12 GiB 的 RSS/MPS guard。它训练以下四个成员：

```
BM^D：5 年     MM：5 年
BM^D：15 年    MM：15 年
```

每个 MM 都使用同期限、已冻结的 BM^D baseline。当前范围不含 swaps 和 `MM^S`。

```bash
uv run deepalm plan --config configs/local-two-policy-m5.yaml

uv run deepalm run --config configs/local-two-policy-m5.yaml

uv run deepalm evaluate \
  --config configs/local-two-policy-m5.yaml \
  --source-run artifacts/corrected-local-validation-pilot

uv run deepalm report \
  --config configs/local-two-policy-m5.yaml \
  --source-run artifacts/corrected-local-validation-pilot \
  --evaluation-run artifacts/corrected-local-validation-pilot-evaluation
```

`run` 默认输出每个成员的开始/结束信息，以及每个已完成 epoch 的策略、期限、训练损失、优化更新次数；发生 selection 时还会显示其 total loss 和 penalty loss。进度写入 stderr，stdout 始终只包含最终 artifact 目录；可用 `--no-verbose` 关闭进度。

`evaluate` 默认也会向 stderr 输出进度：来源产物校验、市场校准、锁定 checkpoint 与测试路径数量，以及每个已完成 checkpoint 的一行状态。stdout 始终只包含最终 evaluation 目录；可用 `evaluate --no-verbose` 关闭进度。

默认情况下，`run` 不会替换已有的 artifact 目录。若确认要在新运行成功完成后丢弃并替换当前配置所指定的同名目录，可显式增加 `--overwrite`：

```bash
uv run deepalm run --config configs/local-four-policy-m5.yaml --overwrite
```

本机两轮试跑按预算跑完，关闭 early stopping。每个 epoch 后，模型在固定、独立的
selection 集上测量；锁定 test 集绝不参与选模。最终 `.pt` 使用 selection total loss
最低的 epoch；total loss 完全相同时选择 penalty loss 更低的一轮；两者都相同时保留较早
的一轮。`.recovery.pt` 则保留最后完成 epoch、对应的优化器/scheduler 状态，以及最佳策略
权重副本，供后续安全续训。

长训练通过 `run_scale.selection_start_epoch`、
`run_scale.early_stopping_patience` 和
`run_scale.minimum_relative_improvement`（本项目的相对 `min_delta`）配置。patience
统计 selection total loss 没有足够相对下降的验证次数。轻微下降仍可替换最佳模型，
但不会重置 patience；patience 耗尽时交付保存的最佳策略，而非最后完成 epoch 的策略。
checkpoint 的 `training_summary` 记录选中 epoch、最后完成 epoch、loss、配置值以及
`max_epochs` 或 `patience_exhausted` 停止原因。固定 profile 不接受这些字段的覆盖；
银行环境如需自行配置，应使用 `run_scale.profile: bank_training` 并显式填写三个字段。
具体选模、阈值与 patience 是项目补充规则，论文未公开这些细节，见
[实现版勘误中的训练说明](docs/implementation-errata.md#训练与选模规则项目补充)。

检查各 run bundle 的 `manifest.json` 和生成的报告 JSON。本机验收的标准是：四个成员的证据链完整、身份关联正确、优化更新和评估结果均为有限值；它不是经济验收。目标 M5 MacBook 上的一次实测中，pilot 用时 151.30 秒，随后锁定评估用时 21.04 秒；这只是容量规划观察值，不是性能承诺。

### 同一台 MacBook 上的四策略初步比较

如果需要让 BM^E、BM^C、BM^D、MM 在相同条件下得到一轮初步比较，使用独立的四策略试跑。它保持 MPS/`float32`、compact 网络、600 秒和 12GiB guard 不变，但会运行 8 个任务、共 32 次更新。目标 M5 MacBook 上预计约 5--7 分钟；仍保留 10 分钟上限，因为它只是有界的技术比较，不是收敛实验。

```bash
uv run deepalm plan --config configs/local-four-policy-m5.yaml

uv run deepalm run --config configs/local-four-policy-m5.yaml

uv run deepalm evaluate \
  --config configs/local-four-policy-m5.yaml \
  --source-run artifacts/four-policy-corrected-local-validation-pilot

uv run deepalm report \
  --config configs/local-four-policy-m5.yaml \
  --source-run artifacts/four-policy-corrected-local-validation-pilot \
  --evaluation-run artifacts/four-policy-corrected-local-validation-pilot-evaluation
```

报告会在同一批锁定测试情景下列出 8 个“策略 × 期限”成员。年化收益、综合约束惩罚和权益风险只能作为点估计比较；单一随机种子、64 条测试路径和 2 个 epoch 不能证明某个 TreasuryPolicy 在经济上更好。

## 2. 研究路径：尽可能贴近论文设定

当前配置模型中，最接近论文的组合为：

| 设置 | 论文导向取值 |
| --- | --- |
| `run_scale.profile` | `paper_scale`：100 epochs；每 epoch 40,000 条训练路径；1,600 条 selection 路径；1,600 条测试路径；batch size 32；第 20 个 epoch 才开始 selection。 |
| `architecture.profile` | `paper`：`512/512/256/128`。 |
| `experiment.horizons_years` | `[5, 15]`。 |
| `policy.names` | `[BM^E, BM^C, BM^D, MM]`。 |
| `acceptance` | `purpose: research`，`required_status: methodologically-reproduced`。 |

从 `configs/paper-oriented-research-plan.yaml` 开始，并在版本控制之外复制一份，再修改设备、输出目录、资源上限和新的 run 名称。先执行 `plan`；该命令只解析和展示配置，不会生成情景或开始训练：

```bash
uv run deepalm plan --config configs/paper-oriented-research-plan.yaml
```

这里有一个必须明确的当前边界：`paper-oriented-research-plan` 是可规划的合同，不是已交付的正式训练能力。今天执行通用 `run` 最多只能写出最低限度的审计 bundle；它不会训练论文尺度的策略矩阵，更不会生成“方法论已复现”的证据。要完成真正的论文导向训练，需要先交付 research training 能力，并在选定的计算环境上从头训练。

即使 research runner 完成，公开输入也无法逐数复现论文：论文使用私有银行数据；本项目不含 swaps/`MM^S`；新发贷款目前以统一的六个月收益率定价，而不是按期限使用完整的发放收益率曲线。因此可作的研究主张应是“在明确替代假设下的方法论比较”，而不是与论文图表逐值一致。

## 3. 银行路径：真实数据与 GPU 配置

先做单 GPU 工程验证，不要直接提交大规模训练：

```bash
uv run deepalm plan --config configs/bank-single-gpu-commissioning.yaml
uv run deepalm device-check --config configs/bank-single-gpu-commissioning.yaml --horizon 5
uv run deepalm device-check --config configs/bank-single-gpu-commissioning.yaml --horizon 15
uv run deepalm device-check --config configs/bank-single-gpu-commissioning.yaml --policy MM --horizon 5
uv run deepalm device-check --config configs/bank-single-gpu-commissioning.yaml --policy MM --horizon 15
```

该配置请求 CUDA/`float32`，但有意只使用 2 条路径和 1 个 epoch。它验证的是可移植 PyTorch 路径和 checkpoint 恢复，不是训练。

### 真实银行训练前应做的数据与配置调整

1. **制作导入型 Reference Bank 快照，并在配置中选择它。** 按照 [Reference Bank 输入合同](docs/reference-bank-inputs.md) 准备：估值日和市场身份、六组 180 月合同梯、固定利率贷款 cohort、四种存款参考期限类别、有日期的初始存款利率历史、产品假设、单位和来源信息。在银行受控的 `configs/bank-single-gpu-commissioning.yaml` 副本中，将 `reference_bank.snapshot_path` 设为已校验 JSON；它会记录在解析后的配置和产物 manifest 中。随后独立校验快照：

   ```bash
   uv run deepalm reference-bank \
     --config configs/bank-single-gpu-commissioning.yaml
   ```

2. **成对替换市场输入。** 导入快照中的 `as_of_date` 与
   `initial_curve_identity` 必须和历史市场数据及每个生成的情景批次一致。曲线或预处理身份一旦改变，旧 checkpoint 就失效。

3. **建立银行训练配置。** 使用 `run_scale.profile: bank_training`，显式填写 `epochs`、训练/selection/test 路径数、batch size、selection 起始 epoch、early-stopping patience 和最小相对改进。初始使用 `optimization.device: cuda` 和 `float32`；输出目录使用银行受控路径；总时限、RSS 与 GPU 内存上限应来自 profiling，而不是复制 MacBook 的数值。

4. **经 profiling 和审批后再放大。** 网络宽度、路径数、batch size、随机种子数、数据或市场发生变化后，都从新初始化的权重重新训练。始终保留修正后的金融语义，以及 MM 对同期限冻结 BM^D baseline 的依赖；每个新实验身份都要重新生成 checkpoint、baseline、锁定评估和报告。

5. **将银行运行物放在仓库之外。** 私有源数据、凭据与结果应放在银行控制的存储中。银行自己的交付流程还应包括：数据映射签字、账务与经济价值核对、GPU/驱动/PyTorch 版本记录、模型风险审查与验收标准。

当前仓库已经能通过 `deepalm reference-bank` 校验导入快照，并能完成单张 CUDA 卡的 commissioning；但“导入快照接入完整策略训练”、多 GPU/DDP、混合精度调优、集群调度及银行/监管验收，仍是明确列出的后续交付项。面向未来集群实现的并行边界、可复现性规则和银行侧 rollout，见[分布式训练设计（规划中）](docs/distributed-training-design.md)；当前目标端的步骤见[单 GPU commissioning 指南](docs/bank-single-gpu-commissioning.md)。

## 延伸阅读

- [实现版勘误与实现边界](docs/implementation-errata.md)
- [Reference Bank 输入合同](docs/reference-bank-inputs.md)
- [银行单 GPU commissioning 指南](docs/bank-single-gpu-commissioning.md)
- [分布式训练设计（规划中）](docs/distributed-training-design.md)
- [论文 PDF](docs/Deep%20treasury%20management%20for%20banks.pdf)
