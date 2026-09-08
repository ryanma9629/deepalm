# 21: Bound training interruptions and verify recovery

**What to build:** 一次小型真实训练可以在预算耗尽/取消后保存一致进度，重启恢复并与未中断的 CPU 对照一致。

**Blocked by:** 11/Train BM^E locally and measure resource use.

**Status:** resolved

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [x] 保存最后完成 epoch 的模型、优化器、scheduler、选择历史、数据/配置/seed 身份；重放被中断的 epoch，不丢失或重复已经完成的 epoch。
- [x] 中断首轮且无完整 checkpoint 时明确从初始登记状态重放，不伪造 checkpoint；后续 MM 的 preprocessing 与冻结 BM^D 依赖使用相同恢复契约。
- [x] 校验语义兼容性，拒绝更换宽度、数据、口径、期限的续训；设备、输出位置、增加资源预算等允许覆盖须单独记录。
- [x] 短夹具覆盖完整学习率升降周期、early stopping 配置与恢复；保留 quick/paper_scale 历史参数及 bank_training 显式配置能力，不执行 40,000 路径矩阵。
- [x] 超预算、OOM、取消和操作失败可区分；保留产物、未完成项及实际资源，非零退出且不给出 development-validated。
- [x] CPU synthetic interruption/resume fixtures compare an uninterrupted run with epoch-boundary recovery, including optimizer/scheduler, selection history and deterministic scenario/objective streams. Record measured evidence rather than accepting a caller-supplied success flag.
- [x] Preserve opt-in quick at 256/128/128 paths, five epochs and batch 32, and paper_scale at 40,000/1,600/1,600 paths, at most 100 epochs and batch 32, with paper widths. Default local_flow keeps its two-epoch per-epoch selection and requires no statistical early stopping.
- [x] Short synthetic selection-history fixtures verify paper_scale selection after epoch 20, patience 15 and 0.1% relative improvement, including penalty-loss tie-breaks. Bank-training limits/early stopping remain explicitly configured. No full research matrix is executed to test these control paths.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。从原纸面规模矩阵执行中抽取有界训练恢复能力并提前到 11 之后；研究参数保留可配置，三 seed 和大矩阵执行范围移入 24。原编号保留；本次只更新待办，不表示本票实现已经完成。

2026-09-08 — 已实现并验证。每个完整 epoch 都保存可恢复训练状态；`TrainingControl` 支持安全 epoch 边界停止、同输出目录恢复，以及从显式恢复路径恢复。恢复身份锁定 policy、期限、语义配置、数据与校准身份；设备/输出可覆盖，三个资源上限只能增加。预算耗尽、OOM、取消和操作失败分别写入 `.interruption.json`，并以非零的 `TrainingInterrupted` 结束。

2026-09-08 — CPU 夹具验证 4 个更新的未中断训练与 epoch 2 中断后恢复完全一致，覆盖 scheduler、selection history、学习率与模型状态；paper_scale selection 从 epoch 20 开始，patience 15、0.1% 相对改进及 penalty tie-break 均有确定性测试。真实 CPU/local_flow 证据在 `artifacts/ticket21-recovery/`（gitignore）：epoch 1 中断后从 `BM_E_5y.recovery.pt` 恢复，最终完成 8 个更新并在 epoch 2 选模。
