# 13: Train BM^D and freeze the local baselines

**What to build:** 两种期限分别得到按决策时点学习的 benchmark 及可冻结、可追溯的 MM baseline。

**Blocked by:** 11/Train BM^E locally and measure resource use.

**Status:** resolved

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [x] BM^D learns independent investment and funding scales and maturity distributions for every actual decision date in both resolved horizons.
- [x] The policy remains scenario-independent, emits non-negative common-contract actions, and uses the same simulator and optimization machinery as the other benchmarks.
- [x] Actual parameter counts are reported and the paper's indexing mismatch is recorded rather than silently changing the 60/180-transition timeline.
- [x] The selected parameters can be loaded through an immutable baseline reference whose identity is validated before MM use.
- [x] Behavioral tests show time dependence, absence of scenario dependence, and correct per-date maturity availability.
- [x] 保留 60/180 个实际决策日、参数量与论文索引差异记录；默认 local_flow，仅用 selection 选模。
- [x] Both horizons complete genuine local_flow training with one seed each, and selection uses only the fixed selection scenarios. Freeze the selected baseline even if its small-run economic performance is poor; do not introduce an economic-superiority gate.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。以 local_flow 训练两期限并冻结所选 BM^D；不以经济效果决定是否可以作为 MM 依赖。原编号保留；本次只更新待办，不表示本票实现已经完成。

2026-09-07 — 已实现并验证。BMDatePolicy 为每个实际决策日配置独立的投资/融资规模与 13/16 个期限分布：5 年为 60 行、1,860 参数；15 年为 180 行、5,580 参数。checkpoint 的 `policy_metadata` 明确记录模拟器使用 `T` 个动作日（0..T-1）和一个被动终止转换，因此不为终点额外创建 T+1 动作参数行。冻结引用 `BM_D_<horizon>y.baseline.json` 对 checkpoint 内容哈希、期限、配置身份与数据身份进行验证，验证通过后才加载为不可训练策略。

2026-09-07 — 已完成真实 local_flow 训练并固定 selection 结果，无经济优越性门槛。BM^D 的共同动作规模不读取情景路径，两个期限均以 CPU/float64 完成：5 年期 selected epoch 2，前反向更新 0.99 s、selection 0.78 s、峰值 RSS 535,412,736 B；15 年期 selected epoch 2，前反向更新 2.99 s、selection 2.35 s、峰值 RSS 548,241,408 B。产物位于 `artifacts/quick-skeleton/`（gitignore），包括两种期限的 `.pt`、`.profile.json` 与文件名含 SHA-256 的 `.baseline.<hash>.json`。
