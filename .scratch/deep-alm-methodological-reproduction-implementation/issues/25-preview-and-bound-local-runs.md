# 25: Preview and bound local workflow runs

**What to build:** 用户可先查看完整运行计划，再执行受预算约束的真实市场阶段；默认不会启动大规模训练。

**Blocked by:** 03/Generate HJM-PCA scenarios.

**Status:** completed

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [x] 增加 `local_flow` 默认配置：Corrected，32/32/32 条训练/选择/测试路径，两轮，batch 8，compact 网络；保留显式选择的旧 quick/paper_scale，并要求 bank_training 显式指定规模、架构、设备和限制。
- [x] 运行规模、网络宽度、金融口径和验收目的独立解析、校验、版本化及哈希；设备在生成 manifest 前一次解析为 CUDA、MPS 或 CPU，拒绝不支持的组合。
- [x] 计划列出 8 个主训练任务、64 次主更新、两次 paper-width 检查，以及选择/测试、分析、报告和额外恢复/性能探针的成本。未知训练耗时标为尚未测量，不编造估时。
- [x] 用实际 HJM 阶段演示计时、批量路径生成及预算耗尽后的非零退出与诊断保留；此时不能声称训练已跑通。
- [x] 时间默认 1,800 秒，RSS 和可用的 accelerator allocated memory 各自默认 4 GiB 软限制；分开报告、不相加，缺失计数器标为不可用。检查阶段/批次边界与大分配之前，记录在途工作超时，不承诺硬实时截止。
- [x] CPU 路径身份由 split、epoch、seed、global path index 决定；不同 batch 划分回放一致，5/15 年创新前缀一致。使用注入时钟/内存读数验证守卫，不靠长时间等待或故意耗尽内存。
- [x] Resolve compact widths 64/64/32/32 and paper widths 512/512/256/128 independently from run scale and convention. Preserve the same four 32-feature encoders, 145-feature observation, 64-feature final encoding and 29 actions; parameterized topology is implemented in 14.
- [x] Do not reinterpret old quick/paper_scale values, auto-expand the policy/seed matrix, trigger sensitivity retraining, or implicitly select paper_scale/bank_training during ordinary run, fast-test or resume commands. auto device selection resolves available CUDA, then MPS, then CPU once before hashing.
- [x] The local plan lists one seed for each of four policies at two horizons, eight updates per job and 64 primary updates, two separately counted paper-width checks, and no training for MM(15y|5y). Include profiling, recovery fixtures, evaluations, optional stages and artifact work with their identities and estimates.
- [x] Bound CPU generation and downstream transfer to batches rather than all epochs; expose split/epoch/seed/global-index identity to the later trainer. Selection/test identities remain stable and chunkable, and finite negative/extreme rates must not be clipped.
- [x] Budget exhaustion during this pre-training slice preserves completed market-stage artifacts and marks remaining work incomplete. Epoch checkpoint/replay handling belongs to 21; this slice cannot claim a completed training workflow.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。新增优先切片：运行计划、预算和批量市场回放先于首次训练落地；未知性能先标记未测量。使用追加编号；本次只更新待办，不表示本票实现已经完成。

2026-09-07 — 已完成。增加 local_flow、独立 architecture/resources/purpose 配置及 schema 版本；`plan` 在任何市场/训练工作前输出任务、唯一 policy/horizon seed、宽度、更新量、预算和未测量成本；`preflight` 实际加载 SNB、校准 HJM-PCA，并以 8 个本机有界批次生成 5/15 年情景。预检实例耗时 0.2923 秒，RSS 428,294,144 bytes，accelerator 指标 unavailable，最大曲线 round-trip error 为 0。预算耗尽会留下已完成阶段和 resource snapshot，状态为 incomplete/budget_exhausted，且不可能产生 development-validated。完整测试 36 passed；ruff 通过。此完成不包含 Reference Bank、金融滚动、策略训练、checkpoint 恢复或 CUDA 硬件验收，分别由后续票据覆盖。
