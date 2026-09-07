# 11: Train BM^E locally and measure resource use

**What to build:** BM^E 在两种完整期限上真正训练、按选择集挑选并重载 checkpoint，同时产生首次训练资源实测。

**Blocked by:** 10/Integrate constraints and the training objective; 25/Preview and bound local workflow runs.

**Status:** ready-for-agent

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [ ] BM^E uses equal maturity distributions and two learned time-shared scale adjustments while satisfying the common policy contract.
- [ ] The benchmark trains through the differentiable simulator with RAdam, the resolved triangular learning-rate cycle, gradient clipping, and fresh deterministic epoch paths and objective parameters.
- [ ] Checkpoint selection follows selection total loss with the penalty tie-break and records configuration, seed, data, convention, policy, horizon, optimizer, and history identities.
- [ ] 保留等期限分配、两个共享规模调整量、RAdam、四轮三角学习率周期、0.2 梯度裁剪和逐轮更新的路径及 mu/lambda。
- [ ] 默认每个期限 2 轮、每轮 4 次更新；每轮末评估 selection 并以总损失/罚项决定 checkpoint，不要求 20 轮 warm-up。最终测试集在 17 中统一使用，本票不提前消费它。
- [ ] 只保留当前训练轨迹批次和至多一个预取情景批次；保留完整时序计算图，更新后释放。评估无梯度，数值失败保留位置诊断。
- [ ] 测量 60/180 月完整 forward/backward/update 的时间与峰值内存，分离 warm-up、情景、评估和写产物时间；设备同步计时。优先复用主训练更新，额外探针单列预算。
- [ ] CPU 和可用 MPS 验证有限值、裁剪与合适夹具上的真实参数更新；设备不可用记录 not-run。完整恢复语义在 21 验证，不在两张票重复实现。
- [ ] Both horizons use the resolved local_flow defaults: 32 fresh training paths per epoch, 32 selection paths, 32 reserved locked-test paths, two epochs, batch size eight, and one registered seed per policy/horizon. This ticket consumes selection, not final-test results.
- [ ] Training and objective-parameter streams, model initialization, ordering and evaluation identities are recorded separately; each selected checkpoint can reproduce evaluation without retraining.
- [ ] Keep the agreed RAdam defaults with zero weight decay, learning-rate bounds 5e-4 and 5e-3 over a four-epoch per-update cycle with momentum cycling disabled, and global gradient clipping at 0.2. Do not shorten financial horizons or truncate action-dependent temporal gradients to meet the resource budget.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。原 quick 完整训练要求替换为 local_flow；首次训练即实测资源，完整恢复验证移入提前的 21，最终测试统一留给 17。原编号保留；本次只更新待办，不表示本票实现已经完成。
