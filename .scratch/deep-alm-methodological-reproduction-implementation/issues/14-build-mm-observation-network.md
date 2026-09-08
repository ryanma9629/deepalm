# 14: Validate MM observations and configurable network widths

**What to build:** 同一个 MM 实现可按 compact 或 paper 宽度，从银行状态与冻结 BM^D 产生合法行动并通过梯度路径测试。

**Blocked by:** 13/Train BM^D and freeze the local baselines.

**Status:** resolved

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [x] Yield-curve features use a centered, non-standardized three-component PCA fitted only on the approved deterministic subset of training states.
- [x] Four independent cash-flow encoders transform investment, funding, aggregate-loan, and aggregate-deposit ladders into 128 features.
- [x] The observation contains exactly 145 features: portfolio encodings, curve factors, five relative balance-sheet values, six prior constraints, normalized time, `mu`, and `lambda`.
- [x] The shared ELU residual network uses the approved widths and 64-dimensional final encoding without batch normalization.
- [x] Investment and funding heads each produce a scale deviation and softmax maturity distribution, add the frozen BM^D baseline, and apply final non-negativity.
- [x] Stop-gradient preserves forward values, blocks gradient into the current observation state, preserves parameter gradients, and leaves future action-to-loss gradients intact.
- [x] Observation, action, baseline, device, dtype, and shape contracts fail early on incompatible inputs.
- [x] 保留四个独立 32 维编码器、145 维观测、64 维最终编码、29 维行动、ELU residual topology 与跨时点共享参数；compact 为 64/64/32/32，paper 为 512/512/256/128。
- [x] 曲线特征 PCA 仅使用登记的 training states，保存中心、投影、抽样索引及身份，不使用 selection/test 拟合。
- [x] 验证 stop-gradient 前向不变、阻断当前观测历史的输入梯度、保留策略参数和行动经未来转移到损失的梯度。容量变化不改变金融口径。
- [x] Divide each portfolio ladder by 100 before its independent encoder; the centered, non-standardized curve-feature PCA uses all training states up to 50,000, or the specified deterministic path/time-stratified subset above that limit. Persist preprocessing and reject mismatched data/calibration.
- [x] Apply stop-gradient at the agreed complete policy-observation boundary; neither compact widths nor CPU/MPS/CUDA selection may alter that boundary, the shared-time architecture, or the financial transition formulas.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。单一实现支持 compact/paper 宽度；保留观测、共享参数、baseline 与梯度语义，不构建第二套简化金融模型。原编号保留；本次只更新待办，不表示本票实现已经完成。

2026-09-08 — 已实现并验证。新增单一 MMPolicy：强制从内容寻址的冻结 BM^D reference 加载 baseline，使用四个独立 32 维编码器、145 维完整观测、ELU residual 共享网络和 64 维最终编码；compact/paper 宽度均受契约锁定。CurveFeaturePCA 仅接收 RegisteredTrainingCurves，50,000 条以下全量拟合，以上按 time 分层并在层内确定性抽样，且持久化中心、投影、抽样索引、数据/校准/training-state identity。停止梯度只截断完整观测的状态输入，不截断编码器或后续参数；测试覆盖行动经债券转换、未来曲线重估和完整 60 时点滚动终期权益的反向路径。全量验证：ruff 通过，96 passed；规格与规范双轴复审无 P1/P2。
