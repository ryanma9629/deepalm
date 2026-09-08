# 20: Run the paired Paper-convention experiment

**What to build:** 未来执行同银行、同创新的 BM^D/MM 两期限 Paper/Corrected 配对实验，量化两个口径差异。

**Blocked by:** 17/Lock test evaluation and small-sample statistics.

**Status:** needs-triage

**Execution scope:** opt-in-research

**Specification revision:** 2 (2026-09-07)

本票是延期研究范围记录，不是当前可领取的本机实现票。须显式确认研究环境及资源计划后再激活；24 还需先细拆。下列勾选项是未来研究验收要求，不阻塞 local_flow。

- [ ] The Paper matrix contains BM^D and MM at five and fifteen years plus `MM(15y|5y)` using the primary training seed.
- [ ] Paper and Corrected runs reuse the same canonical Reference Bank and Brownian innovations wherever comparison requires pairing.
- [ ] Convention contract tests prove that only PCA scaling and loan-interest annualization differ; any override resolves to custom and is excluded from locked pairing.
- [ ] The experiment reports performance, constraints, and numerical behavior without treating poor finite Paper outcomes as Corrected acceptance failures.
- [ ] A non-finite Paper result preserves diagnostics and identifies the exact model stage, path, time, and value that failed.
- [ ] Paired outputs clearly label convention, formula choices, seed, paths, horizon, and baseline dependency.
- [ ] 不阻塞本机报告或验收；本机的 Paper 公式覆盖已分配给 25、07、14、23，不以训练这个矩阵为前提。
- [ ] 保留差异边界与数值失败披露，不把有限但较差的 Paper 表现变成 Corrected 失败。
- [ ] This is a deferred research record, not part of local-delivery acceptance or the automatic implementation frontier. Starting it requires explicit research activation and a resolved environment/resource plan; bounded Paper/Corrected formula tests remain mandatory in the active local tickets.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。按用户确认转为延期研究记录，不进入当前 frontier，不再阻塞 22 或本机交付。启用需要明确资源计划。原编号保留；本次只更新待办，不表示本票实现已经完成。

2026-09-08 — 用户确认 MacBook M5 / 32 GiB 的配对口径研究试跑边界：只运行 Paper 与 Corrected 各自的 BM^D/MM、5/15 年（八个训练任务），并由各自 15 年 MM checkpoint 生成零训练更新的 MM(15y|5y)。使用单进程 MPS float32、compact 网络、两 epochs、每 epoch 16 条训练路径、batch 8、16 条 selection 路径、64 条 locked-test 路径和一个主 seed；两口径固定共享 canonical Reference Bank、市场创新路径及种子注册表，但不共享 checkpoint。总训练为 32 次主更新，100 次 paired bootstrap；排除 Paper width、BME/BMC、多 seed、敏感性重训和完整论文图表。总运行预算为 600 秒、RSS/MPS 各 12 GiB；两个 15 年 MM 的首 epoch 后若预测全矩阵纯训练超过 420 秒，则停止未启动任务并按 incomplete 保留诊断。Paper 的有限但较差表现是研究发现，不影响 Corrected；Paper 非有限时保留 stage/path/month/value 诊断并继续独立子任务。此试跑量化公式差异和数值行为，不声明收敛、methodologically-reproduced 或银行模型获批。仍需把本票拆分为可执行子票后才启动实现。
