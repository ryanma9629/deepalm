# 22: Generate a compact report and paper coverage inventory

**What to build:** 从已完成的本机证据生成精简可审计报告，显示已覆盖内容、资源消耗与未执行的研究项目。

**Blocked by:** 04/Add the Hull-White scenario comparison; 18/Demonstrate horizon effects and rate scenarios; 19/Evaluate a representative Reference Bank sensitivity.

**Status:** ready-for-agent

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [ ] Swap columns, swap-only figures, swap cash flows, and MM^S are absent from generated outputs.
- [ ] Each table and figure identifies policy, horizon, convention, sample size, units, and source run; semantic artifact tests avoid pixel-level assertions.
- [ ] The run manifest includes resolved configuration, software and device identity, data and paper hashes, bank and calibration hashes, weekly dates, seeds, checkpoints, metrics, and acceptance evidence.
- [ ] Long-end extrapolation diagnostics, PCA evidence, HJM-versus-Hull-White comparison, policy actions, equity, constraints, durations, sensitivities, and representative paths are included.
- [ ] Artifact publication is atomic at run level; failed or incomplete runs retain diagnostics but cannot appear as complete bundles.
- [ ] Caption-number discrepancies in the paper are disclosed and visual similarity is not used as an acceptance test.
- [ ] 包含参考银行/假设、校准/PCA 及用户已接受的拟合偏差、所有本机政策/期限指标、恢复、截断、代表性敏感性、情景与 bootstrap 演示。
- [ ] 使用最多 32 条 5 年路径的 HJM/Hull-White 比较；包含情景多样性、权益、行动/约束轨迹等所需图表，注明样本、架构、口径、单位、seed 与来源。
- [ ] 为无互换 Tables 1–5 / Figures 3–17 提供逐项覆盖清单；未生成的完整论文式产物说明延期原因，不伪装成已复现。完整绘制工作转入 24 的延期范围。
- [ ] 原子写入 run bundle，包含计划/实际任务及更新数、配置与输入身份、资源、产物/延期清单及验收证据；失败保留诊断而非完整成功包。
- [ ] Every generated output identifies architecture and parameter counts as well as policy, horizon, convention, rate/monetary units, sample size, seed and source run. Label the small-sample output a local workflow demonstration, not a reproduction of trained paper results.
- [ ] Record each original component-15% and aggregate-10% cubic-fit reference level as a waived diagnostic together with observed error and waiver provenance; the reference three-component explained-variance threshold remains at least 90%.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。完整论文图表要求改为精简本机报告和逐项覆盖清单；取消对延期 20 的依赖，扩展论文式绘制范围留在 24。原编号保留；本次只更新待办，不表示本票实现已经完成。
