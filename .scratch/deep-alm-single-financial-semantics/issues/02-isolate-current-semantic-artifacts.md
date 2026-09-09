# 02: 隔离当前金融语义的 artifact 重用

**What to build:** 让训练、恢复、冻结 BM^D baseline、评估和报告只互认当前唯一金融语义的证据，防止任何历史 Learned state 在形状相同的情况下进入当前模型链路。

**Blocked by:** 01 — 建立唯一已修正金融语义契约.

**Status:** ready-for-agent

- [ ] 新生成的 checkpoint、冻结 baseline、恢复状态、评估与报告均带有 `corrected-financial-semantics-v1`，且不再写入可选择的 convention 身份。
- [ ] 任一重用入口对缺失、旧版或不同金融语义版本的输入明确拒绝，不做 metadata-only 升级或权重重解释。
- [ ] 当前版本的同语义 artifact 能够完成既有恢复/冻结引用的公开验证；历史 Paper、paired 与修复前 Corrected artifact 不会被运行时代码加载。
