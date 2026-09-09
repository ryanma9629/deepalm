# 02: 隔离当前金融语义的 artifact 重用

**What to build:** 让训练、恢复、冻结 BM^D baseline、评估和报告只互认当前唯一金融语义的证据，防止任何历史 Learned state 在形状相同的情况下进入当前模型链路。

**Blocked by:** 01 — 建立唯一已修正金融语义契约.

**Status:** resolved

- [x] 新生成的 checkpoint、冻结 baseline、恢复状态、评估与报告均带有 `corrected-financial-semantics-v1`，且不再写入可选择的 convention 身份。
- [x] 任一重用入口对缺失、旧版或不同金融语义版本的输入明确拒绝，不做 metadata-only 升级或权重重解释。
- [x] 当前版本的同语义 artifact 能够完成既有恢复/冻结引用的公开验证；历史 Paper、paired 与修复前 Corrected artifact 不会被运行时代码加载。

## Answer

2026-09-09 — artifact envelope 的金融字段统一为
`financial_semantics_version = corrected-financial-semantics-v1`。因兼容性判断
对整个 envelope 精确匹配，缺失、历史 `financial_version` 字段、较旧或不同版本均
fail-closed；没有迁移、元数据升级或权重重解释路径。checkpoint、恢复状态、评估
与报告沿用既有封装写入路径；冻结 BM^D sidecar 也以该封装记录并在内容哈希验证后
先做语义检查。

验证：`uv run ruff check src tests`、`uv run python -m compileall -q src tests` 和
`uv run pytest -q` 均通过；完整套件为 240 passed（117.35s）。以 `a1c0984` 为
实施前基线的 Standards 与 Spec 双轴复核均无未解决问题；复核新增了 hash-valid、
但金融版本过期的 frozen sidecar 加载拒绝用例。paired 生命周期的退役仍由 03–05
票负责。
