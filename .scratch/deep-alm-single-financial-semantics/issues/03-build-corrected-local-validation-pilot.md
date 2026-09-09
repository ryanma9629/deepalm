# 03: 交付已修正本机验证试跑

**What to build:** 让本机开发者通过一个 corrected-only 入口运行 BM^D 与 MM、各 5 年和 15 年的完整四任务训练矩阵，并在每个 MM 中使用当次、同期限、同金融语义的冻结 BM^D baseline。

**Blocked by:** 01 — 建立唯一已修正金融语义契约; 02 — 隔离当前金融语义的 artifact 重用.

**Status:** resolved

- [x] `corrected-pilot` 生成且仅生成四个预期训练成员；每个成员在受限本机预算内有四次有限、非零优化信号且参数确实更新，总计 16 次更新。
- [x] 每个 MM 成员只加载同期限、当前版本、内容验证的冻结 BM^D reference，并保持该 baseline 不可训练。
- [x] 原 paired 训练命令、八任务展开规则和 paired 试跑配置不再是公开或可执行的训练入口。

## Answer

2026-09-09：交付 `corrected-pilot` 与 `configs/corrected-pilot.yaml`。该入口只训练 BM^D/MM × 5/15 年四个成员，每个成员四次更新、合计 16 次；每个 MM 在同一次运行内加载同期限冻结 BM^D reference。旧的 paired 训练命令、八任务训练展开和配置已移除；paired 评估与报告路径由 Ticket 04 迁移。已通过 `uv run ruff check src tests`、`uv run python -m compileall -q src tests` 与完整 `uv run pytest -q`（234 passed）。
