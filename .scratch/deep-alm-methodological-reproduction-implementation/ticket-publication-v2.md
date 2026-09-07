# Ticket publication — Revision 2

Type: publication-record
Status: published
Date: 2026-09-07

用户已确认增量拆分。正式任务现已逐张发布到本地 Markdown tracker，后续以独立票据的内容、状态和依赖为准。本记录是发布时快照，不自动跟随实现进度变化。

来源：[已确认的拆分草案](ticket-revision-v2-proposal.md)；[Revision 2 规格](../deep-alm-methodological-reproduction/spec.md)。草案保留确认前原文，本记录补记用户确认与正式发布结果；规格和 Wayfinder map 本次未改动。

## 发布结果

- 共 27 张正式票据：4 张已完成、21 张本机待实现、2 张延期研究记录。
- 01 经核对既有提交和当前行为测试后补记 completed；只表示原审计骨架切片完成，不表示新的完整流程验收通过。
- 02–04 的内容与完成历史逐字保持不变。
- 05–10 保留金融建模要求；11–19、21–23 调整本机规模、资源、恢复、分析和验收要求。
- 新增 25–27；旧票编号和文件名保留。新依赖可能指向更大的编号，执行顺序由依赖而非编号决定。
- 20、24 为 needs-triage / opt-in-research，不进入本机实现 frontier；启动时先明确环境、资源及必要的进一步拆分。
- 每张未完成本机票为 ready-for-agent / local-delivery，并包含可验证的切片、验收标准、依赖标题及 Revision 2 变更注释。

## 下一步

建议优先 [25 · 预览并限制本机运行](issues/25-preview-and-bound-local-runs.md)：先让计划、时间/内存预算和批量情景回放可验证，再进入真实训练。

发布时可开始的 frontier 是 25 和 [05 · 构建 Reference Bank](issues/05-build-reference-bank.md)；05 不依赖 25。这里只完成待办发布，没有启动任何一张新实现票或训练任务。

关键后续入口：

- [21 · 受预算约束的训练中断与恢复](issues/21-enable-paper-scale-training.md)：在第一次 BM^E 训练之后，提前验证恢复。
- [26 · 验证可替换的银行与市场输入](issues/26-validate-replaceable-bank-and-market-inputs.md)：合成非标准输入的导入、校验与滚动。
- [27 · 验证单设备移植与银行交接](issues/27-validate-single-device-portability-and-bank-handoff.md)：可移植 checkpoint、CUDA 验证入口和银行侧交接。
- [23 · 一键本机流程验收](issues/23-run-quick-reproduction.md)：实际证据齐全后才给出 development-validated。
- [20 · 延期 Paper 配对训练](issues/20-run-paired-paper-convention.md)、[24 · 延期完整方法论研究验收](issues/24-assess-methodological-reproduction.md)：不阻塞上述本机交付。

## 核验

- 实际发布的全部依赖与用户确认草案一致；依赖图无环，引用票号存在且标题匹配。
- 本机票对延期研究票没有直接或间接依赖；新实现票没有提前勾选完成。
- 原运行骨架行为测试：18 passed，0.70 秒；仅用于核对 01 的已有能力，不代表金融模拟器、模型训练、CUDA 或端到端验收已经实现。
- 已核对规格、map 和 02–04 未改变；源代码、测试和环境配置未修改，也未启动模型训练。
- 本次是本地 Markdown 发布，不创建 Git commit。之前已有的规格修订和拆分草案保留在工作区。
