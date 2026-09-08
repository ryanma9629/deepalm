# 27: Validate single-device portability and bank handoff

**What to build:** checkpoint 可经 CPU 读取并映射至明确设备；提供可在银行执行的小型单 GPU 验证配置与完整交接说明。

**Blocked by:** 16/Train the fifteen-year MM and verify five-year truncation; 26/Validate replaceable bank and market inputs.

**Status:** resolved

**Execution scope:** local-delivery

**Specification revision:** 2 (2026-09-07)

本票按规格 Revision 2 实现所列切片；默认只使用有界 local_flow 或更小的确定性验证夹具。完整任务的 development-validated 由 23 根据实际证据判定。

- [x] 同一 PyTorch simulator/policy/trainer 使用设备感知的张量、优化器和恢复路径，支持 CPU/MPS/CUDA 选择，不嵌入 Mac 本机路径或设备专属金融公式。
- [x] checkpoint 包含架构/参数、优化器/scheduler、epoch、特征 PCA、冻结 BM^D、seed registry、数据/快照/配置/代码身份和 schema 版本；拒绝不兼容预处理/数据及跨宽度续训。
- [x] 实际执行本机可用设备的加载与短恢复测试；提供同样的 CUDA 集成测试。无 CUDA 时记录 not-run，不以 mock 通过冒充硬件验证，也不承诺跨设备逐位相同。
- [x] 提供 uv/Linux 安装及运行说明，银行依次检查运行时/设备、合成输入、短 CUDA 训练与恢复、真实数据映射及对账、性能测量，再显式扩大训练规模。
- [x] compact 权重只作流程证据；银行通常用真实数据重新训练。实际连接/映射签字、多 GPU/DDP、调度、混合精度和集群吞吐优化均须银行侧另行立项及验收，本票不声称已经实现或验证。
- [x] Verify CPU-readable artifacts and allowed explicit device remapping while rejecting incompatible data, preprocessing, architecture, convention and horizon. A runtime-only device override is audited; cross-device bitwise training equality is not promised.
- [x] Commissioning evidence must distinguish code-path availability, actual local CPU/MPS execution, CUDA runtime tests not run without hardware, and the separately deferred real-bank mapping/multi-GPU boundary. Synthetic examples contain no private bank data.

## Comments

2026-09-07 — 用户确认 Revision 2 增量拆分后发布。新增单设备可移植代码与交接验证；无 CUDA 硬件时诚实标记 not-run，银行真实映射和多 GPU 工作明确另行验收。使用追加编号；本次只更新待办，不表示本票实现已经完成。

## Answer

2026-09-08 — 已交付 CPU recovery → 可用 CPU/MPS/CUDA 短续训 → checkpoint 重载的真实单设备验证；无可用硬件时明确写入 `not-run` 原因。`device-check` 覆盖 BM^E/MM 与 5/15 年，运行包保存可移植性证据而不承诺跨设备逐位相同。checkpoint 记录参数、优化器、scheduler、epoch、PCA/冻结 BM^D 依赖、seed 与数据/快照/配置/Git/schema 身份；通用加载器拒绝金融口径、架构、实际期限、预处理和数据身份不一致。提供银行单 GPU 配置及 uv/Linux 交接步骤；真实银行映射、多 GPU 与正式训练仍明确留给银行侧验收。提交前完整 `ruff`、`pytest` 与 `git diff --check` 均通过。
