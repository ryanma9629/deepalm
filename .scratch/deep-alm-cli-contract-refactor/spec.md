# Deep ALM：以配置合同驱动的通用 CLI 重构

Type: spec
Status: ready-for-agent
Date: 2026-09-09
Scope: remove scenario names from public execution commands; make one resolved configuration the sole definition of a Deep ALM workflow

## Problem Statement

作为 Deep ALM 的使用者，我目前必须同时在命令和 `--config` 文件名中重复说明场景。例如，四策略已修正本机验证既通过 `four-policy-pilot` 表示，又通过 `four-policy-corrected-pilot.yaml` 表示。两处都含有“four-policy”和“pilot”，但它们各自又隐含不同的训练成员、冻结 BM^D baseline reference、锁定评估和报告规则。

这使使用者无法清楚判断哪一处才是 Reference Bank、TreasuryPolicy 矩阵、已修正金融语义、资源预算和验收边界的唯一来源。它也让 `corrected-pilot`、`four-policy-pilot`、`workflow`、`train` 等命令把 Runner 的内部编排细节暴露为公共 CLI。随着本机验证、方法论复现和银行 GPU 运行增多，命令数量会按场景增长，而不是按用户要执行的动作增长。

用户需要一个动作导向的 CLI：命令只表达“计划、运行、评估、报告”等动作；完整配置只表达“要运行什么 Deep ALM 工作流、使用什么数据、哪些 TreasuryPolicy、什么资源、什么输出和什么验收合同”。这不能削弱当前 Corrected Financial Semantics、冻结 BM^D baseline reference 或 artifact identity 的严格性。

## Solution

将一次实验的场景定义收敛为配置内唯一的 **Workflow Contract（工作流合同）**。Workflow Contract 是可审计的配置身份，定义 TreasuryPolicy 成员、5/15 年期限、是否包含 swaps、MM 对同期限冻结 BM^D baseline reference 的依赖、训练/评估/报告生命周期以及所需的 artifact 证据。解析后的配置推导并记录合同要求；用户不再在 CLI 子命令中重复输入场景名称。

将公开 CLI 收敛为动作导向的 `plan`、`run`、`evaluate`、`report`，并保留确实独立的准备/诊断动作（市场 preflight、Reference Bank 导入/构建、单设备 commissioning 和可恢复训练的 resume）。`run` 根据 Workflow Contract 分派到正确的内部编排；`evaluate` 和 `report` 从同一合同验证 source run 与 evaluation run。当前两策略已修正本机验证和四策略初步比较都成为配置选择，而不是命令名称。

将资源选择从场景名称中拆出为 **Execution Profile（执行配置档）**。Execution Profile 表达设备、数值类型、网络规模、路径/epoch、时间和内存 guard；Workflow Contract 表达金融和实验语义。一个最终 YAML 仍然是单一、完整、可复现的运行输入，且 resolved configuration 与 manifest 必须展开记录实际合同和实际资源值。

## User Stories

1. As a treasury model developer, I want `deepalm run --config <file>` to execute the workflow defined by that file, so that I do not repeat the scenario in the command name.
2. As a treasury model developer, I want `deepalm plan --config <file>` to show the resolved Workflow Contract, TreasuryPolicy matrix and Execution Profile before market or training work starts, so that I can detect an incorrect run early.
3. As a treasury model developer, I want `deepalm evaluate --config <file> --source-run <directory>` to evaluate only the contract defined in the configuration, so that an incomplete or different source cannot be promoted to a result.
4. As a treasury model developer, I want `deepalm report --config <file> --source-run <directory> --evaluation-run <directory>` to produce the report required by the same Workflow Contract, so that the report is not selected by a scenario-specific command name.
5. As a MacBook user, I want a local two-policy validation configuration to specify BM^D and MM at both 5 and 15 years, so that I can verify the frozen BM^D baseline reference path without naming that scenario in the CLI.
6. As a MacBook user, I want a local four-policy comparison configuration to specify BM^E, BM^C, BM^D and MM at both supported horizons, so that I can receive initial like-for-like point estimates from one generic lifecycle.
7. As a MacBook user, I want the local two-policy and four-policy configurations to share an Execution Profile where appropriate, so that resource limits are not confused with the policy matrix.
8. As a research user, I want a paper-oriented Workflow Contract to express the disclosed no-swap TreasuryPolicy matrix and paper-width training target, so that methodological reproduction is not encoded in a bespoke CLI name.
9. As a bank integrator, I want a bank commissioning or bank-training Workflow Contract to select CUDA-oriented resources and an imported Reference Bank path through configuration, so that the same public actions work after handoff.
10. As a bank integrator, I want bank data, market identity and Execution Profile to remain explicit in the resolved configuration, so that a checkpoint cannot cross an incompatible Reference Bank or market boundary.
11. As a reviewer, I want the Workflow Contract to retain Corrected Financial Semantics as the only executable financial semantics, so that the CLI refactor cannot revive a Paper convention or paired-convention path.
12. As a reviewer, I want MM to require a same-horizon frozen BM^D baseline reference whenever the contract contains MM, so that generic command names cannot weaken the policy dependency.
13. As a reviewer, I want every source run, locked evaluation and report to record the resolved Workflow Contract and Execution Profile identity, so that CLI simplification does not reduce auditability.
14. As a reviewer, I want `evaluate` and `report` to reject source evidence whose contract, Reference Bank, market identities, policy members, horizons or corrected financial semantics differ, so that point estimates remain comparable only within one declared experiment.
15. As a reviewer, I want the local four-policy comparison report to contain exactly eight policy-horizon members and use the same locked test scenarios by horizon, so that it can answer an initial MM/BM^E/BM^C/BM^D comparison question honestly.
16. As a reviewer, I want the local validation reports to disclose that they use one seed, small path counts and two epochs, so that no user mistakes the initial comparison for convergence evidence or bank-model approval.
17. As a maintainer, I want one internal contract dispatcher rather than separate public corrected-pilot and four-policy-pilot command branches, so that adding a future Workflow Contract does not multiply CLI parsing logic.
18. As a maintainer, I want scenario validation to live with the Workflow Contract, so that a policy list, horizon set, baseline dependency and report requirements cannot disagree across configuration validation, Runner and reporting code.
19. As a maintainer, I want an Execution Profile to contain only execution-scale choices, so that names such as `m5-small`, `cuda-single-gpu` and `cuda-cluster` do not silently choose TreasuryPolicy members or financial semantics.
20. As a maintainer, I want resolved configurations to materialize concrete resource values rather than merely store a profile label, so that manifests remain self-contained audit evidence.
21. As a CLI user, I want generic help text organized by actions and preparation stages, so that I can learn the lifecycle without first learning internal pilot names.
22. As a CLI user, I want retired scenario-specific execution commands to fail as unknown commands after the migration, so that there is one public lifecycle rather than two overlapping interfaces.
23. As a CLI user, I want existing `preflight`, `bank`, `device-check` and `resume` actions to remain available when their distinct outputs are needed, so that the refactor does not remove useful preparation, portability or recovery operations.
24. As a documentation reader, I want the English and Chinese READMEs to explain that commands mean actions and configuration means the complete experiment, so that file names can be concise and scenario names do not recur in the command line.
25. As a documentation reader, I want configuration examples named by scenario and environment rather than only by vague size labels, so that `local-four-policy`, `paper-oriented` and `bank-single-gpu` communicate the intended experiment.
26. As a future contributor, I want a documented extension point for new Workflow Contracts, so that swap support, a research runner or bank training can be added without adding `deepalm <new-scenario>-pilot` commands.
27. As a repository maintainer, I want historic scenario-specific artifacts retained as historical evidence but not accepted by the new generic lifecycle unless their contract identity is explicitly compatible, so that migration does not reinterpret old output directories.
28. As a repository maintainer, I want the current two-policy and four-policy behavior preserved behind the new contract dispatcher before any broader research or bank capabilities are added, so that the CLI cleanup does not expand financial-model scope.

## Implementation Decisions

- Add an ADR that establishes the public CLI rule: a subcommand denotes an action, while a configuration denotes the full experiment. This ADR must explicitly preserve ADR-0002: Corrected Financial Semantics remains the sole executable financial semantics.
- Introduce a resolved Workflow Contract field. The contract is the sole configuration authority for the scenario-specific invariants currently spread among special run-scale profiles, policy lists, Runner methods, evaluation validators and report builders.
- The first supported Workflow Contracts are: compact two-policy local validation; compact four-policy local comparison; existing bounded local workflow; paper-oriented research plan; and single-device bank commissioning. A contract may be planned even when its full training capability is not yet delivered; the plan/report must disclose that boundary rather than simulate a completed run.
- A local two-policy contract resolves exactly BM^D/MM at 5 and 15 years, four optimizer updates per member, MM's same-horizon frozen BM^D baseline reference, 16 total updates, compatible locked evaluation and a four-member report.
- A local four-policy contract resolves exactly BM^E/BM^C/BM^D/MM at 5 and 15 years, four optimizer updates per member, MM's same-horizon frozen BM^D baseline reference, 32 total updates, compatible locked evaluation and an eight-member comparison report.
- Introduce an Execution Profile field that is independent of Workflow Contract. It owns the resolved device, dtype, network width, path counts, epoch count, batch size, time guard and RSS/accelerator memory guards. Initial named profiles must distinguish at least the target M5 compact validation profile and CUDA single-GPU commissioning profile; future CUDA-cluster profiles are a declared extension, not an implementation in this change.
- The resolved configuration must expand both the Workflow Contract and Execution Profile into concrete values. The manifest records this resolved identity, the existing market and Reference Bank identities, the seed registry and corrected financial semantics. No runtime default may be absent from the recorded configuration.
- Remove scenario-bearing public execution commands: `workflow`, `train`, `corrected-pilot`, `corrected-evaluate`, `corrected-report`, `four-policy-pilot`, `four-policy-evaluate` and `four-policy-report`. Do not retain hidden aliases or compatibility wrappers. `run` becomes the generic primary execution action.
- Make `plan`, `run`, `evaluate` and `report` the primary public lifecycle. `evaluate` and `report` receive source/evaluation artifact locations as required by the selected contract. The public command parser must not use a scenario name to select a Runner method.
- Retain `preflight`, `bank`, `device-check` and `resume` as action-specific commands. They may use the configuration's Workflow Contract to validate compatibility, but are not alternative public scenario lifecycle names. Remove `profile` and `calibrate` aliases if they duplicate `plan` and `preflight` without providing distinct observable behavior.
- Centralize contract dispatch behind one Runner entry seam. It chooses the internal training, evaluation and reporting implementation from the resolved Workflow Contract; public CLI parsing remains a thin action-to-runner adapter. Existing specialized methods may become private adapters during migration but must not define public CLI behavior.
- Centralize expected policy-horizon member construction, optimizer-update counts, MM frozen-baseline requirements, artifact key names and report member validation in the Workflow Contract. Evaluation and reporting must consume this definition rather than hard-code separate two-policy and four-policy lists.
- Rename shipped configuration examples so their filenames communicate scenario plus environment, such as local two-policy, local four-policy and bank single-GPU. File names are descriptive only; behavior is determined by YAML content and resolved identity.
- Migrate README examples in English and Chinese to generic commands. They must explain that a local four-policy configuration is still only an initial comparison: same locked scenarios, one seed, 64 test paths and two epochs do not demonstrate economic superiority.
- Historic artifacts and previously published `corrected-*`/`four-policy-*` output directories remain on disk. The new lifecycle does not infer a Workflow Contract from a filename and does not promote historic evidence without exact contract and existing artifact-identity checks.
- Keep the current no-swap product scope, Reference Bank contract, market provenance rules, MM Curve Feature Preprocessing, Recoverable Training semantics and all corrected financial formulas unchanged. This is a CLI/configuration architecture change, not a financial-model change.

## Testing Decisions

- The primary test seam is the public generic CLI lifecycle: `plan`, `run`, `evaluate` and `report` receive a complete configuration and publish observable artifacts. Tests must not assert which former specialized Runner method happens to execute internally.
- For each shipped local Workflow Contract, one high-level lifecycle test verifies that `plan` resolves its contract and Execution Profile, `run` publishes the exact expected members, `evaluate` uses compatible locked scenarios, and `report` publishes only the expected members and disclosure.
- The local two-policy lifecycle test verifies exactly BM^D/MM at 5/15 years, 16 updates and same-horizon frozen BM^D baseline references for MM. It is the successor to current corrected-pilot lifecycle coverage.
- The local four-policy lifecycle test verifies exactly BM^E/BM^C/BM^D/MM at 5/15 years, 32 updates, the same MM baseline rule and a common locked-test identity per horizon. It is the successor to current four-policy command coverage.
- Configuration tests verify that each Workflow Contract resolves its complete policy/horizon matrix and rejects contradictory scenario fields, incompatible Execution Profiles, swaps in no-swap contracts, unavailable device requirements and altered locked local budgets.
- Artifact-compatibility tests verify that `evaluate` and `report` reject evidence with a different contract identity, TreasuryPolicy matrix, horizon, corrected-financial-semantics marker, market identity, Reference Bank identity or seed/locked-scenario identity where the existing contract requires equality.
- Command-surface tests verify that generic actions appear in help, scenario-bearing commands and duplicate aliases are rejected as unknown, and retained preparation/recovery actions remain independently callable.
- Report tests verify that every report records the resolved Workflow Contract, concrete Execution Profile, policy members, scope disclosure, market and Reference Bank provenance, and the current limitation of compact local evidence.
- Existing focused CLI tests for corrected local validation, four-policy local validation, generic configuration and documentation are prior art. They should be migrated to assert the new public lifecycle rather than duplicated alongside the old command surface.
- README tests inspect both Chinese and English examples for generic action commands and complete configuration selection. Documentation must not advertise removed scenario-specific commands.
- Acceptance requires format/lint checks, the full automated test suite, generic CLI help, and smoke `plan` runs for each shipped Workflow Contract. Compact test doubles may be used at the lifecycle seam; the acceptance suite must not require a five-to-ten-minute M5 training run.

## Out of Scope

- Replacing the existing Corrected Financial Semantics, reintroducing a Paper convention, or changing any loan, deposit, cash, valuation or risk formula.
- Adding swaps, changing the no-swap TreasuryPolicy universe, or extending the product model beyond the already disclosed methodological-reproduction scope.
- Delivering paper-scale convergence, matching the paper's numerical results, or asserting that MM economically dominates BM^E, BM^C or BM^D from compact local pilots.
- Implementing real-bank data ingestion, production governance, GPU-cluster distributed training, scheduler integration, secrets handling or model approval workflows.
- Providing automatic migration or reinterpretation of historic scenario-specific run directories. Historic evidence remains readable but must satisfy explicit compatibility checks before reuse.
- Introducing a general multi-file configuration-composition system, environment-variable interpolation or an arbitrary plug-in system for contracts in this refactor.
- Removing or redesigning distinct preparation, device-diagnosis or recovery capabilities merely to reduce the command count.

## Further Notes

- The key boundary is intentionally conceptual rather than cosmetic: Workflow Contract answers “what financial experiment is this?”, while Execution Profile answers “what resources execute it?”. A config such as `local-four-policy` should combine both because it is a complete reproducible experiment; a bare `small` file name cannot tell a reviewer which TreasuryPolicy matrix it runs.
- The primary user-facing seam is one generic lifecycle. This is preferable to testing individual parsers or private Runner branches because it protects the behavior users care about: a single configuration determines planning, artifacts, compatibility checks and reporting.
- The migration is deliberately breaking for old scenario-bearing CLI names. Retaining aliases would preserve the ambiguity this refactor is meant to eliminate. Historical output stays preserved, but public documentation and examples move to the generic interface at the same time.
- The contract registry is the approved extension point. A future research, swap-enabled or bank GPU workflow first gains a documented Workflow Contract and artifact requirements, then becomes available through the same actions; it does not gain another scenario-named subcommand.
