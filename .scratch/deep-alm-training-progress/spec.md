# Deep ALM：默认训练进度输出

Type: spec
Status: ready-for-agent
Date: 2026-09-10
Scope: expose bounded local-training progress through the generic `run` action without changing Corrected Financial Semantics, workflow identity, or stored artifacts

## Problem Statement

作为 Deep ALM 的使用者，我通过 `deepalm run --config <file>` 执行本机四策略或两策略工作流时，实际会训练多个 TreasuryPolicy/期限成员，但终端在训练完成前几乎没有可见反馈。特别是在 MacBook 上执行 5 年和 15 年、BM^E、BM^C、BM^D 与 MM 的矩阵时，我无法判断当前正在训练哪一个成员、已经完成第几个 epoch、训练是否仍在推进，或当前损失大致如何变化。

我需要类似 PyTorch 常见训练循环的简洁进度输出：默认可见，按 epoch 显示当前训练成员、epoch 进度和损失；必要时可显式关闭，以便把 CLI 用于脚本、CI 或安静的批量运行。这个可观察性改进不得改变 TreasuryPolicy 的梯度计算、选择逻辑、冻结 BM^D baseline reference 或任何已修正金融语义；终端输出与其他 `run` I/O 一样计入真实的 wall-clock guard。

## Solution

在通用 `run` 动作增加默认开启的布尔型 verbose 控制，支持 `--verbose` 和 `--no-verbose`，其中不写任何开关时等价于 verbose 开启。对于真正执行本机训练矩阵的 Workflow Contract，终端按成员报告开始和完成，并在每个完成的 epoch 后报告：TreasuryPolicy、期限、`当前 epoch/总 epoch`、累计 optimizer updates、该 epoch 的平均训练损失，以及在发生 selection 时的 selection total loss 和 penalty loss。

训练循环把 epoch 级的结构化进度事件交给 Runner；Runner 只负责面向人类的格式化和输出。进度事件不进入 checkpoint、manifest、Reference Bank、市场情景或评估 artifact，因此当 verbose 与 non-verbose 都在相同 Execution Profile 预算内完成时，它们产生相同的训练语义和可审计身份。终端 I/O 的真实耗时仍属于 Execution Profile 的 wall-clock 计量；在异常缓慢或阻塞的输出端，它可能消耗该预算，但不会被隐式扣除或伪装为训练时间。

## User Stories

1. As a Deep ALM user, I want `deepalm run --config <file>` to show training progress by default, so that I can tell that a local run is advancing without inspecting processes or artifacts.
2. As a MacBook user, I want each progress line to identify the TreasuryPolicy and the 5-year or 15-year horizon, so that I know which member of the configured matrix is training.
3. As a MacBook user, I want each completed epoch to show `epoch/total epochs`, so that I can estimate remaining local validation work.
4. As a MacBook user, I want each completed epoch to show an average training loss, so that I can see whether optimization is producing finite, changing values.
5. As a model developer, I want selection total loss and penalty loss shown whenever selection is evaluated, so that I can distinguish optimization-batch loss from the fixed selection measurement used for checkpoint choice.
6. As a model developer, I want each progress line to include cumulative optimizer updates, so that I can reconcile terminal progress with the bounded Workflow Contract update count.
7. As a model developer, I want a member-start message before training and a member-completion message after selection, so that longer member boundaries are clear even when selection is sparse.
8. As a scripting user, I want `--no-verbose` to suppress human-oriented training progress, so that stdout can remain limited to the completed artifact directory and errors keep their current behavior.
9. As a scripting user, I want `--verbose` to be accepted explicitly as well as by default, so that invocation intent is readable in automation.
10. As a reviewer, I want verbose output to be observational only when both display modes complete within the same Execution Profile budget, so that changing the display flag cannot change model parameters, random seeds, optimizer updates, selected checkpoints, manifests, or financial results.
11. As a reviewer, I want Corrected Financial Semantics to remain the sole executable financial semantics, so that progress reporting cannot revive a Paper convention branch.
12. As a reviewer, I want the frozen same-horizon BM^D baseline reference required by MM to remain unchanged, so that reporting does not weaken the policy dependency.
13. As a maintainer, I want the public generic `run` action to remain the primary control seam, so that no scenario-named CLI command or duplicate training entry point is introduced.
14. As a maintainer, I want a small structured epoch-progress boundary between trainer and Runner, so that presentation does not become embedded in the financial optimization loop.
15. As a maintainer, I want the behavior to apply consistently to BM^E, BM^C, BM^D, and MM whenever their Workflow Contract runs real local training, so that the four-policy comparison has uniform observability.
16. As a maintainer, I want workflows that only create a planning/audit bundle to remain valid, so that the feature does not claim epoch progress where no model is actually trained.
17. As a documentation reader, I want CLI help and the runnable examples to state that progress is on by default and can be disabled with `--no-verbose`, so that the terminal behavior is discoverable.

## Implementation Decisions

- Retain the action-oriented public CLI established by ADR-0003. `run` receives the verbosity control; no workflow-specific command or scenario-bearing alias is added.
- Model the parser option as a paired boolean flag with a default of enabled. Both `--verbose` and `--no-verbose` are valid; the latter is the only way to suppress progress for a normal `run` invocation.
- Use the existing generic CLI-to-Runner `run` dispatch as the highest behavioral seam. The resolved Workflow Contract continues to select the internal local-training implementation.
- Add one optional, structured epoch-progress callback at the BenchmarkTrainer boundary. A progress event represents a completed epoch and carries the policy identity, horizon, total epochs, completed epoch, optimizer-update count, average training loss, and any selection measurement available at that boundary.
- The callback is observational and optional. With no callback, trainer behavior remains silent and identical to current behavior; checkpoint/recovery cadence remains at the existing complete-epoch safe boundary.
- Calculate average training loss from the scalar losses actually used for that epoch's optimizer updates. Do not add extra forward passes, resample market paths, alter gradient flow, or run selection more often solely to print a value.
- Preserve the distinction between epoch training loss and fixed selection loss. When a selection is not scheduled for an epoch, the formatted progress must say so by omitting selection fields rather than inventing a value.
- Make Runner the sole owner of terminal wording and member lifecycle messages. It converts structured events into concise, stable human-readable stdout lines and passes callbacks to every real local matrix member.
- Keep final CLI success behavior: after a successful run, stdout still prints the artifact directory. Progress lines precede that final path; `--no-verbose` preserves the prior one-line-success form.
- Do not persist verbose preference or individual progress events as manifest identity. The resolved configuration, Workflow Contract, Execution Profile, Reference Bank identity, market identity, seed registry, checkpoints, and corrected-financial-semantics marker remain unchanged by this display setting.
- Preserve the existing resource monitor, wall-clock, memory, cancellation, recovery, early-stopping, selection, and frozen baseline behaviors. Progress I/O is deliberately included in the real wall-clock guard and is never excluded or rewritten in resource measurements. Errors remain on stderr under the existing exit-status contract.
- Update user-facing CLI documentation in both English and Chinese only where command examples or option descriptions need the new default/opt-out behavior.

## Testing Decisions

- The primary test seam is the public `deepalm run --config <file>` behavior. A compact Runner double verifies that the default invocation passes verbose enabled and that `--no-verbose` passes it disabled; tests assert observable action behavior rather than parser internals.
- A trainer-level focused test verifies that one completed epoch emits one structured progress event with finite average training loss, epoch and total-epoch values, and the actual optimizer-update count. It verifies selection fields only when the existing selection schedule evaluates that epoch.
- A local Workflow Contract test uses compact trainer doubles to verify that all configured policy/horizon members receive progress reporting and that member-start/member-completion messages identify the same matrix as the contract.
- An output test verifies that verbose mode presents policy, horizon, epoch progress, training loss, and selection loss when applicable, while non-verbose mode suppresses those lines and retains the final artifact-directory output.
- A semantic-invariance test executes equivalent compact deterministic training that completes within its Execution Profile budget, with and without the optional callback, and compares the selected checkpoint/result identity, optimizer-update count, and selection history. The test must not require a five-to-ten-minute M5 run.
- Existing generic CLI surface tests, local two-policy/four-policy pilot tests, and bounded training/recovery tests are prior art. Extend them rather than restoring retired scenario-specific command tests.
- Acceptance requires formatting/lint checks, the focused progress and CLI tests, the full automated suite, and generic CLI help inspection. A real multi-minute M5 training run is not required for automated acceptance.

## Out of Scope

- Changing Corrected Financial Semantics, any runoff, deposit, loan, cash, valuation, capital, liquidity, or risk formula.
- Reintroducing a Paper convention, a parallel convention switch, or a paper-scale result claim.
- Altering the TreasuryPolicy matrix, supported 5/15-year horizons, no-swap scope, MM observation network, PCA preprocessing, or frozen BM^D baseline-reference rule.
- Adding rich progress bars, TUI rendering, log files, TensorBoard, experiment tracking services, metrics dashboards, remote telemetry, or a machine-readable streaming protocol.
- Changing training resource budgets, MacBook execution profiles, GPU/cluster portability, checkpoint schema, manifest schema, or artifact compatibility rules.
- Making `plan`, `evaluate`, `report`, `preflight`, `bank`, `device-check`, or `resume` verbose unless a separately specified need arises.

## Further Notes

- The approved test seam is intentionally the generic `run` action, with one narrow trainer-to-Runner progress-event boundary beneath it. This keeps the public behavior testable while avoiding terminal I/O in financial-model code.
- Loss values are diagnostic optimization evidence, not economic evaluation results. Compact local output must not be interpreted as proof that MM outperforms BM^E, BM^C, or BM^D.
- Because selection cadence can differ by Execution Profile, an epoch can have a training-loss value without a selection-loss value. This is expected and should remain legible in output.
- The visible labels use the glossary's `TreasuryPolicy`, `Workflow Contract`, `Execution Profile`, BM^E/BM^C/BM^D/MM, and Corrected Financial Semantics terminology.
