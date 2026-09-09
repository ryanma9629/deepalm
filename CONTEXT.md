# Deep ALM

This project reconstructs the treasury-management problem described in *Deep treasury management for banks* using public market data and a transparent substitute for unavailable bank data.

## Language

**Methodological reproduction（方法论复现）**:
A reconstruction of the paper's disclosed data transformations, financial state transitions, optimization method, and evaluation logic whose results are assessed for structural and behavioral agreement rather than exact numerical identity.
_Avoid_: Exact replication, numerical reproduction

**Reference Bank（参考银行）**:
The internally consistent bank representation used in place of the paper's unavailable proprietary bank data, with assumptions stated explicitly and kept replaceable by real bank data.
_Avoid_: Mock bank, fake bank

**已修正金融语义（Corrected Financial Semantics）**:
本项目唯一可执行的金融公式与计量定义，基于论文披露、errata 和实现审计共同确定；运行时不提供与之并列的替代口径。
_Avoid_: Paper convention, 可切换口径

**实现版勘误（Implementation Errata）**:
项目维护的、从论文及其 errata 映射到已修正金融语义的唯一审计说明；它区分已采纳的公式修正、明确保留的建模简化与尚未接通的能力缺口。
_Avoid_: 可执行的论文口径, 隐式修正

**工作流合同（Workflow Contract）**:
一次 Deep ALM 实验的可审计语义身份，唯一规定 TreasuryPolicy 成员、期限、无互换范围、验收目的和所需的冻结基准依赖；它回答“要执行什么金融实验”。
_Avoid_: 命令名中的场景, 资源配置

**执行配置档（Execution Profile）**:
独立于工作流合同的、完整且固定的运行资源信封，规定设备、数值类型、网络规模、路径/epoch、批大小、时间与内存 guard；它回答“以什么资源执行已声明的实验”。
_Avoid_: 场景选择器, 仅含设备的标签

**BM^E、BM^C、BM^D 与 MM**:
论文中的四类无互换 TreasuryPolicy：等期限分配 benchmark、固定期限分配 benchmark、按决策日期分配 benchmark，以及共享参数的多期模型。
_Avoid_: bme, bmc, bmd, mm

**TreasuryPolicy 决策状态**:
某一 Treasury 决策日期可交易的投资与融资名义梯子及其在有限决策期限中的位置；benchmark 只消费这些核心字段，MM 额外消费完整银行状态、当前曲线、上期约束、`mu` 与 `lambda`，以决定本期无互换交易动作。
_Avoid_: 完整资产负债表快照, 通用状态

**MM 曲线特征预处理（MM Curve Feature Preprocessing）**:
仅由登记的 training states 拟合的、中心化但不标准化的三成分曲线 PCA；它保存中心、投影、确定性 path/time 抽样索引以及数据和校准身份，并拒绝 selection/test 或身份不匹配的重载。
_Avoid_: 使用 selection/test 曲线拟合, 无身份的 PCA

**冻结 BM^D baseline reference（冻结基准引用）**:
与一个已选择 BM^D checkpoint 并列保存的、内容寻址的只读引用；引用文件名内含其内容哈希，并记录 checkpoint 哈希、期限、配置与数据身份。MM 先从引用路径验证该哈希与 checkpoint 身份，再加载为不可训练的策略。
_Avoid_: 文件名引用, 未验证的 checkpoint

**可恢复训练（Recoverable Training）**:
只在完整 epoch 的 selection 完成后保存的训练进度；它包含当前模型、优化器、scheduler、selection 历史和语义身份。重启时会重放未完成 epoch，并仅允许设备、输出位置和增加资源预算的覆盖。
_Avoid_: batch 级别快照, 不经兼容性检查的续训

**已修正本机验证试跑（Corrected Local Validation Pilot）**:
在固定本机资源预算内运行 BM^D 与 MM、各 5 年和 15 年的四任务训练矩阵；它验证训练、冻结 baseline、评估和报告链路，不宣称收敛、论文数值复现或银行模型获批。
_Avoid_: 配对口径试跑, 正式经济验收

**期限结构（Term Structure）**:
某一估值日、同一组月度期限上的连续复利即期利率、贴现因子和离散远期利率，它们是同一条利率曲线的三种一致表示。
_Avoid_: 三条独立曲线

**校准窗口（Calibration Window）**:
用于估计市场情景模型的、完整历史期限结构的固定时间区间；本项目采用 2005 年 1 月 1 日至 2022 年 7 月 15 日。
_Avoid_: 任意训练数据

**存款参考历史（Deposit Reference History）**:
截至初始估值日的完整历史期限结构，用于计算初始存款参考利率，而非用于 HJM 校准。
_Avoid_: 校准窗口

**标准初始曲线（Canonical Initial Curve）**:
2022 年 7 月 15 日的期限结构；它是 Reference Bank 的估值起点和市场情景的共同起点。
_Avoid_: 最新曲线, 任意起始曲线

### Regulatory constraints

**流动性覆盖率（Liquidity Coverage Ratio, LCR）**:
高质量流动资产相对于 30 日净流出的比率；本复现的约束下限为 105%。
_Avoid_: 流动性比例, 流动性缓冲

**净稳定资金比率（Net Stable Funding Ratio, NSFR）**:
可用稳定资金相对于所需稳定资金的比率；本复现的约束下限为 105%。
_Avoid_: 长期流动性比例

**现金对最低准备金比率（Cash-to-Minimum-Reserve Ratio, CMR）**:
现金相对于瑞士央行最低准备金的比率；本复现的约束下限为 100%。
_Avoid_: 准备金覆盖率

**权益/风险加权资产比率（Equity/RWA）**:
经济权益相对于风险加权资产的资本约束比率；本复现的约束下限为 17%。
_Avoid_: 杠杆率, 资本率

**利率敏感度（Interest Rate Sensitivity, IRS）**:
平行上、下移 100 个基点后最差的绝对权益变化，相对于当前权益的比率；本复现的约束上限为 8.5%。
_Avoid_: 久期, DV01

**超额年度权益回报（Excess Yearly Equity Return, EYR）**:
年度分红前权益变化扣除 6 mCHF 后，相对于上一年权益的比率；仅在年度关闭时适用，约束下限为 0%。
_Avoid_: 年收益率, 股东回报
