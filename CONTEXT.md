# Deep ALM

This project reconstructs the treasury-management problem described in *Deep treasury management for banks* using public market data and a transparent substitute for unavailable bank data.

## Language

**Methodological reproduction（方法论复现）**:
A reconstruction of the paper's disclosed data transformations, financial state transitions, optimization method, and evaluation logic whose results are assessed for structural and behavioral agreement rather than exact numerical identity.
_Avoid_: Exact replication, numerical reproduction

**Reference Bank（参考银行）**:
The internally consistent bank representation used in place of the paper's unavailable proprietary bank data, with assumptions stated explicitly and kept replaceable by real bank data.
_Avoid_: Mock bank, fake bank

**Paper convention（论文口径）**:
An implementation choice that follows the paper's stated formula or procedure when the disclosure is sufficiently precise, including choices that differ from common practice.
_Avoid_: Correct implementation

**Corrected convention（修正口径）**:
An explicitly labeled alternative used when evidence supports a correction or conventional interpretation of an ambiguous paper procedure.
_Avoid_: Silent fix, improved version

**BM^E、BM^C、BM^D 与 MM**:
论文中的四类无互换 TreasuryPolicy：等期限分配 benchmark、固定期限分配 benchmark、按决策日期分配 benchmark，以及共享参数的多期模型。
_Avoid_: bme, bmc, bmd, mm

**TreasuryPolicy 决策状态**:
某一 Treasury 决策日期可交易的投资与融资名义梯子及其在有限决策期限中的位置；它是策略决定本期无互换交易动作所依据的状态。
_Avoid_: 完整资产负债表快照, 通用状态

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
