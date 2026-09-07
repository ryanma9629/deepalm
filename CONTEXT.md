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
