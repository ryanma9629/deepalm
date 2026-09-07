# SNB, Svensson, and HJM-PCA numerical conventions

## Scope and conclusion

This note resolves what the sources do and do not determine for the term-structure part of the methodological reproduction. It does not select an implementation policy.

The paper determines the broad pipeline:

1. reconstruct continuously compounded Swiss Confederation spot curves from daily SNB Nelson-Siegel-Svensson (NSS) parameters;
2. convert them to forward curves;
3. form weekly forward-curve differences over 1 January 2005 through 15 July 2022;
4. multiply the estimated weekly covariance by 52;
5. retain three principal components and fit cubic tenor functions;
6. use those functions as time-homogeneous HJM volatilities, calculate the risk-neutral drift, and simulate curves;
7. convert simulated forwards to discount factors and zero yields for the monthly ALM model.

The sources do **not** uniquely determine the weekly observation rule, the exact yield-to-forward discretization, covariance estimator details, the HJM tenor grid, the HJM simulation step, the weekly-to-monthly interface, the initial curve date, or several boundary conventions. Most importantly, the paper literally scales covariance eigenvectors by eigenvalues, but a Brownian volatility factor that reproduces the estimated covariance must scale them by the **square roots** of the eigenvalues. Both variants therefore need explicit names if both are implemented; they must never be silently interchanged.

## Primary-source findings

### 1. SNB source and data units

The official SNB page identifies cube `rendopar` as **Nelson-Siegel-Svensson parameters for calculating the daily yield curve of Confederation bonds**. Its notes say that the published curves are spot rates on synthetic zero-coupon bonds, that maturity is measured in years, and that the spot rates are continuously compounded. The page also says that the series shown here uses the SNB's 2002 methodology through July 2025. The paper's calibration window ends in July 2022, so the entire research sample belongs to that methodology. [SNB cube](https://data.snb.ch/en/topics/ziredev/cube/rendopar), [SNB methodology notes](https://data.snb.ch/en/topics/ziredev/doc/explanations_ziredev#interest_rates_meth_par_siegel)

The official API exposes six series, `b0`, `b1`, `b2`, `b3`, `t1`, and `t2`, with daily frequency metadata. The local extract stores one date/parameter/value per row, with a UTF-8 BOM, semicolon separators, and three metadata/header lines before the data. [SNB dimensions API](https://data.snb.ch/api/cube/rendopar/dimensions/en), [SNB data API example](https://data.snb.ch/api/cube/rendopar/data/json/en?dimSel=d0%28b0%2Cb1%2Cb2%2Cb3%2Ct1%2Ct2%29&fromDate=2005-01-03&toDate=2005-01-03), [`rendopar` extract](../../../data/snb-data-rendopar-en-all_19880401-20250731.csv)

The NSS spot curve used in the paper is

\[
R(m)=\beta_0+\beta_1 L(m,\tau_1)
      +\beta_2\left[L(m,\tau_1)-e^{-m/\tau_1}\right]
      +\beta_3\left[L(m,\tau_2)-e^{-m/\tau_2}\right],
\]

where

\[
L(m,\tau)=\frac{1-e^{-m/\tau}}{m/\tau}, \qquad m>0.
\]

This is Equation 37 of the paper and the zero-yield form derived in Svensson's original work. The instantaneous forward form is

\[
f(m)=\beta_0+\beta_1e^{-m/\tau_1}
 +\beta_2\frac{m}{\tau_1}e^{-m/\tau_1}
 +\beta_3\frac{m}{\tau_2}e^{-m/\tau_2}.
\]

[Englisch et al., Sections 2.7.2-2.7.5](../../../docs/Deep%20treasury%20management%20for%20banks.pdf), [Svensson 1994, Sections II-III and Appendix](https://doi.org/10.5089/9781451853759.001)

The numerical values of `b0` through `b3` in the SNB file are percentage points per annum, while `t1` and `t2` are year-scale parameters. This follows from the official definition of `R` as an annual continuously compounded spot rate at maturity `m` years and is confirmed by the magnitude of the data: for example, treating `b0 = 3.666` on 3 January 2005 as a decimal would imply a 366.6% long-end rate. Before discounting or applying the HJM drift formula, rates should therefore be represented as decimals, or an exactly equivalent scale conversion must be carried through every formula.

The distinction is material. In decimal units,

\[
D(m)=e^{-mR(m)}.
\]

If `R` remains in percentage points, the exponent must use `R/100`. More subtly, an HJM drift computed directly from percentage-point volatilities needs a scale correction: if \(\sigma_{pp}=100\sigma\), then the percentage-point drift is \(\sigma_{pp}\int\sigma_{pp}/100\), not \(\sigma_{pp}\int\sigma_{pp}\). A decimal-rate internal representation avoids this trap. The paper's Equation 4 and the SNB's own technical work both use exponential discounting. [Englisch et al., Equation 4](../../../docs/Deep%20treasury%20management%20for%20banks.pdf), [Christensen and Mirkov, SNB Working Paper 2021/2, Appendix C](https://www.snb.ch/public/asset/en/www-snb-ch/publications/research/working-papers/2021/working_paper_2021_02/publications0_en/working_paper_2021_02.n.pdf)

For numerical stability, `L` can be evaluated as `-expm1(-x)/x`, with \(x=m/\tau\), and its limit at zero is 1. The available grid starts above zero, but this treatment avoids cancellation for small \(x\).

### 2. Tenor grid and curve conversion

The ALM model is explicit about its own grid: one month per step, 60 or 180 horizon steps for five- or fifteen-year experiments, and `N = 180` future cash-flow buckets. Its yield vector therefore needs maturities

\[
m_k=k/12,\qquad k=1,\ldots,180.
\]

[Englisch et al., Section 2.1.1 and Appendix notation table](../../../docs/Deep%20treasury%20management%20for%20banks.pdf)

Section 2.7 defines the HJM forward vector on a generic grid \(x_k=k\Delta\tau\), but never explicitly sets \(\Delta\tau=1/12\). Using the ALM monthly tenor grid is a defensible reading because the simulated yield vector is described as the same `N`-dimensional input needed by Deep ALM. A weekly HJM tenor grid followed by interpolation to monthly tenors is also defensible because Section 2.7 later calls the term-structure discretization weekly. This remains unresolved by the text.

There are two source-compatible ways to obtain historical forwards:

1. **Analytic instantaneous forward:** evaluate the closed-form Svensson forward equation above at each tenor.
2. **Discrete forwards consistent with Equation 42:** first set \(D_k=\exp(-m_kR_k)\), \(D_0=1\), then compute

   \[
   F_k=-\frac{\log D_k-\log D_{k-1}}{\Delta\tau}
      =\frac{m_kR_k-m_{k-1}R_{k-1}}{\Delta\tau}.
   \]

The second definition makes the paper's forward-to-discount approximation exact on the chosen piecewise-constant grid:

\[
D_k=\exp\left(-\sum_{j=1}^{k}F_j\Delta\tau\right),
\qquad
Y_k=-\frac{\log D_k}{m_k}.
\]

The paper says only that PCA is applied to historical CHF forward curves. It does not state which of these two conversions it used. They converge as \(\Delta\tau\) becomes fine but are not identical on monthly tenors.

### 3. Weekly observations and covariance

The following points are explicit in Section 2.7.5 and Table 2:

- use **forward-curve changes**, not NSS parameter changes and not percentage returns;
- use weekly changes;
- use 1 January 2005 through 15 July 2022;
- estimate their covariance and multiply it by 52;
- keep three components;
- fit degree-three polynomials.

The first usable complete record in the local extract after the stated start date is 3 January 2005. The end date has a complete observation. The paper does not prescribe treatment of holidays, missing values, or multiple daily observations within a calendar week.

Defensible weekly rules include:

- the last available observation in each calendar week, conventionally a `W-FRI` resample;
- Friday only, dropping weeks without a Friday observation;
- every fifth valid observation;
- an exact seven-calendar-day lag on daily curves.

These produce different samples and must be configuration-visible. Interpolating the six NSS parameters across missing dates is not required by any source and can be fragile because large offsetting parameter changes can leave the fitted curve almost unchanged. Resampling reconstructed curves is better supported than resampling individual parameters.

“Covariance” conventionally means centering the arithmetic differences by their sample mean. The paper does not state whether the divisor is \(n-1\) or \(n\), although standard sample covariance and common PCA implementations use \(n-1\). It also does not state an outlier rule. These are smaller but reproducibility-relevant choices. [scikit-learn PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

Annualizing the weekly covariance before or after eigendecomposition is equivalent when done consistently:

\[
\widehat\Sigma_{ann}=52\widehat\Sigma_{week},
\qquad
\lambda_{ann,j}=52\lambda_{week,j}.
\]

The corresponding annual volatility loading scales by \(\sqrt{52}\), not by 52.

### 4. PCA scaling: eigenvalue versus square root

The paper states

\[
\widehat\Sigma_{ann}=Q\Lambda Q^{-1}
\]

and then literally defines the retained loading vectors as

\[
\widetilde V^{(j)}=\lambda_jQ^{(j)}.
\]

Because the covariance matrix is symmetric, \(Q^{-1}=Q^\top\) for an orthonormal eigensystem. If an HJM shock is \(VdW\), then its instantaneous covariance is \(VV^\top dt\). Therefore a rank-three factor that reproduces the retained covariance is

\[
V_{sqrt}=Q_{1:3}\Lambda_{1:3}^{1/2},
\qquad
V_{sqrt}V_{sqrt}^{\top}
=Q_{1:3}\Lambda_{1:3}Q_{1:3}^{\top}.
\]

The paper's literal factor instead gives

\[
V_{literal}V_{literal}^{\top}
=Q_{1:3}\Lambda_{1:3}^{2}Q_{1:3}^{\top},
\]

which squares the retained covariance eigenvalues. It also assigns variance units to a quantity used as volatility. PyTorch documents the same covariance-factor identity, `covariance_matrix = cov_factor @ cov_factor.T + cov_diag`. [PyTorch `LowRankMultivariateNormal`](https://docs.pytorch.org/docs/stable/distributions.html#lowrankmultivariatenormal)

The evidence therefore supports two distinguishable conventions without establishing which code the authors actually ran:

- **Literal paper convention:** \(V^{(j)}=\lambda_jq_j\).
- **Covariance-factor convention:** \(V^{(j)}=\sqrt{\lambda_j}q_j\).

The private-code footnote in the publication does not resolve the discrepancy. The figures alone are not sufficient evidence to infer the implementation. A reproduction should report the empirical covariance implied by the fitted `V` under either convention.

Eigenvector signs are arbitrary. Flipping one fitted factor's sign and the corresponding Brownian coordinate leaves its distribution and its own HJM drift contribution unchanged. A deterministic sign rule is still useful for reproducible plots and polynomial coefficients.

### 5. Cubic tenor regularization

The paper says to approximate each of the three scaled eigenvectors as a polynomial function of tenor and uses degree three. The direct reading is an ordinary cubic regression with an intercept:

\[
v_j(x)=c_{j,0}+c_{j,1}x+c_{j,2}x^2+c_{j,3}x^3.
\]

It does not state:

- whether regression is weighted by tenor or observation precision;
- whether raw years or a normalized tenor coordinate is used;
- whether an intercept or a boundary constraint such as \(v_j(0)=0\) is imposed;
- how eigenvector signs are stabilized;
- whether the fit uses only 1m-15y nodes or an extended grid;
- whether extrapolated polynomial values are clipped.

Ordinary unweighted least squares with an intercept is the minimal literal interpretation. Rescaling the tenor coordinate does not change the exact cubic function space, but it improves numerical conditioning. Polynomial fitting intentionally changes the retained covariance: after fitting, \(VV^\top\) will generally differ from the rank-three PCA reconstruction, so both approximation errors should be measured separately.

### 6. HJM drift and path discretization

Under the risk-neutral measure, the paper uses the HJM restriction

\[
\alpha(t,T)
=\sum_{j=1}^{d}\sigma_j(t,T)
  \int_t^T\sigma_j(t,s)\,ds.
\]

It then assumes \(\mathbb P=\mathbb Q\), constant-in-calendar-time volatility, and tenor-only functions \(v_j(x)\), where \(x=T-t\). In Musiela time-to-maturity coordinates the drift is

\[
A(x)=\sum_{j=1}^{d}v_j(x)\int_0^x v_j(u)\,du,
\]

and the forward-curve dynamics become

\[
dF_t(x)=\left[\partial_xF_t(x)+A(x)\right]dt
        +\sum_{j=1}^{d}v_j(x)dW_{t,j}.
\]

This matches the original HJM no-arbitrage result and the paper's Equations 39-41. [Heath, Jarrow, and Morton 1992](https://doi.org/10.2307/2951677), [Englisch et al., Sections 2.7.4-2.7.5](../../../docs/Deep%20treasury%20management%20for%20banks.pdf)

The paper specifies a trapezoidal approximation for the volatility integral and a forward difference for the tenor derivative. A direct explicit Euler-Maruyama step is therefore

\[
F_{n+1,k}=F_{n,k}
+\left[
  \frac{F_{n,k+1}-F_{n,k}}{\Delta\tau}+A_k
 \right]\Delta t
+\sum_{j=1}^{d}V_{k,j}\sqrt{\Delta t}\,Z_{n,j},
\]

with independent standard-normal \(Z_{n,j}\). The \(\sqrt{\Delta t}\) factor is implicit in the paper's Brownian increment \(W_{t+\Delta t}-W_t\), and omitting it is an error.

Material undisclosed details are:

- whether the trapezoid starts from a polynomial evaluation at tenor zero or from the first positive tenor;
- whether a cubic's analytic integral is used instead of the stated trapezoid;
- the relationship between \(\Delta t\) and \(\Delta\tau\);
- the order in which the curve is advanced, truncated, and exposed to ALM;
- whether negative rates or extreme paths are clipped. The paper reports that more than 95% of simulated 1m yields remain above -1%, but does not impose a floor;
- the initial simulation curve. The calibration end date is disclosed, but the paper does not explicitly say that the 15 July 2022 curve is \(Y_0\).

The paper resolves the last-node forward-difference problem only at a high level: start with a higher-dimensional curve containing longer maturities and reduce its dimension step by step. It does not give the initial extended length. If \(\Delta t=\Delta\tau=1/12\), one-node-per-step reduction implies `N + H` initial tenor nodes: through 20 years for a five-year run and through 30 years for a fifteen-year run. The NSS formula can provide those extrapolated initial tenors. If the simulator is weekly or \(\Delta t\ne\Delta\tau\), this interpretation no longer follows uniquely; interpolation or a different boundary scheme is required.

### 7. Weekly calibration versus monthly ALM

The ALM clock is monthly: \(\Delta t=1/12\), with 60 or 180 transitions. Section 2.7 calls the term-structure time discretization weekly and annualizes weekly changes by 52. It never states how weekly paths are converted to the monthly ALM clock.

Three defensible interfaces remain:

1. **Weekly calibration, monthly simulation.** Estimate weekly covariance, annualize it, construct annualized `V`, and apply Euler-Maruyama directly with \(\Delta t=1/12\). A continuous-time diffusion with annualized coefficients does not require intermediate weekly steps. This also permits \(\Delta t=\Delta\tau=1/12\) and makes the paper's one-node boundary reduction natural.
2. **Weekly simulation, exact-month observation.** Simulate with \(\Delta t=1/52\), then interpolate forward curves to the exact times \(m/12\) for ALM. The interpolation rule is not disclosed.
3. **Weekly simulation, nearest/period-end sampling.** Simulate weekly and select a nearest week or last week in each month. This introduces unequal or approximate month lengths and requires an explicit calendar convention.

A fixed “four weeks per month” conversion gives 48 weeks per year and conflicts with the paper's covariance multiplier of 52 unless an additional scaling convention is introduced. It has no direct textual support.

There is also an off-by-one inconsistency in the paper. Section 2.1.1 describes `H` equidistant times with \(T=(H-1)\Delta t\), whereas the Appendix labels `H = 60 or 180` as the number of steps, defines \(\Delta t=T/H=1/12\), and lists times \(\{0,\Delta t,\ldots,H\Delta t\}\). The latter means 60/180 transitions and 61/181 curve states and is consistent with a five-/fifteen-year horizon. The path tensor shape should therefore be a named convention, not an implicit array-length assumption.

## Ambiguity register for the later convention decision

| Topic | Fixed by evidence | Defensible alternatives still open |
| --- | --- | --- |
| Source series | SNB `rendopar`, 2002-methodology daily NSS parameters | None for the paper sample |
| Rate units | Betas are annual percentage points; spot rates are continuously compounded; maturity and `t1/t2` use years | Store decimals internally, or store percentage points with explicit conversion everywhere |
| Calibration dates | 2005-01-01 through 2022-07-15, inclusive intent | First/last complete data observation handling |
| ALM tenor grid | 1m increments, 180 maturities through 15y | Whether HJM calibration/simulation uses the same grid or a finer weekly grid |
| Yield to forward | PCA uses forward curves | Analytic instantaneous NSS forwards; discrete forwards consistent with Equation 42 |
| Weekly sampling | Weekly forward differences; covariance multiplied by 52 | Week-end last available; Friday-only; every fifth observation; exact seven-day lag |
| Missing values | No rule disclosed | Drop incomplete observations/weeks; curve-level carry-forward under an explicit rule |
| Covariance | Arithmetic forward differences and three leading components | `n-1` vs `n`; de-mean vs zero second moment; outlier policy |
| PCA loading scale | Paper literally says \(\lambda q\) | Literal \(\lambda q\); covariance factor \(\sqrt\lambda q\) |
| Eigenvector sign | Distribution is sign invariant | Any deterministic sign normalization for reproducibility |
| Polynomial | Three separate degree-three tenor fits | Intercept/boundary constraint; weighting; coordinate scaling; fit/extrapolation grid |
| HJM measure | Paper imposes \(\mathbb P=\mathbb Q\) | No alternative within literal paper scope; a market-price-of-risk extension is outside it |
| HJM drift | HJM restriction; trapezoidal integral | Treatment of tenor zero; analytic cubic integral as a disclosed variant |
| Numerical step | Forward tenor difference and Brownian increment | Monthly direct step; weekly substeps plus interpolation/sampling |
| Long-end boundary | Extend the curve and reduce dimension stepwise | Exact extension length and truncation order; alternative one-sided boundary |
| Initial curve | Must be supplied and is matched by HJM | Final calibration-date curve; separately configured as-of curve |
| Extreme paths | No hard floor is disclosed | No clipping; an explicitly non-paper stress guard |
| Path length | Five-/fifteen-year monthly experiments | 60/180 transitions plus initial state versus the contradictory Section 2.1.1 indexing |

## Reproduction checks implied by the evidence

These checks can compare conventions without selecting one:

1. Reconstruct several SNB spot curves and confirm \(D_k=\exp(-m_kR_k)\) in decimal units.
2. For the discrete-forward route, confirm the round trip `spot -> discount -> forward -> discount -> spot` to numerical tolerance.
3. Record the exact weekly dates selected and the number of differences; a hidden resampling default is not reproducible.
4. Verify eigenpairs are sorted descending and \(Q^\top Q=I\).
5. Before polynomial fitting, compare `V @ V.T` with the retained covariance. The square-root convention should match; the literal eigenvalue convention should match \(Q\Lambda^2Q^\top\).
6. After polynomial fitting, report covariance reconstruction error separately from PCA truncation error.
7. Verify Brownian innovations have empirical covariance approximately `V @ V.T * dt`.
8. Compare trapezoidal and analytic cubic HJM integrals, especially at the 1m tenor.
9. Verify every simulated discount curve and spot curve has the expected 180 monthly maturities at each ALM decision date.
10. Run identical seeds under the weekly-substep and direct-monthly variants and quantify terminal-curve differences rather than treating them as equivalent.

## Source list

- Englisch, H., Krabichler, T., Müller, K. J., and Schwarz, M. (2023), *Deep treasury management for banks*, especially Sections 2.1.1-2.1.2 and 2.7.2-2.7.5, Equations 4 and 37-42, Table 2, footnote 15, and the Appendix notation table. [Local paper](../../../docs/Deep%20treasury%20management%20for%20banks.pdf), [publisher HTML](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1120297/full), [DOI](https://doi.org/10.3389/frai.2023.1120297).
- Swiss National Bank, *Nelson-Siegel-Svensson parameters: For calculating the daily yield curve of Confederation bonds*. [Cube](https://data.snb.ch/en/topics/ziredev/cube/rendopar), [methodology notes](https://data.snb.ch/en/topics/ziredev/doc/explanations_ziredev#interest_rates_meth_par_siegel), [dimensions API](https://data.snb.ch/api/cube/rendopar/dimensions/en).
- Svensson, L. E. O. (1994), *Estimating and Interpreting Forward Interest Rates: Sweden 1992-1994*, IMF Working Paper 94/114. [DOI and full text](https://doi.org/10.5089/9781451853759.001).
- Heath, D., Jarrow, R., and Morton, A. (1992), *Bond Pricing and the Term Structure of Interest Rates: A New Methodology for Contingent Claims Valuation*, Econometrica 60(1), 77-105. [DOI](https://doi.org/10.2307/2951677).
- Christensen, J. H. E., and Mirkov, N. (2021), *The Safety Premium of Safe Assets*, SNB Working Paper 2021/2, Appendix C. [Official SNB PDF](https://www.snb.ch/public/asset/en/www-snb-ch/publications/research/working-papers/2021/working_paper_2021_02/publications0_en/working_paper_2021_02.n.pdf).
- PyTorch documentation, covariance-factor identity for `LowRankMultivariateNormal`. [Official documentation](https://docs.pytorch.org/docs/stable/distributions.html#lowrankmultivariatenormal).
- scikit-learn documentation, PCA centering, eigenvectors, and explained-variance definition. [Official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
