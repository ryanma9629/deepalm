# Resolve SNB, Svensson, and HJM-PCA numerical conventions

Type: research
Status: resolved

## Question

What exact data units, compounding conventions, tenor grids, yield-to-forward transformations, weekly sampling rules, covariance annualization, PCA scaling, polynomial regularization, HJM drift discretization, boundary handling, and weekly-to-monthly path conversion are supported by the paper, the official SNB source, and primary term-structure references? Identify every material ambiguity, especially eigenvalue versus square-root-eigenvalue scaling, and state the defensible alternatives without choosing between them.

## Answer

The SNB source and paper determine the data units and broad pipeline: continuously compounded NSS spot curves, weekly forward-curve changes, covariance annualized by 52, three PCA components, cubic tenor fits, and HJM simulation. Several numerical conventions remain undisclosed and must be configuration-visible. Most importantly, the paper's literal `lambda * q` scaling does not reproduce the retained covariance; the standard covariance factor is `sqrt(lambda) * q`. These become explicit Paper and Corrected convention candidates for the later decision.

See [the research report](../research/term-structure-conventions.md).
