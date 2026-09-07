# 02: Reconstruct SNB term structures

**What to build:** Turn the supplied SNB NSS export into validated historical monthly spot, discount, and forward curves that downstream market models can consume.

**Blocked by:** 01/Establish an auditable run skeleton.

**Status:** completed

- [x] Ingestion handles metadata rows, semicolon separation, long-form parameter codes, missing observations, duplicates, dates, and declared units.
- [x] Calibration data is restricted to 1 January 2005 through 15 July 2022 and beta parameters are converted from percentage points to decimal rates.
- [x] Continuously compounded NSS spot rates, discount factors, and monthly discrete forwards are produced on the required tenor grid.
- [x] Spot, discount, and forward representations round-trip within `1e-10` on deterministic fixtures.
- [x] The 15 July 2022 curve is exposed as the canonical initial curve and source identity is preserved in the run bundle.
- [x] Invalid source structure, units, dates, or non-finite reconstructed values fail at ingestion rather than downstream.
