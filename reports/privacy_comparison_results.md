# Privacy-matched prior-art comparison results

## Execution status

Not run. The comparison implementation, config, schemas, and claim gates are prepared, but all smoke tests, optimization runs, certificate verification, attack fitting, plotting, pytest, and ruff were intentionally deferred while the user-reported experiment PID 13298 is active. No numeric result, certificate, confidence interval, figure, runtime estimate, or review packet is claimed in this file.

## 1. Formally matched privacy results

Pending execution. Only rows with an independently valid robust certificate and `certified_upper_epsilon <= 1` may be summarized here.

## 2. Oracle or different-online-input results

Pending execution. PBP oracle will remain non-deployable and visually separated even if its utility is higher.

## 3. Empirically matched privacy results

Pending conditional gate. This analysis runs only if the certified MaSS comparison is unavailable or collapses to nearly constant cover, and it must remain labeled as an empirical diagnostic.

## 4. Incomparable prior methods

The feasibility audit rejects Pufferfish central-query mechanisms and DistP tupling from the one-token frontier because their mechanism interfaces do not match. This is not a finding that their native privacy promises fail.

## 5. Claims currently allowed

- Implementation/provenance claims documented in the feasibility and methodology reports.
- No performance or privacy-comparison claim is currently allowed because no comparison has run.

## 6. Claims currently disallowed

- Any numeric CAPT-versus-MaSS/PBP utility claim.
- Any statement that a configured MaSS `m` is an achieved robust epsilon.
- Any statement that a lower audit proves safety.
- Any description of MaSS-12 as official MaSS.
- Any formal-comparison label without a valid robust certificate.
