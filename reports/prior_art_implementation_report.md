# Prior-art comparison implementation report

## Implemented without experiment execution

- Method contracts encode online inputs, `A_S` use, deployability, output interface, formal eligibility, oracle/diagnostic/reject role, and exclusion reasons.
- PBP oracle and common-channel nominal LPs implement ordered nominal profile constraints and D_design-only fitting guards.
- MaSS-12 learns a finite block distribution, lifts it through the common decoder, computes discrete conditional MI exactly, preserves the paper's `m/n` directions and operational checks, and exposes no sensitive value on inference paths.
- Common-cover calibration freezes a full-support D_design cover, bisects to tolerance at most `1e-8`, independently reverifies the endpoint, and records utility/TV/constant-channel provenance.
- The attack-CMI diagnostic freezes equal-budget, user-day cross-fitted attacker families on `D_attack_train` and evaluates/bootstraps on `D_test` only.
- Results validation and Figures A–E enforce the formal/empirical separation. Review ZIP creation and plot metadata are deterministic by construction.
- Config and CLI accept every requested MaSS grid control with hyphen and underscore aliases.

## Difference from official MaSS code

No official upstream file is vendored, modified, imported, or called. The CAPT adaptation replaces continuous neural generation and adversarial CE estimates with an exact finite categorical channel and exact discrete conditional MI over the design split. It retains the paper's constraint directions and penalty form. Consequently, it must be named `MaSS-12 (finite-output adaptation)`; the cover-wrapped method must be named `MaSS-12 + certified cover calibration`.

## Deferred verification

Tests have been authored but not run in this turn. The required smoke/full grids, ruff, pytest, independent certificate verification, deterministic two-render hashes, runtime/memory report, and review packet creation must be executed only after the ongoing experiment is confirmed complete.
