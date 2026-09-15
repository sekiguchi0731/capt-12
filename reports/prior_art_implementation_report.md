# Prior-art comparison implementation report

## Implemented without experiment execution

- Method contracts encode online inputs, `A_S` use, deployability, output interface, formal eligibility, oracle/diagnostic/reject role, and exclusion reasons.
- PBP oracle and common-channel nominal LPs implement ordered nominal profile constraints and D_design-only fitting guards.
- MaSS-12 learns a finite block distribution, lifts it through the common decoder, computes discrete conditional MI exactly, preserves the paper's `m/n` directions and operational checks, and exposes no sensitive value on inference paths.
- Common-cover calibration freezes a full-support D_design cover, bisects to tolerance at most `1e-8`, independently reverifies the endpoint, and records utility/TV/constant-channel provenance.
- The attack-CMI diagnostic freezes equal-budget, user-day cross-fitted attacker families on `D_attack_train` and evaluates/bootstraps on `D_test` only.
- Results validation and Figures A–E enforce the formal/empirical separation. Review ZIP creation and plot metadata are deterministic by construction.
- `capt12 prior-art-comparison --phase pilot|full` now connects native CAPT/LDP runs to D_design MaSS fitting, exact robust verification, cover calibration, D_test evaluation, checkpointed result writing, and rendering. `run-grid` rejects prior-art/MaSS controls to prevent an accidental CAPT-only run.
- Comparison certificates store the K-by-L input-to-block factor, L-by-K decoder, full-support cover, confidence boxes, and ordered adjacency. The independent verifier reconstructs the exact K-by-K `Q` per public context and reruns the existing support verifier; it does not accept configured MaSS `m` as achieved privacy.
- The main renderer emits the requested certified privacy–utility frontier and epsilon=1 L-sensitivity view. The former Figures A–E are emitted under `figures/supplementary/`.

## Difference from official MaSS code

No official upstream file is vendored, modified, imported, or called. The CAPT adaptation replaces continuous neural generation and adversarial CE estimates with an exact finite categorical channel and exact discrete conditional MI over the design split. It retains the paper's constraint directions and penalty form. Consequently, it must be named `MaSS-12 (finite-output adaptation)`; the cover-wrapped method must be named `MaSS-12 + certified cover calibration`.

## Deferred verification

Tests have been authored but not run in this turn. The required pilot/full grids, ruff, pytest, independent certificate verification, deterministic two-render hashes, runtime/memory report, and review packet creation must be executed only after the ongoing experiment is confirmed complete.
