# Prior-art feasibility audit

Status: implementation-time audit completed; comparison experiments have not been run. Retrieval date: 2026-08-25.

## Repository provenance gate

- Requested current-series commit: `cfb8aecfcc2ebb98e625c7c1595857331acf5476` (`Support compressed context channel grids`).
- Requested 3-cost commit: `a143413ae0a735fcaf2c1ca97cfa3ae226940ac8` (`Aggregate objective-aligned cost comparisons`).
- Ancestry: `cfb8aec…` is the direct parent of `a143413…`; the comparison implementation starts from later descendant `4d78be8ad0288a74aeebb45b749af772527fc01c`.
- Initial checkout: detached HEAD at `4d78be8…`; initial worktree status was clean. Work moved to `prior-art-privacy-matched` before edits.
- Reproduction gate: the existing 3-cost code and commit are present, but no reproduction, smoke test, solver, certificate verifier, pytest, ruff, or plotting command was run because PID 13298 was reported as an active experiment and the user explicitly deferred every execution.

## Feasibility table

`A_S` means the realized protected value/profile. “CAPT verifier” means the existing robust upper verifier, not raw-data provenance authentication.

| candidate | privacy definition | protected secret | side information | online inputs | `A_S` online | known profile/distribution | output | one 12-bit token | adjustable | CAPT verifier | official implementation | decision | rejection/qualification |
|:--|:--|:--|:--|:--|:--:|:--:|:--|:--:|:--:|:--:|:--|:--|:--|
| CAPT-12 | robust profile privacy over confidence sets | protected profile tuple | frozen `B,S`, adjacency and uncertainty sets | `Z,B,S` | no | confidence set from `D_cert` | one token | yes | epsilon | native | repository | main | formal reference method |
| optimal LDP | epsilon-LDP | input token | arbitrary input distinction | `Z,B,S` | no | no | one token | yes | epsilon | yes | repository LP | main | optimal in the same cost/channel class; fixed GRR cannot be claimed to dominate it |
| best constant cover | input-independent, epsilon 0 | all input/profile information | any | `B,S` | no | `D_design` output mass only | one token | yes | cover distribution | yes | repository adaptation | main | full-support cover is frozen before certification |
| PBP oracle | nominal profile-based privacy | source profile | profile graph and nominal distributions | `Z,B,S,A_S` | **yes** | yes | one token | yes | epsilon | no shared-channel robust check | paper-derived LP; no official code used | oracle | non-deployable utility upper bound |
| PBP common nominal | point-estimate profile privacy with a shared channel | source profile | profile graph and nominal `D_design` distributions | `Z,B,S` | no | yes | one token | yes | epsilon | frozen channel: yes | paper-derived LP; no official code used | diagnostic, promoted only if certified | call it uncertified nominal unless robust verification passes |
| PBP common nominal + cover | robust profile privacy after CAPT wrapper | same CAPT profile | same CAPT certificate objects | `Z,B,S` | no | confidence sets | one token | yes | lambda by bisection | yes | CAPT adaptation | main | wrapper is not from the PBP paper |
| MaSS official | constraints `I(X';S_i)<=m_i`, `I(X';U_j)>=n_j` in nats | annotated sensitive attributes | learned attribute inference networks | original representation `X` | no at inference | empirical joint data | continuous/modal representation | no | `m,n`, penalties | no | yes, SHA `6fbe9be1…`, Apache-2.0 | reject as direct baseline | AudioMNIST/MotionSense-oriented representation interface does not match a finite 12-bit channel |
| MaSS-12 raw | exact finite-output adaptation of MaSS MI penalty directions | protected attributes | `D_design` labels during training | `Z,B,S` | no | empirical `D_design` joint distribution | one token through common decoder | yes | `m,n`, weights, temperature | frozen channel: yes | CAPT adaptation, not official MaSS | diagnostic, promoted only if certified | MI controls are not achieved robust epsilon values |
| MaSS-12 + cover | robust profile privacy after CAPT wrapper | same CAPT profile | same CAPT certificate objects | `Z,B,S` | no | confidence sets | one token | yes | lambda by bisection | yes | CAPT adaptation | main | call it `MaSS-12 + certified cover calibration`, never original MaSS |
| Pufferfish framework | Pufferfish privacy | configured secret pairs | configured data-evolution family | none by itself | n/a | yes | none by itself | no | framework epsilon | no mechanism to inspect | definition, not implementation | reject | a framework is not a local token mechanism |
| Pufferfish Wasserstein mechanism | Pufferfish privacy | central secret pair | data-evolution family | database query | n/a | yes | noisy numeric query | no | epsilon | no | paper mechanism | reject | central query release cannot be recast as the local token interface without defining a new method |
| Pufferfish Markov Quilt | Pufferfish privacy for correlated data | state/secret pair | Markov model | central query and model | n/a | yes | noisy query | no | epsilon/quilt | no | paper mechanism | reject | Markov central-query assumptions do not match Criteo one-display local release |
| DistP tupling | distribution privacy | source distribution | candidate distributions | local datum | no | yes | tuple of real plus dummy outputs | **no** | epsilon/tuple size | no | paper mechanism | reject | interface requires multiple outputs; compressing to one token would be a new mechanism |
| GRR/RAPPOR-style fixed LDP | LDP | token | arbitrary input distinction | `Z` | no | no | one token/bit vector | GRR yes | epsilon | yes | standard formulas | diagnostic | optimal LDP is already the cost-optimal comparator |

## MaSS loss audit

The [MaSS paper](https://arxiv.org/html/2405.14981) defines `m_i` as an upper bound on sensitive mutual information and `n_j` as a lower bound on useful mutual information, both in nats:

- sensitive: `I(X';S_i) <= m_i`, relaxed through the violation side of `H(S_i)-m_i <= CE(S_i)`;
- useful: `I(X';U_j) >= n_j`, relaxed through the violation side of `CE(U_j) <= H(U_j)-n_j`;
- operational checks: `m_i >= 0`, `n_j <= H(U_j)`, and `n_j <= m_i + H(U_j|S_i)` under deterministic labels.

The official code at `6fbe9be1…` implements the same directions using `min(CE-H(S)+m,0)` for suppression and `max(CE-H(U)+n,0)` for preservation, followed by a squared plus absolute penalty. One code path prints the cross-attribute infeasibility message without exiting, so this repository performs the mathematical feasibility checks explicitly. Parameter names are controls, not measurements of achieved leakage.

## Primary sources

- [MaSS paper](https://proceedings.mlr.press/v235/chen24f.html) and [official code](https://github.com/jpmorganchase/MaSS)
- [Profile-Based Privacy](https://arxiv.org/abs/1903.09084)
- [Pufferfish mechanisms](https://arxiv.org/abs/1603.03977)
- [Distribution Privacy / tupling](https://arxiv.org/abs/1812.00939)
