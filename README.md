# CAPT-12

This repository is a reproducible research implementation of **block CAPT**, a
post-processing sanitizer for a frozen token encoder `Phi_0: X -> [K]`, with
`K <= 4096`.  The advertiser receives only the sampled sanitized token `O`.
The main guarantee treats the selected profile `S` as known and protects the
selected sensitive-proxy tuple `A_S`, conditional on an advertiser-visible
context bucket `B`.

The main mechanism partitions tokens into non-empty blocks, learns a
profile-specific row-stochastic block transport `R_S`, then samples from a
profile-common decoder inside the destination block. Scalar keep-or-cover is
included only as a baseline. `f_ref` and the outcome label are utility inputs;
this project does **not** claim to hide `Y`, make a Blackwell-order claim, or
treat the fixed encoder as a contribution.

## Quick start

```bash
uv sync --extra dev
uv run pytest -q
uv run capt12 smoke --config configs/smoke.yaml
uv run capt12 run-grid --config configs/synthetic_theorem4.yaml
uv run capt12 plot --suite paper --input outputs/runs
```

For local CriteoPrivateAd shards:

```bash
uv run capt12 inspect-data --data-root data/CriteoPrivateAd_release/data
uv run capt12 smoke --config configs/criteo_main.yaml --max-rows 5000
```

`configs/criteo_main.yaml` names anonymized columns as *candidate protected
attributes* or *sensitive proxy attributes*. It never assigns real-world
semantics such as age or region to them. Five temporal splits are disjoint:
model fit, design, certification, attack training, and final testing.

## Guarantees and bounds

The default privacy scope is `tuple_adjacent`; `marginal` is an ablation and
`joint_all_pairs` is optional. Rare groups are never silently discarded.
`cp_box` and `hoeffding_box` yield finite-sample upper certificates. The
DP-aware construction is deliberately labeled experimental and cannot be
emitted as a certified result until its coverage validation is accepted.
`point` confidence is restricted to known synthetic populations. A positive
TV radius expands the finite-sample confidence set and therefore cannot make a
certificate less conservative.

Finite-sample Criteo coverage is conditional on the declared `user_day_iid`
assumption after one-display-per-uuid-day contribution sampling. The policy
bounds each privacy epoch's contribution but does not itself prove i.i.d.
sampling or physical-user independence across days.

Expected protected/context tuples are formed from the frozen pre-certificate
domains, including `__OTHER__` and `__MISSING__`. By default, an unobserved
tuple or a disconnected hybrid path triggers the configured explicit
failure/common-cover policy; it is never removed from the claimed scope
silently. The explicit `missing_group_policy: full_simplex` mode instead gives
every zero-count group the full block-probability simplex and keeps CP/TV boxes
for all observed groups, including rare ones. The certificate binds and
reconstructs this mixed uncertainty policy.
`merge_to_other` is disabled until an identical frozen runtime coarsening map
can be included in the deployment certificate.

The first fixed-design full-simplex comparison is run with:

```bash
uv run capt12 simplex-completion --config configs/criteo_simplex_completion.yaml
```

It compares common cover, k-ary randomized response, the optimal row-wise LDP
block LP, and simplex-completed CAPT in the same fixed partition, common
decoder, cost, and objective class.

Before expanding that comparison, screen whether the fixed block class has any
utility reason to use an input-dependent channel and compare utility-aware
alternatives with:

```bash
uv run capt12 utility-design --config configs/criteo_utility_design.yaml
```

The diagnostic records the best constant and no-privacy row-wise distortions,
their information gap, row argmins, and no-privacy row TV. A class with a
zero information gap, one shared row argmin, or zero no-privacy row TV is not
sent to the privacy LP. Informative classes compare utility medoid, cost
medoid, joint weighted cost k-medoids, and the `L=K` singleton/identity positive
control against optimal LDP and independently verified simplex-CAPT.

To isolate sparse public contexts instead of making one global channel pay for
all of them, run the prescribed L=16 primary diagnostic first:

```bash
uv run capt12 context-stratified --config configs/criteo_context_stratified.yaml
```

Override the frozen design seed for one run with
`--frozen-design-seed 7`. To assess whether the result is stable across seeds,
run the seeds sequentially and aggregate them in one command:

```bash
uv run capt12 context-seed-stability \
  --config configs/criteo_context_stratified.yaml \
  --frozen-design-seeds 0,1,2,3,4 \
  --utility-objective teacher_kl
```

The LP objective can be changed without changing the frozen support,
partition/decoder, privacy constraints, or certificate definition. Supported
values are `teacher_kl`, `empirical_logloss` (D_design click labels only), and
`hybrid_logloss_kl`; the hybrid additionally accepts
`--hybrid-empirical-weight 0.5`. Never select the objective or hybrid weight
from D_test.

The stability command keeps epsilon=1 and L=16 fixed. It reuses completed runs
from the same source commit, writes per-seed and across-seed CSVs plus one
stability figure/report and D_attack_train-fixed privacy lower audits under
`outputs/context_stratified_seed_summaries/`, and
creates `sol_seed_stability_review_bundle.zip` containing all constituent
mechanisms and certificates for one-file review.

Run the substantially larger L=K=64 positive control separately, so a stopped
positive-control solve cannot discard the completed primary artifacts:

```bash
uv run capt12 context-stratified --config configs/criteo_context_stratified_l64.yaml
```

This maps missing and unseen sensitive values to one frozen `__UNKNOWN__`
secret and indexes the channel only by token `Z`, profile, and public context
`B`; the realized protected value is never an online selector. It compares
context CAPT against the fair context-specific optimal-LDP and best-constant
baselines. The primary uses joint k-medoids `L=16`; the separate positive
control uses singleton/identity `L=K=64`. The
guarantee concerns the coarsened sensitive value conditional on an attacker
already knowing `B`; leakage through `B` itself is outside scope.

The context CAPT channel is post-solve repaired with the minimum per-context
mixture of a uniform input-independent channel needed for strict pure-epsilon
feasibility. The released mechanism and all certificates contain the repaired
channel. `tables/privacy_repair_offending_constraints.csv` records every
pre-repair positive-over-zero constraint, and the result tables report the
mixing weights plus ordinary and tolerance-free Decimal verification.

This diagnostic is intentionally verbose. It flushes stage, design, context,
LP-size, HiGHS, cutting-plane, support-oracle, verification, memory, and
artifact events to the terminal and to `progress.log` and `progress.jsonl` in
the run directory. While a HiGHS call is still running, a heartbeat identifies
the exact solve label and elapsed time every 30 seconds. Robust optimization
shares each group/output support bound across all adjacent pairs and writes a
fingerprinted witness checkpoint after every cutting-plane iteration. Repeating
the same command after an interruption resumes those cuts; a checkpoint from a
different cost, confidence set, adjacency family, or tolerance is rejected.
If HiGHS dual simplex returns numerical status 4/Unknown, the identical LP is
retried once with HiGHS IPM at the same feasibility tolerance. A retry result
is never accepted without the usual support-oracle and independent certificate
verification stages.
If both algorithms reject the auxiliary-variable shared-support master, CAPT
automatically retries the same robust feasible set with the direct
paired-witness cutting-plane formulation. This removes the ill-conditioned
support-bound auxiliary variables without changing the objective, privacy
constraints, confidence sets, or certificate tolerance.

To reproduce the deterministic output directory from a Git revision and the
same config used for the run:

```bash
capt12 resolve-run-id a760ce9 \
  --config configs/criteo_context_stratified.yaml
```

The revision is resolved to its full commit SHA before hashing. A commit alone
is insufficient because the run ID also binds the complete resolved config.

Every successfully completed context-stratified run also writes
`sol_review_bundle.zip` in its run directory. The bundle contains the report,
tables, figures, resolved config, manifests, models, mechanisms, and every
context certificate. `sol_review_bundle_manifest.json` records the size and
SHA-256 of each included file. Cutting-plane checkpoints and verbose progress
logs are excluded because they are unnecessary for result review.

`verify-certificate` checks the self-contained confidence/count construction,
adjacency, block channel, partition/common decoder lift, component hashes, and
robust constraints using a verifier-owned tolerance. It is bundle-consistency
verification, not authentication of the original raw data; deployment still
requires a trusted manifest or signature for provenance.

Certificate generation also requires the clean committed worktree containing
the actually imported `capt12` package; process CWD is irrelevant, and loaded
CAPT modules must be tracked below that source root. The embedded
`source_git_sha` and `code_git_sha` must agree, so publication artifacts must
be regenerated after the final implementation commit.

Serialized external `f_ref` artifacts must declare feature columns, and a
precomputed prediction column requires `prediction_manifest_path` declaring
its upstream features. Certified Criteo runs reject either path if protected
columns are declared or inputs fall outside `phi_source_cols`.

For exact-token retention,

`U_CAPT(L) <= U_full <= U_envelope`.

At singleton blocks (`L=K`) CAPT reaches `U_full`, but need not reach the
Theorem-4 envelope. The latter is a relaxation, not an achievable mechanism.
For general distortions the comparison is the full-channel LP and its solver
bound, never the retention-only envelope.

Current Criteo smoke results that lack full protected/context support are
reported as a successful fail-safe retreat to profile-wide common cover, not
as evidence of a nontrivial CAPT gain. Reported Criteo utility is a frozen
reference-score surrogate; downstream CTR or auction utility is not claimed.

## Deployment requirements

Distribute all profile tables as one authenticated bundle with the same URL,
version, size, and timing behavior. Do not expose profile, raw token, or the
sampling branch to the advertiser. Reject unauthenticated/expired tables and
tables failing shift monitoring; configured fallback is common cover or
K-ary randomized response. Repeated releases can be memoized per user/privacy
epoch/static token. Network side-channel defenses and cryptographic secure
aggregation are operational prerequisites, not implementations claimed here.

See [theory](docs/theory.md), [data schema](docs/data_schema.md), [DP confidence](docs/dp_confidence.md), and [reproducibility](docs/reproducibility.md).

## Optional profile privacy

`--protect-profile off` is the default. The separate diagnostic supports
`counterfactual` comparison (profile-specific channels under one common input
distribution) and `observational` comparison (including each observed
`P(Z|S=s)`). The latter mixes sanitizer differences with population
composition and is never folded into the main theorem. Criteo has no observed
profile choices, so this diagnostic requires the deterministic synthetic
user/epoch assignment utility. Empty profile `S=∅` is excluded unless an
explicit API option includes it.
