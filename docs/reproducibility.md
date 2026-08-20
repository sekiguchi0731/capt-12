# Reproducibility protocol

## Split discipline

The Criteo default uses days 1–6 for `D_model`, 7–12 for `D_design`, 13–18 for
`D_cert`, 19–24 for `D_attack_train`, and 25–30 for `D_test`. The loader asserts
that day sets and `(day_int,id)` keys do not overlap. Every fit-capable
component stores its fit split and rejects later splits:

- `D_model`: category preprocessing, `f_ref`, and `Phi_0`;
- `D_design`: partition, common decoder, utility cost;
- `D_cert`: group histograms/confidence sets and the upper certificate;
- `D_attack_train`: attacker scores and event thresholds;
- `D_test`: final utility and the lower leakage witness.

The one-display-per-uuid-day contribution policy is applied independently to
`D_cert`, `D_attack_train`, and `D_test` before finite-sample inference. The
lower-audit candidate groups, context strata, token events, attacker model,
and thresholds are fixed from `D_attack_train`; D_test-only values do not
expand the family. Its `T` event/pair/stratum tests use `alpha/(2T)` for each
of the two one-sided bounds. Day-wise audit ranges are descriptive sensitivity
checks and not cluster-robust intervals.

No component is tuned after observing `D_test`. On Criteo, profile selection is
counterfactual because actual user choices are not present. Simulated
per-user/epoch profiles and optional profile privacy are labeled synthetic.

## Artifacts

Resolved config is canonicalized and SHA-256 hashed to obtain a resume-stable
run ID. Each ignored run directory records seed, component names, git SHA,
dependency versions, split IDs, input file metadata, solver status/size, model
and mechanism hashes, metrics, certificate, and plot source tables.

Certificate-producing runs require a clean worktree at a committed revision.
The Git root is resolved from the actually imported `capt12.__file__`, never
from the process working directory. Every loaded `capt12` source module must
be a tracked file below that one root. A wheel or copied package without a Git
root is rejected unless a future build-attestation path is implemented. The
resolved config records the revision as `source_git_sha`, and the writer
requires it to equal the certificate's `code_git_sha`. Verification rejects a
bundle whose two SHA fields disagree. Generated artifacts must therefore be
rerun after the last source commit; an older certificate is not relabeled.

Version-2 certificates embed histogram counts, structured groups, partition,
and common decoder. `verify-certificate` reconstructs adjacency and confidence
boxes, validates the block-to-token lift and component hashes, and rechecks the
robust constraints with a verifier-owned tolerance capped at `1e-8`. This
establishes internal deployment-bundle consistency; it does not independently
prove that embedded counts came from the named raw shards. Raw-data provenance
requires a separately trusted manifest or signature.

Theorem-4 rows include a `problem_signature` over the distributions,
adjacency, objective weights, and cost matrix. Figure aggregation matches this
signature together with K, epsilon, seed, case, partition, decoder, distortion,
weighting, and confidence so unrelated smoke runs cannot enter the comparison.
The Theorem-4 figure reports the mean and observed min–max across its three
seeds; it does not label a three-point bootstrap interval as a confidence
interval. Other Seaborn bootstrap intervals use a fixed random seed. PDF/SVG
metadata and SVG hash salt are fixed, and repeated rendering is byte-checked
in the integration suite.

Generated data, source data, DP histograms, checkpoints, caches, certificates,
results, and model artifacts are ignored by Git. Source manifests under
`data/` also remain covered by the pre-existing `data/*` rule.

## Assumptions selected during implementation

- Local release files are a one-shard-per-day sample, not the full release.
- `user_id` is only a within-day privacy-epoch contribution key; no cross-day
  persistence is assumed.
- After contribution sampling, user-day epochs are explicitly assumed
  conditionally i.i.d. within protected group and context. This is an external
  modeling assumption required by the CP/Hoeffding coverage statements, not a
  fact established by the contribution code or certificate verifier.
- `is_clicked` is the default CTR utility label.
- `features_kv_bits_constrained_2` and `_3` are anonymous sensitive proxies;
  `features_ctx_not_constrained_0` is an anonymous visible-context candidate.
- A hash encoder is the Criteo default because no precomputed 12-bit token was
  found. It is fitted/frozen on the configured source columns and never treated
  as the research novelty.
- DP-aware confidence remains experimental; certified main results use
  non-DP `cp_box`.
- Current Criteo smoke rows that trigger incomplete/rare support use a
  profile-wide input-independent common cover. They demonstrate the fail-safe,
  not a nontrivial CAPT privacy-utility improvement.

## Fixed-support certificate scaling

Use the focused experiment before interpreting a larger refitted Criteo run:

```bash
uv run capt12 fixed-support-scaling \
  --config configs/criteo_fixed_support_scaling.yaml
```

This command freezes the `features_kv_bits_constrained_2` mapper, hash encoder,
reference model, partition, decoder, common-cover distribution, utility cost,
objective weights, support sets, and adjacency hashes from the complete
`D_model` and `D_design` splits. It then selects one display per user-day by a
stable event hash and evaluates nested BLAKE2b-ranked `D_cert` prefixes at 5k,
10k, 25k, 50k, 100k, and full for sampling seeds 0, 1, and 2.

The output separates the frozen Cartesian-minus-design tuple set from the
design-minus-certificate tuple set. The current certified mechanism continues
to use the Cartesian complete-coverage gate. A reported
`design_support_ready_without_boundary_certificate` value is diagnostic only:
it is not permission to certify on observed design support without a frozen
runtime support rule and privacy constraints across the supported/unsupported
boundary. Sample-size projections use design frequencies and require temporal
stationarity; they are not certificate guarantees.

Each of the 18 cells writes a self-contained certificate and immediately runs
the independent bundle verifier. The ignored run directory also contains the
frozen support table, component artifacts, CSV/Parquet results, deterministic
PDF/PNG diagnostics, metadata, and a report.

## Full-simplex completion comparison

After the fixed-support diagnostic, run the prescribed one-condition
graceful-degradation experiment with:

```bash
uv run capt12 simplex-completion \
  --config configs/criteo_simplex_completion.yaml
```

This reuses the same full `D_model`/`D_design` frozen design and the complete
one-display-per-user-day `D_cert` split for
`features_kv_bits_constrained_2`, `L=16`, and `epsilon=1`. Positive-count
groups use their simultaneous CP/TV boxes regardless of the count-20
diagnostic threshold. Zero-count expected Cartesian groups use the exact full
simplex. The four methods—common cover, k-ary randomized response, optimal
row-wise LDP, and simplex-completed CAPT—share the identical block class,
common decoder, cost, and objective weights.

The run saves one independently verified certificate per method, all four
channels, a group-level uncertainty audit table, an exact comparison table,
deterministic PDF/PNG bars for utility gain and input dependence, resource
metadata, and a concise interpretation report. The first condition is a gate:
expand nested `D_cert`, epsilon, and L only after checking whether the optimized
simplex-CAPT channel is non-input-independent.

## Utility-aware channel-class diagnostic

If the fixed simplex-completion design is input-independent even without
privacy, run the focused follow-up:

```bash
uv run capt12 utility-design \
  --config configs/criteo_utility_design.yaml
```

The command freezes the same mapper, encoder, reference model, expected
support, and full certificate population. It first screens five fixed channel
classes using `D_const`, `D_free`, `G_info`, row argmins, and no-privacy row TV.
The current L=16 design is retained as a negative control but is not re-solved
when it fails the gate. Passing classes are solved once at epsilon 1 as both
optimal row-wise LDP and full-simplex CAPT, followed by robust and independent
certificate verification.

Outputs include `utility_design_screen.csv`, `utility_design_privacy.csv`, a
deterministic PDF/PNG diagnostic, per-design channels and design arrays, one
certificate per solved class, resource metadata, and an interpretation report.
Do not start an epsilon/L/sample grid unless CAPT has positive row TV, strictly
better utility than optimal LDP, positive gain over the best constant channel,
and a valid certificate. A full-simplex/full-simplex edge makes CAPT exactly
row-wise LDP, so utility-aware design can repair constant-channel degeneration
without by itself producing a CAPT-over-LDP advantage.

## Public-context stratified diagnostic

Run the prescribed one-condition follow-up with:

```bash
uv run capt12 context-stratified \
  --config configs/criteo_context_stratified.yaml
```

The run keeps `features_kv_bits_constrained_2`, epsilon 1, and the frozen
partition/decoder family. It solves the joint weighted k-medoids `L=16`
primary design and the singleton/identity `L=K=64` positive control. Missing
and unseen sensitive values are mapped by the frozen runtime mapper to the one
coarsened secret `__UNKNOWN__`. A separate channel is optimized for every
frozen public-context value, using only `Z`, profile, and public `B` online.

Required comparisons are the best context-specific constant, context-specific
optimal LDP, and context-specific CAPT. Shared optimal LDP and shared CAPT are
supplementary baselines. The output records full-simplex edge context count and
design mass, strict CAPT-over-context-LDP count and mass, aggregate distortion,
row TV, table size, resource use, and expected-randomized D_test log loss. AUC
and calibration are calculated from the expected prediction under mechanism
randomness; no downstream model is retrained.

Each context gets an independently verified certificate with alpha divided by
the frozen number of context values. Every certificate binds the unified
sensitive mapper and the complete context-channel manifest. The experiment is
a gate: expand epsilon or L only if context CAPT strictly improves on
context-specific optimal LDP with valid certificates.

## Full-scale command

```bash
uv run capt12 run-grid \
  --config configs/criteo_main.yaml \
  --fixed-model ctr_model \
  --phi-list hash,ctr_quantile \
  --K-list 64,256,4096 \
  --profiles 'features_kv_bits_constrained_2;features_kv_bits_constrained_3;features_kv_bits_constrained_2+features_kv_bits_constrained_3' \
  --privacy-scope tuple_adjacent \
  --epsilon-list 0.1,0.25,0.5,1.0,2.0 \
  --distortion-list bernoulli_kl,brier,retention \
  --partition-list frequency_balanced,score_quantile,risk_utility \
  --decoder-list pi0_conditional,uniform_within_block \
  --mechanism-list common_cover,k_ary_rr,scalar_keep_or_cover,capt_block,capt_full \
  --L-list 16,32,64,128 \
  --confidence cp_box --alpha-cert 0.05 \
  --shift-tv-list 0,0.01,0.05 \
  --seeds 0,1,2,3,4 --target-ctr-list 0.001,0.005,0.01 --resume
```

Invalid `L>K` combinations are skipped with a recorded reason. Full LP runs
above `K=256` are absent unless `--force-full`; they are never extrapolated.

### Local resource estimate (2026-08-15 smoke calibration)

The command above expands to 32,400 candidates; 2,700 invalid `L>K`
combinations are reasoned skips, leaving 29,700 candidate configurations. A
five-split, 500-row-per-split Criteo smoke with seven mechanisms and three
profiles took about 20 seconds on the local machine. This is a functional
smoke calibration, not a linear full-scale runtime forecast: the current grid
runner refits each configuration and should be run in phased sweeps with
`--resume`, not as one desktop invocation.

Deep pandas memory on the selected ten columns was 133.9 bytes/row in a bounded
sample, or about 1.20 GiB for the 9,634,806 local rows before preprocessing,
model, solver, and copies. A practical full-data run should budget at least
4–8 GiB per worker. With 386 ordered adjacent pairs (the smoke profile scale),
the sparse coefficient/workspace estimator reports approximately 0.036 GiB at
dimension 64, 0.142 GiB at 128, 0.568 GiB at 256, and 145.5 GiB at 4096. The
last case has 16,777,216 variables and is therefore skipped by the default
`full_max_k=256`; it is not an invitation to use `--force-full` on a desktop.
