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
uv run capt12 reference-benchmark \
  --config configs/criteo_context_stratified_l32.yaml \
  --output outputs/reference_benchmarks/ctr_logistic_quantile_k64_dtest.json
```

`configs/criteo_main.yaml` names anonymized columns as *candidate protected
attributes* or *sensitive proxy attributes*. It never assigns real-world
semantics such as age or region to them. Five temporal splits are disjoint:
model fit, design, certification, attack training, and final testing.

The Criteo default encoder is supervised but frozen: it fits a sparse logistic
CTR score only on `D_model`, using inference-available scalar fields while
excluding the declared protected columns, outcomes, delayed outcomes,
unavailable features, row/user IDs, and list-valued fields. Integer/string fields are
nominal one-hot inputs and floating-point fields are imputed and standardized.
The score is quantized at `D_model` quantiles into `K` tokens; the categorical
token plus public context is then calibrated by the ordinary `f_ref`. The
`reference-benchmark` command evaluates this unsanitized K-token predictor on
the untouched temporal `D_test`. The previous four-column whole-row hash is
available as `configs/criteo_context_stratified_l32_legacy_hash.yaml`.

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

For example, suppose a sensitive attribute has the following group counts:

```text
A = a     1000
A = b      800
A = c        3
A = d        2
```

The groups `c` and `d` are extremely sparse, so their finite-sample confidence
sets can become very wide. A merge_to_other policy would coarsen these rare
values into a shared fallback category:

```text
c -> __OTHER__
d -> __OTHER__
```

so that the resulting grouped count becomes:

```text
__OTHER__ = 5
```

The important requirement is that this coarsening rule must be frozen and
applied identically at certification time and at runtime. If `c` is treated as
`__OTHER__` when constructing the certificate but is treated as `c` itself
during deployment, the certified privacy guarantee would no longer correspond to
the actually deployed mechanism. This is why `merge_to_other` is currently
disabled until the same frozen coarsening map can be included in the deployment
certificate and enforced at runtime.

This version makes the reason for disabling `merge_to_other` explicit, rather
than only stating that a frozen runtime coarsening map is required.

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
  --mechanism-seed-mode per-seed \
  --frozen-design-seeds 0,1,2,3,4 \
  --utility-objective teacher_kl \
  --representation-mode teacher_kl_fixed
```

`--mechanism-seed-mode per-seed` rebuilds the complete mechanism for every
listed design seed, including the encoder, reference model, partition,
decoder, context-specific `R` matrices, and certificates. To select one
canonical deployable mechanism and optimize it only once, use
`--mechanism-seed-mode fixed --fixed-mechanism-seed SEED`. Fixed mode does not
copy a numeric `R` into designs from other seeds; it fixes the complete design
at the canonical seed and reuses an already completed matching run.

For the prescribed L=32 condition, the two execution policies are:

```bash
# Full design-sensitivity run: reoptimize R as part of every seeded mechanism.
uv run capt12 context-seed-stability \
  --config configs/criteo_context_stratified_l32.yaml \
  --mechanism-seed-mode per-seed \
  --frozen-design-seeds 0,1,2,3,4 \
  --utility-objective teacher_kl \
  --representation-mode teacher_kl_fixed

# Canonical deployment run: optimize/certify the whole mechanism once at seed 0.
uv run capt12 context-seed-stability \
  --config configs/criteo_context_stratified_l32.yaml \
  --mechanism-seed-mode fixed \
  --fixed-mechanism-seed 0 \
  --test-seeds 0,1,2,3,4 \
  --sensitive-domain-policy closed-domain \
  --utility-objective teacher_kl \
  --representation-mode teacher_kl_fixed
```

`--sensitive-domain-policy closed-domain` freezes every protected-attribute
domain from `D_model`, omits the artificial `__UNKNOWN__` protected value from
the Cartesian certificate support, and rejects a missing or unseen protected
value on every later split. The default config remains the conservative
open-world `unified-unknown` policy; it can also be selected explicitly with
`--sensitive-domain-policy unified-unknown` (alias `open-world`). The chosen
policy is part of the run ID, runtime-mapper hash, channel manifest, and every
context certificate.

The fixed command returns the ordinary single-run `sol_review_bundle.zip`;
the per-seed command returns the aggregated
`sol_seed_stability_review_bundle.zip`. Choose the fixed seed before inspecting
`D_test`; fixed mode is a single-mechanism result, not a seed-stability claim.
When `--test-seeds` is present, the fixed mechanism is still optimized only
once. The additional seeds draw released tokens on the same fixed `D_test`
rows using common random numbers for CAPT and LDP, and write per-seed sampled
log loss, ROC-AUC, PR-AUC, ECE, paired differences, and summaries under
`outputs/context_fixed_test_seed_evaluations/`. The primary expected randomized
log loss and expected-score metrics remain analytic and seed-invariant; the
seeded table is a Monte Carlo variation diagnostic.

To compare any subset of the private mechanisms, non-private controls, and
the D_test null reference without rebuilding the fixed CAPT run, use:

```bash
uv run capt12 context-fixed-test-seeds \
  --run outputs/context_stratified_l32_runs/RUN_ID \
  --test-seeds 0,1,2,3,4 \
  --methods capt,block-ldp,rr,context-token-ldp,nonprivate-k64,nonprivate-l32,constant
```

`--methods all` selects the same seven conditions. `nonprivate-k64` is the
identity release on the original K-token alphabet; `nonprivate-l32` keeps the
run's fixed L-block compression/decoder but removes privacy randomization.
`constant` predicts the empirical D_test prevalence and is an evaluation-only
LLHCompVN reference, not a deployable mechanism. `context-token-ldp` solves an
unrestricted KxK epsilon-LDP channel independently for each public context.
Those LPs are checkpointed per context under
`outputs/context_token_ldp_channels/`; rerunning the command resumes unfinished
contexts. They can also be precomputed explicitly with:

```bash
uv run capt12 context-token-ldp \
  --run outputs/context_stratified_l32_runs/RUN_ID \
  --solver-time-limit 1800
```

The older `--baselines block-ldp,rr` selector remains available for backward
compatibility but cannot be combined with `--methods`. The same selectors are
available as `--test-methods` and `--test-baselines` on fixed-mode
`context-seed-stability` runs.

The R LP objective supports `teacher_kl`, `empirical_logloss` (D_design click
labels only), and `hybrid_logloss_kl`; the hybrid additionally accepts
`--hybrid-empirical-weight 0.5`. Partition/decoder construction is selected
independently with `--representation-mode`:

- `teacher_kl_fixed` preserves the original teacher-KL partition/decoder. It
  is the R-objective-only ablation when paired with an empirical or hybrid R.
- `objective_aligned` builds the common partition/decoder from
  `C_repr(z,o)=sum_b P(b|z) C_b(z,o)`, using the same selected utility estimand
  that each public-context R LP uses.

Thus the cost-matched empirical main condition is
`--utility-objective empirical_logloss --representation-mode
objective_aligned`; the corresponding hybrid condition uses
`--utility-objective hybrid_logloss_kl --representation-mode
objective_aligned`. Support, adjacency, confidence sets, and certificate
definitions do not change between these modes. Never select the objective,
representation mode, or hybrid weight from D_test.

The seed-stability command keeps one requested epsilon and L fixed. It reuses completed runs
from the same source commit, writes per-seed and across-seed CSVs plus one
stability figure/report and D_attack_train-fixed privacy lower audits under
`outputs/context_stratified_seed_summaries/`, and
creates `sol_seed_stability_review_bundle.zip` containing all constituent
mechanisms and certificates for one-file review.

Run a matched positive-epsilon grid with the objective-aligned empirical
log-loss design as follows:

```bash
uv run capt12 context-epsilon-grid \
  --config configs/criteo_context_epsilon_grid.yaml \
  --epsilon-values 0.5,1,2 \
  --frozen-design-seeds 0,1,2,3,4
```

The command runs or reuses one complete seed-stability family per epsilon,
then writes paired epsilon-by-seed tables, a deterministic figure/report, and
`sol_context_epsilon_grid_bundle.zip` under
`outputs/context_epsilon_grids/`. It verifies that each paired seed has the
same encoder, partition, and decoder hashes across epsilon, and that every
serialized channel passes its own conservative pure-epsilon certificate.
Epsilon must be strictly positive: the current deterministic full-support
repair obtains strict floating-point slack from `exp(epsilon)-1`; epsilon zero
requires a separate exact-equality mechanism and is intentionally rejected.

Render the compact paper figure from a completed empirical L=16 epsilon grid
and the available certified L=8/L=16 matched-cost bundles with:

```bash
uv run capt12 context-paper-figure \
  --epsilon-grid-bundle outputs/context_epsilon_grids/<id>/sol_context_epsilon_grid_bundle.zip \
  --block-comparison-bundle outputs/context_cost_comparisons/<L8-id>/sol_context_cost_comparison_bundle.zip \
  --block-comparison-bundle outputs/context_cost_comparisons/<L16-id>/sol_context_cost_comparison_bundle.zip
```

Panel (a) uses the conservative realized robust certificate epsilon on the x
axis. Panel (b) pairs frozen-design seeds across the available block sizes at
target epsilon one. Both panels report held-out expected randomized log-loss
gain over the context-specific optimal-LDP baseline. The renderer rejects
invalid certificates and mismatched objectives, seed families, source SHAs,
or encoders; it does not treat seed spread as a confidence interval.

For the stronger end-to-end baseline, solve one unrestricted global K=64
token-level optimal-LDP channel per paired seed and privacy budget, then
compare the same CAPT results against it:

```bash
uv run capt12 context-global-token-ldp \
  --epsilon-grid-bundle outputs/context_epsilon_grids/<id>/sol_context_epsilon_grid_bundle.zip \
  --block-comparison-bundle outputs/context_cost_comparisons/<L8-id>/sol_context_cost_comparison_bundle.zip \
  --block-comparison-bundle outputs/context_cost_comparisons/<L16-id>/sol_context_cost_comparison_bundle.zip
```

This baseline has a singleton partition, identity decoder, and an unrestricted
64 by 64 channel. It is global: public context is used to aggregate the frozen
empirical-log-loss objective but is not an online channel selector. The command
reuses the frozen encoder, objective, and probability grid embedded in the
certified epsilon-grid bundle, loads only D_test, solves the 15 LDP LPs
sequentially, directly verifies the released channels, and emits a main-figure
candidate plus a compact Sol review bundle.

For empirical and hybrid objectives, plots normalize the CAPT-LDP gain by the
excess objective after subtracting the channel-invariant empirical entropy
floor. Exact raw objective, floor, excess objective, and absolute gain in
micro-objective-units per display are all retained in the CSVs; this reporting
normalization does not alter the LP.

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
Before every HiGHS call, nonzero inequality rows whose largest coefficient is
below one are scaled upward to unit magnitude; rows are never scaled downward,
so the effective original-space feasibility tolerance is not weakened. If the
usual HiGHS and strict IPM paths return a time or numerical status, the same
scaled LP is retried with crossover enabled and finally with explicit primal
simplex. Presolve choices may change between paths, but the objective,
constraints, bounds, and feasibility/optimality tolerances do not. Every
solver-reported optimum is recomputed against the original unscaled LP before
and after channel cleanup, and is rejected if any equality, inequality, or
bound residual exceeds the configured tolerance. A retry result is never
accepted without the usual support-oracle and independent certificate
verification stages.
If all solver paths reject the auxiliary-variable shared-support master, CAPT
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
columns are declared or inputs fall outside `phi_source_cols`. The built-in
token-conditioned `f_ref` treats the integer hash-token ID as an uncapped
categorical feature and records this schema plus the reference-model hash in
downstream context artifacts. Pre-schema context artifacts must be regenerated;
they are not mixed with categorical-token results.

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

## Prior-art comparison with neural MaSS

The dedicated `prior-art-comparison` runner now uses neural MaSS. Install its
optional dependency with `uv sync --extra mass`. The original MaSS source is
[JPMorgan Chase MaSS](https://github.com/jpmorganchase/MaSS), pinned to
`6fbe9be1ff155b3fdd2677de7977d801d10000b5` (Apache-2.0).

MaSS takes the frozen input token Z directly. A residual MLP with Gaussian
input noise generates logits over all K output tokens, then samples one
integer O. It uses no CAPT partition, medoid, shared decoder, confidence set,
or post-training cover. `--K`/`--input-count` and `--input-bits` set the common
input/transmitted-token alphabet for CAPT and MaSS. `--L`/`--capt-blocks`,
`--L-list`, and `--capt-block-bits` control CAPT only. The historical aliases
`--output-count` and `--output-bits` also refer only to CAPT blocks.

```bash
capt12 prior-art-comparison \
  --config configs/criteo_prior_art_privacy_matched.yaml \
  --K 32 --L 3 --phase pilot
```

The config filename is retained for compatibility; the MaSS result is no
longer a privacy-matched CAPT certificate. The runner emits `mass12_raw` with
`mass_backend: neural_token`, and never emits `mass12_calibrated`. Its result
column L identifies the paired CAPT reference, while `mass_output_count=K`
records MaSS's output capacity. A model is trained once per data/seed/control
cell and reused across CAPT L values. Models contain no L or decoder.

The learning structure follows official MaSS: supervised pretraining of
attribute-prediction MLPs on the original Z, identity-pair pretraining of a
shared InfoNCE MLP, then alternating generator/InfoNCE updates and supervised
attribute-network updates on transformed tokens. Attribute MI is estimated
as entropy minus classifier cross-entropy; exact finite-channel MI is not
used in this training. Sensitive violations use `min(CE-H+m,0)` and useful
violations use `max(CE-H+n,0)`, with the upstream squared-plus-absolute penalty.
Both InfoNCE directions use cross-view-only in-batch negatives.

Two interface adaptations remain explicit. Nominal token IDs use learned
input embeddings, and the original continuous output is replaced with a
K-way categorical head. Gradients through its hard samples use an unbiased
score-function estimator with a baseline based only on past batches. The
joint batch log probability includes every sampled output, including InfoNCE
negative candidates. The Gaussian input noise is retained. These are learned
randomization inside MaSS, not post-training CAPT randomization.

All training/pretraining uses `D_design`. Online inputs are Z and public
context/profile only; sensitive values are labels, never generator inputs.
`--mass-mi-conditioning conditional` (default) uses H(attribute|B,S) and
context-aware prediction networks; `unconditional` uses the original
unconditioned entropy/MI form. Conditional InfoNCE draws negatives within
public contexts. This choice is recorded in model provenance.

Main training defaults to 50 full-data epochs, batch size 128, AdamW with
base learning rate 0.0005 (generator/attribute joint-training LR is scaled by
batch_size/256), weight decay 0.001, and cosine decay. Token embeddings are
32-dimensional; attribute hidden widths are 512/256 and the InfoNCE encoder
has two 512-wide layers by default. `--mass-epochs`, `--mass-batch-size`,
`--mass-learning-rate`, `--mass-weight-decay`, `--mass-token-dim`, and
`--mass-hidden-dim` expose these controls.

Attribute and InfoNCE pretraining each default to 100 sampled-batch updates;
these stages do not count complete dataset passes. They are controlled by
`--mass-attribute-pretrain-epochs`, `--mass-infonce-pretrain-epochs`, and
`--mass-infonce-pretrain-learning-rate`. InfoNCE also supports weight,
temperature, batch-size, embedding-dimension and direction controls. Zero
pretraining updates selects random initialization; zero InfoNCE weight skips
its pretraining. Pretrained network states initialize joint training, and
optimizer state is reset between stages.

`--mass-m-list`, `--mass-n-list`, `--mass-privacy-weight-list`, and
`--mass-utility-weight-list` override both pilot and full grids. The four
quadratic/linear penalty coefficients remain separately configurable with
`--mass-{privacy,useful}-{quadratic,linear}-weight`. The optional CTR extension
uses `--mass-ctr-weight` (default 0); both settings are evaluated on held-out
CTR log loss with the same fixed f_ref.

Each `models/mass-nn-*.pt` contains generator, attribute and shared InfoNCE
network weights, all pretrained initial states, configuration, label/context
vocabularies, entropy statistics, evaluation-probe indices and histories.
`MassNN.load(path)` reconstructs the frozen model. A JSON sidecar identifies
the upstream source, estimator and interface adaptations. Main history CSVs
save pre/post measurements on the same fixed design probe and random seed,
plus optimization/discriminator losses. Separate pretraining CSVs save both
stages' pre/post losses. Negative H-CE estimates are retained; they indicate
classifier approximation error rather than negative true information.

MaSS runtime sampling uses fresh Gaussian noise. For expected CTR evaluation,
`--mass-eval-noise-samples` (default 64) averages categorical probabilities
over a reproducible Monte Carlo sample of that noise. This is a numerical
utility estimate, not a replacement mechanism or exact privacy channel.
Bootstrap intervals are conditional on that evaluation noise sample; they do
not quantify its Monte Carlo error. Model SHA and evaluation controls are
saved with each result.

### GPU execution on the remote GTX 1080 Ti host

`--mass-device cpu|cuda[:visible-index]` selects the MaSS training, runtime
sampling, and utility-evaluation device. CPU remains the default. CUDA is
probed before Criteo loading/CAPT optimization and an unusable build fails
explicitly. CAPT optimization and categorical attack statistics stay on CPU.
Only minibatches move to CUDA; full reference datasets stay in host memory.
Training epochs are reported in the progress log. Checkpoints retain CPU
tensors and their training-runtime metadata; loading defaults to CPU, even
for GPU-trained models. CPU and GPU RNG streams/numerics need not coincide.

On hpc5000-xsl only physical GPUs **2 and 3** are allocated to this experiment.
Set `CUDA_VISIBLE_DEVICES` before Python starts. With `CUDA_VISIBLE_DEVICES=2`,
`cuda:0` means physical GPU 2. With `CUDA_VISIBLE_DEVICES=3`, the same `cuda:0`
means physical GPU 3. With the mask `2,3`, they become visible indices 0 and 1.
One process uses one selected GPU; this is not distributed multi-GPU training.
The implementation does not fork/seed CUDA RNG state on every installed GPU.

The reported torch 2.15.0+cu132 build does not work with this host's driver
555.42.06/Pascal cards. Use the official torch 2.6.0 CUDA 12.4 wheel in a
separate `.venv-pascal` environment. The setup script locks base dependencies
from uv.lock and installs that optional wheel from PyTorch's CUDA 12.4 index;
the ordinary `.venv` is preserved. Invoke `.venv-pascal/bin/python` or
`.venv-pascal/bin/capt12` directly, because a later `uv run/sync --extra mass`
would reselect the generic torch version. Driver changes are not needed for
this configuration, subject to the actual kernel/training probe below.

Run long remote commands with **nohup**. From the repository root:

```bash
mkdir -p logs
nohup bash tools/setup_mass_pascal.sh \
  > logs/mass-pascal-setup.log 2>&1 < /dev/null &
```

After setup reports `Ready`, probe a small neural fit, Gaussian/categorical
sampling, MC utility channel, and GPU/CPU checkpoint round trip:

```bash
nohup env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv-pascal/bin/python -u tools/check_mass_cuda.py --device cuda:0 \
  > logs/mass-gpu2-check.log 2>&1 < /dev/null &

nohup env CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv-pascal/bin/python -u tools/check_mass_cuda.py --device cuda:0 \
  > logs/mass-gpu3-check.log 2>&1 < /dev/null &
```

These probes need no Criteo data. After `status: ok`, the focused full-alphabet
pilot uses K=4096, CAPT L=16, empirical logloss, seed 0, 50 MaSS epochs, and
m in {0, 0.1, 0.5}, with the configured data path
`data/CriteoPrivateAd_release/data`:

```bash
nohup env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venv-pascal/bin/capt12 prior-art-comparison \
  --config configs/criteo_prior_art_gpu_pilot.yaml --phase pilot \
  > logs/mass-gpu-pilot.log 2>&1 < /dev/null &
```

The pilot is a real-data experiment, not a small-data smoke test. For saved
models, `evaluate-prior-art-attacks --mass-device cuda:0` selects CUDA inference
without retraining and writes a separate evaluation directory. Its device is
part of the attack cache identity, since sampled CPU/GPU outputs may differ.

Compatibility sources: [PyTorch 2.6 official wheel commands](https://docs.pytorch.org/get-started/previous-versions/),
[NVIDIA CUDA 12.4 driver requirements](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/),
and [CUDA 13 removal of Pascal support](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html).

MaSS has no CAPT upper certificate and undergoes no common-cover calibration.
Its configured m/n values and learned classifier MI estimates are not achieved
privacy guarantees. Results always mark neural MaSS as `certificate_valid=false`
and `formal_comparable=false`, leave epsilon/certificate fields empty, and
record zero post-training cover. Figures show its MI/CTR results separately
from the CAPT/LDP certified-epsilon axis. The pilot gate checks variation in
the classifier MI estimate, not variation in certified epsilon. A full grid
requires a completed pilot from the same source/config controls unless the
explicit pilot gate is disabled in config.

The finite-table utilities in `comparison/mass12.py` and their tests are
retained only for historical diagnostics. The dedicated comparison runner
cannot select the old `finite_block` backend. Historical result files remain
renderable with their original metadata; rerunning the neural comparison
requires new training.

### Shared empirical attribute-inference evaluation

`prior-art-comparison` also evaluates CAPT, LDP, and MaSS with the same
`likelihood_ratio` attack. It estimates categorical output frequencies from
`D_attack_train` within each fixed public context and selected profile, and
compares ordered attribute tuples differing in one coordinate (or all pairs
when `joint_all_pairs` is configured). It does not try to recover the original
input token. MaSS attacks use actual Gaussian-plus-categorical runtime draws,
not its Monte Carlo utility channel. Generation is streamed to bound memory.

Both attack and test data retain one display per user-day. A stable user-day
hash divides `D_attack_train` into frequency-fit and threshold-calibration
parts. The default add-one smoothing is fixed before evaluation. A calibrated
likelihood-ratio threshold targets FPR=0.01; randomized decisions at tied
scores permit a categorical attacker to reach that target on calibration data.
This randomization changes only the attacker's decision, never the released
token. The test FPR is measured and can differ from the target. Test labels
do not choose the attack, thresholds, smoothing, or eligible comparison family.

Controls are `--attack-method likelihood_ratio|none`, `--attack-smoothing`,
`--attack-target-fpr`, `--attack-confidence`, `--attack-calibration-fraction`,
and `--attack-seed`. The matching config names have the `attribute_attack_`
prefix. `ATTACK_FACTORIES` in `comparison/attribute_inference.py` separates
the score-fitting backend from sampling, calibration, and evaluation, so a
different token-scoring attack can replace it later.

Each `attribute_attacks/<id>/` saves `pairs.csv` (per-context ordered-pair
counts, calibrated thresholds, tie probabilities, AUC, actual TPR/FPR and
intervals), `roc.csv` (all frozen score thresholds), `attacker_scores.npz`,
`roc.png/pdf`, and `summary.json`. Clopper-Pearson operating-point rate
intervals use Bonferroni correction across both rates and all frozen pairs
within that mechanism cell, under the declared `user_day_iid` assumption;
they are not simultaneous across the whole experiment grid. AUC and ROC
interpolation do not carry a worst-case or finite-sample optimality guarantee.

Missing fit/calibration groups are explicitly ineligible. Missing test groups
are reported as incomplete coverage; they are never assigned zero leakage.
Mean and largest observed pair AUC/TPR/FPR appear in results.csv only when
all eligible pairs are evaluated. The eligible-family hash and counts are
recorded for comparison. `attribute_inference_privacy_utility.png/pdf` uses
common empirical axes for all methods; TPR points use calibrated thresholds
and must be read together with the realized FPR in pairs.csv. These are attack
measurements, not MI estimates or epsilon certificates. The existing CAPT
certificate and MaSS training objectives are unaffected.

Saved mechanisms can be evaluated again with new attack settings:

```bash
capt12 evaluate-prior-art-attacks \
  --run-dir outputs/prior_art_comparison/runs/<run-id> \
  --attack-smoothing 0.5 --attack-target-fpr 0.05
```

This writes a separate `attack_evaluations/<id>/` directory and does not
retrain or overwrite the original models, certificates, or comparison table.
The original training and current evaluation source revisions are recorded.

The evaluation applies the Neyman-Pearson viewpoint in
[Liu et al., PoPETs 2019](https://petsymposium.org/popets/2019/popets-2019-0045.pdf):
Section 2.2, Definition 3 / Equation (1); Section 3, Theorem 2 and the PR/ROC
correspondence; Section 5.3.1 on optimal versus realistic attackers; and
Section 6 on comparing privacy notions. Here the distributions are estimated
from finite data and the hypotheses concern conditional attribute values,
rather than reproducing that paper's analytic comparisons of DP frameworks.
No epsilon conversion or hypothesis test of a claimed privacy budget is
performed by this empirical evaluation.

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
