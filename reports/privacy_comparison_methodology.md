# Privacy-matched prior-art comparison methodology

## Comparison question

The formal question is whether CAPT-12 attains equal or better held-out CTR utility while satisfying the same robust profile-privacy condition, especially `robust upper epsilon <= 1`. Nominal MI parameters, attack estimates, and lower audits are never substituted for this upper certificate.

## Frozen data and online contract

All methods must reuse the same Criteo contribution table, one-display-per-user-day policy, temporal split IDs, fixed 12-bit encoder, `L=16` objective-aligned partition/decoder, ordered adjacency, hybrid paths, reference CTR service, costs, output budget, and seeds. Training reads only `D_design`; formal confidence objects read only `D_cert`; attack selection reads only `D_attack_train`; reporting reads `D_test`. The code rejects PBP/MaSS fitting calls whose declared source split is not `D_design`, rejects attack fitting outside `D_attack_train`, and rejects held-out attack evaluation outside `D_test`.

The deployable inference signature is `Q(o|z,b,s)`. Sensitive values may label MaSS adversaries during training but are absent from `Mass12FiniteChannel.block_probabilities` and `.channel`. Profile-indexed PBP requires `A_S` to select `Q_g`, so it is always an oracle.

## Formal robust upper epsilon

For every ordered edge and output, the existing confidence-box support solver computes the numerator supremum and denominator infimum with no denominator floor. Positive numerator over zero denominator is infinity; zero over zero contributes an impossible-output constraint with realized value minus infinity and zero additive violation. Ordered reverse edges are explicit adjacency entries. Tuple claims use only existing hybrid paths and sum their per-attribute epsilon values.

A row-stochastic frozen channel is formal-comparable only when its independent robust verification is valid and its realized upper epsilon is at most the selected target. Criteo point confidence is rejected by existing config and certificate code.

## PBP baselines

The oracle LP contains one stochastic matrix per nominal profile and constrains `p_g Q_g <= exp(epsilon) p_g' Q_g'` for every ordered edge/output. Its objective uses profile weights and the selected objective-aligned cost. It is never promoted to deployable or common-channel formal comparison.

The common nominal LP shares one channel across profiles and uses nominal distributions from `D_design`. It is frozen before being sent to the robust verifier. Passing only nominal constraints yields `uncertified_nominal_baseline`; passing the robust verifier promotes the exact frozen channel.

## MaSS-12 finite-output adaptation

For each frozen public context, including predeclared contexts absent from `D_design`, the adaptation learns logits over `L` blocks for every input token. Softmax probabilities are multiplied by the common decoder to enumerate `Q=pi D` exactly. No privacy verification samples from the mechanism.

Unlike the official neural MaSS implementation, this finite discrete adaptation computes `I(A;O|B,S)` exactly from `D_design` joint masses and applies the paper's same violation direction and `d^2+|d|` penalty. If useful labels and `n` are configured, their exact discrete conditional MI uses the symmetric lower-bound penalty. This is `MaSS-12 (finite-output adaptation)`, not official or original MaSS.

The main condition uses the MI penalties and InfoNCE without an additional CTR cost (`mass_ctr_weight: 0`, the default). The CTR-oriented extension independently adds `mass_ctr_weight * E[d_B(Z,O)]` for the selected teacher-KL, empirical-logloss, or hybrid cost; `--mass-ctr-weight` accepts any finite nonnegative coefficient. The useful-MI weight no longer scales this term. To recover an older cell's objective, set the CTR weight equal to that cell's useful-MI weight. All variants retain the same frozen CTR evaluation, cost-derived common decoder, and certification/calibration procedure. They remain finite-output adaptations even at CTR weight zero. Results, model/certificate metadata, and pre/post history record the CTR weight; distortion remains measurable when its training coefficient is zero. Resume and plotting distinguish weights, and pilot/full phases must use the same CTR weight.

The adaptation additionally preserves source information with conditional symmetric InfoNCE, the arithmetic mean of output-to-source and source-to-output losses. Within each public context, source tokens and their in-batch negatives are sampled iid from the design marginal. The reverse loss enumerates outputs through `Q=pi D` exactly. The forward loss samples an output batch and enumerates each output slot conditional on the other sampled outputs (Rao-Blackwellization). Channel gradients therefore include the effect of changing negative-output distributions; they are unbiased estimates of the full finite-batch expectation, not gradients that treat sampled negatives as fixed. Source/output randomness is separate and reproducible. Single-direction modes are available as ablations. History records both directional losses and their mean before/after each update, both seeds, and final embeddings are retained. The training InfoNCE estimate is not a privacy certificate.

Before the joint fit, the shared unit-normalized token embedding is pretrained with identity-pair InfoNCE on `D_design`: each original token instance is its own anchor/positive and other instances in the same public-context batch are negatives, including duplicates. This follows the upstream pretraining pattern while retaining the finite token lookup in place of its NN. The stage uses no sensitive/click labels, decoder outputs, `D_cert`, or `D_test`, and never changes the fixed encoder. The pretrained weights initialize joint MaSS training; optimizer moments are reset. Defaults are 100 updates (one batch per context per update) at learning rate 0.01, controlled by `--mass-infonce-pretrain-epochs` and `--mass-infonce-pretrain-learning-rate`. Epochs 0 selects the earlier random initialization and InfoNCE weight 0 skips pretraining. The calculation uses batch-sized similarities rather than a K-by-K identity channel. Pre/post pretraining losses and seeds are saved in a separate CSV, and the main-training initializer is saved alongside final embeddings in the model NPZ. Results, certificate metadata, resume/plot keys, and pilot/full checks retain the controls and completed update counts where applicable. Legacy result files are interpreted as no-pretraining runs.

The dedicated CLI overrides m/n and overall privacy/utility grids in both pilot and full phases. Quadratic and linear coefficients for sensitive and useful penalties are independently configurable; default coefficients of 1 reproduce the original adaptation's `d^2+|d|` shape. All values are recorded in results and model/certificate metadata.

Pilot controls are `mass_m_list`, `mass_n_list`, privacy/utility weights, training seed, epochs, finite-block mode, and temperature. A full grid is allowed only after raw achieved leakage changes across the pilot. The configured `m,n` values are never copied into achieved-privacy fields.

## Certified cover calibration

The cover distribution is frozen once from `D_design`, assigned a strictly positive probability floor, and shared across methods/seeds. For each raw channel, bisection finds the smallest `lambda` within `1e-8` such that the existing robust verifier reports upper epsilon at most 1. The endpoint is independently reverified. Artifacts record pre/post epsilon and utility, lambda, maximum per-input row TV change, and constant-channel status. The same wrapper applies to raw MaSS-12 and common nominal PBP. It is a CAPT comparison wrapper, not part of either prior paper.

The certificate bundle is factorized rather than a redundant 4096-by-4096 JSON table: it stores `pi`, the decoder, cover and per-context `lambda`, confidence boxes, and ordered adjacency. Independent verification checks hashes, reconstructs the exact square token channel for each context, and sends it to the existing robust verifier. This compression changes storage only, not the verified channel.

## Empirical diagnostic

Only if no useful certified MaSS point exists, the auxiliary metric is the `cross-fitted attack-CMI proxy`: baseline conditional cross-entropy minus output-conditioned cross-entropy on `D_test`. One-hot logistic attackers use identical candidate grids and user-day group folds; hyperparameters and fold ensembles are frozen using `D_attack_train` only. User-day cluster bootstrap supplies 95% intervals. Raw and zero-clipped values, attribute-wise values, worst attribute, and profile-mass-weighted means are separate fields.

The empirical plot uses only overlapping observed leakage ranges and does not extrapolate. Lower audit above 1 is a violation witness; a value at or below 1 is not a safety proof. No certificate upper/lower sandwich is claimed without a stationarity or shift bridge.

## Output and claim gates

The results schema refuses a `formal_comparable=true` row without a valid certificate, or a deployable row that uses `A_S` online. The main artifact uses at most two panels: an actual certified-epsilon/held-out-log-loss frontier and the paired L=8,16,32 sensitivity at epsilon 1. Frozen-design seed points and paths are shown without treating seeds as independent test samples. The detailed prior Figures A–E are Supplementary, and the empirical panel remains labeled `not a certificate`. Deterministic plotting fixes timestamps and metadata, and review packets use sorted members with fixed ZIP timestamps. A two-render hash check remains mandatory when execution is authorized.
