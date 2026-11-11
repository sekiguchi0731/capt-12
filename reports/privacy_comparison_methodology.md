# CAPT / neural MaSS comparison methodology

## Comparison question and privacy interpretation

Compare held-out CTR utility with CAPT/LDP and with a neural token adaptation
of MaSS under each method's own privacy objective. MaSS is not calibrated to
a CAPT epsilon budget. Its m/n controls and classifier-based MI estimates
must never be described as CAPT certificates or equated numerically to epsilon.

## Common data and evaluation contract

Reuse the same frozen encoder Phi0, K-token input/output alphabet, public
context/profile, contribution policy, temporal splits, fixed CTR service and
paired seeds. L and the partition/decoder constrain CAPT only. MaSS models
are trained independently of L and reused across CAPT references. Actual
protected values are offline training labels and are absent from inference.
All MaSS fitting and pretraining reads D_design, never D_cert or D_test.
D_test is used only for evaluation. Fixed f_ref evaluation is the chosen CTR
protocol; it differs from refitting a useful-attribute classifier on transformed
data as done in the original MaSS experiments.

## Neural MaSS learning

The reference is jpmorganchase/MaSS at commit
6fbe9be1ff155b3fdd2677de7977d801d10000b5 (Apache-2.0), particularly
models/audiomnist/gen_model.py, models/audiomnist/disc_model.py,
models/trainer.py, utils/losses/infonce.py, and engine/mass_engine.py.

A Gaussian-noise residual generator takes the original integer token and
public context and generates a K-way categorical distribution. Attribute
MLPs estimate MI through entropy minus cross-entropy, using the original
sensitive/useful violation directions and quadratic-plus-absolute penalties.
A single shared InfoNCE MLP evaluates original/transformed views with only
cross-view negatives. Attribute networks are first pretrained on original
tokens with their labels; InfoNCE is pretrained with identity pairs. Their
weights initialize joint learning. Each minibatch first updates the generator
and InfoNCE network, then updates attribute networks by supervised CE on the
same detached outputs drawn before the generator update. This replaces the
previous probability-table/exact-MI training in the dedicated runner.

A nominal token embedding and categorical output head are interface
adaptations. A hard integer sample has no pathwise derivative, so generator
gradients use the score-function estimator with a baseline independent of the
current sampled outputs. The log probability is summed over the complete
batch; negative InfoNCE outputs also depend on the generator. A tiny fully
enumerated test checks that the expected gradient matches the exact coupled
batch objective. Attribute/feature networks see the actual hard token, not a
soft vector containing information that is absent from the released output.

The default conditional mode uses H(A|C)-CE(A|O,C), C=(B,S), and within-context
InfoNCE negatives. The explicit unconditional mode uses H(A)-CE(A|O) and
unconditioned negatives, as in original MaSS. Both remain identifiable in
metadata. A finite penalty requests the MI target; it does not prove target
attainment. Negative H-CE estimates are saved without clipping.

The CTR-cost extension is opt-in with mass_ctr_weight (default 0). Its weight
is independent of the MI and InfoNCE terms. Main training counts complete
dataset epochs. Pretraining controls count sampled-batch updates, as recorded
in the README and provenance. All initial/final networks, labels/context
vocabularies and histories are saved in a reloadable .pt checkpoint.

## No CAPT postprocessing of MaSS

There is no shared decoder, medoid restriction, confidence-box optimization,
common-cover mixture, or certificate repair on the MaSS path. Gaussian input
noise and categorical sampling are part of MaSS itself. The runner emits
only mass12_raw with mass_backend=neural_token. No mass12_calibrated result or
MaSS certificate is generated. Results validation rejects neural rows with
post-training cover or CAPT certification claims.

## Evaluation and reporting

For CTR evaluation, average the categorical probabilities over a reproducible
Monte Carlo sample of Gaussian input noise. Record the sample count, seed and
model SHA; this estimates utility and is not an exact privacy channel.
User-day bootstrap intervals condition on the evaluation noise sample and do
not include its Monte Carlo error. Increase the sample count for sensitivity
analysis. Inference itself uses fresh noise, not a frozen approximation.

MaSS result fields report requested m/n, classifier MI estimates and CE,
pre/post design-probe losses, held-out CTR metrics, and zero cover. L identifies
the CAPT reference and mass_output_count=K identifies MaSS capacity. No MaSS
point is placed on a certified epsilon axis. The neural pilot gate uses MI
estimate variation across controls, not raw/certified epsilon. CAPT/LDP retain
their existing independent certificates. The legacy finite-table and PBP
utilities are preserved for historical diagnostics but are not the neural
MaSS comparison path.
