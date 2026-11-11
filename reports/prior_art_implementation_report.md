# Prior-art comparison implementation report

## Current neural MaSS path (2026-11-11)

The dedicated runner uses comparison/mass_nn.py. It implements a
Gaussian-noise residual generator, pretrained attribute MLPs, a pretrained
shared InfoNCE MLP, CE-based MI penalties, and alternating generator/feature
then attribute updates following the official MaSS learning structure. The
upstream source is pinned to 6fbe9be1ff155b3fdd2677de7977d801d10000b5.

MaSS receives the frozen input token directly and can generate all K output
tokens. CAPT L and its medoid decoder are absent from the model and training
signature. Models are reused across CAPT L values. The integer interface
requires a categorical output head and score-function generator gradients;
these adaptations, the public-context conditioning option, model sizes and
training/evaluation controls are explicit in provenance.

No post-training common-cover mixture or MaSS CAPT certificate is generated.
Only mass12_raw with mass_backend=neural_token is produced. Its own m/n and
classifier MI estimates are saved separately from CAPT epsilon fields.
Prediction/CTR evaluation integrates the categorical output analytically and
uses Monte Carlo for Gaussian input noise. This approximation is identified
in model metadata and does not change runtime sampling.

.pt checkpoints save all three network types, pretrained initial states,
configuration, context/label vocabularies, entropy statistics, fixed design
probe indices, and pre/post histories. Models reload through MassNN.load.

The old finite-table utilities and PBP calibration helpers remain available
for historical diagnostics; they are not selected by the dedicated MaSS
runner. Existing historical measurement files are not rewritten.

## Verification

Tests compare the neural CE penalties and cross-view InfoNCE with the official
formulas. A fully enumerated tiny categorical problem verifies the expected
generator score-function gradient, including in-batch negatives. Tests cover
pretrained initialization transfer, alternating network updates, hard-token
inference, checkpoint roundtrip, direct K-way output, missing postprocessing,
and real CLI CAPT/MaSS integration. The multi-L CLI smoke checks one shared
MaSS checkpoint and identical MaSS CTR results across CAPT L values.

Synthetic smoke measurements are implementation checks, not Criteo research
results. No full Criteo experiment is claimed here.
