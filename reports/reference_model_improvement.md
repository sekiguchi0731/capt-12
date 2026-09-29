# K=64 non-private reference-model check

## Decision

The previous result was primarily an encoder problem, not evidence that
`K=64` is intrinsically too small. The default four-column whole-row BLAKE2b
hash was replaced by a supervised, frozen CTR-score quantizer. The first
literature-guided candidate improved the held-out metric substantially, so the
predeclared stop rule was met and no second model replacement was retained.

## Literature used

1. [CriteoPrivateAd](https://arxiv.org/abs/2502.12103) documents that categorical
   values are hashed, continuous values receive monotone transforms, and its
   baseline is logistic regression over the anonymized bidding features with a
   temporal split. The CAPT split remains stricter and unchanged: model fitting
   uses days 1--6 and the final check uses days 25--30.
2. [Ad Click Prediction: a View from the Trenches](https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/)
   motivates sparse logistic CTR learning with per-coordinate updates and
   reports no useful accuracy benefit from collision hashing. This supports
   removing the label-blind whole-row modulo hash.
3. [Practical Lessons from Predicting Clicks on Ads at Facebook](https://quinonero.net/Publications/predicting-clicks-facebook.pdf)
   was reviewed as the next candidate: supervised boosted-tree transforms plus
   a sparse linear classifier. It was not implemented because candidate 1 had
   already passed the improvement stop rule.

## Implementation change

Previous pipeline:

- concatenate `campaign_id`, `publisher_id`, `display_order`, and public
  context;
- BLAKE2b-hash the complete row and reduce modulo 64 without labels;
- fit an additive logistic `f_ref` on nominal token plus public context.

Current pipeline:

- select all scalar, inference-available Criteo feature families;
- exclude the declared protected `_2` and `_3`, outcome/delay columns,
  `features_not_available_*`, row/user IDs, and the list-valued context field;
- treat integer/string fields as nominal one-hot features (at most 256 retained
  levels per field), and median-impute plus standardize floating-point fields;
- learn an averaged sparse logistic CTR score on `D_model` only;
- freeze 64 quantile cut points from the `D_model` decision score;
- retain the existing final calibration model on nominal token plus public
  context.

`features_kv_bits_constrained_5` is declared but all-missing in `D_model`, so it
is recorded and automatically dropped rather than imputed from later splits.

## Held-out result

All values below use the same 2,082,610 rows from `D_test` days 25--30 and the
ordinary, unweighted binary log-loss used in the CAPT experiments.

| representation | K | D_test log-loss | ROC-AUC |
|---|---:|---:|---:|
| constant at D_test prevalence | -- | 0.6586481253 | 0.5000000 |
| previous four-column whole-row hash | 64 | 0.6474476390 | 0.5698350 |
| supervised logistic-score quantiles | 64 | **0.5009144170** | **0.8147066** |

The absolute log-loss reduction versus the old K=64 representation is
0.1465332220 (22.63% relative). The new representation uses all 64 tokens and
its mean D_test prediction is 0.36679 versus an observed click rate of 0.36942.
The fitted result is written to
`outputs/reference_benchmarks/ctr_logistic_quantile_k64_dtest.json`.

This benchmark establishes that useful CTR information survives a 64-category
non-private representation. It does not establish that the subsequent
epsilon-1 sanitizer preserves that information; CAPT/LDP experiments must be
rerun because the encoder, partition, decoder, distortion costs, and optimized
channels all change with the new token semantics.
