# CriteoPrivateAd local data inspection

Inspection date: 2026-08-15 (Asia/Tokyo). Source directory:
`data/CriteoPrivateAd_release/data`. Values and identifiers were not copied into
this document. The inspection used Parquet footer metadata plus the first 2,048
rows of each local shard (61,440 sampled rows); it did not materialize the full
dataset.

## Physical layout

- 30 Hive-style partitions, `day_int=1` through `day_int=30`.
- One local Parquet shard and one row group per day; all 30 schemas are equal.
- 9,634,806 rows total. Daily shard counts range from 168,930 to 398,630.
- 150 physical columns. `day_int` is a partition value in the path, not a
  physical Parquet column.
- Local files comprise a selected subset of the release, not the complete
  34GB release. The manifest selection is one shard per day.

One row has display-level structure: it has a unique-looking `id`, a `user_id`,
`display_order`, auction identifiers/features, and delayed outcome arrays. The
sample contained repeated `user_id` values within a day. The code therefore
does not assume independent users per row and defaults certificate histograms
to one display per `user_id` and day. No claim is made that `user_id` is a
persistent cross-day identity.

## Named identifiers and outcomes

The cardinality column below is sample cardinality, not an exact population
cardinality. Missing counts come from footer statistics and are exact for the
30 inspected local shards.

| column | Arrow type | missing / 9,634,806 | sample cardinality | interpretation used by code |
|---|---:|---:|---:|---|
| `id` | string | 0 | 61,440 | display/row identifier |
| `user_id` | string | 0 | 61,420 | privacy-epoch contribution key; not treated as cross-day permanent ID |
| `display_order` | int32 | 0 | 65 | auction metadata |
| `campaign_id` | int64 | 0 | 4,933 | auction metadata |
| `publisher_id` | int32 | 347 | 8,186 | auction metadata/context candidate |
| `is_clicked` | int32 | 0 | 2 | default CTR utility label |
| `is_click_landed` | double | 0 | 2 | optional landed-click label |
| `is_visit` | int32 | 0 | 2 | optional visit label |
| `nb_sales` | int64 | 9,506,593 | 12 | optional sales count/outcome |
| `sale_delay_after_display_array` | list<int32> | 9,562,661 | list, not computed | delayed sales outcome |
| `click_delay_after_display_array` | list<int32> | 6,110,508 | list, not computed | delayed click outcome |
| `landed_click_delay_after_display_array` | list<int32> | 6,855,258 | list, not computed | delayed landed-click outcome |

## Anonymized feature families

| prefix | count | observed Arrow types | missingness/cardinality notes |
|---|---:|---|---|
| `features_kv_bits_constrained_` | 31 | double, int64 | heterogeneous; one all-null column; sampled cardinalities 1 to 46,336 |
| `features_kv_not_constrained_` | 8 | double, int64 | mostly complete; sampled cardinalities 2 to 52,038 |
| `features_browser_bits_constrained_` | 11 | double | heterogeneous missingness; sampled cardinalities 369 to 15,735 |
| `features_ctx_not_constrained_` | 8 | double, int64, list<int64> | context candidates; cardinalities 6 to 33,867 for scalar columns |
| `features_not_available_` | 80 | double, int64, list<int64> | intentionally unavailable feature family; several all-null/nearly-null columns |

The low-cardinality defaults in `configs/criteo_main.yaml` are
`features_kv_bits_constrained_2` (sample cardinality 9) and
`features_kv_bits_constrained_3` (sample cardinality 14). They are called
*candidate protected attributes* or *sensitive proxy attributes* only; their
real-world semantics are unknown. The default context is
`features_ctx_not_constrained_0` (sample cardinality 29).

There is no physical `modelingSignals` or other precomputed 12-bit token column
in the inspected schema. Experiments must therefore fit/freeze an encoder on
`D_model`, or receive an explicitly configured precomputed token column.

`capt12 inspect-data` reproduces the footer/sample inspection and can emit a
machine-readable JSON report under an ignored output directory.

