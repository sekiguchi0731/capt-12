# DP histogram and confidence status

Certificate histograms default to `one-display-per-uuid-day`: after a
deterministic seeded selection, each user/privacy-epoch contributes one one-hot
vector to exactly one group/block cell. Under add/remove adjacency the vector
histogram has L1 sensitivity 1. Under replacement adjacency one user can move
between two cells, so L1 sensitivity is 2. `capped-c` multiplies these bounds
by `c`. `per-row` is debug-only and is never described as a user-level
guarantee.

Contribution bounding is not an i.i.d. theorem. Certified finite-sample runs
add the external `user_day_iid` assumption: after the one-display selection,
user-day epochs are conditionally i.i.d. within protected group and context.
Because the release has no stable physical-user identifier across days, this
does not establish physical-user independence. Day-wise lower-audit summaries
are emitted only as robustness diagnostics, not as clustered confidence
intervals.

The code simulates the **one-shot central-DP histogram received by a server
after secure aggregation**. It does not implement or claim cryptographic secure
aggregation.

## Experimental DP-aware box

Noisy counts are never treated as binomial counts and are never passed directly
to Clopper–Pearson. The current experimental construction allocates half of
`alpha_cert` to a simultaneous two-sided Laplace tail event, converts each
noisy cell into a true-count interval, and allocates the other half to a
simultaneous Hoeffding sampling interval. The optional TV model expands the
resulting box-simplex: the shifted population may lie within the configured TV
radius of any distribution in the sampling confidence set.

This composition is intentionally marked `experimental`: finite-sample
coverage under random contribution selection, unknown total count after
noise, and all data-dependent grouping operations has not yet received a
complete proof and broad coverage simulation. `certify` rejects
`dp_aware_box`; it can only be used in diagnostic/ablation output. Non-DP
`cp_box` is the default certified path.

Enabling DP-aware certification requires (1) a complete joint coverage proof,
(2) a simulation suite spanning sparse cells, epsilon, delta/noise mechanism,
group counts, and contribution caps, and (3) explicit accounting across every
group, cell, privacy comparison, and parameter sweep.
