# Theory and implementation contract

## Scope

Let the frozen external encoder be `Phi_0: X -> [K]`, `K <= 4096`, and
`Z=Phi_0(X)`. For a protected profile `S`, `A_S` is the tuple of selected
sensitive-proxy values, `B` is advertiser-visible context, and `Y` is used only
for utility evaluation through a frozen reference predictor. The main theorem
treats `S` as known and protects `A_S`; it does not protect the fact that `S`
was selected. Optional profile-privacy experiments use synthetic assignments
and are reported separately. No Blackwell-order or outcome-privacy claim is
made.

## Block CAPT and the common-decoder lift

Let `C_1,...,C_L` be a non-empty partition of `[K]`, with assignment `c(z)`.
For each profile, `R_S` is a row-stochastic block channel. The within-block
decoder `nu_l(o)` is common to every profile, fitted from `D_design` only, and
frozen before `D_cert`. Online sampling is

1. `l=c(z)`;
2. sample `l' ~ R_S[l,:]`;
3. sample `o ~ nu_l'`.

Thus

`Q_S(o|z) = R_S[c(z),c(o)] nu_{c(o)}(o)`.

For group `g=(S,a,b)`, define `p_g(l)=P(c(Z)=l|g)`. If `l'=c(o)`, then

`P(O=o|g) = nu_l'(o) sum_l p_g(l) R_S[l,l']`.

**Proposition (block privacy implies token privacy).** Suppose for an ordered
adjacent pair `(g,g')` and every destination block `l'`,

`p_g^T R_S[:,l'] <= exp(epsilon) p_g'^T R_S[:,l']`.

Multiplying both sides by the nonnegative, group-independent quantity
`nu_l'(o)` proves the same inequality for each token `o in C_l'`. If the
decoder probability is zero, both token probabilities are zero and the
inequality remains true. Therefore the block constraints imply every
token-output likelihood-ratio constraint. Profile-dependent decoders would
invalidate this cancellation, which is why the default decoder is common.

## Tuple adjacency and hybrid composition

Within fixed `(S,B=b)`, tuples are adjacent when exactly one coordinate
`j in S` differs. The robust ordered constraint is

`sup_{p in C_g} p^T R_S[:,l'] <= exp(epsilon_j) inf_{p' in C_g'} p'^T R_S[:,l']`.

For tuples differing in coordinates `j_1,...,j_m`, connect them by a path that
changes one coordinate at a time. Multiplying the `m` likelihood-ratio
inequalities and cancelling the intermediate probabilities gives

`epsilon_tuple <= sum_t epsilon_{j_t}`.

The implementation checks every ordered edge. `marginal` is an ablation;
`joint_all_pairs` directly compares all tuple pairs. Rare groups are handled by
the explicit `fail`, `force_cover`, or finite-sample `confidence_box` policy.
`merge_to_other` is rejected by configuration, certificate creation, and
verification until a frozen coarsening map is applied identically at runtime
and embedded in the bundle.

## Full-simplex completion for unobserved groups

For a fixed expected support, observed group `g` receives its simultaneous
finite-sample CP/TV set. If `g` has zero certificate observations, the explicit
`missing_group_policy: full_simplex` mode sets

`C_g = Delta_L = {p >= 0: sum_l p(l)=1}`.

For every channel column `r`, its exact support values are

`sup_{p in Delta_L} p^T r = max_l r(l)` and
`inf_{p in Delta_L} p^T r = min_l r(l)`.

The same robust ordered-adjacency constraints therefore apply without an
estimated conditional distribution for the missing group. A zero-count group
is never assigned a point estimate. The certificate embeds zero histograms,
labels their boxes `full_simplex`, and independently reconstructs both the
mixed box family and all expected-support adjacency edges.

On the simultaneous coverage event for all observed-group boxes, robust
feasibility implies the requested group privacy for every expected group;
missing groups are covered distribution-free because every possible block
distribution lies in `Delta_L`. Every row-wise epsilon-LDP block channel is
feasible when all ordered edges use that epsilon: `max_l R(l,o) <=
exp(epsilon) min_l R(l,o)` is exactly the full-simplex robust inequality.
Consequently, for the same fixed channel class, decoder, cost, and objective,

`U_simplex-CAPT >= U_optimal-LDP >= U_common-cover`.

If every confidence set is replaced by a subset, the robust feasible region
can only expand and the optimal utility cannot decrease. This is a deterministic
set-inclusion statement. CP boxes recomputed from larger random samples need
not be nested realization by realization, so ordinary sample-size curves do
not automatically satisfy pointwise monotonicity without an explicitly nested
confidence construction.

A full-simplex/full-simplex ordered edge has a stronger structural
consequence. For every output `o`, its robust inequality is exactly

`max_l R(l,o) <= exp(epsilon) min_l R(l,o)`.

Thus one such edge imposes global row-wise epsilon-LDP on the channel in its
scope. With several such edges, the smallest full-simplex-edge epsilon applies.
The implementation replaces the robust LP by the pure LDP LP only when that
epsilon is no larger than every other edge budget. If a non-simplex edge has a
smaller budget, the LDP inequalities are used as seed constraints and all
remaining robust constraints are retained and separated by the support oracle.
Every returned channel is independently rechecked against every original edge.

## Public-context stratification and sensitive fallback coarsening

When `B` is explicitly public and there is no privacy adjacency between
different values of `B`, CAPT may use a separate block channel `R_b` for each
public context. Online selection uses only `(Z, profile, B)`; the realized
protected value is not an input. The guarantee is conditional:

`P(O=o | A~=a, B=b) <= exp(epsilon) P(O=o | A~=a', B=b)`.

It protects the additional disclosure in `O` to an attacker who already knows
`B`. It does not protect attribute leakage through `B` itself.

When repeated-release protection is enabled, the online sanitizer memoizes by
`(user privacy epoch, profile, public context)`, or by `(user privacy epoch,
profile)` for a shared channel. The raw token `Z` is not part of the cache key:
the first sanitized output is reused if `Z` changes later in the same epoch.
Including `Z` in the key would permit multiple randomized releases and require
an explicit composition analysis.

The frozen sensitive mapper coarsens both a missing value and any value outside
its pre-certificate known domain to one secret value `__UNKNOWN__`. Therefore
the certified secret is `A~=c(A)`, not the original uncoarsened `A`. Comparisons
between distinct original values that both map to `__UNKNOWN__` are outside the
declared secret domain. The runtime mapper artifact, exact online selector, and
the complete context-to-channel table are hash-bound to every context
certificate.

For fixed partition, decoder, support, and objectives, absence of cross-context
edges gives the product decomposition

`F = product_b F_b` and `D({R_b}) = sum_b omega_b D_b(R_b)`.

The shared-channel class is the restriction `R_b=R` for every `b`, hence the
optimal context-stratified distortion cannot exceed the optimal shared-channel
distortion. A full-simplex/full-simplex edge in context `b` makes only `R_b`
row-wise LDP; it does not force dense contexts to pay the same restriction.
This is context-local graceful degradation.

The decomposition and certificate arguments require only a fixed linear cost;
they do not require Bernoulli KL specifically. The implementation therefore
supports teacher KL, D_design empirical randomized log loss, and a predeclared
linear blend. With a fixed decoder and output-token CTR probabilities, the
empirical cost is the D_design average of
`-y log(q_o) - (1-y) log(1-q_o)` for each input/output token pair, so the
mechanism optimization remains an LP. D_model fits the teacher, D_design fixes
the cost, D_cert constructs the privacy confidence sets, and D_test is reserved
for final evaluation.

Because the partition and decoder are common across public contexts, an
objective-aligned representation uses

`C_repr(z,o) = sum_b P_Ddesign(b | z) C_b(z,o)`.

Partition and common decoder are constructed from `C_repr`, while each
context-indexed channel `R_b` is optimized with its own `C_b`. Thus all three
stages target the same utility estimand without allowing `B`-specific decoders
or using the protected value online. The legacy `teacher_kl_fixed` mode is a
strictly defined ablation in which only R's objective changes.

For empirical log loss, `C_b(z,o)` decomposes into the empirical binary entropy
of the input cell plus `KL(Ber(q_hat_zb) || Ber(p_ob))`. The entropy term is
independent of the channel. It may be subtracted when normalizing reported
relative improvements, but is retained in the LP cost; absolute optimizer and
certificate results are unchanged.

One certificate is emitted for each public context and design. To obtain a
simultaneous confidence statement across all `B` context tables within a
design, each certificate uses `alpha_cert / |B|` (Bonferroni). The design-level
statement is conditional on all these context confidence events holding.

### Certificate-safe post-solve repair

An LP solver enforces inequalities to an additive feasibility tolerance. A
returned channel can consequently contain a tiny positive entry in one row of
an otherwise zero output column. A robust denominator can then be exactly zero
while its numerator is positive, so the pure-epsilon ratio is infinite even
though the additive violation is below the solver tolerance. Such a channel is
not released unchanged.

For each context, the implementation deterministically mixes the LP output
with the input-independent uniform channel `U(l,o)=1/L`:

`R_lambda = (1-lambda) R + lambda U`.

For any group distribution `p`, `p^T U[:,o]=1/L`. Thus a robust additive
violation `v=n-exp(epsilon)d` becomes

`v(lambda) = (1-lambda)v - lambda(exp(epsilon)-1)/L`.

For positive epsilon, the common component supplies strict slack. The code
takes the maximum closed-form minimum `lambda` required by every constraint in
that context, adds an explicit small additive safety margin, and stores
`R_lambda`—not the unmodified solver output—as the released and certified
channel. It then recomputes all constraints from scratch. A separate Decimal
check evaluates the serialized binary64 channel with no privacy feasibility
tolerance and no denominator floor; any non-finite realized epsilon or
positive additive violation rejects the certificate. Because the objective is
linear, the distortion change is also linear in `lambda`, avoiding a larger
arbitrary utility penalty.

For each directed robust constraint, the reported realized privacy loss is
`log(max(1, numerator/denominator))`. A ratio below one and a jointly zero
output column therefore contribute zero. A positive numerator over a zero
denominator remains infinite; neither the floating-point nor Decimal verifier
uses a denominator floor.

## Utility informativeness gate

For a fixed partition, common decoder, block cost `C`, and normalized input
weights `w`, define

`D_const = min_o sum_l w_l C(l,o)`,

`D_free = sum_l w_l min_o C(l,o)`, and

`G_info = D_const - D_free`.

`G_info` is the largest distortion reduction that input dependence could offer
before privacy constraints. The experiment gate rejects a channel class when
`G_info` is at numerical zero, every input row has the same minimizing output,
or the deterministic no-privacy row-wise minimizer has zero maximum row TV.
This is an experiment-eligibility diagnostic, not a privacy theorem and not a
certificate failure.

The `cost_medoid` decoder chooses, inside each fixed output block, the token
minimizing weighted expected token distortion. The deterministic weighted
cost k-medoids design jointly changes the partition and uses cost medoids. The
`L=K` singleton partition with identity decoder is a positive control that
represents every token channel; it does not remove any LDP restriction caused
by full-simplex edges.

## Sampling assumption

Finite-sample CP/Hoeffding certificates and the held-out lower audit are
conditional on the explicit `sampling_assumption: user_day_iid`: after
one-display-per-uuid-day contribution sampling, user-day privacy epochs are
assumed conditionally i.i.d. within each protected group and visible-context
stratum. Contribution bounding prevents one epoch from contributing multiple
displays; it does not prove independence, identical distributions, or
physical-user independence across days. The guarantee is therefore at the
user-day privacy-epoch population level. TV expansion models population shift
inside its declared radius but does not repair sampling dependence.

## Utility cost

The frozen reference model is clipped to `[eta,1-eta]`, with `eta=10^-6`, and
uses `(Z,B)` inputs without direct protected attributes by default. Because
Criteo contains no observed profile-selection variable, design data estimate a
profile-common utility table

`c(z,o)=E[d(z,o,B)|Z=z]`

and

`C(l,l')=E[sum_{o in C_l'} nu_l'(o)d(Z,o,B)|c(Z)=l]`.

Each protected profile still has its own feasible set and optimized channel
`R_S`, but the objective coefficients are frozen globally on `D_design`. For
each profile the block objective is

`sum_{l,l'} pi(l) R_S[l,l'] C(l,l')`.

Empirical and uniform input-token weighting are configuration options.
`profile_weighting=global_design` records this identifiable objective scope;
profile-specific `pi_S`, `c_S`, or `omega_S` are not claimed without observed
profile assignments. The default decoder adds a small uniform floor to global
`D_design` token frequencies and normalizes within each block.

The reported Criteo utility is a frozen reference-score surrogate. No
downstream CTR model is retrained on sanitized `(O,B)`, so deployed CTR,
auction, or bidding-utility preservation is not claimed.

## Full-channel oracle and singleton equivalence

The full oracle optimizes a row-stochastic `Q_S(z,o)` with the identical
adjacency, confidence sets, epsilon and weighted token cost. If `L=K`, every
block is a singleton, `c(z)=z`, and `nu_l=delta_l`; therefore

`Q_S(o|z)=R_S[z,o]`.

The feasible sets, objective coefficients, and constraints are identical, so

`OPT_CAPT(L=K) = OPT_full`.

Results label a row as a comparison-eligible full optimum only when the solver
is optimal, the channel is feasible, independent robust verification is valid,
and all problem-defining inputs match the CAPT comparison. Merely emitting a
`capt_full` row or a common-cover fallback does not establish `U_full`.

This is tested numerically. It is computationally useful only for small `K`:
`K=4096` gives 16,777,216 variables per profile, whereas `L=64` and `L=128`
give 4,096 and 16,384. The full solver defaults to `K<=256` and reports its
size estimate instead of constructing a prohibited dense large matrix.

## Theorem-4 envelope is a relaxation

For exact-token retention `r_z=Q_S(z|z)`, each reference group gives the
necessary constraint

`sum_z m_g0(z) r_z <= 1`,

where `m_g0(z)=max(p_g0(z), max_{g~g0} exp(-epsilon(g,g0))p_g(z))`. Maximizing
weighted retention over all such constraints and `0<=r<=1` is called
`theorem4_envelope`. It is a `K`-variable relaxation and is not a channel.

Consequently,

`U_CAPT(L) <= U_full <= U_envelope`.

Singleton CAPT reaches `U_full`, not necessarily `U_envelope`. For
`K=3`, `epsilon=0`, `p0=(0.3,0.6,0.1)` and `p1=(0.1,0.8,0.1)`, true privacy
forces rows 0 and 1 of `Q` to coincide, yielding uniform-weight retention
`U_full=2/3`. The envelope allows `r=(1,0.75,1)` and yields `11/12`. This strict
gap is a regression test. The envelope applies only to exact-token retention,
not block retention or CTR distortion; general distortions use the full LP
optimum/solver dual bound.

Plot captions distinguish the CAPT-to-full algorithmic/block approximation gap
from the full-to-envelope converse-relaxation looseness. Non-nested partitions
need not be utility-monotone in `L`.

For gap normalization, `U_decoder_cover(L)` is the best input-independent
channel representable by the same fixed partition and common decoder as the
CAPT row. If decoder row `D_l` is the output distribution for block `l`, then
for exact-token retention

`U_decoder_cover(L) = max_l sum_z pi(z) D_l(z)`.

This channel is feasible in the fixed CAPT class by sending every source block
to the maximizing destination block. Thus `U_decoder_cover <= U_CAPT`, and the
reported CAPT-to-full/envelope attainment fractions use this within-class
baseline. The separate token-level empirical common-cover mechanism is not
used as their denominator.

## Risk-utility partition

On `D_design` only, token `z` receives

`h(z) = normalize(f_ref score(z)) + lambda_risk normalize(max_g p_g(z)-min_g p_g(z))`.

Tokens are stably ordered by `h` and divided into balanced contiguous blocks.
`lambda_risk` is configurable (default 1). This is a heuristic partition
builder, not a new privacy theorem; certification still uses independent
`D_cert` confidence sets.

## Feasibility

The common-cover block channel has identical rows. Its output distribution is
independent of the input and therefore satisfies every finite-epsilon
constraint, including epsilon zero. A solver failure is reported as a numerical
failure, not mathematical infeasibility.
