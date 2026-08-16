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
the explicit `fail` or `force_cover` policy. `merge_to_other` is rejected by
configuration, certificate creation, and verification until a frozen
coarsening map is applied identically at runtime and embedded in the bundle.

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
