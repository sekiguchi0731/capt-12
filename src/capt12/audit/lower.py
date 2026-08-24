from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class LowerAuditResult:
    epsilon_lower: float
    witness: dict | None
    events_tested: int
    bounds_tested: int = 0
    per_bound_alpha: float = math.nan
    candidate_events: int = 0
    candidate_groups: int = 0
    candidate_contexts: int = 0
    interpretation: str = "lower leakage witness; not a safety certificate"


def _one_sided_cp(successes: int, trials: int, alpha: float, lower: bool) -> float:
    if trials <= 0:
        return 0.0 if lower else 1.0
    if lower:
        return 0.0 if successes == 0 else float(beta.ppf(alpha, successes, trials - successes + 1))
    return 1.0 if successes == trials else float(beta.ppf(1 - alpha, successes + 1, trials - successes))


def lower_audit(
    attack_train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    group_col: str,
    output_col: str,
    context_cols: list[str] | None = None,
    alpha: float = 0.05,
) -> LowerAuditResult:
    context_cols = context_cols or []
    # Candidate groups, context strata, singleton events, model parameters, and
    # thresholds are learned only on D_attack_train.  Conditional on that split,
    # the complete test family is fixed before D_test is examined.
    if context_cols:
        attack_context = attack_train[context_cols].astype(str).agg("|".join, axis=1)
        test_context = test[context_cols].astype(str).agg("|".join, axis=1)
        context_values = sorted(attack_context.dropna().unique().tolist(), key=str)
    else:
        attack_context = pd.Series("all", index=attack_train.index)
        test_context = pd.Series("all", index=test.index)
        context_values = ["all"]
    groups_by_context = {
        context_value: sorted(
            attack_train.loc[attack_context == context_value, group_col]
            .dropna()
            .unique()
            .tolist(),
            key=str,
        )
        for context_value in context_values
    }
    candidate_group_count = sum(len(groups) for groups in groups_by_context.values())
    events: list[tuple[str, object]] = [
        ("singleton", value)
        for value in sorted(attack_train[output_col].dropna().unique().tolist())
    ]
    # Attacker sees (O,B), learns scores/threshold candidates only on D_attack_train.
    feature_cols = [output_col, *context_cols]
    if attack_train[group_col].nunique() == 2:
        model = make_pipeline(
            ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols)]),
            LogisticRegression(max_iter=200),
        )
        model.fit(attack_train[feature_cols].astype(str), attack_train[group_col].astype(str))
        train_scores = model.predict_proba(attack_train[feature_cols].astype(str))[:, 1]
        thresholds = np.unique(np.quantile(train_scores, [0.25, 0.5, 0.75]))
        test_scores = model.predict_proba(test[feature_cols].astype(str))[:, 1]
        for threshold in thresholds:
            events.append(("score_threshold", (float(threshold), test_scores)))
    tests = len(events) * sum(
        len(groups) * (len(groups) - 1) for groups in groups_by_context.values()
    )
    if tests == 0:
        return LowerAuditResult(
            0.0,
            None,
            0,
            candidate_events=len(events),
            candidate_groups=candidate_group_count,
            candidate_contexts=len(context_values),
        )
    # Every event/pair/stratum test uses a numerator LCB and denominator UCB.
    # Allocating alpha/(2T) to all 2T one-sided bounds gives a global union-bound
    # failure probability of at most alpha within this audit run.
    per_bound_alpha = alpha / (2 * tests)
    best = 0.0
    witness = None
    tested = 0
    singleton_only = all(event_type == "singleton" for event_type, _ in events)
    if singleton_only:
        # The Criteo audit normally has more than two protected groups, hence
        # singleton output events only.  Aggregate once so a large D_test is
        # not rescanned for every ordered pair and output value.  This computes
        # exactly the same successes/trials and CP bounds as the row-wise path.
        audit_table = pd.DataFrame(
            {
                "__context__": test_context.astype(str).to_numpy(),
                "__group__": test[group_col].to_numpy(),
                "__output__": test[output_col].to_numpy(),
            }
        )
        trial_counts = audit_table.groupby(
            ["__context__", "__group__"], dropna=False, sort=False
        ).size()
        event_counts = audit_table.groupby(
            ["__context__", "__group__", "__output__"], dropna=False, sort=False
        ).size()
    for context_value in context_values:
        stratum_mask = None if singleton_only else test_context == context_value
        groups = groups_by_context[context_value]
        for left, right in permutations(groups, 2):
            if singleton_only:
                left_trials = int(trial_counts.get((str(context_value), left), 0))
                right_trials = int(trial_counts.get((str(context_value), right), 0))
            else:
                assert stratum_mask is not None
                left_mask = stratum_mask & (test[group_col] == left)
                right_mask = stratum_mask & (test[group_col] == right)
                left_trials = int(left_mask.sum())
                right_trials = int(right_mask.sum())
            for event_type, value in events:
                if event_type == "singleton":
                    event_description = {"type": event_type, "output": str(value)}
                    if singleton_only:
                        left_success = int(
                            event_counts.get((str(context_value), left, value), 0)
                        )
                        right_success = int(
                            event_counts.get((str(context_value), right, value), 0)
                        )
                    else:
                        event = test[output_col].to_numpy() == value
                        left_success = int(np.sum(event & left_mask.to_numpy()))
                        right_success = int(np.sum(event & right_mask.to_numpy()))
                else:
                    threshold, scores = value
                    event = scores >= threshold
                    event_description = {"type": event_type, "threshold": threshold}
                    left_success = int(np.sum(event & left_mask.to_numpy()))
                    right_success = int(np.sum(event & right_mask.to_numpy()))
                lcb = _one_sided_cp(
                    left_success, left_trials, per_bound_alpha, True
                )
                ucb = _one_sided_cp(
                    right_success, right_trials, per_bound_alpha, False
                )
                epsilon = (
                    math.log(lcb / ucb)
                    if lcb > 0 and ucb > 0
                    else -math.inf
                )
                tested += 1
                if epsilon > best:
                    best = epsilon
                    witness = {
                        "left_group": str(left),
                        "right_group": str(right),
                        "left_group_values": (
                            list(left) if isinstance(left, tuple) else [left]
                        ),
                        "right_group_values": (
                            list(right) if isinstance(right, tuple) else [right]
                        ),
                        "context": context_value,
                        "event": event_description,
                        "left_lcb": lcb,
                        "right_ucb": ucb,
                        "familywise_alpha": alpha,
                        "per_bound_alpha": per_bound_alpha,
                    }
    return LowerAuditResult(
        best,
        witness,
        tested,
        bounds_tested=2 * tested,
        per_bound_alpha=per_bound_alpha,
        candidate_events=len(events),
        candidate_groups=candidate_group_count,
        candidate_contexts=len(context_values),
    )
