from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _feature_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or len(matrix) == 0:
        raise ValueError("attack features must be a nonempty row matrix")
    return matrix.astype(str)


def _attacker(column_count: int, regularization: float, seed: int) -> Pipeline:
    transformer = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), list(range(column_count)))],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", transformer),
            (
                "model",
                LogisticRegression(
                    C=regularization,
                    max_iter=500,
                    random_state=seed,
                ),
            ),
        ]
    )


def _aligned_probabilities(
    model: Pipeline, features: np.ndarray, classes: tuple[str, ...]
) -> np.ndarray:
    raw = model.predict_proba(features)
    learned = tuple(map(str, model.named_steps["model"].classes_))
    if set(learned) != set(classes):
        raise ValueError("an attack fold did not contain every sensitive class")
    return raw[:, [learned.index(value) for value in classes]]


@dataclass(frozen=True)
class CrossFittedAttackFamily:
    """Attacker family selected exclusively inside D_attack_train."""

    classes: tuple[str, ...]
    baseline_models: tuple[Pipeline, ...]
    output_models: tuple[Pipeline, ...]
    baseline_regularization: float
    output_regularization: float
    candidate_regularization: tuple[float, ...]
    fold_count: int
    source_split: str = "D_attack_train"
    model_class: str = "one-hot multinomial logistic regression"

    def __post_init__(self) -> None:
        if self.source_split != "D_attack_train":
            raise ValueError("attack family must be frozen on D_attack_train")
        if len(self.baseline_models) != self.fold_count or len(self.output_models) != self.fold_count:
            raise ValueError("attack ensemble does not match the frozen fold count")


@dataclass(frozen=True)
class AttackCMIProxy:
    """Held-out conditional attack log-loss improvement; not a certificate."""

    raw_nats: float
    zero_clipped_nats: float
    ci95_low_nats: float
    ci95_high_nats: float
    baseline_cross_entropy_nats: float
    output_cross_entropy_nats: float
    bootstrap_replicates: int
    metric_name: str = "cross-fitted attack-CMI proxy"
    certificate: bool = False


def _select_regularization(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: tuple[str, ...],
    candidates: tuple[float, ...],
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> float:
    scores: list[tuple[float, float]] = []
    for regularization in candidates:
        fold_losses = []
        for fold_index, (train, validation) in enumerate(folds):
            model = _attacker(features.shape[1], regularization, seed + fold_index)
            model.fit(features[train], labels[train])
            probabilities = _aligned_probabilities(model, features[validation], classes)
            fold_losses.append(
                float(log_loss(labels[validation], probabilities, labels=list(classes)))
            )
        scores.append((float(np.mean(fold_losses)), regularization))
    return min(scores, key=lambda item: (item[0], item[1]))[1]


def fit_cross_fitted_attack_family(
    baseline_features: np.ndarray,
    output_features: np.ndarray,
    sensitive_labels: np.ndarray,
    user_day_ids: np.ndarray,
    *,
    regularization_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
    fold_count: int = 5,
    seed: int = 0,
    split_name: str = "D_attack_train",
) -> CrossFittedAttackFamily:
    """Select and fit equal-budget baseline/output attackers on D_attack_train."""
    if split_name != "D_attack_train":
        raise ValueError("attack models and hyperparameters may use D_attack_train only")
    baseline = _feature_matrix(baseline_features)
    output = _feature_matrix(output_features)
    labels = np.asarray(sensitive_labels).astype(str)
    groups = np.asarray(user_day_ids).astype(str)
    if not (len(baseline) == len(output) == len(labels) == len(groups)):
        raise ValueError("attack-training arrays must have equal row counts")
    if len(np.unique(groups)) < fold_count:
        raise ValueError("not enough user-day clusters for requested cross fitting")
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    candidates = tuple(sorted(set(map(float, regularization_grid))))
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("regularization_grid values must be positive")
    classes = tuple(sorted(np.unique(labels).tolist()))
    if len(classes) < 2:
        raise ValueError("sensitive attack requires at least two classes")

    splitter = GroupKFold(n_splits=fold_count)
    folds = list(splitter.split(baseline, labels, groups))
    baseline_c = _select_regularization(
        baseline, labels, groups, classes, candidates, folds, seed
    )
    output_c = _select_regularization(output, labels, groups, classes, candidates, folds, seed)
    baseline_models: list[Pipeline] = []
    output_models: list[Pipeline] = []
    # Cross-fitted ensembles use the same group folds and training budget for
    # baseline and output attackers.  D_test is never seen here.
    for fold_index, (train, _validation) in enumerate(folds):
        baseline_model = _attacker(baseline.shape[1], baseline_c, seed + fold_index)
        output_model = _attacker(output.shape[1], output_c, seed + fold_index)
        baseline_model.fit(baseline[train], labels[train])
        output_model.fit(output[train], labels[train])
        baseline_models.append(baseline_model)
        output_models.append(output_model)
    return CrossFittedAttackFamily(
        classes=classes,
        baseline_models=tuple(baseline_models),
        output_models=tuple(output_models),
        baseline_regularization=baseline_c,
        output_regularization=output_c,
        candidate_regularization=candidates,
        fold_count=fold_count,
    )


def _ensemble_probabilities(
    models: tuple[Pipeline, ...], features: np.ndarray, classes: tuple[str, ...]
) -> np.ndarray:
    predictions = [_aligned_probabilities(model, features, classes) for model in models]
    return np.mean(predictions, axis=0)


def evaluate_attack_cmi_proxy(
    family: CrossFittedAttackFamily,
    baseline_features: np.ndarray,
    output_features: np.ndarray,
    sensitive_labels: np.ndarray,
    user_day_ids: np.ndarray,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 0,
    split_name: str = "D_test",
) -> AttackCMIProxy:
    """Evaluate the frozen attack family on D_test with cluster-bootstrap CI."""
    if split_name != "D_test":
        raise ValueError("the held-out attack-CMI proxy is evaluated on D_test only")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    baseline = _feature_matrix(baseline_features)
    output = _feature_matrix(output_features)
    labels = np.asarray(sensitive_labels).astype(str)
    groups = np.asarray(user_day_ids).astype(str)
    if not (len(baseline) == len(output) == len(labels) == len(groups)):
        raise ValueError("D_test attack arrays must have equal row counts")
    if not set(np.unique(labels)).issubset(family.classes):
        raise ValueError("D_test contains a sensitive class unseen in D_attack_train")

    baseline_probabilities = _ensemble_probabilities(
        family.baseline_models, baseline, family.classes
    )
    output_probabilities = _ensemble_probabilities(family.output_models, output, family.classes)
    class_index = {value: index for index, value in enumerate(family.classes)}
    indices = np.asarray([class_index[value] for value in labels], dtype=int)
    row = np.arange(len(labels))
    baseline_loss = -np.log(np.clip(baseline_probabilities[row, indices], 1e-15, 1.0))
    output_loss = -np.log(np.clip(output_probabilities[row, indices], 1e-15, 1.0))
    improvement = baseline_loss - output_loss

    unique_groups = np.unique(groups)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_replicates, dtype=float)
    for replicate in range(bootstrap_replicates):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_rows = np.concatenate([rows_by_group[group] for group in sampled])
        bootstrap[replicate] = float(improvement[sampled_rows].mean())
    raw = float(improvement.mean())
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return AttackCMIProxy(
        raw_nats=raw,
        zero_clipped_nats=max(raw, 0.0),
        ci95_low_nats=float(low),
        ci95_high_nats=float(high),
        baseline_cross_entropy_nats=float(baseline_loss.mean()),
        output_cross_entropy_nats=float(output_loss.mean()),
        bootstrap_replicates=bootstrap_replicates,
    )


def aggregate_attribute_proxies(
    values: dict[str, AttackCMIProxy], profile_mass: dict[str, float]
) -> dict[str, float]:
    if not values:
        raise ValueError("at least one protected-attribute proxy is required")
    missing = sorted(set(values) - set(profile_mass))
    if missing:
        raise ValueError(f"missing profile masses for attributes: {missing}")
    weights = np.asarray([profile_mass[name] for name in values], dtype=float)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("profile masses must be nonnegative with positive total mass")
    raw = np.asarray([values[name].raw_nats for name in values], dtype=float)
    clipped = np.asarray([values[name].zero_clipped_nats for name in values], dtype=float)
    weights /= weights.sum()
    return {
        "worst_attribute_raw_nats": float(raw.max()),
        "worst_attribute_zero_clipped_nats": float(clipped.max()),
        "profile_mass_weighted_mean_raw_nats": float(weights @ raw),
        "profile_mass_weighted_mean_zero_clipped_nats": float(weights @ clipped),
    }
