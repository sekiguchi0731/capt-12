"""Two-output CAPT on a frozen partition, certified on the released threshold event."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.special import xlogy

from capt12.comparison.optimal_ldp.retraining import FixedBlocks
from capt12.confidence.boxes import ConfidenceBox, cp_box, full_simplex_box
from capt12.privacy.adjacency import AdjacentPair


def design_label_counts(
    path: Path, context: str, assignment: np.ndarray, blocks: int
) -> np.ndarray:
    """Read the D_design display counts saved by the stratified mechanism run."""
    with np.load(path, allow_pickle=False) as frozen:
        levels = list(map(str, frozen["context_levels"]))
        n = frozen["token_context_label_count"]
        y = frozen["token_context_label_sum"]
    if (
        len(set(levels)) != len(levels)
        or n.shape != (len(assignment), len(levels))
        or y.shape != n.shape
    ):
        raise ValueError("inconsistent frozen token/context design tables")
    if (
        not np.isfinite(n).all()
        or not np.isfinite(y).all()
        or (y < 0).any()
        or (n < y).any()
        or (n != np.floor(n)).any()
        or (y != np.floor(y)).any()
    ):
        raise ValueError("invalid D_design label counts")
    design = np.zeros((blocks, 2), dtype=np.int64)
    if context in levels:
        index = levels.index(context)
        np.add.at(design[:, 0], assignment, (n[:, index] - y[:, index]).astype(np.int64))
        np.add.at(design[:, 1], assignment, y[:, index].astype(np.int64))
    return design


@dataclass
class BinaryCandidate:
    channel: FixedBlocks
    high_blocks: list[int]
    probabilities: tuple[float, float]
    boxes: dict[str, ConfidenceBox]


def _release_probabilities(boxes, edges):
    # Write the two probabilities as a/(a+c) and (a+1)/(a+c).
    # Each directed likelihood-ratio constraint for each output gives
    # a lower bound on a or c. Smaller values reduce randomization.
    lo = {g: max(box.lower[1], 1 - box.upper[0]) for g, box in boxes.items()}
    hi = {g: min(box.upper[1], 1 - box.lower[0]) for g, box in boxes.items()}
    a, c = 0.0, 1.0
    for edge in edges:
        # A small interior margin protects the serialized channel. For a
        # zero budget return a constant if the event probabilities can differ.
        eps = max(0.0, min(edge.epsilon, 100.0) - 1e-9)
        left, right = edge.left, edge.right
        if eps == 0:
            if hi[left] > lo[right] or hi[right] > lo[left]:
                return (0.5, 0.5)
            continue
        ratio = np.exp(eps)
        denom = np.expm1(eps)
        a = max(a, (hi[left] - ratio * lo[right]) / denom)
        c = max(c, (ratio * hi[right] - lo[left]) / denom)
    probs = 0.5 + ((a + np.array([0.0, 1.0])) / (a + c) - 0.5) * (1 - 1e-10)
    return float(probs[0]), float(probs[1])


def fit_binary_capt(
    assignment: np.ndarray,
    design: np.ndarray,
    cert_hist: dict,
    edges: list[AdjacentPair],
    alpha: float,
) -> BinaryCandidate:
    """Rank on D_design and certify all possible cuts simultaneously on D_cert."""
    blocks = len(design)
    n = design.sum(axis=1)
    order = np.argsort(
        np.divide(design[:, 1], n, out=np.full(blocks, 0.5), where=n > 0), kind="stable"
    )
    candidates = max(1, blocks - 1)
    best_loss = np.inf
    for cut in range(1, candidates + 1):
        high = np.zeros(blocks, dtype=bool)
        high[order[cut:]] = True
        boxes = {}
        for group, histogram in cert_hist.items():
            v = np.asarray(histogram)
            if v.shape != (blocks,) or not np.issubdtype(v.dtype, np.integer) or (v < 0).any():
                raise ValueError("invalid D_cert group histogram for source partition")
            counts = np.array([v[~high].sum(), v[high].sum()])
            boxes[group] = (
                cp_box(counts, alpha=alpha, group_count=len(cert_hist), comparisons=candidates)
                if counts.sum()
                else full_simplex_box(counts)
            )
        probs = _release_probabilities(boxes, edges)
        p = np.where(high, probs[1], probs[0])
        joint = np.array([(1 - p) @ design, p @ design])
        totals = joint.sum(axis=1, keepdims=True)
        posterior = np.divide(joint, totals, out=np.zeros_like(joint), where=totals > 0)
        loss = -xlogy(joint, posterior).sum()
        if loss < best_loss:
            best_loss = loss
            rows = tuple((1 - Fraction(q), Fraction(q)) for q in probs)
            best = BinaryCandidate(
                FixedBlocks(high[assignment].astype(int), rows),
                np.flatnonzero(high).tolist(),
                probs,
                boxes,
            )
    return best
