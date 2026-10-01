from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from capt12.experiments.context_fixed_global_ldp import (
    _plot,
    _sample_global_outputs,
    _scores_for_outputs,
)


def test_sample_global_outputs_uses_one_shared_channel() -> None:
    channel = np.asarray([[1.0, 0.0], [0.25, 0.75]])
    outputs = _sample_global_outputs(
        np.asarray([0, 1, 1, 0]),
        channel,
        np.asarray([0.9, 0.1, 0.9, 0.2]),
    )
    assert outputs.tolist() == [0, 0, 1, 0]


def test_scores_for_outputs_uses_public_context_only_after_release() -> None:
    scores = _scores_for_outputs(
        np.asarray([0, 1, 1, 0]),
        np.asarray(["a", "a", "b", "b"]),
        np.asarray(["a", "b"]),
        np.asarray([[0.1, 0.8], [0.2, 0.9]]),
    )
    assert np.allclose(scores, [0.1, 0.8, 0.9, 0.2])


def test_plot_writes_absolute_log_loss_and_auc_outputs(tmp_path: Path) -> None:
    rows = []
    for seed in range(2):
        rows.extend(
            [
                {
                    "test_seed": seed,
                    "method": "context_capt",
                    "sampled_log_loss": 0.62 + seed * 1e-4,
                    "ROC_AUC": 0.65 - seed * 1e-4,
                },
                {
                    "test_seed": seed,
                    "method": "global_ldp",
                    "sampled_log_loss": 0.63 + seed * 1e-4,
                    "ROC_AUC": 0.64 - seed * 1e-4,
                },
            ]
        )
    _plot(pd.DataFrame(rows), tmp_path)
    assert (tmp_path / "figures" / "fixed_capt_vs_global_ldp_absolute.png").is_file()
    assert (tmp_path / "figures" / "fixed_capt_vs_global_ldp_absolute.pdf").is_file()
