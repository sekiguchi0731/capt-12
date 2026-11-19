"""Four-state population control, with an analytic channel and a matching negative control."""

import csv
import math
from pathlib import Path

import numpy as np

from capt12.comparison.optimal_ldp.utility import solve_spr, staircase_lp


def h(p):
    return -p * math.log(p) - (1 - p) * math.log1p(-p)


def run():
    epsilon, label_noise = 1.0, 0.1
    rr_keep = 1 / (1 + math.exp(-epsilon))
    v_ldp = h(label_noise + (1 - 2 * label_noise) * (1 - rr_keep))
    rows = []
    for alignment in (rr_keep, 0.8, 0.9, 1.0):
        # A, U balanced, Z=(U,W), W=A is a perfect attribute proxy in this control.
        # For any epsilon-LDP on Z the channel averaged given U is epsilon-LDP on U.
        # RR(U) Blackwell-dominates every binary-input epsilon-LDP and is feasible on Z.
        joint = []
        for u in (0, 1):
            for a in (0, 1):
                mass = 0.5 * (alignment if u == a else 1 - alignment)
                eta_u = label_noise if u == 0 else 1 - label_noise
                joint.append([mass * (1 - eta_u), mass * eta_u])
        flip = 0.5 * (1 - (2 * rr_keep - 1) / (2 * alignment - 1))
        attr_keep = alignment * (1 - flip) + (1 - alignment) * flip
        beta = math.log(attr_keep / (1 - attr_keep))
        assert math.isclose(beta, epsilon, abs_tol=2e-15)
        law = np.array(joint)
        spr = solve_spr(law, epsilon)
        lp = staircase_lp(law, epsilon)
        assert math.isclose(v_ldp, spr.estimate, abs_tol=2e-14, rel_tol=0.0)
        assert math.isclose(v_ldp, lp, abs_tol=2e-13, rel_tol=0.0)
        d_capt = h(label_noise + (1 - 2 * label_noise) * flip)
        rows.append(
            dict(
                alignment=alignment,
                flip=flip,
                beta=beta,
                tv=1.0,
                j_beta=1.0,
                analytic_capt_type_loss=d_capt,
                analytic_v_z=v_ldp,
                gap=v_ldp - d_capt,
                spr_lower=spr.lower,
                spr_upper=spr.upper,
                staircase_lp=lp,
            )
        )
    with Path(__file__).with_name("binary-control.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"q={row['alignment']:.6f}, beta={row['beta']:.6f}, loss={row['analytic_capt_type_loss']:.9f}, "
            f"V={v_ldp:.9f}, gap={row['gap']:.9f}"
        )


if __name__ == "__main__":
    run()
