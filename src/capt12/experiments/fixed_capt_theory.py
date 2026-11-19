"""Read-only diagnostic of fixed contextual CAPT on validation counts."""

from __future__ import annotations

import csv
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from capt12.comparison.optimal_ldp.calibration import calibrate
from capt12.comparison.optimal_ldp.theory import (
    Bounds,
    empirical_decision,
    empirical_ldp_oracle,
    find_crossing,
    information_decomposition,
    sdpi_retention_check,
    weighted_bounds,
)
from capt12.experiments.fixed_capt_ctr_comparison import (
    ComparisonSettings,
    _check_context_groups,
    _encode_tokens,
    _fixed_input_columns,
    _json_finite,
    _load_selected_displays,
    _verified_fixed_channel,
)
from capt12.utils.artifacts import sha256_file

METHODS = ("equal_epsilon", "tv_calibrated", "uniform_calibrated")
EMPIRICAL_PREFIX = "empirical_plugin_"
CONTEXT_INDEX = re.compile(r"context-(\d+)\.json$")


def _index(path: Path) -> int:
    match = CONTEXT_INDEX.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse context index from {path.name}")
    return int(match[1])


@dataclass(frozen=True)
class TheorySearch:
    initial: float = 2.0
    cap: float = 16.0
    tolerance: float = 0.001

    def crossing(self, oracle, capt, certificate, *, informative, preserves):
        crossed = find_crossing(
            oracle,
            capt,
            certificate,
            released_informative=informative,
            preserves_raw_information=preserves,
            initial=self.initial,
            cap=self.cap,
            tolerance=self.tolerance,
        )
        return {
            "epsilon_cross_lower": crossed.lower,
            "epsilon_cross_upper": crossed.upper,
            "epsilon_cross_status": crossed.status,
            "empirical_win_interval_empty": crossed.win_interval_empty,
        }


def _epsilon_one(capt: Bounds, ldp: Bounds, epsilon_cert: float) -> dict:
    gap = ldp.minus(capt)
    eligible = epsilon_cert <= 1
    decision = empirical_decision(capt, ldp, privacy_eligible=eligible)
    return {
        "epsilon_1_privacy_eligible": eligible,
        "epsilon_1_v_z_lower": ldp.lower,
        "epsilon_1_v_z_upper": ldp.upper,
        "epsilon_1_gap_lower": gap.lower,
        "epsilon_1_gap_upper": gap.upper,
        "epsilon_1_empirical_decision": decision,
        "epsilon_1_decision": {
            "empirical_plugin_win": "certified_win",
            "empirical_plugin_loss": "certified_loss",
            "empirical_plugin_inconclusive": "inconclusive",
        }[decision],
    }


def _population_status(saved: dict | None, epsilon_cert: float) -> dict:
    if saved is None or not saved["settings"]["iid_utility_target"]:
        return {"population_decision": "assumptions_not_enabled"}
    equal = saved["comparisons"]["equal_epsilon"]
    if equal["lambda"] != 1 or equal["population_v_z_bounds"] is None:
        return {"population_decision": "population_inconclusive"}
    capt = Bounds(saved["capt_population_bayes_lcb"], saved["capt_test_loss_ucb"])
    ldp = Bounds(*equal["population_v_z_bounds"])
    state = empirical_decision(capt, ldp, privacy_eligible=epsilon_cert <= 1)
    return {
        "population_decision": {
            "empirical_plugin_win": "population_certified_win",
            "empirical_plugin_loss": "population_certified_loss",
            "empirical_plugin_inconclusive": "population_inconclusive",
        }[state],
        "saved_population_capt_bounds": [capt.lower, capt.upper],
        "saved_population_v_z_bounds": [ldp.lower, ldp.upper],
        "saved_simultaneous_error_upper": saved["simultaneous_error_upper"],
    }


def _read_result_files(results_dir: Path) -> dict:
    files = sorted(results_dir.glob("context-*.json"))
    if not files:
        raise ValueError(f"no context result JSONs in {results_dir}")
    saved = {}
    for path in files:
        index = _index(path)
        if index in saved:
            raise ValueError(f"duplicate result for context index {index}")
        saved[index] = json.loads(path.read_text(encoding="utf-8"))
    first = next(iter(saved.values()))
    for doc in saved.values():
        if any(
            doc[key] != first[key]
            for key in (
                "settings",
                "splits",
                "profile",
                "context_column",
                "K",
                "L",
                "data_root",
                "certificate_source_git_sha",
            )
        ):
            raise ValueError(
                "context results must share one fitted experiment and split definition"
            )
        if Path(doc["certificate"]).parent != Path(first["certificate"]).parent:
            raise ValueError("context results refer to different original certificate directories")
    return saved


def _collect_selected_counts(bundle, config, settings, splits, data_root, cells, *, batch_size):
    models = bundle / "models"
    mapper = joblib.load(models / "category_mapper.joblib")
    encoder = joblib.load(models / "encoder.joblib")
    id_col, user_col = config.get("id_col", "id"), config.get("user_col", "user_id")
    label = config.get("label_col", "is_clicked")
    context_col = config["context_cols"][0]
    profile = config["profiles"][0]
    attributes = profile.split("+")
    k = int(encoder.k)
    seen = pd.Index([], dtype=str)
    counts = {key: {"train": 0, "validation": 0, "test": 0} for key in cells}
    joint = {key: np.zeros((k, 2), dtype=np.int64) for key in cells}
    count_columns = list(
        dict.fromkeys([id_col, user_col, *mapper.mappings, *attributes, context_col])
    )
    validate_columns = [*_fixed_input_columns(config), label]
    for kind, days in [
        ("train", settings.predictor_train_days),
        ("validation", settings.predictor_validation_days),
        ("test", splits["D_test"]),
    ]:
        for day in days:
            frame, seen = _load_selected_displays(
                data_root,
                [day],
                validate_columns if kind == "validation" else count_columns,
                mapper,
                id_col=id_col,
                user_col=user_col,
                seen_ids=seen,
            )
            for value, group in frame.groupby(context_col, sort=False):
                context = str(value)
                if context not in cells:
                    raise ValueError(f"observed uncatalogued public context {context}")
                _check_context_groups(group, attributes, profile, context, cells[context]["groups"])
                counts[context][kind] += len(group)
                if kind != "validation":
                    continue
                labels = group[label].to_numpy()
                if not np.isin(labels, [0, 1]).all():
                    raise ValueError("utility requires binary click labels")
                for start in range(0, len(group), batch_size):
                    stop = start + batch_size
                    tokens = _encode_tokens(encoder, group.iloc[start:stop], k)
                    indices = 2 * tokens + labels[start:stop].astype(int)
                    joint[context] += np.bincount(indices, minlength=2 * k).reshape(k, 2)
            del frame
    return counts, joint


def _load_bundle(saved, bundle, settings):
    first = next(iter(saved.values()))
    first_path = bundle / Path(first["certificate"]).name
    first_cert, capt, boxes, edges, beta = _verified_fixed_channel(first_path)
    config = first_cert.resolved_config
    design = config["context_design"]
    manifest = json.loads((bundle / "mechanism/context_channel_manifest.json").read_text())
    declared = set(manifest["designs"][design]["contexts"])
    paths = sorted(bundle.glob(f"certificate-{design}-context-*.json"), key=_index)
    settings.validate(first_cert.split_identifiers)
    bindings = ("encoder", "mapper", "model", "context_channel_manifest", "partition", "decoder")
    same_input_keys = (
        "dataset",
        "profiles",
        "context_cols",
        "phi_source_cols",
        "label_col",
        "id_col",
        "user_col",
        "data_root",
        "context_design",
    )
    cells = {}
    used_indices = set()
    for path in paths:
        certificate, channel, available, adjacency, eps_cert = (
            (first_cert, capt, boxes, edges, beta)
            if path == first_path
            else _verified_fixed_channel(path)
        )
        context = str(certificate.resolved_config["public_context_value"])
        idx = _index(path)
        if context in cells or idx in used_indices:
            raise ValueError("duplicate certificate for context or index")
        if (len(channel.assignment), len(channel.rows)) != (first["K"], first["L"]):
            raise ValueError(f"unexpected alphabet or block count in {path.name}")
        if (
            certificate.split_identifiers != first_cert.split_identifiers
            or any(
                certificate.component_hashes[key] != first_cert.component_hashes[key]
                for key in bindings
            )
            or any(
                certificate.resolved_config.get(key) != config.get(key) for key in same_input_keys
            )
        ):
            raise ValueError(f"mismatched shared inputs in {path.name}")
        old = saved.get(idx)
        if old is not None:
            existing_splits = {key: old["splits"][key] for key in first_cert.split_identifiers}
            if (
                Path(old["certificate"]).name != path.name
                or old["context"] != context
                or old["profile"] != config["profiles"][0]
                or old["context_column"] != config["context_cols"][0]
                or old["certificate_source_git_sha"] != certificate.code_git_sha
                or sha256_file(path) != old["certificate_sha256"]
                or existing_splits != first_cert.split_identifiers
                or old["splits"]["D_predict_validation"] != list(settings.predictor_validation_days)
                or old["splits"]["D_predict_train"] != list(settings.predictor_train_days)
                or not math.isclose(
                    old["calibration"]["beta_R"], eps_cert, abs_tol=5e-12, rel_tol=0
                )
            ):
                raise ValueError(f"saved result does not bind verified certificate {path.name}")
            budgets = {name: float(old["comparisons"][name]["lambda"]) for name in METHODS}
        else:
            # Without a saved calibrated result, reconstruct the same fallback
            # only for a run that did not use raw-token shift information.
            budgets = dict.fromkeys(METHODS)
            budgets["equal_epsilon"] = float(certificate.resolved_config["epsilon"])
            if settings.raw_shift_tv is None:
                calibrated = calibrate(
                    eps_cert, None, [(e.left, e.right) for e in adjacency], alphabet_size=first["K"]
                )
                budgets.update(
                    tv_calibrated=calibrated.tv_budget, uniform_calibrated=calibrated.uniform_budget
                )
        eps_design = float(certificate.resolved_config["epsilon"])
        if budgets["equal_epsilon"] != eps_design:
            raise ValueError(f"saved equal-epsilon budget differs from design in {path.name}")
        cells[context] = {
            "index": idx,
            "certificate": path,
            "channel": channel,
            "groups": set(available),
            "epsilon_cert": eps_cert,
            "epsilon_design": eps_design,
            "epsilon_certificate_target": certificate.target_epsilon,
            "budgets": budgets,
            "saved": old,
        }
        used_indices.add(idx)
    if set(cells) != declared or not set(saved) <= used_indices:
        raise ValueError("certificate files do not cover the manifest and saved results")
    return first_cert, cells


def _flat(value, prefix="") -> dict:
    cells = {}
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(item, dict):
            cells.update(_flat(item, name))
        elif isinstance(item, list):
            cells[name] = json.dumps(item, ensure_ascii=False)
        else:
            cells[name] = item
    return cells


def _aggregate(included, search, epsilon_cert):
    records = [row for row, _, _ in included]
    n = sum(row["validation_n"] for row in records)
    weights = [Fraction(row["validation_n"], n) for row in records]
    infos = [info for _, info, _ in included]
    cap = weighted_bounds([info.entropy_release for info in infos], weights)

    @cache
    def oracle(epsilon):
        return weighted_bounds([value(epsilon) for _, _, value in included], weights)

    scalar_keys = (
        "h_y",
        "h_y_given_z",
        "h_y_given_c",
        "h_y_given_release",
        "i_y_z",
        "i_y_c",
        "i_y_release",
        "i_y_z_given_c",
        "i_y_c_given_release",
        "decomposition_residual",
    )
    metrics = {
        key: math.fsum(float(w) * r[key] for r, w in zip(records, weights, strict=True))
        for key in scalar_keys
    }
    return {
        **metrics,
        "validation_n": n,
        "epsilon_cert": epsilon_cert,
        "evaluated_epsilon_cert_max": max(r["epsilon_cert"] for r in records),
        "d_capt": cap.midpoint,
        "d_capt_lower": cap.lower,
        "d_capt_upper": cap.upper,
        "information_retention": metrics["i_y_release"] / metrics["i_y_z"]
        if metrics["i_y_z"]
        else None,
        **_epsilon_one(cap, oracle(1.0), epsilon_cert),
        **search.crossing(
            oracle,
            cap,
            epsilon_cert,
            informative=any(i.released_informative for i in infos),
            preserves=all(i.preserves_raw_information for i in infos),
        ),
        "population_decision": (
            "assumptions_not_enabled"
            if all(r["population_decision"] == "assumptions_not_enabled" for r in records)
            else "population_inconclusive"
        ),
    }


def run_fixed_capt_theory(
    results_dir: Path,
    output: Path,
    *,
    certificate_dir: Path | None = None,
    data_root: Path | None = None,
    search: TheorySearch = TheorySearch(),
    grid_subdivisions: int = 1,
    batch_size: int = 16384,
) -> dict:
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(f"use a new output directory: {output}")
    if batch_size <= 0 or not 1 <= grid_subdivisions <= 8:
        raise ValueError("positive batch size and 1 to 8 grid subdivisions required")
    saved = _read_result_files(results_dir)
    first = next(iter(saved.values()))
    settings = ComparisonSettings(**first["settings"])
    grid = sorted(
        set(settings.ctr_grid)
        | {
            a + (b - a) * i / grid_subdivisions
            for a, b in zip(settings.ctr_grid[:-1], settings.ctr_grid[1:], strict=True)
            for i in range(1, grid_subdivisions)
        }
    )
    bundle = certificate_dir or Path(first["certificate"]).parent
    first_cert, cells = _load_bundle(saved, bundle, settings)
    config, splits = first_cert.resolved_config, first_cert.split_identifiers
    data_root = data_root or Path(config["data_root"])
    counts, joint = _collect_selected_counts(
        bundle, config, settings, splits, data_root, cells, batch_size=batch_size
    )
    included, records = [], []
    for context, cell in cells.items():
        sizes = counts[context]
        old = cell["saved"]
        if old is not None and sizes != old["utility_sample_sizes"]:
            raise ValueError(f"sample counts differ from saved result at context {context}")
        row = {
            "context_index": cell["index"],
            "public_context_value": context,
            "certificate": str(cell["certificate"].resolve()),
            **{f"{kind}_n": n for kind, n in sizes.items()},
            "epsilon_design": cell["epsilon_design"],
            "epsilon_cert": cell["epsilon_cert"],
            "epsilon_certificate_target": cell["epsilon_certificate_target"],
            "ldp_budgets": cell["budgets"],
            "has_saved_comparison": old is not None,
        }
        records.append(row)
        if sizes["validation"] == 0:
            row.update(status="skipped", skip_reason="no_validation_examples")
            continue
        info = information_decomposition(joint[context], cell["channel"])
        if int(joint[context].sum()) != sizes["validation"]:
            raise ValueError(f"joint histogram count mismatch for {context}")
        if old is not None and any(
            not math.isclose(actual, expected, abs_tol=5e-13, rel_tol=0)
            for actual, expected in zip(
                (info.entropy_release.lower, info.entropy_release.upper),
                old["capt_validation_plugin_bayes_bounds"],
                strict=True,
            )
        ):
            raise ValueError(f"CAPT Bayes loss differs from saved result at context {context}")
        oracle = empirical_ldp_oracle(info, grid, settings.exact_max_k)
        row.update(
            status="evaluated",
            **info.metrics(),
            **_epsilon_one(info.entropy_release, oracle(1.0), cell["epsilon_cert"]),
            **search.crossing(
                oracle,
                info.entropy_release,
                cell["epsilon_cert"],
                informative=info.released_informative,
                preserves=info.preserves_raw_information,
            ),
            sdpi={
                name: sdpi_retention_check(info, epsilon, cell["epsilon_cert"])
                if epsilon is not None
                else None
                for name, epsilon in {"epsilon_1": 1.0, **cell["budgets"]}.items()
            },
            **_population_status(old, cell["epsilon_cert"]),
        )
        included.append((row, info, oracle))
    if not included:
        raise ValueError("all certified contexts have zero validation examples")
    aggregate = _aggregate(included, search, max(cell["epsilon_cert"] for cell in cells.values()))
    all_test = sum(r["test_n"] for r in records)
    for row in records:
        row["validation_weight"] = row["validation_n"] / aggregate["validation_n"]
        row["test_share_all_certified_contexts"] = row["test_n"] / all_test if all_test else None
    summary = {
        "K": first["K"],
        "L": first["L"],
        "profile": first["profile"],
        "context_column": first["context_column"],
        "results_dir": str(results_dir.resolve()),
        "certificate_dir": str(bundle.resolve()),
        "data_root": str(data_root.resolve()),
        "validation_days": list(settings.predictor_validation_days),
        "test_days": splits["D_test"],
        "grid": grid,
        "grid_subdivisions": grid_subdivisions,
        "exact_max_k": settings.exact_max_k,
        "cross_search": vars(search),
        "validation_weighting": "n_val_context / sum(n_val_context)",
        "test_n_all_certified_contexts": all_test,
        "aggregate": aggregate,
        "contexts": records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary = _json_finite(summary)
    output.mkdir(parents=True, exist_ok=False)
    (output / "theory.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    flat = [_flat(r) for r in summary["contexts"]]
    with (output / "contexts.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(dict.fromkeys(key for row in flat for key in row))
        )
        writer.writeheader()
        writer.writerows(flat)
    (output / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _report(summary: dict) -> str:
    results = [r for r in summary["contexts"] if r["status"] == "evaluated"]
    skipped = [r for r in summary["contexts"] if r["status"] == "skipped"]
    aggregate = summary["aggregate"]
    decisions = Counter(r["epsilon_1_empirical_decision"] for r in results)
    win, loss, unknown = (
        decisions[EMPIRICAL_PREFIX + name] for name in ("win", "loss", "inconclusive")
    )
    sdpi = {
        tag: sum(
            (r["sdpi"][tag] or {}).get("status") == "sufficient_condition_met" for r in results
        )
        for tag in ("epsilon_1", *METHODS)
    }
    losing = [r for r in results if r["epsilon_1_empirical_decision"] == "empirical_plugin_loss"]

    def components(rows):
        total = sum(r["validation_n"] for r in rows)
        return tuple(
            math.fsum(r[key] * r["validation_n"] for r in rows) / total if total else 0.0
            for key in ("i_y_z_given_c", "i_y_c_given_release")
        )

    def interval(row):
        upper = row["epsilon_cross_upper"]
        return f"{row['epsilon_cross_lower']}〜{upper if upper is not None else '未確定'}"

    interval_state = {True: "空", False: "非空", None: "未確定"}
    block, random = components(results)
    loss_block, loss_random = components(losing)
    block_dominant = sum(r["i_y_z_given_c"] > r["i_y_c_given_release"] for r in losing)
    random_dominant = sum(r["i_y_z_given_c"] < r["i_y_c_given_release"] for r in losing)
    lines = [
        "# 固定CAPTと元トークン上の最適LDP",
        "",
        f"ε=1の検証経験分布では **CAPT勝利 {win}、LDP勝利 {loss}、未確定 {unknown}**。"
        f"文脈内でチャネルと予測器を最適化するLDPを比較対象とした。母集団判定は `{aggregate['population_decision']}`。",
        "",
        "## 全体集約",
        "",
        f"K={summary['K']}、L={summary['L']}。検証日 {summary['validation_days']} の "
        f"**{aggregate['validation_n']:,} ユーザー日**を、文脈ごとの検証件数の比で加重した。"
        f"テスト日 {summary['test_days']} の {summary['test_n_all_certified_contexts']:,} 件は件数照合と内訳に使った。"
        "CAPTとLDPの損失は同じ検証分布で計算した。",
        "",
        f"- CAPTのBayes損失は **{aggregate['d_capt']:.9f}**。"
        f"最適LDPの境界は **{aggregate['epsilon_1_v_z_lower']:.9f}〜{aggregate['epsilon_1_v_z_upper']:.9f}** nats。",
        f"- 損失差 LDP − CAPT は **{aggregate['epsilon_1_gap_lower']:+.9f}〜{aggregate['epsilon_1_gap_upper']:+.9f}**。"
        f"全体判定は `{aggregate['epsilon_1_empirical_decision']}`。",
        f"- 全証明文脈の最大認証予算は {aggregate['epsilon_cert']:.12f}。"
        f"共通予算の交差点は {interval(aggregate)}（`{aggregate['epsilon_cross_status']}`）。"
        f"勝利区間は **{interval_state[aggregate['empirical_win_interval_empty']]}**。",
        f"- 追加損失はブロック化 **{block:.9f}**、ランダム化 **{random:.9f}** nats。",
        f"- 情報保持率の十分条件を満たした文脈数は ε=1で **{sdpi['epsilon_1']}**、"
        f"TV較正で **{sdpi['tv_calibrated']}**、一律較正で **{sdpi['uniform_calibrated']}**。"
        "条件が成立しない場合の判定はこの十分条件では未確定。",
        "",
    ]
    if losing:
        higher = "ブロック化" if loss_block > loss_random else "ランダム化"
        comparison = f"{higher}が大きい。" if loss_block != loss_random else "両者は同じ。"
        lines += [
            f"LDP勝利の {loss} 文脈に限ると、検証件数加重のブロック化損失は **{loss_block:.9f}**、"
            f"ランダム化損失は **{loss_random:.9f}** natsで、{comparison}"
            f"文脈別にはブロック化が大きいものが {block_dominant}、ランダム化が大きいものが {random_dominant}。"
            "この分解は固定済みチャネルの情報損失を配分している。",
            "",
        ]
    lines += ["## 予算と文脈別の判定", ""]
    for label, values in [
        ("CAPT設計ε", [r["epsilon_design"] for r in results]),
        ("CAPT認証ε", [r["epsilon_cert"] for r in results]),
        *((name, [r["ldp_budgets"][name] for r in results]) for name in METHODS),
    ]:
        found = [float(x) for x in values if x is not None]
        lines.append(
            f"- {label} の範囲は {min(found):.12f}〜{max(found):.12f}。"
            if found
            else f"- {label} の保存値は取得できなかった。"
        )
    lines += [
        "",
        "| 番号 | 公開文脈値 | 検証件数 | 認証ε | ε=1 | 交差点範囲 | 勝利区間 | ブロック化 | ランダム化 |",
        "| ---: | --- | ---: | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for r in results:
        verdict = r["epsilon_1_empirical_decision"].removeprefix(EMPIRICAL_PREFIX)
        state = interval_state[r["empirical_win_interval_empty"]]
        lines.append(
            f"| {r['context_index']:02d} | `{r['public_context_value']}` | {r['validation_n']:,} |"
            f" {r['epsilon_cert']:.12f} | {verdict} | {interval(r)} | {state} |"
            f" {r['i_y_z_given_c']:.9f} | {r['i_y_c_given_release']:.9f} |"
        )
    lines += ["", "## 評価対象外と解釈", ""]
    for r in skipped:
        lines.append(
            f"- 文脈番号 {r['context_index']:02d}（`{r['public_context_value']}`）は検証0件。"
            f"訓練 {r['train_n']:,} 件、テスト {r['test_n']:,} 件。"
        )
    lines += [
        "",
        "情報損失は検証カウントから平滑化なしで計算し、条件付き相互情報量は条件付きKLから独立に照合した。"
        f"恒等式の最大絶対残差は {max(abs(r['decomposition_residual']) for r in results):.3g} nats。",
        "",
        "格子の比較分布は、元トークンLDPすべてに対する下界と、元トークン上で実行可能な機構による上界を与える。"
        f"格子点は {len(summary['grid'])} 個。交差点探索は上限 {summary['cross_search']['cap']}、"
        f"二分探索の予算幅 {summary['cross_search']['tolerance']}。"
        "境界未確定の場合は得られた片側のみを報告し、有限の交差点を探索範囲内で挟めた場合と区別した。",
        "",
        "この結果は指定プロファイルを適用したログの経験分布についての比較。"
        "母集団の結論には、検証・テストの標本が同じ対象法則を表す根拠、依存やシフトを考慮した効用区間、"
        "および文脈横断の被覆確率が必要。勝利の認証にはCAPTの上側信頼限界が最適LDPの下側信頼限界を下回る必要がある。"
        "既存の母集団判定がある場合は、予算1と一致する文脈別の区間を引き継ぐ。",
        "",
        f"計算時間は {summary['elapsed_seconds']:.1f} 秒。入力比較結果、証明書、設定、格子は `theory.json` に記録した。",
    ]
    return "\n".join(lines) + "\n"
