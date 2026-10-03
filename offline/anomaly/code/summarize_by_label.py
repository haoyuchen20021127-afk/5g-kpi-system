"""Summarize saved per-KPI decisions by label without fitting or changing a detector.

Record decisions are OR decisions within each method, NOT the separately calibrated
row__METHOD__flag fields in the source files. Unknown is never counted as normal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


METHODS = ("three_sigma", "iqr", "isolation_forest", "one_class_svm")
LABEL_NAMES = {0: "benign", 1: "attack", -1: "unknown"}
ALL_CONTEXTS = "ALL_CONTEXTS"


def validate_flags(values, name="flags"):
    """Require actual finite numeric -1/0/1 values; never silently fill missing."""
    array = np.asarray(values)
    if not is_numeric_dtype(array.dtype) or is_bool_dtype(array.dtype):
        raise ValueError(f"{name}: flags must have a numeric, non-boolean dtype")
    if not np.isfinite(array).all() or not np.isin(array, (-1, 0, 1)).all():
        raise ValueError(f"{name}: expected only -1, 0, 1; missing/invalid flags found")
    return array.astype(np.int8, copy=False)


def label_groups(labels, label_scope):
    labels = validate_flags(labels, "labels")
    if label_scope == "attack_state":
        return np.array([LABEL_NAMES[int(x)] for x in labels], dtype=object)
    if label_scope not in {"unlabelled", "unlabelled_exploration"}:
        raise ValueError(f"Unsupported label_scope: {label_scope}")
    if (labels != -1).any():
        raise ValueError("Unlabelled task unexpectedly contains a known attack-state label")
    return np.full(len(labels), "unlabelled", dtype=object)


def record_or(flags, applicable):
    """Return OR plus counts, using the applicable KPI set defined before scoring.

    1 if any applicable flag is 1; 0 only if there is an applicable KPI and all
    applicable flags are 0; -1 otherwise. Inapplicable KPIs must carry flag -1.
    """
    flags = validate_flags(flags)
    applicable = np.asarray(applicable)
    if flags.ndim != 2 or applicable.dtype != np.bool_ or flags.shape != applicable.shape:
        raise ValueError("flags and boolean applicable must be equally shaped 2-D matrices")
    if np.any((~applicable) & (flags != -1)):
        raise ValueError("A non-applicable KPI has a scored flag")
    applicable_n = applicable.sum(axis=1)
    observed_n = ((flags != -1) & applicable).sum(axis=1)
    abnormal_n = ((flags == 1) & applicable).sum(axis=1)
    missing_n = applicable_n - observed_n
    decision = np.full(len(flags), -1, dtype=np.int8)
    decision[(applicable_n > 0) & (missing_n == 0) & (abnormal_n == 0)] = 0
    decision[abnormal_n > 0] = 1
    return decision, applicable_n, observed_n, abnormal_n, missing_n


def flag_statistics(flags):
    flags = validate_flags(flags)
    counts = {"total_n": int(flags.size), "abnormal_n": int((flags == 1).sum()),
              "not_abnormal_n": int((flags == 0).sum()),
              "undetermined_n": int((flags == -1).sum())}
    counts["evaluable_n"] = counts["abnormal_n"] + counts["not_abnormal_n"]
    counts["abnormal_fraction"] = (counts["abnormal_n"] / counts["evaluable_n"]
                                     if counts["evaluable_n"] else np.nan)
    counts["abnormal_fraction_all"] = (counts["abnormal_n"] / counts["total_n"]
                                         if counts["total_n"] else np.nan)
    counts["coverage"] = counts["evaluable_n"] / counts["total_n"] if counts["total_n"] else np.nan
    # Independent count checks, rather than counting unknown as non-abnormal.
    frequency = dict(zip(*np.unique(flags, return_counts=True)))
    assert counts["abnormal_n"] == frequency.get(1, 0)
    assert counts["not_abnormal_n"] == frequency.get(0, 0)
    assert counts["undetermined_n"] == frequency.get(-1, 0)
    assert counts["total_n"] == sum(frequency.values())
    assert counts["total_n"] == counts["evaluable_n"] + counts["undetermined_n"]
    return counts


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_task(folder: Path, output: Path):
    manifest_path = folder / "task_manifest.json"
    summary_path = folder / "kpi_summary.csv"
    predictions_path = folder / "test_predictions.csv.gz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset, task = manifest["dataset"], manifest["task_id"]
    features = manifest["features"]
    methods = manifest["methods"]
    if not features or len(features) != len(set(features)) or set(methods) != set(METHODS):
        raise ValueError(f"{dataset}/{task}: invalid features or method definitions")
    summary = pd.read_csv(summary_path, keep_default_na=False,
                          dtype={"context": str, "feature": str, "method": str, "status": str})
    if summary.duplicated(["context", "feature", "method"]).any():
        raise ValueError("Duplicate context-feature-method status definitions")
    required = ["row_id", "context", "label"]
    optional = ["source", "session", "time", "utc", "test_order", "attack_category", "attack_subcategory"]
    flag_columns = [f"{feature}__{method}__flag" for method in METHODS for feature in features]
    available = pd.read_csv(predictions_path, nrows=0).columns
    missing = set(required + flag_columns) - set(available)
    if missing:
        raise ValueError(f"{dataset}/{task}: missing prediction fields: {sorted(missing)}")
    metadata_columns = required + [x for x in optional if x in available]
    frame = pd.read_csv(predictions_path, usecols=metadata_columns + flag_columns,
                        dtype={"row_id": str, "context": str}, keep_default_na=False)
    if frame.row_id.eq("").any() or frame.row_id.duplicated().any() or frame.context.eq("").any():
        raise ValueError("Empty/duplicate row IDs or empty contexts")
    if len(frame) != int(manifest["rows_test"]):
        raise ValueError("Saved test row count does not match task manifest")
    if frame.context.eq(ALL_CONTEXTS).any():
        raise ValueError("Reserved summary context name occurs in real data")
    frame["label_group"] = label_groups(frame.label.to_numpy(), manifest["label_scope"])
    statuses = summary.set_index(["context", "feature", "method"])["status"]
    output_frame = frame[metadata_columns + ["label_group"]].copy()
    output_frame.insert(0, "task", task)
    output_frame.insert(0, "dataset", dataset)
    contexts = frame.context.to_numpy()
    groups = [("context", str(context), np.flatnonzero(contexts == context))
              for context in sorted(frame.context.unique())]
    groups.append(("all_contexts", ALL_CONTEXTS, np.arange(len(frame))))
    kpi_rows, record_rows = [], []
    subtype_kpi_rows, subtype_record_rows = [], []
    has_subtypes = dataset == "netslab_5g_oran_idd" and task == "radio_kpi" and "attack_subcategory" in frame
    for method in METHODS:
        columns = [f"{feature}__{method}__flag" for feature in features]
        values = validate_flags(frame[columns].to_numpy(), f"{dataset}/{task}/{method}")
        applicable = np.zeros(values.shape, dtype=bool)
        status_by_context = {}
        for context in frame.context.unique():
            key = [(context, feature, method) for feature in features]
            if any(k not in statuses.index for k in key):
                raise ValueError(f"Incomplete applicable-KPI definition: {context}/{method}")
            feature_status = statuses.loc[key].to_numpy()
            status_by_context[context] = feature_status
            applicable[contexts == context] = feature_status == "ok"
        decision, applicable_n, observed_n, abnormal_n, missing_n = record_or(values, applicable)
        output_frame[f"any_kpi__{method}__flag"] = decision
        output_frame[f"any_kpi__{method}__applicable_n"] = applicable_n
        output_frame[f"any_kpi__{method}__observed_n"] = observed_n
        output_frame[f"any_kpi__{method}__abnormal_n"] = abnormal_n
        output_frame[f"any_kpi__{method}__missing_n"] = missing_n
        # A second, independent formulation validates the complete row decision vector.
        expected = np.where((values == 1).any(axis=1), 1,
                            np.where((applicable_n > 0) & (((values == 0) | (~applicable)).all(axis=1)), 0, -1))
        np.testing.assert_array_equal(decision, expected)
        for scope, context, indexes in groups:
            for label in sorted(frame.iloc[indexes].label_group.unique()):
                ix = indexes[frame.label_group.to_numpy()[indexes] == label]
                header = {"dataset": dataset, "task": task, "label_scope": manifest["label_scope"],
                          "scope": scope, "context": context, "label_group": label,
                          "label_code": next((k for k, v in LABEL_NAMES.items() if v == label), -1),
                          "method": method}
                record_counts = flag_statistics(decision[ix])
                record_rows.append({**header, **record_counts,
                                    "applicable_kpi_min": int(applicable_n[ix].min()),
                                    "applicable_kpi_max": int(applicable_n[ix].max()),
                                    "no_applicable_kpi_n": int((applicable_n[ix] == 0).sum()),
                                    "rows_with_missing_applicable_kpi_n": int((missing_n[ix] > 0).sum())})
                for j, feature in enumerate(features):
                    counts = flag_statistics(values[ix, j])
                    # Compare counts on the same row set, never unequal-denominator rates.
                    assert record_counts["abnormal_n"] >= counts["abnormal_n"]
                    assert np.all(decision[ix][values[ix, j] == 1] == 1)
                    kpi_rows.append({**header, "feature": feature,
                                     "detector_status": str(status_by_context[context][j]) if scope == "context" else "aggregate",
                                     "applicable_n": int(applicable[ix, j].sum()), **counts})
            if has_subtypes:
                for (subtype, parent), local_rows in frame.iloc[indexes].groupby(
                        ["attack_subcategory", "label_group"], dropna=False, sort=True):
                    ix = local_rows.index.to_numpy()
                    header = {"dataset": dataset, "task": task, "label_scope": manifest["label_scope"],
                              "source_label_field": "attack_subcategory", "scope": scope,
                              "context": context, "label_group": subtype, "parent_label_group": parent,
                              "method": method}
                    record_counts = flag_statistics(decision[ix])
                    subtype_record_rows.append({**header, **record_counts,
                                                "applicable_kpi_min": int(applicable_n[ix].min()),
                                                "applicable_kpi_max": int(applicable_n[ix].max()),
                                                "no_applicable_kpi_n": int((applicable_n[ix] == 0).sum()),
                                                "rows_with_missing_applicable_kpi_n": int((missing_n[ix] > 0).sum())})
                    for j, feature in enumerate(features):
                        counts = flag_statistics(values[ix, j])
                        assert record_counts["abnormal_n"] >= counts["abnormal_n"]
                        subtype_kpi_rows.append({**header, "feature": feature,
                                                 "detector_status": str(status_by_context[context][j]) if scope == "context" else "aggregate",
                                                 "applicable_n": int(applicable[ix, j].sum()), **counts})
    kpi_frame, record_frame = pd.DataFrame(kpi_rows), pd.DataFrame(record_rows)
    validate_scope_totals(kpi_frame, ["method", "feature", "label_group"])
    validate_scope_totals(record_frame, ["method", "label_group"])
    supplement = {"available": False}
    if has_subtypes:
        sub_kpi, sub_record = pd.DataFrame(subtype_kpi_rows), pd.DataFrame(subtype_record_rows)
        validate_scope_totals(sub_kpi, ["method", "feature", "label_group", "parent_label_group"])
        validate_scope_totals(sub_record, ["method", "label_group", "parent_label_group"])
        validate_subcategory_totals(sub_kpi, kpi_frame, ["method", "feature"])
        validate_subcategory_totals(sub_record, record_frame, ["method"])
        output.mkdir(parents=True, exist_ok=True)
        sub_kpi.to_csv(output / "per_kpi_by_attack_subcategory.csv", index=False, encoding="utf-8-sig")
        sub_record.to_csv(output / "record_any_kpi_by_attack_subcategory.csv", index=False, encoding="utf-8-sig")
        supplement = {"available": True, "source_label_field": "attack_subcategory",
                      "labels": {str(k): int(v) for k, v in frame.attack_subcategory.value_counts(dropna=False).items()},
                      "test_n": len(frame), "category_total_verified": True,
                      "source_values_preserved": True, "parent_label_totals_verified": True,
                      "files": ["per_kpi_by_attack_subcategory.csv", "record_any_kpi_by_attack_subcategory.csv"],
                      "separate_supplement_not_added_to_binary_totals": True}
    task_output = output / "datasets" / dataset / task
    task_output.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(task_output / "record_or_predictions.csv.gz", index=False,
                        compression="gzip", encoding="utf-8")
    info = {"dataset": dataset, "task": task, "label_scope": manifest["label_scope"],
            "test_n": len(frame), "features": features, "methods": list(METHODS),
            "context_n": int(frame.context.nunique()),
            "labels": {str(k): int(v) for k, v in frame.label_group.value_counts().items()},
            "input_sha256": {p.name: sha256(p) for p in (manifest_path, summary_path, predictions_path)},
            "output_record_file": str((task_output / "record_or_predictions.csv.gz").relative_to(output)),
            "attack_subcategory_supplement": supplement,
            "source_predictions_unchanged": True}
    (task_output / "task_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return kpi_frame, record_frame, info


def validate_scope_totals(table, group_columns):
    """Independently reconstruct all-context integer denominators from context rows."""
    counts = ["total_n", "abnormal_n", "not_abnormal_n", "undetermined_n", "evaluable_n"]
    local = table[table.scope == "context"].groupby(group_columns, dropna=False)[counts].sum().sort_index()
    overall = table[table.scope == "all_contexts"].set_index(group_columns)[counts].sort_index()
    pd.testing.assert_frame_equal(local, overall, check_dtype=False)
    for row in table.itertuples(index=False):
        assert row.total_n == row.abnormal_n + row.not_abnormal_n + row.undetermined_n
        assert row.evaluable_n == row.abnormal_n + row.not_abnormal_n
        assert (np.isnan(row.abnormal_fraction) if row.evaluable_n == 0
                else np.isclose(row.abnormal_fraction, row.abnormal_n / row.evaluable_n))
        assert np.isclose(row.abnormal_fraction_all, row.abnormal_n / row.total_n)
        assert np.isclose(row.coverage, row.evaluable_n / row.total_n)


def validate_subcategory_totals(supplement, primary, additional_keys):
    """Detailed source labels must partition, not enlarge, each binary-label group."""
    counts = ["total_n", "abnormal_n", "not_abnormal_n", "undetermined_n", "evaluable_n"]
    keys = ["scope", "context", *additional_keys, "label_group"]
    detailed = supplement.drop(columns=["label_group"]).rename(columns={"parent_label_group": "label_group"})
    detailed = detailed.groupby(keys, dropna=False)[counts].sum().sort_index()
    baseline = primary.set_index(keys)[counts].sort_index()
    pd.testing.assert_frame_equal(detailed, baseline, check_dtype=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "analysis")
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--expected-tasks", type=int)
    args = parser.parse_args()
    start = time.perf_counter()
    manifests = sorted((args.results / "datasets").glob("*/*/task_manifest.json"))
    if not manifests or (args.expected_tasks is not None and len(manifests) != args.expected_tasks):
        raise ValueError(f"Unexpected number of tasks: {len(manifests)}")
    args.output.mkdir(parents=True, exist_ok=True)
    kpi_frames, record_frames, information = [], [], []
    for manifest in manifests:
        kpi, record, info = summarize_task(manifest.parent, args.output)
        kpi_frames.append(kpi)
        record_frames.append(record)
        information.append(info)
        print(f"Summarized {info['dataset']}/{info['task']}: {info['test_n']} test records", flush=True)
    total = sum(x["test_n"] for x in information)
    if args.expected_total is not None and total != args.expected_total:
        raise ValueError(f"Unexpected test row count: {total} != {args.expected_total}")
    pd.concat(kpi_frames, ignore_index=True).to_csv(args.output / "per_kpi_by_label.csv", index=False, encoding="utf-8-sig")
    pd.concat(record_frames, ignore_index=True).to_csv(args.output / "record_any_kpi_by_label.csv", index=False, encoding="utf-8-sig")
    audit = {"status": "complete", "results_source": str(args.results.resolve()),
             "test_n": total, "task_n": len(information), "elapsed_seconds": round(time.perf_counter() - start, 3),
             "rule": "OR separately within each method across context-applicable (status=ok) KPI flags; no cross-method OR",
             "undetermined_rule": "No observed anomaly and any applicable KPI missing, or no applicable KPI",
             "rate_denominator": "evaluable_n = abnormal_n + not_abnormal_n; zero denominator -> NaN",
             "all_record_rate": "abnormal_n / total_n; reported separately with coverage",
             "label_caution": "Attack-state labels are not KPI-health truth. Unknown and unlabelled are not benign.",
             "record_caution": "OR alarm probability can increase with the number of applicable KPIs; not a calibrated simultaneous error guarantee.",
             "model_refitted": False, "source_flags_modified": False, "old_calibrated_row_flags_used": False,
             "all_scope_integer_denominators_verified": True, "record_or_contains_every_kpi_alarm_verified": True,
             "attack_subcategory_supplements": [{"dataset": x["dataset"], "task": x["task"], **x["attack_subcategory_supplement"]}
                                                for x in information if x["attack_subcategory_supplement"]["available"]],
             "tasks": information}
    (args.output / "task_info.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Complete: {len(information)} tasks, {total} test records, {audit['elapsed_seconds']} seconds", flush=True)


if __name__ == "__main__":
    main()
