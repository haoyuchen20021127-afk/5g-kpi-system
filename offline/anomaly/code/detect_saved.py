"""Score packaged test observations with their original per-context detectors.

No refitting, threshold tuning, clipping, or changed train/test assignments.
Record decisions use OR within ONE method; unknown never means normal.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from calibrated_detectors import METHODS, compare_score
from summarize_by_label import record_or

RESULTS = Path(__file__).resolve().parents[1] / "results"


def task_folders():
    return {f"{p.parent.parent.name}/{p.parent.name}": p.parent
            for p in sorted((RESULTS / "datasets").glob("*/*/task_manifest.json"))}


def load_task(folder):
    manifest = json.loads((folder / "task_manifest.json").read_text(encoding="utf-8"))
    # Round-trip parsing matters for values lying extremely close to a threshold.
    inputs = pd.read_csv(folder / "model_inputs.csv.gz", float_precision="round_trip",
                         dtype={"row_id": str, "context": str, "source": str})
    inputs = inputs.loc[inputs.split.eq("test")].reset_index(drop=True)
    if inputs.row_id.duplicated().any() or len(inputs) != int(manifest["rows_test"]):
        raise ValueError("Invalid test row IDs/count.")
    summary = pd.read_csv(folder / "kpi_summary.csv", float_precision="round_trip",
                          keep_default_na=False, dtype={"context": str})
    if summary.duplicated(["context", "feature", "method"]).any():
        raise ValueError("Duplicate detector definitions.")
    return manifest, inputs, summary


def get_detector(folder, row):
    path = (folder / row["model_file"]).resolve()
    if not path.is_relative_to((folder / "models").resolve()):
        raise ValueError("Model path escapes this task's model directory.")
    detector = joblib.load(path)
    if detector.method != row["method"] or detector.status != "ok":
        raise ValueError("Model method/status differs from the saved summary.")
    if not np.isclose(detector.threshold, float(row["threshold"]), rtol=1e-12, atol=1e-12):
        raise ValueError("Model threshold differs from the saved summary.")
    return detector


def score_frame(folder, manifest, inputs, summary, method):
    features = manifest["features"]
    scores = np.full((len(inputs), len(features)), np.nan)
    flags = np.full(scores.shape, -1, dtype=np.int8)
    applicable = np.zeros(scores.shape, dtype=bool)
    used = 0
    for context, positions in inputs.groupby("context", sort=True).indices.items():
        definitions = summary.loc[summary.context.eq(context) & summary.method.eq(method)]
        for j, feature in enumerate(features):
            matches = definitions.loc[definitions.feature.eq(feature)]
            if len(matches) != 1:
                raise ValueError(f"Missing/duplicate detector: {context}/{feature}/{method}")
            row = matches.iloc[0]
            if row["status"] != "ok":
                continue  # No usable fitted/reference rule: retain score NaN, flag -1.
            detector = get_detector(folder, row)
            values = inputs.iloc[positions][f"{feature}__model_input"].to_numpy(float)
            score = detector.score(values)
            scores[positions, j] = score
            flags[positions, j] = compare_score(score, detector.threshold,
                                               detector.comparison_method())[0]
            applicable[positions, j] = True
            used += 1
    identifiers = [c for c in ("row_id", "source", "context", "session", "time", "label")
                   if c in inputs]
    result = inputs[identifiers].copy()
    for j, feature in enumerate(features):
        result[f"{feature}__model_input"] = inputs[f"{feature}__model_input"].to_numpy()
        result[f"{feature}__input_imputed"] = inputs[f"{feature}__input_imputed"].to_numpy()
        result[f"{feature}__{method}__score"] = scores[:, j]
        result[f"{feature}__{method}__flag"] = flags[:, j]
    decision, count, observed, abnormal, missing = record_or(flags, applicable)
    for suffix, values in zip(("flag", "applicable_n", "observed_n", "abnormal_n", "missing_n"),
                              (decision, count, observed, abnormal, missing)):
        result[f"any_kpi__{method}__{suffix}"] = values
    return result, used


def compare_archived(folder, result, features, method):
    columns = [f"{f}__{method}__{kind}" for f in features for kind in ("score", "flag")]
    archived = pd.read_csv(folder / "test_predictions.csv.gz", usecols=["row_id", *columns],
                           float_precision="round_trip", dtype={"row_id": str})
    if archived.row_id.duplicated().any():
        raise ValueError("Duplicate archived row IDs.")
    archived = archived.set_index("row_id").loc[result.row_id]
    maximum = 0.
    for column in columns:
        actual, expected = result[column].to_numpy(), archived[column].to_numpy()
        if column.endswith("__flag"):
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
            finite = np.isfinite(actual) & np.isfinite(expected)
            if finite.any():
                maximum = max(maximum, float(np.abs(actual[finite]-expected[finite]).max()))
    return maximum


def main():
    folders = task_folders()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(folders), required=True)
    parser.add_argument("--method", choices=METHODS, default="isolation_forest")
    parser.add_argument("--output", type=Path, required=True, help="A NEW result directory.")
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(RESULTS.resolve()):
        parser.error("Output must be a new directory outside packaged results.")
    folder = folders[args.task]
    manifest, inputs, summary = load_task(folder)
    result, used = score_frame(folder, manifest, inputs, summary, args.method)
    difference = compare_archived(folder, result, manifest["features"], args.method)
    flag = result[f"any_kpi__{args.method}__flag"]
    report = dict(task=args.task, method=args.method, test_rows=len(inputs), loaded_detectors=used,
                  suspected=int(flag.eq(1).sum()), not_flagged=int(flag.eq(0).sum()),
                  undetermined=int(flag.eq(-1).sum()), archived_flags_equal=True,
                  maximum_score_difference=difference, retrained=False,
                  label_scope=manifest["label_scope"], rule="OR within this method only")
    args.output.mkdir(parents=True, exist_ok=False)
    result.to_csv(args.output / "predictions.csv.gz", index=False, compression="gzip")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
