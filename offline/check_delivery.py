"""Audit portable assets and frozen-model inference; never train or modify assets.

Forecast checks cover every saved seed/model on deterministic test samples.
Anomaly checks cover contexts with test observations, with up to 24 rows per context,
across all four methods and selected KPIs. All saved artifacts are hash-checked.
Use detect_saved.py to reproduce an entire anomaly task/method separately.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
sys.path.insert(0, str(BASE / "forecast/code"))
sys.path.insert(0, str(BASE / "anomaly/code"))
from common import TASKS, MODELS, NEURAL_MODELS, load_data
from evaluate import predict_saved
from detect_saved import load_task, task_folders, score_frame, compare_archived
from calibrated_detectors import METHODS


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def test_sample_ids(length, maximum):
    return np.unique(np.linspace(0, length - 1, min(length, maximum), dtype=int))


def verify_hashes():
    inventory = json.loads((BASE / "verification/source_inventory.json").read_text(encoding="utf-8"))
    for item in inventory["entries"]:
        path = PROJECT / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Artifact changed: {item['path']}")
    baseline = json.loads((BASE / "verification/gotsf_unchanged.json").read_text(encoding="utf-8"))
    for name, expected in baseline["files"].items():
        if digest(PROJECT / name) != expected:
            raise ValueError(f"GOTSF file changed: {name}")
    return dict(source_files=len(inventory["entries"]), gotsf_files=len(baseline["files"]), passed=True)


def audit_forecasts():
    records = []
    for task in TASKS:
        folder = BASE / "forecast/results" / task
        x, y, meta, spec = load_data(folder / "data")
        test = meta.split.eq("test").to_numpy()
        test_x, test_y = x[test], y[test]
        selected = test_sample_ids(len(test_x), 32)
        test_index = meta.loc[test].reset_index(drop=True)
        for name in MODELS:
            seeds = (42,) if name == "Persistence" else (42, 43, 44)
            for seed in seeds:
                model_dir = folder / "models" / name
                archived_name = "test_predictions.csv" if name == "Persistence" else f"test_predictions_seed_{seed}.csv"
                saved = pd.read_csv(model_dir / archived_name, float_precision="round_trip")
                if not saved[["series_id", "target_sequence"]].equals(test_index[["series_id", "target_sequence"]]):
                    raise ValueError(f"Prediction row alignment mismatch: {task}/{name}/{seed}")
                np.testing.assert_allclose(saved[[f"actual__{k}" for k in spec["targets"]]].to_numpy(),
                                           test_y, rtol=1e-10, atol=1e-10)
                if name != "Persistence":
                    run = json.loads((model_dir / "run.json").read_text(encoding="utf-8"))
                    entry = next(v for v in run["seed_runs"] if v["seed"] == seed)
                    suffix = "pt" if name in NEURAL_MODELS else "joblib"
                    if digest(model_dir / f"seed_{seed}.{suffix}") != entry["artifact_sha256"]:
                        raise ValueError("Model differs from original training manifest.")
                predicted = predict_saved(folder, name, seed, test_x[selected], spec)
                expected = saved.iloc[selected][[f"predicted__{k}" for k in spec["targets"]]].to_numpy()
                # Original neural CSVs used CUDA; current portable inference uses CPU.
                tolerance = 2e-5 if name in NEURAL_MODELS else 1e-9
                np.testing.assert_allclose(predicted, expected, rtol=tolerance, atol=tolerance)
                record = dict(task=task, model=name, seed=seed, checked_rows=len(selected),
                              total_test_rows=int(test.sum()), passed=True,
                              maximum_prediction_difference=float(np.max(np.abs(predicted-expected))),
                              atol=tolerance, rtol=tolerance)
                records.append(record)
                del predicted, saved
                gc.collect()
            print(f"PASS forecast {task}/{name}", flush=True)
        del x, y, test_x, test_y
        gc.collect()
    return records


def audit_anomalies():
    records = []
    for task, folder in task_folders().items():
        manifest, inputs, summary = load_task(folder)
        positions = []
        for indexes in inputs.groupby("context", sort=True).indices.values():
            positions.extend(indexes[test_sample_ids(len(indexes), 24)])
        sampled = inputs.iloc[sorted(positions)].reset_index(drop=True)
        archived_or = None
        or_path = BASE / "anomaly/label_statistics/datasets" / task / "record_or_predictions.csv.gz"
        if or_path.exists():
            archived_or = pd.read_csv(or_path, dtype={"row_id": str}).set_index("row_id")
        for method in METHODS:
            result, models = score_frame(folder, manifest, sampled, summary, method)
            maximum = compare_archived(folder, result, manifest["features"], method)
            if archived_or is not None:
                col = f"any_kpi__{method}__flag"
                np.testing.assert_array_equal(result[col], archived_or.loc[result.row_id, col])
            records.append(dict(task=task, method=method, checked_rows=len(sampled),
                                total_test_rows=len(inputs), loaded_detectors=models,
                                contexts=int(sampled.context.nunique()), passed=True,
                                archived_OR_checked=archived_or is not None,
                                maximum_score_difference=maximum))
            print(f"PASS anomaly {task}/{method}: {models} rules, {len(sampled)} rows", flush=True)
        del inputs, sampled
        gc.collect()
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="NEW JSON report path")
    p.add_argument("--only", choices=["all", "hashes", "forecast", "anomaly"], default="all")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output report already exists.")
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")
    report = dict(scope=args.only, datasets=["donext", "netslab_5g_oran_idd"],
                  versions={n: importlib.metadata.version(n) for n in
                            ["numpy", "pandas", "torch", "scikit-learn", "xgboost", "lightgbm", "joblib"]})
    if args.only in ("all", "hashes"):
        report["hashes"] = verify_hashes()
        print("PASS all packaged source hashes and GOTSF runtime hashes", flush=True)
    if args.only in ("all", "forecast"):
        report["forecasts"] = audit_forecasts()
    if args.only in ("all", "anomaly"):
        report["anomalies"] = audit_anomalies()
    report.update(passed=True, seconds=round(time.perf_counter()-start, 3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"PASS {args.only}; report -> {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
