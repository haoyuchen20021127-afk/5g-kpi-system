"""统一的多变量窗口、指标及实验文件接口。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "test")
TRADITIONAL_MODELS = ("LinearRegression", "DecisionTree", "RandomForest", "XGBoost", "LightGBM")
NEURAL_MODELS = ("LSTM", "GRU", "Transformer")
MODELS = ("Persistence", *TRADITIONAL_MODELS, *NEURAL_MODELS)
SEEDS = (42, 43, 44)
TASKS = {
    "donext/cell": {"targets": ["RSRP", "RSRQ", "SINR", "CQI"], "raw": ["rsrp", "rsrq", "sinr", "cqi"], "window": 12, "resolution": "5 minutes", "stems": ["hbahn__cell_data", "mobile__cell_data", "static__cell_data"], "groups": []},
    "donext/iperf": {"targets": ["DataRate", "RSRP", "SINR", "CQI"], "raw": ["datarate", "rsrp", "sinr", "cqi"], "window": 12, "resolution": "5 minutes", "stems": ["hbahn__iperf_data", "mobile__iperf_data"], "groups": ["protocol", "direction"]},
    "donext/latency": {"targets": ["MeanLatency", "RSRP", "SINR", "CQI"], "raw": ["mean_latency", "rsrp", "sinr", "cqi"], "window": 12, "resolution": "5 minutes", "stems": ["hbahn__latency_data", "mobile__latency_data", "static__latency_data"], "groups": ["server_ip"]},
    "netslab_5g_oran_idd/joint_kpi": {"targets": ["RSRP", "RSRQ", "SINR", "CQI", "DLBLER", "ULBLER"], "window": 24, "resolution": "1 second"},
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array_digest(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    result = hashlib.sha256(str((contiguous.shape, contiguous.dtype.str)).encode())
    result.update(contiguous.tobytes())
    return result.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.12g")
    temporary.replace(path)


def metrics(actual: np.ndarray, predicted: np.ndarray, targets: list[str]) -> list[dict]:
    y, p = np.asarray(actual, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != len(targets) or not len(y):
        raise ValueError("多输出预测与标签形状不一致。")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("指标输入含缺失值或无穷值。")
    rows = []
    for index, target in enumerate(targets):
        error = p[:, index] - y[:, index]
        sst = np.square(y[:, index] - y[:, index].mean()).sum()
        rows.append({"target": target, "MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt(np.square(error).mean())), "R2": float(1 - np.square(error).sum() / sst) if sst > 0 else None, "n_test": len(y)})
    return rows


def macro_rmse(actual, predicted) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)), axis=0)).mean())


def make_windows(series: pd.DataFrame, targets: list[str], window: int, minimum_points: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    arrays, labels, metadata = [], [], []
    audit = {"input_bins": len(series), "segments_total": 0, "segments_retained": 0, "bins_in_short_segments": 0, "warmup_bins": 0}
    for series_id, group in series.groupby("series_id", sort=True, observed=True):
        group = group.sort_values("time_index", kind="stable").reset_index(drop=True)
        if group["time_index"].duplicated().any() or not np.all(np.diff(group["time_index"]) == 1):
            raise ValueError("窗口构建要求单一实体的连续时间箱。")
        audit["segments_total"] += 1
        if len(group) < max(minimum_points, window + 3):
            audit["bins_in_short_segments"] += len(group)
            continue
        values = group[targets].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("共同观测序列包含无效 KPI 值。")
        windows = np.moveaxis(np.lib.stride_tricks.sliding_window_view(values, window + 1, axis=0), -1, 1)
        arrays.append(windows[:, :window].copy())
        labels.append(windows[:, window].copy())
        meta = group.loc[window:, ["series_id", "time_index", "target_time", "time_basis"]].copy()
        meta = meta.rename(columns={"time_index": "target_sequence"}).reset_index(drop=True)
        fraction = np.arange(len(meta)) / len(meta)
        meta["split"] = np.select([fraction < .70, fraction < .85], ["train", "validation"], default="test")
        if set(meta["split"]) != set(SPLITS):
            raise ValueError("保留片段不足以形成三种划分。")
        metadata.append(meta)
        audit["segments_retained"] += 1
        audit["warmup_bins"] += window
    if not arrays:
        raise ValueError("没有足够长的共同连续序列。")
    return np.concatenate(arrays), np.concatenate(labels), pd.concat(metadata, ignore_index=True), audit


def validate_arrays(x: np.ndarray, y: np.ndarray, meta: pd.DataFrame, spec: dict) -> dict:
    n, window, channels = x.shape
    if (window, channels) != (spec["window"], len(spec["targets"])) or y.shape != (n, channels) or len(meta) != n:
        raise ValueError("统一数据形状错误。")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or meta[["series_id", "target_sequence", "split"]].isna().any().any():
        raise ValueError("数据含无效值。")
    if meta.duplicated(["series_id", "target_sequence"]).any():
        raise ValueError("样本键重复。")
    checked = 0
    for _, group in meta.groupby("series_id", sort=False):
        positions = group.sort_values("target_sequence").index.to_numpy()
        ranks = meta.loc[positions, "split"].map({"train": 0, "validation": 1, "test": 2}).to_numpy()
        if np.any(np.diff(ranks) < 0):
            raise ValueError("序列内部划分时间顺序错误。")
        adjacent = np.diff(meta.loc[positions, "target_sequence"]) == 1
        earlier, later = positions[:-1][adjacent], positions[1:][adjacent]
        if not np.allclose(x[later, :-1], x[earlier, 1:], atol=2e-6, rtol=2e-6) or not np.allclose(x[later, -1], y[earlier], atol=2e-6, rtol=2e-6):
            raise ValueError("历史窗口和目标的连续对应关系错误。")
        checked += len(earlier)
    flat = x.reshape(n, -1)
    if not np.array_equal(flat.reshape(x.shape), x):
        raise ValueError("传统模型与 LSTM 输入值不一致。")
    counts = {split: int(meta["split"].eq(split).sum()) for split in SPLITS}
    if min(counts.values()) == 0:
        raise ValueError("某个划分为空。")
    return {"split_rows": counts, "adjacent_pairs_checked": checked, "flat_sequence_exact_equality": True, "X_sha256": array_digest(x), "Y_sha256": array_digest(y), "sequence_count": int(meta["series_id"].nunique())}


def save_data(folder: Path, task: str, x, y, meta, details) -> dict:
    spec = TASKS[task]
    audit = validate_arrays(x, y, meta, spec)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "samples.npz.part").open("wb") as handle:
        np.savez_compressed(handle, X=x, Y=y)
    (folder / "samples.npz.part").replace(folder / "samples.npz")
    save_csv(folder / "sample_index.csv", meta)
    feature_names = [f"lag_{lag}__{target}" for lag in range(spec["window"], 0, -1) for target in spec["targets"]]
    manifest = {"task": task, **spec, **audit, **details, "n_samples": len(meta), "input_shape": list(x.shape), "output_shape": list(y.shape), "flat_feature_order": feature_names, "unit": "z-score", "new_scaling_or_imputation": False, "input_contract": "all models receive exactly the same X values; traditional estimators use C-order flattening", "limitation": "Inherited full-source preprocessing statistics; not an end-to-end leakage-free deployment benchmark."}
    save_json(folder / "manifest.json", manifest)
    return manifest


def load_data(folder: Path):
    spec = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    with np.load(folder / "samples.npz", allow_pickle=False) as payload:
        x, y = payload["X"], payload["Y"]
    meta = pd.read_csv(folder / "sample_index.csv")
    audit = validate_arrays(x, y, meta, spec)
    if audit["X_sha256"] != spec["X_sha256"] or audit["Y_sha256"] != spec["Y_sha256"]:
        raise ValueError("数据哈希与清单不一致。")
    return x, y, meta, spec
