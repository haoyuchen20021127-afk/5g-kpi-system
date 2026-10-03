"""从既有 Z-score 数据组织多输入多输出窗口，不重新清洗或拟合标准化器。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import TASKS, digest, make_windows, save_data, save_json

ROOT = Path(__file__).resolve().parents[1]


def join_existing(week4_root: Path, task: str):
    spec = TASKS[task]
    dataset = task.split("/")[0]
    xparts, yparts, sources, base = [], [], [], None
    for target in spec["targets"]:
        path = week4_root / dataset / "data" / target / "supervised_samples.csv.gz"
        frame = pd.read_csv(path)
        if dataset == "gotsf_ds":
            frame["series_id"] = frame["series_id"].str.removeprefix(target + "_")
            frame["target_time"] = frame["series_id"] + ":hour_" + frame["target_sequence"].astype(str)
        frame = frame.sort_values(["series_id", "target_sequence"], kind="stable").reset_index(drop=True)
        index = frame[["series_id", "target_sequence", "split", "target_time"]]
        if base is not None and not index.equals(base):
            raise ValueError(f"{task}/{target} 与其他输出的样本键或划分不同，禁止按行硬拼。")
        base = index
        xparts.append(frame[[f"lag_{lag}" for lag in range(spec["window"], 0, -1)]].to_numpy(dtype=np.float32))
        yparts.append(frame["target"].to_numpy(dtype=np.float32))
        sources.append({"path": str(path), "sha256": digest(path), "rows": len(frame)})
    base = base.copy()
    base["time_basis"] = "relative_hour" if dataset == "gotsf_ds" else "UTC"
    return np.stack(xparts, axis=-1), np.stack(yparts, axis=-1), base, {"sources": sources, "construction": "exact sample-key join of existing target/lag columns", "split_policy": "preserved existing source splits", "rows_excluded_during_join": 0}


def aggregate_file(path: Path, spec: dict, chunksize: int):
    available = set(pd.read_csv(path, nrows=0).columns)
    required = set(spec["raw"] + ["series_id"] + spec["groups"])
    if not required.issubset(available):
        raise ValueError(f"{path.name} 缺少共同观测字段：{required - available}")
    time_columns = [name for name in ["event_time_utc", "relative_time_index"] if name in available]
    selected = sorted(required | set(time_columns))
    partials, n_rows, rejected = [], 0, 0
    for number, chunk in enumerate(pd.read_csv(path, usecols=selected, chunksize=chunksize, low_memory=False)):
        n_rows += len(chunk)
        absolute = pd.to_datetime(chunk.get("event_time_utc", pd.Series(index=chunk.index, dtype="object")), errors="coerce", utc=True)
        seconds = pd.Series(np.nan, index=chunk.index, dtype=float)
        valid_abs = absolute.notna()
        seconds.loc[valid_abs] = absolute.loc[valid_abs].astype("int64") / 1e9
        relative = pd.to_numeric(chunk.get("relative_time_index", pd.Series(index=chunk.index, dtype=float)), errors="coerce")
        seconds = seconds.fillna(relative)
        chunk["time_basis"] = np.where(valid_abs, "UTC", "relative_second")
        chunk["time_index"] = np.floor(seconds / 300)
        base = path.stem.removesuffix(".csv") + "|" + chunk["time_basis"] + "|" + chunk["series_id"].astype("string")
        for column in spec["groups"]:
            base = base + "|" + column + "=" + chunk[column].astype("string").fillna("<missing>")
        chunk["series_base"] = base
        for column in spec["raw"]:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        valid = np.isfinite(chunk[spec["raw"] + ["time_index"]].to_numpy(dtype=float)).all(axis=1) & chunk["series_base"].notna().to_numpy()
        rejected += int((~valid).sum())
        chunk = chunk.loc[valid].copy()
        chunk["time_index"] = chunk["time_index"].astype(np.int64)
        groups = chunk.groupby(["series_base", "time_basis", "time_index"], observed=True)
        subtotal = groups[spec["raw"]].sum()
        subtotal["_count"] = groups.size()
        partials.append(subtotal.reset_index())
        if number % 8 == 0:
            print(f"  {path.name}: {n_rows:,} rows", flush=True)
    grouped = pd.concat(partials, ignore_index=True).groupby(["series_base", "time_basis", "time_index"], observed=True, as_index=False)[spec["raw"] + ["_count"]].sum()
    for column in spec["raw"]:
        grouped[column] = grouped[column] / grouped["_count"]
    details = {"path": str(path), "sha256": digest(path), "rows_read": n_rows, "rows_excluded_invalid_common_observation": rejected, "bins": len(grouped)}
    return grouped, details


def prepare_donext(processed_root: Path, task: str, chunksize: int):
    spec = TASKS[task]
    parts, sources = [], []
    for stem in spec["stems"]:
        frame, source = aggregate_file(processed_root / "donext" / "zscore" / f"{stem}.csv.gz", spec, chunksize)
        parts.append(frame)
        sources.append(source)
    frame = pd.concat(parts, ignore_index=True).sort_values(["series_base", "time_index"], kind="stable").reset_index(drop=True)
    gap = frame.groupby("series_base", observed=True)["time_index"].diff()
    segment = (gap.ne(1) | gap.isna()).groupby(frame["series_base"], observed=True).cumsum().astype(int)
    frame["series_id"] = frame["series_base"] + "|segment_" + segment.astype(str)
    frame["target_time"] = frame["time_basis"] + ":bin_" + frame["time_index"].astype(str)
    utc_mask = frame["time_basis"].eq("UTC")
    frame.loc[utc_mask, "target_time"] = pd.to_datetime(frame.loc[utc_mask, "time_index"] * 300, unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame = frame.rename(columns=dict(zip(spec["raw"], spec["targets"])))
    x, y, meta, audit = make_windows(frame, spec["targets"], spec["window"], 30)
    return x, y, meta, {"sources": sources, "construction": "joint observations from each existing Z-score source table; 5-minute means; no added filling", "split_policy": "within each continuous source/entity/business segment, chronological 70/15/15 after warmup", "organization_audit": audit, "entity_limit": "source-provided series identifiers; device strings may identify models rather than unique physical devices"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=Path("E:/5G_Project/week3/data/processed_data"))
    parser.add_argument("--week4-root", type=Path, default=Path("E:/5G_Project/week4/final_models_zscore"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--chunksize", type=int, default=250000)
    args = parser.parse_args()
    summaries = []
    for task in args.tasks:
        folder = args.output_root / task / "data"
        if (folder / "manifest.json").exists():
            from common import load_data
            *_, manifest = load_data(folder)
            print(f"Verified existing data: {task}", flush=True)
        else:
            print(f"PREPARE {task}", flush=True)
            payload = prepare_donext(args.processed_root, task, args.chunksize) if task.startswith("donext/") else join_existing(args.week4_root, task)
            manifest = save_data(folder, task, *payload)
            print(f"  READY {manifest['input_shape']} -> {manifest['output_shape']}; {manifest['split_rows']}", flush=True)
        summaries.append({key: manifest[key] for key in ("task", "targets", "window", "resolution", "split_rows", "n_samples", "sequence_count")})
    save_json(args.output_root / "prepared_tasks.json", summaries)


if __name__ == "__main__":
    main()
