"""使用保存的多输出模型预测按清单顺序排列的历史窗口。"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from models import load_neural_checkpoint
from common import save_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("输出文件已存在，请指定新文件名。")
    spec = json.loads(args.manifest.read_text(encoding="utf-8"))
    run = json.loads((args.model.parent / "run.json").read_text(encoding="utf-8"))
    if run["task"] != spec["task"] or run["signature"]["X_sha256"] != spec["X_sha256"]:
        raise ValueError("模型与数据任务或输入顺序不匹配。")
    frame = pd.read_csv(args.input)
    flat = frame[spec["flat_feature_order"]].to_numpy(dtype=np.float32)
    if len(flat) == 0 or not np.isfinite(flat).all():
        raise ValueError("预测输入为空或包含缺失值。")
    if args.model.suffix == ".pt":
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
        if checkpoint["targets"] != spec["targets"] or checkpoint["window"] != spec["window"]:
            raise ValueError("模型与输入清单不匹配。")
        model = load_neural_checkpoint(checkpoint)
        model.eval()
        arrays = flat.reshape(len(flat), spec["window"], len(spec["targets"]))
        pieces = []
        with torch.inference_mode():
            for start in range(0, len(arrays), 1024):
                pieces.append(model(torch.from_numpy(arrays[start:start + 1024])).numpy())
        predicted = np.concatenate(pieces)
    else:
        model = joblib.load(args.model)
        predicted = model.predict(flat)
    if predicted.shape != (len(frame), len(spec["targets"])) or not np.isfinite(predicted).all():
        raise ValueError("模型输出形状或数值无效。")
    result = frame[[c for c in ("series_id", "target_sequence", "target_time") if c in frame]].copy()
    for j, target in enumerate(spec["targets"]):
        result[f"predicted__{target}"] = predicted[:, j]
    save_csv(args.output, result)


if __name__ == "__main__":
    main()
