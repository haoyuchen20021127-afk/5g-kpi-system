"""在相同多变量窗口上训练八种回归算法，并逐目标、逐种子评估。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import time
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from threadpoolctl import threadpool_limits

from common import MODELS, NEURAL_MODELS, SEEDS, SPLITS, TASKS, load_data, macro_rmse, metrics, digest, save_csv, save_json
from models import MECHANISM, SEARCH, make_traditional, make_neural, load_neural_checkpoint

ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def predict_neural(model, x, device):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(x), 1024):
            result.append(model(torch.from_numpy(x[start:start + 1024]).to(device)).cpu().numpy())
    return np.concatenate(result)


def fit_neural(name, config, x, y, val_x, val_y, seed, args, device, epochs, phase):
    seed_everything(seed)
    model = make_neural(name, config, x.shape[2], y.shape[1], x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    loss_fn = nn.MSELoss()
    best_score, best_epoch, stale, best_weights = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        squared_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), labels)
            if not torch.isfinite(loss):
                raise ValueError(f"{name} 损失非有限值。")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            squared_loss += float(loss.detach()) * len(inputs)
        row = {"phase": phase, "seed": seed, "epoch": epoch, "train_MSE": squared_loss / len(x)}
        if val_x is not None:
            score = macro_rmse(val_y, predict_neural(model, val_x, device))
            row["validation_macro_RMSE"] = score
            if score < best_score - 1e-5:
                best_score, best_epoch, stale = score, epoch, 0
                best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(f"    {phase} seed={seed} epoch={epoch}: MSE={row['train_MSE']:.6f}" + (f" val={score:.6f}" if val_x is not None else ""), flush=True)
        if val_x is not None and stale >= 12:
            break
    if val_x is not None:
        model.load_state_dict(best_weights)
    else:
        best_epoch = epochs
    return model, history, best_epoch, best_score


def code_snapshot(output_root):
    files = ("common.py", "models.py", "prepare_shared_data.py", "train_comparison.py")
    hashes = {name: digest(Path(__file__).parent / name) for name in files}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    snapshot = output_root / "training_sources" / identity
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in files:
        source = Path(__file__).parent / name
        target = snapshot / name
        hashes[name] = digest(source)
        if target.exists() and digest(target) != hashes[name]:
            raise ValueError("训练源代码与已冻结快照不同；请使用新的输出目录。")
        if not target.exists():
            shutil.copy2(source, target)
    return hashes, str(snapshot.relative_to(output_root))


def verify_existing(folder, previous, spec, args):
    """只复用输入、候选、训练规则和已保存模型都可核验的完成结果。"""
    signature = previous["signature"]
    expected = {"X_sha256": spec["X_sha256"], "Y_sha256": spec["Y_sha256"],
                "seeds": list(SEEDS), "max_epochs": args.epochs, "threads": args.threads}
    if any(signature[key] != value for key, value in expected.items()):
        raise ValueError("完成任务的数据或训练参数不一致，请使用独立输出目录。")
    name = previous["model"]
    selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    if [entry["config"] for entry in selection["candidates"]] != SEARCH[name]:
        raise ValueError("已完成模型的候选配置不一致。")
    snapshot = args.output_root / previous.get("source_snapshot", "training_source")
    for filename, sha in signature["code_sha256"].items():
        if digest(snapshot / filename) != sha:
            raise ValueError("已完成模型的训练快照被改变。")
    extension = "pt" if name in NEURAL_MODELS else "joblib"
    for entry in previous["seed_runs"]:
        if not entry["reload_verified"] or digest(folder / f"seed_{entry['seed']}.{extension}") != entry["artifact_sha256"]:
            raise ValueError("已完成模型文件校验失败。")


def run_model(task, name, x, y, meta, spec, args, device, hashes):
    folder = args.output_root / task / "models" / name
    signature = {"X_sha256": spec["X_sha256"], "Y_sha256": spec["Y_sha256"], "seeds": list(SEEDS), "max_epochs": args.epochs, "code_sha256": hashes, "threads": args.threads}
    complete = folder / "run.json"
    if complete.exists():
        previous = json.loads(complete.read_text(encoding="utf-8"))
        verify_existing(folder, previous, spec, args)
        print(f"SKIP verified {task}/{name}", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    split_ids = {split: np.flatnonzero(meta["split"].eq(split)) for split in SPLITS}
    train_id, val_id, test_id = [split_ids[s] for s in SPLITS]
    fit_id = np.concatenate([train_id, val_id])
    features = x if name in NEURAL_MODELS else x.reshape(len(x), -1)
    search, histories = [], []
    selection_start = time.perf_counter()
    for index, config in enumerate(SEARCH[name]):
        print(f"TUNE {task}/{name} candidate={index}", flush=True)
        started = time.perf_counter()
        if name in NEURAL_MODELS:
            model, history, epoch, score = fit_neural(name, config, features[train_id], y[train_id], features[val_id], y[val_id], 42, args, device, args.epochs, f"candidate_{index}")
            histories.extend(history)
        else:
            model = make_traditional(name, config, 42, args.threads)
            model.fit(features[train_id], y[train_id])
            score = macro_rmse(y[val_id], model.predict(features[val_id]))
            epoch = None
        search.append({"candidate": index, "config": config, "validation_macro_RMSE": score, "best_epoch": epoch, "seconds": time.perf_counter() - started})
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    winner = min(search, key=lambda r: r["validation_macro_RMSE"])
    config = winner["config"]
    selection_seconds = time.perf_counter() - selection_start
    save_json(folder / "selection.json", {"selection_seed": 42, "criterion": "mean per-target validation RMSE", "candidates": search, "winner": winner, "selection_seconds": selection_seconds})
    predictions, rows, seed_runs = [], [], []
    for seed in SEEDS:
        print(f"FIT {task}/{name} seed={seed}", flush=True)
        started = time.perf_counter()
        if name in NEURAL_MODELS:
            model, history, _, _ = fit_neural(name, config, features[fit_id], y[fit_id], None, None, seed, args, device, winner["best_epoch"], "final_fit")
            histories.extend(history)
        else:
            model = make_traditional(name, config, seed, args.threads)
            model.fit(features[fit_id], y[fit_id])
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predicted = predict_neural(model, features[test_id], device) if name in NEURAL_MODELS else model.predict(features[test_id])
        predict_seconds = time.perf_counter() - started
        for result in metrics(y[test_id], predicted, spec["targets"]):
            rows.append({"task": task, "dataset": task.split("/")[0], "model": name, "seed": seed, **result, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds})
        if name in NEURAL_MODELS:
            checkpoint = {"model_name": name, "config": config, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "input_size": x.shape[2], "output_size": y.shape[1], "window": x.shape[1], "targets": spec["targets"], "seed": seed, "unit": "z-score", "data_X_sha256": spec["X_sha256"]}
            path = folder / f"seed_{seed}.pt"
            torch.save(checkpoint, path)
            reloaded = torch.load(path, map_location="cpu", weights_only=True)
            restored = load_neural_checkpoint(reloaded).to(device)
            check = predict_neural(restored, features[test_id], device)
            parameters = sum(p.numel() for p in model.parameters())
            del restored
        else:
            path = folder / f"seed_{seed}.joblib"
            joblib.dump(model, path, compress=3)
            restored = joblib.load(path)
            check = restored.predict(features[test_id])
            parameters = None
            del restored
        np.testing.assert_allclose(check, predicted, rtol=1e-6, atol=1e-6)
        predictions.append(predicted)
        seed_runs.append({"seed": seed, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds, "reload_verified": True, "artifact_sha256": digest(path), "trainable_parameters": parameters})
        print(f"  TEST {task}/{name} seed={seed}: macro RMSE={macro_rmse(y[test_id], predicted):.6f}", flush=True)
        del model
        gc.collect()
    for seed, predicted in zip(SEEDS, predictions):
        frame = meta.iloc[test_id].reset_index(drop=True).copy()
        for j, target in enumerate(spec["targets"]):
            frame[f"actual__{target}"] = y[test_id, j]
            frame[f"predicted__{target}"] = predicted[:, j]
        save_csv(folder / f"test_predictions_seed_{seed}.csv", frame)
    save_csv(folder / "metrics.csv", pd.DataFrame(rows))
    save_json(folder / "selection.json", {"selection_seed": 42, "criterion": "mean per-target validation RMSE", "candidates": search, "winner": winner, "selection_seconds": selection_seconds})
    if histories:
        save_csv(folder / "training_history.csv", pd.DataFrame(histories))
    save_json(complete, {"task": task, "model": name, "signature": signature, "source_snapshot": args.source_snapshot, "mechanism": MECHANISM[name], "selected_config": config, "selected_epochs": winner["best_epoch"], "final_fit_rows": len(fit_id), "test_rows": len(test_id), "selection_seconds": selection_seconds, "seed_runs": seed_runs, "device": str(device) if name in NEURAL_MODELS else "cpu", "gpu": torch.cuda.get_device_name(0) if name in NEURAL_MODELS and device.type == "cuda" else None, "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "torch", "scikit-learn", "xgboost", "lightgbm", "joblib"]}, "common_settings": {"optimizer": "Adam", "loss": "mean squared error across samples and outputs", "batch_size": 256, "weight_decay": 1e-5, "gradient_clip": 1.0, "patience": 12, "min_delta": 1e-5, "layers": 1, "dropout": 0, "transformer_position_encoding": "fixed sinusoidal", "transformer_attention_scope": "all observed history positions only", "transformer_norm_first": True, "transformer_activation": "gelu", "attention_backend": "math", "mha_fastpath": False}})


def persist_baseline(folder, task, x, y, meta, spec):
    ids = np.flatnonzero(meta["split"].eq("test"))
    predicted = x[ids, -1, :]
    frame = meta.iloc[ids].reset_index(drop=True).copy()
    for j, target in enumerate(spec["targets"]):
        frame[f"actual__{target}"] = y[ids, j]
        frame[f"predicted__{target}"] = predicted[:, j]
    save_csv(folder / "test_predictions.csv", frame)
    rows = [{"task": task, "dataset": task.split("/")[0], "model": "Persistence", "seed": seed, **row, "fit_seconds": 0., "predict_seconds": 0.} for seed in SEEDS for row in metrics(y[ids], predicted, spec["targets"])]
    save_csv(folder / "metrics.csv", pd.DataFrame(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--models", nargs="+", default=list(SEARCH), choices=list(SEARCH))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("epochs 和 threads 必须为正数。")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。")
    hashes, args.source_snapshot = code_snapshot(args.output_root)
    with threadpool_limits(limits=args.threads):
        for task in args.tasks:
            x, y, meta, spec = load_data(args.output_root / task / "data")
            persist_baseline(args.output_root / task / "models" / "Persistence", task, x, y, meta, spec)
            for name in args.models:
                run_model(task, name, x, y, meta, spec, args, device, hashes)


if __name__ == "__main__":
    main()
