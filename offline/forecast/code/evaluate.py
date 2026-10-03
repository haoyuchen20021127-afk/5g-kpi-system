"""Run a frozen forecasting model on the complete packaged test split."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from common import MODELS, NEURAL_MODELS, TASKS, load_data, metrics, save_csv, save_json
from models import load_neural_checkpoint

RESULTS = Path(__file__).resolve().parents[1] / "results"


def predict_saved(folder, name, seed, x, spec):
    if name == "Persistence":
        return x[:, -1, :].copy()
    model_dir = folder / "models" / name
    run = json.loads((model_dir / "run.json").read_text(encoding="utf-8"))
    if run["task"] != spec["task"] or run["signature"]["X_sha256"] != spec["X_sha256"]:
        raise ValueError("Model and prepared data do not match.")
    if name in NEURAL_MODELS:
        state = torch.load(model_dir / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
        if state["targets"] != spec["targets"] or state["window"] != spec["window"]:
            raise ValueError("Checkpoint target order or history window does not match.")
        model = load_neural_checkpoint(state).eval()
        with torch.inference_mode():
            values = [model(torch.from_numpy(x[i:i + 1024])).numpy()
                      for i in range(0, len(x), 1024)]
        predicted = np.concatenate(values)
    else:
        # Only load trusted artifacts shipped with this project.
        model = joblib.load(model_dir / f"seed_{seed}.joblib")
        predicted = model.predict(x.reshape(len(x), -1))
    if predicted.shape != (len(x), len(spec["targets"])) or not np.isfinite(predicted).all():
        raise ValueError("Invalid prediction shape or values.")
    return predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--model", choices=list(MODELS), default="LSTM")
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], default=42)
    parser.add_argument("--output", type=Path, required=True, help="A NEW result directory.")
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(RESULTS.resolve()):
        parser.error("Output must be a new directory outside packaged results.")
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    folder = RESULTS / args.task
    x, y, index, spec = load_data(folder / "data")
    selected = index.split.eq("test").to_numpy()
    predicted = predict_saved(folder, args.model, args.seed, x[selected], spec)
    frame = index.loc[selected].reset_index(drop=True).copy()
    for i, target in enumerate(spec["targets"]):
        frame[f"actual__{target}"] = y[selected, i]
        frame[f"predicted__{target}"] = predicted[:, i]
    args.output.mkdir(parents=True, exist_ok=False)
    save_csv(args.output / "predictions.csv", frame)
    save_json(args.output / "metrics.json", metrics(y[selected], predicted, spec["targets"]))
    save_json(args.output / "run_info.json", dict(task=args.task, model=args.model,
              seed=args.seed, rows=len(frame), split="test", unit=spec["unit"],
              retrained=False, data_X_sha256=spec["X_sha256"]))
    print(f"Predicted {len(frame)} held-out windows -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
