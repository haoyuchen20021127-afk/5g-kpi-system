"""具有统一多输出接口的 LSTM 和传统回归模型。"""
import torch
from torch import nn


class JointLSTM(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 32):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, history):
        if history.ndim != 3 or history.shape[2] != self.input_size:
            raise ValueError("输入必须为 [batch, time, input_size]。")
        sequence, _ = self.lstm(history)
        return self.head(sequence[:, -1, :])


def make_traditional(name: str, config: dict, seed: int, threads: int):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    if name == "LinearRegression":
        return LinearRegression(n_jobs=threads)
    if name == "DecisionTree":
        return DecisionTreeRegressor(random_state=seed, **config)
    if name == "RandomForest":
        return RandomForestRegressor(random_state=seed, n_jobs=threads, **config)
    if name == "XGBoost":
        return MultiOutputRegressor(XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=seed, n_jobs=threads, **config), n_jobs=1)
    if name == "LightGBM":
        return MultiOutputRegressor(LGBMRegressor(objective="regression", random_state=seed, n_jobs=threads, verbosity=-1, **config), n_jobs=1)
    raise ValueError(name)


SEARCH = {
    "LinearRegression": [{}],
    "DecisionTree": [{"max_depth": 5, "min_samples_leaf": 3}, {"max_depth": 10, "min_samples_leaf": 3}],
    "RandomForest": [{"n_estimators": 180, "max_depth": 10, "min_samples_leaf": 2, "max_features": .85}, {"n_estimators": 180, "max_depth": None, "min_samples_leaf": 2, "max_features": .85}],
    "XGBoost": [{"n_estimators": 300, "max_depth": depth, "learning_rate": .05, "min_child_weight": child, "subsample": .85, "colsample_bytree": .85, "reg_lambda": 1.0} for depth, child in [(3, 1), (5, 3)]],
    "LightGBM": [{"n_estimators": 300, "num_leaves": leaves, "learning_rate": .05, "min_child_samples": child, "subsample": .85, "subsample_freq": 1, "colsample_bytree": .85, "reg_lambda": 1.0} for leaves, child in [(15, 20), (31, 30)]],
    "LSTM": [{"hidden_size": 32, "learning_rate": .003}, {"hidden_size": 64, "learning_rate": .001}],
}

MECHANISM = {
    "LinearRegression": "native multi-target regression; separate coefficient vector per target",
    "DecisionTree": "native multi-output tree with shared partitions",
    "RandomForest": "native multi-output forest with shared partitions",
    "XGBoost": "MultiOutputRegressor; independent boosted regressor per target, shared inputs/configuration",
    "LightGBM": "MultiOutputRegressor; independent boosted regressor per target, shared inputs/configuration",
    "LSTM": "shared multivariate recurrent representation and joint linear output head",
}
