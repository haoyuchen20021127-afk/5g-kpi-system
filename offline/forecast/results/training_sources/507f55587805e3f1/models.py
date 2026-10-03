"""具有统一多输出接口的循环网络、Transformer 和传统回归模型。"""
import math
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


class JointGRU(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 32):
        super().__init__()
        self.input_size = input_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, history):
        if history.ndim != 3 or history.shape[2] != self.input_size:
            raise ValueError("输入必须为 [batch, time, input_size]。")
        sequence, _ = self.gru(history)
        return self.head(sequence[:, -1, :])


class JointTransformer(nn.Module):
    """只编码已观测历史；位置编码来自窗口顺序，不引入未来观测。"""
    def __init__(self, input_size: int, output_size: int, window: int, d_model: int = 32,
                 nhead: int = 4, dim_feedforward: int = 128, num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        if d_model % nhead or d_model % 2 or window < 1:
            raise ValueError("隐藏维度必须为偶数且能被注意力头数整除。")
        self.input_size, self.window = input_size, window
        self.input_projection = nn.Linear(input_size, d_model)
        position = torch.arange(window, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.) / d_model))
        encoding = torch.zeros(window, d_model)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        encoding[:, 1::2] = torch.cos(position * frequencies)
        self.register_buffer("position_encoding", encoding.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.head = nn.Linear(d_model, output_size)

    def forward(self, history):
        if history.ndim != 3 or history.shape[1:] != (self.window, self.input_size):
            raise ValueError("Transformer 输入必须匹配 [batch, window, input_size]。")
        tokens = self.input_projection(history) + self.position_encoding
        # The whole encoder input precedes the single forecast target; no decoder or target token is used.
        encoded = self.encoder(tokens)
        return self.head(encoded[:, -1, :])


def make_neural(name: str, config: dict, input_size: int, output_size: int, window: int):
    if name == "LSTM":
        return JointLSTM(input_size, output_size, config["hidden_size"])
    if name == "GRU":
        return JointGRU(input_size, output_size, config["hidden_size"])
    if name == "Transformer":
        options = {key: value for key, value in config.items() if key != "learning_rate"}
        return JointTransformer(input_size, output_size, window, **options)
    raise ValueError(name)


def load_neural_checkpoint(checkpoint):
    name = checkpoint.get("model_name", "LSTM")
    config = checkpoint.get("config") or {"hidden_size": checkpoint["hidden_size"]}
    model = make_neural(name, config, checkpoint["input_size"], checkpoint["output_size"], checkpoint["window"])
    model.load_state_dict(checkpoint["state_dict"])
    return model


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
    "GRU": [{"hidden_size": 32, "learning_rate": .003}, {"hidden_size": 64, "learning_rate": .001}],
    "Transformer": [{"d_model": width, "nhead": 4, "dim_feedforward": width * 4,
                     "num_layers": 1, "dropout": 0.0, "learning_rate": lr}
                    for width, lr in [(32, .003), (64, .001)]],
}

MECHANISM = {
    "LinearRegression": "native multi-target regression; separate coefficient vector per target",
    "DecisionTree": "native multi-output tree with shared partitions",
    "RandomForest": "native multi-output forest with shared partitions",
    "XGBoost": "MultiOutputRegressor; independent boosted regressor per target, shared inputs/configuration",
    "LightGBM": "MultiOutputRegressor; independent boosted regressor per target, shared inputs/configuration",
    "LSTM": "shared multivariate recurrent representation and joint linear output head",
    "GRU": "shared multivariate GRU representation and joint linear output head",
    "Transformer": "history-only Transformer encoder with sinusoidal positions and joint linear output head",
}
