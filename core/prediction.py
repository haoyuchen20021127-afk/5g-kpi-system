from pathlib import Path
import hashlib
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st



from core.config import PREDICTION_ROOT as TASK_DIR, KPIS as TARGETS, LABELS
from core.models import load_neural_checkpoint


def file_signature():
    """文件变化后更新缓存，避免继续显示旧结果。"""
    paths = [
        TASK_DIR / "data" / "manifest.json",
        TASK_DIR / "data" / "samples.npz",
        TASK_DIR / "data" / "sample_index.csv",
        TASK_DIR / "models" / "LSTM" / "seed_42.pt",
        TASK_DIR / "models" / "LSTM" / "run.json",
        Path(__file__).with_name("models.py"),
    ]

    return tuple(
        (str(p), p.stat().st_size, p.stat().st_mtime_ns)
        for p in paths
    )


def array_hash(values):
    """核对数据是否与第五周保存的数据一致。"""
    contiguous = np.ascontiguousarray(values)

    digest = hashlib.sha256(
        str((contiguous.shape, contiguous.dtype.str)).encode()
    )
    digest.update(contiguous.tobytes())

    return digest.hexdigest()


def read_test_data():
    """只读取既定测试集，不重新划分或处理数据。"""
    spec = json.loads(
        (TASK_DIR / "data" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    with np.load(
        TASK_DIR / "data" / "samples.npz",
        allow_pickle=False,
    ) as data:
        x = data["X"].copy()
        y = data["Y"].copy()

    index = pd.read_csv(
        TASK_DIR / "data" / "sample_index.csv"
    )

    if (
        spec["task"] != "gotsf_ds/joint_kpi"
        or spec["targets"] != TARGETS
    ):
        raise ValueError("任务或 KPI 顺序不匹配。")

    if spec["window"] != 24 or spec["unit"] != "z-score":
        raise ValueError("需要 24 小时的 Z-score 历史窗口。")

    if (
        x.shape != (len(index), 24, 4)
        or y.shape != (len(index), 4)
    ):
        raise ValueError("输入、真实值与样本索引的形状不匹配。")

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("数据包含缺失值或无穷值。")

    if (
        array_hash(x) != spec["X_sha256"]
        or array_hash(y) != spec["Y_sha256"]
    ):
        raise ValueError("数据与第五周保存的清单不一致。")

    if index.duplicated(
        ["series_id", "target_sequence"]
    ).any():
        raise ValueError("样本索引存在重复键。")

    counts = index["split"].value_counts().to_dict()

    if counts != spec["split_rows"]:
        raise ValueError("数据划分数量与第五周清单不一致。")

    # 只选择标记为 test 的样本。
    ids = np.flatnonzero(
        index["split"].eq("test").to_numpy()
    )

    if len(ids) < 2:
        raise ValueError("测试样本不足。")

    order = (
        index.iloc[ids]
        .sort_values(["series_id", "target_sequence"])
        .index.to_numpy()
    )

    meta = index.iloc[order].reset_index(drop=True)
    x_test = x[order]
    y_test = y[order]

    # 检查相邻窗口的时间和数值是否对齐。
    for i in range(1, len(meta)):
        same_series = (
            meta.loc[i, "series_id"]
            == meta.loc[i - 1, "series_id"]
        )
        adjacent = (
            meta.loc[i, "target_sequence"]
            == meta.loc[i - 1, "target_sequence"] + 1
        )

        if same_series and adjacent:
            if not np.array_equal(
                x_test[i - 1, 1:],
                x_test[i, :-1],
            ):
                raise ValueError("相邻历史窗口不连续。")

            if not np.array_equal(
                y_test[i - 1],
                x_test[i, -1],
            ):
                raise ValueError("真实值与后一个历史窗口不匹配。")

    # 只用于提示输入是否超出训练期单项取值范围，不重新拟合任何预处理器。
    train = x[index["split"].eq("train").to_numpy()]
    spec["training_input_min"] = train.min(axis=(0, 1)).tolist()
    spec["training_input_max"] = train.max(axis=(0, 1)).tolist()
    return x_test, y_test, meta, spec


@st.cache_data(max_entries=2, show_spinner=False)
def load_test_data(signature, schema_version="interactive-v1"):
    # 缓存不会自动追踪 read_test_data 的内部改动，用版本参数隔离旧页面缓存。
    return read_test_data()



@st.cache_resource(max_entries=2, show_spinner=False)
def load_model(signature, data_hash):
    import torch

    checkpoint = torch.load(
        TASK_DIR / "models" / "LSTM" / "seed_42.pt",
        map_location="cpu",
        weights_only=True,
    )

    run = json.loads(
        (TASK_DIR / "models" / "LSTM" / "run.json").read_text(
            encoding="utf-8"
        )
    )

    if (
        run["task"] != "gotsf_ds/joint_kpi"
        or run["signature"]["X_sha256"] != data_hash
    ):
        raise ValueError("模型训练清单与输入数据不匹配。")

    if (
        checkpoint["data_X_sha256"] != data_hash
        or checkpoint["targets"] != TARGETS
    ):
        raise ValueError("模型权重与输入顺序不匹配。")

    dimensions = (
        checkpoint["window"],
        checkpoint["input_size"],
        checkpoint["output_size"],
    )

    if dimensions != (24, 4, 4):
        raise ValueError("模型要求的窗口或维度不匹配。")

    if (
        checkpoint["seed"] != 42
        or checkpoint["unit"] != "z-score"
    ):
        raise ValueError("模型种子或数据尺度不匹配。")

    model = load_neural_checkpoint(checkpoint)
    model.eval()

    return model


def predict_history(model, history):
    """模型只接收历史输入 X，不接收待预测时刻的真实值 Y。"""
    import torch

    history = np.asarray(history, dtype=np.float32)

    if (
        history.ndim != 3
        or history.shape[1:] != (24, 4)
        or len(history) == 0
    ):
        raise ValueError("历史输入必须为 [样本数, 24, 4]。")

    if not np.isfinite(history).all():
        raise ValueError("历史输入含无效数值。")

    with torch.inference_mode():
        predicted = model(
            torch.from_numpy(
                np.ascontiguousarray(history)
            )
        ).cpu().numpy()

    if (
        predicted.shape != (len(history), 4)
        or not np.isfinite(predicted).all()
    ):
        raise ValueError("模型输出形状或数值无效。")

    return predicted



def evaluate(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if (
        actual.shape != predicted.shape
        or actual.ndim != 2
        or actual.shape[1] != 4
    ):
        raise ValueError("评分数据形状不匹配。")

    if (
        len(actual) < 1
        or not np.isfinite(actual).all()
        or not np.isfinite(predicted).all()
    ):
        raise ValueError("评分数据不足或包含无效值。")

    error = predicted - actual

    mae = np.abs(error).mean(axis=0)
    rmse = np.sqrt((error ** 2).mean(axis=0))

    sse = (error ** 2).sum(axis=0)
    sst = (
        (actual - actual.mean(axis=0)) ** 2
    ).sum(axis=0)

    r2 = np.full(4, np.nan)
    valid = sst > 0
    r2[valid] = 1 - sse[valid] / sst[valid]

    return pd.DataFrame({
        "KPI": TARGETS,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "测试样本数": len(actual),
    })


def history_table(history, target_hour):
    frame = pd.DataFrame(history, columns=TARGETS)

    frame.insert(
        0,
        "历史小时序号",
        np.arange(
            int(target_hour) - 24,
            int(target_hour),
        ),
    )

    return frame


def result_table(meta, actual, predicted):
    result = meta[
        ["series_id", "target_sequence", "target_time"]
    ].copy()

    for j, name in enumerate(TARGETS):
        result[f"actual__{name}"] = actual[:, j]
        result[f"predicted__{name}"] = predicted[:, j]
        result[f"error__{name}"] = (
            predicted[:, j] - actual[:, j]
        )
        result[f"abs_error__{name}"] = np.abs(
            predicted[:, j] - actual[:, j]
        )

    return result


def comparison_chart(meta, actual, predicted, kpi):
    j = TARGETS.index(kpi)

    frame = meta[
        ["series_id", "target_sequence"]
    ].copy()

    # 不把不同序列或不连续时间段连成一条线。
    new_segment = (
        frame["series_id"].ne(
            frame["series_id"].shift()
        )
        | frame["target_sequence"].diff().ne(1)
    )

    frame["连续片段"] = (
        new_segment.cumsum().astype(str)
    )

    frame["真实值"] = actual[:, j]
    frame["预测值"] = predicted[:, j]

    long = frame.melt(
        id_vars=[
            "series_id",
            "target_sequence",
            "连续片段",
        ],
        value_vars=["真实值", "预测值"],
        var_name="曲线",
        value_name="数值",
    )

    return (
        alt.Chart(long)
        .mark_line(strokeWidth=2, point=True)
        .encode(
            x=alt.X(
                "target_sequence:Q",
                title="目标小时序号（序列内相对位置）",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "数值:Q",
                title=f"{kpi}（Z-score）",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "曲线:N",
                scale=alt.Scale(
                    domain=["真实值", "预测值"],
                    range=["#253247", "#2463EB"],
                ),
                title=None,
            ),
            detail="连续片段:N",
            tooltip=[
                alt.Tooltip(
                    "series_id:N",
                    title="序列",
                ),
                alt.Tooltip(
                    "target_sequence:Q",
                    title="目标小时",
                ),
                alt.Tooltip("曲线:N"),
                alt.Tooltip(
                    "数值:Q",
                    format=".5f",
                ),
            ],
        )
        .properties(height=330)
    )



APP_VERSION = "interactive-v1.1"
HOUR = "历史小时序号"
CUSTOM = "自定义输入预测"
HISTORICAL = "历史测试查询"
FILTERS = ["全部记录", "预测值高于阈值", "绝对误差高于阈值", "误差最大的前 N 条"]


def validate_editor(frame, target_hour):
    """在服务端核对 24×4、固定时间轴、有限数值和 float32 可表示性。"""
    if len(frame) != 24 or list(frame.columns) != [HOUR, *TARGETS]:
        raise ValueError("必须保留 24 行历史数据和全部 4 项 KPI，不能增加、删除或重排列。")
    hours = pd.to_numeric(frame[HOUR], errors="coerce").to_numpy(float)
    if not np.array_equal(hours, np.arange(target_hour - 24, target_hour)):
        raise ValueError("历史时间必须是预测时刻前连续 24 小时，不能修改或重排。")
    values = frame[TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("请填写所有 KPI 数值，不能留空或输入非数字、无穷值。")
    if np.any(np.abs(values) > np.finfo(np.float32).max):
        raise ValueError("输入数值过大，超出模型可处理范围。")
    return values.astype(np.float32)


def shifted_frame(frame, target_hour, kpi, last_n, offset):
    if kpi not in TARGETS or not isinstance(last_n, int) or not 1 <= last_n <= 24:
        raise ValueError("批量调整的 KPI 或小时数量无效。")
    if not np.isfinite(offset):
        raise ValueError("调整量必须是有限数值。")
    validate_editor(frame, target_hour)
    result = frame.copy(deep=True)
    result.loc[result.index[-last_n:], kpi] += offset
    validate_editor(result, target_hour)
    return result


def filter_indices(meta, actual, predicted, series, hours, rule, kpi, threshold=0., top_n=10):
    """先预测完整历史窗口，再筛选结果；不通过删行拼接模型输入。"""
    if rule not in FILTERS or kpi not in TARGETS or hours[0] > hours[1]:
        raise ValueError("查询条件无效。")
    if not np.isfinite(threshold) or top_n < 1:
        raise ValueError("阈值或条数无效。")
    ids = np.flatnonzero((meta["series_id"].eq(series)
                         & meta["target_sequence"].between(*hours)).to_numpy())
    j = TARGETS.index(kpi)
    if rule == FILTERS[1]:
        ids = ids[predicted[ids, j] > threshold]
    elif rule == FILTERS[2]:
        ids = ids[np.abs(predicted[ids, j] - actual[ids, j]) > threshold]
    elif rule == FILTERS[3]:
        ranking = np.argsort(-np.abs(predicted[ids, j] - actual[ids, j]), kind="stable")
        ids = ids[ranking[:top_n]]
    # 显示按时间排序，曲线中的非连续小时仍分段。
    return np.sort(ids)


def scenario_table(original_pred, current_pred, row, input_hash):
    return pd.DataFrame({
        "KPI": TARGETS,
        "原始输入的预测": original_pred,
        "当前输入的预测": current_pred,
        "变化量": current_pred - original_pred,
        "样本序列": row["series_id"],
        "目标小时": int(row["target_sequence"]),
        "模型": "LSTM seed 42",
        "单位": "Z-score",
        "当前模型输入校验值": input_hash,
    })


def scenario_chart(original, current, predictions, target_hour, kpi):
    j = TARGETS.index(kpi)
    long = pd.DataFrame({
        "小时": np.tile(np.arange(target_hour - 24, target_hour), 2),
        "数值": np.concatenate([original[:, j], current[:, j]]),
        "输入版本": np.repeat(["原始输入", "当前输入"], 24),
    })
    forecast = pd.DataFrame({
        "小时": [target_hour, target_hour], "数值": predictions[:, j],
        "输入版本": ["原始输入", "当前输入"],
    })
    color = alt.Color("输入版本:N", scale=alt.Scale(
        domain=["原始输入", "当前输入"], range=["#6b7280", "#2563eb"]), title=None)
    history = alt.Chart(long).mark_line().encode(
        x=alt.X("小时:Q", title="序列内相对小时", scale=alt.Scale(zero=False)),
        y=alt.Y("数值:Q", title=f"{kpi}（Z-score）", scale=alt.Scale(zero=False)),
        color=color, strokeDash="输入版本:N",
        tooltip=["小时:Q", "输入版本:N", alt.Tooltip("数值:Q", format=".5f")],
    )
    points = alt.Chart(forecast).mark_point(size=130, filled=False).encode(
        x="小时:Q", y="数值:Q", color=color, shape="输入版本:N",
        tooltip=["小时:Q", "输入版本:N", alt.Tooltip("数值:Q", title="下一小时预测", format=".5f")],
    )
    return (history + points).properties(height=250)


def replace_draft(frame, clear_result=True):
    """更换编辑器基底时换 key，避免旧的单元格修改覆盖新样本。"""
    st.session_state["ip_seed"] = frame.copy(deep=True)
    st.session_state["ip_draft"] = frame.copy(deep=True)
    st.session_state["ip_epoch"] += 1
    if clear_result:
        st.session_state.pop("ip_scenario", None)
