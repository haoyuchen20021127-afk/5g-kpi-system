from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


from core.config import ANOMALY_ROOT as DATA_ROOT, KPIS, LABELS
from core import detectors

METHODS = {
    "isolation_forest": "孤立森林",
    "three_sigma": "3σ",
    "iqr": "IQR（箱线图规则）",
    "one_class_svm": "One-Class SVM",
}
INPUT = "seasonal_residual__model_input"
IMPUTED = "seasonal_residual__input_imputed"
VERSION = "anomaly-v2-custom"
STATES = {1: "疑似异常", 0: "未判异常", -1: "无法判定"}


def check_options(kpi, method=None):
    if kpi not in KPIS or (method is not None and method not in METHODS):
        raise ValueError("未知的 KPI 或检测方法。")


def signature(kpi):
    check_options(kpi)
    folder = DATA_ROOT / kpi
    paths = [folder / name for name in (
        "prepared_observations.csv.gz", "model_inputs.csv.gz", "kpi_summary.csv"
    )]
    paths += sorted((folder / "models").glob("*.joblib"))
    paths += [Path(detectors.__file__)]
    return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)


@st.cache_data(max_entries=4, show_spinner=False)
def load_test_data(kpi, stamp, version=VERSION):
    check_options(kpi)
    folder = DATA_ROOT / kpi
    columns = ["row_id", "source", "cell", "time", "split", "raw_value",
               "seasonal_expected", "training_cell_scale", "seasonal_residual",
               "observed_beams"]
    observed = pd.read_csv(folder / "prepared_observations.csv.gz", usecols=columns)
    inputs = pd.read_csv(folder / "model_inputs.csv.gz",
                         usecols=["row_id", "split", INPUT, IMPUTED])
    observed = observed.loc[observed["split"].eq("test")].copy()
    inputs = inputs.loc[inputs["split"].eq("test")].drop(columns="split")
    data = observed.merge(inputs, on="row_id", how="left", validate="one_to_one")
    if data.empty or not np.isfinite(data[INPUT].to_numpy(float)).all():
        raise ValueError("测试输入为空或存在无效值，请检查第六周的数据文件。")
    if data[IMPUTED].isna().any():
        raise ValueError("部分观测记录没有对应的模型输入，不能继续检测。")
    filled = data[IMPUTED].astype(str).str.lower()
    if not filled.isin(["true", "false"]).all():
        raise ValueError("输入填补标记无效。")
    data[IMPUTED] = filled.eq("true")
    if not data["source"].str.startswith(kpi + "_").all():
        raise ValueError("来源与 KPI 不一致。")
    data["window"] = data["source"].str.removeprefix(kpi + "_")
    if data.duplicated(["window", "cell", "time"]).any():
        raise ValueError("同小区、时段、小时存在重复记录。")
    actual = (data["raw_value"] - data["seasonal_expected"]) / data["training_cell_scale"]
    measured = ~data[IMPUTED]
    if not np.allclose(actual[measured], data.loc[measured, INPUT], rtol=1e-9, atol=1e-9):
        raise ValueError("原始观测与保存的检测输入不匹配，请勿混用不同版本。")
    data["cell"] = data["cell"].astype(str)
    return data.sort_values(["source", "cell", "time"]).reset_index(drop=True)


@st.cache_resource(max_entries=16, show_spinner=False)
def load_detector(kpi, method, stamp):
    check_options(kpi, method)
    # 只读取本项目自己训练并保存的模型，不加载用户上传的 joblib 文件。
    folder = DATA_ROOT / kpi
    summary = pd.read_csv(folder / "kpi_summary.csv")
    rows = summary.loc[summary["method"].eq(method)
                       & summary["feature"].eq("seasonal_residual")]
    if len(rows) != 1 or rows.iloc[0]["status"] != "ok":
        raise ValueError("未找到唯一、可用的检测模型。")
    model_path = (folder / rows.iloc[0]["model_file"]).resolve()
    if not model_path.is_relative_to((folder / "models").resolve()):
        raise ValueError("模型路径不在本任务的 models 文件夹内。")
    detector = joblib.load(model_path)
    if detector.method != method or detector.status != "ok":
        raise ValueError("模型的方法或状态与清单不一致。")
    if not np.isclose(detector.threshold, float(rows.iloc[0]["threshold"]), rtol=1e-10, atol=1e-12):
        raise ValueError("模型阈值与结果清单不一致。")
    return detector


def detect(data, detector):
    result = data.copy()
    result["score"] = detector.score(result[INPUT].to_numpy(float))
    flags, guard = detectors.compare_score(
        result["score"], detector.threshold, detector.comparison_method()
    )
    result["flag"] = flags
    result["limit"] = detector.threshold + guard
    result["判定"] = result["flag"].map({1: "疑似异常", 0: "未判异常", -1: "无法判定"})
    return result


def observed_chart(result):
    result = result.sort_values("time").copy()
    result["segment"] = result["time"].diff().ne(1).cumsum()
    long = result.melt(id_vars=["time", "segment"], value_vars=["raw_value", "seasonal_expected"],
                       var_name="曲线", value_name="数值")
    long["曲线"] = long["曲线"].map({"raw_value": "实际观测", "seasonal_expected": "历史同小时基准"})
    lines = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("time:Q", title="项目相对小时", scale=alt.Scale(zero=False)),
        y=alt.Y("数值:Q", title="KPI 观测值（数据集原始尺度）", scale=alt.Scale(zero=False)),
        color=alt.Color("曲线:N", legend=alt.Legend(orient="bottom")),
        strokeDash=alt.StrokeDash("曲线:N", legend=None),
        detail="segment:N",
        tooltip=["time:Q", "曲线:N", alt.Tooltip("数值:Q", format=".5g")],
    )
    points = alt.Chart(result.loc[result["flag"].eq(1)]).mark_point(
        color="#d62728", size=65, filled=True
    ).encode(x="time:Q", y="raw_value:Q", tooltip=["time:Q", "raw_value:Q", "score:Q"])
    return (lines + points).properties(height=280)


def select_observations(data, cell, window):
    selected = data.loc[data["cell"].eq(cell) & data["window"].eq(window)].sort_values("time").copy()
    if selected.empty or selected["time"].duplicated().any():
        raise ValueError("所选小区、时段为空或小时重复。")
    if not np.isfinite(selected["time"]).all():
        raise ValueError("小时编号无效。")
    return selected.reset_index(drop=True)


def detect_group(cell, window, method, stamps):
    """Four independent univariate models, not a multivariate detector."""
    results = []
    for kpi in KPIS:
        data = select_observations(load_test_data(kpi, stamps[kpi]), cell, window)
        detector = load_detector(kpi, method, stamps[kpi])
        result = detect(data, detector)
        result["kpi"], result["method"] = kpi, method
        results.append(result)
    return pd.concat(results, ignore_index=True)


def record_summary(result):
    """Explicit OR: positive if any positive; negative only if all 4 negative."""
    keys = ["cell", "window", "time"]
    if result.empty or result.duplicated(keys + ["kpi"]).any():
        raise ValueError("检测结果为空或同一 KPI 记录重复。")
    if not result["kpi"].isin(KPIS).all() or not result["flag"].isin([-1, 0, 1]).all():
        raise ValueError("未知 KPI 或异常标记。")
    flags = result.pivot(index=keys, columns="kpi", values="flag").reindex(columns=KPIS).fillna(-1)
    out = flags.reset_index()[keys].copy()
    values = flags.to_numpy()
    out["flag"] = np.where((values == 1).any(axis=1), 1,
                           np.where((values == 0).all(axis=1), 0, -1))
    for kpi in KPIS:
        out[kpi] = flags[kpi].map(STATES).to_numpy()
    out["异常KPI"] = ["、".join(k for k, f in zip(KPIS, row) if f == 1) for row in values]
    out["可判定KPI数"] = (values >= 0).sum(axis=1)
    out["记录判定"] = out["flag"].map(STATES)
    return out.sort_values(keys).reset_index(drop=True)


def score_chart(result):
    result = result.sort_values("time").copy()
    result["segment"] = result["time"].diff().ne(1).cumsum()
    long = result.melt(id_vars=["time", "segment"], value_vars=["score", "limit"],
                       var_name="曲线", value_name="分数")
    long["曲线"] = long["曲线"].map({"score": "异常分数", "limit": "有效阈值"})
    return alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("time:Q", title="项目相对小时", scale=alt.Scale(zero=False)),
        y=alt.Y("分数:Q", scale=alt.Scale(zero=False)), detail="segment:N",
        color=alt.Color("曲线:N", legend=alt.Legend(orient="bottom")),
        strokeDash="曲线:N", tooltip=["time:Q", "曲线:N", alt.Tooltip("分数:Q", format=".8g")],
    ).properties(height=240)


def details_table(result):
    return result[["cell", "window", "time", "kpi", "method", "raw_value", "seasonal_expected",
                   INPUT, IMPUTED, "observed_beams", "score", "limit", "判定"]].rename(columns={
        "cell": "小区", "window": "测试时段", "time": "项目相对小时", "kpi": "KPI",
        "method": "检测方法", "raw_value": "实际观测", "seasonal_expected": "历史同小时基准",
        INPUT: "检测输入", IMPUTED: "输入是否填补", "observed_beams": "有效波束数",
        "score": "异常分数", "limit": "有效阈值"})


@st.cache_data(max_entries=4, show_spinner=False)
def load_training_ranges(kpi, stamp, version="raw-ranges-v1"):
    """Descriptive warning ranges only; never used to fit or clip inputs."""
    check_options(kpi)
    frame = pd.read_csv(DATA_ROOT / kpi / "prepared_observations.csv.gz",
                        usecols=["cell", "split", "raw_value"])
    frame = frame.loc[frame.split.eq("train") & np.isfinite(frame.raw_value)].copy()
    frame["cell"] = frame.cell.astype(str)
    return frame.groupby("cell").raw_value.agg(["min", "max"])


def custom_reference(cell, window, hour, stamps):
    """Existing per-cell/hour transforms, without fitting on edited values."""
    rows = []
    for kpi in KPIS:
        data = select_observations(load_test_data(kpi, stamps[kpi]), cell, window)
        row = data.loc[data.time.eq(hour)].copy()
        if len(row) != 1:
            raise ValueError(f"{kpi} 没有唯一对应小时，请选择其他记录。")
        row["kpi"] = kpi
        ranges = load_training_ranges(kpi, stamps[kpi])
        row["train_raw_min"] = ranges.loc[cell, "min"] if cell in ranges.index else np.nan
        row["train_raw_max"] = ranges.loc[cell, "max"] if cell in ranges.index else np.nan
        rows.append(row)
    result = pd.concat(rows, ignore_index=True)
    validate_reference(result)
    return result


def validate_reference(reference):
    if len(reference) != 4 or reference.kpi.tolist() != KPIS:
        raise ValueError("需要按固定顺序提供 4 项 KPI 的参考记录。")
    if any(reference[c].nunique(dropna=False) != 1 for c in ["cell", "window", "time"]):
        raise ValueError("不能混合不同小区、时段或小时的记录。")
    columns = ["raw_value", "seasonal_expected", "training_cell_scale", INPUT]
    if not np.isfinite(reference[columns].to_numpy(float)).all() or (reference.training_cell_scale <= 0).any():
        raise ValueError("本小时缺少有效原始观测或历史转换参数，暂不能进行前后对比，请换一个小时。")
    if reference[IMPUTED].any():
        raise ValueError("本小时含填补输入，不能冒充实测值做前后对比，请换一个小时。")


def custom_editor_frame(reference):
    validate_reference(reference)
    return pd.DataFrame({"KPI": KPIS, "原始观测值": reference.raw_value.to_numpy(float),
                         "当前输入值": reference.raw_value.to_numpy(float)})


def validate_custom(frame, reference):
    validate_reference(reference)
    if list(frame.columns) != ["KPI", "原始观测值", "当前输入值"] or len(frame) != 4:
        raise ValueError("请保留 4 行 KPI 和原来的列，只修改“当前输入值”。")
    if frame.KPI.tolist() != KPIS or not np.array_equal(
            pd.to_numeric(frame["原始观测值"], errors="coerce"), reference.raw_value.to_numpy(float)):
        raise ValueError("KPI 顺序和原始观测值不允许修改。")
    values = pd.to_numeric(frame["当前输入值"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("当前输入必须是完整的有限数值，不能留空、输入文字或无穷值。")
    if (values < 0).any():
        raise ValueError("这 4 项 KPI 的原始尺度输入不能为负数；不要填写预测页的 Z-score 值。")
    with np.errstate(over="ignore", invalid="ignore"):
        residual = (values - reference.seasonal_expected.to_numpy(float)) / reference.training_cell_scale.to_numpy(float)
    if not np.isfinite(residual).all() or (np.abs(residual) > 1e30).any():
        raise ValueError("输入过大，超出本页安全数值计算范围。请检查数值和单位。")
    return values


def compare_custom(reference, frame, method, stamps):
    """Run original and hypothetical points through identical saved detectors."""
    values = validate_custom(frame, reference)
    check_options(KPIS[0], method)
    originals, modified, comparison = [], [], []
    for j, kpi in enumerate(KPIS):
        row = reference.iloc[[j]].copy()
        detector = load_detector(kpi, method, stamps[kpi])
        original = detect(row, detector)
        current = row.copy()
        current["raw_value"] = values[j]
        # Reuse the precise stored model input for unchanged values. This avoids
        # round-off moving an unchanged point across a boundary on reprocessing.
        if values[j] != row.raw_value.iloc[0]:
            current[INPUT] = (values[j] - row.seasonal_expected.iloc[0]) / row.training_cell_scale.iloc[0]
        internal = (current[INPUT].iloc[0] - detector.center) / detector.scale
        if not np.isfinite(internal) or abs(internal) > np.finfo(np.float32).max / 2:
            raise ValueError("模型内部转换结果无效，请检查输入。")
        current[IMPUTED] = False
        current = detect(current, detector)
        if not np.isfinite(current.score).all():
            raise ValueError("模型未能得到有限分数，请检查输入大小。")
        original["method"] = current["method"] = method
        original["scenario"] = "原始观测"
        current["scenario"] = "用户假设输入（非实测）"
        originals.append(original)
        modified.append(current)
        comparison.append({
            "KPI": kpi, "原始观测值": row.raw_value.iloc[0], "当前输入值": values[j],
            "是否修改": bool(values[j] != row.raw_value.iloc[0]),
            "原始分数": original.score.iloc[0], "当前分数": current.score.iloc[0],
            "有效阈值": current.limit.iloc[0],
            "原始判定": original["判定"].iloc[0], "当前判定": current["判定"].iloc[0],
            "历史同小时基准": row.seasonal_expected.iloc[0], "训练期残差标准差": row.training_cell_scale.iloc[0],
            "原始检测输入": original[INPUT].iloc[0], "当前检测输入": current[INPUT].iloc[0],
            "小区": row.cell.iloc[0], "测试时段": row.window.iloc[0], "项目相对小时": row.time.iloc[0],
            "检测方法": METHODS[method], "输入尺度": "数据集原始尺度的小区波束均值",
            "当前场景": "用户假设输入（非实测）",
        })
    return pd.DataFrame(comparison), pd.concat(originals, ignore_index=True), pd.concat(modified, ignore_index=True)


def custom_context_chart(history, hour, value, kpi):
    """Unmodified neighbours are context only, not input to the point detector."""
    near = history.loc[history.time.between(hour - 12, hour + 12)].sort_values("time").copy()
    near["segment"] = near.time.diff().ne(1).cumsum()
    line = alt.Chart(near).mark_line(point=True, color="#64748b").encode(
        x=alt.X("time:Q", title="项目相对小时", scale=alt.Scale(zero=False)),
        y=alt.Y("raw_value:Q", title=f"{kpi}（数据集原始尺度）", scale=alt.Scale(zero=False)),
        detail="segment:N", tooltip=["time:Q", "raw_value:Q"],
    )
    point = alt.Chart(pd.DataFrame({"time": [hour], "value": [value], "来源": ["当前假设输入（非实测）"]})).mark_point(
        color="#d97706", size=150, shape="diamond", filled=True).encode(
        x="time:Q", y="value:Q", tooltip=["time:Q", "来源:N", "value:Q"])
    return (line + point).properties(height=240)
