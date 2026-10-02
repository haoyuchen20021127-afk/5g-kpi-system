"""Prediction controls; computation lives in core.prediction."""
import numpy as np
import streamlit as st
from core.prediction import (
    APP_VERSION,
    CUSTOM,
    FILTERS,
    HISTORICAL,
    HOUR,
    LABELS,
    TARGETS,
    array_hash,
    comparison_chart,
    evaluate,
    file_signature,
    filter_indices,
    history_table,
    load_model,
    load_test_data,
    predict_history,
    replace_draft,
    result_table,
    scenario_chart,
    scenario_table,
    shifted_frame,
    validate_editor
)


st.session_state.setdefault("ip_epoch", 0)
with st.sidebar:
    st.header("预测工作台")
    mode = st.radio("选择功能", [CUSTOM, HISTORICAL], key="ip_mode", persist_state="session")
    visible = st.multiselect("查看哪些 KPI", TARGETS, default=["DLPRB"],
                             format_func=lambda k: f"{k}｜{LABELS[k]}", key="ip_visible", persist_state="session")
    st.caption("这里仅选择显示内容，模型始终输入并输出 4 项 KPI。")
    st.caption("固定模型：LSTM seed 42；24 小时历史 → 下一小时。")

with st.container(height=500, border=True, key="gotsf_page_scroll"):
    st.title("基于GOTSF数据集格式的KPI 预测")
    st.caption("可自由编辑数据")
    try:
        signature = file_signature()
        x_test, y_test, meta, spec = load_test_data(signature)
    except Exception as error:
        st.error(f"无法读取模型或数据：{error}")
        st.stop()

    data_token = (APP_VERSION, signature)
    if st.session_state.get("ip_data_token") != data_token:
        st.session_state.pop("ip_full", None)
        st.session_state.pop("ip_context", None)
        st.session_state.pop("ip_scenario", None)
        st.session_state["ip_data_token"] = data_token

    if mode == CUSTOM:
        st.subheader("1. 选择样本")
        st.info("请输入 Z-score 标准化数值。"
                "例如把 -0.50 改成 -0.20，表示增加 0.30 个 Z-score 单位。")
        selected = st.selectbox(
            "选择起始历史样本", list(range(len(meta))),
            format_func=lambda i: f"样本 {i+1}｜{meta.loc[i, 'series_id']}｜预测小时 {int(meta.loc[i, 'target_sequence'])}",
            key="ip_sample", persist_state="session",
        )
        target_hour = int(meta.loc[selected, "target_sequence"])
        original = x_test[selected].copy()
        original_frame = history_table(original.astype(float), target_hour)
        context = (data_token, selected)
        if st.session_state.get("ip_context") != context:
            replace_draft(original_frame)
            st.session_state["ip_context"] = context
        elif st.session_state.get("ip_last_mode") != CUSTOM:
            replace_draft(st.session_state["ip_draft"], clear_result=False)

        if st.button("恢复该样本的原始输入", key="ip_restore"):
            replace_draft(original_frame)

        with st.expander("可选：批量调整最后几小时"):
            a, b, c = st.columns(3)
            shift_kpi = a.selectbox("调整哪项 KPI", TARGETS, key="ip_shift_kpi")
            last_n = b.number_input("最后几小时", 1, 24, 3, step=1, key="ip_last_n")
            offset = c.number_input("统一增加量（负数表示减少）", value=0.3, step=0.1,
                                    format="%.3f", key="ip_offset")
            st.caption("在当前输入基础上增加相同数值。")
            if st.button("应用到输入表", key="ip_apply_shift"):
                try:
                    replace_draft(shifted_frame(st.session_state["ip_draft"], target_hour,
                                                shift_kpi, int(last_n), offset))
                except ValueError as error:
                    st.error(str(error))

        # 编辑结果只保存为草稿，不反写 seed，避免隔次编辑丢失。
        editor_key = f"ip_grid_{st.session_state['ip_epoch']}"
        for old_key in list(st.session_state):
            if old_key.startswith("ip_grid_") and old_key != editor_key:
                del st.session_state[old_key]
        st.caption(f"输入小时 {target_hour-24}—{target_hour-1}，预测小时 {target_hour}。"
                   "双击数字编辑，按 Enter 确认。")
        edited = st.data_editor(
            st.session_state["ip_seed"], num_rows="fixed", disabled=[HOUR],
            hide_index=True, height=330, width="stretch", key=editor_key,
            column_config={HOUR: st.column_config.NumberColumn(format="%d", pinned=True),
                           **{k: st.column_config.NumberColumn(k, format="%.5f", help=LABELS[k]) for k in TARGETS}},
        )
        st.session_state["ip_draft"] = edited.copy(deep=True)
        valid_input = None
        try:
            valid_input = validate_editor(edited, target_hour)
            current_hash = array_hash(valid_input)
            modified_count = int(np.count_nonzero(valid_input != original))
            st.caption(f"当前改变了 {modified_count} / 96 个模型输入值。")
            outside = ((valid_input < np.asarray(spec["training_input_min"]))
                       | (valid_input > np.asarray(spec["training_input_max"])))
            if outside.any():
                st.warning(f"有 {int(outside.sum())} 个输入超出训练期对应 KPI 的取值范围。"
                           "可以作探索性预测，但结果可能不可靠；程序不会自动截断这些值。")
            else:
                st.caption("单项取值未超出训练期范围，也不保证修改后的组合符合实际网络规律。")
        except ValueError as error:
            current_hash = None
            st.error(str(error))

        payload = st.session_state.get("ip_scenario")
        if payload is not None and payload["input_hash"] != current_hash:
            st.session_state.pop("ip_scenario", None)
            st.info("输入已改变，旧预测已清除。请重新点击下方按钮。")

        st.subheader("2. 使用当前输入预测")
        if st.button("预测并比较修改前后", type="primary", key="ip_predict_custom", disabled=valid_input is None):
            st.session_state.pop("ip_scenario", None)
            try:
                with st.spinner("模型正在计算原始输入和当前输入的两组预测……"):
                    model = load_model(signature, spec["X_sha256"])
                    pair = predict_history(model, np.stack([original, valid_input]))
                    st.session_state["ip_scenario"] = {"input_hash": current_hash, "prediction": pair}
            except Exception as error:
                st.error(f"预测失败：{error}")

        st.caption("只比较同一个模型对两份输入的输出"
                   "也不能据此证明某项 KPI 是另一项变化的原因。")
        payload = st.session_state.get("ip_scenario")
        if payload is not None and payload["input_hash"] == current_hash:
            pair = payload["prediction"]
            result = scenario_table(pair[0], pair[1], meta.loc[selected], current_hash)
            if modified_count == 0:
                st.info("当前输入与原始输入一致，两组预测应相同。改几个数再预测即可观察变化。")
            if visible:
                st.dataframe(result.loc[result["KPI"].isin(visible),
                                         ["KPI", "原始输入的预测", "当前输入的预测", "变化量"]],
                             hide_index=True, key="ip_scenario_table")
                for kpi in visible:
                    st.altair_chart(scenario_chart(original, valid_input, pair, target_hour, kpi),
                                    key=f"ip_scenario_chart_{kpi}")
                st.caption("线条表示历史输入；最右侧符号表示下一小时的预测，不是真实观测。")
            else:
                st.info("请在左侧选择至少一项 KPI 查看结果；模型仍已预测全部 4 项。")
            st.download_button("下载 4 项 KPI 的场景对比结果", result.to_csv(index=False).encode("utf-8-sig"),
                               "gotsf_scenario_predictions.csv", "text/csv", key="ip_scenario_download")
        if valid_input is not None:
            export_input = history_table(valid_input, target_hour)
            export_input.insert(0, "series_id", meta.loc[selected, "series_id"])
            st.download_button("下载当前 24 小时输入（Z-score）", export_input.to_csv(index=False).encode("utf-8-sig"),
                               "gotsf_custom_input_zscore.csv", "text/csv", key="ip_input_download")

    else:
        st.subheader("历史测试查询")
        st.info("先用原始 24 小时窗口计算，再筛选预测结果。每个窗口独立预测下一小时，"
                "真实值仅用于评分，未使用自定义页面的修改。")
        with st.sidebar:
            st.subheader("查询条件")
            series = st.selectbox("测试序列", list(meta["series_id"].unique()), key="ip_series")
            series_hours = meta.loc[meta["series_id"].eq(series), "target_sequence"]
            lo, hi = int(series_hours.min()), int(series_hours.max())
            hours = st.slider("目标小时范围", lo, hi, (lo, hi), key="ip_hours")
            rule = st.selectbox("进一步筛选", FILTERS, key="ip_filter_rule")
            filter_kpi = st.selectbox("筛选依据 KPI", TARGETS, key="ip_filter_kpi")
            threshold, top_n = 0., 10
            if rule in FILTERS[1:3]:
                threshold = st.number_input("阈值（Z-score）", value=0., step=0.05, key="ip_threshold")
                if rule == FILTERS[2] and threshold < 0:
                    st.caption("绝对误差非负；负阈值会保留范围内全部记录。")
            elif rule == FILTERS[3]:
                top_n = int(st.number_input("最多显示几条", 1, len(meta), 10, key="ip_top_n"))
            st.caption("筛选仅改变查询范围，不改变模型及其历史输入。")
        if st.button(f"运行历史测试预测（{len(meta)} 个窗口）", type="primary", key="ip_predict_history"):
            st.session_state.pop("ip_full", None)
            try:
                with st.spinner("调用已训练模型计算历史预测……"):
                    model = load_model(signature, spec["X_sha256"])
                    st.session_state["ip_full"] = predict_history(model, x_test)
            except Exception as error:
                st.error(f"预测失败：{error}")
        if "ip_full" not in st.session_state:
            st.caption("点击按钮计算后，可以反复筛选，无需每次重新运行模型。")
        else:
            predicted = st.session_state["ip_full"]
            ids = filter_indices(meta, y_test, predicted, series, hours, rule, filter_kpi, threshold, top_n)
            st.caption(f"全部测试窗口 {len(meta)} 个；当前筛选命中 {len(ids)} 个。")
            if not len(ids):
                st.warning("没有符合条件的结果，请扩大时间范围或调整筛选条件。")
            else:
                metrics = evaluate(y_test[ids], predicted[ids])
                st.subheader("当前筛选范围的评估")
                st.caption("这里的指标只描述筛选后的数据，不是全测试集成绩；按误差筛选会改变指标。"
                           "仅 1 条记录或真实值恒定时，R² 不定义，显示为空。")
                st.dataframe(metrics.loc[metrics["KPI"].isin(visible)], hide_index=True, key="ip_filtered_metrics")
                if not visible:
                    st.info("请在左侧选择至少一项 KPI 查看曲线和数值。")
                for kpi in visible:
                    st.subheader(f"{kpi}：真实值与预测值")
                    st.altair_chart(comparison_chart(meta.iloc[ids], y_test[ids], predicted[ids], kpi),
                                    key=f"ip_history_chart_{kpi}")
                result = result_table(meta.iloc[ids], y_test[ids], predicted[ids])
                columns = ["series_id", "target_sequence", "target_time"]
                columns += [f"{kind}__{k}" for k in visible for kind in ("actual", "predicted", "abs_error")]
                st.caption("actual = 真实值；predicted = 预测值；abs_error = 绝对误差。明细按时间排序。")
                st.dataframe(result[columns], hide_index=True, key="ip_filtered_results")
                st.download_button("下载当前筛选结果", result[columns].to_csv(index=False).encode("utf-8-sig"),
                                   "gotsf_filtered_predictions.csv", "text/csv", key="ip_filter_download")

            with st.expander("完整测试集成绩与下载（不受筛选影响）"):
                all_metrics = evaluate(y_test, predicted)
                st.dataframe(all_metrics, hide_index=True, key="ip_full_metrics")
                st.download_button("下载全部预测与误差", result_table(meta, y_test, predicted).to_csv(index=False).encode("utf-8-sig"),
                                   "gotsf_lstm_test_predictions.csv", "text/csv", key="ip_full_download")
                st.download_button("下载完整测试集评估", all_metrics.to_csv(index=False).encode("utf-8-sig"),
                                   "gotsf_lstm_test_metrics.csv", "text/csv", key="ip_metrics_download")

    st.session_state["ip_last_mode"] = mode
    with st.expander("数据与模型边界"):
        st.markdown(
            "- 使用第五周 GOTSF 小时级网络汇总数据，输入输出均为既有 Z-score 尺度，不是单小区预测。\n"
            "- 固定 LSTM seed 42，24 小时、4 项 KPI 输入，联合预测下一小时的 4 项 KPI。\n"
            "- 手动修改只保存在当前网页会话，不改源数据、不改模型权重；需要保留时请下载。\n"
            "- 原始单位 CSV 上传、多模型切换、重新训练和多步预测尚未开放。\n"
            "- 沿用的预处理使用过全表统计量，存在信息泄漏限制，只作为离线实验与交互演示。\n"
            "- 假设场景没有对应真实答案，预测变化不是因果证据。异常检测请切换到独立页面，其输入是小区级观测。"
        )
