import numpy as np
import pandas as pd
import streamlit as st
from core import anomaly as ad

with st.sidebar:
    st.header("异常检测设置")
    mode = st.radio("检测功能", ["历史检测查询", "自定义数据检测"], key="ad_mode", persist_state="session")
    method = st.selectbox("检测方法", list(ad.METHODS), format_func=ad.METHODS.get,
                          key="ad_method", persist_state="session")
    st.caption("沿用第六周模型和阈值；这里不重新训练。")

with st.container(height=500, border=True, key="anomaly_scroll"):
    st.title("GOTSF KPI 异常检测")
    st.caption("选择小区和时段，调用 4 个 KPI 各自的检测模型，再汇总逐小时结果。")
    st.info("历史模式检测测试数据集观测；自定义模式检测你填写自定义数值。")
    try:
        stamps = {k: ad.signature(k) for k in ad.KPIS}
        data = ad.load_test_data(ad.KPIS[0], stamps[ad.KPIS[0]])
    except Exception as exc:
        st.error(f"无法读取第六周数据：{exc}")
        st.stop()

    with st.sidebar:
        cell = st.selectbox("小区编号", sorted(data["cell"].unique()), key="ad_cell", persist_state="session")
        windows = data.groupby("window")["time"].min().sort_values().index.tolist()
        window = st.selectbox("测试时段", windows, key="ad_window", persist_state="session")
        kpi = st.selectbox("查看哪项 KPI 的曲线", ad.KPIS, format_func=lambda k: f"{k}｜{ad.LABELS[k]}",
                           key="ad_kpi", persist_state="session")
        st.caption("KPI 选项只切换图表；每次运行仍检测全部 4 项。")

    try:
        selected = ad.select_observations(data, cell, window)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    lo, hi = int(selected.time.min()), int(selected.time.max())
    if mode == "自定义数据检测":
        st.subheader("1. 选一个小时，修改该小时的观测值")
        st.caption("这是单小时试验，不需要 24 小时输入窗口。切换小区、时段或小时会载入新原值；"
                   "切换方法保留当前修改。原始数据和历史检测结果不受影响。")
        hour = st.selectbox("选择要试验的项目相对小时", selected.time.astype(int).tolist(),
                            key=f"ac_hour_{window}", persist_state="session")
        try:
            reference = ad.custom_reference(cell, window, hour, stamps)
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
            st.stop()
        original_frame = ad.custom_editor_frame(reference)
        context = (ad.VERSION, cell, window, hour, tuple(stamps.items()))
        st.session_state.setdefault("ac_epoch", 0)
        new_context = st.session_state.get("ac_context") != context
        restore = st.button("恢复这个小时的原始值", key="ac_restore")
        if new_context or restore:
            st.session_state["ac_seed"] = original_frame.copy(deep=True)
            st.session_state["ac_draft"] = original_frame.copy(deep=True)
            st.session_state["ac_epoch"] += 1
            st.session_state["ac_context"] = context
            st.session_state.pop("ac_result", None)
        elif st.session_state.get("ac_remount", False):
            st.session_state["ac_seed"] = st.session_state["ac_draft"].copy(deep=True)
            st.session_state["ac_epoch"] += 1
        st.session_state["ac_remount"] = False
        editor_key = f"ac_grid_{st.session_state['ac_epoch']}"
        for old_key in list(st.session_state):
            if old_key.startswith("ac_grid_") and old_key != editor_key:
                del st.session_state[old_key]
        st.info("请修改“当前输入值”，使用与原始观测相同的数据集尺度。"
                "这里不是预测页的 Z-score，也不要自行换算成百分数、Mbps 等单位。")
        st.caption("双击数值编辑，按 Enter 确认。数据是小区的波束均值，因此数量类指标也可能为小数。")
        edited = st.data_editor(
            st.session_state["ac_seed"], key=editor_key, hide_index=True, num_rows="fixed",
            disabled=["KPI", "原始观测值"], height=180,
            column_config={
                "原始观测值": st.column_config.NumberColumn(format="%.6f"),
                "当前输入值": st.column_config.NumberColumn(format="%.6f", min_value=0., required=True),
            },
        )
        # Keep a stable seed, never feed the returned draft back on each rerun.
        st.session_state["ac_draft"] = edited.copy(deep=True)
        values, result_key = None, None
        try:
            values = ad.validate_custom(edited, reference)
            result_key = (context, method, tuple(values))
            changed = int(np.count_nonzero(values != reference.raw_value.to_numpy(float)))
            st.caption(f"当前修改了 {changed} / 4 项 KPI；小区 {cell}，项目相对小时 {hour}。")
            outside = ((values < reference.train_raw_min.to_numpy(float))
                       | (values > reference.train_raw_max.to_numpy(float)))
            if outside.any():
                names = "、".join(np.asarray(ad.KPIS)[outside])
                st.warning(f"{names} 超出该小区训练期的观测范围。可以作探索性检测，但模型对远离训练范围的数值可能不可靠；不会自动截断。")
        except ValueError as exc:
            st.error(str(exc))
        payload = st.session_state.get("ac_result")
        if payload is not None and payload["key"] != result_key:
            st.session_state.pop("ac_result", None)
            st.info("输入或方法已改变，旧结果已清除，请重新检测。")
        st.subheader("2. 比较修改前后的异常判定")
        if st.button("检测并比较修改前后", key="ac_run", type="primary", disabled=values is None):
            st.session_state.pop("ac_result", None)
            try:
                with st.spinner("正在用相同模型和阈值检测原始值与当前输入……"):
                    table, original, current = ad.compare_custom(reference, edited, method, stamps)
                    st.session_state["ac_result"] = {"key": result_key, "table": table,
                                                    "original": original, "current": current}
            except Exception as exc:
                st.error(f"检测失败：{exc}")
        payload = st.session_state.get("ac_result")
        if payload is not None and payload["key"] == result_key:
            table = payload["table"]
            old_record = ad.record_summary(payload["original"]).iloc[0]
            new_record = ad.record_summary(payload["current"]).iloc[0]
            with st.container(horizontal=True):
                st.metric("原始记录判定", old_record["记录判定"], border=True)
                st.metric("修改后记录判定", new_record["记录判定"], border=True)
            st.write(f"修改后疑似异常 KPI：{new_record['异常KPI'] or '无'}。任一项异常，该小时记录就标为疑似异常。")
            if changed == 0:
                st.caption("当前没有修改，前后分数和判定应当完全相同。")
            st.dataframe(table[["KPI", "原始观测值", "当前输入值", "原始分数", "当前分数",
                                "有效阈值", "原始判定", "当前判定"]], hide_index=True, key="ac_comparison")
            st.caption("分数严格高于有效阈值才判为疑似异常；分数不是概率。各 KPI 独立检测，修改一项不会改变其他项的分数。")
            export = table.copy()
            export["原始记录判定"] = old_record["记录判定"]
            export["修改后记录判定"] = new_record["记录判定"]
            st.download_button("下载本次 4 项 KPI 的前后对比 CSV", export.to_csv(index=False).encode("utf-8-sig"),
                               f"gotsf_custom_{cell}_{hour}_{method}.csv", "text/csv", key="ac_download")
            with st.expander("查看输入转换过程（历史参数保持不变）"):
                st.caption("检测输入 =（当前输入值 − 历史同小时基准）÷ 训练期残差标准差；随后应用模型已保存的后续变换。")
                st.dataframe(table[["KPI", "历史同小时基准", "训练期残差标准差", "原始检测输入", "当前检测输入"]],
                             hide_index=True, key="ac_transform")
        if values is not None:
            st.subheader(f"3. {kpi}：把当前输入放到原始曲线中看")
            history = ad.select_observations(ad.load_test_data(kpi, stamps[kpi]), cell, window)
            st.altair_chart(ad.custom_context_chart(history, hour, values[ad.KPIS.index(kpi)], kpi), key="ac_context_chart")
            st.caption("灰线是原始观测，橙色菱形是当前假设输入。附近曲线仅供对照，不参与这一小时的判定。")
        st.caption("这只是‘如果数值变成这样，模型会怎么判断’的试验；不证明真实故障或因果关系。"
                   "历史基准、模型和阈值均未调整。尚未开放任意小区、新时间点或 CSV 上传。")
        st.stop()

    st.session_state["ac_remount"] = True
    selection = (ad.VERSION, cell, window, method, tuple(stamps.items()))
    if st.session_state.get("ad_selection") != selection:
        st.session_state.pop("ad_result", None)
    st.subheader(f"小区 {cell} · {ad.METHODS[method]}")
    st.write(f"项目编号 {lo}—{hi}；本段 {len(selected)} 条小时记录，每条检测 4 项 KPI。")
    st.caption("小时编号从项目起点累计")
    if st.button("运行异常检测", type="primary", key="ad_run"):
        st.session_state.pop("ad_result", None)
        try:
            with st.spinner("正在计算 4 项 KPI 的异常分数……"):
                result = ad.detect_group(cell, window, method, stamps)
                ad.record_summary(result)  # Validate before storing.
                st.session_state["ad_result"] = result
                st.session_state["ad_selection"] = selection
        except Exception as exc:
            st.error(f"检测失败：{exc}")
    if "ad_result" not in st.session_state:
        st.caption("点击上方按钮开始。切换小区、时段或方法后，需要重新运行；切换显示 KPI 不需要。")
        st.stop()

    result = st.session_state["ad_result"]
    st.subheader("1. 选择查看范围")
    hours = st.slider("项目相对小时范围（仅筛选结果）", lo, hi, (lo, hi),
                      key=f"ad_hours_{window}", persist_state="session")
    view = result.loc[result["time"].between(*hours)].copy()
    records = ad.record_summary(view)
    with st.container(horizontal=True):
        st.metric("当前范围小时数", len(records), border=True)
        st.metric("任一 KPI 疑似异常的小时数", int(records.flag.eq(1).sum()), border=True)
        st.metric("四项均未判异常的小时数", int(records.flag.eq(0).sum()), border=True)
        st.metric("记录无法判定的小时数", int(records.flag.eq(-1).sum()), border=True)
    st.caption("按同一小区、同一时段、同一小时合并：只要至少 1 项异常，该条记录就判为疑似异常；"
               "不是把 4 项分数相加，也不是把四种方法投票。某项缺失且其他项均未异常时，不强行判正常。")
    counts = pd.DataFrame([{
        "KPI": k, "疑似异常小时数": int(view.loc[view.kpi.eq(k), "flag"].eq(1).sum()),
        "未判异常小时数": int(view.loc[view.kpi.eq(k), "flag"].eq(0).sum()),
        "无法判定小时数": int(view.loc[view.kpi.eq(k), "flag"].eq(-1).sum()),
    } for k in ad.KPIS])
    st.dataframe(counts, hide_index=True, key="ad_counts")

    focus = view.loc[view.kpi.eq(kpi)].sort_values("time")
    st.subheader(f"2. {kpi}｜{ad.LABELS[kpi]}：观测曲线与异常点")
    st.altair_chart(ad.observed_chart(focus), key="ad_observed_chart")
    st.caption("红点：当前方法判出的疑似异常。历史同小时基准来自训练数据，不是 LSTM 预测曲线。"
               "每条原始尺度观测是该小区最多 32 个有效波束值的均值。")
    st.subheader(f"3. {kpi}：异常分数与阈值")
    st.altair_chart(ad.score_chart(focus), key="ad_score_chart")
    st.write(f"有效阈值：{focus['limit'].iloc[0]:.10g}。分数严格高于阈值才判为疑似异常。")
    st.caption("分数不是异常概率；不同方法或不同 KPI 的分数不能直接比较大小。"
               "有效阈值沿用第六周校准结果及数值保护量，不根据当前筛选数据重算。")

    st.subheader("4. 逐小时结果与下载")
    only_flags = st.checkbox("表格只看疑似异常", key="ad_only_flags", persist_state="session")
    record_table = records.rename(columns={"cell": "小区", "window": "测试时段", "time": "项目相对小时"})
    record_table.insert(2, "检测方法", ad.METHODS[method])
    filtered_records = record_table.loc[records.flag.eq(1)] if only_flags else record_table
    filtered_records = filtered_records.drop(columns="flag")
    st.caption("记录级结果：4 项 KPI 中任一项异常即记为异常；计数和曲线仍保留所选时间范围的全貌。")
    if filtered_records.empty:
        st.info("当前筛选下没有疑似异常记录。")
    st.dataframe(filtered_records, hide_index=True, key="ad_record_table")
    st.download_button("下载当前记录级表格 CSV", filtered_records.to_csv(index=False).encode("utf-8-sig"),
                       f"gotsf_{cell}_{window}_{method}_records.csv", "text/csv", key="ad_records_download")
    detail = ad.details_table(focus.loc[focus.flag.eq(1)] if only_flags else focus)
    st.caption(f"下面是当前 KPI（{kpi}）的分数明细。")
    st.dataframe(detail, hide_index=True, key="ad_detail_table")
    st.download_button("下载当前 KPI 明细 CSV", detail.to_csv(index=False).encode("utf-8-sig"),
                       f"gotsf_{cell}_{window}_{method}_{kpi}.csv", "text/csv", key="ad_detail_download")
    st.download_button("下载所选时间范围全部 4 项 KPI 明细 CSV", ad.details_table(view).to_csv(index=False).encode("utf-8-sig"),
                       f"gotsf_{cell}_{window}_{method}_all_kpis.csv", "text/csv", key="ad_all_download")

    with st.expander("检测依据与适用范围"):
        st.markdown(
            "- 先减去训练期同小区、同小时的中位数，再除以训练期该小区残差的标准差，"
            "得到检测输入。模型沿用第六周保存的进一步变换和阈值，不现场重新拟合。\n"
            "- 4 种方法各自独立；3σ 和 IQR 也沿用原项目的校准规则，不宣称这里是未经校准的原始规则。\n"
            "- 检测页是小区级，预测页是网络级汇总，两者不能直接逐点比较或共用输入。\n"
            "- GOTSF 没有人工异常标签，因此不展示准确率、召回率或误报率。未判异常也不等于确认正常。\n"
            "- 当前是历史数据查询，不是实时监控。可切换到‘自定义数据检测’修改单小时输入；不影响这里的历史结果，也未开放上传。"
        )
        st.write(f"所选范围使用填补输入的 KPI 观测共 {int(view[ad.IMPUTED].sum())} 条；明细保留填补标记。")
        st.code(str(ad.DATA_ROOT), language="text")
