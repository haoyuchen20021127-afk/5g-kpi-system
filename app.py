"""Week 7 entry point: keep forecasting and anomaly inputs separate."""
from pathlib import Path
import sys
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

st.set_page_config(page_title="5G KPI 分析系统", layout="wide")
page = st.navigation([
    st.Page("app_pages/prediction.py", title="KPI 预测", icon=":material/show_chart:", default=True),
    st.Page("app_pages/anomaly.py", title="异常检测", icon=":material/troubleshoot:"),
])
# data_editor is not a persistent input widget. Rebuild it from its saved draft
# when returning from the other page, without re-applying its old edit patches.
if st.session_state.get("system_last_page") != page.title:
    if page.title == "KPI 预测":
        st.session_state["ip_last_mode"] = None
    elif page.title == "异常检测":
        st.session_state["ac_remount"] = True
    st.session_state["system_last_page"] = page.title
page.run()
