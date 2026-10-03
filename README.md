# 5G KPI 工程：GOTSF Web + DONeXT / NetLab 离线实验

本版在原 GOTSF 独立工程上增加 DONeXT 与 NetLab 的代码、已训练模型、模型实际使用的完整数据及原实验结果。**不包含 5G-Datasets，也不包含原始全量采集文件。**

- **网页仍然只有 GOTSF**：交互式 LSTM 预测和四种异常检测，页面、计算逻辑、模型、阈值及资产保持不变。
- **新增 offline/**：DONeXT 三组预测/八组检测；NetLab 一组预测/两组检测，支持命令行加载、复核和可选重训。
- 旧 Week1–Week7 工程、原 GOTSF 文件夹及旧提交包保留，不删除、不覆盖。

## 先看哪里

| 你要做的事 | 入口 |
|---|---|
| 运行 GOTSF 网页 | 根目录 app.py / start.ps1，见下面启动命令 |
| 了解网页操作 | docs/用户操作说明.md |
| 运行 DONeXT / NetLab 预测或检测 | offline/README.md |
| 查看每个任务有哪些数据和多少样本 | offline/任务与数据清单.md |
| 找预测样本、模型及历史指标 | offline/forecast/results/数据集/任务 |
| 找检测观测、输入、模型及阈值 | offline/anomaly/results/datasets/数据集/任务 |
| 查看技术报告 / PPT | docs/；原报告和八页 PPT 仍以 GOTSF Web 为主题，未扩写成多数据集网页报告 |

## 在原电脑启动 GOTSF 网页

在本文件所在目录打开 PowerShell：

```powershell
& "E:\5G_Project\week5\.venv\Scripts\python.exe" -m streamlit run .\app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

浏览器访问 http://127.0.0.1:8501。保持终端运行；停止按 Ctrl+C。端口占用时改为 --server.port 8502。也可使用 `start.ps1 -Python <解释器路径>`。
命令中的 Week5 仅是借用已有 Python 环境；代码、数据和模型从本包读取，不依赖旧 Week4/5/6/7 的数据目录。

## 新电脑环境

已验证环境为 Windows / Python 3.13.12。推理使用 CPU；无需 GPU。安装时需要网络，正常加载本地数据推理不需要下载模型。

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r ".\offline\requirements.txt"
& ".\.venv\Scripts\python.exe" -m streamlit run .\app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

`offline/requirements.txt` 包含原网页依赖及离线 XGBoost/LightGBM 依赖。只运行 GOTSF 时可只安装根目录 requirements.txt。请保留已验证版本，勿默认升级；新电脑全新安装未实测。

## 数据说明

预测的 samples.npz 包含窗口输入 X 和真实目标 Y，sample_index.csv 保留既有 train/validation/test 划分；样本不是再次随机抽取的五条演示数据。
异常检测的 prepared_observations.csv.gz、model_inputs.csv.gz 分别保留整理后的观测和实际检测输入，以及 split/phase、缺失标记与参考划分。
两类数据粒度和尺度不同，不能直接混用；预测沿用 Z-score 尺度，检测使用各任务已定义的指标与上下文。

## 验证

```powershell
& "E:\5G_Project\week5\.venv\Scripts\python.exe" -m unittest discover -s tests -v
& "E:\5G_Project\week5\.venv\Scripts\python.exe" .\offline\check_delivery.py --output .\runs\offline_check.json
```

结果文件必须使用新的路径。原 GOTSF 测试与新离线核验分开记录。详细验证范围、容差、限制及检查结果见 offline/verification，不把抽样回归说成全量重算。

## 已知边界与交付

- 预测继承早期全来源统计预处理，存在信息泄漏局限；本次打包不消除该问题。
- GOTSF/DONeXT 无故障真值；NetLab 状态标签不等于逐 KPI 健康真值。异常不等于确认故障，相关不等于原因。
- 此次不改变历史划分、不调参、不重训全量模型。可选重训从已处理数据开始，不包含完整原始清洗流水线。
- 不新增多数据集 Web、实时采集、任意新场景自动适配或自动根因诊断。
- 本地 Git 仓库继承 GOTSF 交付历史；ZIP 不含 .git/虚拟环境/缓存，bundle 可恢复完整提交历史。
- 没有配置或上传 GitHub。文件较大，未来公开上传需先处理大模型存储与数据再分发许可。
