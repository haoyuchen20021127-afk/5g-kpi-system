# GOTSF KPI 预测与异常检测系统

这是第八周整理后的独立工程：保留 GOTSF 的交互式 LSTM 预测、4 种异常检测方法及必要数据和已训练模型。运行时不再访问 week5、week6、week7。旧工程保持原样，作为历史备份。

## 启动

在本文件所在目录打开 PowerShell。如果沿用原项目的 Python 环境：

```powershell
.\start.ps1 -Python "E:\5G_Project\week5\.venv\Scripts\python.exe"
```

也可以直接执行，避免 PowerShell 脚本执行策略限制：

```powershell
& "E:\5G_Project\week5\.venv\Scripts\python.exe" -m streamlit run .\app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

浏览器访问 http://127.0.0.1:8501 。终端需保持运行；停止时按 Ctrl+C。若 8501 已被占用，可在启动命令后加 `--server.port 8502`，相应访问 8502。

换电脑时，建议创建独立 Python 环境并安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

已验证环境是 Python 3.13.12。`torch==2.8.0` 不强制 CUDA，网页使用 CPU。新电脑全新安装尚未实测；若依赖版本不可用，不要随意替换后宣称结果一致，应重新运行测试。

## 使用

预测页面有两个模式。自定义模式先选择一个测试窗口，编辑过去 24 小时的 4 项 KPI，再点击比较预测；历史模式对 144 个测试窗口运行预测，再按时段、预测值、误差或 Top N 筛选。修改输入不会重新训练或写回原始数据。

异常检测也有两个模式。历史模式选择方法、小区、测试时段后运行，查看各 KPI 分数和记录级判定；自定义模式选一个已有小时，修改 4 个 KPI 的观测值后比较前后结果。任一项疑似异常，该小时即标为疑似异常。

**两页不能共用输入。** 预测是网络级汇总的 Z-score；检测是小区级波束均值的源尺度，内部再去除日周期并缩放。检测不是针对 LSTM 预测值，分数不是异常概率。详见 `docs/用户操作说明.md`。

## 目录

| 路径 | 作用 |
|---|---|
| `app.py` | 页面导航入口 |
| `app_pages/` | 两个页面的控件、交互和展示 |
| `core/config.py` | 相对路径和 KPI 配置 |
| `core/models.py` | 1 层、32 隐藏单元的 LSTM 结构 |
| `core/prediction.py` | 输入校验、加载、推理、误差和图表 |
| `core/detectors.py` | 3σ、IQR、孤立森林、One-Class SVM 的评分和离线拟合 |
| `core/anomaly.py` | 小区数据、输入转换、检测、前后对比及记录级汇总 |
| `assets/` | 必需的已处理数据、权重、阈值和训练清单 |
| `scripts/retrain.py` | 可选离线重训，不由网页调用 |
| `tests/` | 资产、推理、边界和页面交互测试 |
| `evidence/` | 结果核对、文件哈希及历史实验指标 |
| `docs/` | 技术报告、演示 PPT、操作说明和整理记录 |

## 验证与离线重训

```powershell
python -m unittest discover -s tests -v
python -m scripts.retrain lstm --output runs/lstm_new
python -m scripts.retrain detectors --output runs/detectors_new
```

请将 `python` 换成实际环境解释器。重训要求新的输出目录，不覆盖交付模型，也不自动切换网页模型。LSTM 重训复用历史已选参数和 80 轮最终拟合，不重复原来的候选搜索；输入是已处理的样本，不是原始数据。CPU 与历史 CUDA 训练不保证逐位相同。

## 已知边界

- GOTSF 无人工故障标签，不能由异常比例推导准确率、召回率或误报率。
- 预测沿用了早期全源数据统计的预处理，存在信息泄漏局限。表中误差用于复核已有离线实验，不是严格无泄漏部署基准。
- 不提供实时采集、任意新小区、文件上传、自动根因确认或在线训练。相关性案例是离线分析材料。
- 只加载本工程可信的 `.pt`、`.joblib`；不加载来历不明的模型文件。

## Git 与交付

本目录是本地 Git 仓库，没有配置远程地址，也没有上传 GitHub。提交包 ZIP 包含运行所需代码、资产和文档，但不包含虚拟环境或 `.git`。Git bundle 是可离线恢复历史的另一份交付：`git clone <bundle路径> gotsf-kpi-system`。

未来公开发布前，请核实 GOTSF 数据及模型再分发条款。此交付只在本地整理，不新增许可声明。
