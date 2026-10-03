# DONeXT 与 NetLab 离线实验

这是已完成实验的可运行整理版，不是新增 Web 页面，也不是重新训练得到的新结果。
仅新增 DONeXT 和 NetLab；按用户要求不包含 5G-Datasets。GOTSF 继续使用工程根目录的网页、代码和资产。

## 包含什么

| 数据集 | 预测任务 | 异常检测任务 |
|---|---|---|
| DONeXT | cell、iperf、latency 三组多输入多输出任务 | hbahn 的 cell/iperf/latency，mobile 的 cell/iperf/latency，static 的 cell/latency，共八组 |
| NetLab | 无线侧 joint_kpi 联合预测 | radio_kpi 无线侧、network_dataset 网络侧，共两组 |

- 每个预测任务保留 LSTM、GRU、Transformer、线性回归、决策树、随机森林、XGBoost、LightGBM 的三个种子（42/43/44）模型，并保留持续性基线结果。
- 每个异常检测任务保留所选 KPI 的 3σ、IQR、孤立森林和 One-Class SVM 规则/模型；按原上下文分开加载。参考不足、常量参考及不适用状态不改成“正常”。
- 保留配套的全部已构建预测样本、异常分析观测、模型输入及划分；未随机抽取一个小演示集替代它们。
- 不包含原始采集文件，不从最初的原始数据重做预处理；也没有把第三周用于预测的缩尾数据混作异常检测输入。

## 目录

| 目录 | 用途 |
|---|---|
| `forecast/code` | 网络结构、传统模型工厂、预测、指标、选参及重训代码 |
| `forecast/results/数据集/任务/data` | samples.npz、sample_index.csv、manifest.json |
| `forecast/results/数据集/任务/models/算法` | 三个种子的模型、真实/预测值、指标、配置及训练记录 |
| `forecast/results/training_source*` | 训练时冻结的源码快照，仅供溯源；其中的原始数据准备脚本不是本包入口 |
| `anomaly/code` | 完整通用检测器、参考填补、离线拟合、保存模型推理和标签统计 |
| `anomaly/results/datasets/数据集/任务` | prepared_observations、model_inputs、models、kpi_summary、原测试分数及参考划分 |
| `anomaly/label_statistics` | 已保存的逐 KPI 标签统计及同方法内的记录级 OR 结果 |
| `verification` | 资产来源和校验值、GOTSF 不变核对、扩展版测试结果 |

任务和样本数量详见 `任务与数据清单.md`。`kpi_summary.csv` 中的 model_file 对应每个上下文、指标和方法的模型，不能仅根据 joblib 文件名推断所属指标。

## 环境

在工程根目录打开 PowerShell。原 Windows 环境可继续使用：

```powershell
$py = "E:\5G_Project\week5\.venv\Scripts\python.exe"
```

换电脑时，在根目录创建独立环境，安装本文件夹的 requirements.txt：

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r ".\offline\requirements.txt"
$py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
```

这是当前已验证环境的版本清单，不自动升级依赖。新电脑全新安装未实测。推理可使用 CPU，不需要 CUDA；随机森林模型较大，一次只加载一个模型，加载时需要较多内存。只有可信的本地模型文件才能通过 joblib/torch 加载。

## 1. 加载已训练模型预测（不训练）

```powershell
& $py ".\offline\forecast\code\evaluate.py" --task donext/iperf --model LSTM --seed 42 --output ".\runs\donext_iperf_lstm"
```

这会对该任务的**全部测试窗口**预测，生成 predictions.csv、metrics.json 和 run_info.json。下一次运行请换一个新的输出目录，不覆盖原结果。

可选任务：`donext/cell`、`donext/iperf`、`donext/latency`、`netslab_5g_oran_idd/joint_kpi`。
可选模型：`Persistence`、`LinearRegression`、`DecisionTree`、`RandomForest`、`XGBoost`、`LightGBM`、`LSTM`、`GRU`、`Transformer`。

若要对自己编辑的 CSV 预测，可用 predict.py，列顺序必须符合 manifest.json 的 flat_feature_order。example_predict_input.csv 是五个真实测试窗口的输入示例，不代表完整数据只有五条：

```powershell
& $py ".\offline\forecast\code\predict.py" --model ".\offline\forecast\results\donext\iperf\models\LSTM\seed_42.pt" --manifest ".\offline\forecast\results\donext\iperf\data\manifest.json" --input ".\offline\forecast\results\donext\iperf\example_predict_input.csv" --output ".\runs\donext_custom_prediction.csv"
```

输入必须沿用该任务已有的 Z-score 尺度；本包不支持将任意物理单位原始 CSV 自动变换为兼容输入。

## 2. 加载保存的检测器（不训练）

```powershell
& $py ".\offline\anomaly\code\detect_saved.py" --task netslab_5g_oran_idd/radio_kpi --method isolation_forest --output ".\runs\netlab_radio_if"
```

它会重新计算全部测试记录的逐 KPI 分数和判定，校验与原保存结果一致，再输出 predictions.csv.gz、summary.json。记录级结果是在**同一种方法内**对适用 KPI 作 OR：任一疑似异常即异常，无法判定不会自动计为正常。

检测方法可选 `three_sigma`、`iqr`、`isolation_forest`、`one_class_svm`。任务清单见 `任务与数据清单.md`。NetLab 网络侧记录多，SVM 重算可能明显慢于统计规则。

## 3. 可选：离线重新训练

只有需要重做实验时才运行；启动 GOTSF 网页或复核已有模型不需要这些步骤。所有输出必须是新的独立目录，不会切换网页模型。

预测重训会从保存的 train/validation/test 样本开始，按原候选空间在验证集选参数，之后 train+validation 合并拟合三个种子，测试集只评估：

```powershell
& $py ".\offline\forecast\code\train_comparison.py" --tasks donext/iperf --models LSTM --device cpu --output-root ".\runs\donext_iperf_retrain"
```

异常重训从保存的分析观测开始，复用原分组及划分，重新估计参考填补参数、拟合检测器和校准阈值：

```powershell
& $py ".\offline\anomaly\code\refit_prepared_inputs.py" --task netslab_5g_oran_idd/radio_kpi --output ".\runs\netlab_radio_retrain"
& $py ".\offline\anomaly\code\summarize_by_label.py" --results ".\runs\netlab_radio_retrain" --output ".\runs\netlab_radio_retrain_labels"
```

CPU 重训不保证逐位复现原 CUDA 神经网络权重。此次交付整理不重跑全量模型训练，也不重新优化参数。

## 4. 检查交付是否完整

```powershell
& $py ".\offline\check_delivery.py" --output ".\runs\offline_check.json"
& $py -m unittest discover -s tests -v
```

第一条默认核对所有来源文件 SHA256，并对全部保存种子/模型及测试上下文作抽样推理回归。它**不是对所有模型的全部测试记录穷举重算**；完整单任务复核使用 evaluate.py/detect_saved.py。第二条是原 GOTSF 网页的自动化回归。

## 解释边界

- 使用第五周全算法对比和第六周所选 KPI 结果为基准，不混入其他版本的重划分或阈值。
- 预测沿用早期全来源统计预处理，存在信息泄漏局限；相同样本比较不消除该问题。
- NetLab 的 benign/attack 是状态标签，不是逐 KPI 健康真值；DONeXT 没有真实故障标签。异常比例不是确认故障率。
- 预测数据和检测数据的粒度、尺度、预处理和分组不同，不能把预测样本直接当检测输入。
- 旧 task_manifest 的 row__ 综合判定说明属于原拟合流程的辅助结果；本包标签统计及新推理入口使用逐 KPI OR，不混用两种记录规则。
- 某些 JSON 里的旧 E 盘路径仅是历史来源信息；新预测/检测入口从本包相对位置读取配套数据和模型。
- 只做本地交付，未上传任何数据。未来公开发布前需核实数据再分发许可；大模型文件不能直接当普通 GitHub 文件上传，应另行规划 Git LFS 或资产分发。
