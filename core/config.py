"""All runtime assets are resolved relative to this project, never to old weeks."""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_ROOT = PROJECT_ROOT / "assets" / "prediction"
ANOMALY_ROOT = PROJECT_ROOT / "assets" / "anomaly"
KPIS = ["DLPRB", "DLThpVol", "DLThpTime", "MR_number"]
LABELS = {"DLPRB": "下行无线资源利用率", "DLThpVol": "下行传输量",
          "DLThpTime": "下行传输时间", "MR_number": "测量报告数量"}
