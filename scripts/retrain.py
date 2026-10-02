"""Refit GOTSF models from packaged prepared inputs; keep shipped models frozen.

This repeats the final selected LSTM recipe, not the old cross-model search.
It does not fix the historical full-source preprocessing leakage limitation.
"""
import argparse
import json
from pathlib import Path
import random
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from core.config import PREDICTION_ROOT, ANOMALY_ROOT, KPIS
from core.models import JointLSTM
from core.prediction import array_hash, evaluate, predict_history
from core.detectors import METHODS, fit_detector


def train_lstm(output):
    spec = json.loads((PREDICTION_ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    run = json.loads((PREDICTION_ROOT/'models/LSTM/run.json').read_text(encoding='utf-8'))
    with np.load(PREDICTION_ROOT/'data/samples.npz',allow_pickle=False) as arrays:
        x, y = arrays['X'].copy(), arrays['Y'].copy()
    if array_hash(x) != spec['X_sha256'] or array_hash(y) != spec['Y_sha256']:
        raise ValueError('输入数据校验失败')
    index = pd.read_csv(PREDICTION_ROOT/'data/sample_index.csv')
    fit = index.split.isin(['train','validation']).to_numpy()
    test = index.split.eq('test').to_numpy()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = JointLSTM(hidden_size=run['selected_config']['hidden_size'])
    optimizer = torch.optim.Adam(model.parameters(), lr=run['selected_config']['learning_rate'], weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[fit]),torch.from_numpy(y[fit])),
                        batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(42))
    loss_fn, history = nn.MSELoss(), []
    for epoch in range(1,run['selected_epochs']+1):
        model.train()
        total = 0.
        for inputs, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs),labels)
            if not torch.isfinite(loss): raise ValueError('损失不是有限值')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()*len(inputs)
        history.append({'epoch':epoch,'train_MSE':total/int(fit.sum())})
    model.eval()
    checkpoint = dict(model_name='LSTM',config=run['selected_config'],state_dict=model.state_dict(),
                      input_size=4,output_size=4,window=24,targets=KPIS,seed=42,
                      unit='z-score',data_X_sha256=spec['X_sha256'])
    torch.save(checkpoint,output/'lstm_seed_42.pt')
    pd.DataFrame(history).to_csv(output/'training_history.csv',index=False)
    evaluate(y[test],predict_history(model,x[test])).to_csv(output/'test_metrics.csv',index=False)


def train_detectors(output):
    for kpi in KPIS:
        folder = ANOMALY_ROOT/kpi
        frame = pd.read_csv(folder/'model_inputs.csv.gz')
        original = pd.read_csv(folder/'prepared_observations.csv.gz',usecols=['row_id','seasonal_residual'])
        frame = frame.merge(original,on='row_id',validate='one_to_one')
        fit = frame.loc[frame.split.eq('train')]
        cal = frame.loc[frame.split.eq('validation')]
        if fit.empty or cal.empty: raise ValueError('缺少独立的拟合或校准阶段')
        for method in METHODS:
            detector = fit_detector(method,fit.seasonal_residual__model_input,
                                    cal.seasonal_residual__model_input,
                                    fit.seasonal_residual,cal.seasonal_residual)
            if detector.status != 'ok': raise ValueError(f'{kpi}/{method}: {detector.status}')
            joblib.dump(detector,output/f'{kpi}_{method}.joblib',compress=3)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('module',choices=['lstm','detectors'])
    parser.add_argument('--output',type=Path,required=True,help='必须是尚不存在的新目录，不覆盖已交付模型')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (train_lstm if args.module=='lstm' else train_detectors)(args.output)
    print(f'已保存到 {args.output.resolve()}；Web 应用仍使用原始模型。')


if __name__=='__main__':
    main()
