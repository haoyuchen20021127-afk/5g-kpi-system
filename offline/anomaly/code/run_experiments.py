"""Run calibrated per-KPI detectors and supplementary record-level evaluation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from calibrated_detectors import CONFIG, METHODS, fit_detector, finite_quantile, compare_score
from reference_imputation import fit_reference_imputer, transform


def write_json(path, value):
    def convert(x):
        if isinstance(x,dict): return {str(k):convert(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [convert(v) for v in x]
        if isinstance(x,np.generic): return convert(x.item())
        if isinstance(x,float) and not np.isfinite(x): return None
        if isinstance(x,Path): return str(x)
        return x
    Path(path).write_text(json.dumps(convert(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def choose_reference(frame, labelled):
    reference = frame.label.eq(0) if labelled else pd.Series(True,index=frame.index)
    train = frame.split.eq('train') & reference
    cal = frame.split.eq('validation') & reference
    if cal.sum() >= CONFIG['min_cal']:
        return train, cal, 'original_validation_reference'
    ordered = np.sort(frame.loc[train,'time'].dropna().unique())
    if len(ordered)<2:
        return train, cal, 'no_separate_calibration'
    cutoff = ordered[min(int(len(ordered)*.75),len(ordered)-1)]
    cal = train & frame.time.ge(cutoff)
    fit = train & frame.time.lt(cutoff)
    return fit,cal,'last_quarter_of_original_training_reference_time'


def metrics(y,flag,score):
    y=np.asarray(y);flag=np.asarray(flag);score=np.asarray(score)
    eligible=np.isin(y,[0,1]) & np.isin(flag,[0,1])
    if not eligible.any():
        return dict(known_scored_n=0,tn=0,fp=0,fn=0,tp=0,benign_n=0,attack_n=0,
                    false_positive_rate=None,recall=None,precision=None,f1=None,
                    roc_auc=None,average_precision=None)
    a,b,c,d=confusion_matrix(y[eligible],flag[eligible],labels=[0,1]).ravel()
    div=lambda x,z: float(x/z) if z else None
    both=len(np.unique(y[eligible]))==2
    return dict(known_scored_n=int(eligible.sum()),tn=int(a),fp=int(b),fn=int(c),tp=int(d),
        benign_n=int(a+b),attack_n=int(c+d),false_positive_rate=div(b,a+b),recall=div(d,c+d),
        precision=div(d,b+d),f1=div(2*d,2*d+b+c),
        roc_auc=float(roc_auc_score(y[eligible],score[eligible])) if both else None,
        average_precision=float(average_precision_score(y[eligible],score[eligible])) if both else None)


def calibration_percentile(reference, values):
    a=np.sort(np.asarray(reference,dtype=float))
    a=a[np.isfinite(a)]
    v=np.asarray(values,dtype=float)
    out=np.full(v.shape,np.nan)
    if len(a):
        valid=np.isfinite(v)
        out[valid]=np.searchsorted(a,v[valid],side='left')/len(a)
    return out


def run_task(task,out_root,reuse_models=None,impute_missing=False):
    started=time.perf_counter()
    dataset,task_id=task['dataset'],task['task_id']
    folder=out_root/'datasets'/dataset/task_id
    if (folder/'completion.json').exists():
        print(f'SKIP completed {dataset}/{task_id}',flush=True)
        return
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'models').mkdir(exist_ok=True)
    frame=task['frame'].copy().reset_index(drop=True)
    features=task['features']
    reusable={}
    if reuse_models is not None and impute_missing:
        raise ValueError('Changed model inputs require refitting; cached detectors may not be reused.')
    if reuse_models is not None:
        old=pd.read_csv(reuse_models/'kpi_summary.csv',float_precision='round_trip')
        for item in old.to_dict('records'):
            if item['method']!='iqr' and item['status']=='ok':
                reusable[(str(item['context']),item['feature'],item['method'])]=reuse_models/item['model_file']
    assert not frame.row_id.duplicated().any(),f'{dataset}/{task_id}: duplicate row_id'
    assert all(f not in ['label','time','row_id','context'] for f in features)
    labelled=task['label_scope']=='attack_state'
    frame['phase']='not_used_for_fitting'
    frame.loc[frame.split.eq('test'),'phase']='test'
    test=frame[frame.split.eq('test')].copy()
    test['test_order']=np.arange(len(test))
    calparts=[]; summary=[]; shift=[]; assignments=[]; row_thresholds=[]; imputation=[]
    model_inputs={}
    # Score columns are built once to avoid fragmented DataFrames.
    extra={}
    for feature in features:
        if impute_missing:
            extra[f'{feature}__model_input']=np.full(len(test),np.nan)
            extra[f'{feature}__input_imputed']=np.zeros(len(test),dtype=bool)
            extra[f'{feature}__input_missing_before']=~np.isfinite(test[feature].to_numpy(float))
            model_inputs[f'{feature}__model_input']=np.full(len(frame),np.nan)
            model_inputs[f'{feature}__input_imputed']=np.zeros(len(frame),dtype=bool)
        for method in METHODS:
            extra[f'{feature}__{method}__score']=np.full(len(test),np.nan)
            extra[f'{feature}__{method}__flag']=np.full(len(test),-1,dtype=np.int8)
    for method in METHODS:
        extra[f'row__{method}__score']=np.full(len(test),np.nan)
        extra[f'row__{method}__flag']=np.full(len(test),-1,dtype=np.int8)
    test=pd.concat([test.reset_index().rename(columns={'index':'frame_index'}),pd.DataFrame(extra)],axis=1)
    for context, sub in frame.groupby('context',sort=True,dropna=False):
        fitmask,calmask,policy=choose_reference(sub,labelled)
        fit=sub.loc[fitmask];cal=sub.loc[calmask]
        frame.loc[fit.index,'phase']='fit_reference'
        frame.loc[cal.index,'phase']='calibration_reference'
        tpos=test.index[test.context.eq(context)]
        ts=test.loc[tpos]
        cp=cal.copy()
        cp_extra={}
        cscore={m:[] for m in METHODS};tscore={m:[] for m in METHODS}
        assignments.append(dict(context=str(context),fit_n=len(fit),calibration_n=len(cal),test_n=len(ts),policy=policy,
            fit_time_min=fit.time.min(),fit_time_max=fit.time.max(),cal_time_min=cal.time.min(),cal_time_max=cal.time.max(),
            fit_label_counts=fit.label.value_counts().to_dict(),cal_label_counts=cal.label.value_counts().to_dict()))
        for feature in features:
            f=fit[feature].to_numpy(float);c=cal[feature].to_numpy(float);x=ts[feature].to_numpy(float)
            f_native,c_native,x_native=f.copy(),c.copy(),x.copy()
            imputed_test=np.zeros(len(x),dtype=bool)
            if impute_missing:
                params=fit_reference_imputer(f_native,feature=feature)
                f,f_missing=transform(f_native,params)
                c,c_missing=transform(c_native,params)
                x,x_missing=transform(x_native,params)
                imputed_test=x_missing & np.isfinite(x)
                all_input,all_missing=transform(sub[feature].to_numpy(float),params)
                model_inputs[f'{feature}__model_input'][sub.index]=all_input
                model_inputs[f'{feature}__input_imputed'][sub.index]=all_missing & np.isfinite(all_input)
                test.loc[tpos,f'{feature}__model_input']=x
                test.loc[tpos,f'{feature}__input_imputed']=imputed_test
                cp_extra[f'{feature}__model_input']=c
                cp_extra[f'{feature}__input_imputed']=c_missing & np.isfinite(c)
                cp_extra[f'{feature}__input_missing_before']=c_missing
                imputation.append(dict(context=str(context),feature=feature,**params,
                    calibration_total_n=len(c),calibration_observed_n=int(np.isfinite(c_native).sum()),
                    calibration_imputed_n=int((c_missing & np.isfinite(c)).sum()),
                    test_total_n=len(x),test_observed_n=int(np.isfinite(x_native).sum()),
                    test_imputed_n=int(imputed_test.sum()),test_unresolved_n=int((~np.isfinite(x)).sum()),
                    applied_after_feature_construction=True))
            ff=f_native[np.isfinite(f_native)];xx=x_native[np.isfinite(x_native)]
            benign=ts.loc[ts.label.eq(0),feature].dropna()
            shift.append(dict(context=str(context),feature=feature,fit_observed=len(ff),test_observed=len(xx),
                fit_median=np.median(ff) if len(ff) else None,test_median=np.median(xx) if len(xx) else None,
                benign_test_median=float(benign.median()) if len(benign) else None,
                test_outside_fit_range_fraction=float(((xx<ff.min())|(xx>ff.max())).mean()) if len(ff) and len(xx) else None,
                fit_unique=len(np.unique(ff))))
            for method in METHODS:
                cached=reusable.get((str(context),feature,method))
                detector=joblib.load(cached) if cached else fit_detector(method,f,c,feature=feature,
                    observed_fit=f_native,observed_calibration=c_native)
                cs=detector.score(c);scores=detector.score(x)
                comparison_method=detector.comparison_method()
                cf=compare_score(cs,detector.threshold,comparison_method)[0]
                flags=compare_score(scores,detector.threshold,comparison_method)[0]
                sc=f'{feature}__{method}__score';fc=f'{feature}__{method}__flag'
                test.loc[tpos,sc]=scores;test.loc[tpos,fc]=flags
                cp_extra[sc]=cs;cp_extra[fc]=cf
                if method=='iqr':
                    base_test=compare_score(scores,0.,method)[0]
                    base_cal=compare_score(cs,0.,method)[0]
                    test.loc[tpos,f'{feature}__iqr__base_flag']=base_test
                    cp_extra[f'{feature}__iqr__base_flag']=base_cal
                valid=flags>=0
                info=dict(dataset=dataset,task=task_id,context=str(context),feature=feature,method=method,
                    status=detector.status,reference_policy=policy,total_test_n=len(ts),test_scored_n=int(valid.sum()),
                    test_flag_n=int((flags==1).sum()),test_flag_fraction=float((flags[valid]==1).mean()) if valid.any() else None,
                    test_coverage_fraction=float(valid.mean()) if len(valid) else None,
                    **{k:v for k,v in (detector.metadata or {}).items() if k!='feature'})
                info.update(test_observed_scored_n=int((valid & np.isfinite(x_native)).sum()),
                    test_imputed_scored_n=int((valid & imputed_test).sum()),
                    test_imputed_flag_n=int(((flags==1) & imputed_test).sum()),
                    input_imputation=params['strategy'] if impute_missing else 'none')
                for label,name in [(0,'benign'),(1,'attack')]:
                    m=(ts.label.to_numpy()==label)&valid
                    info[f'{name}_test_scored_n']=int(m.sum())
                    info[f'{name}_test_flag_n']=int((flags[m]==1).sum())
                summary.append(info)
                if method=='iqr':
                    info['base_test_flag_n']=int((base_test==1).sum())
                    info['base_test_flag_fraction']=float((base_test[valid]==1).mean()) if valid.any() else None
                if detector.status=='ok':
                    cscore[method].append(cs);tscore[method].append(scores)
                    key=hashlib.sha256(f'{context}|{feature}|{method}'.encode()).hexdigest()[:18]
                    joblib.dump(detector,folder/'models'/f'{key}.joblib',compress=3)
                    info['model_file']=f'models/{key}.joblib'
        # Supplementary record-level threshold is fitted separately; no uncalibrated
        # OR over many per-KPI flags. Only complete scored vectors are evaluated.
        for method in METHODS:
            if not cscore[method]: continue
            # Method scores from separate KPI fits have different scales. Compare
            # empirical calibration percentiles before taking a row maximum.
            cm=np.column_stack([calibration_percentile(c,c) for c in cscore[method]])
            tm=np.column_stack([calibration_percentile(c,t) for c,t in zip(cscore[method],tscore[method])])
            cg=np.isfinite(cm).all(axis=1);tg=np.isfinite(tm).all(axis=1)
            cs=np.full(len(cal),np.nan);ss=np.full(len(ts),np.nan)
            cs[cg]=cm[cg].max(axis=1);ss[tg]=tm[tg].max(axis=1)
            if cg.sum()<CONFIG['min_cal']: continue
            threshold=finite_quantile(cs[cg],CONFIG['alpha'])
            flag,guard=compare_score(ss,threshold,method)
            test.loc[tpos,f'row__{method}__score']=ss
            test.loc[tpos,f'row__{method}__flag']=flag
            cp_extra[f'row__{method}__score']=cs
            cp_extra[f'row__{method}__flag']=compare_score(cs,threshold,method)[0]
            row_thresholds.append(dict(context=str(context),method=method,features_used=len(cscore[method]),
                calibration_complete_n=int(cg.sum()),threshold=threshold,numerical_guard=guard,
                test_complete_n=int(tg.sum()),calibration_flag_fraction=float((compare_score(cs[cg],threshold,method)[0]==1).mean())))
        cp=pd.concat([cp.reset_index(drop=True),pd.DataFrame(cp_extra)],axis=1)
        calparts.append(cp)
        print(f'  {dataset}/{task_id} context={str(context)[:75]} fit={len(fit)} cal={len(cal)} test={len(ts)}',flush=True)
    frame.to_csv(folder/'prepared_observations.csv.gz',index=False,compression='gzip')
    if impute_missing:
        identifiers=[c for c in ['row_id','source','context','session','split','phase','time','label'] if c in frame]
        pd.concat([frame[identifiers],pd.DataFrame(model_inputs)],axis=1).to_csv(
            folder/'model_inputs.csv.gz',index=False,compression='gzip')
        write_json(folder/'imputation_parameters.json',imputation)
    test.to_csv(folder/'test_predictions.csv.gz',index=False,compression='gzip')
    calibration=pd.concat(calparts,ignore_index=True) if calparts else pd.DataFrame()
    calibration.to_csv(folder/'calibration_predictions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(summary).to_csv(folder/'kpi_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(shift).to_csv(folder/'reference_shift.csv',index=False,encoding='utf-8-sig')
    write_json(folder/'reference_assignments.json',assignments)
    write_json(folder/'row_thresholds.json',row_thresholds)
    records=[]
    if labelled:
        for group_name,mask in [('all',np.ones(len(test),dtype=bool))]+[(f'source:{s}',test.source.eq(s).to_numpy()) for s in test.source.unique()]:
            for method in METHODS:
                g=test.loc[mask]
                records.append(dict(dataset=dataset,task=task_id,group=group_name,method=method,
                    total_test_n=len(g),metric_scope='record_attack_state_not_KPI_truth',
                    **metrics(g.label,g[f'row__{method}__flag'],g[f'row__{method}__score'])))
    for record in records:
        record['benign_label_flag_fraction']=record['false_positive_rate']
        record['attack_label_flag_fraction']=record['recall']
        record['health_truth_available']=False
    pd.DataFrame(records).to_csv(folder/'attack_metrics.csv',index=False,encoding='utf-8-sig')
    # Candidate associations are calculated within a context, never declared causes.
    corr=[]
    for ctx,g in test.groupby('context',sort=False):
        matrix=g[features].corr(method='spearman',min_periods=30)
        for i,a in enumerate(features):
            for b in features[i+1:]:
                if np.isfinite(matrix.loc[a,b]):
                    corr.append(dict(context=str(ctx),kpi_a=a,kpi_b=b,spearman_r=float(matrix.loc[a,b]),
                        paired_n=int(g[[a,b]].notna().all(axis=1).sum()),interpretation='association_not_causation'))
    pd.DataFrame(corr).to_csv(folder/'kpi_associations.csv',index=False,encoding='utf-8-sig')
    note={k:v for k,v in task.items() if k!='frame'}
    note.update(config=CONFIG,methods=METHODS,rows_prepared=len(frame),rows_test=len(test),
        rows_calibration=len(calibration),seconds=round(time.perf_counter()-started,2),
        interpretation='Retrospective calibrated exploration; calibration fraction is not independent performance or a guaranteed future false-positive rate.',
        joint_rule='maximum per-KPI empirical calibration percentile; separate calibration threshold, complete-case vectors only; not a formal simultaneous error guarantee')
    note['label_interpretation']='benign and attack state are not per-KPI health truth; false_positive_rate in supplementary metrics is attack-label-relative only'
    note['input_imputation']=dict(enabled=impute_missing,rule='fit-reference mode for CQI/RI/pcmax; median for other KPIs; per-context',
        original_observations_preserved=True,clipping=False,threshold_evidence_counts='observed before imputation',
        prediction_value_columns='feature__model_input',imputation_flag_columns='feature__input_imputed',
        no_fit_observations='leave missing and unscored; do not borrow calibration/test values',
        qualification='Predictions on imputed values are model outputs for estimates, not measured KPI health.')
    write_json(folder/'task_manifest.json',note)
    assert set(frame.loc[frame.phase.eq('fit_reference'),'row_id']).isdisjoint(frame.loc[frame.phase.eq('calibration_reference'),'row_id'])
    assert set(frame.loc[frame.phase.isin(['fit_reference','calibration_reference']),'row_id']).isdisjoint(test.row_id)
    write_json(folder/'completion.json',dict(status='complete',rows_prepared=len(frame),rows_test=len(test),
        fit_calibration_test_row_ids_disjoint=True,source_files_unchanged=True,seconds=note['seconds']))
    print(f'DONE {dataset}/{task_id}: {len(features)} KPI; test={len(test)}; {note["seconds"]}s',flush=True)
