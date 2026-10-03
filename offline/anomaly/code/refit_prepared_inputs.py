"""Fit the detectors from preserved, pre-imputation KPI observations."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from run_experiments import run_task, write_json


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-results',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',help='Optional dataset/task for an individual run')
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.prepared_results.resolve()):
        raise ValueError('Output must be a NEW directory outside packaged results.')
    args.output.mkdir(parents=True,exist_ok=False)
    code=Path(__file__).parent
    code_hashes={name:sha(code/name) for name in ['run_experiments.py','calibrated_detectors.py','reference_imputation.py']}
    sources=[]
    for manifest_path in sorted((args.prepared_results/'datasets').glob('*/*/task_manifest.json')):
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        key=manifest['dataset']+'/'+manifest['task_id']
        if args.task and args.task != key:
            continue
        source=manifest_path.parent/'prepared_observations.csv.gz'
        source_sha=sha(source)
        folder=args.output/'datasets'/manifest['dataset']/manifest['task_id']
        provenance=folder/'refit_provenance.json'
        entry=dict(task=key,source=str(source.resolve()),source_sha256=source_sha,
                   manifest_sha256=sha(manifest_path),code_sha256=code_hashes)
        if (folder/'completion.json').exists():
            if not provenance.exists() or json.loads(provenance.read_text(encoding='utf-8'))!=entry:
                raise ValueError(f'{key}: completed output has incompatible provenance')
            sources.append(entry)
            print(f'SKIP verified {key}',flush=True)
            continue
        frame=pd.read_csv(source,float_precision='round_trip',dtype={
            'row_id':str,'context':str,'session':str,'source':str,'split':str})
        task={k:manifest[k] for k in ['dataset','task_id','features','group_col','label_scope','source_coverage']}
        task['notes']=manifest.get('notes',[])
        task['quality']=manifest.get('quality',[])
        task['frame']=frame
        run_task(task,args.output,impute_missing=True)
        assert sha(source)==source_sha, 'Source observations changed during fitting'
        write_json(provenance,entry)
        sources.append(entry)
    if not sources:
        raise ValueError('No selected tasks found')
    write_json(args.output/('refit_inventory.json' if not args.task else 'selected_refit_inventory.json'),
               dict(tasks=sources,task_n=len(sources),imputation='fit-reference mode for CQI/RI/pcmax, median otherwise',
                    preserves_finite_values=True,source_values_unchanged=True))


if __name__=='__main__':
    main()
