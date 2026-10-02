"""Read-only audit of the frozen validation outputs. Does not import GPU code."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sqlite3
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
REF = 'stageA-b12-identity'
BEST = 'stageA-b9-identity'


def records(df):
    return df.to_dict(orient='records')


def wilson(f, n):
    z = 1.959963984540054
    p = f/n
    d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h, c+h]


def stats(g):
    t = g.wall_seconds
    return dict(n=len(g), mean=float(t.mean()), median=float(t.median()),
                p90=float(t.quantile(.9)), p95=float(t.quantile(.95)), p99=float(t.quantile(.99)),
                maximum=float(t.max()), hours=float(t.sum()/3600),
                oracle_mean=float(g.oracle_seconds.mean()), oracle_share=float(g.oracle_seconds.sum()/t.sum()),
                calls_mean=float(g.oracle_calls.mean()), calls_max=int(g.oracle_calls.max()),
                calls_p99=float(g.oracle_calls.quantile(.99)), two_calls=int((g.oracle_calls==2).sum()),
                calls_hist={str(k):int(v) for k,v in g.oracle_calls.value_counts().sort_index().items()},
                leaves_mean=float(g.leaves_evaluated.mean()), leaves_max=int(g.leaves_evaluated.max()),
                failure=int((g.logical_success==0).sum()), ler=float(1-g.logical_success.mean()),
                ler_wilson95=wilson(int((g.logical_success==0).sum()),len(g)),
                clipped_shots=int((g.clipped_negative_entries>0).sum()),
                clipped_entries=int(g.clipped_negative_entries.sum()),
                min_mass=float(g.most_negative_raw_mass.min()),
                max_negative_root_ratio=float((-g.most_negative_raw_mass/g.root_mass).max()),
                max_guard_root_ratio=float((g.max_local_guard_abs/g.root_mass).max()))


def main():
    all_frames=[]; audits=[]; groups=[]; consist=[]; exceptions=[]; mismatches=[]; dsinfo=[]
    metas=[]; csv_issues=[]; hashes={}; stress_arrays=[]
    for folder in sorted(ROOT.glob('validation_*')):
        if not folder.is_dir():
            continue
        db=folder/'results.sqlite'
        # immutable is safe for these copied snapshots only if no live WAL exists.
        if Path(str(db)+'-wal').exists():
            raise RuntimeError(f'WAL exists: {db}. Take a consistent snapshot first.')
        hashes[str(db.relative_to(ROOT))]=hashlib.sha256(db.read_bytes()).hexdigest()
        conn=sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True)
        integrity=conn.execute('PRAGMA integrity_check').fetchall()
        frame=pd.read_sql_query('SELECT * FROM shots',conn)
        meta=pd.read_sql_query('SELECT * FROM layout_meta',conn)
        conn.close()
        manifest=json.loads((folder/'run_manifest.json').read_text())
        dsmanifest=json.loads((folder/'datasets/dataset_manifest.json').read_text())
        issues=[]
        if dsmanifest!=manifest['datasets']: issues.append('dataset manifest mismatch')
        settings=manifest['datasets']
        expected={}
        for p in settings['p_values']:
            expected['natural_p'+format(p,'.6g').replace('.','p')]=(settings['natural_shots_per_p'],p)
        expected['stress_w'+'-'.join(map(str,settings['stress_weights']))+'_p'+format(settings['stress_p'],'.6g').replace('.','p')]=(settings['stress_shots'],settings['stress_p'])
        datasets={}
        for name,(count,p) in expected.items():
            with np.load(folder/'datasets'/f'{name}.npz',allow_pickle=False) as arr:
                data={key:arr[key].copy() for key in arr.files}
            datasets[name]=data
            shapes={'errors':(count,72),'syndromes':(count,36),'true_lambdas':(count,12),'error_weights':(count,),'syndrome_weights':(count,)}
            for key,shape in shapes.items():
                if data[key].shape!=shape: issues.append(f'{name}/{key}: shape {data[key].shape} != {shape}')
            if not np.array_equal(data['errors'].sum(axis=1),data['error_weights']): issues.append(f'{name}: error weights inconsistent')
            if not np.array_equal(data['syndromes'].sum(axis=1),data['syndrome_weights']): issues.append(f'{name}: syndrome weights inconsistent')
            if count:
                dsinfo.append(dict(run=folder.name,dataset=name,p=p,n=count,
                    unique_syndromes=int(len(np.unique(data['syndromes'],axis=0))),
                    unique_errors=int(len(np.unique(data['errors'],axis=0))),
                    zero_syndromes=int((data['syndrome_weights']==0).sum()),
                    mean_weight=float(data['error_weights'].mean()),
                    weights={str(k):int(v) for k,v in zip(*np.unique(data['error_weights'],return_counts=True))},
                    errors_sha256=hashlib.sha256(data['errors'].tobytes()).hexdigest()))
                if name.startswith('stress'): stress_arrays.append(data)
        expected_rows=sum(c for c,p in expected.values())*len(manifest['layout_tags'])
        if set(frame.layout_tag)!=set(manifest['layout_tags']): issues.append('layout tags differ')
        if frame.duplicated(['layout_tag','dataset_name','shot_id']).any(): issues.append('duplicate keys')
        num=frame.select_dtypes('number')
        if not np.isfinite(num.to_numpy()).all(): issues.append('nonfinite numeric values')
        if frame.isna().any().any(): issues.append('NULL values')
        if ((frame.wall_seconds<=0)|(frame.oracle_seconds<0)|(frame.oracle_seconds>frame.wall_seconds)|(frame.root_mass<=0)).any(): issues.append('invalid times or mass')
        if not ((frame.best_probability>=0)&(frame.best_probability<=1+1e-9)).all(): issues.append('probability outside tolerance')
        if not frame.best_lambda.str.fullmatch('[01]{12}').all(): issues.append('invalid best lambda')
        if not np.array_equal((frame.best_lambda==frame.true_lambda).astype(int),frame.logical_success): issues.append('success flag inconsistent')
        if not (frame.leaves_evaluated+frame.pruned_leaves==4096).all(): issues.append('leaf accounting inconsistent')
        if not (frame.oracle_calls==frame.expanded_internal_nodes).all(): issues.append('node accounting inconsistent')
        topology=[json.loads(s) for s in frame.topology_call_counts]
        if not np.array_equal([sum(v.values()) for v in topology],frame.oracle_calls): issues.append('topology calls inconsistent')
        if len(frame)!=expected_rows: issues.append('total row count differs')
        for tag in manifest['layout_tags']:
            for name,(count,p) in expected.items():
                g=frame[(frame.layout_tag==tag)&(frame.dataset_name==name)].sort_values('shot_id')
                if not np.array_equal(g.shot_id.to_numpy(),np.arange(count)):
                    issues.append(f'{tag}/{name}: missing or extra shot ids')
                    continue
                if not count: continue
                data=datasets[name]
                truth=[''.join(map(str,row)) for row in data['true_lambdas']]
                if not np.array_equal(g.true_lambda.to_numpy(),truth): issues.append(f'{tag}/{name}: true lambda mismatch')
                if not np.array_equal(g.error_weight,data['error_weights']): issues.append(f'{tag}/{name}: recorded weight mismatch')
                if not np.array_equal(g.syndrome_weight,data['syndrome_weights']): issues.append(f'{tag}/{name}: recorded syndrome weight mismatch')
                if not (g.p==p).all(): issues.append(f'{tag}/{name}: p mismatch')
                groups.append(dict(run=folder.name,kind='natural' if name.startswith('natural') else 'stress',tag=tag,dataset=name,p=p,**stats(g)))
        unknown=set(frame.dataset_name)-set(expected)
        if unknown: issues.append(f'unknown datasets: {unknown}')
        audit=dict(run=folder.name,expected=expected_rows,actual=len(frame),integrity=integrity,issues=issues,
                   first_utc=frame.created_at.min(),last_utc=frame.created_at.max(),
                   online_hours=float(frame.wall_seconds.sum()/3600),manifest=manifest)
        audits.append(audit)
        # Recompute every field in the existing summaries from the database.
        existing=pd.read_csv(folder/'summary.csv')
        wanted=[r for r in groups if r['run']==folder.name]
        if len(existing)!=len(wanted): csv_issues.append(f'{folder.name}: summary row count')
        for r in wanted:
            e=existing[(existing.layout_tag==r['tag'])&(existing.dataset_name==r['dataset'])]
            if len(e)!=1:
                csv_issues.append(f'{folder.name}: summary key {r["tag"]}/{r["dataset"]}')
                continue
            values=dict(completed=r['n'],mean_wall_s=r['mean'],median_wall_s=r['median'],p90_wall_s=r['p90'],p95_wall_s=r['p95'],p99_wall_s=r['p99'],mean_oracle_s=r['oracle_mean'],mean_oracle_calls=r['calls_mean'],mean_leaves=r['leaves_mean'],logical_success_rate=1-r['ler'],logical_error_rate=r['ler'])
            for key,value in values.items():
                if not math.isclose(float(e.iloc[0][key]),value,rel_tol=1e-10,abs_tol=1e-12): csv_issues.append(f'{folder.name}/{r["tag"]}/{r["dataset"]}: {key}')
        ref=frame[frame.layout_tag==REF]
        cx=pd.read_csv(folder/'consistency_vs_b12.csv')
        for tag in manifest['layout_tags']:
            if tag==REF: continue
            m=frame[frame.layout_tag==tag].merge(ref,on=['dataset_name','shot_id'],suffixes=('','_ref'),validate='one_to_one')
            m['gap']=(m.best_probability-m.best_probability_ref).abs()
            m['same']=m.best_lambda==m.best_lambda_ref
            def comp(g):
                return dict(n=len(g),same=int(g.same.sum()),different=int((~g.same).sum()),
                            probability_not_close=int((g.gap>1e-10).sum()),max_probability_gap=float(g.gap.max()),
                            max_root_relative_gap=float(((g.root_mass-g.root_mass_ref).abs()/g.root_mass_ref).max()),
                            candidate_only_success=int(((g.logical_success==1)&(g.logical_success_ref==0)).sum()),
                            reference_only_success=int(((g.logical_success==0)&(g.logical_success_ref==1)).sum()),
                            both_fail_different=int(((g.logical_success==0)&(g.logical_success_ref==0)&~g.same).sum()))
            c=comp(m)
            check=cx[cx.layout_tag==tag]
            values={'overlap_shots':c['n'],'same_lambda':c['same'],'same_lambda_fraction':c['same']/c['n'], 'prob_close_1e-10':c['n']-c['probability_not_close'],'prob_close_fraction':1-c['probability_not_close']/c['n']}
            if len(check)!=1 or any(not math.isclose(float(check.iloc[0][k]),v,rel_tol=1e-10,abs_tol=1e-12) for k,v in values.items()): csv_issues.append(f'{folder.name}: consistency {tag}')
            for name,g in m.groupby('dataset_name'):
                consist.append(dict(run=folder.name,tag=tag,dataset=name,p=float(g.p.iloc[0]),kind='natural' if name.startswith('natural') else 'stress',**comp(g)))
            cols=['dataset_name','shot_id','p','error_weight','syndrome_weight','best_lambda','best_lambda_ref','true_lambda','best_probability','best_probability_ref','gap','root_mass','root_mass_ref','logical_success','logical_success_ref','oracle_calls','leaves_evaluated','clipped_negative_entries','most_negative_raw_mass','max_local_guard_abs','created_at']
            exceptions.extend([dict(run=folder.name,tag=tag,**v) for v in records(m[m.gap>1e-10][cols])])
            mismatches.extend([dict(run=folder.name,tag=tag,**v) for v in records(m[~m.same][cols])])
        meta['run']=folder.name
        metas.extend(records(meta))
        frame['run']=folder.name
        frame['kind']=np.where(frame.dataset_name.str.startswith('natural'),'natural','stress')
        all_frames.append(frame)
        print(folder.name,len(frame),'issues',len(issues),flush=True)
    all_data=pd.concat(all_frames,ignore_index=True)
    aggregate=[]
    for (kind,tag),g in all_data.groupby(['kind','layout_tag']):
        aggregate.append(dict(kind=kind,tag=tag,**stats(g)))
    weight_stats=[]
    for (kind,p,tag,w),g in all_data.groupby(['kind','p','layout_tag','error_weight']):
        if tag in (BEST,REF): weight_stats.append(dict(kind=kind,p=p,tag=tag,weight=int(w),**stats(g)))
    slowest=records(all_data.nlargest(12,'wall_seconds')[['run','layout_tag','dataset_name','shot_id','wall_seconds','oracle_calls','error_weight','syndrome_weight','created_at']])
    # B9 second-block counts are oracle_calls - 1; all runs use exactly two blocks.
    stress_equal=all(all(np.array_equal(a[k],stress_arrays[0][k]) for k in a) for a in stress_arrays)
    report=dict(source=str(ROOT),audits=audits,groups=groups,aggregate=aggregate,
                consistency=consist,probability_exceptions=exceptions,sector_mismatch_count=len(mismatches),
                csv_issues=csv_issues,datasets=dsinfo,stress_arrays_identical=stress_equal,
                layout_meta=metas,weight_stats=weight_stats,slowest=slowest,sha256=hashes,
                total_rows=len(all_data),total_online_hours=float(all_data.wall_seconds.sum()/3600))
    (OUT/'audit_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (OUT/'sector_mismatches.json').write_text(json.dumps(mismatches,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(rows=len(all_data),hours=report['total_online_hours'],csv_issues=csv_issues,
                         sector_mismatches=len(mismatches),probability_exceptions=len(exceptions)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
