"""Audit elapsed search time once per saved execution, without overlapping budget sums.

Run from the repository root: python report/scan_runtime.py
Reads results/runs only; writes auditable timing CSV/JSON under report/data.
These are summed recorded run-wall-times, not CPU hours or calendar duration.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json, csv, math, hashlib

ROOT=Path(__file__).resolve().parent;RUNS=ROOT.parent/'results/runs';OUT=ROOT/'data'
rows=[];issues=[];file_hashes={}
def number(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0
for path in sorted(RUNS.glob('*.json')):
    try:
        content=path.read_bytes();r=json.loads(content);file_hashes[path.name]=hashlib.sha256(content).hexdigest()
    except Exception as e:
        issues.append({'file':path.name,'error':str(e)});continue
    c=r.get('config',{});trials=r.get('trials',[]);summaries=r.get('summaries',[])
    cumulative=[t.get('elapsed_seconds') for t in trials if number(t.get('elapsed_seconds'))]
    totals=[s.get('optimization_seconds') for s in summaries if number(s.get('optimization_seconds'))]
    seconds=max(cumulative) if cumulative else (max(totals) if totals else None)
    if cumulative and totals and abs(max(cumulative)-max(totals))>1:
        issues.append({'file':path.name,'error':'Final cumulative trial and maximum summary time disagree by >1 second.'})
    if seconds is None:issues.append({'file':path.name,'error':'No recorded elapsed time; retained as unknown rather than zero.'})
    rows.append({'file':path.name,'run_id':r.get('run_id',path.stem),'status':r.get('status','unknown'),
        **{k:c.get(k) for k in ['dataset','condition','model','optimizer','seed','budget','source_hash','cv_jobs','n_jobs','model_threads']},
        'trials':len(trials),'budget_summaries':len(summaries),'recorded_seconds':seconds,
        'recorded_hours':None if seconds is None else seconds/3600,'timing_source':'trial_cumulative_max' if cumulative else ('summary_max' if totals else 'missing'),
        'mtime_ns':path.stat().st_mtime_ns,'sha256':file_hashes[path.name]})

def write(name,records):
    records=list(records)
    if not records:return
    keys=list(dict.fromkeys(k for r in records for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)

# Repeated filenames/run_ids are saved representations, not proof of new execution.
# Different run IDs remain distinct recorded attempts even when configuration repeats.
by_id={}
for row in sorted(rows,key=lambda r:r['mtime_ns']):by_id[row['run_id']]=row
executions=list(by_id.values())
duplicated_ids={k for k,n in Counter(r['run_id'] for r in rows).items() if n>1}
for key in duplicated_ids:
    copies=[r for r in rows if r['run_id']==key]
    assert len({r['sha256'] for r in copies})==1, 'Nonidentical duplicate run ID needs manual resolution: '+key
write('runtime_duplicate_files.csv',[{**r,'retained_file':by_id[r['run_id']]['file']} for r in rows if r['run_id'] in duplicated_ids])
key_fields=['source_hash','dataset','condition','model','optimizer','seed','budget']
by_combination={}
for row in sorted(executions,key=lambda r:r['mtime_ns']):
    if row['status']=='complete':by_combination[tuple(row[k] for k in key_fields)]=row
completed=[r for r in executions if r['status']=='complete']
manifest=json.loads((OUT/'snapshot_manifest.json').read_text())
primary_names=set(manifest['files']);primary=[r for r in executions if r['file'] in primary_names]

def total(records):return sum(r['recorded_seconds'] for r in records if r['recorded_seconds'] is not None)
def stats(records):
    return {'execution_records':len(records),'timed_records':sum(r['recorded_seconds'] is not None for r in records),
            'unknown_timing_records':sum(r['recorded_seconds'] is None for r in records),
            'complete_records':sum(r['status']=='complete' for r in records),
            'incomplete_records':sum(r['status']!='complete' for r in records),
            'seconds':total(records),'hours':total(records)/3600,'sequential_days_equivalent':total(records)/86400}

def grouped(field):
    groups=defaultdict(list)
    for r in executions:groups[r[field]].append(r)
    out=[]
    for label,group in sorted(groups.items()):
        cs=[r for r in group if r['status']=='complete'];ps=[r for r in group if r['file'] in primary_names]
        out.append({field:label,**stats(group),'complete_hours':total(cs)/3600,
            'incomplete_hours':total([r for r in group if r['status']!='complete'])/3600,
            'primary_hours':total(ps)/3600,'share_percent':100*total(group)/total(executions)})
    return out
write('runtime_all_files.csv',rows)
write('runtime_unique_executions.csv',executions)
write('runtime_by_dataset.csv',grouped('dataset'))
write('runtime_by_optimizer.csv',grouped('optimizer'))
write('runtime_by_model.csv',grouped('model'))
write('runtime_by_source.csv',grouped('source_hash'))
write('runtime_by_seed.csv',grouped('seed'))
cross=defaultdict(list)
for r in executions:cross[(r['dataset'],r['optimizer'],r['model'])].append(r)
write('runtime_dataset_optimizer_model.csv',({'dataset':k[0],'optimizer':k[1],'model':k[2],**stats(v)} for k,v in sorted(cross.items())))
issues_path=OUT/'runtime_scan_issues.json';issues_path.write_text(json.dumps(issues,indent=2)+'\n')
summary={'scan_time_utc':datetime.now(timezone.utc).isoformat(),'file_count':len(rows),'unreadable_files':len(list(RUNS.glob('*.json')))-len(rows),
    'status_counts':dict(Counter(r['status'] for r in rows)),'duplicate_run_id_files':len(rows)-len(executions),
    'all_recorded_executions':stats(executions),'completed_recorded_executions':stats(completed),
    'incomplete_recorded_executions':stats([r for r in executions if r['status']!='complete']),
    'completed_latest_per_cohort_combination':stats(list(by_combination.values())),
    'repeated_complete_combination_records':len(completed)-len(by_combination),
    'primary_paper_snapshot':stats(primary),'primary_missing_files':sorted(primary_names-{r['file'] for r in executions}),
    'timing_definition':'Maximum cumulative elapsed_seconds per execution (fallback: maximum optimization_seconds among budget summaries). Never sum cumulative trials or overlapping budget summaries.',
    'interpretation':'Aggregate recorded search-wall-time; not CPU time or observed calendar duration. Excludes unlogged data preparation, final refits, post-final-trial cleanup, and unfinished uncheckpointed work. Repeated configurations with distinct run IDs are retained as recorded attempts.',
    'hardware':{'processor':'Apple M4 Max','cpu_cores':16,'memory_gb':128,'source':'User-provided specification'},
    'file_sha256':file_hashes}
assert not summary['primary_missing_files']
for field in ['dataset','optimizer','model','source_hash','seed']:
    assert math.isclose(sum(r['seconds'] for r in grouped(field)),total(executions),abs_tol=1e-6)
(OUT/'runtime_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='file_sha256'},indent=2))
for field in ['dataset','optimizer','model']:
    print('\nBY',field)
    for r in grouped(field):print(r[field],f"{r['hours']:.6f} h",f"{r['share_percent']:.2f}%",'primary',f"{r['primary_hours']:.6f} h")
