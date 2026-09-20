#!/usr/bin/env python3
import json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np,pandas as pd,torch
from rdkit import RDLogger
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'src'));O=R/'results/repeat_shift_invariance';SEED=20260920
from materials_adv.robustness.repeat_shift import generate_repeat_shift,tokenizer_roundtrip
from materials_adv.data.tokenizer import tokenize
from materials_adv.data.graph_dataset import GraphDataset
from materials_adv.data.scaler import TargetScaler
from materials_adv.models.graph_predictor import GraphPredictor
from scripts.run_sequence_equivalence_audit import load_frozen_model
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
def graph_predict(m,s,texts):
 ds=GraphDataset(pd.DataFrame({'original_representation':texts,'property_value':0.}),strict=True);out=[]
 with torch.no_grad():
  for b in DataLoader(ds,batch_size=32):out.extend(s.inverse_transform(m(b['x'],b['adj'],b['mask']).view(-1).numpy().reshape(-1,1)).ravel())
 return np.asarray(out)
def displacement(a,b):
 x,y=tokenize(a),tokenize(b);p=defaultdict(list);q=defaultdict(list)
 for i,t in enumerate(x):p[t].append(i)
 for i,t in enumerate(y):q[t].append(i)
 z=[abs(i-j) for t in set(p)&set(q) for i,j in zip(p[t],q[t])];return np.mean(z) if z else np.nan
def main():
 O.mkdir(parents=True,exist_ok=True);model,pred,vocab=load_frozen_model();s=TargetScaler.load(R/'results/models/transformer_regressor/scaler.json');g=GraphPredictor(7,64,3);g.load_state_dict(torch.load(R/'results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt',weights_only=True));g.eval()
 cohort=json.loads((R/'results/representation_equivalence_sequence/cohort_manifest.json').read_text())['sources'];apps=[];rows=[];excluded=[]
 for c in cohort:
  vs=generate_repeat_shift(c['representation'],cap=20);valid=[]
  for j,v in enumerate(vs):
   rt=tokenizer_roundtrip(v.variant,vocab);rec={'source_id':c['source_id'],'variant_id':f"{c['source_id']}_rs{j:03d}",'family':v.transform_family,'source_representation':v.source,'variant_representation':v.variant,'repeat_factor':v.transform_parameters.get('repeat_factor',1),'shift_offset':v.transform_parameters.get('shift_offset',0),'orientation':v.transform_parameters.get('orientation','forward'),'equivalence_pass':v.equivalence_status=='VERIFIED_EQUIVALENT','validation_reason':v.equivalence_reason,'token_count_source':len(tokenize(v.source)),'token_count_variant':rt['token_count'],'unknown_count':rt['unknown_count'],'seed':SEED,'position_displacement':displacement(v.source,v.variant)}
   if rec['equivalence_pass'] and rt['roundtrip'] and rt['unknown_count']==0 and rt['within_max_length']:valid.append(rec)
   else:excluded.append(rec)
  rows+=valid;apps.append({'source_id':c['source_id'],'repeat_shift_applicable':bool(valid),'reason':'verified transforms generated' if valid else 'no valid linear repeat/shift transform','num_possible_transforms':len(valid)})
 pd.DataFrame(apps).to_csv(O/'cohort_applicability.csv',index=False);pd.DataFrame(excluded).to_csv(O/'excluded_variants.csv',index=False);v=pd.DataFrame(rows);v.to_csv(O/'variants.csv',index=False)
 texts=[x for r in v.itertuples() for x in (r.source_representation,r.variant_representation)];p=pred.predict(texts);out=v[['source_id','variant_id','family']].copy();out['prediction_source']=p[::2];out['prediction_variant']=p[1::2];out['signed_drift']=out.prediction_variant-out.prediction_source;out['absolute_drift']=out.signed_drift.abs();out.to_csv(O/'predictions.csv',index=False);v=v.merge(out,on=['source_id','variant_id','family'])
 gp=graph_predict(g,s,texts);gc=out[['source_id','variant_id','family']].copy();gc['graph_prediction_source']=gp[::2];gc['graph_prediction_variant']=gp[1::2];gc['graph_absolute_drift']=(gc.graph_prediction_variant-gc.graph_prediction_source).abs();gc.to_csv(O/'graph_control.csv',index=False)
 sm=[]
 for sid,q in v.groupby('source_id'):
  d=q.absolute_drift;allp=np.r_[q.prediction_source.iloc[0],q.prediction_variant];sm.append({'source_id':sid,'n_variants':len(q),'max_drift':d.max(),'mean_drift':d.mean(),'median_drift':d.median(),'prediction_std':allp.std(),'prediction_range':allp.max()-allp.min()})
 sm=pd.DataFrame(sm);sm.to_csv(O/'source_metrics.csv',index=False);fm=[]
 rng_boot=np.random.default_rng(SEED);boot=[];source_worst=sm.max_drift.to_numpy()
 for i in range(5000):
  sample=rng_boot.choice(source_worst,len(source_worst),replace=True)
  boot.append({'replicate':i,'mean_source_worst':sample.mean(),'median_source_worst':np.median(sample),'P90_source_worst':np.quantile(sample,.9)})
 boot=pd.DataFrame(boot);boot.to_csv(O/'source_bootstrap.csv',index=False)
 for f,q in v.groupby('family'):
  w=q.groupby('source_id').absolute_drift.max();fm.append({'family':f,'n_sources':q.source_id.nunique(),'n_variants':len(q),'mean_drift':q.absolute_drift.mean(),'median_drift':q.absolute_drift.median(),'P95':q.absolute_drift.quantile(.95),'max':q.absolute_drift.max(),'source_worst_mean':w.mean(),'source_worst_median':w.median(),'source_worst_P95':w.quantile(.95)})
 pd.DataFrame(fm).to_csv(O/'family_results.csv',index=False)
 graph_family=[]
 for f,q in gc.groupby('family'):
  graph_family.append({'family':f,'n_variants':len(q),'mean_graph_drift':q.graph_absolute_drift.mean(),'median_graph_drift':q.graph_absolute_drift.median(),'P95_graph_drift':q.graph_absolute_drift.quantile(.95),'max_graph_drift':q.graph_absolute_drift.max()})
 pd.DataFrame(graph_family).to_csv(O/'graph_family_results.csv',index=False)
 broader=pd.read_csv(R/'results/representation_equivalence_validation/token_pair_audit.csv');broader=broader[['source_id','variant_id','strict_variant','absolute_drift']];cmp=[]
 for sid,q in v.groupby('source_id'):
  b=broader[broader.source_id==sid];rw=q.absolute_drift.max();bw=b.absolute_drift.max();rm=q.absolute_drift.median();bm=b.absolute_drift.median();reachable=set(q.variant_representation)&set(b.strict_variant);status='TIE' if abs(rw-bw)<1e-6 else ('REPEAT_SHIFT_SUFFICIENT' if rw>bw else 'BROADER_EQUIVALENCE_NEEDED')
  cmp.append({'source_id':sid,'repeat_shift_worst':rw,'broader_worst':bw,'paired_worst_difference':bw-rw,'repeat_shift_median':rm,'broader_median':bm,'paired_median_difference':bm-rm,'broader_worse':bw>rw+1e-6,'repeat_shift_reaches_broader_worst':any(abs(b[b.strict_variant.isin(reachable)].absolute_drift-bw)<1e-6),'inclusion_class':status,'repeat_shift_count':len(q),'broader_count':len(b),'overlap':len(reachable),'unique_broader':len(set(b.strict_variant)-set(q.variant_representation))})
 pd.DataFrame(cmp).to_csv(O/'repeat_shift_vs_broader.csv',index=False)
 rng=np.random.default_rng(SEED);mk=[]
 for k in [1,2,5,10,20]:
  for sid,q in v.groupby('source_id'):
   b=broader[broader.source_id==sid]
   if len(q)<k or len(b)<k:continue
   r=q.absolute_drift.to_numpy();x=b.absolute_drift.to_numpy();rd=[];bd=[]
   for _ in range(1000):rd.append(rng.choice(r,k,False).max());bd.append(rng.choice(x,k,False).max())
   mk.append({'k':k,'source_id':sid,'repeat_shift_expected_worst':np.mean(rd),'broader_expected_worst':np.mean(bd),'paired_difference':np.mean(bd)-np.mean(rd)})
 pd.DataFrame(mk).to_csv(O/'matched_k_comparison.csv',index=False)
 pos=v[v.family=='SHIFT'][['source_id','variant_id','position_displacement','absolute_drift']];rho=spearmanr(pos.position_displacement,pos.absolute_drift).statistic if len(pos)>2 else np.nan;pos.assign(spearman_rho=rho).to_csv(O/'position_analysis.csv',index=False)
 summary={'protocol_status':'FAITHFUL_REIMPLEMENTATION','applicable_sources':len(sm),'verified_variants':len(v),'excluded_variants':len(excluded),'variant_mean':v.absolute_drift.mean(),'variant_median':v.absolute_drift.median(),'variant_p90':v.absolute_drift.quantile(.9),'variant_p95':v.absolute_drift.quantile(.95),'variant_max':v.absolute_drift.max(),'source_worst_mean':sm.max_drift.mean(),'source_worst_median':sm.max_drift.median(),'source_worst_p90':sm.max_drift.quantile(.9),'source_worst_p95':sm.max_drift.quantile(.95),'source_worst_max':sm.max_drift.max(),'source_worst_mean_CI':[boot.mean_source_worst.quantile(.025),boot.mean_source_worst.quantile(.975)],'source_worst_median_CI':[boot.median_source_worst.quantile(.025),boot.median_source_worst.quantile(.975)],'source_worst_P90_CI':[boot.P90_source_worst.quantile(.025),boot.P90_source_worst.quantile(.975)],'graph_mean':gc.graph_absolute_drift.mean(),'graph_median':gc.graph_absolute_drift.median(),'graph_p95':gc.graph_absolute_drift.quantile(.95),'graph_max':gc.graph_absolute_drift.max(),'fraction_broader_worse':pd.DataFrame(cmp).broader_worse.mean(),'fraction_repeat_reaches_broader_worst':pd.DataFrame(cmp).repeat_shift_reaches_broader_worst.mean(),'position_rho':rho};(O/'summary.json').write_text(json.dumps(summary,indent=2,default=float)+'\n');print(json.dumps(summary,indent=2,default=float))
if __name__=='__main__':main()
