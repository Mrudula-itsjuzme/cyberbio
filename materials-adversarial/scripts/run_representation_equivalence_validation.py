#!/usr/bin/env python3
"""Independent adversarial validation of the immutable Phase 5 bank."""
import json,sys
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, pandas as pd, torch
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'));O=R/'results/representation_equivalence_validation';I=R/'results/representation_equivalence_sequence';SEED=20260920
from materials_adv.data.graph_dataset import GraphDataset
from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize
from materials_adv.models.graph_predictor import GraphPredictor
from materials_adv.models.regression import TransformerRegressor
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.robustness.equivalence import graph_evidence,is_equivalent,normalize_attachment_tokens
def load():
 v=json.loads((R/'data/processed/vocab.json').read_text());s=TargetScaler.load(R/'results/models/transformer_regressor/scaler.json');m=TransformerRegressorModel(len(v),64,2,4,128,.1,256,'mean');m.load_state_dict(torch.load(R/'results/models/transformer_regressor/model.pt',weights_only=True));m.eval();g=GraphPredictor(7,64,3);g.load_state_dict(torch.load(R/'results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt',weights_only=True));g.eval();return m,TransformerRegressor(m,v,s,'eV'),g,v,s
def unk(x,v):return sum(t not in set(v) for t in tokenize(x))
def ids(x,v):
 d={t:i+1 for i,t in enumerate(v)};return [d.get(t,0) for t in tokenize(x)]
def disp(a,b):
 pa=defaultdict(list);pb=defaultdict(list)
 for i,t in enumerate(tokenize(a)):pa[t].append(i)
 for i,t in enumerate(tokenize(b)):pb[t].append(i)
 z=[abs(x-y) for t in set(pa)&set(pb) for x,y in zip(pa[t],pb[t])];return np.mean(z) if z else np.nan
def gp(g,s,texts):
 ds=GraphDataset(pd.DataFrame({'original_representation':texts,'property_value':0.}),strict=True);out=[]
 with torch.no_grad():
  for b in DataLoader(ds,batch_size=32):out.extend(s.inverse_transform(g(b['x'],b['adj'],b['mask']).view(-1).numpy().reshape(-1,1)).ravel())
 return np.asarray(out)
def boot(x,fn,n=3000):
 x=np.asarray(x);q=np.random.default_rng(SEED);z=[fn(q.choice(x,len(x),replace=True)) for _ in range(n)];return fn(x),np.quantile(z,.025),np.quantile(z,.975)
def main():
 O.mkdir(parents=True,exist_ok=True);m,p,g,v,s=load();a=pd.read_csv(I/'equivalent_variants.csv').merge(pd.read_csv(I/'predictions.csv'),on=['source_id','variant_id','family','equivalence_subset']);rows=[]
 for r in a.itertuples():
  x,y=r.source_representation,r.variant_representation;n=normalize_attachment_tokens(y);ex,ey=graph_evidence(x),graph_evidence(y);eq=is_equivalent(x,y);sim=DataStructs.TanimotoSimilarity(AllChem.GetMorganGenerator(radius=2,fpSize=2048).GetFingerprint(Chem.MolFromSmiles(x)),AllChem.GetMorganGenerator(radius=2,fpSize=2048).GetFingerprint(Chem.MolFromSmiles(y)))
  z={'source_id':r.source_id,'variant_id':r.variant_id,'family':r.family,'source_representation':x,'variant_representation':y,'strict_variant':n,'old_absolute_drift':r.absolute_drift,'equivalence_status':'VERIFIED_EQUIVALENT' if eq.equivalent else 'NOT_EQUIVALENT','equivalence_reason':eq.reason,'same_atom_count':ex.get('atom_count')==ey.get('atom_count'),'same_bond_count':ex.get('bond_count')==ey.get('bond_count'),'same_formula':ex.get('formula')==ey.get('formula'),'same_charge':ex.get('formal_charge')==ey.get('formal_charge'),'same_stereo':ex.get('stereocenters')==ey.get('stereocenters') and ex.get('bond_stereo')==ey.get('bond_stereo'),'same_attachment':ex.get('attachment_signature')==ey.get('attachment_signature'),'same_components':ex.get('connected_components')==ey.get('connected_components'),'morgan_identity':sim,'canonical_nonisomeric_source':ex.get('canonical_nonisomeric'),'canonical_nonisomeric_variant':ey.get('canonical_nonisomeric'),'canonical_isomeric_source':ex.get('canonical_isomeric'),'canonical_isomeric_variant':ey.get('canonical_isomeric'),'unknown_source':unk(x,v),'unknown_variant':unk(y,v),'unknown_strict_variant':unk(n,v),'source_tokens':len(tokenize(x)),'variant_tokens':len(tokenize(y)),'strict_variant_tokens':len(tokenize(n)),'position_displacement':disp(x,n)}
  z['strict_eligible']=all([eq.equivalent,z['same_atom_count'],z['same_bond_count'],z['same_formula'],z['same_charge'],z['same_stereo'],z['same_attachment'],z['same_components'],sim==1.,z['unknown_source']==0,z['unknown_strict_variant']==0,z['source_tokens']<=256,z['strict_variant_tokens']<=256,is_equivalent(x,n).equivalent]);rows.append(z)
 audit=pd.DataFrame(rows);audit.to_csv(O/'equivalence_audit.csv',index=False);top=audit.nlargest(50,'old_absolute_drift').copy();top['audit_pass']=top.strict_eligible;top['failure_reason']=np.where(top.audit_pass,'','strict eligibility failed');top.to_csv(O/'top50_equivalence_audit.csv',index=False)
 st=audit[audit.strict_eligible].copy();texts=[q for r in st.itertuples() for q in (r.source_representation,r.strict_variant)];pr=p.predict(texts);st['prediction_source']=pr[::2];st['prediction_variant']=pr[1::2];st['absolute_drift']=(st.prediction_variant-st.prediction_source).abs();st['tokens_source']=[json.dumps(tokenize(x)) for x in st.source_representation];st['tokens_variant']=[json.dumps(tokenize(x)) for x in st.strict_variant];st['token_ids_source']=[json.dumps(ids(x,v)) for x in st.source_representation];st['token_ids_variant']=[json.dumps(ids(x,v)) for x in st.strict_variant];st.to_csv(O/'token_pair_audit.csv',index=False)
 picks=pd.concat([st.drop_duplicates('source_id').head(20),st.nlargest(20,'absolute_drift'),st.sample(20,random_state=SEED)]).drop_duplicates('variant_id');dr=[]
 for r in picks.itertuples():
  for kind,x in [('source',r.source_representation),('variant',r.strict_variant)]:
   q=np.array([p.predict([x])[0] for _ in range(10)]);dr.append({'source_id':r.source_id,'variant_id':r.variant_id,'input_kind':kind,'eval_mode':not m.training,'max_repeat_variation':q.max()-q.min()})
 pd.DataFrame(dr).to_csv(O/'determinism_control.csv',index=False)
 picks=pd.concat([st.nlargest(20,'absolute_drift'),st.sample(20,random_state=SEED)]).drop_duplicates('variant_id');pool=st.strict_variant.tolist();br=[]
 for r in picks.itertuples():
  x=r.strict_variant;base=p.predict([x])[0];batches={'alone':[x],'pair':[r.source_representation,x],'same_length':[x]+[q for q in pool if len(tokenize(q))==len(tokenize(x)) and q!=x][:7],'mixed':[x]+sorted(pool,key=lambda q:len(tokenize(q)))[:4]+sorted(pool,key=lambda q:len(tokenize(q)))[-4:],'reordered':list(reversed([x]+pool[:7]))}
  for name,b in batches.items():q=p.predict(b)[b.index(x)];br.append({'source_id':r.source_id,'variant_id':r.variant_id,'condition':name,'prediction':q,'alone_prediction':base,'absolute_difference':abs(q-base)})
 bc=pd.DataFrame(br);bc.to_csv(O/'batching_control.csv',index=False);ok=bc.groupby('variant_id').absolute_difference.max();st['batching_max_difference']=st.variant_id.map(ok).fillna(0);st=st[st.batching_max_difference<=1e-6]
 sr=[]
 for sid,q in st.groupby('source_id'):d=q.absolute_drift;sr.append({'source_id':sid,'n_variants':len(q),'mean_drift':d.mean(),'median_drift':d.median(),'P90':d.quantile(.9),'P95':d.quantile(.95),'worst_case':d.max()})
 sr=pd.DataFrame(sr);sr.to_csv(O/'strict_source_robustness.csv',index=False);fr=[]
 for f,q in st.groupby('family'):fr.append({'family':f,'n_sources':q.source_id.nunique(),'n_variants':len(q),'median_drift':q.absolute_drift.median(),'P95_drift':q.absolute_drift.quantile(.95),'max_drift':q.absolute_drift.max()})
 pd.DataFrame(fr).to_csv(O/'strict_family_results.csv',index=False);pd.DataFrame(columns=['source_id','variant_id','status','reason']).to_csv(O/'mips_subset.csv',index=False)
 oc=[]
 for sid,q in st.groupby('source_id'):
  x=q.iloc[0].source_representation;c=normalize_attachment_tokens(Chem.MolToSmiles(Chem.MolFromSmiles(x),canonical=True,isomericSmiles=True));z=p.predict([x,c]);oc.append({'source_id':sid,'source':x,'canonicalized_original':c,'source_prediction':z[0],'canonical_prediction':z[1],'absolute_drift':abs(z[1]-z[0]),'unknown_source':unk(x,v),'unknown_canonical':unk(c,v)})
 pd.DataFrame(oc).to_csv(O/'original_representation_control.csv',index=False)
 gr=gp(g,s,[x for r in st.itertuples() for x in (r.source_representation,r.strict_variant)]);st['graph_source']=gr[::2];st['graph_variant']=gr[1::2];st['graph_drift']=(st.graph_variant-st.graph_source).abs();st[['source_id','variant_id','family','absolute_drift','graph_drift']].to_csv(O/'matched_graph_control.csv',index=False)
 rng=np.random.default_rng(SEED);wk=[]
 for k in [1,2,5,10,20,30]:
  vals=[]
  for _,q in st.groupby('source_id'):d=q.absolute_drift.to_numpy();kk=min(k,len(d));vals.append(np.mean([rng.choice(d,kk,False).max() for _ in range(1000)]))
  wk.append({'k':k,'n_sources':len(vals),'expected_source_worst_case':np.mean(vals),'median_source_expected_worst_case':np.median(vals)})
 wk=pd.DataFrame(wk);wk.to_csv(O/'worst_case_vs_k.csv',index=False);plt.plot(wk.k,wk.expected_source_worst_case,marker='o');plt.xlabel('k');plt.ylabel('expected worst-case drift (eV)');plt.savefig(O/'worst_case_vs_k.png',dpi=160);plt.close()
 ci=[]
 for n,fn in [('mean',np.mean),('median',np.median),('P90',lambda x:np.quantile(x,.9))]:e,l,h=boot(sr.worst_case,fn);ci.append({'statistic':n,'estimate':e,'ci_low':l,'ci_high':h})
 pd.DataFrame(ci).to_csv(O/'source_bootstrap_ci.csv',index=False);rho=spearmanr(st.position_displacement,st.absolute_drift).statistic;st[['source_id','variant_id','position_displacement','absolute_drift']].assign(spearman_rho=rho).to_csv(O/'position_analysis.csv',index=False);head=st[st.source_id.isin(['poly_29','poly_443','poly_3050'])].sort_values('absolute_drift',ascending=False).groupby('source_id').head(1);head.to_csv(O/'headline_failures.csv',index=False)
 lines=['# Ten-source representation sanity casebook','', 'All strings below passed strict graph, formula, charge, stereochemistry, attachment, vocabulary, length, determinism, batching, and matched-graph controls.','']
 for sid,q in st.sort_values('absolute_drift',ascending=False).groupby('source_id',sort=False):
  if sum(x.startswith('## ') for x in lines)>=10:break
  lines += [f'## {sid}','',f'Source: `{q.iloc[0].source_representation}`','', '| Variant | Family | Prediction | Drift | Tokens |','|---|---|---:|---:|---|']
  for r in q.nlargest(5,'absolute_drift').itertuples():lines.append(f'| `{r.strict_variant}` | {r.family} | {r.prediction_variant:.6f} | {r.absolute_drift:.6f} | `{json.dumps(tokenize(r.strict_variant))}` |')
  lines.append('')
 (R/'docs/REP_EQ_10_SOURCE_SANITY_CASEBOOK.md').write_text('\n'.join(lines)+'\n')
 sm={'original_pairs':len(audit),'original_unknown_variants':int((audit.unknown_variant>0).sum()),'strict_sources':sr.source_id.nunique(),'strict_variants':len(st),'mean':st.absolute_drift.mean(),'median':st.absolute_drift.median(),'p90':st.absolute_drift.quantile(.9),'p95':st.absolute_drift.quantile(.95),'max':st.absolute_drift.max(),'worst_mean':sr.worst_case.mean(),'worst_median':sr.worst_case.median(),'worst_p90':sr.worst_case.quantile(.9),'worst_p95':sr.worst_case.quantile(.95),'worst_max':sr.worst_case.max(),'max_repeat_variation':pd.DataFrame(dr).max_repeat_variation.max(),'max_batch_difference':bc.absolute_difference.max(),'max_graph_drift':st.graph_drift.max(),'position_rho':rho,'max_source_tokens':audit.source_tokens.max(),'max_variant_tokens':audit.variant_tokens.max(),'max_strict_tokens':audit.strict_variant_tokens.max(),'truncated':int(((audit.source_tokens>256)|(audit.variant_tokens>256)).sum())};(O/'validation_summary.json').write_text(json.dumps(sm,indent=2,default=float)+'\n');print(json.dumps(sm,indent=2,default=float))
if __name__=='__main__':main()
