import json
from pathlib import Path
import numpy as np, pandas as pd, pytest, torch
from rdkit import Chem
from materials_adv.data.tokenizer import tokenize
from materials_adv.robustness.equivalence import graph_evidence,is_equivalent,normalize_attachment_tokens
ROOT=Path(__file__).resolve().parents[1]
def test_attachment_formula_stereo_preservation():
 a='[*][C@H](F)C(=O)N[*]';b=normalize_attachment_tokens(Chem.MolToSmiles(Chem.MolFromSmiles(a),canonical=False,rootedAtAtom=3,isomericSmiles=True))
 assert is_equivalent(a,b).equivalent
 x,y=graph_evidence(a),graph_evidence(b)
 assert x['formula']==y['formula'] and x['attachment_signature']==y['attachment_signature'] and x['stereocenters']==y['stereocenters']
def test_unknown_accounting_and_normalization():
 v=json.loads((ROOT/'data/processed/vocab.json').read_text()); assert '*' not in v and '[*]' in v
 assert sum(t not in v for t in tokenize('*CC*'))==2
 assert sum(t not in v for t in tokenize(normalize_attachment_tokens('*CC*')))==0
def test_max_length_behavior_and_determinism():
 from scripts.run_sequence_equivalence_audit import load_frozen_model
 m,p,_=load_frozen_model();x='[*]CC[*]';assert p.predict([x])[0]==p.predict([x])[0] and not m.training
 with pytest.raises(ValueError):p.predict(['C'*257])
def test_batch_composition_invariance():
 from scripts.run_sequence_equivalence_audit import load_frozen_model
 _,p,_=load_frozen_model();x='[*]CC[*]';a=p.predict([x])[0];b=p.predict(['[*]C[*]',x,'[*]CCCC[*]'])[1];assert abs(a-b)<1e-6
def test_masked_pooling_excludes_padding():
 from scripts.run_sequence_equivalence_audit import load_frozen_model
 m,_,_=load_frozen_model();m.eval();a_ids=torch.tensor([[1,2]]);b_ids=torch.tensor([[1,2,0]])
 with torch.no_grad():
  a=m(a_ids,padding_mask=torch.tensor([[False,False]]));b=m(b_ids,padding_mask=torch.tensor([[False,False,True]]))
 assert torch.allclose(a,b,atol=1e-6)
def test_strict_outputs_bootstrap_and_worst_k():
 o=ROOT/'results/representation_equivalence_validation';s=pd.read_csv(o/'strict_source_robustness.csv');k=pd.read_csv(o/'worst_case_vs_k.csv');ci=pd.read_csv(o/'source_bootstrap_ci.csv')
 assert len(s)==100 and (s.n_variants>0).all();assert k.k.tolist()==[1,2,5,10,20,30] and k.expected_source_worst_case.is_monotonic_increasing;assert set(ci.statistic)=={'mean','median','P90'}
