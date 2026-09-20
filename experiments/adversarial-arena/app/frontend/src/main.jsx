import React,{useEffect,useState} from 'react';
import{createRoot}from'react-dom/client';
import'./style.css';

const fmt=n=>n==null?'—':Number(n).toFixed(3);
function App(){
 const[runs,setRuns]=useState([]),[run,setRun]=useState(null),[candidates,setCandidates]=useState([]),[selected,setSelected]=useState(null),[matrix,setMatrix]=useState([]);
 useEffect(()=>{fetch('/api/runs').then(r=>r.json()).then(x=>{setRuns(x);if(x[0])load(x[0].run_id)})},[]);
 const load=id=>Promise.all([fetch(`/api/runs/${id}`).then(r=>r.json()),fetch(`/api/runs/${id}/candidates`).then(r=>r.json()),fetch(`/api/runs/${id}/matrix`).then(r=>r.json())]).then(([a,b,c])=>{setRun(a);setCandidates(b);setSelected(b[0]);setMatrix(c)});
 if(!run)return <main><h1>CYBERBIO // ADVERSARIAL ARENA</h1><p>No completed artifacts found.</p></main>;
 const latest=run.rounds.at(-1),arms=[...new Set(matrix.map(x=>x.attacker))],rounds=[...new Set(matrix.map(x=>x.round))];
 return <main><header><div><small>OFFLINE RESEARCH INSTRUMENT</small><h1>CYBERBIO <i>ADVERSARIAL ARENA</i></h1></div><select onChange={e=>load(e.target.value)}>{runs.map(x=><option key={x.run_id}>{x.run_id}</option>)}</select></header>
 <section className="status"><b>{run.status}</b><span>{run.scientific_status}</span><span>ROUND {latest.round+1}/{run.rounds.length}</span></section>
 <div className="loop">{['ATTACK','VALIDATE','SCORE','SELECT HARD EXAMPLES','DEFEND','RE-EVALUATE','NEXT ROUND'].map((x,i)=><React.Fragment key={x}><b>{x}</b>{i<6&&<em>→</em>}</React.Fragment>)}</div>
 <section className="grid"><Panel title="ATTACKER" data={{family:latest.active_attacker,proposals:latest.proposals,valid:latest.valid,mean_drift:fmt(latest.mean_drift),max_drift:fmt(latest.max_drift),drift_query:fmt(latest.drift_per_query)}}/><Panel title="DEFENDER" data={{checkpoint:`round-${latest.round+1}`,clean_rmse:fmt(latest.defender_clean_rmse),robustness:fmt(latest.defender_robustness),gain:fmt(latest.defender_gain),hard_examples:latest.hard_example_buffer_size}}/><Panel title="VALIDATOR" data={{validity:fmt(latest.validity_rate),invalid:latest.invalid,authority:'RDKit + canonical checks'}}/></section>
 <section className="wide"><h2>ROUND TIMELINE</h2><table><thead><tr><th>ROUND</th><th>ATTACKER</th><th>MEAN DRIFT</th><th>MAX</th><th>VALIDITY</th><th>CLEAN RMSE</th><th>GAIN</th></tr></thead><tbody>{run.rounds.map(r=><tr key={r.round}><td>{r.round}</td><td>{r.active_attacker}</td><td>{fmt(r.mean_drift)}</td><td>{fmt(r.max_drift)}</td><td>{fmt(r.validity_rate)}</td><td>{fmt(r.defender_clean_rmse)}</td><td>{fmt(r.defender_gain)}</td></tr>)}</tbody></table></section>
 <section className="split"><div><h2>ATTACKER × DEFENDER</h2><div className="heat">{matrix.map((x,i)=><button key={i} style={{opacity:.35+Math.min(1,Number(x.mean_robustness_effect))*0.65}}>{x.defender_checkpoint}<strong>{x.attacker}</strong>{fmt(x.mean_robustness_effect)}</button>)}</div></div><div><h2>CANDIDATE INSPECTOR</h2><select onChange={e=>setSelected(candidates[e.target.value])}>{candidates.map((c,i)=><option key={`${c.source_id}-${i}`} value={i}>{c.source_id} · {c.attack_family}</option>)}</select>{selected&&<pre>{JSON.stringify(selected,null,2)}</pre>}</div></section>
 <section className="wide"><h2>RESEARCH QUESTIONS</h2><div className="rq">{Array.from({length:10},(_,i)=><div key={i}><b>RQ{i+1}</b><span>{run.scientific_status==='SMOKE_ONLY_NON_CANONICAL'?'UNANSWERED':'PARTIAL'}</span></div>)}</div></section>
 </main>}
function Panel({title,data}){return <article><h2>{title}</h2>{Object.entries(data).map(([k,v])=><div key={k}><span>{k.replace('_',' ')}</span><b>{v}</b></div>)}</article>}
createRoot(document.getElementById('root')).render(<App/>);
