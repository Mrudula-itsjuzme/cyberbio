import React, { useEffect, useState, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const fmt = n => n == null ? 'N/A' : Number(n).toFixed(3);
const fmtPct = n => n == null ? 'N/A' : (Number(n) * 100).toFixed(1) + '%';
const pm = (mean, std) => (mean == null ? 'N/A' : `${fmt(mean)} ± ${fmt(std)}`);

const SEEDS = ['20260920', '20260921', '20260922', 'AGGREGATE'];

const RUN_TYPE_COLORS = {
  'SMOKE_ONLY_NON_CANONICAL': '#c0392b',
  'SCIENTIFIC_PILOT': '#2980b9',
  'MULTI-SEED AGGREGATE': '#27ae60',
  'AGGREGATE': '#27ae60',
};

const METRIC_TOOLTIPS = {
  'DEFENDER GAIN': 'R(A, D_before) / R(A, D_after). >1 = defender improved against active attacker.',
  'ADAPTIVE SELECTION GAIN': 'G_select(t) = R(A_{t+1}, D_t) - R(A_t, D_t). Measures controller\'s curriculum selection quality. NOT within-attacker policy adaptation.',
  'HELD-OUT TRANSFER': 'T_Evo(t) = R(Evo, D_t) - R(Evo, D_{t+1}). Transfer of robustness improvements to held-out Evolutionary attacker.',
};

function Tooltip({ text, children }) {
  const [vis, setVis] = useState(false);
  return (
    <span style={{ position: 'relative', cursor: 'help' }}
      onMouseEnter={() => setVis(true)} onMouseLeave={() => setVis(false)}>
      {children}
      {vis && <span style={{
        position: 'absolute', bottom: '120%', left: 0, background: '#222',
        color: '#eee', padding: '6px 10px', borderRadius: 4, fontSize: 11,
        width: 280, zIndex: 99, whiteSpace: 'normal', lineHeight: 1.4
      }}>{text}</span>}
    </span>
  );
}

function Badge({ type }) {
  const label = type === 'SMOKE_ONLY_NON_CANONICAL' ? 'SMOKE'
    : type === 'SCIENTIFIC_PILOT' ? 'SCIENTIFIC PILOT'
    : type === 'MULTI-SEED AGGREGATE' || type === 'AGGREGATE' ? 'MULTI-SEED AGGREGATE'
    : type || '—';
  return (
    <span style={{
      background: RUN_TYPE_COLORS[type] || '#555',
      color: '#fff', padding: '3px 10px', borderRadius: 3,
      fontSize: 11, fontWeight: 700, letterSpacing: 1
    }}>{label}</span>
  );
}

function Panel({ title, data, tooltip }) {
  return (
    <article>
      <h2>{tooltip ? <Tooltip text={tooltip}>{title} ℹ</Tooltip> : title}</h2>
      {Object.entries(data).map(([k, v]) => (
        <div key={k}>
          <span>{k.replace(/_/g, ' ')}</span>
          <b style={{ color: v === 'N/A' ? '#888' : undefined }}>{v}</b>
        </div>
      ))}
    </article>
  );
}

function SeedSummaryPanel({ data }) {
  if (!data) return null;
  const r = data;
  return (
    <section className="grid">
      <Panel title="MULTI-SEED SUMMARY" data={{
        'Seeds': '3 (20260920, 20260921, 20260922)',
        'Rounds per seed': '3',
        'Curriculum': 'Thompson leave-one-out',
        'Held-out attacker': 'Evolutionary',
        'Query budget per source': '25',
      }} />
      <Panel title="CLEAN PERFORMANCE"
        tooltip={METRIC_TOOLTIPS['DEFENDER GAIN']}
        data={{
          'Clean RMSE (R0)': pm(r.rmse_mean?.[0], r.rmse_std?.[0]),
          'Clean RMSE (R1)': pm(r.rmse_mean?.[1], r.rmse_std?.[1]),
          'Clean RMSE (R2)': pm(r.rmse_mean?.[2], r.rmse_std?.[2]),
        }} />
      <Panel title="DEFENDER GAIN"
        tooltip={METRIC_TOOLTIPS['DEFENDER GAIN']}
        data={{
          'Gain R0': pm(r.gain_mean?.[0], r.gain_std?.[0]),
          'Gain R1': pm(r.gain_mean?.[1], r.gain_std?.[1]),
          'Gain R2': pm(r.gain_mean?.[2], r.gain_std?.[2]),
        }} />
    </section>
  );
}

function RedistPanel({ events }) {
  if (!events || events.length === 0) return null;
  const trueEvents = events.filter(e => e.redistribution_observed === 'True' || e.redistribution_observed === true);
  return (
    <section className="wide">
      <h2>ROBUSTNESS REDISTRIBUTION</h2>
      <p style={{ fontSize: 12, color: '#888' }}>
        Redistribution observed when active attacker vulnerability decreases while another attacker's vulnerability increases.
        {' '}{trueEvents.length} / {events.length} events across all seeds.
      </p>
      {trueEvents.length > 0 ? (
        <table>
          <thead><tr>
            <th>SEED</th><th>ROUND</th><th>TRAINED ATTACK ↓</th><th>BEFORE</th><th>AFTER</th>
            <th>OTHER ATTACK ↑</th><th>BEFORE</th><th>AFTER</th>
          </tr></thead>
          <tbody>{trueEvents.slice(0, 8).map((e, i) => (
            <tr key={i}>
              <td>{e.seed}</td>
              <td>{e.round}</td>
              <td style={{ color: '#27ae60' }}>{e.active_attacker}</td>
              <td>{fmt(e.active_before)}</td>
              <td>{fmt(e.active_after)}</td>
              <td style={{ color: '#e67e22' }}>{e.other_attacker}</td>
              <td>{fmt(e.other_before)}</td>
              <td>{fmt(e.other_after)}</td>
            </tr>
          ))}</tbody>
        </table>
      ) : <p style={{ color: '#aaa' }}>No redistribution events detected.</p>}
    </section>
  );
}

function CrossAttackMatrix({ matrix, isAggregate }) {
  if (!matrix || matrix.length === 0) return null;
  const attackers = [...new Set(matrix.map(x => x.attacker))];
  const rounds = [...new Set(matrix.map(x => x.round))].sort((a, b) => Number(a) - Number(b));
  const cell = (round, att) => {
    const rows = matrix.filter(x => String(x.round) === String(round) && x.attacker === att);
    if (!rows.length) return null;
    return Number(rows[0].mean_robustness_effect);
  };
  return (
    <div>
      <h2>ATTACKER × DEFENDER MATRIX {isAggregate ? '(mean across seeds)' : ''}</h2>
      <div className="heat">
        {rounds.map(r => (
          <React.Fragment key={r}>
            <b style={{ fontSize: 10, alignSelf: 'center' }}>D_{Number(r) + 1}</b>
            {attackers.map(att => {
              const v = cell(r, att);
              const heldOut = att === 'Evolutionary';
              return (
                <button key={att}
                  style={{ opacity: v == null ? 0.2 : 0.35 + Math.min(1, v) * 0.65,
                    outline: heldOut ? '2px solid #e67e22' : undefined }}
                  title={`${att} vs D_${Number(r) + 1}: ${v == null ? 'N/A' : v.toFixed(3)}`}>
                  <strong>{att}{heldOut ? ' ★' : ''}</strong>
                  {v == null ? 'N/A' : fmt(v)}
                </button>
              );
            })}
          </React.Fragment>
        ))}
      </div>
      {attackers.includes('Evolutionary') && (
        <small style={{ color: '#e67e22' }}>★ HELD OUT – not used in Thompson training selection</small>
      )}
    </div>
  );
}

function App() {
  const [runs, setRuns] = useState([]);
  const [selectedSeed, setSelectedSeed] = useState('AGGREGATE');
  const [run, setRun] = useState(null);
  const [runData, setRunData] = useState(null);
  const [redistData, setRedistData] = useState([]);
  const [candidates, setCandidates] = useState([]);
  const [selectedCand, setSelectedCand] = useState(null);

  useEffect(() => {
    fetch('/api/runs').then(r => r.json()).then(x => {
      setRuns(x);
      loadSeed('AGGREGATE');
    });
    // Load redistribution for all views
    fetch('/api/aggregate/redistribution').then(r => r.ok ? r.json() : []).then(setRedistData).catch(() => {});
  }, []);

  const loadSeed = useCallback((seedId) => {
    setSelectedSeed(seedId);
    const rid = seedId === 'AGGREGATE' ? 'AGGREGATE'
      : runs.find(r => r.run_id === `arena_transformer_seed${seedId}`)?.run_id
      || `arena_transformer_seed${seedId}`;

    Promise.all([
      fetch(`/api/runs/${rid}`).then(r => r.json()).catch(() => null),
      fetch(`/api/runs/${rid}/data`).then(r => r.json()).catch(() => null),
      fetch(`/api/runs/${rid}/candidates`).then(r => r.json()).catch(() => []),
    ]).then(([summary, data, cands]) => {
      setRun(summary);
      setRunData(data);
      setCandidates(cands || []);
      setSelectedCand(cands?.[0] || null);
    });
  }, [runs]);

  const isAggregate = selectedSeed === 'AGGREGATE';

  const agg_rows = runData?.rounds || [];

  // Build summary stats from agg_rows for the multi-seed summary panel
  const aggStats = {
    rmse_mean: agg_rows.map(r => r.defender_clean_rmse_mean != null ? r.defender_clean_rmse_mean : r.defender_clean_rmse),
    rmse_std: agg_rows.map(r => r.defender_clean_rmse_std ?? null),
    gain_mean: agg_rows.map(r => r.defender_gain_mean != null ? r.defender_gain_mean : r.defender_gain),
    gain_std: agg_rows.map(r => r.defender_gain_std ?? null),
  };

  const latest = run?.rounds?.length ? run.rounds.at(-1) : null;
  const sciStatus = run?.scientific_status || '';

  return (
    <main>
      <header>
        <div>
          <small>OFFLINE RESEARCH INSTRUMENT</small>
          <h1>CYBERBIO <i>ADVERSARIAL ARENA</i></h1>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'flex-end' }}>
          <Badge type={sciStatus} />
          <div style={{ display: 'flex', gap: 6 }}>
            {SEEDS.map(s => (
              <button key={s}
                onClick={() => loadSeed(s)}
                style={{
                  background: selectedSeed === s ? '#2c3e50' : '#1a252f',
                  color: selectedSeed === s ? '#fff' : '#aaa',
                  border: `1px solid ${selectedSeed === s ? '#3498db' : '#444'}`,
                  padding: '4px 10px', fontSize: 11, cursor: 'pointer', borderRadius: 3
                }}>
                {s === 'AGGREGATE' ? '⊕ AGGREGATE' : s}
              </button>
            ))}
          </div>
        </div>
      </header>

      <div className="loop">
        {['ATTACK', 'VALIDATE', 'SCORE', 'SELECT HARD EXAMPLES', 'DEFEND', 'RE-EVALUATE', 'NEXT ROUND'].map((x, i) => (
          <React.Fragment key={x}><b>{x}</b>{i < 6 && <em>→</em>}</React.Fragment>
        ))}
      </div>

      {isAggregate ? (
        <>
          <SeedSummaryPanel data={aggStats} />
          <section className="wide">
            <h2>AGGREGATE ROUND TIMELINE (mean ± std, n=3 seeds)</h2>
            <table>
              <thead><tr>
                <th>ROUND</th>
                <th><Tooltip text={METRIC_TOOLTIPS['DEFENDER GAIN']}>DEFENDER GAIN ℹ</Tooltip></th>
                <th>MEAN DRIFT</th><th>CLEAN RMSE</th>
                <th><Tooltip text={METRIC_TOOLTIPS['HELD-OUT TRANSFER']}>HELD-OUT TRANSFER ℹ</Tooltip></th>
              </tr></thead>
              <tbody>
                {agg_rows.map((r, i) => {
                  const trow = runData?.transfer?.find(t => String(t.round) === String(r.round));
                  return (
                    <tr key={i}>
                      <td>{r.round}</td>
                      <td>{pm(r.defender_gain_mean, r.defender_gain_std)}</td>
                      <td>{pm(r.mean_drift_mean, r.mean_drift_std)}</td>
                      <td>{pm(r.defender_clean_rmse_mean, r.defender_clean_rmse_std)}</td>
                      <td>{trow ? pm(trow.transfer_gain_mean, trow.transfer_gain_std) : 'N/A'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </section>

          <section className="wide">
            <h2>
              <Tooltip text={METRIC_TOOLTIPS['ADAPTIVE SELECTION GAIN']}>ADAPTIVE SELECTION GAIN ℹ</Tooltip>
            </h2>
            <table>
              <thead><tr><th>TRANSITION</th><th>G_select (mean ± std)</th></tr></thead>
              <tbody>
                {(runData?.selection_gain || []).map((r, i) => (
                  <tr key={i}>
                    <td>Round {r.round} → {Number(r.round) + 1}</td>
                    <td>{pm(r.selection_gain_mean, r.selection_gain_std)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className="split">
            <CrossAttackMatrix
              matrix={(() => {
                // Build mean matrix from all seeds' data
                const rows = agg_rows.map((r, i) => {
                  const attackers = ['RandomSMILES', 'MCMC', 'Evolutionary'];
                  return attackers.map(a => ({
                    round: r.round, attacker: a,
                    mean_robustness_effect: r[`mean_drift_mean`] ?? null,
                  }));
                }).flat();
                return rows;
              })()}
              isAggregate={true}
            />
            <div>
              <h2>THOMPSON CONTROLLER (posterior means)</h2>
              <table>
                <thead><tr><th>ROUND</th><th>ATTACKER</th><th>POSTERIOR MEAN</th></tr></thead>
                <tbody>
                  {(runData?.thompson || []).map((r, i) => (
                    <tr key={i}>
                      <td>{r.round}</td><td>{r.attacker}</td>
                      <td>{pm(r.posterior_mean_mean, r.posterior_mean_std)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <RedistPanel events={redistData.length ? redistData : (runData?.redistribution || [])} />
        </>
      ) : (
        <>
          {run && run.rounds && (
            <>
              <section className="status">
                <b>{run.status}</b>
                <Badge type={run.scientific_status} />
                <span>ROUNDS: {run.rounds?.length ?? 0}</span>
              </section>
              {latest && (
                <section className="grid">
                  <Panel title="ATTACKER (LATEST ROUND)" data={{
                    family: latest.active_attacker,
                    proposals: latest.proposals,
                    valid: latest.valid,
                    mean_drift: fmt(latest.mean_drift),
                    max_drift: fmt(latest.max_drift),
                    drift_per_query: fmt(latest.drift_per_query),
                  }} />
                  <Panel title="DEFENDER (LATEST ROUND)"
                    tooltip={METRIC_TOOLTIPS['DEFENDER GAIN']}
                    data={{
                      checkpoint: `D_${latest.round + 1}`,
                      clean_rmse: fmt(latest.defender_clean_rmse),
                      robustness: fmt(latest.defender_robustness),
                      gain: fmt(latest.defender_gain),
                      hard_examples: latest.hard_example_buffer_size,
                    }} />
                  <Panel title="VALIDATOR" data={{
                    validity: fmtPct(latest.validity_rate),
                    invalid: latest.invalid,
                    authority: 'RDKit + canonical checks',
                  }} />
                </section>
              )}
              <section className="wide">
                <h2>ROUND TIMELINE</h2>
                <table>
                  <thead><tr>
                    <th>ROUND</th><th>ATTACKER</th><th>MEAN DRIFT</th><th>MAX</th>
                    <th>VALIDITY</th><th>CLEAN RMSE</th>
                    <th><Tooltip text={METRIC_TOOLTIPS['DEFENDER GAIN']}>GAIN ℹ</Tooltip></th>
                  </tr></thead>
                  <tbody>
                    {run.rounds.map(r => (
                      <tr key={r.round}>
                        <td>{r.round}</td><td>{r.active_attacker}</td>
                        <td>{fmt(r.mean_drift)}</td><td>{fmt(r.max_drift)}</td>
                        <td>{fmtPct(r.validity_rate)}</td><td>{fmt(r.defender_clean_rmse)}</td>
                        <td>{fmt(r.defender_gain)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
              <section className="split">
                <CrossAttackMatrix matrix={runData?.matrix || []} isAggregate={false} />
                <div>
                  <h2>CANDIDATE INSPECTOR</h2>
                  <select onChange={e => setSelectedCand(candidates[e.target.value])}>
                    {candidates.map((c, i) => (
                      <option key={`${c.source_id}-${i}`} value={i}>
                        R{c.round} · {c.source_id} · {c.attack_family}
                        {c.is_hard_example ? ' ★' : ''}
                        {c.held_out_eval ? ' (HELD OUT)' : ''}
                      </option>
                    ))}
                  </select>
                  {selectedCand && (
                    <div style={{ fontSize: 11 }}>
                      <div><b>Seed:</b> {selectedCand.seed || selectedSeed}</div>
                      <div><b>Checkpoint:</b> {selectedCand.defender_checkpoint || '—'}</div>
                      <div><b>Hard example:</b> {selectedCand.is_hard_example ? 'YES ★' : 'no'}</div>
                      <div><b>Held-out eval:</b> {selectedCand.held_out_eval ? 'YES (Evolutionary)' : 'no'}</div>
                      <details><summary>Raw JSON</summary><pre>{JSON.stringify(selectedCand, null, 2)}</pre></details>
                    </div>
                  )}
                </div>
              </section>
            </>
          )}
        </>
      )}
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
