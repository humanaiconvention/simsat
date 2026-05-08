// SimSat — main inspector showing the active encounter (Rotterdam case).

const GATES = [
  { name: 'information_gain',    state: 'pass',  detail: 'ΔH = 0.42 nats · threshold 0.10' },
  { name: 'observation_quality', state: 'pass',  detail: 'cloud 48.78 % · geometry nominal' },
  { name: 'metadata_consistency',state: 'pass',  detail: 'sentinel-2c · datetime within window' },
  { name: 'update_magnitude',    state: 'log',   detail: 'lora_delta_l2 = 0.012 · cap 0.30' },
  { name: 'update_rate',         state: 'log',   detail: '34 / 1000 cumulative' },
  { name: 'error_balance',       state: 'pass',  detail: 'last 10 errors balanced — proceed' },
];

function GateRow({ g }) {
  const colors = { pass: '#68d391', block: '#fc8181', log: '#a0aec0' };
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: '24px 1fr auto',
      alignItems: 'center', gap: 14, padding: '10px 0',
      borderBottom: '1px solid rgba(220,230,245,0.05)',
      fontFamily: 'var(--font-mono)', fontSize: 12, color: '#dce6f5',
    }}>
      <span style={{ color: '#6b7693', fontSize: 11 }}>0{GATES.indexOf(g) + 1}</span>
      <div>
        <div>{g.name}</div>
        <div style={{ color: '#6b7693', fontSize: 11, marginTop: 2 }}>{g.detail}</div>
      </div>
      <span style={{
        fontSize: 10, letterSpacing: '0.14em', textTransform: 'uppercase',
        padding: '3px 10px', borderRadius: 999,
        border: `1px solid ${colors[g.state]}55`, color: colors[g.state],
      }}>{g.state}</span>
    </div>
  );
}

function Inspector() {
  return (
    <main style={{ flex: 1, padding: '28px 32px', overflowY: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 22 }}>
        <div>
          <Eyebrow>urban-coastal · trace_4f65355f5c95</Eyebrow>
          <h1 style={{ font: '300 32px/1.1 var(--font-sans)', color: '#fff', margin: '6px 0 0', letterSpacing: '-0.02em' }}>Port of Rotterdam</h1>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button style={{ font: '500 13px/1 var(--font-sans)', padding: '10px 16px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.18)', color: '#dce6f5', background: 'transparent', cursor: 'pointer' }}>Defer</button>
          <button style={{ font: '500 13px/1 var(--font-sans)', padding: '10px 16px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.18)', color: '#dce6f5', background: 'transparent', cursor: 'pointer' }}>Refine</button>
          <button style={{ font: '500 13px/1 var(--font-sans)', padding: '10px 18px', borderRadius: 8, border: '1px solid #fef3c7', color: '#fef3c7', background: 'transparent', cursor: 'pointer' }}>Accept</button>
        </div>
      </div>

      {/* Sentinel tile */}
      <div style={{
        height: 280, borderRadius: 8, overflow: 'hidden',
        border: '1px solid rgba(220,230,245,0.08)',
        background: `#000 url('../../assets/sentinel-rotterdam.png') center / cover no-repeat`,
        marginBottom: 14,
      }}/>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--font-mono)', fontSize: 11, color: '#a0aec0', letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 28 }}>
        <span><b style={{ color: '#dce6f5', fontWeight: 500, marginRight: 6 }}>Sensor</b>sentinel-2c</span>
        <span><b style={{ color: '#dce6f5', fontWeight: 500, marginRight: 6 }}>Cloud</b>48.78 %</span>
        <span><b style={{ color: '#dce6f5', fontWeight: 500, marginRight: 6 }}>Window</b>2026-05-07 04:51 UTC</span>
        <span><b style={{ color: '#dce6f5', fontWeight: 500, marginRight: 6 }}>Bands</b>B04 · B08 · B11 · B12</span>
      </div>

      {/* scaffold → trust → operator */}
      <Card style={{ marginBottom: 22 }}>
        <Eyebrow>Calibration loop</Eyebrow>
        <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: '1fr auto 1fr auto 1fr', alignItems: 'center', gap: 18 }}>
          <div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#6b7693', letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 8 }}>Scaffold</div>
            <ActionPill action="accept" />
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#6b7693', marginTop: 8 }}>geometry nominal</div>
          </div>
          <div style={{ color: '#6b7693', fontFamily: 'var(--font-mono)' }}>→</div>
          <div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#6b7693', letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 8 }}>Trust layer · WCLI</div>
            <ActionPill action="refine" />
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#6b7693', marginTop: 8 }}>cloud may hide containers</div>
          </div>
          <div style={{ color: '#6b7693', fontFamily: 'var(--font-mono)' }}>→</div>
          <div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#6b7693', letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 8 }}>Operator · ben</div>
            <ActionPill action="accept" />
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#6b7693', marginTop: 8 }}>visible 51 % was operationally enough</div>
          </div>
        </div>
      </Card>

      {/* viability gates */}
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 }}>
          <Eyebrow>Six viability gates</Eyebrow>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#68d391' }}>5 pass · 0 block</span>
        </div>
        {GATES.map(g => <GateRow key={g.name} g={g} />)}
      </Card>
    </main>
  );
}

window.Inspector = Inspector;
