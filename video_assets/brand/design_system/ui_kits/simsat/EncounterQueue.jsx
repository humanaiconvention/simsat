// SimSat — encounter-window queue (left rail).

const ENCOUNTERS = [
  { id: 'enc_3957…', target: 'Suez Canal',                pack: 'maritime',         cloud: 0.13,  scaffold: 'accept', trust: 'accept', operator: 'accept', active: false, sensor: 'sentinel-2a' },
  { id: 'enc_d886…', target: 'Houston Ship Channel',      pack: 'disaster',         cloud: 4.05,  scaffold: 'accept', trust: 'accept', operator: 'accept', active: false, sensor: 'sentinel-2c' },
  { id: 'enc_4f65…', target: 'Port of Rotterdam',         pack: 'urban-coastal',    cloud: 48.78, scaffold: 'accept', trust: 'refine', operator: 'accept', active: true,  sensor: 'sentinel-2c' },
  { id: 'enc_3d10…', target: 'Nile Delta',                pack: 'pedospheric',      cloud: 0.00,  scaffold: 'accept', trust: 'accept', operator: 'accept', active: false, sensor: 'sentinel-2c' },
  { id: 'enc_a2c1…', target: 'Mato Grosso frontier',      pack: 'pedospheric',      cloud: 12.4,  scaffold: 'refine', trust: 'refine', operator: '—',       active: false, sensor: 'sentinel-2a' },
  { id: 'enc_b91e…', target: 'Singapore Strait',          pack: 'maritime',         cloud: 6.7,   scaffold: 'accept', trust: 'refine', operator: '—',       active: false, sensor: 'sentinel-2c' },
  { id: 'enc_c4f7…', target: 'San Francisco Bay',         pack: 'urban-coastal',    cloud: 22.1,  scaffold: 'refine', trust: 'defer',  operator: '—',       active: false, sensor: 'sentinel-2a' },
];

function EncounterQueue({ activeId, onPick }) {
  return (
    <aside style={{
      width: 320, flex: '0 0 320px',
      background: '#0a0e27',
      borderRight: '1px solid rgba(220,230,245,0.08)',
      padding: '20px 0', display: 'flex', flexDirection: 'column',
    }}>
      <div style={{ padding: '0 20px 14px', display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
        <Eyebrow>Encounter queue</Eyebrow>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#6b7693' }}>{ENCOUNTERS.length} · 5d cadence</span>
      </div>
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {ENCOUNTERS.map(e => (
          <button key={e.id} onClick={() => onPick(e.id)} style={{
            width: '100%', textAlign: 'left', background: e.id === activeId ? 'rgba(254,243,199,0.05)' : 'transparent',
            borderLeft: e.id === activeId ? '2px solid #fef3c7' : '2px solid transparent',
            border: 'none', borderBottom: '1px solid rgba(220,230,245,0.05)',
            padding: '14px 18px', cursor: 'pointer', color: '#dce6f5',
            fontFamily: 'var(--font-sans)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
              <span style={{ fontWeight: 500, fontSize: 14 }}>{e.target}</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#6b7693' }}>{e.cloud.toFixed(2)}%</span>
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#6b7693', marginBottom: 8 }}>
              {e.id} · {e.sensor} · {e.pack}
            </div>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center', fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#a0aec0' }}>
              <span style={{ color: ACTION_COLOR[e.scaffold] }}>{e.scaffold}</span>
              <span style={{ color: '#6b7693' }}>→</span>
              <span style={{ color: ACTION_COLOR[e.trust] }}>{e.trust}</span>
              <span style={{ color: '#6b7693' }}>→</span>
              <span style={{ color: e.operator === '—' ? '#6b7693' : ACTION_COLOR[e.operator] }}>{e.operator}</span>
            </div>
          </button>
        ))}
      </div>
    </aside>
  );
}

Object.assign(window, { EncounterQueue, ENCOUNTERS });
