// SimSat — top bar with the small lockup, mission clock, operator badge.

function TopBar() {
  return (
    <header style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '14px 24px', borderBottom: '1px solid rgba(220,230,245,0.08)',
      background: '#0a0e27',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 28 }}>
        <Lockup small />
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '0.18em',
          textTransform: 'uppercase', color: '#a0aec0', borderLeft: '1px solid rgba(220,230,245,0.16)',
          paddingLeft: 16,
        }}>SimSat · operator console · v3 canonical</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 28, fontFamily: 'var(--font-mono)', fontSize: 12, color: '#a0aec0' }}>
        <span><b style={{ color: '#dce6f5', fontWeight: 500, marginRight: 6, textTransform: 'uppercase', letterSpacing: '0.12em', fontSize: 11 }}>UTC</b>2026-05-07T04:51:12Z</span>
        <span><b style={{ color: '#dce6f5', fontWeight: 500, marginRight: 6, textTransform: 'uppercase', letterSpacing: '0.12em', fontSize: 11 }}>Operator</b>ben</span>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
          <span style={{ width: 8, height: 8, borderRadius: 999, background: '#68d391' }}/>
          link nominal · 5 MB up / 10 MB down
        </span>
      </div>
    </header>
  );
}

window.TopBar = TopBar;
