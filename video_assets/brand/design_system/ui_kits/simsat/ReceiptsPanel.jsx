// SimSat — right rail: numeric receipts.

function ReceiptsPanel() {
  return (
    <aside style={{
      width: 300, flex: '0 0 300px', padding: '24px 22px',
      background: '#050812', borderLeft: '1px solid rgba(220,230,245,0.08)',
      overflowY: 'auto',
    }}>
      <Eyebrow>Receipts · v3 holdout</Eyebrow>
      <div style={{ display: 'grid', gap: 18, marginTop: 14, marginBottom: 26 }}>
        <StatNumber value="+68.8 pp" label="exact_action_agreement · 0.156 → 0.844" />
        <StatNumber value="−31.0 pp" label="score_mae · 0.365 → 0.055 (lower better)" color="#68d391" />
        <StatNumber value="1.000" label="parse_rate · base & tuned" color="#dce6f5" />
      </div>

      <div style={{ height: 1, background: 'rgba(220,230,245,0.08)', margin: '6px 0 22px' }}/>

      <Eyebrow>Class-targeted TTT · headline</Eyebrow>
      <div style={{ display: 'grid', gap: 18, marginTop: 14, marginBottom: 26 }}>
        <StatNumber value="+37.5 pp" label="skip · 0.375 → 0.750 · 16 steps" color="#68d391" />
        <StatNumber value="+75.0 pp" label="defer · 0.125 → 0.875 · 16 steps" color="#68d391" />
      </div>

      <div style={{ height: 1, background: 'rgba(220,230,245,0.08)', margin: '6px 0 22px' }}/>

      <Eyebrow>Honest negatives</Eyebrow>
      <div style={{ display: 'grid', gap: 18, marginTop: 14 }}>
        <StatNumber value="−6.3 pp"  label="v4 · imbalanced data"          color="#fc8181" />
        <StatNumber value="−15.6 pp" label="v5 · class-balanced data"      color="#fc8181" />
        <StatNumber value="−3.1 pp"  label="v3+ · recipe variant"          color="#fc8181" />
      </div>

      <p style={{ fontFamily: 'var(--font-sans)', fontSize: 12, lineHeight: 1.5, color: '#a0aec0', marginTop: 26 }}>
        Three independent angles, same conclusion: v3 sits at a local optimum on this architecture and this holdout.
      </p>
    </aside>
  );
}

window.ReceiptsPanel = ReceiptsPanel;
