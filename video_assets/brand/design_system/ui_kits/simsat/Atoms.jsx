// SimSat — small, reusable components shared across the UI.
// Exported to window so the index <script> block can pick them up.

const { useState } = React;

function Mark({ size = 22 }) {
  return React.createElement('img', {
    src: '../../assets/logo.svg', alt: '', style: { width: size, height: 'auto' }
  });
}

function Lockup({ small = false }) {
  const fs = small ? 14 : 18;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <Mark size={small ? 18 : 28} />
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, lineHeight: 1, color: '#fff' }}>
        <span style={{ fontWeight: 200, fontSize: fs, letterSpacing: '-0.01em' }}>Human</span>
        <span style={{ fontWeight: 400, fontSize: fs, letterSpacing: '-0.04em' }}>AI</span>
        <span style={{ fontWeight: 300, fontSize: fs - 4, letterSpacing: '0.22em', textTransform: 'uppercase', opacity: 0.7, marginLeft: 10 }}>Convention</span>
      </div>
    </div>
  );
}

const ACTION_COLOR = {
  accept: '#68d391',
  refine: '#7aa7ff',
  defer:  '#a0aec0',
  skip:   '#fc8181',
};

function ActionPill({ action }) {
  const c = ACTION_COLOR[action] || '#a0aec0';
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 8,
      fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '0.12em',
      textTransform: 'uppercase', padding: '4px 12px 4px 10px',
      borderRadius: 999, border: '1px solid rgba(220,230,245,0.16)',
      color: '#dce6f5', background: 'rgba(220,230,245,0.04)',
    }}>
      <span style={{
        width: 8, height: 8, borderRadius: 999,
        background: action === 'defer' ? 'transparent' : c,
        border: action === 'defer' ? `1.5px solid ${c}` : 'none',
        boxSizing: 'border-box',
      }}/>
      {action}
    </span>
  );
}

function Eyebrow({ children }) {
  return (
    <div style={{
      fontFamily: 'var(--font-sans)', fontSize: 10, letterSpacing: '0.18em',
      textTransform: 'uppercase', color: '#a0aec0', fontWeight: 500,
    }}>{children}</div>
  );
}

function StatNumber({ value, label, color = '#fef3c7' }) {
  return (
    <div>
      <div style={{
        fontFamily: 'var(--font-mono)', fontVariantNumeric: 'tabular-nums',
        fontSize: 30, color, letterSpacing: '-0.01em', lineHeight: 1,
      }}>{value}</div>
      {label && <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#6b7693', marginTop: 6, letterSpacing: '0.04em' }}>{label}</div>}
    </div>
  );
}

function Card({ children, padding = 18, style = {} }) {
  return (
    <div style={{
      background: '#131838',
      border: '1px solid rgba(220,230,245,0.08)',
      borderRadius: 8,
      boxShadow: '0 1px 0 rgba(255,255,255,0.04) inset, 0 24px 48px -24px rgba(0,0,0,0.6)',
      padding,
      ...style,
    }}>{children}</div>
  );
}

Object.assign(window, { Mark, Lockup, ActionPill, Eyebrow, StatNumber, Card, ACTION_COLOR, useState });
