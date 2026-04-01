/**
 * PRISMPanel — displays PRISM entropy measurements and viability gate results.
 *
 * Shows before/after spectral entropy, entropy delta proof,
 * geometric health scores, and viability gate pass/fail.
 */
import React from "react";
import type { ConventionSession } from "./haicApi";

interface Props {
  session: ConventionSession;
  onContinue: () => void;
}

export const PRISMPanel: React.FC<Props> = ({ session, onContinue }) => {
  const { entropy_delta, geometric_health_before, geometric_health_after, viability_gates } = session;

  const delta_se = entropy_delta?.delta_spectral_entropy ?? null;
  const reduction_verified = entropy_delta?.reduction_verified ?? false;
  const gates = viability_gates ?? {};
  const gateCount = Object.keys(gates).length;
  const passCount = Object.values(gates).filter(Boolean).length;

  return (
    <div className="haic-prism">
      <h3 className="haic-section-title">PRISM Measurement</h3>

      {/* Entropy Delta */}
      <div className="prism-card">
        <div className="prism-card-title">Spectral Entropy</div>
        <div className="prism-metrics">
          <EntropyMeter
            label="Before"
            value={geometric_health_before?.spectral_entropy ?? null}
          />
          <div className="prism-arrow">→</div>
          <EntropyMeter
            label="After"
            value={geometric_health_after?.spectral_entropy ?? null}
          />
        </div>
        {delta_se !== null && (
          <div className={`prism-delta ${delta_se < 0 ? "positive" : "negative"}`}>
            ΔS = {delta_se > 0 ? "+" : ""}{delta_se.toFixed(4)}
            {" "}
            {reduction_verified ? "✓ Reduction verified" : "✗ No reduction"}
          </div>
        )}
        {delta_se === null && (
          <div className="prism-delta neutral">
            No PRISM model configured — synthetic mode
          </div>
        )}
      </div>

      {/* Geometric Health */}
      {(geometric_health_before || geometric_health_after) && (
        <div className="prism-card">
          <div className="prism-card-title">Geometric Health Score</div>
          <div className="prism-metrics">
            <HealthGauge
              label="Before"
              value={geometric_health_before?.composite_score ?? null}
            />
            <div className="prism-arrow">→</div>
            <HealthGauge
              label="After"
              value={geometric_health_after?.composite_score ?? null}
            />
          </div>
          <p className="prism-note">
            Higher score = greater entropy crisis = higher grounding demand
          </p>
        </div>
      )}

      {/* Viability Gates */}
      {gateCount > 0 && (
        <div className="prism-card">
          <div className="prism-card-title">
            Viability Gates — {passCount}/{gateCount} passed
          </div>
          <div className="prism-gates">
            {Object.entries(gates).map(([gate, passed]) => (
              <div key={gate} className={`prism-gate ${passed ? "pass" : "fail"}`}>
                <span className="prism-gate-icon">{passed ? "✓" : "✗"}</span>
                <span className="prism-gate-name">{gate.replace(/_/g, " ")}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <button className="haic-btn primary" onClick={onContinue}>
        View Receipt
      </button>
    </div>
  );
};

// ---- Sub-components ----

const EntropyMeter: React.FC<{ label: string; value: number | null }> = ({ label, value }) => (
  <div className="prism-meter">
    <span className="prism-meter-label">{label}</span>
    <span className="prism-meter-value">
      {value !== null ? value.toFixed(3) : "—"}
    </span>
    {value !== null && (
      <div className="prism-meter-bar">
        <div
          className="prism-meter-fill entropy"
          style={{ width: `${Math.min(100, (value / 8) * 100)}%` }}
        />
      </div>
    )}
  </div>
);

const HealthGauge: React.FC<{ label: string; value: number | null }> = ({ label, value }) => (
  <div className="prism-meter">
    <span className="prism-meter-label">{label}</span>
    <span className="prism-meter-value">
      {value !== null ? (value * 100).toFixed(1) + "%" : "—"}
    </span>
    {value !== null && (
      <div className="prism-meter-bar">
        <div
          className="prism-meter-fill health"
          style={{ width: `${Math.min(100, value * 100)}%` }}
        />
      </div>
    )}
  </div>
);
