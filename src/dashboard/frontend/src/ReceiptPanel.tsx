/**
 * ReceiptPanel — displays the Merkle-rooted participation receipt.
 *
 * Shows the settlement outcome, Merkle root, summary, and a
 * visual breakdown of the leaf commitments.
 */
import React, { useState } from "react";
import type { ConventionSession } from "./haicApi";

interface Props {
  session: ConventionSession;
  onReset: () => void;
}

export const ReceiptPanel: React.FC<Props> = ({ session, onReset }) => {
  const receipt = session.settlement_result;
  const [showLeaves, setShowLeaves] = useState(false);

  const statusColor =
    session.status === "settled"
      ? "#10b981"
      : session.status === "failed"
      ? "#ef4444"
      : "#f59e0b";

  return (
    <div className="haic-receipt">
      <h3 className="haic-section-title">Participation Receipt</h3>

      {/* Status badge */}
      <div className="receipt-status-row">
        <span className="receipt-status-label">Status</span>
        <span className="receipt-status-badge" style={{ background: statusColor }}>
          {session.status.toUpperCase()}
        </span>
      </div>

      {receipt ? (
        <>
          {/* Summary */}
          <div className="receipt-summary">
            {receipt.summary}
          </div>

          {/* Merkle root */}
          <div className="receipt-card">
            <div className="receipt-card-title">Merkle Root</div>
            <div className="receipt-hash">{receipt.merkle_root}</div>
            <div className="receipt-meta">
              Issued: {new Date(receipt.issued_at).toISOString().replace("T", " ").substring(0, 19)} UTC
            </div>
          </div>

          {/* Gate summary */}
          <div className="receipt-card">
            <div className="receipt-card-title">Viability Gates</div>
            {session.viability_gates && (
              <div className="prism-gates">
                {Object.entries(session.viability_gates).map(([gate, passed]) => (
                  <div key={gate} className={`prism-gate ${passed ? "pass" : "fail"}`}>
                    <span className="prism-gate-icon">{passed ? "✓" : "✗"}</span>
                    <span className="prism-gate-name">{gate.replace(/_/g, " ")}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* PoG */}
          <div className="receipt-card">
            <div className="receipt-card-title">Proof-of-Grounding</div>
            <div className={`receipt-pog ${session.pog_verified ? "verified" : "unverified"}`}>
              {session.pog_verified ? "✓ Biological origin verified" : "⚠ Not verified"}
            </div>
            {session.pog_telemetry && "result" in session.pog_telemetry && (
              <div className="receipt-pog-score">
                Provenance score: {(session.pog_telemetry.result as { provenance_score?: number })?.provenance_score?.toFixed(3) ?? "—"}
              </div>
            )}
          </div>

          {/* Leaf commitments (collapsible) */}
          {receipt.leaves && (
            <div className="receipt-card">
              <div
                className="receipt-card-title clickable"
                onClick={() => setShowLeaves((v) => !v)}
              >
                Leaf Commitments ({receipt.leaves.length}) {showLeaves ? "▲" : "▼"}
              </div>
              {showLeaves && (
                <div className="receipt-leaves">
                  {receipt.leaves.map((leaf: string, i: number) => (
                    <div key={i} className="receipt-leaf">
                      <span className="receipt-leaf-index">{i + 1}</span>
                      <span className="receipt-leaf-hash">{leaf.substring(0, 24)}…</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Session ID */}
          <div className="receipt-meta receipt-session-id">
            Session: {session.session_id}
          </div>
        </>
      ) : (
        <div className="haic-error">
          No receipt generated.{" "}
          {session.status === "failed"
            ? "Session failed viability checks."
            : "Session may still be processing."}
        </div>
      )}

      <button className="haic-btn primary" style={{ marginTop: "1rem" }} onClick={onReset}>
        Start New Session
      </button>
    </div>
  );
};
