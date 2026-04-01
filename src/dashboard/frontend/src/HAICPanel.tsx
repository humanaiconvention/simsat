/**
 * HAICPanel — top-level HAIC convention panel.
 *
 * Renders as a collapsible section below the existing side-panel.
 * Manages the full lifecycle:
 *   Stimulus → Session → Interview → PRISM → Receipt
 */
import React, { useCallback, useEffect, useState } from "react";
import {
  buildStimulus,
  closeSession,
  createSession,
  getSession,
  type ConventionSession,
  type GroundingStimulus,
} from "./haicApi";
import { InterviewPanel } from "./InterviewPanel";
import { PRISMPanel } from "./PRISMPanel";
import { ReceiptPanel } from "./ReceiptPanel";

type Phase = "idle" | "stimulus" | "session" | "interview" | "prism" | "receipt";

export const HAICPanel: React.FC = () => {
  const [phase, setPhase] = useState<Phase>("idle");
  const [stimulus, setStimulus] = useState<GroundingStimulus | null>(null);
  const [session, setSession] = useState<ConventionSession | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(false);

  // Poll session state while interview is active
  useEffect(() => {
    if (!session || phase !== "interview") return;
    const handle = setInterval(async () => {
      try {
        const updated = await getSession(session.session_id);
        setSession(updated);
      } catch {
        // silent
      }
    }, 3000);
    return () => clearInterval(handle);
  }, [session, phase]);

  const handleBuildStimulus = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const s = await buildStimulus({ include_sentinel: true, include_mapbox: true });
      setStimulus(s);
      setPhase("stimulus");
    } catch (e: unknown) {
      setError(`Stimulus failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }, []);

  const handleStartSession = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const sess = await createSession(
        stimulus
          ? { stimulus_id: stimulus.stimulus_id, auto_stimulus: false }
          : { auto_stimulus: true }
      );
      setSession(sess);
      setPhase("interview");
    } catch (e: unknown) {
      setError(`Session failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }, [stimulus]);

  const handleCloseSession = useCallback(async () => {
    if (!session) return;
    setBusy(true);
    setError(null);
    try {
      const closed = await closeSession(session.session_id);
      setSession(closed);
      // Decide which phase to show next
      if (closed.entropy_delta || closed.geometric_health_after) {
        setPhase("prism");
      } else if (closed.receipt_merkle_root) {
        setPhase("receipt");
      } else {
        setPhase("receipt");
      }
    } catch (e: unknown) {
      setError(`Close failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }, [session]);

  const handleReset = () => {
    setPhase("idle");
    setStimulus(null);
    setSession(null);
    setError(null);
  };

  const handleShowReceipt = () => setPhase("receipt");

  return (
    <div className="haic-panel">
      <div className="haic-header" onClick={() => setCollapsed((c) => !c)}>
        <span className="haic-title">
          <span className="haic-dot" />
          HumanAI Convention
        </span>
        <span className="haic-chevron">{collapsed ? "▶" : "▼"}</span>
      </div>

      {!collapsed && (
        <div className="haic-body">
          {error && <div className="haic-error">{error}</div>}

          {/* Phase indicator */}
          <div className="haic-phases">
            {(["stimulus", "session", "interview", "prism", "receipt"] as Phase[]).map((p) => (
              <div
                key={p}
                className={`haic-phase-dot ${phase === p ? "active" : ""} ${
                  phaseOrder(phase) > phaseOrder(p) ? "done" : ""
                }`}
                title={p}
              />
            ))}
          </div>

          {/* ---- IDLE ---- */}
          {phase === "idle" && (
            <div className="haic-section">
              <p className="haic-description">
                Ground this satellite pass in human lived experience via the{" "}
                <strong>HumanAI Convention</strong> protocol.
              </p>
              <button className="haic-btn primary" disabled={busy} onClick={handleBuildStimulus}>
                {busy ? "Loading…" : "Build Stimulus"}
              </button>
            </div>
          )}

          {/* ---- STIMULUS READY ---- */}
          {phase === "stimulus" && stimulus && (
            <div className="haic-section">
              <StimulusSummary stimulus={stimulus} />
              <button className="haic-btn primary" disabled={busy} onClick={handleStartSession}>
                {busy ? "Starting…" : "Start Convention Session"}
              </button>
              <button className="haic-btn secondary" disabled={busy} onClick={handleBuildStimulus}>
                Refresh Stimulus
              </button>
            </div>
          )}

          {/* ---- INTERVIEW ---- */}
          {phase === "interview" && session && (
            <InterviewPanel
              session={session}
              onSessionUpdate={setSession}
              onClose={handleCloseSession}
              busy={busy}
            />
          )}

          {/* ---- PRISM ---- */}
          {phase === "prism" && session && (
            <PRISMPanel session={session} onContinue={handleShowReceipt} />
          )}

          {/* ---- RECEIPT ---- */}
          {phase === "receipt" && session && (
            <ReceiptPanel session={session} onReset={handleReset} />
          )}

          {/* Reset always available when not idle */}
          {phase !== "idle" && (
            <button className="haic-btn ghost" style={{ marginTop: "0.5rem" }} onClick={handleReset}>
              ↩ New Session
            </button>
          )}
        </div>
      )}
    </div>
  );
};

// ---- Sub-component: stimulus summary ----

const StimulusSummary: React.FC<{ stimulus: GroundingStimulus }> = ({ stimulus }) => {
  const [lon, lat, alt] = stimulus.satellite_position;
  const available = stimulus.images.filter((i) => i.has_image);
  return (
    <div className="haic-stimulus-summary">
      <div className="haic-kv">
        <span className="haic-k">Position</span>
        <span className="haic-v">{lat.toFixed(3)}°, {lon.toFixed(3)}°</span>
      </div>
      <div className="haic-kv">
        <span className="haic-k">Altitude</span>
        <span className="haic-v">{alt.toFixed(0)} km</span>
      </div>
      <div className="haic-kv">
        <span className="haic-k">Perspectives</span>
        <span className="haic-v">{available.length}/{stimulus.images.length} available</span>
      </div>
      {stimulus.images.map((img) => (
        <div key={img.stimulus_type} className="haic-image-badge">
          <span className={`haic-badge ${img.has_image ? "ok" : "na"}`}>
            {img.stimulus_type.replace(/_/g, " ")}
          </span>
          {img.metadata?.cloud_cover != null && (
            <span className="haic-cloud">☁ {img.metadata.cloud_cover.toFixed(0)}%</span>
          )}
        </div>
      ))}
      <p className="haic-context">{stimulus.observation_context}</p>
    </div>
  );
};

// ---- Helpers ----

function phaseOrder(p: Phase): number {
  return ["idle", "stimulus", "session", "interview", "prism", "receipt"].indexOf(p);
}
