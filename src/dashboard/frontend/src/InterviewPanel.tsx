/**
 * InterviewPanel — the grounding interview UI.
 *
 * Shows the conversation, captures typing telemetry for PoG,
 * and submits turns to the HAIC API.
 */
import React, { useEffect, useRef, useState } from "react";
import { submitTurn, type ConventionSession } from "./haicApi";
import { StimulusViewer } from "./StimulusViewer";

interface Props {
  session: ConventionSession;
  onSessionUpdate: (s: ConventionSession) => void;
  onClose: () => void;
  busy: boolean;
}

export const InterviewPanel: React.FC<Props> = ({ session, onSessionUpdate, onClose, busy }) => {
  const [input, setInput] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  // PoG telemetry tracking
  const startMsRef = useRef<number>(0);
  const keystrokesRef = useRef<number[]>([]);
  const lastKeyTimeRef = useRef<number>(0);

  const turns = session.interview_turns;
  const userTurnCount = turns.filter((t) => t.role === "user").length;
  const canClose = userTurnCount >= 2;

  // Auto-scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    const now = Date.now();
    if (lastKeyTimeRef.current > 0) {
      keystrokesRef.current.push(now - lastKeyTimeRef.current);
    }
    lastKeyTimeRef.current = now;
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFocus = () => {
    startMsRef.current = Date.now();
    keystrokesRef.current = [];
    lastKeyTimeRef.current = 0;
  };

  const handleSubmit = async () => {
    const text = input.trim();
    if (!text || submitting) return;

    const endMs = Date.now();
    const telemetry = {
      client_start_ms: startMsRef.current || endMs - 5000,
      client_end_ms: endMs,
      keystroke_intervals: [...keystrokesRef.current],
    };

    setSubmitting(true);
    setError(null);
    try {
      const result = await submitTurn(session.session_id, text, telemetry);
      // Optimistically update turns
      const updated: ConventionSession = {
        ...session,
        status: result.session_status,
        interview_turns: [
          ...session.interview_turns,
          { role: "user", content: text },
          { role: "assistant", content: result.assistant_response },
        ],
      };
      onSessionUpdate(updated);
      setInput("");
      keystrokesRef.current = [];
    } catch (e: unknown) {
      setError(`Submit failed: ${(e as Error).message}`);
    } finally {
      setSubmitting(false);
    }
  };

  // Show opening message if no turns yet
  const showOpening = turns.length === 0;

  return (
    <div className="haic-interview">
      <div className="haic-interview-header">
        <span className="haic-interview-title">Grounding Interview</span>
        <span className="haic-turn-count">{userTurnCount} turns</span>
      </div>

      {/* Stimulus imagery */}
      {session.stimulus && (
        <StimulusViewer stimulus={session.stimulus} />
      )}

      {/* Chat */}
      <div className="haic-chat">
        {showOpening && (
          <div className="haic-msg assistant">
            <span className="haic-msg-role">Interviewer</span>
            <p>
              Welcome. {session.stimulus
                ? `I have loaded a satellite observation: ${session.stimulus.observation_context}`
                : "Let's begin a grounding session."
              }{" "}
              Take a moment to look at the image, then describe what you see.
            </p>
          </div>
        )}

        {turns.map((turn, i) => (
          <div key={i} className={`haic-msg ${turn.role}`}>
            <span className="haic-msg-role">
              {turn.role === "user" ? "You" : "Interviewer"}
            </span>
            <p>{turn.content}</p>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      {error && <div className="haic-error">{error}</div>}

      {/* Input */}
      <div className="haic-input-row">
        <textarea
          className="haic-textarea"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={handleFocus}
          placeholder="Describe what you observe… (Enter to send)"
          rows={3}
          disabled={submitting || busy}
        />
        <button
          className="haic-btn primary"
          disabled={!input.trim() || submitting || busy}
          onClick={handleSubmit}
        >
          {submitting ? "…" : "Send"}
        </button>
      </div>

      {/* PoG indicator */}
      <div className="haic-pog-hint">
        <span className="haic-pog-dot" />
        Typing cadence recorded for Proof-of-Grounding
      </div>

      {/* Close button */}
      <button
        className={`haic-btn ${canClose ? "secondary" : "ghost"}`}
        disabled={!canClose || busy || submitting}
        onClick={onClose}
        title={canClose ? "End interview and run PRISM analysis" : "Complete at least 2 turns first"}
      >
        {busy ? "Analysing…" : "Close & Analyse"}
      </button>
      {!canClose && (
        <p className="haic-hint">Complete at least 2 turns to close the session.</p>
      )}
    </div>
  );
};
