import React, { useEffect, useState } from "react";
import {
  createEncounterTarget,
  deleteEncounterTarget,
  evaluateEncounters,
  getEncounterWindows,
  listEncounterDecisions,
  listEncounterEvaluations,
  listEncounterTargets,
  materializeEncounterDecision,
  planEncounters,
  updateEncounterTarget,
  type EncounterDecision,
  type EncounterEvaluation,
  type EncounterTarget,
  type EncounterWindow,
} from "./encounterApi";

type TargetFormState = {
  target_id: string;
  label: string;
  lon: string;
  lat: string;
  priority: string;
  size_km: string;
  tags: string;
  scenario_pack: string;
  scenario_role: string;
};

const DEFAULT_FORM: TargetFormState = {
  target_id: "",
  label: "",
  lon: "",
  lat: "",
  priority: "0.5",
  size_km: "5",
  tags: "",
  scenario_pack: "",
  scenario_role: "",
};

export const EncounterPanel: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [targets, setTargets] = useState<EncounterTarget[]>([]);
  const [scenarioPacks, setScenarioPacks] = useState<string[]>([]);
  const [windows, setWindows] = useState<EncounterWindow[]>([]);
  const [decisions, setDecisions] = useState<EncounterDecision[]>([]);
  const [evaluation, setEvaluation] = useState<EncounterEvaluation | null>(null);
  const [hours, setHours] = useState("6");
  const [topK, setTopK] = useState("5");
  const [scenarioPack, setScenarioPack] = useState("all");
  const [editingTargetId, setEditingTargetId] = useState<string | null>(null);
  const [editingMetadata, setEditingMetadata] = useState<Record<string, unknown>>({});
  const [form, setForm] = useState<TargetFormState>(DEFAULT_FORM);

  const loadPanel = async () => {
    const nextHours = Number(hours) || 6;
    const nextTopK = Number(topK) || 5;
    const [loadedTargets, loadedWindows, loadedDecisions, loadedEvaluations] = await Promise.all([
      listEncounterTargets(),
      getEncounterWindows({
        hours: nextHours,
        step_seconds: 120,
        top_k: nextTopK,
        scenario_pack: scenarioPack === "all" ? undefined : scenarioPack,
      }),
      listEncounterDecisions(nextTopK),
      listEncounterEvaluations(5, scenarioPack === "all" ? undefined : scenarioPack),
    ]);
    setTargets(loadedTargets.targets);
    setScenarioPacks(loadedTargets.scenario_packs);
    setWindows(loadedWindows);
    setDecisions(loadedDecisions);
    setEvaluation(
      loadedEvaluations[0] ?? null,
    );
  };

  useEffect(() => {
    loadPanel().catch((err: unknown) => {
      setError(`Encounter panel failed to load: ${(err as Error).message}`);
    });
  }, [scenarioPack]);

  const resetForm = () => {
    setEditingTargetId(null);
    setEditingMetadata({});
    setForm(DEFAULT_FORM);
  };

  const handleRefresh = async () => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await loadPanel();
    } catch (err: unknown) {
      setError(`Refresh failed: ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const handlePlan = async () => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const planned = await planEncounters({
        hours: Number(hours) || 6,
        step_seconds: 120,
        top_k: Number(topK) || 5,
        scenario_pack: scenarioPack === "all" ? undefined : scenarioPack,
      });
      setDecisions(planned);
      setNotice(`Planned ${planned.length} encounter decisions.`);
    } catch (err: unknown) {
      setError(`Planning failed: ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const handleEvaluate = async () => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const nextEvaluation = await evaluateEncounters({
        hours: Number(hours) || 6,
        step_seconds: 120,
        top_k: Number(topK) || 5,
        materialize_top_k: Math.min(Number(topK) || 5, 3),
        scenario_pack: scenarioPack === "all" ? undefined : scenarioPack,
      });
      setEvaluation(nextEvaluation);
      setNotice(`Evaluated ${nextEvaluation.window_count} shared windows across scaffold and WCLI-trust planners.`);
    } catch (err: unknown) {
      setError(`Evaluation failed: ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const handleSaveTarget = async () => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const payload: EncounterTarget = {
        target_id: form.target_id.trim(),
        label: form.label.trim(),
        lon: Number(form.lon),
        lat: Number(form.lat),
        priority: Number(form.priority),
        size_km: Number(form.size_km),
        tags: form.tags
          .split(",")
          .map((tag) => tag.trim())
          .filter(Boolean),
        metadata: {
          ...editingMetadata,
          scenario_pack: form.scenario_pack.trim() || undefined,
          scenario_role: form.scenario_role.trim() || undefined,
        },
      };

      if (editingTargetId) {
        await updateEncounterTarget(editingTargetId, payload);
        setNotice(`Updated target ${payload.target_id}.`);
      } else {
        await createEncounterTarget(payload);
        setNotice(`Created target ${payload.target_id}.`);
      }

      resetForm();
      await loadPanel();
    } catch (err: unknown) {
      setError(`Target save failed: ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const handleEditTarget = (target: EncounterTarget) => {
    setEditingTargetId(target.target_id);
    setEditingMetadata({ ...(target.metadata ?? {}) });
    setForm({
      target_id: target.target_id,
      label: target.label,
      lon: String(target.lon),
      lat: String(target.lat),
      priority: String(target.priority),
      size_km: String(target.size_km),
      tags: target.tags.join(", "),
      scenario_pack: String(target.metadata?.scenario_pack ?? ""),
      scenario_role: String(target.metadata?.scenario_role ?? ""),
    });
  };

  const handleDeleteTarget = async (targetId: string) => {
    if (!window.confirm(`Delete target ${targetId}?`)) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await deleteEncounterTarget(targetId);
      if (editingTargetId === targetId) {
        resetForm();
      }
      await loadPanel();
      setNotice(`Deleted target ${targetId}.`);
    } catch (err: unknown) {
      setError(`Delete failed: ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const handleMaterialize = async (decisionId: string) => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const result = await materializeEncounterDecision(decisionId);
      setNotice(`Materialized ${decisionId} as stimulus ${result.stimulus_id}.`);
      setDecisions(await listEncounterDecisions(Number(topK) || 5));
    } catch (err: unknown) {
      setError(`Materialization failed: ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="encounter-panel">
      <div className="encounter-header" onClick={() => setCollapsed((value) => !value)}>
        <span className="encounter-title">
          <span className="encounter-dot" />
          Encounter Planner
        </span>
        <span className="encounter-chevron">{collapsed ? "▶" : "▼"}</span>
      </div>

      {!collapsed && (
        <div className="encounter-body">
          <p className="encounter-description">
            Shadow-plan high-value observation windows with a WCLI-style scaffold, trust score, and
            trust-gated refinement path.
          </p>

          {error && <div className="encounter-error">{error}</div>}
          {notice && <div className="encounter-notice">{notice}</div>}

          <div className="encounter-controls">
            <label>
              <span>Hours</span>
              <input value={hours} onChange={(e) => setHours(e.target.value)} />
            </label>
            <label>
              <span>Top K</span>
              <input value={topK} onChange={(e) => setTopK(e.target.value)} />
            </label>
            <label>
              <span>Scenario</span>
              <select value={scenarioPack} onChange={(e) => setScenarioPack(e.target.value)}>
                <option value="all">all</option>
                {scenarioPacks.map((pack) => (
                  <option key={pack} value={pack}>
                    {pack}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="encounter-button-row">
            <button className="encounter-btn primary" disabled={busy} onClick={handleRefresh}>
              {busy ? "Working…" : "Refresh"}
            </button>
            <button className="encounter-btn secondary" disabled={busy} onClick={handlePlan}>
              Plan
            </button>
            <button className="encounter-btn secondary" disabled={busy} onClick={handleEvaluate}>
              Evaluate
            </button>
          </div>

          {evaluation && (
            <section className="encounter-section">
              <div className="encounter-section-header">
                <h3>Challenge Eval</h3>
                <span>{new Date(evaluation.created_at).toLocaleTimeString()}</span>
              </div>
              <div className="encounter-card-meta">
                Scenario {evaluation.scenario_pack} · Windows {evaluation.window_count}
              </div>
              <div className="encounter-eval-grid">
                <div className="encounter-card">
                  <div className="encounter-card-title">Scaffold</div>
                  <div className="encounter-card-meta">
                    Accept {evaluation.scaffold_summary?.action_counts.accept ?? 0} · Defer {evaluation.scaffold_summary?.action_counts.defer ?? 0} · Refine {evaluation.scaffold_summary?.action_counts.refine ?? 0}
                  </div>
                  <div className="encounter-card-meta">
                    Mean final {(evaluation.scaffold_summary?.mean_combined_score ?? 0).toFixed(2)} · Yield {((evaluation.scaffold_summary?.materialization_yield ?? 0) * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="encounter-card">
                  <div className="encounter-card-title">WCLI Trust</div>
                  <div className="encounter-card-meta">
                    Accept {evaluation.trust_summary?.action_counts.accept ?? 0} · Defer {evaluation.trust_summary?.action_counts.defer ?? 0} · Refine {evaluation.trust_summary?.action_counts.refine ?? 0}
                  </div>
                  <div className="encounter-card-meta">
                    Mean final {(evaluation.trust_summary?.mean_combined_score ?? 0).toFixed(2)} · Yield {((evaluation.trust_summary?.materialization_yield ?? 0) * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
              <div className="encounter-card">
                <div className="encounter-card-title">Action Transitions</div>
                <div className="encounter-transition-list">
                  {Object.entries(evaluation.action_transition_counts).map(([key, value]) => (
                    <span key={key} className="encounter-transition-tag">
                      {key}: {value}
                    </span>
                  ))}
                </div>
              </div>
              {evaluation.decision_deltas.length > 0 && (
                <div className="encounter-card">
                  <div className="encounter-card-title">Decision Deltas</div>
                  <div className="encounter-list compact">
                    {evaluation.decision_deltas.map((delta) => (
                      <div key={delta.window_id} className="encounter-card delta">
                        <div className="encounter-card-row">
                          <div>
                            <div className="encounter-card-title">{delta.target_label || delta.target_id}</div>
                            <div className="encounter-card-subtitle">{delta.scenario_pack}</div>
                          </div>
                          <span className={`encounter-action ${delta.trust_action}`}>{delta.scaffold_action} → {delta.trust_action}</span>
                        </div>
                        <div className="encounter-card-meta">
                          Scaffold {delta.scaffold_score.toFixed(2)} · Final {delta.trust_combined_score.toFixed(2)} · Trust {delta.trust_score.toFixed(2)}
                        </div>
                        <div className="encounter-card-meta">
                          Δ {delta.score_delta >= 0 ? "+" : ""}{delta.score_delta.toFixed(2)}
                          {delta.refinement_reason ? ` · ${delta.refinement_reason}` : ""}
                        </div>
                        <div className="encounter-card-meta">
                          {delta.trust_reason_codes.join(", ")}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>
          )}

          <section className="encounter-section">
            <div className="encounter-section-header">
              <h3>Targets</h3>
              <span>{targets.length}</span>
            </div>

            <div className="encounter-form-grid">
              <input
                placeholder="target_id"
                value={form.target_id}
                disabled={editingTargetId !== null}
                onChange={(e) => setForm((current) => ({ ...current, target_id: e.target.value }))}
              />
              <input
                placeholder="Label"
                value={form.label}
                onChange={(e) => setForm((current) => ({ ...current, label: e.target.value }))}
              />
              <input
                placeholder="Longitude"
                value={form.lon}
                onChange={(e) => setForm((current) => ({ ...current, lon: e.target.value }))}
              />
              <input
                placeholder="Latitude"
                value={form.lat}
                onChange={(e) => setForm((current) => ({ ...current, lat: e.target.value }))}
              />
              <input
                placeholder="Priority"
                value={form.priority}
                onChange={(e) => setForm((current) => ({ ...current, priority: e.target.value }))}
              />
              <input
                placeholder="Size km"
                value={form.size_km}
                onChange={(e) => setForm((current) => ({ ...current, size_km: e.target.value }))}
              />
              <input
                className="encounter-form-wide"
                placeholder="tags, comma-separated"
                value={form.tags}
                onChange={(e) => setForm((current) => ({ ...current, tags: e.target.value }))}
              />
              <input
                placeholder="scenario_pack"
                value={form.scenario_pack}
                onChange={(e) => setForm((current) => ({ ...current, scenario_pack: e.target.value }))}
              />
              <input
                placeholder="scenario_role"
                value={form.scenario_role}
                onChange={(e) => setForm((current) => ({ ...current, scenario_role: e.target.value }))}
              />
            </div>

            <div className="encounter-button-row compact">
              <button className="encounter-btn primary" disabled={busy} onClick={handleSaveTarget}>
                {editingTargetId ? "Update Target" : "Add Target"}
              </button>
              <button className="encounter-btn ghost" disabled={busy} onClick={resetForm}>
                Clear
              </button>
            </div>

            <div className="encounter-list">
              {targets.map((target) => (
                <div key={target.target_id} className="encounter-card">
                  <div className="encounter-card-row">
                    <div>
                      <div className="encounter-card-title">{target.label}</div>
                      <div className="encounter-card-subtitle">{target.target_id}</div>
                    </div>
                    <span className="encounter-priority">{target.priority.toFixed(2)}</span>
                  </div>
                  <div className="encounter-card-meta">
                    {target.lat.toFixed(3)}°, {target.lon.toFixed(3)}° · {target.size_km.toFixed(1)} km
                  </div>
                  <div className="encounter-card-meta">
                    {String(target.metadata?.scenario_pack ?? "default")}
                    {target.metadata?.scenario_role ? ` · ${String(target.metadata.scenario_role)}` : ""}
                  </div>
                  <div className="encounter-inline-actions">
                    <button className="encounter-link-btn" onClick={() => handleEditTarget(target)}>
                      Edit
                    </button>
                    <button className="encounter-link-btn danger" onClick={() => handleDeleteTarget(target.target_id)}>
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="encounter-section">
            <div className="encounter-section-header">
              <h3>Preview Windows</h3>
              <span>{windows.length}</span>
            </div>
            <div className="encounter-list">
              {windows.map((windowItem) => (
                <div key={windowItem.window_id} className="encounter-card">
                  <div className="encounter-card-row">
                    <div className="encounter-card-title">{windowItem.target_id}</div>
                    <span className="encounter-score">{windowItem.pre_rank_score.toFixed(2)}</span>
                  </div>
                  <div className="encounter-card-meta">
                    Peak {new Date(windowItem.peak_time).toLocaleString()} · {windowItem.duration_seconds.toFixed(0)}s
                  </div>
                  <div className="encounter-card-meta">
                    Elev {windowItem.geometry.elevation_degrees.toFixed(1)}° · Range {windowItem.geometry.slant_range_km.toFixed(0)} km
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="encounter-section">
            <div className="encounter-section-header">
              <h3>Decisions</h3>
              <span>{decisions.length}</span>
            </div>
            <div className="encounter-list">
              {decisions.map((decision) => (
                <div key={decision.decision_id} className="encounter-card">
                  <div className="encounter-card-row">
                    <div className="encounter-card-title">{decision.target_id}</div>
                    <span className={`encounter-action ${decision.action}`}>{decision.action}</span>
                  </div>
                  <div className="encounter-card-meta">
                    Scaffold {decision.scaffold_score.toFixed(2)} · Trust {decision.trust_score.toFixed(2)} · Final {decision.combined_score.toFixed(2)}
                  </div>
                  <div className="encounter-card-meta">
                    Support {decision.learned_score.toFixed(2)} · Residual {decision.residual_score.toFixed(3)} · {decision.trust_band} trust
                  </div>
                  <div className="encounter-card-meta">
                    {decision.reason_codes.join(", ")}
                    {decision.refinement_reason ? ` · ${decision.refinement_reason}` : ""}
                  </div>
                  {decision.stimulus_id ? (
                    <div className="encounter-stimulus-tag">Stimulus {decision.stimulus_id}</div>
                  ) : (
                    decision.action !== "skip" && (
                      <button
                        className="encounter-btn ghost"
                        disabled={busy}
                        onClick={() => handleMaterialize(decision.decision_id)}
                      >
                        Materialize
                      </button>
                    )
                  )}
                </div>
              ))}
            </div>
          </section>
        </div>
      )}
    </div>
  );
};
