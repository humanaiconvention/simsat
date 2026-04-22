import axios from "axios";

const SIM_BASE = (import.meta as any).env?.VITE_SIM_URL ?? "/sim";

const encounterClient = axios.create({ baseURL: SIM_BASE });

export interface EncounterTarget {
  target_id: string;
  label: string;
  lon: number;
  lat: number;
  priority: number;
  size_km: number;
  tags: string[];
  metadata: Record<string, unknown>;
}

export interface EncounterTargetsResponse {
  targets: EncounterTarget[];
  scenario_packs: string[];
}

export interface EncounterGeometry {
  elevation_degrees: number;
  off_nadir_degrees: number;
  slant_range_km: number;
  target_visible: boolean;
  bearing?: number | null;
  pitch?: number | null;
}

export interface EncounterWindow {
  window_id: string;
  target_id: string;
  encounter_type: string;
  start_time: string;
  end_time: string;
  peak_time: string;
  satellite_position_peak: [number, number, number];
  target_position: [number, number];
  geometry: EncounterGeometry;
  target_priority: number;
  duration_seconds: number;
  pre_rank_score: number;
}

export interface EncounterDecision {
  decision_id: string;
  window_id: string;
  target_id: string;
  created_at: string;
  policy_version: string;
  scaffold_score: number;
  learned_score: number;
  analytic_score: number;
  trust_score: number;
  residual_score: number;
  combined_score: number;
  trust_band: "high" | "medium" | "low";
  trust_details: Record<string, number>;
  action: "accept" | "defer" | "refine" | "skip";
  reason_codes: string[];
  needs_refinement: boolean;
  refinement_reason?: string | null;
  stimulus_id?: string | null;
  artifact_id?: string | null;
}

export interface PlannerEvaluationSummary {
  planner_id: string;
  policy_version: string;
  model_id: string;
  evaluated_windows: number;
  action_counts: Record<string, number>;
  mean_scaffold_score: number;
  mean_combined_score: number;
  mean_trust_score: number;
  materialization_attempts: number;
  materialization_successes: number;
  materialization_yield: number;
  top_decision_ids: string[];
}

export interface EncounterEvaluation {
  evaluation_id: string;
  created_at: string;
  parameters: Record<string, unknown>;
  scenario_pack: string;
  window_count: number;
  sample_window_ids: string[];
  scaffold_summary?: PlannerEvaluationSummary;
  trust_summary?: PlannerEvaluationSummary;
  action_transition_counts: Record<string, number>;
  decision_deltas: DecisionDelta[];
}

export interface DecisionDelta {
  window_id: string;
  target_id: string;
  target_label: string;
  scenario_pack: string;
  scaffold_action: "accept" | "defer" | "refine" | "skip";
  trust_action: "accept" | "defer" | "refine" | "skip";
  scaffold_score: number;
  trust_score: number;
  trust_combined_score: number;
  score_delta: number;
  action_changed: boolean;
  changed_to_refine: boolean;
  refinement_reason?: string | null;
  scaffold_reason_codes: string[];
  trust_reason_codes: string[];
}

export async function listEncounterTargets(): Promise<EncounterTargetsResponse> {
  const res = await encounterClient.get("/encounter/targets");
  return {
    targets: res.data.targets ?? [],
    scenario_packs: res.data.scenario_packs ?? [],
  };
}

export async function createEncounterTarget(target: EncounterTarget): Promise<EncounterTarget> {
  const res = await encounterClient.post("/encounter/targets", target);
  return res.data;
}

export async function updateEncounterTarget(targetId: string, target: EncounterTarget): Promise<EncounterTarget> {
  const res = await encounterClient.put(`/encounter/targets/${targetId}`, target);
  return res.data;
}

export async function deleteEncounterTarget(targetId: string): Promise<void> {
  await encounterClient.delete(`/encounter/targets/${targetId}`);
}

export async function getEncounterWindows(opts?: {
  hours?: number;
  step_seconds?: number;
  top_k?: number;
  scenario_pack?: string;
}): Promise<EncounterWindow[]> {
  const res = await encounterClient.get("/encounter/windows", { params: opts ?? {} });
  return res.data.windows ?? [];
}

export async function planEncounters(opts?: {
  hours?: number;
  step_seconds?: number;
  top_k?: number;
  scenario_pack?: string;
}): Promise<EncounterDecision[]> {
  const res = await encounterClient.post("/encounter/plan", opts ?? {});
  return res.data.decisions ?? [];
}

export async function listEncounterDecisions(limit = 10): Promise<EncounterDecision[]> {
  const res = await encounterClient.get("/encounter/decisions", { params: { limit } });
  return res.data.decisions ?? [];
}

export async function evaluateEncounters(opts?: {
  hours?: number;
  step_seconds?: number;
  top_k?: number;
  materialize_top_k?: number;
  scenario_pack?: string;
}): Promise<EncounterEvaluation> {
  const res = await encounterClient.post("/encounter/evaluate", opts ?? {});
  return res.data;
}

export async function listEncounterEvaluations(limit = 5, scenario_pack?: string): Promise<EncounterEvaluation[]> {
  const res = await encounterClient.get("/encounter/evaluations", { params: { limit, scenario_pack } });
  return res.data.evaluations ?? [];
}

export async function materializeEncounterDecision(decisionId: string): Promise<{ stimulus_id: string; decision_id: string }> {
  const res = await encounterClient.post(`/encounter/decision/${decisionId}/materialize`);
  return res.data;
}
