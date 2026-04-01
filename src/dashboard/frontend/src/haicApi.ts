/**
 * HAIC API client — talks to the SimSat simulator's /haic/* endpoints
 * (port 9005 in Docker, proxied through Django in dev).
 */
import axios from "axios";

// In Docker/production: requests go through Django proxy at /sim/
// In local dev: hit simulator directly at port 9005 via VITE_SIM_URL override.
const SIM_BASE = (import.meta as any).env?.VITE_SIM_URL ?? "/sim";

const haicClient = axios.create({ baseURL: SIM_BASE });

// ---- Types ----

export interface ObservationMeta {
  satellite_position: [number, number, number];
  timestamp: string;
  footprint?: [number, number, number, number];
  cloud_cover?: number;
  source?: string;
  spectral_bands?: string[];
  elevation_degrees?: number;
  bearing?: number;
  pitch?: number;
  size_km?: number;
  target_visible?: boolean;
  image_available: boolean;
}

export interface StimulusImage {
  stimulus_type: "sentinel_rgb" | "sentinel_multispectral" | "mapbox_perspective" | "composite";
  has_image: boolean;
  image_size_bytes?: number;
  metadata?: ObservationMeta;
}

export interface GroundingStimulus {
  stimulus_id: string;
  created_at: string;
  satellite_position: [number, number, number];
  simulation_timestamp: string;
  images: StimulusImage[];
  location_description: string;
  observation_context: string;
  content_hash: string;
}

export interface ConventionSession {
  session_id: string;
  created_at: string;
  status: string;
  stimulus?: GroundingStimulus;
  interview_turns: { role: "user" | "assistant"; content: string }[];
  participant_id?: string;
  pog_verified: boolean;
  pog_telemetry?: Record<string, unknown>;
  prism_snapshot_before?: Record<string, unknown>;
  prism_snapshot_after?: Record<string, unknown>;
  entropy_delta?: {
    delta_spectral_entropy: number;
    delta_effective_dimension: number;
    reduction_verified: boolean;
    epsilon_threshold: number;
  };
  geometric_health_before?: { composite_score: number; spectral_entropy: number };
  geometric_health_after?: { composite_score: number; spectral_entropy: number };
  viability_gates?: Record<string, boolean>;
  settlement_result?: {
    receipt_id: string;
    merkle_root: string;
    issued_at: string;
    summary: string;
    all_gates_passed: boolean;
    leaves?: string[];
    leaf_count?: number;
    pog_verified?: boolean;
  };
  receipt_merkle_root?: string;
}

export interface SessionSummary {
  session_id: string;
  status: string;
  created_at: string;
  turn_count: number;
  pog_verified: boolean;
  receipt_merkle_root?: string;
}

export interface ObservationWindow {
  window_id: string;
  start_time: string;
  end_time: string;
  satellite_position_start: [number, number, number];
  region_description: string;
}

// ---- API calls ----

export async function buildStimulus(opts?: {
  include_sentinel?: boolean;
  include_mapbox?: boolean;
  sentinel_bands?: string[];
  size_km?: number;
}): Promise<GroundingStimulus> {
  const res = await haicClient.post("/haic/stimulus", opts ?? {});
  return res.data;
}

export async function createSession(opts?: {
  stimulus_id?: string;
  participant_id?: string;
  auto_stimulus?: boolean;
}): Promise<ConventionSession> {
  const res = await haicClient.post("/haic/session", opts ?? { auto_stimulus: true });
  return res.data;
}

export async function getSession(sessionId: string): Promise<ConventionSession> {
  const res = await haicClient.get(`/haic/session/${sessionId}`);
  return res.data;
}

export async function listSessions(): Promise<SessionSummary[]> {
  const res = await haicClient.get("/haic/sessions");
  return res.data.sessions ?? [];
}

export async function submitTurn(
  sessionId: string,
  content: string,
  telemetry?: { client_start_ms: number; client_end_ms: number; keystroke_intervals: number[] }
): Promise<{ assistant_response: string; session_status: string; total_turns: number }> {
  const res = await haicClient.post(`/haic/session/${sessionId}/turn`, {
    content,
    ...telemetry,
  });
  return res.data;
}

export async function closeSession(sessionId: string): Promise<ConventionSession> {
  const res = await haicClient.post(`/haic/session/${sessionId}/close`);
  return res.data;
}

export async function getReceipt(sessionId: string): Promise<ConventionSession["settlement_result"] & { merkle_root: string }> {
  const res = await haicClient.get(`/haic/session/${sessionId}/receipt`);
  return res.data;
}

export async function getObservationWindows(count = 5): Promise<ObservationWindow[]> {
  const res = await haicClient.get(`/haic/windows?count=${count}`);
  return res.data.windows ?? [];
}

export async function haicHealth(): Promise<Record<string, unknown>> {
  const res = await haicClient.get("/haic/health");
  return res.data;
}

/** Returns a URL to fetch a stimulus image as PNG for display in an <img> tag. */
export function stimulusImageUrl(stimulusId: string, imageIndex: number): string {
  return `${SIM_BASE}/haic/stimulus/${stimulusId}/image/${imageIndex}`;
}

/** Fetch a stimulus image as a base64 data URI for inline display. */
export async function fetchStimulusImageDataUrl(
  stimulusId: string,
  imageIndex: number
): Promise<string | null> {
  try {
    const url = stimulusImageUrl(stimulusId, imageIndex);
    const resp = await fetch(url);
    if (!resp.ok) return null;
    const blob = await resp.blob();
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
}
