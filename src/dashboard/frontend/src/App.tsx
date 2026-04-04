import React, { Suspense, lazy, useEffect, useState } from "react";
import { TelemetryPoint, fetchRecentTelemetry } from "./api";
import { TelemetryPanel } from "./TelemetryPanel";
import { SimulationControls } from "./SimulationControls";
import { EncounterPanel } from "./EncounterPanel";
import { HAICPanel } from "./HAICPanel";
import { ArchitecturePanel } from "./ArchitecturePanel";

const GlobeView = lazy(() =>
  import("./GlobeView").then((module) => ({ default: module.GlobeView })),
);

export const App: React.FC = () => {
  const [telemetry, setTelemetry] = useState<TelemetryPoint[]>([]);

  // Poll telemetry ~1 Hz
  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const data = await fetchRecentTelemetry();
        if (!cancelled) {
          setTelemetry(data);
        }
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error("Failed to fetch telemetry", err);
      }
    };

    poll();
    const handle = setInterval(poll, 1000);

    return () => {
      cancelled = true;
      clearInterval(handle);
    };
  }, []);

  const latest = telemetry[0] ?? null;
  const [archOpen, setArchOpen] = useState(false);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Satellite Simulation Dashboard</h1>
        <span className="app-header-badge">HAIC Convention Layer</span>
        <button
          className="app-header-badge"
          style={{ cursor: "pointer", marginLeft: 8, background: "#1a1a3a", border: "1px solid #334" }}
          onClick={() => setArchOpen(true)}
        >
          Architecture Explorer
        </button>
      </header>
      <ArchitecturePanel open={archOpen} onClose={() => setArchOpen(false)} />
      <main className="app-main">
        <section className="globe-section">
          <Suspense fallback={<div className="globe-loading">Loading globe…</div>}>
            <GlobeView telemetry={telemetry} />
          </Suspense>
        </section>
        <section className="side-panel">
          <TelemetryPanel latest={latest} />
          <SimulationControls />
          <EncounterPanel />
          <HAICPanel />
        </section>
      </main>
    </div>
  );
};
