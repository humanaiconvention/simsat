import React, { useState } from "react";

interface ModelInfo {
  id: string;
  name: string;
  family: string;
  file: string;
}

const MODELS: ModelInfo[] = [
  { id: "gemma4-2b", name: "Gemma 4 2B (HAIC)", family: "Transformer + AltUp", file: "/static/arch/gemma4-2b_arch.html" },
  { id: "lfm2-8b", name: "LFM2 8B A1B", family: "MoE + ShortConv", file: "/static/arch/lfm2-8b_arch.html" },
  { id: "qwen3.5-2b", name: "Qwen3.5 2B", family: "Hybrid SSM", file: "/static/arch/qwen3.5-2b_arch.html" },
  { id: "nemotron-4b", name: "Nemotron 4B", family: "Hybrid SSM", file: "/static/arch/nemotron-4b_arch.html" },
  { id: "llama3.1-8b", name: "Llama 3.1 8B", family: "Dense Transformer", file: "/static/arch/llama3.1-8b_arch.html" },
  { id: "qwen3-8b", name: "Qwen3 8B", family: "Dense Transformer", file: "/static/arch/qwen3-8b_arch.html" },
  { id: "phi4-mini", name: "Phi-4 Mini", family: "Dense Transformer", file: "/static/arch/phi4-mini_arch.html" },
  { id: "ministral-3b", name: "Ministral 3B", family: "Dense Transformer", file: "/static/arch/ministral-3b_arch.html" },
  { id: "smollm3-3b", name: "SmolLM3 3B", family: "Dense Transformer", file: "/static/arch/smollm3-3b_arch.html" },
];

interface Props {
  open: boolean;
  onClose: () => void;
}

export const ArchitecturePanel: React.FC<Props> = ({ open, onClose }) => {
  const [selected, setSelected] = useState(MODELS[0]);

  if (!open) return null;

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 1000,
      background: "rgba(0,0,0,0.85)", display: "flex",
    }}>
      {/* Sidebar */}
      <div style={{
        width: 260, background: "#04060f", borderRight: "1px solid #1a2a44",
        padding: "16px 12px", overflowY: "auto",
        fontFamily: "'Courier New', monospace", fontSize: 10,
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <span style={{ color: "#c8d8ff", fontSize: 12, letterSpacing: 1 }}>ARCHITECTURES</span>
          <button onClick={onClose} style={{
            background: "transparent", border: "1px solid #334", color: "#668",
            borderRadius: 4, padding: "2px 8px", cursor: "pointer", fontFamily: "inherit",
          }}>ESC</button>
        </div>
        {MODELS.map((m) => (
          <button
            key={m.id}
            onClick={() => setSelected(m)}
            style={{
              display: "block", width: "100%", textAlign: "left",
              background: selected.id === m.id ? "rgba(10,21,48,0.7)" : "transparent",
              border: `1px solid ${selected.id === m.id ? "#2244aa" : "#152030"}`,
              borderRadius: 5, padding: "8px 10px", marginBottom: 6,
              cursor: "pointer", fontFamily: "inherit", color: "#8899bb", fontSize: 10,
            }}
          >
            <div style={{ color: "#b8c8ee", fontWeight: "bold", marginBottom: 2 }}>{m.name}</div>
            <div style={{ color: "#445566", fontSize: 9 }}>{m.family}</div>
          </button>
        ))}
        <div style={{ marginTop: 12, color: "#1a2a3a", fontSize: 8, lineHeight: 1.7 }}>
          Source: GGUF tensor info sections<br />
          All shapes verified, no estimates
        </div>
      </div>
      {/* Viewer */}
      <div style={{ flex: 1 }}>
        <iframe
          key={selected.id}
          src={selected.file}
          style={{ width: "100%", height: "100%", border: "none", background: "#04040f" }}
          title={`${selected.name} Architecture`}
        />
      </div>
    </div>
  );
};
