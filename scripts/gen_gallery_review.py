#!/usr/bin/env python3
"""Generate gallery_review.html — a single-page review tool showing all
unreviewed traces in a grid with thumbnails, label buttons, and keyboard
shortcuts.

Output: a self-contained HTML file with embedded JPEG thumbnails. Saves a
labels.json on download which the user feeds to scripts/apply_labels.py.
"""
from __future__ import annotations
import json
import base64
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
from operator_review import InProcessClient  # type: ignore


def main() -> None:
    client = InProcessClient()
    data = client.get_json("/observation-vla/traces", limit=1000)
    traces = data.get("traces", [])
    print(f"Total traces in pool: {len(traces)}")

    thumb_dir = pathlib.Path("/tmp/gallery_thumbs")
    asset_dir = REPO_ROOT / "review_queue_assets"

    cards = []
    for t in traces:
        tid = t.get("trace_id")
        if not tid:
            continue
        scenario = t.get("scenario_pack", "?")
        # Pull detailed metadata
        d = client.get_json(f"/observation-vla/trace/{tid}")
        trace = d.get("trace", {})
        sample = trace.get("sample", {})
        probe = sample.get("probe", {})
        geom = sample.get("geometry", {})
        decision_after = trace.get("decision_after") or trace.get("decision_before") or {}

        # Skip if already reviewed
        cur_outcome = d.get("current_outcome") or {}
        if cur_outcome.get("label_source") == "operator_review":
            continue

        # Find thumbnail
        thumb_path = thumb_dir / f"{tid}.jpg"
        if thumb_path.exists():
            img_b64 = "data:image/jpeg;base64," + base64.b64encode(thumb_path.read_bytes()).decode()
        else:
            img_b64 = None

        cards.append({
            "id": tid,
            "scenario": scenario,
            "target": sample.get("target_label", "?"),
            "target_id": sample.get("target_id", "?"),
            "cloud": round(probe.get("sentinel_cloud_cover", 0) or 0, 1),
            "elev": round(geom.get("elevation_degrees", 0) or 0, 1),
            "off_nadir": round(geom.get("off_nadir_degrees", 0) or 0, 1),
            "visible": geom.get("target_visible", False),
            "date": (probe.get("sentinel_datetime") or "?")[:10],
            "auto": decision_after.get("action", "?"),
            "img": img_b64,
        })

    print(f"Building gallery for {len(cards)} unreviewed traces")
    cards_json = json.dumps(cards)

    html = HTML_TEMPLATE.replace("__CARDS_JSON__", cards_json).replace("__TOTAL_COUNT__", str(len(cards)))
    out = REPO_ROOT / "gallery_review.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Open in your browser; review all; click 'Save Labels' to download labels.json;")
    print(f"then run: python scripts/apply_labels.py <path-to-labels.json>")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SimSat Gallery Review</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0a0c14; color: #e2e8f0; font-family: 'Segoe UI', system-ui, sans-serif; }

  header {
    background: #14172a; border-bottom: 1px solid #2d3148;
    padding: 12px 20px;
    position: sticky; top: 0; z-index: 100;
    display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
  }
  header h1 { font-size: 14px; font-weight: 700; color: #a0aec0; letter-spacing: 0.06em; text-transform: uppercase; flex-shrink: 0; }
  #progress-bar-wrap { flex: 1; min-width: 240px; background: #2d3148; border-radius: 4px; height: 6px; overflow: hidden; }
  #progress-bar { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); width: 0%; transition: width 0.3s; }
  .progress-counts { font-size: 12px; color: #a0aec0; white-space: nowrap; }
  .progress-counts span { font-weight: 700; }
  .ct-accept { color: #68d391; }
  .ct-refine { color: #63b3ed; }
  .ct-defer  { color: #f6ad55; }
  .ct-skip   { color: #fc8181; }

  .filter-bar {
    background: #14172a; padding: 10px 20px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
    border-bottom: 1px solid #2d3148;
  }
  .filter-bar label { font-size: 12px; color: #718096; }
  .filter-bar select, .filter-bar input { background: #0f1117; color: #e2e8f0; border: 1px solid #2d3148; border-radius: 4px; padding: 4px 8px; font-size: 12px; }
  .filter-bar button { background: transparent; color: #a0aec0; border: 1px solid #2d3148; border-radius: 4px; padding: 4px 10px; font-size: 12px; cursor: pointer; }
  .filter-bar button:hover { background: #2d3148; color: #e2e8f0; }
  .filter-bar button.active { background: #4a5fc1; color: #fff; border-color: #4a5fc1; }

  .legend { font-size: 11px; color: #718096; padding: 8px 20px; background: #0f1117; border-bottom: 1px solid #2d3148; }
  .legend kbd { background: #2d3148; color: #cbd5e0; padding: 1px 6px; border-radius: 3px; font-family: monospace; font-size: 10px; margin: 0 2px; }
  .legend .lab { font-weight: 700; }

  .grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 14px; padding: 16px;
  }

  .card {
    background: #14172a; border: 1px solid #2d3148; border-radius: 10px; overflow: hidden;
    transition: all 0.15s; position: relative;
    display: flex; flex-direction: column;
  }
  .card.focused { border-color: #667eea; box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.4); }
  .card.labeled { border-color: rgba(255,255,255,0.06); opacity: 0.85; }
  .card.labeled.accept { border-left: 4px solid #38a169; }
  .card.labeled.refine { border-left: 4px solid #3182ce; }
  .card.labeled.defer  { border-left: 4px solid #d69e2e; }
  .card.labeled.skip   { border-left: 4px solid #c53030; }

  .card-img { width: 100%; aspect-ratio: 1; object-fit: cover; background: #0a0c14; display: block; cursor: pointer; }
  .card-img-missing { width: 100%; aspect-ratio: 1; background: #0a0c14; display: flex; align-items: center; justify-content: center; color: #4a5568; font-size: 11px; }

  .card-body { padding: 8px 10px; flex: 1; display: flex; flex-direction: column; gap: 6px; }
  .card-title { font-size: 12px; font-weight: 700; color: #e2e8f0; line-height: 1.2; }
  .card-meta { font-size: 10px; color: #a0aec0; display: flex; gap: 8px; flex-wrap: wrap; }
  .card-meta .pill {
    background: #0f1117; padding: 1px 6px; border-radius: 3px; font-weight: 600;
  }
  .scenario-badge {
    font-size: 9px; font-weight: 700; padding: 1px 5px; border-radius: 3px;
    text-transform: uppercase; letter-spacing: 0.04em;
  }
  .sb-mar { background: #1a2e3a; color: #63b3ed; }
  .sb-dis { background: #2e1f1a; color: #f6ad55; }
  .sb-urb { background: #2e1a3a; color: #b794f4; }
  .sb-ped { background: #1a2e1f; color: #68d391; }

  .cloud-high { color: #fc8181; }
  .cloud-mid  { color: #f6ad55; }
  .cloud-low  { color: #68d391; }

  .auto-row { font-size: 9px; color: #718096; }
  .auto-row .auto-tag { color: #a0aec0; }

  .btn-row { display: flex; gap: 4px; }
  .btn {
    flex: 1; padding: 6px 4px; border: none; border-radius: 4px;
    font-size: 11px; font-weight: 700; cursor: pointer; transition: all 0.1s;
    text-transform: uppercase; letter-spacing: 0.04em;
  }
  .btn:hover { filter: brightness(1.2); }
  .btn.selected { box-shadow: 0 0 0 2px #fff; }
  .btn-accept { background: #276749; color: #c6f6d5; }
  .btn-refine { background: #2b4c7e; color: #bee3f8; }
  .btn-defer  { background: #744210; color: #fefcbf; }
  .btn-skip   { background: #742a2a; color: #fed7d7; }

  /* Modal for full-size image */
  .modal {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.92); display: none; align-items: center; justify-content: center;
    z-index: 1000; cursor: zoom-out;
  }
  .modal.open { display: flex; }
  .modal img { max-width: 90vw; max-height: 90vh; box-shadow: 0 0 40px rgba(102, 126, 234, 0.4); }

  /* Footer */
  footer {
    position: sticky; bottom: 0; background: #14172a; border-top: 1px solid #2d3148;
    padding: 12px 20px; display: flex; gap: 12px; align-items: center; justify-content: space-between;
    z-index: 99;
  }
  .footer-info { font-size: 12px; color: #a0aec0; }
  .footer-info b { color: #e2e8f0; }
  .footer-actions { display: flex; gap: 8px; }
  .save-btn {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: #fff; border: none; border-radius: 6px; padding: 8px 18px;
    font-weight: 700; font-size: 13px; cursor: pointer;
    transition: all 0.15s;
  }
  .save-btn:hover { filter: brightness(1.15); }
  .save-btn:disabled { opacity: 0.5; cursor: not-allowed; filter: none; }
  .reviewer-input { background: #0f1117; color: #e2e8f0; border: 1px solid #2d3148; border-radius: 6px; padding: 6px 10px; font-size: 12px; width: 110px; }
</style>
</head>
<body>

<header>
  <h1>SimSat Gallery Review</h1>
  <div id="progress-bar-wrap"><div id="progress-bar"></div></div>
  <span class="progress-counts">
    <span id="ct-total">0</span> / __TOTAL_COUNT__ &nbsp;·&nbsp;
    <span class="ct-accept">a:<span id="ct-accept">0</span></span> &nbsp;
    <span class="ct-refine">r:<span id="ct-refine">0</span></span> &nbsp;
    <span class="ct-defer">d:<span id="ct-defer">0</span></span> &nbsp;
    <span class="ct-skip">s:<span id="ct-skip">0</span></span>
  </span>
</header>

<div class="filter-bar">
  <label>Filter:</label>
  <button data-filter="all" class="active">all</button>
  <button data-filter="unrated">unrated</button>
  <button data-filter="rated">rated</button>
  <button data-filter="pedospheric_integrity">pedospheric</button>
  <button data-filter="maritime_chokepoints">maritime</button>
  <button data-filter="disaster_response_weather">disaster</button>
  <button data-filter="urban_coastal_ambiguity">urban-coastal</button>
  <span style="flex:1"></span>
  <label>Sort:</label>
  <select id="sort-by">
    <option value="default">default</option>
    <option value="cloud-asc">cloud ↑</option>
    <option value="cloud-desc">cloud ↓</option>
    <option value="elev-desc">elevation ↓</option>
  </select>
</div>

<div class="legend">
  <span class="lab" style="color:#68d391">accept</span> clean, full-frame, commit &nbsp;·&nbsp;
  <span class="lab" style="color:#63b3ed">refine</span> partial occlusion / marginal cloud, secondary pass &nbsp;·&nbsp;
  <span class="lab" style="color:#f6ad55">defer</span> too compromised, wait for next overpass &nbsp;·&nbsp;
  <span class="lab" style="color:#fc8181">skip</span> not worth bandwidth (heavy cloud / off-cycle) &nbsp;|&nbsp;
  Hover a card and press <kbd>A</kbd> <kbd>R</kbd> <kbd>D</kbd> <kbd>S</kbd> &nbsp;·&nbsp; click image to zoom &nbsp;·&nbsp; click outside any card to deselect
</div>

<div class="grid" id="grid"></div>

<div class="modal" id="modal"><img id="modal-img" src=""></div>

<footer>
  <div class="footer-info">
    <b id="rated-count">0</b> labeled · <b id="remaining-count">__TOTAL_COUNT__</b> remaining
  </div>
  <div class="footer-actions">
    <input type="text" id="reviewer-input" class="reviewer-input" value="ben" placeholder="reviewer">
    <button class="save-btn" id="save-btn">💾 Save Labels (labels.json)</button>
  </div>
</footer>

<script>
const CARDS = __CARDS_JSON__;
const TOTAL = CARDS.length;
const labels = {}; // trace_id -> action

const SCENARIO_BADGE = {
  "maritime_chokepoints": "sb-mar",
  "disaster_response_weather": "sb-dis",
  "urban_coastal_ambiguity": "sb-urb",
  "pedospheric_integrity": "sb-ped",
};
const SCENARIO_SHORT = {
  "maritime_chokepoints": "MAR",
  "disaster_response_weather": "DIS",
  "urban_coastal_ambiguity": "URB",
  "pedospheric_integrity": "PED",
};

let focused = null;
let activeFilter = "all";
let activeSort = "default";

function cloudClass(c) {
  if (c >= 75) return "cloud-high";
  if (c >= 50) return "cloud-mid";
  return "cloud-low";
}

function build() {
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  let visible = CARDS;
  if (activeFilter === "unrated") visible = CARDS.filter(c => !labels[c.id]);
  else if (activeFilter === "rated") visible = CARDS.filter(c => labels[c.id]);
  else if (activeFilter !== "all") visible = CARDS.filter(c => c.scenario === activeFilter);

  if (activeSort === "cloud-asc") visible = [...visible].sort((a, b) => a.cloud - b.cloud);
  else if (activeSort === "cloud-desc") visible = [...visible].sort((a, b) => b.cloud - a.cloud);
  else if (activeSort === "elev-desc") visible = [...visible].sort((a, b) => b.elev - a.elev);

  for (const c of visible) {
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.id = c.id;
    if (labels[c.id]) {
      card.classList.add("labeled", labels[c.id]);
    }

    const imgHtml = c.img
      ? `<img class="card-img" src="${c.img}" alt="${c.target}" data-id="${c.id}">`
      : `<div class="card-img-missing">no image</div>`;

    const sb = SCENARIO_BADGE[c.scenario] || "";
    const sbShort = SCENARIO_SHORT[c.scenario] || c.scenario;

    card.innerHTML = `
      ${imgHtml}
      <div class="card-body">
        <div class="card-title">${c.target}</div>
        <div class="card-meta">
          <span class="scenario-badge ${sb}">${sbShort}</span>
          <span class="pill ${cloudClass(c.cloud)}">${c.cloud}%</span>
          <span class="pill">${c.elev}°</span>
          <span class="pill" style="color:#718096">${c.date}</span>
        </div>
        <div class="auto-row"><span class="auto-tag">auto:</span> ${c.auto}</div>
        <div class="btn-row">
          <button class="btn btn-accept" data-action="accept" data-id="${c.id}">A</button>
          <button class="btn btn-refine" data-action="refine" data-id="${c.id}">R</button>
          <button class="btn btn-defer"  data-action="defer"  data-id="${c.id}">D</button>
          <button class="btn btn-skip"   data-action="skip"   data-id="${c.id}">S</button>
        </div>
      </div>`;

    // Highlight selected button
    if (labels[c.id]) {
      const b = card.querySelector(`button[data-action="${labels[c.id]}"]`);
      if (b) b.classList.add("selected");
    }

    grid.appendChild(card);
  }

  // Listeners (delegated could be used; doing direct here for simplicity)
  grid.querySelectorAll(".btn").forEach(b => {
    b.addEventListener("click", (e) => {
      const id = b.dataset.id;
      const action = b.dataset.action;
      setLabel(id, action);
      e.stopPropagation();
    });
  });
  grid.querySelectorAll(".card-img").forEach(img => {
    img.addEventListener("click", (e) => {
      const modal = document.getElementById("modal");
      document.getElementById("modal-img").src = img.src;
      modal.classList.add("open");
      e.stopPropagation();
    });
  });
  grid.querySelectorAll(".card").forEach(card => {
    card.addEventListener("mouseenter", () => focusCard(card.dataset.id));
    card.addEventListener("mouseleave", () => { /* keep focus until next */ });
  });
  updateProgress();
}

function focusCard(id) {
  if (focused === id) return;
  document.querySelectorAll(".card.focused").forEach(c => c.classList.remove("focused"));
  const card = document.querySelector(`.card[data-id="${id}"]`);
  if (card) card.classList.add("focused");
  focused = id;
}

function setLabel(id, action) {
  labels[id] = action;
  // Update card visuals without rebuild
  const card = document.querySelector(`.card[data-id="${id}"]`);
  if (card) {
    card.classList.remove("accept", "refine", "defer", "skip");
    card.classList.add("labeled", action);
    card.querySelectorAll(".btn").forEach(b => b.classList.remove("selected"));
    const b = card.querySelector(`button[data-action="${action}"]`);
    if (b) b.classList.add("selected");
  }
  updateProgress();
}

function updateProgress() {
  const total = TOTAL;
  const labeled = Object.keys(labels).length;
  document.getElementById("progress-bar").style.width = (labeled / total * 100) + "%";
  document.getElementById("ct-total").textContent = labeled;
  document.getElementById("rated-count").textContent = labeled;
  document.getElementById("remaining-count").textContent = total - labeled;
  let ct = { accept: 0, refine: 0, defer: 0, skip: 0 };
  for (const a of Object.values(labels)) ct[a] = (ct[a] || 0) + 1;
  document.getElementById("ct-accept").textContent = ct.accept;
  document.getElementById("ct-refine").textContent = ct.refine;
  document.getElementById("ct-defer").textContent = ct.defer;
  document.getElementById("ct-skip").textContent = ct.skip;
}

// Filter buttons
document.querySelectorAll(".filter-bar [data-filter]").forEach(b => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".filter-bar [data-filter]").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    activeFilter = b.dataset.filter;
    build();
  });
});

document.getElementById("sort-by").addEventListener("change", (e) => {
  activeSort = e.target.value;
  build();
});

// Modal close
document.getElementById("modal").addEventListener("click", () => {
  document.getElementById("modal").classList.remove("open");
});

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") return;
  const k = e.key.toLowerCase();
  const map = { a: "accept", r: "refine", d: "defer", s: "skip" };
  if (k === "escape") {
    document.getElementById("modal").classList.remove("open");
    return;
  }
  if (focused && map[k]) {
    setLabel(focused, map[k]);
    e.preventDefault();
  }
});

// Save labels
document.getElementById("save-btn").addEventListener("click", () => {
  const reviewer = document.getElementById("reviewer-input").value.trim() || "ben";
  const out = {
    reviewer: reviewer,
    generated_at: new Date().toISOString(),
    total_in_pool: TOTAL,
    labels: Object.entries(labels).map(([trace_id, action]) => ({ trace_id, action })),
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "labels.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

build();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
