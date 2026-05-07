#!/usr/bin/env python3
"""Generate defer_review.html with embedded images and keyboard-driven UI."""
import json, base64, pathlib, textwrap

assets_dir = pathlib.Path(__file__).parent.parent / "review_queue_assets"
out_path   = pathlib.Path(__file__).parent.parent / "defer_review.html"

traces = [
    {"id": "trace_c3e2a26eeab540a3b34a2b5185ce6613", "scenario": "urban_coastal_ambiguity",   "target": "Port of Los Angeles",  "cloud": 61.9, "elev": 60.0,  "off_nadir": 26.4, "date": "2026-04-02", "auto": "refine"},
    {"id": "trace_f760ebf72a964cab8ac6372e6e2dd820", "scenario": "urban_coastal_ambiguity",   "target": "Shenzhen Bay",         "cloud": 70.8, "elev": 73.7,  "off_nadir": 14.4, "date": "2026-04-03", "auto": "refine"},
    {"id": "trace_e603eaf267c448b1a4ed059e6ac3e638", "scenario": "urban_coastal_ambiguity",   "target": "Port of Rotterdam",    "cloud": 76.8, "elev": 75.4,  "off_nadir": 13.0, "date": "2026-04-01", "auto": "defer"},
    {"id": "trace_63d2f16600264d93b7443459166cfde7", "scenario": "urban_coastal_ambiguity",   "target": "Port of Rotterdam",    "cloud": 66.8, "elev": 65.1,  "off_nadir": 22.0, "date": "2026-03-25", "auto": "refine"},
    {"id": "trace_ad3a863eae974301b903539c40a708b2", "scenario": "urban_coastal_ambiguity",   "target": "Shenzhen Bay",         "cloud": 61.8, "elev": 74.4,  "off_nadir": 13.8, "date": "2026-03-24", "auto": "defer"},
    {"id": "trace_85ddfa677bfd459d95dbed455dd38505", "scenario": "disaster_response_weather", "target": "Fort Myers Coast",     "cloud": 65.9, "elev": 78.5,  "off_nadir": 10.2, "date": "2026-03-18", "auto": "refine"},
    {"id": "trace_978e6a6a5bf8466e99aa7b38ef7e35a8", "scenario": "disaster_response_weather", "target": "Fort Myers Coast",     "cloud": 89.5, "elev": 71.7,  "off_nadir": 16.2, "date": "2026-04-07", "auto": "skip"},
    {"id": "trace_1d9e98b6bc7d493f8c292947938b6d4e", "scenario": "maritime_chokepoints",      "target": "Port of Singapore",    "cloud": 93.1, "elev": 81.0,  "off_nadir":  8.0, "date": "2026-04-09", "auto": "skip"},
    {"id": "trace_31ec6996796d40c2b09e946f854f0d4a", "scenario": "maritime_chokepoints",      "target": "Panama Canal",         "cloud": 64.1, "elev": 59.1,  "off_nadir": 27.2, "date": "2026-04-08", "auto": "refine"},
    {"id": "trace_10d844ffa6a14e39b7e933375f05c9cd", "scenario": "urban_coastal_ambiguity",   "target": "Port of Los Angeles",  "cloud": 74.5, "elev": 51.0,  "off_nadir": 34.0, "date": "2026-04-09", "auto": "skip"},
    {"id": "trace_2c2e81628756417dbb0093a1babf0ffd", "scenario": "urban_coastal_ambiguity",   "target": "Shenzhen Bay",         "cloud":100.0, "elev": 56.5,  "off_nadir": 29.4, "date": "2026-04-08", "auto": "skip"},
    {"id": "trace_22155d6f9ee14be18d072bcda4aa4a2c", "scenario": "disaster_response_weather", "target": "Fort Myers Coast",     "cloud": 89.5, "elev": 64.0,  "off_nadir": 22.9, "date": "2026-04-07", "auto": "refine"},
]

# Embed images
for t in traces:
    matches = sorted(assets_dir.glob(f"*{t['id']}*.png"))
    if matches:
        t["img"] = "data:image/png;base64," + base64.b64encode(matches[0].read_bytes()).decode()
    else:
        t["img"] = ""

traces_json = json.dumps(traces)

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SimSat Defer Review — 12 Cases</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f1117; color: #e2e8f0; font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }

  header {
    background: #1a1d2e; border-bottom: 1px solid #2d3148;
    padding: 14px 24px; display: flex; align-items: center; gap: 16px;
    position: sticky; top: 0; z-index: 100;
  }
  header h1 { font-size: 16px; font-weight: 600; color: #a0aec0; letter-spacing: 0.05em; text-transform: uppercase; }
  #progress-bar-wrap { flex: 1; background: #2d3148; border-radius: 4px; height: 6px; overflow: hidden; }
  #progress-bar { height: 100%; background: #667eea; width: 0%; transition: width 0.3s; border-radius: 4px; }
  #progress-label { font-size: 13px; color: #718096; white-space: nowrap; }

  .carousel { max-width: 900px; margin: 32px auto; padding: 0 20px; }

  .card { background: #1a1d2e; border: 1px solid #2d3148; border-radius: 12px; overflow: hidden; display: none; }
  .card.active { display: block; }

  .card-img { width: 100%; max-height: 420px; object-fit: cover; display: block; background: #0f1117; }
  .card-img-placeholder { width: 100%; height: 260px; background: #0f1117; display: flex; align-items: center; justify-content: center; color: #4a5568; font-size: 14px; }

  .card-body { padding: 20px 24px; }
  .card-header { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 14px; gap: 12px; }
  .card-title { font-size: 20px; font-weight: 700; color: #f7fafc; }
  .scenario-badge { font-size: 11px; font-weight: 600; padding: 4px 10px; border-radius: 20px; letter-spacing: 0.04em; white-space: nowrap; flex-shrink: 0; margin-top: 4px; }

  .meta-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px; }
  .meta-item { background: #0f1117; border-radius: 8px; padding: 10px 12px; }
  .meta-label { font-size: 10px; color: #718096; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
  .meta-value { font-size: 16px; font-weight: 600; }
  .cloud-high { color: #fc8181; }
  .cloud-mid  { color: #f6ad55; }
  .cloud-low  { color: #68d391; }

  .auto-row { display: flex; align-items: center; gap: 8px; margin-bottom: 20px; font-size: 13px; color: #718096; }
  .auto-badge { font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
  .ab-accept { background: #1a3a2a; color: #68d391; }
  .ab-refine { background: #1a2d3a; color: #63b3ed; }
  .ab-defer  { background: #2d2a1a; color: #f6ad55; }
  .ab-skip   { background: #2d1a1a; color: #fc8181; }

  .btn-row { display: flex; gap: 10px; }
  .btn {
    flex: 1; padding: 13px 8px; border: none; border-radius: 8px;
    font-size: 14px; font-weight: 700; cursor: pointer; transition: all 0.15s;
    text-transform: uppercase; letter-spacing: 0.06em; position: relative;
  }
  .btn:hover { transform: translateY(-1px); filter: brightness(1.15); }
  .btn.selected { outline: 3px solid #fff; outline-offset: 2px; }
  .btn kbd {
    position: absolute; top: 5px; right: 7px;
    font-size: 9px; background: rgba(0,0,0,0.3); border-radius: 3px;
    padding: 1px 4px; font-family: monospace; letter-spacing: 0; font-weight: 400;
  }
  .btn-accept { background: #276749; color: #c6f6d5; }
  .btn-refine { background: #2b4c7e; color: #bee3f8; }
  .btn-defer  { background: #744210; color: #fefcbf; }
  .btn-skip   { background: #742a2a; color: #fed7d7; }

  .nav-row { display: flex; gap: 10px; margin-top: 14px; }
  .nav-btn { flex: 1; padding: 9px; border: 1px solid #2d3148; border-radius: 8px; background: transparent; color: #a0aec0; font-size: 13px; cursor: pointer; transition: all 0.15s; }
  .nav-btn:hover { background: #2d3148; color: #e2e8f0; }

  .dot-row { display: flex; justify-content: center; gap: 7px; margin: 20px 0 0; flex-wrap: wrap; }
  .dot { width: 12px; height: 12px; border-radius: 50%; background: #2d3148; cursor: pointer; transition: all 0.2s; border: 2px solid transparent; }
  .dot.current  { border-color: #667eea; }
  .dot.done-accept { background: #276749; }
  .dot.done-refine { background: #2b4c7e; }
  .dot.done-defer  { background: #744210; }
  .dot.done-skip   { background: #742a2a; }

  #results-panel { display: none; max-width: 900px; margin: 32px auto; padding: 0 20px 60px; }
  #results-panel.visible { display: block; }
  .results-card { background: #1a1d2e; border: 1px solid #2d3148; border-radius: 12px; padding: 28px; }
  .results-card h2 { font-size: 18px; font-weight: 700; margin-bottom: 6px; color: #f7fafc; }
  .results-card p { font-size: 13px; color: #718096; margin-bottom: 20px; }
  .results-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 24px; }
  .results-table th { text-align: left; padding: 8px 10px; color: #718096; font-weight: 600; border-bottom: 1px solid #2d3148; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; }
  .results-table td { padding: 8px 10px; border-bottom: 1px solid #1a1d2e; }
  .results-table tr:last-child td { border-bottom: none; }
  .changed { color: #f6ad55; font-size: 11px; margin-left: 4px; }

  .code-block { background: #0f1117; border: 1px solid #2d3148; border-radius: 8px; padding: 16px; font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 12px; color: #a0aec0; white-space: pre-wrap; word-break: break-all; max-height: 320px; overflow-y: auto; margin-bottom: 14px; }
  .copy-btn { background: #667eea; color: #fff; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.15s; }
  .copy-btn:hover { background: #5a6fd6; }
  .copy-btn.copied { background: #276749; }
  .kbd-hint { font-size: 12px; color: #4a5568; margin-top: 10px; text-align: center; padding-bottom: 8px; }
</style>
</head>
<body>

<header>
  <h1>SimSat Defer Review</h1>
  <div id="progress-bar-wrap"><div id="progress-bar"></div></div>
  <span id="progress-label">0 / 12</span>
</header>

<div class="carousel" id="carousel"></div>
<div class="dot-row" id="dot-row"></div>
<p class="kbd-hint" id="kbd-hint">&#8592; &#8594; navigate &nbsp;·&nbsp; <b>A</b> accept &nbsp; <b>R</b> refine &nbsp; <b>D</b> defer &nbsp; <b>S</b> skip</p>

<div id="results-panel">
  <div class="results-card">
    <h2>Review Complete &#10003;</h2>
    <p>All 12 cases reviewed. Copy the command below and run it from your SimSat repo root.</p>
    <table class="results-table" id="results-table">
      <thead><tr><th>Target</th><th>Cloud</th><th>Auto</th><th>Your Decision</th></tr></thead>
      <tbody id="results-tbody"></tbody>
    </table>
    <div class="code-block" id="code-block"></div>
    <button class="copy-btn" id="copy-btn" onclick="copyCode()">Copy Command</button>
  </div>
</div>

<script>
var TRACES = TRACES_JSON_PLACEHOLDER;

var scenarioColors = {
  "urban_coastal_ambiguity":   "#4a90d9",
  "disaster_response_weather": "#e07b39",
  "maritime_chokepoints":      "#5cb85c"
};
var actionClass = {accept:"ab-accept", refine:"ab-refine", defer:"ab-defer", skip:"ab-skip"};
var usefulness  = {accept:0.85, refine:0.55, defer:0.40, skip:0.20};
var useful_flag = {accept:true, refine:true, defer:false, skip:false};

var decisions = new Array(TRACES.length).fill(null);
var current = 0;

function cloudClass(c) {
  if (c >= 80) return "cloud-high";
  if (c >= 60) return "cloud-mid";
  return "cloud-low";
}

function buildCards() {
  var carousel = document.getElementById("carousel");
  var dotRow   = document.getElementById("dot-row");
  TRACES.forEach(function(t, i) {
    var card = document.createElement("div");
    card.className = "card" + (i === 0 ? " active" : "");
    card.id = "card-" + i;
    var imgHtml = t.img
      ? '<img class="card-img" src="' + t.img + '" alt="' + t.target + '">'
      : '<div class="card-img-placeholder">No image available</div>';
    var sc = t.scenario.replace(/_/g, " ");
    var color = scenarioColors[t.scenario] || "#888";
    var aClass = actionClass[t.auto] || "";
    card.innerHTML = imgHtml +
      '<div class="card-body">' +
        '<div class="card-header">' +
          '<div class="card-title">' + t.target + '</div>' +
          '<span class="scenario-badge" style="background:' + color + '22;color:' + color + '">' + sc + '</span>' +
        '</div>' +
        '<div class="meta-grid">' +
          '<div class="meta-item"><div class="meta-label">Cloud Cover</div><div class="meta-value ' + cloudClass(t.cloud) + '">' + t.cloud + '%</div></div>' +
          '<div class="meta-item"><div class="meta-label">Elevation</div><div class="meta-value">' + t.elev + '&deg;</div></div>' +
          '<div class="meta-item"><div class="meta-label">Off-Nadir</div><div class="meta-value">' + t.off_nadir + '&deg;</div></div>' +
          '<div class="meta-item"><div class="meta-label">Sentinel Date</div><div class="meta-value" style="font-size:13px;padding-top:2px">' + t.date + '</div></div>' +
        '</div>' +
        '<div class="auto-row">Auto-assigned: <span class="auto-badge ' + aClass + '">' + t.auto + '</span>&nbsp;&middot;&nbsp;' + (i+1) + ' of ' + TRACES.length + '</div>' +
        '<div class="btn-row">' +
          '<button class="btn btn-accept" id="btn-' + i + '-accept" onclick="decide(' + i + ',\'accept\')">Accept<kbd>A</kbd></button>' +
          '<button class="btn btn-refine" id="btn-' + i + '-refine" onclick="decide(' + i + ',\'refine\')">Refine<kbd>R</kbd></button>' +
          '<button class="btn btn-defer"  id="btn-' + i + '-defer"  onclick="decide(' + i + ',\'defer\')">Defer<kbd>D</kbd></button>' +
          '<button class="btn btn-skip"   id="btn-' + i + '-skip"   onclick="decide(' + i + ',\'skip\')">Skip<kbd>S</kbd></button>' +
        '</div>' +
        '<div class="nav-row">' +
          '<button class="nav-btn" onclick="navigate(-1)">&larr; Prev</button>' +
          '<button class="nav-btn" onclick="navigate(1)">Next &rarr;</button>' +
        '</div>' +
      '</div>';
    carousel.appendChild(card);

    var dot = document.createElement("div");
    dot.className = "dot" + (i === 0 ? " current" : "");
    dot.id = "dot-" + i;
    dot.title = t.target + " (" + t.cloud + "% cloud)";
    dot.onclick = (function(idx){ return function(){ goTo(idx); }; })(i);
    dotRow.appendChild(dot);
  });
}

function goTo(i) {
  document.getElementById("card-" + current).classList.remove("active");
  var oldDot = document.getElementById("dot-" + current);
  oldDot.classList.remove("current");
  current = i;
  document.getElementById("card-" + current).classList.add("active");
  var newDot = document.getElementById("dot-" + current);
  newDot.classList.add("current");
}

function navigate(dir) {
  var next = current + dir;
  if (next < 0) next = TRACES.length - 1;
  if (next >= TRACES.length) next = 0;
  goTo(next);
}

function decide(i, action) {
  decisions[i] = action;
  var dot = document.getElementById("dot-" + i);
  dot.className = "dot done-" + action + (i === current ? " current" : "");
  ["accept","refine","defer","skip"].forEach(function(a) {
    var b = document.getElementById("btn-" + i + "-" + a);
    if (b) { if (a === action) b.classList.add("selected"); else b.classList.remove("selected"); }
  });
  updateProgress();
  if (decisions.every(function(d){ return d !== null; })) {
    setTimeout(showResults, 400);
  } else if (i < TRACES.length - 1 && current === i) {
    setTimeout(function(){ navigate(1); }, 180);
  }
}

function updateProgress() {
  var done = decisions.filter(function(d){ return d !== null; }).length;
  document.getElementById("progress-bar").style.width = (done / TRACES.length * 100) + "%";
  document.getElementById("progress-label").textContent = done + " / " + TRACES.length;
}

function showResults() {
  document.getElementById("carousel").style.display = "none";
  document.getElementById("dot-row").style.display = "none";
  document.getElementById("kbd-hint").style.display = "none";
  document.getElementById("results-panel").classList.add("visible");

  var tbody = document.getElementById("results-tbody");
  TRACES.forEach(function(t, i) {
    var tr = document.createElement("tr");
    var aClass = actionClass[t.auto] || "";
    var dClass = actionClass[decisions[i]] || "";
    var changed = decisions[i] !== t.auto ? '<span class="changed">&#9664; changed</span>' : '';
    tr.innerHTML = '<td>' + t.target + '</td>' +
      '<td>' + t.cloud + '%</td>' +
      '<td><span class="auto-badge ' + aClass + '">' + t.auto + '</span></td>' +
      '<td><span class="auto-badge ' + dClass + '">' + decisions[i] + '</span>' + changed + '</td>';
    tbody.appendChild(tr);
  });

  var lines = [
    "import sys; sys.path.insert(0, 'scripts')",
    "from operator_review import InProcessClient",
    "client = InProcessClient()",
    ""
  ];
  TRACES.forEach(function(t, i) {
    var d = decisions[i];
    lines.push("client.post_json('/observation-vla/trace/" + t.id + "/operator-review', " +
      "{'action_taken': '" + d + "', 'reviewer': 'ben', " +
      "'useful': " + (useful_flag[d] ? "True" : "False") + ", " +
      "'usefulness_score': " + usefulness[d] + "})");
  });
  lines.push("");
  lines.push("print('Done — all 12 reviews posted.')");

  document.getElementById("code-block").textContent = lines.join("\\n");
}

function copyCode() {
  var code = document.getElementById("code-block").textContent;
  navigator.clipboard.writeText(code).then(function() {
    var btn = document.getElementById("copy-btn");
    btn.textContent = "Copied!";
    btn.classList.add("copied");
    setTimeout(function(){ btn.textContent = "Copy Command"; btn.classList.remove("copied"); }, 2000);
  });
}

document.addEventListener("keydown", function(e) {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  var key = e.key.toLowerCase();
  if      (key === "a") decide(current, "accept");
  else if (key === "r") decide(current, "refine");
  else if (key === "d") decide(current, "defer");
  else if (key === "s") decide(current, "skip");
  else if (key === "arrowleft")  { e.preventDefault(); navigate(-1); }
  else if (key === "arrowright") { e.preventDefault(); navigate(1); }
});

buildCards();
updateProgress();
</script>
</body>
</html>
"""

html = html.replace("TRACES_JSON_PLACEHOLDER", traces_json)
out_path.write_text(html, encoding="utf-8")
print(f"Written: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
