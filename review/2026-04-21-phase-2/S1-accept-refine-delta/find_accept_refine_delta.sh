#!/usr/bin/env bash
# Phase 2 S1 - Find accept->refine deltas via encounter_eval sweep
#
# Sweeps multiple start times and scenario packs, looking for any
# accept->refine transition in the WCLI-trust planner output. Writes a
# markdown summary to deltas_found.md if any are found.
#
# Requires: docker compose up running locally (http://127.0.0.1:8000/sim),
#           jq installed (most Git Bash / WSL / Linux setups)
#
# Usage:
#   cd D:\SimSat
#   bash review/2026-04-21-phase-2/S1-accept-refine-delta/find_accept_refine_delta.sh
#
# Optional: edit START_TIMES below to cover different dates / cloud conditions.

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/sim}"
OUT="review/2026-04-21-phase-2/S1-accept-refine-delta/deltas_found.md"
SCENARIOS=("maritime_chokepoints" "disaster_response_weather" "urban_coastal_ambiguity")
START_TIMES=(
    "2026-01-15T00:00:00Z"
    "2026-02-01T06:00:00Z"
    "2026-02-15T12:00:00Z"
    "2026-03-01T18:00:00Z"
    "2026-03-15T00:00:00Z"
    "2026-04-01T06:00:00Z"
    "2026-04-15T12:00:00Z"
    "2026-05-01T18:00:00Z"
    "2026-05-15T00:00:00Z"
    "2026-06-01T06:00:00Z"
)
HOURS=12
STEP=120
TOPK=20
MATERIALIZE_TOPK=0  # skip materialization; we only care about decisions here

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

echo "# SimSat accept→refine delta sweep" > "$OUT"
echo "" >> "$OUT"
echo "- Base URL: \`$BASE_URL\`" >> "$OUT"
echo "- Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT"
echo "- Scanned scenarios: ${SCENARIOS[*]}" >> "$OUT"
echo "- Start times: ${#START_TIMES[@]} passes (see below)" >> "$OUT"
echo "" >> "$OUT"

hits=0

for start in "${START_TIMES[@]}"; do
    for pack in "${SCENARIOS[@]}"; do
        payload=$(cat <<JSON
{"scenario_pack":"$pack","start_time":"$start","hours":$HOURS,"step_seconds":$STEP,"top_k":$TOPK,"materialize_top_k":$MATERIALIZE_TOPK}
JSON
        )
        resp_file="$TMP_DIR/resp_${pack}_${start//[:T-]/_}.json"
        if ! curl -sS -X POST "$BASE_URL/encounter/evaluate" \
            -H "Content-Type: application/json" \
            -d "$payload" \
            -o "$resp_file"; then
            echo "  [warn] curl failed for $pack @ $start" >&2
            continue
        fi

        transitions=$(jq -r '.action_transition_counts // {}' "$resp_file")
        refine_count=$(jq -r '.action_transition_counts["accept->refine"] // 0' "$resp_file")

        if [[ "$refine_count" != "0" ]]; then
            echo "## accept→refine FOUND — $pack @ $start" >> "$OUT"
            echo "" >> "$OUT"
            echo "- Refine transitions: $refine_count" >> "$OUT"
            echo "- Full transition counts: \`$transitions\`" >> "$OUT"
            echo "- Evaluation ID: \`$(jq -r '.evaluation_id' "$resp_file")\`" >> "$OUT"
            echo "" >> "$OUT"
            echo "### Decision deltas that changed to refine" >> "$OUT"
            echo "" >> "$OUT"
            echo "| target | scaffold_action | trust_action | trust_score | refinement_reason |" >> "$OUT"
            echo "| --- | --- | --- | --- | --- |" >> "$OUT"
            jq -r '.decision_deltas[] | select(.trust_action == "refine" and .scaffold_action == "accept") | "| \(.target_label) | \(.scaffold_action) | \(.trust_action) | \(.trust_score) | \(.refinement_reason // "-") |"' "$resp_file" >> "$OUT"
            echo "" >> "$OUT"
            hits=$((hits + refine_count))
        fi
    done
done

if [[ $hits -eq 0 ]]; then
    echo "## No accept→refine transitions found in this sweep" >> "$OUT"
    echo "" >> "$OUT"
    echo "Consider:" >> "$OUT"
    echo "" >> "$OUT"
    echo "1. Extending the \`START_TIMES\` list with more dates or high-cloud periods." >> "$OUT"
    echo "2. Running the sweep against scenario packs with lower-priority targets." >> "$OUT"
    echo "3. Using the Fallback in \`DEMO_DELTA_ADDENDUM.md\` to temporarily lower \`trust_refine_threshold\` to 0.65." >> "$OUT"
    echo "" >> "$OUT"
else
    echo "" >> "$OUT"
    echo "## Summary" >> "$OUT"
    echo "" >> "$OUT"
    echo "- Total accept→refine transitions found: **$hits**" >> "$OUT"
    echo "- Pick the delta with the clearest \`refinement_reason\` (e.g. \`trust_below_refine_threshold\`, \`cloud_risk\`, \`edge_geometry\`) for the demo." >> "$OUT"
    echo "- Use \`operator_review.py --show-bundle-only --trace-id <trace_id>\` for the chosen delta to inspect the window in detail." >> "$OUT"
    echo "" >> "$OUT"
fi

echo "Sweep complete. $hits transitions found. See $OUT"
