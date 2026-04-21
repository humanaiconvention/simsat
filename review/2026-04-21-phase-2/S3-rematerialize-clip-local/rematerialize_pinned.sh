#!/usr/bin/env bash
# Phase 2 S3 - Re-assess the 3 pinned submission cases under clip_local
#
# Calls POST /observation-vla/reassess/{trace_id}?persist=true for each
# pinned trace so the stored runtime_mode flips from 'stub' to 'clip_local'.
#
# Requires: docker compose up running at http://127.0.0.1:8000/sim
#           jq for response parsing
#
# Usage:
#   cd D:\SimSat
#   bash review/2026-04-21-phase-2/S3-rematerialize-clip-local/rematerialize_pinned.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/sim}"

# 3 pinned submission cases as of 2026-04-21 per SUBMISSION_PACKET.md
TRACES=(
    "trace_1926b646ee4b48478913681a33fcdfb1:disaster_response_weather:Houston Ship Channel"
    "trace_2507337b7939460ebf01cbc9fcef8055:maritime_chokepoints:Suez Canal"
    "trace_a4e31b4c39224d8fbdb2c4bf0f444823:urban_coastal_ambiguity:San Francisco Bay"
)

echo "Re-assessing 3 pinned traces under the current ObservationVLA backend..."
echo ""

success=0
for entry in "${TRACES[@]}"; do
    IFS=':' read -r trace_id scenario target <<< "$entry"
    echo "---"
    echo "[$scenario] $target"
    echo "  trace_id: $trace_id"

    resp=$(curl -sS -w "\n%{http_code}" -X POST \
        "$BASE_URL/observation-vla/reassess/$trace_id?persist=true" \
        -H "Content-Type: application/json")
    body="${resp%$'\n'*}"
    code="${resp##*$'\n'}"

    if [[ "$code" != "200" ]]; then
        echo "  [ERROR] HTTP $code: $body"
        continue
    fi

    runtime=$(echo "$body" | jq -r '.assessment.runtime_mode // "?"')
    action=$(echo "$body" | jq -r '.assessment.recommended_action // "?"')
    mode=$(echo "$body" | jq -r '.assessment.assessment_mode // "?"')
    echo "  [OK] runtime_mode=$runtime, assessment_mode=$mode, recommended_action=$action"
    success=$((success + 1))
done

echo ""
echo "---"
echo "Reassessed $success / ${#TRACES[@]} pinned traces."
echo ""
echo "Next: regenerate SUBMISSION_CASEBOOK.md with"
echo "  python scripts/submission_casebook.py --base-url $BASE_URL"
echo "Then grep -c 'Observation runtime: \`clip_local\`' SUBMISSION_CASEBOOK.md (should be 3)"
