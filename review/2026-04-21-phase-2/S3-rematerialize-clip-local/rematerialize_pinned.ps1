# Phase 2 S3 - Re-assess the 3 pinned submission cases under clip_local (PowerShell)
#
# Requires: docker compose up running at http://127.0.0.1:8000/sim
#
# Usage:
#   cd D:\SimSat
#   pwsh .\review\2026-04-21-phase-2\S3-rematerialize-clip-local\rematerialize_pinned.ps1

$ErrorActionPreference = "Stop"
$BaseUrl = if ($env:BASE_URL) { $env:BASE_URL } else { "http://127.0.0.1:8000/sim" }

$Traces = @(
    @{ TraceId = "trace_1926b646ee4b48478913681a33fcdfb1"; Scenario = "disaster_response_weather"; Target = "Houston Ship Channel" },
    @{ TraceId = "trace_2507337b7939460ebf01cbc9fcef8055"; Scenario = "maritime_chokepoints";      Target = "Suez Canal" },
    @{ TraceId = "trace_a4e31b4c39224d8fbdb2c4bf0f444823"; Scenario = "urban_coastal_ambiguity";   Target = "San Francisco Bay" }
)

Write-Host "Re-assessing 3 pinned traces under the current ObservationVLA backend..."
Write-Host ""

$success = 0
foreach ($t in $Traces) {
    Write-Host "---"
    Write-Host ("[{0}] {1}" -f $t.Scenario, $t.Target)
    Write-Host ("  trace_id: {0}" -f $t.TraceId)

    $url = "$BaseUrl/observation-vla/reassess/$($t.TraceId)?persist=true"
    try {
        $resp = Invoke-RestMethod -Method Post -Uri $url -ContentType "application/json" -TimeoutSec 180
    } catch {
        Write-Host ("  [ERROR] {0}" -f $_.Exception.Message)
        continue
    }

    $runtime = $resp.assessment.runtime_mode
    $action  = $resp.assessment.recommended_action
    $mode    = $resp.assessment.assessment_mode
    Write-Host ("  [OK] runtime_mode={0}, assessment_mode={1}, recommended_action={2}" -f $runtime, $mode, $action)
    $success++
}

Write-Host ""
Write-Host "---"
Write-Host ("Reassessed {0} / {1} pinned traces." -f $success, $Traces.Count)
Write-Host ""
Write-Host "Next: regenerate SUBMISSION_CASEBOOK.md with"
Write-Host ("  python scripts\submission_casebook.py --base-url {0}" -f $BaseUrl)
Write-Host "Then grep -c 'Observation runtime: `clip_local`' SUBMISSION_CASEBOOK.md (should be 3)"
