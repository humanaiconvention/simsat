# SimSat submission hygiene cleanup - Phase 1, M2
#
# Run from repo root:
#   cd D:\SimSat
#   pwsh .\review\2026-04-21-phase-1\M2-cleanup\cleanup.ps1
#
# Removes leftover tmp*_simsat_api.py files from src\sim\ and scans for a
# suspect duplicate gui.py with a hardcoded LAN IP.

$ErrorActionPreference = "Stop"

$simDir = "src\sim"
if (-not (Test-Path $simDir)) {
    Write-Error "Expected to find $simDir relative to current directory. Run this script from the repo root (D:\SimSat)."
    exit 1
}

Write-Host "=== Removing stray tmp*_simsat_api.py files from $simDir ==="
$removed = 0
Get-ChildItem -Path $simDir -Filter "tmp*_simsat_api.py" -File | ForEach-Object {
    Write-Host "  rm $($_.FullName)"
    Remove-Item $_.FullName
    $removed++
}
if ($removed -eq 0) {
    Write-Host "  (none found - already clean)"
} else {
    Write-Host "  removed $removed file(s)"
}

Write-Host ""
Write-Host "=== Scanning for duplicate gui.py with hardcoded LAN IP ==="
$canonical = (Resolve-Path "$simDir\gui.py" -ErrorAction SilentlyContinue).Path
$suspects = Get-ChildItem -Path . -Recurse -Filter "gui.py" -File |
    Where-Object { $_.FullName -ne $canonical -and (Select-String -Path $_.FullName -Pattern "192\.168\.115\.95" -Quiet) }

if ($suspects) {
    Write-Host "  Found suspect duplicate(s):"
    foreach ($dup in $suspects) {
        Write-Host "    $($dup.FullName)"
    }
    Write-Host ""
    Write-Host "  These files contain a hardcoded LAN IP (192.168.115.95) and likely"
    Write-Host "  missing imports. Review manually, then remove with:"
    Write-Host ""
    foreach ($dup in $suspects) {
        Write-Host "    Remove-Item '$($dup.FullName)'"
    }
} else {
    Write-Host "  No duplicate gui.py found outside $simDir\gui.py"
}

Write-Host ""
Write-Host "=== Done ==="
Write-Host "Review changes with: git status"
Write-Host "Commit with:         git commit -am 'chore: remove stray tmp api files and duplicate gui.py'"
