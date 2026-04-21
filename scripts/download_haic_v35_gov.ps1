# Download HAIC v35-gov weights from Kaggle for Entry B (Windows / PowerShell)
#
# Populates .\weights\haic-v35-gov\ with base / adapter / gguf / eval subfolders.
#
# Requires:
#   - kaggle CLI installed  (pip install kaggle)
#   - %USERPROFILE%\.kaggle\kaggle.json API token
#
# Usage:
#   cd D:\SimSat
#   pwsh .\scripts\download_haic_v35_gov.ps1

$ErrorActionPreference = "Stop"

$Dest = if ($env:HAIC_GEMMA4_WEIGHTS_DIR) { $env:HAIC_GEMMA4_WEIGHTS_DIR } else { ".\weights\haic-v35-gov" }
$KernelUnsloth = "benhaslam/haic-gemma4-v35-gov-unsloth"
$KernelDataset = "benhaslam/haic-gemma4-v35-gov-dataset-generator"

$kaggle = Get-Command kaggle -ErrorAction SilentlyContinue
if (-not $kaggle) {
    Write-Error "kaggle CLI not found. Install with: pip install kaggle"
    exit 1
}

$tokenPath = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if (-not (Test-Path $tokenPath)) {
    Write-Warning "Kaggle API token not found at $tokenPath - download will fail for private kernels."
    Write-Warning "Get one at https://www.kaggle.com/settings (Account -> Create New Token)."
}

$RawDir = Join-Path $Dest "_raw_kernel_output"
New-Item -ItemType Directory -Force -Path $RawDir | Out-Null

Write-Host "=== Downloading main training kernel output ==="
Write-Host "  $KernelUnsloth -> $RawDir"
kaggle kernels output $KernelUnsloth -p $RawDir

Write-Host ""
Write-Host "=== Organizing weights into predictable layout ==="

$Base = Join-Path $Dest "base"
$Adapter = Join-Path $Dest "adapter"
$Gguf = Join-Path $Dest "gguf"
$Eval = Join-Path $Dest "eval"

foreach ($d in @($Base, $Adapter, $Gguf, $Eval)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

function Copy-IfExists {
    param([string]$Pattern, [string]$Target)
    Get-ChildItem -Path $RawDir -Filter $Pattern -File -ErrorAction SilentlyContinue |
        ForEach-Object { Copy-Item -Force $_.FullName -Destination $Target }
}

# Merged full model
Copy-IfExists "model-*-of-*.safetensors" $Base
Copy-IfExists "model.safetensors.index.json" $Base
Copy-IfExists "config.json" $Base

# Tokenizer & chat template -> base AND adapter
foreach ($f in @("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")) {
    $src = Join-Path $RawDir $f
    if (Test-Path $src) {
        Copy-Item -Force $src -Destination $Base
        Copy-Item -Force $src -Destination $Adapter
    }
}

# LoRA adapter
Copy-IfExists "adapter_config.json" $Adapter
Copy-IfExists "adapter_model.safetensors" $Adapter

# GGUF
$ggufSub = Join-Path $RawDir "haic-gemma4-v35-gov-gguf"
if (Test-Path $ggufSub) {
    Get-ChildItem -Path $ggufSub -Filter "*.gguf" -File | ForEach-Object { Copy-Item -Force $_.FullName -Destination $Gguf }
    $ggufTpl = Join-Path $ggufSub "chat_template.jinja"
    if (Test-Path $ggufTpl) { Copy-Item -Force $ggufTpl -Destination $Gguf }
}
Copy-IfExists "*.gguf" $Gguf

# Eval artifacts
foreach ($f in @("haic_v35_gov_full_results.json", "prism_gemma4_v35_gov.json", "trainer_state.json", "training_args.bin")) {
    $src = Join-Path $RawDir $f
    if (Test-Path $src) { Copy-Item -Force $src -Destination $Eval }
}

Write-Host ""
Write-Host "=== Summary ==="
foreach ($pair in @(@{name="Base"; path=$Base}, @{name="Adapter"; path=$Adapter}, @{name="GGUF"; path=$Gguf}, @{name="Eval"; path=$Eval})) {
    Write-Host ""
    Write-Host ("{0}:" -f $pair.name)
    $items = Get-ChildItem -Path $pair.path -File -ErrorAction SilentlyContinue
    if ($items) { $items | ForEach-Object { "  {0,10}  {1}" -f $_.Length, $_.Name } | Write-Host } else { Write-Host "  (empty)" }
}

Write-Host ""
Write-Host "=== Done ==="
$absDest = (Resolve-Path $Dest).Path
Write-Host "Point the sim service at these weights by setting:"
Write-Host "  OBSERVATION_VLA_BACKEND=gemma4_haic_local"
Write-Host ("  HAIC_GEMMA4_WEIGHTS_DIR={0}" -f $absDest)
Write-Host ""
Write-Host "Optional: download training-data snapshots too (for the viability-grounding demo)"
Write-Host ("  kaggle kernels output {0} -p {1}\dataset_snapshots" -f $KernelDataset, $Dest)
