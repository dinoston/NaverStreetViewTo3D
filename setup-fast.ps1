$ErrorActionPreference = "Stop"

$fullEnvRoot = Join-Path $PSScriptRoot "tools\gs-env"
$basicEnvRoot = Join-Path $PSScriptRoot ".venv"
$python = if (Test-Path (Join-Path $fullEnvRoot "python.exe")) {
    Join-Path $fullEnvRoot "python.exe"
} else {
    Join-Path $basicEnvRoot "Scripts\python.exe"
}
if (-not (Test-Path $python)) {
    throw "Project Python environment not found. Run setup.ps1 first."
}

$vggtRoot = Join-Path $PSScriptRoot "external\vggt"
if (-not (Test-Path (Join-Path $vggtRoot "vggt\models\vggt.py"))) {
    git clone https://github.com/facebookresearch/vggt.git $vggtRoot
    if ($LASTEXITCODE -ne 0) { throw "Could not clone VGGT." }
}

& $python -m pip install einops huggingface_hub hf_xet plyfile
if ($LASTEXITCODE -ne 0) { throw "Could not install fast-mode dependencies." }

$env:HF_HOME = Join-Path $PSScriptRoot "tools\model-cache"
& $python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/VGGT-1B', allow_patterns=['config.json','model.safetensors'])"
if ($LASTEXITCODE -ne 0) { throw "Could not download the VGGT model." }

Write-Host "Fast mode setup complete. Run: .\run.ps1 -Stage fast -Force"
