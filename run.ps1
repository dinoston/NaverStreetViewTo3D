param(
    [ValidateSet("doctor", "preprocess", "align", "train", "mesh", "fast", "all")]
    [string]$Stage = "all",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$fullEnvRoot = Join-Path $PSScriptRoot "tools\gs-env"
$basicEnvRoot = Join-Path $PSScriptRoot ".venv"
$envRoot = if (Test-Path (Join-Path $fullEnvRoot "Scripts\street3d.exe")) {
    $fullEnvRoot
} else {
    $basicEnvRoot
}
$env:CUDA_HOME = $envRoot
$env:HF_HOME = Join-Path $PSScriptRoot "tools\model-cache"
$env:PATH = @(
    $envRoot
    (Join-Path $envRoot "bin")
    (Join-Path $envRoot "Library")
    (Join-Path $envRoot "Library\bin")
    (Join-Path $envRoot "Scripts")
    $env:PATH
) -join ";"
$exe = Join-Path $envRoot "Scripts\street3d.exe"
if (-not (Test-Path $exe)) {
    throw "Project environment not found. Run .\setup.ps1 first."
}
$arguments = @("--project", $PSScriptRoot, $Stage)
if ($Force -and $Stage -in @("preprocess", "align", "train", "fast", "all")) {
    $arguments += "--force"
}
& $exe @arguments
