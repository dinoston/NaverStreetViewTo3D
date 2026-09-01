param(
    [ValidateSet("preprocess", "align", "panorama-fast")]
    [string]$Stage = "panorama-fast",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$envRoot = Join-Path $PSScriptRoot "tools\gs-env"
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
    throw "Project environment not found. Run .\setup-fast.ps1 first."
}
$arguments = @(
    "--project", $PSScriptRoot,
    "--config", (Join-Path $PSScriptRoot "config.panorama-test.json"),
    $Stage
)
if ($Force) {
    $arguments += "--force"
}
& $exe @arguments
