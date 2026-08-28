$ErrorActionPreference = "Stop"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher (py) not found. Install Python 3.10-3.12 first."
}

$selectedVersion = $null
foreach ($version in @("3.11", "3.12", "3.10")) {
    py "-$version" -c "import sys; print(sys.version)" *> $null
    if ($LASTEXITCODE -eq 0) {
        $selectedVersion = $version
        break
    }
}
if (-not $selectedVersion) {
    throw "Python 3.10-3.12 is required. Current system Python 3.7 is too old."
}
py "-$selectedVersion" -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\street3d.exe --project . init
Write-Host "Setup complete. Put panoramas in input\panoramas."
