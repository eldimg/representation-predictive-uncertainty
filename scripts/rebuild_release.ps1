$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$localPython = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$pythonCommand = if (Test-Path -LiteralPath $localPython) { $localPython } else { "python" }

function Invoke-PythonStep {
    param([Parameter(Mandatory = $true)][string]$ScriptPath)
    & $pythonCommand $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "Release rebuild step failed ($LASTEXITCODE): $ScriptPath"
    }
}

Push-Location $repositoryRoot
try {
    Invoke-PythonStep .\scripts\build_publication_outputs.py
    Invoke-PythonStep .\scripts\audit_manuscript_numbers.py
    Invoke-PythonStep .\scripts\build_release_manifest.py
    Invoke-PythonStep .\scripts\validate_release.py
}
finally {
    Pop-Location
}
