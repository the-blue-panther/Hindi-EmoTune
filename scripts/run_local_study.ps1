param(
    [string]$Project = 'D:\Downloads\Projects\Hindi Emotion controlled TTS with Voice cloning'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$runtime = Join-Path $repo '.venv-study'
if (-not (Test-Path (Join-Path $runtime 'Scripts\python.exe'))) {
    & py -3.11 -m venv $runtime
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 environment creation failed' }
}
$python = Join-Path $runtime 'Scripts\python.exe'
& $python -X utf8 -m pip install -r (Join-Path $repo 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. No synthesis result was produced.' }
& $python -X utf8 -c "import torch, TTS, whisper, jiwer, soundfile, librosa; print('Runtime imports passed')"
if ($LASTEXITCODE -ne 0) { throw 'Runtime import check failed' }
$run = Join-Path $repo ('results\study_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $python -X utf8 (Join-Path $PSScriptRoot 'run_hindi_baseline.py') $Project $run --compare
if ($LASTEXITCODE -ne 0) { throw "Study incomplete. Inspect retained outputs in $run" }
Write-Output "Completed baseline, sad, angry and happy conditions: $run"
