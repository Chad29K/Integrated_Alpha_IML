$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$main = Join-Path $PSScriptRoot "..\main.py"
& $python $main run-all
