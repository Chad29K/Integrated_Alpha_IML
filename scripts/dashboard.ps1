$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$app = Join-Path $PSScriptRoot "..\dashboard_app.py"
& $python $app --host 127.0.0.1 --port 8000
