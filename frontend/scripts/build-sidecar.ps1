$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }
& $python -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('PyInstaller') else 1)"
if ($LASTEXITCODE -ne 0) {
    & $python -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller installation failed' }
}
$out = Join-Path $root 'frontend\src-tauri\binaries'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$resources = Join-Path $root 'frontend\src-tauri\resources'
New-Item -ItemType Directory -Force -Path $resources | Out-Null
$llamaSource = Join-Path $root 'runtime\llama'
if ($env:LLAMA_SERVER_EXECUTABLE) { $llamaSource = Split-Path -Parent $env:LLAMA_SERVER_EXECUTABLE }
if (-not (Test-Path -LiteralPath (Join-Path $llamaSource 'llama-server.exe'))) { throw 'Install llama-server and its DLLs in runtime/llama before packaging.' }
$llamaTarget = Join-Path $resources 'llama'
New-Item -ItemType Directory -Force -Path $llamaTarget | Out-Null
Get-ChildItem -LiteralPath $llamaSource -File | Where-Object { $_.Extension -eq '.dll' -or $_.Name -eq 'llama-server.exe' -or $_.Name -like 'LICENSE*' } | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $llamaTarget -Force }
$legacyMarker = Join-Path $resources 'legacy-data-location.txt'
[System.IO.File]::WriteAllText($legacyMarker, (Join-Path $root 'api'), [System.Text.UTF8Encoding]::new($false))
$arguments = @('-m','PyInstaller','--noconfirm','--onefile','--exclude-module','ollama','--exclude-module','langchain_ollama','--name','kontur-backend','--paths',(Join-Path $root 'api'),'--collect-all','chromadb','--collect-all','pymorphy3','--collect-all','pymorphy3_dicts_ru','--hidden-import','whisper_utils','--distpath',$out,'--workpath',(Join-Path $root 'build\pyinstaller'),'--specpath',(Join-Path $root 'build'))
$arguments += @('--add-data', ((Join-Path $root 'api\agent_skills') + ':agent_skills'))
$arguments += @('--copy-metadata','pymorphy3-dicts-ru')
$arguments += (Join-Path $root 'frontend\scripts\backend_entry.py')
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Backend sidecar build failed' }
