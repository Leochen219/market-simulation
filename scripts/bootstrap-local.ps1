param([switch]$SkipModel)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $uv = Get-Command uv -ErrorAction Stop
    if (-not (Test-Path -LiteralPath '.venv-m0/Scripts/python.exe')) {
        & $uv.Source venv --python 3.11 .venv-m0
        if ($LASTEXITCODE -ne 0) { throw 'Creating the project environment failed.' }
    }
    # Install the exact CUDA build first; never resolve torch from the CPU index.
    & $uv.Source pip install --python .venv-m0/Scripts/python.exe 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'Installing CUDA torch failed.' }
    & $uv.Source pip install --python .venv-m0/Scripts/python.exe 'torch==2.11.0+cu128' 'laya==0.4.13' 'transformers==4.57.6' 'PyYAML==6.0.3' 'pytest==9.1.1' 'pytest-asyncio==1.4.0' 'python-dotenv==1.2.4' 'setuptools==78.1.0'
    if ($LASTEXITCODE -ne 0) { throw 'Installing local dependencies failed.' }
    # Reuse this environment's build backend; Windows application control can block
    # temporary Python executables created by an isolated editable build.
    & $uv.Source pip install --python .venv-m0/Scripts/python.exe --no-deps --no-build-isolation -e .
    if ($LASTEXITCODE -ne 0) { throw 'Installing capclaims failed.' }
    if (-not $SkipModel) {
        & .venv-m0/Scripts/python.exe scripts/download_laya.py
        if ($LASTEXITCODE -ne 0) { throw 'Downloading the pinned model failed.' }
    }
    & .venv-m0/Scripts/python.exe -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA is required by local-laya.yaml"'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed.' }
} finally {
    Pop-Location
}
