$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m src.prepare_sanskrit
$env:PINGALA_TRAIN_FILE=(Resolve-Path 'data\sanskrit_train.txt').Path
$env:PINGALA_OUTPUT=(Resolve-Path '.').Path + '\artifacts\sanskrit_model'
.venv\Scripts\python.exe -m src.train_lm
.venv\Scripts\python.exe -m src.generate_sanskrit
.venv\Scripts\python.exe -m src.generate_lm
.venv\Scripts\python.exe -m unittest discover -s tests -v
