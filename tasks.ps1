# AutoForge task runner (PowerShell). Usage:
#   .\tasks.ps1 <verb>
# Verbs: setup, init-db, test-nemotron, dashboard, run, smoke, clean, help
#
# Examples:
#   .\tasks.ps1 setup           # conda env create / update
#   .\tasks.ps1 dashboard       # launch Streamlit
#   .\tasks.ps1 run             # run pipeline with a default test dataset
#   .\tasks.ps1 smoke           # run skeleton smoke test

param(
    [Parameter(Position = 0)]
    [string]$Verb = "help",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

function Invoke-Setup {
    Write-Host "Creating/updating conda env 'autoforge'..." -ForegroundColor Cyan
    conda env update -f environment.yml --prune
}

function Invoke-InitDb {
    Write-Host "Initializing SQLite store..." -ForegroundColor Cyan
    python scripts/init_db.py
}

function Invoke-TestNemotron {
    Write-Host "Pinging Nemotron endpoint..." -ForegroundColor Cyan
    python scripts/test_nemotron.py
}

function Invoke-Dashboard {
    Write-Host "Launching Streamlit dashboard on http://localhost:8501 ..." -ForegroundColor Cyan
    streamlit run dashboard/app.py
}

function Invoke-Run {
    $dataset = if ($Rest.Count -ge 1) { $Rest[0] } else { "data/uploads/test.csv" }
    $objective = if ($Rest.Count -ge 2) { $Rest[1] } else { "predict churn" }
    Write-Host "Running pipeline: dataset=$dataset objective='$objective'" -ForegroundColor Cyan
    python scripts/run_pipeline.py --dataset $dataset --objective $objective
}

function Invoke-Smoke {
    Write-Host "Running skeleton smoke tests..." -ForegroundColor Cyan
    pytest tests/ -v
}

function Invoke-Clean {
    Write-Host "Removing caches and SQLite DB..." -ForegroundColor Yellow
    Get-ChildItem -Path . -Include __pycache__, .pytest_cache, .ruff_cache, .mypy_cache `
        -Recurse -Directory -Force -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force
    if (Test-Path data/autoforge.db) { Remove-Item data/autoforge.db -Force }
}

function Invoke-Help {
    @"
AutoForge tasks
---------------
setup           Create/update the 'autoforge' conda env from environment.yml
init-db         Initialize the SQLite memory store
test-nemotron   Smoke-test the NVIDIA Nemotron endpoint
dashboard       Launch the Streamlit dashboard
run [dataset] [objective]
                Run the pipeline (defaults: data/uploads/test.csv, 'predict churn')
smoke           Run the pytest smoke suite
clean           Remove caches and the SQLite DB
help            Show this message
"@ | Write-Host
}

switch ($Verb.ToLower()) {
    "setup"          { Invoke-Setup }
    "init-db"        { Invoke-InitDb }
    "test-nemotron"  { Invoke-TestNemotron }
    "dashboard"      { Invoke-Dashboard }
    "run"            { Invoke-Run }
    "smoke"          { Invoke-Smoke }
    "clean"          { Invoke-Clean }
    "help"           { Invoke-Help }
    default {
        Write-Host "Unknown verb: $Verb" -ForegroundColor Red
        Invoke-Help
        exit 1
    }
}
