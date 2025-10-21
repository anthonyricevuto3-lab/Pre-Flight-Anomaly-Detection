# PowerShell script to create a virtual environment for the project

$projectPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $projectPath "venv"

if (Test-Path $venvPath) {
    Write-Host "Virtual environment already exists at $venvPath"
} else {
    python -m venv $venvPath
    Write-Host "Virtual environment created at $venvPath"
}

# Activate the virtual environment
& "$venvPath\Scripts\Activate.ps1"

# Install required packages
pip install -r (Join-Path $projectPath "requirements.txt")