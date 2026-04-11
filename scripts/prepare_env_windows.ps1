# PowerShell helper: create venv and install pip requirements (Windows)
param()

python -m venv .venv
# Activate the venv in PowerShell
try {
    . .\.venv\Scripts\Activate.ps1
} catch {
    Write-Host "If activation fails, run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
}
python -m pip install --upgrade pip
pip install -r requirements.txt
Write-Host "Environment created and requirements installed. To activate in a new PowerShell: .\\.venv\\Scripts\\Activate.ps1"