<#
Simple helper to install Poppler on Windows if `choco` or `winget` exists.
Falls back to printed manual instructions when package manager is missing.
#>
param(
    [string]$InstallDir = "$PSScriptRoot\\third_party\\poppler"
)

# Try Chocolatey
if (Get-Command choco -ErrorAction SilentlyContinue) {
    choco install poppler -y
    Write-Host "Installed poppler via Chocolatey"
    exit 0
}

# Try winget
if (Get-Command winget -ErrorAction SilentlyContinue) {
    winget install --id Poppler.Poppler -e
    Write-Host "Installed poppler via winget"
    exit 0
}

Write-Host "No choco/winget detected. Please download Poppler for Windows:" -ForegroundColor Yellow
Write-Host "https://github.com/oschwartz10612/poppler-windows/releases" -ForegroundColor Cyan
Write-Host "Extract to a folder, e.g., C:\\poppler-25.12.0, then add C:\\poppler-25.12.0\\Library\\bin to your PATH."
Write-Host "Temporary add to PATH in PowerShell (current session):"
Write-Host "  $env:PATH += ';C:\\poppler-25.12.0\\Library\\bin'"