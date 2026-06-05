[CmdletBinding(PositionalBinding = $false)]
param(
    [switch]$Help,
    [string]$Config = "",
    [string]$LogRoot = "logs",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArgs = @()
)

$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host "Usage:"
    Write-Host "  .\scripts\run_background.ps1 [-Config .\configs\config.local.toml] [-LogRoot logs] [-- nx end_time forcing_power viscosity resistivity tag_suffix fixed_dt snapshot_dt seed]"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\scripts\run_background.ps1"
    Write-Host "  .\scripts\run_background.ps1 -Config .\configs\config.local.toml"
    Write-Host "  .\scripts\run_background.ps1 -Config .\configs\config.local.toml -- 8 0.001 10 0.01 0.01 smoke 0.001 0.01 1234"
    exit 0
}

if ($RunArgs.Count -gt 0 -and $RunArgs[0] -eq "--") {
    if ($RunArgs.Count -eq 1) {
        $RunArgs = @()
    } else {
        $RunArgs = $RunArgs[1..($RunArgs.Count - 1)]
    }
}

function Quote-Argument {
    param([string]$Value)
    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
$logDir = if ([System.IO.Path]::IsPathRooted($LogRoot)) {
    $LogRoot
} else {
    Join-Path $repoRoot $LogRoot
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutLog = Join-Path $logDir "simulation_$stamp.out.log"
$stderrLog = Join-Path $logDir "simulation_$stamp.err.log"
$pidFile = Join-Path $logDir "simulation_$stamp.pid"

$arguments = @((Join-Path "scripts" "run_simulation.jl"))
if ($Config -ne "") {
    $arguments += @("--config", $Config)
}
$arguments += $RunArgs
$argumentLine = ($arguments | ForEach-Object { Quote-Argument $_ }) -join " "

$process = Start-Process `
    -FilePath "julia" `
    -ArgumentList $argumentLine `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

$process.Id | Set-Content -Path $pidFile -Encoding UTF8

Write-Host "Started background simulation."
Write-Host "PID: $($process.Id)"
Write-Host "stdout: $stdoutLog"
Write-Host "stderr: $stderrLog"
Write-Host "pid file: $pidFile"
