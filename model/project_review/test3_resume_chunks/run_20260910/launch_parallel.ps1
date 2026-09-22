$ErrorActionPreference = "Stop"

$pythonExe = "D:\miniconda\envs\gamstekpeaking\python.exe"
$modelRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$processes = @()
foreach ($i in 1..4) {
    $chunkDir = Join-Path $PSScriptRoot ("chunk_{0}" -f $i)
    $stdout = Join-Path $logDir ("chunk_{0}.stdout.log" -f $i)
    $stderr = Join-Path $logDir ("chunk_{0}.stderr.log" -f $i)
    $arguments = @(
        "-u", "-X", "utf8", "-m", "inference.cli",
        "--config", "configs\massnova.json",
        "--mode", "massnova",
        "--batch_dir", $chunkDir,
        "--exp_name", "test3"
    )
    $processes += Start-Process `
        -FilePath $pythonExe `
        -ArgumentList $arguments `
        -WorkingDirectory $modelRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
}

$processes | Select-Object Id, ProcessName, StartTime
