# Runs the full set of comparison experiments back to back (about 7 h on a GTX 1650), e.g. overnight:
#   powershell -ExecutionPolicy Bypass -File scripts\queue_experiments.ps1
# Finished runs are skipped, so the queue can be restarted after a crash. Progress shows up in monitor/.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
$env:PYTHONIOENCODING = "utf-8"; $env:TORCH_HOME = "E:\cache\torch"; $env:HF_HOME = "E:\cache\hf"
$env:TMP = "E:\cache\tmp"; $env:TEMP = "E:\cache\tmp"; $env:PYTHONWARNINGS = "ignore"
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force reports\logs | Out-Null

# Stop Windows from sleeping while the queue runs (cleared automatically when the script exits).
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

function Log($msg) { $msg | Out-File -Append -Encoding utf8 reports\logs\queue.log; Write-Output $msg }

function Run($name, [string[]]$sets) {
    if (Test-Path "reports\runs\$name.json") { Log "=== $name skipped (already finished)"; return }
    $t = Get-Date
    Log "=== $name started $t"
    $cmd = @("-m", "src.train", "--name", $name)
    if ($sets.Count) { $cmd += "--set"; $cmd += $sets }
    & $py @cmd *> "reports\logs\$name.log"
    Log ("=== $name exit $LASTEXITCODE after {0:N1} min" -f ((Get-Date) - $t).TotalMinutes)
}

# Lighter architecture, worth knowing for CPU deployment
Run "mnv3_s42"   @("train.arch=mobilenetv3_large_100")

# Extra seeds of the main recipe; together with seed 42 they form the deployed ensemble
foreach ($s in 43, 44, 45, 46) { Run "effb0_main_s$s" @("seed=$s") }

# A self-supervised ViT for comparison
Run "dinov2s_s42" @("train.arch=vit_small_patch14_dinov2")

# Ablations of the main recipe
Run "abl_frozen_s42"      @("train.stage2_epochs=0")
Run "abl_aug_none_s42"    @("train.augment=none")
Run "abl_aug_hed_s42"     @("train.augment=geometric_hed")
Run "abl_sampler_s42"     @("train.imbalance=weighted_sampler")
Run "abl_labelsmooth_s42" @("train.label_smoothing=0.1")
Run "abl_scalejit_s42"    @("train.augment=geometric_scale")

& $py -m src.report *> reports\logs\report.log
Log "=== queue done $(Get-Date)"
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
