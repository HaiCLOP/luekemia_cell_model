# Rebuilds every number and figure in reports/ from scratch.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1              # full run, about 8 h on a GTX 1650
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -SkipTraining  # reuse checkpoints, about 15 min
#
# Fixed seeds throughout. The held-out TEST split is never touched here; it gets evaluated exactly once,
# together with the external ALL-IDB2 set, after every modelling choice is frozen.
param([switch]$SkipTraining)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONWARNINGS = "ignore"
$env:HF_HOME = "E:\cache\hf"; $env:TORCH_HOME = "E:\cache\torch"; $env:TMP = "E:\cache\tmp"; $env:TEMP = "E:\cache\tmp"
$py = ".\.venv\Scripts\python.exe"

function Step($title, [string[]]$cmd) {
    Write-Host "`n=== $title" -ForegroundColor Green
    & $py @cmd
    if ($LASTEXITCODE -ne 0) { throw "$title failed (exit $LASTEXITCODE)" }
}

# Data: download, audit, patient-level splits, leakage tests
Step "Download and audit C-NMC"      @("-m", "src.download")
Step "Patient-level splits"           @("-m", "src.split")
Step "Unit and leakage tests"         @("-m", "pytest", "-q")

if (-not $SkipTraining) {
    # Baselines
    Step "Baseline A: frozen features + LR" @("-m", "src.baselines")
    Step "Baseline B: small CNN"            @("-m", "src.train", "--name", "baseline_b_smallcnn", "--set", "train.arch=small_cnn", "train.pretrained=false", "train.batch_size=32", "train.grad_accum=1", "train.stage1_lr=0.0003", "train.early_stop_patience=8")
    # Main model (five seeds), other architectures and ablations
    foreach ($s in 42, 43, 44, 45, 46) { Step "EfficientNet-B0 seed $s" @("-m", "src.train", "--name", "effb0_main_s$s", "--set", "seed=$s") }
    Step "MobileNetV3-L"           @("-m", "src.train", "--name", "mnv3_s42", "--set", "train.arch=mobilenetv3_large_100")
    Step "DINOv2 ViT-S/14"         @("-m", "src.train", "--name", "dinov2s_s42", "--set", "train.arch=vit_small_patch14_dinov2")
    Step "Ablation: frozen"        @("-m", "src.train", "--name", "abl_frozen_s42", "--set", "train.stage2_epochs=0")
    Step "Ablation: no aug"        @("-m", "src.train", "--name", "abl_aug_none_s42", "--set", "train.augment=none")
    Step "Ablation: HED aug"       @("-m", "src.train", "--name", "abl_aug_hed_s42", "--set", "train.augment=geometric_hed")
    Step "Ablation: sampler"       @("-m", "src.train", "--name", "abl_sampler_s42", "--set", "train.imbalance=weighted_sampler")
    Step "Ablation: label smooth"  @("-m", "src.train", "--name", "abl_labelsmooth_s42", "--set", "train.label_smoothing=0.1")
    Step "Ablation: scale jitter"  @("-m", "src.train", "--name", "abl_scalejit_s42", "--set", "train.augment=geometric_scale")
}

# Validation analysis
Step "Comparison table with subject CIs" @("-m", "src.report")
Step "Calibration and referral band"   @("-m", "src.triage", "--mc")
Step "Explainability"                  @("-m", "src.explain", "--n", "200")
Step "Size reliance"                     @("-m", "src.size_check", "effb0_main_s42", "effb0_main_s43", "effb0_main_s44", "effb0_main_s45", "effb0_main_s46", "mnv3_s42", "dinov2s_s42", "baseline_b_smallcnn", "abl_aug_none_s42", "abl_aug_hed_s42", "abl_scalejit_s42")

# Deployable model and website data
Step "Export ONNX ensemble" @("-m", "src.export")
Step "Export website data"  @("scripts\export_web_data.py")

Write-Host "`nAll validation results regenerated in reports\." -ForegroundColor Green
