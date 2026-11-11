# Stage the Kaggle Model upload in dist\kaggle (model + one ONNX instance).
#   powershell -ExecutionPolicy Bypass -File scripts\build_kaggle.ps1 -Owner <kaggle-username>
# Then:
#   kaggle models create -p dist\kaggle
#   kaggle models instances create -p dist\kaggle\onnx
# The Kaggle token is read from KAGGLE_CONFIG_DIR (E:\cache\kaggle); it is never stored in this repo.
param([Parameter(Mandatory = $true)][string]$Owner)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$dst = "dist\kaggle"
if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
New-Item -ItemType Directory -Force "$dst\onnx" | Out-Null

$card = (Get-Content MODEL_CARD.md -Raw) -replace '(?s)^---.*?---\s*', ''   # drop the HF YAML header
$model = [ordered]@{
  ownerSlug   = $Owner
  title       = "Cell Triage"
  slug        = "cell-triage"
  subtitle    = "ALL vs normal white blood cell classifier with a refer-to-expert band"
  isPrivate   = $false
  description = $card
  publishTime = ""
  provenanceSources = "C-NMC 2019 (TCIA, CC BY 3.0), https://doi.org/10.7937/tcia.2019.dc64i46r"
}
$model | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 "$dst\model-metadata.json"

$instance = [ordered]@{
  ownerSlug          = $Owner
  modelSlug          = "cell-triage"
  instanceSlug       = "effb0-ens5-v1"
  framework          = "onnx"
  overview           = "Five-seed EfficientNet-B0 ensemble, one ONNX graph. Input: image float32 [B,3,224,224], ImageNet-normalised. Outputs: logits [B,5], cam [B,7,7]. Apply temperature and band from model_meta.json. Research demo, not for clinical use."
  usage              = "See https://github.com/HaiCLOP/luekemia_cell_model (api/inference.py) for preprocessing, calibration and the referral band."
  licenseName        = "CC BY 4.0"
  fineTunable        = $false
  trainingData       = @("https://doi.org/10.7937/tcia.2019.dc64i46r")
  modelInstanceType  = "Unspecified"
}
$instance | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 "$dst\onnx\model-instance-metadata.json"
Copy-Item models\effb0-ens5-v1.onnx, models\model_meta.json "$dst\onnx\"
Get-ChildItem -Recurse $dst -File | ForEach-Object { "{0,12:N0}  {1}" -f $_.Length, $_.Name }
