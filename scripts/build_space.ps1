# Assemble the Hugging Face Space (Docker SDK) folder in dist\space.
# Upload afterwards with:  .venv\Scripts\hf.exe upload <user>/luekemia-cell-api dist\space . --repo-type space
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$dst = "dist\space"
if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
New-Item -ItemType Directory -Force "$dst\api", "$dst\models", "$dst\src\vision" | Out-Null
Copy-Item api\*.py, api\requirements.txt "$dst\api\"
# the smear cell finder used by /analyze-smear
Copy-Item src\vision\__init__.py, src\vision\segment.py "$dst\src\vision\"
# deployed model, C-NMC colour statistics for smear crops, and the cell-type filter
Copy-Item models\model_meta.json, models\MH51401122.onnx, models\cnmc_stain.json, models\cell_type_filter.onnx, models\cell_type_meta.json "$dst\models\"
Copy-Item api\Dockerfile "$dst\Dockerfile"
Copy-Item deploy\space_README.md "$dst\README.md"
# Space builds from its root; .env is never copied.
Get-ChildItem -Recurse $dst | Where-Object { -not $_.PSIsContainer } | ForEach-Object { "{0,10:N0}  {1}" -f $_.Length, $_.FullName.Substring((Resolve-Path $dst).Path.Length + 1) }
