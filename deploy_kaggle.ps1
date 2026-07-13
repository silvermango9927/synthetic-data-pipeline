# deploy_kaggle.ps1
# Automates the entire Kaggle CLI deployment workflow

$patPath = "kaggle_job\github_pat.txt"

# 1. Check for GitHub PAT
if (-not (Test-Path $patPath)) {
    Write-Error "Error: $patPath not found!"
    Write-Host "Please create a file at '$patPath' and paste your GitHub Personal Access Token (PAT) inside it." -ForegroundColor Yellow
    exit 1
}

# 2. Parse Kaggle Username using 'kaggle config view'
try {
    $configView = & kaggle config view 2>$null
    $usernameLine = $configView | Where-Object { $_ -match "- username:\s*(.*)" }
    if ($usernameLine -and ($Matches[1])) {
        $username = $Matches[1].Trim()
    } else {
        # Fallback to kaggle.json
        $kaggleJsonPath = "$HOME\.kaggle\kaggle.json"
        if (Test-Path $kaggleJsonPath) {
            $kaggleJson = Get-Content $kaggleJsonPath -Raw | ConvertFrom-Json
            $username = $kaggleJson.username
        }
    }
    
    if (-not $username) { throw "Username could not be parsed." }
    Write-Host "Detected Kaggle Username: $username" -ForegroundColor Green
} catch {
    Write-Error "Error: Failed to retrieve Kaggle username. Please ensure kaggle CLI is authenticated."
    exit 1
}

# 4. Replace placeholder in metadata JSONs
Write-Host "Configuring metadata files..." -ForegroundColor Cyan
$metaFiles = @(
    "outputs\hf_datasets\synthetic-asr-hi\dataset-metadata.json",
    "outputs\hf_datasets\synthetic-asr-zh\dataset-metadata.json",
    "kaggle_job\kernel-metadata.json"
)

foreach ($file in $metaFiles) {
    if (Test-Path $file) {
        $content = Get-Content $file -Raw
        $updatedContent = $content -replace "CHANGE_ME_TO_KAGGLE_USERNAME", $username
        Set-Content $file -Value $updatedContent -Encoding utf8
        Write-Host "Updated: $file"
    } else {
        Write-Warning "File not found: $file"
    }
}

# 5. Push Datasets to Kaggle
Write-Host "Uploading synthetic-asr-hi dataset to Kaggle..." -ForegroundColor Cyan
kaggle datasets create -p outputs\hf_datasets\synthetic-asr-hi

Write-Host "Uploading synthetic-asr-zh dataset to Kaggle..." -ForegroundColor Cyan
kaggle datasets create -p outputs\hf_datasets\synthetic-asr-zh

# 6. Push Kernel to Kaggle
Write-Host "Pushing training kernel to Kaggle..." -ForegroundColor Cyan
kaggle kernels push -p kaggle_job

Write-Host "Deployment completed successfully! Check the status using:" -ForegroundColor Green
Write-Host "kaggle kernels status $username/whisper-scaling-laws" -ForegroundColor Yellow
