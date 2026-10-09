# Veyra Backend Deployment to Azure Container Apps
# Prerequisites:
#   - Azure CLI (az) logged in: az login
#   - Docker running locally
#   - Subscription ID for Microsoft for Startups account
#
# Usage:
#   1. Set SUBSCRIPTION_ID below (or pass as env var)
#   2. Run: ./azure/deploy.ps1
#   3. When prompted, paste secret values (they won't be echoed)

param(
    [string]$SUBSCRIPTION_ID = "6d947d68-b871-4787-9681-abc4ec62186c",
    [string]$RESOURCE_GROUP = "rg-veyra",
    [string]$LOCATION = "westeurope",
    [string]$ACR_NAME = "acrveyra",
    [string]$APP_NAME = "ca-veyra-backend",
    [string]$IMAGE_TAG = "latest",
    [string]$BicepFile = (Join-Path $PSScriptRoot "main.bicep")
)

$ErrorActionPreference = "Stop"

Write-Host "=== Veyra Backend Deployment ===" -ForegroundColor Cyan
Write-Host "Subscription:   $SUBSCRIPTION_ID"
Write-Host "Resource Group: $RESOURCE_GROUP"
Write-Host "Location:       $LOCATION"
Write-Host ""

# --- Step 0: Set Azure context ---
Write-Host "[0/6] Setting Azure subscription context..." -ForegroundColor Yellow
az account set --subscription $SUBSCRIPTION_ID
if ($LASTEXITCODE -ne 0) { throw "az account set failed - run 'az login' first" }
$currentSub = az account show --query "name" -o tsv
Write-Host "  Active subscription: $currentSub" -ForegroundColor Green

# --- Step 1: Create / verify resource group ---
Write-Host ""
Write-Host "[1/6] Ensuring resource group '$RESOURCE_GROUP' exists..." -ForegroundColor Yellow
az group create --name $RESOURCE_GROUP --location $LOCATION --output none
Write-Host "  Resource group ready." -ForegroundColor Green

# --- Step 2: Create ACR (if not exists) & build/push image ---
Write-Host ""
Write-Host "[2/6] Azure Container Registry build + push..." -ForegroundColor Yellow
$acrExists = az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query "name" -o tsv 2>$null
if (-not $acrExists) {
    Write-Host "  Creating ACR '$ACR_NAME'..."
    az acr create --name $ACR_NAME --resource-group $RESOURCE_GROUP `
        --sku Basic --admin-enabled true --location $LOCATION --output none
}
az acr build --registry $ACR_NAME --resource-group $RESOURCE_GROUP `
    --image "veyra-backend:$IMAGE_TAG" --file Dockerfile .
if ($LASTEXITCODE -ne 0) { throw "ACR build failed" }
$IMAGE_FQDN = "$ACR_NAME.azurecr.io/veyra-backend:$IMAGE_TAG"
Write-Host "  Image ready: $IMAGE_FQDN" -ForegroundColor Green

# --- Step 3: Collect secure parameters ---
Write-Host ""
Write-Host "[3/6] Collecting deployment parameters..." -ForegroundColor Yellow

function Read-Secret($prompt) {
    $value = Read-Host $prompt -AsSecureString
    return [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($value))
}

function Read-Plain($prompt, $default = "") {
    $val = Read-Host "$prompt (default: $default)"
    if ([string]::IsNullOrWhiteSpace($val)) { return $default }
    return $val
}

$DATABASE_URL             = Read-Secret "DATABASE_URL (Supabase PostgreSQL connection string)"
$AZURE_OPENAI_API_KEY     = Read-Secret "AZURE_OPENAI_API_KEY"
$AZURE_OPENAI_ENDPOINT    = Read-Plain  "AZURE_OPENAI_ENDPOINT (e.g. https://<name>.openai.azure.com/)"
$AZURE_OPENAI_DEPLOYMENT  = Read-Plain  "AZURE_OPENAI_DEPLOYMENT (e.g. gpt-4o-mini)"
$AZURE_OPENAI_API_VERSION = Read-Plain  "AZURE_OPENAI_API_VERSION" "2024-08-01-preview"
$WHATSAPP_ACCESS_TOKEN    = Read-Secret "WHATSAPP_ACCESS_TOKEN"
$WHATSAPP_PHONE_NUMBER_ID = Read-Plain  "WHATSAPP_PHONE_NUMBER_ID"
$WHATSAPP_VERIFY_TOKEN    = Read-Secret "WHATSAPP_VERIFY_TOKEN"
$WHATSAPP_BUSINESS_ACCOUNT_ID = Read-Plain "WHATSAPP_BUSINESS_ACCOUNT_ID"
$CORS_ORIGINS             = Read-Plain  "VEYRA_CORS_ORIGINS (comma-separated)" "https://veyra-1.vercel.app"

# --- Step 4: Deploy Bicep template ---
Write-Host ""
Write-Host "[4/6] Deploying Bicep template (Container Apps + Storage + File Share)..." -ForegroundColor Yellow
$STORAGE_NAME = "stveyrahfcache$(Get-Random -Minimum 1000 -Maximum 9999)"
$DEPLOY_NAME = "veyra-deploy-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

az deployment group create `
    --name $DEPLOY_NAME `
    --resource-group $RESOURCE_GROUP `
    --template-file $BicepFile `
    --parameters `
        location=$LOCATION `
        storageAccountName=$STORAGE_NAME `
        containerImage=$IMAGE_FQDN `
        corsOrigins=$CORS_ORIGINS `
        databaseUrl=$DATABASE_URL `
        azureOpenaiApiKey=$AZURE_OPENAI_API_KEY `
        azureOpenaiEndpoint=$AZURE_OPENAI_ENDPOINT `
        azureOpenaiDeployment=$AZURE_OPENAI_DEPLOYMENT `
        azureOpenaiApiVersion=$AZURE_OPENAI_API_VERSION `
        whatsappAccessToken=$WHATSAPP_ACCESS_TOKEN `
        whatsappPhoneNumberId=$WHATSAPP_PHONE_NUMBER_ID `
        whatsappVerifyToken=$WHATSAPP_VERIFY_TOKEN `
        whatsappBusinessAccountId=$WHATSAPP_BUSINESS_ACCOUNT_ID `
    --output table

if ($LASTEXITCODE -ne 0) { throw "Bicep deployment failed" }

# --- Step 5: Attach ACR to Container Apps (so it can pull images) ---
Write-Host ""
Write-Host "[5/6] Attaching ACR to Container Apps environment..." -ForegroundColor Yellow
$ENV_NAME = az deployment group show --name $DEPLOY_NAME --resource-group $RESOURCE_GROUP `
    --query "properties.outputs.environmentName.value" -o tsv
az containerapp registry set `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --server "$ACR_NAME.azurecr.io" `
    --identity system `
    --output none
# Grant ACR pull to the managed identity
$principalId = az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP `
    --query "identity.principalId" -o tsv 2>$null
if ($principalId) {
    $acrId = az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query "id" -o tsv
    az role assignment create `
        --assignee $principalId `
        --role "AcrPull" `
        --scope $acrId `
        --output none 2>$null
    Write-Host "  ACR pull role assigned." -ForegroundColor Green
}

# --- Step 6: Post-deploy: grab URL, warmup the model ---
Write-Host ""
Write-Host "[6/6] Post-deployment: fetching URL + warmup..." -ForegroundColor Yellow

$BACKEND_URL = az deployment group show --name $DEPLOY_NAME --resource-group $RESOURCE_GROUP `
    --query "properties.outputs.backendUrl.value" -o tsv
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DEPLOYMENT COMPLETE"
Write-Host "  Backend URL: $BACKEND_URL"
Write-Host "  Health:      $BACKEND_URL/health"
Write-Host "  Model:       $BACKEND_URL/model-status"
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Calling /warmup to preload ASR models (may take 1-2 min)..."
try {
    $warmup = Invoke-RestMethod -Uri "$BACKEND_URL/warmup" -Method Get -TimeoutSec 300
    Write-Host "  Warmup result: $($warmup.status)" -ForegroundColor Green
} catch {
    Write-Host "  Warmup call failed (may still be booting): $_" -ForegroundColor DarkYellow
    Write-Host "  Manually call:  curl $BACKEND_URL/warmup"
}

Write-Host ""
Write-Host "WhatsApp webhook URL: $BACKEND_URL/webhook"
Write-Host "Update Meta App webhook callback URL to the above."
Write-Host ""
Write-Host "Frontend (Vercel): set ?api=$BACKEND_URL on the demo page to use this backend."
Write-Host "Or update AZURE_BACKEND in site/demo/demo.js and re-deploy Vercel."
