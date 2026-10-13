# Azure Deployment Script for Veyra
# This script deploys Veyra to Azure Web App with the specified configuration

# Configuration
$resourceGroup = "rg-veyra"
$region = "westeurope"
$appName = "veyra"
$databaseUrl = "postgresql://postgres.ropmzfftqbxkvsnwnyah:yAGYZ%3Fkm5R_TBTu@aws-0-eu-central-1.pooler.supabase.com:6543/postgres"
$vercelUrl = "https://veyra-1.vercel.app"
$whatsappAccessToken = "EAAZBuA5uUWMsBSjl9C2AuypqqPZCtDp3bNxRR16F9L7v7L7FDZAS4ilmuEupuMqXRkO1lcyNL4o4sEMWyTRfkeSOHmQPdALZC7bB3pgwp1ZBlRNMUgJpo6Et9nlqNa0OtYGnfBsAaZBMd05LeNMCBtXZCbXjUK50j3ZC9g2Mj3Jtddbuk3UH07yCMNIqgk3dE4VNOd11ouU68B8uMcoylepOCJ6y9i9R0P1gKCHOVnOyZBJGlxAZAcrvK1peHOSMpOlTprw3WwaoNJb10ZBkiVrd5kR2AYZD"
$whatsappPhoneNumberId = "1394270210433135"
$whatsappVerifyToken = "VeyraWebhook_9xP4mQ7rL2_kT8"
$whatsappBusinessAccountId = "28363089073372159"

Write-Host "=== Veyra Azure Deployment Script ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Resource Group: $resourceGroup"
Write-Host "  Region: $region"
Write-Host "  App Name: $appName"
Write-Host ""

# Check if Azure CLI is available
$azAvailable = Get-Command az -ErrorAction SilentlyContinue
if (-not $azAvailable) {
    Write-Host "Azure CLI not found in PATH. Please:" -ForegroundColor Red
    Write-Host "1. Close and reopen your terminal/IDE"
    Write-Host "2. Or run: refreshenv in PowerShell"
    Write-Host "3. Or add Azure CLI to your PATH manually"
    Write-Host ""
    Write-Host "Azure CLI is typically installed at:" -ForegroundColor Yellow
    Write-Host "  - C:\Program Files (x86)\Microsoft SDKs\Azure\CLI2\az.cmd"
    Write-Host "  - C:\Program Files\Microsoft Azure CLI\az.exe"
    exit 1
}

Write-Host "Azure CLI found! Proceeding with deployment..." -ForegroundColor Green
Write-Host ""

# Login to Azure (interactive)
Write-Host "Step 1: Logging into Azure..." -ForegroundColor Cyan
az login

# Create resource group
Write-Host "Step 2: Creating resource group..." -ForegroundColor Cyan
az group create --name $resourceGroup --location $region

# Create App Service Plan
Write-Host "Step 3: Creating App Service Plan..." -ForegroundColor Cyan
$planName = "$($appName)-plan"
az appservice plan create `
    --name $planName `
    --resource-group $resourceGroup `
    --location $region `
    --sku B1 `
    --is-linux

# Create Web App
Write-Host "Step 4: Creating Web App..." -ForegroundColor Cyan
az webapp create `
    --name $appName `
    --resource-group $resourceGroup `
    --plan $planName `
    --runtime "PYTHON:3.12" `
    --deployment-container-image-name "veyra:latest"

# Configure environment variables
Write-Host "Step 5: Configuring environment variables..." -ForegroundColor Cyan
az webapp config appsettings set `
    --resource-group $resourceGroup `
    --name $appName `
    --settings `
    DATABASE_URL="$databaseUrl" `
    VERCEL_URL="$vercelUrl" `
    WHATSAPP_ACCESS_TOKEN="$whatsappAccessToken" `
    WHATSAPP_PHONE_NUMBER_ID="$whatsappPhoneNumberId" `
    WHATSAPP_VERIFY_TOKEN="$whatsappVerifyToken" `
    WHATSAPP_BUSINESS_ACCOUNT_ID="$whatsappBusinessAccountId" `
    USE_POSTGRES="true"

# Configure CORS
Write-Host "Step 6: Configuring CORS..." -ForegroundColor Cyan
az webapp cors add `
    --resource-group $resourceGroup `
    --name $appName `
    --allowed-origins "$vercelUrl"

# Get deployment credentials
Write-Host "Step 7: Getting deployment credentials..." -ForegroundColor Cyan
$publishingProfile = az webapp deployment list-publishing-profiles `
    --resource-group $resourceGroup `
    --name $appName `
    --query "[0]" `
    --output json

Write-Host ""
Write-Host "=== Deployment Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Deploy your code using:"
Write-Host "   az webapp up --name $appName --resource-group $resourceGroup"
Write-Host ""
Write-Host "2. Or configure GitHub Actions for continuous deployment"
Write-Host ""
Write-Host "3. Your app will be available at:"
Write-Host "   https://$appName.azurewebsites.net"
