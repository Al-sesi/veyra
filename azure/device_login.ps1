$ErrorActionPreference = "Stop"

$TENANT_ID = "c611af88-d359-4ea3-a925-198ad5c6c062"
$CLIENT_ID = "04b07795-8ddb-461a-bbee-02f9e1bf7b46"  # Azure CLI official client id
$SCOPE = "https://management.azure.com/.default openid profile offline_access"

Write-Host "Requesting device-code for tenant $TENANT_ID ..." -ForegroundColor Cyan

$body = @{
    client_id = $CLIENT_ID
    scope     = $SCOPE
}
$resp = Invoke-RestMethod -Method Post `
    -Uri "https://login.microsoftonline.com/$TENANT_ID/oauth2/v2.0/devicecode" `
    -Body $body -ContentType "application/x-www-form-urlencoded"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  OPEN IN BROWSER:  $($resp.verification_uri)" -ForegroundColor White
Write-Host "  ENTER CODE:       $($resp.user_code)" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "  NOTE: Sign in with the account that owns subscription"
Write-Host "        6d947d68-b871-4787-9681-abc4ec62186c"
Write-Host ""
Write-Host "Waiting for you to complete authentication ($($resp.expires_in/60 -as [int]) min timeout)..."

$intervalSec = $resp.interval
$deadline = (Get-Date).AddSeconds($resp.expires_in)
$deviceCode = $resp.device_code

do {
    Start-Sleep -Seconds $intervalSec
    try {
        $tokenBody = @{
            grant_type  = "urn:ietf:params:oauth:grant-type:device_code"
            client_id   = $CLIENT_ID
            device_code = $deviceCode
            tenant      = $TENANT_ID
        }
        $tokenResp = Invoke-RestMethod -Method Post -ErrorAction Stop `
            -Uri "https://login.microsoftonline.com/$TENANT_ID/oauth2/v2.0/token" `
            -Body $tokenBody -ContentType "application/x-www-form-urlencoded"
    } catch {
        $err = $_.ErrorDetails.Message | ConvertFrom-Json -ErrorAction SilentlyContinue
        if ($err.error -eq "authorization_pending") {
            Write-Host "  still waiting... ($(([math]::Round(($deadline - (Get-Date)).TotalSeconds)))s left)"
            continue
        }
        if ($err.error -eq "slow_down") { Start-Sleep -Seconds 2; continue }
        if ($err.error -eq "expired_token") { Write-Host "TIMEOUT - run again." -ForegroundColor Red; exit 1 }
        Write-Host "Error: $($err.error) - $($err.error_description)" -ForegroundColor Red
        exit 1
    }
    break
} while ($true)

Write-Host ""
Write-Host "Authenticated! Access token obtained." -ForegroundColor Green

Write-Host "Listing subscriptions to confirm access..." -ForegroundColor Cyan
$subs = Invoke-RestMethod -Method Get `
    -Uri "https://management.azure.com/subscriptions?api-version=2020-01-01" `
    -Headers @{ Authorization = "Bearer $($tokenResp.access_token)" }

$subs.value | ForEach-Object {
    Write-Host "  SUBSCRIPTION: $($_.displayName)  id=$($_.subscriptionId)  state=$($_.state)"
}

if (-not $subs.value -or $subs.value.Count -eq 0) {
    Write-Host "ERROR: No subscriptions visible. The signed-in identity has no Azure RBAC assignments." -ForegroundColor Red
    exit 1
}

$target = $subs.value | Where-Object { $_.subscriptionId -eq "6d947d68-b871-4787-9681-abc4ec62186c" }
if (-not $target) { $target = $subs.value[0] }
Write-Host "Target subscription for deployment: $($target.displayName) ($($target.subscriptionId))" -ForegroundColor Green

$tokenResp | ConvertTo-Json -Depth 10 | Set-Content -Path "$PSScriptRoot\.az_token.json"
@{ tenantId = $TENANT_ID; targetSubscriptionId = $target.subscriptionId } | ConvertTo-Json | Set-Content -Path "$PSScriptRoot\.az_context.json"

Write-Host ""
Write-Host "Token saved to azure/.az_token.json and context to azure/.az_context.json" -ForegroundColor Green
