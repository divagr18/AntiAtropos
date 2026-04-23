# AntiAtropos - One-Run Teardown Script
# Deletes entire AWS infrastructure: EKS cluster, AMP workspace
#
# Usage: .\deploy\aws\teardown-all.ps1

$ErrorActionPreference = "Stop"

# In PowerShell 7+, prevent native stderr output from becoming terminating errors.
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Region = "ap-south-1"
$ClusterName = "antiatropos"
$AmpAlias = "antiatropos-metrics"
$GeneratedKubeconfig = Join-Path $PSScriptRoot "kubeconfig-antiatropos.yaml"

function Invoke-CheckedCommand {
    param(
        [ScriptBlock]$Command,
        [string]$ErrorMessage
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Command
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($LASTEXITCODE -ne 0) {
        throw $ErrorMessage
    }
}

function Get-EksClusterStatus {
    param(
        [string]$Name,
        [string]$AwsRegion
    )

    try {
        $status = aws eks describe-cluster --name $Name --region $AwsRegion --query 'cluster.status' --output text 2>$null
    } catch {
        return $null
    }

    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($status) -or $status -eq "None") {
        return $null
    }

    return $status.Trim()
}

function Get-EksNodegroups {
    param(
        [string]$Name,
        [string]$AwsRegion
    )

    try {
        $raw = aws eks list-nodegroups --cluster-name $Name --region $AwsRegion --query 'nodegroups' --output text 2>$null
    } catch {
        return @()
    }

    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($raw) -or $raw -eq "None") {
        return @()
    }

    return @($raw -split '\s+' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function Remove-ResidualEksStacks {
    param(
        [string]$Cluster,
        [string]$AwsRegion
    )

    $stackPrefix = "eksctl-$Cluster"
    $stackQuery = "StackSummaries[?starts_with(StackName, '$stackPrefix') && (StackStatus!='DELETE_COMPLETE' && StackStatus!='DELETE_IN_PROGRESS')].StackName"

    $stacksText = aws cloudformation list-stacks --region $AwsRegion --query $stackQuery --output text 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($stacksText) -or $stacksText -eq "None") {
        return
    }

    $stacks = @($stacksText -split '\s+' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    foreach ($stack in $stacks) {
        Write-Host "Deleting residual stack: $stack" -ForegroundColor Yellow
        Invoke-CheckedCommand -Command { aws cloudformation delete-stack --stack-name $stack --region $AwsRegion 2>$null | Out-Null } -ErrorMessage "Failed to delete stack '$stack'"
        Invoke-CheckedCommand -Command { aws cloudformation wait stack-delete-complete --stack-name $stack --region $AwsRegion } -ErrorMessage "Timed out deleting stack '$stack'"
    }
}

function Get-AmpWorkspaceIdByAlias {
    param(
        [string]$Alias,
        [string]$AwsRegion
    )

    try {
        $id = aws amp list-workspaces --alias $Alias --region $AwsRegion --query 'workspaces[0].workspaceId' --output text 2>$null
    } catch {
        return $null
    }

    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($id) -or $id -eq "None") {
        return $null
    }

    return $id.Trim()
}

function Wait-AmpWorkspaceDeleted {
    param(
        [string]$WorkspaceId,
        [string]$AwsRegion
    )

    for ($i = 0; $i -lt 30; $i++) {
        try {
            $status = aws amp describe-workspace --workspace-id $WorkspaceId --region $AwsRegion --query 'workspace.status.statusCode' --output text 2>$null
        } catch {
            return
        }

        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($status) -or $status -eq "None") {
            return
        }
        Start-Sleep -Seconds 10
    }

    throw "AMP workspace '$WorkspaceId' deletion timed out"
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Red
Write-Host "   AntiAtropos AWS Infrastructure Teardown" -ForegroundColor Red
Write-Host "==========================================" -ForegroundColor Red
Write-Host "Region:      $Region"
Write-Host "Cluster:     $ClusterName"
Write-Host ""

# --- Step 1: Delete EKS Cluster ---
Write-Host ">>> Step 1: Deleting EKS cluster..." -ForegroundColor Yellow

$clusterStatus = Get-EksClusterStatus -Name $ClusterName -AwsRegion $Region
if ($clusterStatus) {
    Write-Host "Cluster status: $clusterStatus" -ForegroundColor Yellow

    if ($clusterStatus -ne "DELETING") {
        $nodegroups = Get-EksNodegroups -Name $ClusterName -AwsRegion $Region
        foreach ($ng in $nodegroups) {
            Write-Host "Deleting nodegroup: $ng" -ForegroundColor Yellow
            $ngStatus = aws eks describe-nodegroup --cluster-name $ClusterName --nodegroup-name $ng --region $Region --query 'nodegroup.status' --output text 2>$null
            if ($LASTEXITCODE -eq 0 -and $ngStatus -ne "DELETING") {
                Invoke-CheckedCommand -Command { aws eks delete-nodegroup --cluster-name $ClusterName --nodegroup-name $ng --region $Region --output text 2>$null | Out-Null } -ErrorMessage "Failed to start deletion for nodegroup '$ng'"
            } else {
                Write-Host "Nodegroup '$ng' already deleting" -ForegroundColor Yellow
            }

            Write-Host "Waiting for nodegroup deletion: $ng" -ForegroundColor Yellow
            Invoke-CheckedCommand -Command { aws eks wait nodegroup-deleted --cluster-name $ClusterName --nodegroup-name $ng --region $Region } -ErrorMessage "Timed out waiting for nodegroup '$ng' deletion"
            Write-Host "OK: Nodegroup deleted: $ng" -ForegroundColor Green
        }

        Write-Host "Deleting cluster control plane..." -ForegroundColor Yellow
        Invoke-CheckedCommand -Command { eksctl delete cluster --name $ClusterName --region $Region 2>$null | Out-Null } -ErrorMessage "Failed to delete EKS cluster"
    } else {
        Write-Host "Cluster is already deleting" -ForegroundColor Yellow
    }

    Write-Host "Waiting for cluster deletion..." -ForegroundColor Yellow
    Invoke-CheckedCommand -Command { aws eks wait cluster-deleted --name $ClusterName --region $Region } -ErrorMessage "Timed out waiting for EKS cluster deletion"
    Write-Host "OK: Cluster deleted" -ForegroundColor Green
} else {
    Write-Host "OK: Cluster not found, skipping" -ForegroundColor Green
}

Write-Host "Checking for residual eksctl stacks..." -ForegroundColor Yellow
Remove-ResidualEksStacks -Cluster $ClusterName -AwsRegion $Region
Write-Host "OK: Residual EKS stacks cleaned" -ForegroundColor Green

# --- Step 2: Delete AMP Workspace ---
Write-Host ""
Write-Host ">>> Step 2: Deleting AMP workspace..." -ForegroundColor Yellow

$AmpWsId = Get-AmpWorkspaceIdByAlias -Alias $AmpAlias -AwsRegion $Region

if (-not [string]::IsNullOrWhiteSpace($AmpWsId)) {
    Invoke-CheckedCommand -Command { aws amp delete-workspace --workspace-id $AmpWsId --region $Region | Out-Null } -ErrorMessage "Failed to delete AMP workspace '$AmpWsId'"
    Wait-AmpWorkspaceDeleted -WorkspaceId $AmpWsId -AwsRegion $Region
    Write-Host "OK: AMP workspace deleted: $AmpWsId" -ForegroundColor Green
} else {
    Write-Host "OK: AMP workspace not found, skipping" -ForegroundColor Green
}

# --- Step 3: Local kubeconfig cleanup ---
Write-Host ""
Write-Host ">>> Step 3: Cleaning local kubeconfig entries..." -ForegroundColor Yellow

try { kubectl config delete-context $ClusterName 2>$null | Out-Null } catch {}
try { kubectl config delete-cluster $ClusterName 2>$null | Out-Null } catch {}
try { kubectl config delete-user antiatropos-hf-user 2>$null | Out-Null } catch {}

if (Test-Path $GeneratedKubeconfig) {
    Remove-Item $GeneratedKubeconfig -Force
    Write-Host "OK: Removed generated kubeconfig file" -ForegroundColor Green
} else {
    Write-Host "OK: Generated kubeconfig file not found, skipping" -ForegroundColor Green
}

# --- Step 4: Verify Cleanup ---
Write-Host ""
Write-Host ">>> Step 4: Verifying cleanup..." -ForegroundColor Yellow

$clusterStillExists = [bool](Get-EksClusterStatus -Name $ClusterName -AwsRegion $Region)

if ($clusterStillExists) {
    Write-Host "WARN: Cluster still exists (deletion in progress)" -ForegroundColor Yellow
} else {
    Write-Host "OK: Cluster deleted" -ForegroundColor Green
}

$ampStillExists = -not [string]::IsNullOrWhiteSpace((Get-AmpWorkspaceIdByAlias -Alias $AmpAlias -AwsRegion $Region))

if ($ampStillExists) {
    Write-Host "WARN: AMP workspace alias '$AmpAlias' still exists" -ForegroundColor Yellow
} else {
    Write-Host "OK: AMP workspace deleted" -ForegroundColor Green
}

# --- Done ---
Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "   Teardown Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "All AWS infrastructure has been removed." -ForegroundColor Yellow
Write-Host ""
