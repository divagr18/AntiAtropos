# AntiAtropos - One-Run Deploy Script
# Deploys entire AWS infrastructure: EKS cluster, workloads, AMP, Prometheus, Grafana

$ErrorActionPreference = "Stop"

# In PowerShell 7+, prevent native stderr from becoming terminating errors.
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Region = "ap-south-1"
$ClusterName = "antiatropos"
$AwsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$GrafanaMode = if ([string]::IsNullOrWhiteSpace($env:ANTIATROPOS_GRAFANA_MODE)) { "auto" } else { $env:ANTIATROPOS_GRAFANA_MODE.Trim().ToLowerInvariant() }
$GrafanaModeResolved = "cluster"

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

function Test-EksNodegroupExists {
    param(
        [string]$Cluster,
        [string]$Nodegroup,
        [string]$AwsRegion
    )

    try {
        aws eks describe-nodegroup --cluster-name $Cluster --nodegroup-name $Nodegroup --region $AwsRegion --query 'nodegroup.nodegroupName' --output text 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Get-EksNodegroupInstanceType {
    param(
        [string]$Cluster,
        [string]$Nodegroup,
        [string]$AwsRegion
    )

    try {
        $instanceType = aws eks describe-nodegroup --cluster-name $Cluster --nodegroup-name $Nodegroup --region $AwsRegion --query 'nodegroup.instanceTypes[0]' --output text 2>$null
    } catch {
        return $null
    }

    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($instanceType) -or $instanceType -eq "None") {
        return $null
    }

    return $instanceType.Trim()
}

function Get-NodegroupSubnetSelection {
    param(
        [string]$Cluster,
        [string]$AwsRegion
    )

    try {
        $allSubnetIds = aws eks describe-cluster --name $Cluster --region $AwsRegion --query 'cluster.resourcesVpcConfig.subnetIds' --output text 2>$null
    } catch {
        throw "Failed to read cluster subnet IDs"
    }

    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($allSubnetIds)) {
        throw "Failed to read cluster subnet IDs"
    }

    $subnetArray = @($allSubnetIds -split '\s+' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($subnetArray.Count -eq 0) {
        throw "No subnets found for cluster '$Cluster' in region '$AwsRegion'"
    }

    $describeSubnetArgs = @(
        'ec2', 'describe-subnets',
        '--region', $AwsRegion,
        '--subnet-ids'
    ) + $subnetArray + @(
        '--query', 'Subnets[?MapPublicIpOnLaunch==true].SubnetId',
        '--output', 'text'
    )

    try {
        $publicSubnetIdsText = & aws @describeSubnetArgs 2>$null
    } catch {
        throw "Failed to classify cluster subnets"
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to classify cluster subnets"
    }

    $publicSubnetIds = @($publicSubnetIdsText -split '\s+' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and $_ -ne "None" })
    $privateSubnetIds = @($subnetArray | Where-Object { $publicSubnetIds -notcontains $_ })

    if ($publicSubnetIds.Count -gt 0) {
        return [PSCustomObject]@{
            SubnetCsv = ($publicSubnetIds -join ',')
            UsePrivateNetworking = $false
            SubnetType = "public"
        }
    }

    if ($privateSubnetIds.Count -gt 0) {
        return [PSCustomObject]@{
            SubnetCsv = ($privateSubnetIds -join ',')
            UsePrivateNetworking = $true
            SubnetType = "private"
        }
    }

    throw "Could not determine valid subnets for nodegroup creation"
}

function Get-ReadyNodeCount {
    $nodeLines = kubectl get nodes --no-headers 2>$null
    if (-not $nodeLines) {
        return 0
    }

    return (@($nodeLines | Select-String -Pattern '\sReady\s').Count)
}

function Wait-ForReadyNodes {
    param(
        [int]$MinimumReadyNodes,
        [int]$TimeoutSeconds = 600
    )

    $attempts = [Math]::Ceiling($TimeoutSeconds / 10)
    for ($i = 0; $i -lt $attempts; $i++) {
        $readyCount = Get-ReadyNodeCount
        Write-Host "Nodes ready: $readyCount (target: $MinimumReadyNodes)"
        if ($readyCount -ge $MinimumReadyNodes) {
            return
        }
        Start-Sleep -Seconds 10
    }

    throw "Timed out waiting for $MinimumReadyNodes Ready nodes"
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   AntiAtropos AWS Infrastructure Deploy" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Region:      $Region"
Write-Host "Cluster:     $ClusterName"
Write-Host ""

# Check prerequisites
$missing = @()
foreach ($cmd in @("aws", "eksctl", "kubectl", "helm")) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        $missing += $cmd
    }
}
if ($missing.Count -gt 0) {
    Write-Host "ERROR: Missing: $($missing -join ', ')" -ForegroundColor Red
    exit 1
}

# Phase 1: Create EKS Cluster
Write-Host ">>> Phase 1: Creating EKS cluster..." -ForegroundColor Yellow

$clusterStatus = Get-EksClusterStatus -Name $ClusterName -AwsRegion $Region

if ($clusterStatus -eq "DELETING") {
    Write-Host "Cluster is currently deleting. Waiting for deletion to complete..." -ForegroundColor Yellow
    Invoke-CheckedCommand -Command { aws eks wait cluster-deleted --name $ClusterName --region $Region } -ErrorMessage "Failed while waiting for cluster deletion"
    $clusterStatus = $null
}

if (-not $clusterStatus) {
    $TempConfig = Join-Path $AwsDir "eksctl-cluster-only.yaml"
    $ClusterYaml = Get-Content (Join-Path $AwsDir "eksctl-cluster.yaml") -Raw
    $ClusterOnlyYaml = $ClusterYaml -replace '(?s)(managedNodeGroups:.*)', ''
    $ClusterOnlyYaml | Out-File -FilePath $TempConfig -Encoding utf8
    Invoke-CheckedCommand -Command { eksctl create cluster -f $TempConfig } -ErrorMessage "Failed to create EKS cluster"
    Remove-Item $TempConfig -Force
    Write-Host "Cluster created" -ForegroundColor Green
} else {
    if ($clusterStatus -eq "CREATING") {
        Write-Host "Cluster creation in progress. Waiting until ACTIVE..." -ForegroundColor Yellow
        Invoke-CheckedCommand -Command { aws eks wait cluster-active --name $ClusterName --region $Region } -ErrorMessage "Cluster did not become active"
    }
    Write-Host "Cluster already exists (status: $clusterStatus)" -ForegroundColor Green
}

Invoke-CheckedCommand -Command { aws eks wait cluster-active --name $ClusterName --region $Region } -ErrorMessage "Cluster is not active"
Invoke-CheckedCommand -Command { aws eks update-kubeconfig --name $ClusterName --region $Region | Out-Null } -ErrorMessage "Failed to update kubeconfig"

# Phase 2: Create Nodegroup
Write-Host ""
Write-Host ">>> Phase 2: Ensuring compute nodegroup..." -ForegroundColor Yellow

$NodegroupName = "linux-nodes"
$PreferredInstanceType = "t3.micro"
$ngExists = Test-EksNodegroupExists -Cluster $ClusterName -Nodegroup $NodegroupName -AwsRegion $Region

if (-not $ngExists) {
    $SubnetSelection = Get-NodegroupSubnetSelection -Cluster $ClusterName -AwsRegion $Region
    $SubnetCsv = $SubnetSelection.SubnetCsv
    $UsePrivateNetworking = [bool]$SubnetSelection.UsePrivateNetworking
    
    Write-Host "Using $($SubnetSelection.SubnetType) subnets: $SubnetCsv"
    
    Invoke-CheckedCommand -Command {
        $args = @(
            'create', 'nodegroup',
            '--cluster', $ClusterName,
            '--region', $Region,
            '--name', $NodegroupName,
            '--node-type', $PreferredInstanceType,
            '--nodes', '4',
            '--nodes-min', '2',
            '--nodes-max', '8',
            '--node-volume-size', '20',
            '--subnet-ids', $SubnetCsv
        )

        if ($UsePrivateNetworking) {
            $args += '--node-private-networking'
        }

        eksctl @args
    } -ErrorMessage "Failed to create nodegroup '$NodegroupName'"

    Write-Host "Nodegroup created" -ForegroundColor Green
} else {
    $existingInstanceType = Get-EksNodegroupInstanceType -Cluster $ClusterName -Nodegroup $NodegroupName -AwsRegion $Region
    Write-Host "Nodegroup already exists ($existingInstanceType)" -ForegroundColor Green
}

Invoke-CheckedCommand -Command { aws eks wait nodegroup-active --cluster-name $ClusterName --nodegroup-name $NodegroupName --region $Region } -ErrorMessage "Nodegroup did not become active"

if ($GrafanaMode -in @("auto", "")) {
    $effectiveNodeType = Get-EksNodegroupInstanceType -Cluster $ClusterName -Nodegroup $NodegroupName -AwsRegion $Region
    if ($effectiveNodeType -eq "t3.micro") {
        $GrafanaModeResolved = "external"
    } else {
        $GrafanaModeResolved = "cluster"
    }
} elseif ($GrafanaMode -in @("external", "local", "hf")) {
    $GrafanaModeResolved = "external"
} else {
    $GrafanaModeResolved = "cluster"
}

Write-Host "Grafana mode: $GrafanaModeResolved" -ForegroundColor Cyan

Write-Host "Waiting for nodes..."
for ($i = 0; $i -lt 60; $i++) {
    $nodes = $null
    try {
        $nodes = kubectl get nodes --no-headers --request-timeout=10s 2>$null
    } catch {
        Start-Sleep -Seconds 10
        continue
    }

    if ($nodes) {
        $readyCount = ($nodes | Select-String -Pattern '\sReady\s').Count
        Write-Host "Nodes ready: $readyCount" -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds 10
}

# Phase 3: Deploy Workloads
Write-Host ""
Write-Host ">>> Phase 3: Deploying workloads..." -ForegroundColor Yellow
kubectl create namespace prod-sre --dry-run=client -o yaml | kubectl apply -f - | Out-Null
kubectl apply -f (Join-Path $AwsDir "k8s-workloads.yaml") | Out-Null
Write-Host "Workloads deployed" -ForegroundColor Green

# Phase 4: Create AMP Workspace
Write-Host ""
Write-Host ">>> Phase 4: Creating AMP workspace..." -ForegroundColor Yellow

$AmpWsId = $null
try {
    $AmpWsId = aws amp list-workspaces --alias antiatropos-metrics --region $Region --query 'workspaces[0].workspaceId' --output text 2>$null
    if ($AmpWsId -eq "None") { $AmpWsId = $null }
} catch {}

if ([string]::IsNullOrWhiteSpace($AmpWsId)) {
    $AmpWsId = aws amp create-workspace --alias antiatropos-metrics --region $Region --query 'workspaceId' --output text
}
$AmpUrl = "https://aps-workspaces.$Region.amazonaws.com/workspaces/$AmpWsId"
Write-Host "AMP: $AmpWsId" -ForegroundColor Green

# Phase 5: Install Prometheus
Write-Host ""
Write-Host ">>> Phase 5: Installing Prometheus..." -ForegroundColor Yellow

kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f - | Out-Null
Invoke-CheckedCommand -Command { helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>$null | Out-Null } -ErrorMessage "Failed to add prometheus helm repo"
Invoke-CheckedCommand -Command { helm repo update 2>$null | Out-Null } -ErrorMessage "Failed to update helm repos"

$promValuesYaml = Join-Path $AwsDir "prometheus-agent-values.yaml"
$remoteWriteUrl = "$AmpUrl/api/v1/remote_write"

Invoke-CheckedCommand -Command {
    helm upgrade --install prometheus-agent prometheus-community/prometheus --namespace monitoring --reset-values -f $promValuesYaml `
        --set "alertmanager.enabled=false" `
        --set "kube-state-metrics.enabled=false" `
        --set "prometheus-node-exporter.enabled=false" `
        --set "pushgateway.enabled=false" `
        --set "server.enabled=true" `
        --set "server.persistentVolume.enabled=false" `
        --set "server.resources.requests.cpu=50m" `
        --set "server.resources.requests.memory=128Mi" `
        --set "server.resources.limits.cpu=300m" `
        --set "server.resources.limits.memory=384Mi" `
        --set "server.global.scrape_interval=15s" `
        --set "server.remoteWrite[0].url=$remoteWriteUrl" `
        2>&1 | Out-Null
} -ErrorMessage "Failed to install/upgrade Prometheus"
Write-Host "Prometheus installed" -ForegroundColor Green

# Phase 6: Install Grafana
Write-Host ""
if ($GrafanaModeResolved -eq "cluster") {
    Write-Host ">>> Phase 6: Installing Grafana in-cluster..." -ForegroundColor Yellow

    Invoke-CheckedCommand -Command { helm repo add grafana https://grafana.github.io/helm-charts 2>$null | Out-Null } -ErrorMessage "Failed to add grafana helm repo"
    Invoke-CheckedCommand -Command { helm repo update 2>$null | Out-Null } -ErrorMessage "Failed to update helm repos"

    $GrafanaValuesYaml = Join-Path $AwsDir "grafana-values.yaml"
    Invoke-CheckedCommand -Command { helm upgrade --install grafana grafana/grafana --namespace monitoring -f $GrafanaValuesYaml 2>&1 | Out-Null } -ErrorMessage "Failed to install/upgrade Grafana"

    Write-Host "Waiting for Grafana..."
    try {
        Invoke-CheckedCommand -Command { kubectl rollout status deployment/grafana --namespace monitoring --timeout=120s 2>$null | Out-Null } -ErrorMessage "Grafana rollout timed out"
    } catch {
        $pendingGrafanaPod = kubectl get pods -n monitoring -l app.kubernetes.io/name=grafana --field-selector=status.phase=Pending --no-headers 2>$null | Select-Object -First 1
        $pendingReason = ""

        if ($pendingGrafanaPod) {
            $pendingGrafanaPodName = ($pendingGrafanaPod -split '\s+')[0]
            $pendingReason = kubectl describe pod $pendingGrafanaPodName -n monitoring 2>$null | Select-String -Pattern "FailedScheduling|Insufficient memory|Too many pods|unbound" -Context 0,2 | Out-String
            if (-not [string]::IsNullOrWhiteSpace($pendingReason)) {
                Write-Host "Grafana is pending due to scheduler constraints:" -ForegroundColor Yellow
                Write-Host $pendingReason -ForegroundColor Yellow
            }
        }

        $shouldScale = $pendingReason -match "Too many pods|Insufficient memory"
        if ($shouldScale) {
            Write-Host "Scaling nodegroup to 8 nodes and retrying Grafana rollout..." -ForegroundColor Yellow
            Invoke-CheckedCommand -Command { eksctl scale nodegroup --cluster $ClusterName --region $Region --name $NodegroupName --nodes 8 } -ErrorMessage "Failed to scale nodegroup"
            Invoke-CheckedCommand -Command { aws eks wait nodegroup-active --cluster-name $ClusterName --nodegroup-name $NodegroupName --region $Region } -ErrorMessage "Nodegroup did not become active after scaling"
            Write-Host "Waiting for newly scaled nodes to become Ready..." -ForegroundColor Yellow
            Wait-ForReadyNodes -MinimumReadyNodes 8 -TimeoutSeconds 900

            $pendingGrafanaPodAfterScale = kubectl get pods -n monitoring -l app.kubernetes.io/name=grafana --field-selector=status.phase=Pending --no-headers 2>$null | Select-Object -First 1
            if ($pendingGrafanaPodAfterScale) {
                $pendingGrafanaPodNameAfterScale = ($pendingGrafanaPodAfterScale -split '\s+')[0]
                kubectl delete pod $pendingGrafanaPodNameAfterScale -n monitoring 2>$null | Out-Null
            }

            Invoke-CheckedCommand -Command { kubectl rollout status deployment/grafana --namespace monitoring --timeout=600s 2>$null | Out-Null } -ErrorMessage "Grafana rollout timed out after scaling"
        } else {
            throw "Grafana rollout failed. Check: kubectl -n monitoring get pods ; kubectl -n monitoring describe pod -l app.kubernetes.io/name=grafana"
        }
    }
    Write-Host "Grafana installed (admin/antiatropos)" -ForegroundColor Green
} else {
    Write-Host ">>> Phase 6: Skipping in-cluster Grafana (external mode)..." -ForegroundColor Yellow
    $grafanaRelease = ""
    try {
        $grafanaRelease = helm list -n monitoring --filter '^grafana$' --short 2>$null
    } catch {
        $grafanaRelease = ""
    }

    if (-not [string]::IsNullOrWhiteSpace($grafanaRelease)) {
        helm uninstall grafana -n monitoring 2>$null | Out-Null
        kubectl delete pvc grafana -n monitoring 2>$null | Out-Null
        Write-Host "Removed existing in-cluster Grafana release to save resources" -ForegroundColor Green
    }
}

# Phase 7: Install Cluster Autoscaler
Write-Host ""
Write-Host ">>> Phase 7: Installing Cluster Autoscaler..." -ForegroundColor Yellow

Invoke-CheckedCommand -Command { helm repo add autoscaler https://kubernetes.github.io/autoscaler 2>$null | Out-Null } -ErrorMessage "Failed to add autoscaler helm repo"
Invoke-CheckedCommand -Command { helm repo update 2>$null | Out-Null } -ErrorMessage "Failed to update helm repos"

$autoscalerValues = Join-Path $AwsDir "cluster-autoscaler-values.yaml"
Invoke-CheckedCommand -Command { helm upgrade --install cluster-autoscaler autoscaler/cluster-autoscaler --namespace kube-system -f $autoscalerValues 2>&1 | Out-Null } -ErrorMessage "Failed to install/upgrade Cluster Autoscaler"
Write-Host "Cluster Autoscaler installed" -ForegroundColor Green

# Phase 8: Generate Kubeconfig
Write-Host ""
Write-Host ">>> Phase 8: Generating kubeconfig..." -ForegroundColor Yellow

$ClusterEndpoint = aws eks describe-cluster --name $ClusterName --region $Region --query 'cluster.endpoint' --output text
$ClusterCa = aws eks describe-cluster --name $ClusterName --region $Region --query 'cluster.certificateAuthority.data' --output text
$Timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$output = Join-Path $AwsDir "kubeconfig-antiatropos.yaml"

$kubeconfig = "apiVersion: v1`n" +
"kind: Config`n" +
"clusters:`n" +
"  - cluster:`n" +
"      certificate-authority-data: $ClusterCa`n" +
"      server: $ClusterEndpoint`n" +
"    name: $ClusterName`n" +
"contexts:`n" +
"  - context:`n" +
"      cluster: $ClusterName`n" +
"      user: antiatropos-hf-user`n" +
"    name: $ClusterName`n" +
"current-context: $ClusterName`n" +
"preferences: {}`n" +
"users:`n" +
"  - name: antiatropos-hf-user`n" +
"    user:`n" +
"      exec:`n" +
"        apiVersion: client.authentication.k8s.io/v1beta1`n" +
"        command: aws`n" +
"        args:`n" +
"          - eks`n" +
"          - get-token`n" +
"          - --region`n" +
"          - $Region`n" +
"          - --cluster-name`n" +
"          - $ClusterName`n" +
"        env:`n" +
"          - name: AWS_STS_REGIONAL_ENDPOINTS`n" +
"            value: regional`n" +
"          - name: AWS_DEFAULT_REGION`n" +
"            value: $Region`n" +
"        interactiveMode: IfAvailable`n"

$kubeconfig | Out-File -FilePath $output -Encoding utf8 -Force
Write-Host "Kubeconfig: $output" -ForegroundColor Green

# Done
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   Deployment Complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "AMP: $AmpWsId" -ForegroundColor Yellow
if ($GrafanaModeResolved -eq "cluster") {
    Write-Host "Grafana: kubectl port-forward svc/grafana 3000 -n monitoring" -ForegroundColor Yellow
    Write-Host "Login: admin / antiatropos" -ForegroundColor Yellow
} else {
    Write-Host "Grafana: external/local mode enabled (recommended for free-tier nodes)" -ForegroundColor Yellow
    Write-Host "Use AMP endpoint as Prometheus datasource with SigV4 auth" -ForegroundColor Yellow
}
Write-Host "Kubeconfig: $output" -ForegroundColor Yellow
Write-Host ""
