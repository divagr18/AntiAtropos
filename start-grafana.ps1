docker stop antiatropos-grafana 2>$null
docker rm antiatropos-grafana 2>$null

Write-Host "Fetching AWS credentials..."
$AccessKey = (aws configure get aws_access_key_id).Trim()
$SecretKey = (aws configure get aws_secret_access_key).Trim()
$SessionToken = aws configure get aws_session_token
if ($SessionToken) { $SessionToken = $SessionToken.Trim() }

Write-Host "Starting Grafana with injected AWS credentials..."

$DockerCmd = "docker run -d --name antiatropos-grafana -p 3000:3000 " + `
  "-v ""$PWD\deploy\grafana\provisioning:/etc/grafana/provisioning:ro"" " + `
  "-e GF_AUTH_ANONYMOUS_ENABLED=true " + `
  "-e GF_AUTH_ANONYMOUS_ORG_ROLE=Admin " + `
  "-e GF_AUTH_SIGV4_AUTH_ENABLED=true " + `
  "-e AWS_ACCESS_KEY_ID=""$AccessKey"" " + `
  "-e AWS_SECRET_ACCESS_KEY=""$SecretKey"" " + `
  "-e AWS_REGION=ap-south-1 "

if ($SessionToken) {
    $DockerCmd += "-e AWS_SESSION_TOKEN=""$SessionToken"" "
}

$DockerCmd += "grafana/grafana:latest"

Invoke-Expression $DockerCmd

Write-Host "Grafana is running! Open http://localhost:3000 in your browser."
