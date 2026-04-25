docker stop antiatropos-grafana 2>$null
docker rm antiatropos-grafana 2>$null

Write-Host "Starting local Grafana (datasource -> host.docker.internal:9090)..."

docker run -d --name antiatropos-grafana -p 3000:3000 `
  -v "$PWD\deploy\grafana\provisioning:/etc/grafana/provisioning:ro" `
  -e GF_AUTH_ANONYMOUS_ENABLED=true `
  -e GF_AUTH_ANONYMOUS_ORG_ROLE=Admin `
  grafana/grafana:latest | Out-Null

Write-Host "Grafana is running at http://localhost:3000"
