$ErrorActionPreference = "Stop"
$env:PULUMI_CONFIG_PASSPHRASE = ""

$GithubRepo = if ($env:GITHUB_REPO) { $env:GITHUB_REPO } else { "salimalsazu/modelserve" }
$Stack      = if ($env:STACK) { $env:STACK } else { "prod" }
$AwsRegion  = "ap-southeast-1"

Write-Host "--- ModelServe Infrastructure Bootstrap ---" -ForegroundColor Cyan
Write-Host "Repo  : $GithubRepo"
Write-Host "Stack : $Stack"
Write-Host "Region: $AwsRegion"
Write-Host ""

foreach ($cmd in @("aws", "pulumi", "python")) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Error "'$cmd' not found in PATH"; exit 1
    }
}
Write-Host "OK: prerequisites found" -ForegroundColor Green

$AccountId = aws sts get-caller-identity --query Account --output text
Write-Host "OK: AWS account $AccountId" -ForegroundColor Green
Write-Host ""

Set-Location "$PSScriptRoot\infrastructure"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
pip install -q --upgrade pip
pip install -q -r requirements.txt
Write-Host "OK: Python dependencies installed" -ForegroundColor Green

$stacks = pulumi stack ls --json 2>$null | ConvertFrom-Json
$exists  = $stacks | Where-Object { $_.name -eq $Stack }
if (-not $exists) { pulumi stack init $Stack }
else              { pulumi stack select $Stack }

pulumi config set aws:region  $AwsRegion
pulumi config set github_repo $GithubRepo
Write-Host "OK: Pulumi stack '$Stack' configured" -ForegroundColor Green
Write-Host ""

Write-Host "--- Pulumi preview ---" -ForegroundColor Yellow
pulumi preview

Write-Host ""
$confirm = Read-Host "Apply the above changes? [y/N]"
if ($confirm -notmatch "^[Yy]$") { Write-Host "Aborted."; exit 0 }

pulumi up --yes
Write-Host ""

$Ec2Ip    = pulumi stack output ec2_public_ip
$RoleArn  = pulumi stack output github_actions_role_arn
$EcrApi   = pulumi stack output ecr_api_url
$S3Bucket = pulumi stack output s3_bucket

Write-Host ""
Write-Host "--- Infrastructure ready ---" -ForegroundColor Green
Write-Host "EC2 IP    : $Ec2Ip"
Write-Host "ECR API   : $EcrApi"
Write-Host "S3 bucket : $S3Bucket"
Write-Host "OIDC role : $RoleArn"
Write-Host ""
Write-Host "--- NEXT STEP ---" -ForegroundColor Yellow
Write-Host "Go to: https://github.com/$GithubRepo/settings/variables/actions"
Write-Host "Add a new repository variable:"
Write-Host "  Name : AWS_ACCOUNT_ID"
Write-Host "  Value: $AccountId"
Write-Host ""
Write-Host "After that, every push to main deploys automatically." -ForegroundColor Green
