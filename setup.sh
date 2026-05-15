#!/usr/bin/env bash
# One-time infrastructure bootstrap.
# Run this ONCE to create all AWS resources and the GitHub OIDC role.
# After this, every git push to main deploys automatically.
#
# Prerequisites:
#   - AWS CLI configured (aws configure) with admin credentials
#   - Pulumi CLI installed (https://www.pulumi.com/docs/install/)
#   - Python 3.10+

set -euo pipefail

GITHUB_REPO="${GITHUB_REPO:-salimalsazu/modelserve}"
STACK="${STACK:-prod}"
AWS_REGION="${AWS_REGION:-ap-southeast-1}"

echo "========================================"
echo " ModelServe Infrastructure Bootstrap"
echo " Repo  : $GITHUB_REPO"
echo " Stack : $STACK"
echo " Region: $AWS_REGION"
echo "========================================"
echo ""

# ── 1. Verify prerequisites ───────────────────────────────────────────────

command -v aws    >/dev/null || { echo "ERROR: aws CLI not found"; exit 1; }
command -v pulumi >/dev/null || { echo "ERROR: pulumi CLI not found. Install: https://www.pulumi.com/docs/install/"; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 not found"; exit 1; }

echo "✓ Prerequisites OK"

# Confirm AWS identity
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "✓ AWS account: $AWS_ACCOUNT_ID (region: $AWS_REGION)"
echo ""

# ── 2. Python virtualenv + Pulumi providers ───────────────────────────────

cd "$(dirname "$0")/infrastructure"

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Python dependencies installed"

# ── 3. Pulumi stack ───────────────────────────────────────────────────────

pulumi stack select "$STACK" 2>/dev/null || pulumi stack init "$STACK"
pulumi config set aws:region "$AWS_REGION"
pulumi config set github_repo "$GITHUB_REPO"
echo "✓ Pulumi stack '$STACK' ready"
echo ""

# ── 4. Preview first, then apply ─────────────────────────────────────────

echo "--- Pulumi preview ---"
pulumi preview

echo ""
read -rp "Apply the above changes? [y/N] " CONFIRM
[[ "$CONFIRM" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

pulumi up --yes
echo ""

# ── 5. Capture outputs ───────────────────────────────────────────────────

EC2_IP=$(pulumi stack output ec2_public_ip)
ROLE_ARN=$(pulumi stack output github_actions_role_arn)
ECR_API=$(pulumi stack output ecr_api_url)
S3_BUCKET=$(pulumi stack output s3_bucket)

echo "========================================"
echo " Infrastructure ready!"
echo "========================================"
echo " EC2 public IP  : $EC2_IP"
echo " ECR API repo   : $ECR_API"
echo " S3 bucket      : $S3_BUCKET"
echo " GitHub role ARN: $ROLE_ARN"
echo ""
echo "========================================"
echo " NEXT STEP — set one GitHub variable:"
echo "========================================"
echo ""
echo " Go to: https://github.com/$GITHUB_REPO/settings/variables/actions"
echo " Click 'New repository variable' and add:"
echo ""
echo "   Name : AWS_ACCOUNT_ID"
echo "   Value: $AWS_ACCOUNT_ID"
echo ""
echo " After that, every push to main deploys automatically."
echo "========================================"
