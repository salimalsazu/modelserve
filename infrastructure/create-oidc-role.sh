#!/bin/bash
# Create IAM role for GitHub Actions OIDC

# Set variables
AWS_ACCOUNT_ID="679835924810"
AWS_REGION="ap-southeast-1"
GITHUB_REPO="salimalsazu/modelserve"

# Create OIDC provider
aws iam create-open-id-connect-provider \
  --url "https://token.actions.githubusercontent.com" \
  --client-id-list "sts.amazonaws.com" \
  --thumbprint-list "6938fd4d98bab03faadb97bfa968ef4f9edc5f11" 2>/dev/null || echo "OIDC provider may exist"

# Create IAM role for GitHub Actions
cat > /tmp/github-actions-role-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::679835924810:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:salimalsazu/modelserve:*"
        }
      }
    }
  ]
}
EOF

aws iam create-role \
  --role-name github-actions-role \
  --assume-role-policy-document file:///tmp/github-actions-role-trust-policy.json 2>/dev/null || \
  echo "Role may already exist, updating..."

# Create policy for GitHub Actions
cat > /tmp/github-actions-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus",
        "ec2:DescribeInstanceAttribute",
        "ec2-instance-connect:SendSSHPublicKey"
      ],
      "Resource": "*"
    }
  ]
}
EOF

aws iam put-role-policy \
  --role-name github-actions-role \
  --policy-name github-actions-policy \
  --policy-document file:///tmp/github-actions-policy.json 2>/dev/null || \
  echo "Policy may already exist"

echo "Role created: arn:aws:iam::${AWS_ACCOUNT_ID}:role/github-actions-role"