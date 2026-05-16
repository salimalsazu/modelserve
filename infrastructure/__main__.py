"""ModelServe AWS infrastructure — ap-southeast-1 (Singapore)

Resources:
  - GitHub Actions OIDC provider + IAM role (no static credentials needed)
  - VPC, subnet, internet gateway, route table
  - Security group for all ModelServe ports
  - IAM instance role (EC2 -> ECR read + S3)
  - ECR repositories (API + MLflow) with lifecycle policy
  - S3 bucket for MLflow artifacts (private)
  - EC2 instance (t3.medium, Amazon Linux 2023) with Elastic IP
"""

import json
import pulumi
import pulumi_aws as aws

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

cfg = pulumi.Config()
env = cfg.get("environment") or "prod"
github_repo = cfg.get("github_repo") or "salimalsazu/modelserve"

region = "ap-southeast-1"
account_id = aws.get_caller_identity().account_id

tags = {"Project": "ModelServe", "Environment": env, "ManagedBy": "Pulumi"}

# ---------------------------------------------------------------------------
# GitHub Actions OIDC
# ---------------------------------------------------------------------------

# OIDC provider already exists in AWS — look it up instead of managing it
_existing_oidc = aws.iam.get_open_id_connect_provider(
    url="https://token.actions.githubusercontent.com"
)

class _OidcRef:
    """Thin wrapper so the rest of the code can use .arn like a real resource."""
    arn = pulumi.Output.from_input(_existing_oidc.arn)

oidc_provider = _OidcRef()

github_actions_role = aws.iam.Role(
    "github-actions-role",
    name=f"modelserve-github-actions-{env}",
    assume_role_policy=oidc_provider.arn.apply(
        lambda arn: json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Federated": arn},
                "Action": "sts:AssumeRoleWithWebIdentity",
                "Condition": {
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
                    },
                    "StringLike": {
                        "token.actions.githubusercontent.com:sub": f"repo:{github_repo}:*"
                    },
                },
            }],
        })
    ),
    tags=tags,
)

_artifacts_bucket_name = f"modelserve-mlflow-artifacts-{env}-{account_id}"

github_actions_policy = aws.iam.Policy(
    "github-actions-policy",
    name=f"modelserve-github-actions-policy-{env}",
    policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ECRAuth",
                "Effect": "Allow",
                "Action": "ecr:GetAuthorizationToken",
                "Resource": "*",
            },
            {
                "Sid": "ECRPush",
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:CompleteLayerUpload",
                    "ecr:InitiateLayerUpload",
                    "ecr:PutImage",
                    "ecr:UploadLayerPart",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                ],
                "Resource": f"arn:aws:ecr:{region}:{account_id}:repository/modelserve*",
            },
            {
                "Sid": "EC2Describe",
                "Effect": "Allow",
                "Action": [
                    "ec2:DescribeInstances",
                    "ec2:RebootInstances",
                ],
                "Resource": "*",
            },
            {
                "Sid": "SSMRunCommand",
                "Effect": "Allow",
                "Action": [
                    "ssm:SendCommand",
                    "ssm:GetCommandInvocation",
                    "ssm:ListCommandInvocations",
                    "ssm:DescribeInstanceInformation",
                ],
                "Resource": [
                    f"arn:aws:ssm:{region}::document/AWS-RunShellScript",
                    f"arn:aws:ec2:{region}:{account_id}:instance/*",
                    f"arn:aws:ssm:{region}:{account_id}:*",
                    "*",
                ],
            },
            {
                "Sid": "S3DeployList",
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": f"arn:aws:s3:::{_artifacts_bucket_name}",
                "Condition": {"StringLike": {"s3:prefix": "deploy/*"}},
            },
            {
                "Sid": "S3DeployUpload",
                "Effect": "Allow",
                "Action": ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
                "Resource": f"arn:aws:s3:::{_artifacts_bucket_name}/deploy/*",
            },
        ],
    }),
    tags=tags,
)

aws.iam.RolePolicyAttachment(
    "github-actions-policy-attach",
    role=github_actions_role.name,
    policy_arn=github_actions_policy.arn,
)

# ---------------------------------------------------------------------------
# VPC & Networking
# ---------------------------------------------------------------------------

vpc = aws.ec2.Vpc(
    "vpc",
    cidr_block="10.0.0.0/16",
    enable_dns_hostnames=True,
    enable_dns_support=True,
    tags={**tags, "Name": f"modelserve-vpc-{env}"},
)

igw = aws.ec2.InternetGateway(
    "igw",
    vpc_id=vpc.id,
    tags={**tags, "Name": f"modelserve-igw-{env}"},
)

public_subnet = aws.ec2.Subnet(
    "public-subnet",
    vpc_id=vpc.id,
    cidr_block="10.0.1.0/24",
    availability_zone=f"{region}a",
    map_public_ip_on_launch=True,
    tags={**tags, "Name": f"modelserve-public-subnet-{env}"},
)

route_table = aws.ec2.RouteTable(
    "route-table",
    vpc_id=vpc.id,
    routes=[
        aws.ec2.RouteTableRouteArgs(
            cidr_block="0.0.0.0/0",
            gateway_id=igw.id,
        )
    ],
    tags={**tags, "Name": f"modelserve-rt-{env}"},
)

aws.ec2.RouteTableAssociation(
    "rt-assoc",
    subnet_id=public_subnet.id,
    route_table_id=route_table.id,
)

# ---------------------------------------------------------------------------
# Security Group
# ---------------------------------------------------------------------------

sg = aws.ec2.SecurityGroup(
    "sg",
    name=f"modelserve-sg-{env}",
    description="ModelServe API, MLflow, Prometheus, Grafana, SSH",
    vpc_id=vpc.id,
    ingress=[
        aws.ec2.SecurityGroupIngressArgs(protocol="tcp", from_port=22,   to_port=22,   cidr_blocks=["0.0.0.0/0"], description="SSH"),
        aws.ec2.SecurityGroupIngressArgs(protocol="tcp", from_port=80,   to_port=80,   cidr_blocks=["0.0.0.0/0"], description="HTTP"),
        aws.ec2.SecurityGroupIngressArgs(protocol="tcp", from_port=8000, to_port=8000, cidr_blocks=["0.0.0.0/0"], description="FastAPI"),
        aws.ec2.SecurityGroupIngressArgs(protocol="tcp", from_port=5000, to_port=5000, cidr_blocks=["0.0.0.0/0"], description="MLflow"),
        aws.ec2.SecurityGroupIngressArgs(protocol="tcp", from_port=9090, to_port=9090, cidr_blocks=["0.0.0.0/0"], description="Prometheus"),
        aws.ec2.SecurityGroupIngressArgs(protocol="tcp", from_port=3000, to_port=3000, cidr_blocks=["0.0.0.0/0"], description="Grafana"),
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(protocol="-1", from_port=0, to_port=0, cidr_blocks=["0.0.0.0/0"], description="All outbound"),
    ],
    tags={**tags, "Name": f"modelserve-sg-{env}"},
)

# ---------------------------------------------------------------------------
# IAM Instance Role (EC2 -> ECR read + S3)
# ---------------------------------------------------------------------------

instance_role = aws.iam.Role(
    "instance-role",
    name=f"modelserve-instance-role-{env}",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }),
    tags=tags,
)

for policy_arn, logical_name in [
    ("arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly", "ecr-read"),
    ("arn:aws:iam::aws:policy/AmazonS3FullAccess", "s3-full"),
    ("arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore", "ssm-core"),
]:
    aws.iam.RolePolicyAttachment(
        f"instance-{logical_name}",
        role=instance_role.name,
        policy_arn=policy_arn,
    )

instance_profile = aws.iam.InstanceProfile(
    "instance-profile",
    name=f"modelserve-instance-profile-{env}",
    role=instance_role.name,
)

# ---------------------------------------------------------------------------
# ECR Repositories
# ---------------------------------------------------------------------------

_lifecycle_policy = json.dumps({
    "rules": [{
        "rulePriority": 1,
        "description": "Keep last 10 images",
        "selection": {"tagStatus": "any", "countType": "imageCountMoreThan", "countNumber": 10},
        "action": {"type": "expire"},
    }]
})

api_repo = aws.ecr.Repository(
    "ecr-api",
    name=f"modelserve-api-{env}",
    image_tag_mutability="MUTABLE",
    force_delete=True,
    image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(scan_on_push=True),
    encryption_configurations=[aws.ecr.RepositoryEncryptionConfigurationArgs(encryption_type="AES256")],
    tags={**tags, "Name": f"modelserve-api-{env}"},
)

aws.ecr.LifecyclePolicy("ecr-api-lifecycle", repository=api_repo.name, policy=_lifecycle_policy)

mlflow_repo = aws.ecr.Repository(
    "ecr-mlflow",
    name=f"modelserve-mlflow-{env}",
    image_tag_mutability="MUTABLE",
    force_delete=True,
    image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(scan_on_push=True),
    encryption_configurations=[aws.ecr.RepositoryEncryptionConfigurationArgs(encryption_type="AES256")],
    tags={**tags, "Name": f"modelserve-mlflow-{env}"},
)

aws.ecr.LifecyclePolicy("ecr-mlflow-lifecycle", repository=mlflow_repo.name, policy=_lifecycle_policy)

# ---------------------------------------------------------------------------
# S3 - MLflow artifact store (private)
# ---------------------------------------------------------------------------

artifacts_bucket = aws.s3.BucketV2(
    "mlflow-artifacts",
    bucket=f"modelserve-mlflow-artifacts-{env}-{account_id}",
    tags={**tags, "Name": f"modelserve-artifacts-{env}"},
)

aws.s3.BucketPublicAccessBlock(
    "mlflow-artifacts-block",
    bucket=artifacts_bucket.id,
    block_public_acls=True,
    block_public_policy=True,
    ignore_public_acls=True,
    restrict_public_buckets=True,
)

# ---------------------------------------------------------------------------
# EC2 Instance
# ---------------------------------------------------------------------------

ami = aws.ec2.get_ami(
    most_recent=True,
    owners=["amazon"],
    filters=[aws.ec2.GetAmiFilterArgs(name="name", values=["al2023-ami-*-x86_64"])],
)

user_data = """#!/bin/bash
set -e
dnf update -y
dnf install -y docker
systemctl enable --now docker
usermod -aG docker ec2-user
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fsSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
mkdir -p /home/ec2-user/modelserve
chown ec2-user:ec2-user /home/ec2-user/modelserve
"""

ec2 = aws.ec2.Instance(
    "ec2",
    ami=ami.id,
    instance_type="t3.medium",
    subnet_id=public_subnet.id,
    vpc_security_group_ids=[sg.id],
    iam_instance_profile=instance_profile.name,
    associate_public_ip_address=True,
    user_data=user_data,
    root_block_device=aws.ec2.InstanceRootBlockDeviceArgs(
        volume_size=30,
        volume_type="gp3",
        delete_on_termination=True,
    ),
    tags={**tags, "Name": f"modelserve-{env}"},
)

eip = aws.ec2.Eip(
    "eip",
    domain="vpc",
    instance=ec2.id,
    tags={**tags, "Name": f"modelserve-eip-{env}"},
)

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

pulumi.export("ec2_public_ip", eip.public_ip)
pulumi.export("ec2_instance_id", ec2.id)
pulumi.export("ecr_api_url", api_repo.repository_url)
pulumi.export("ecr_mlflow_url", mlflow_repo.repository_url)
pulumi.export("ecr_registry", api_repo.repository_url.apply(lambda u: u.split("/")[0]))
pulumi.export("s3_bucket", artifacts_bucket.bucket)
pulumi.export("github_actions_role_arn", github_actions_role.arn)
