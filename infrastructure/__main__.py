"""
Pulumi Infrastructure for ModelServe AWS Environment.

This module provisions:
- EC2 instance for ModelServe API
- S3 bucket for MLflow artifacts
- ECR repository for Docker images
- Security groups and networking
"""

import os
import pulumi
import pulumi_aws as aws

# Configuration
config = pulumi.Config()
environment = config.get("environment") or "dev"
aws_region = config.get("aws_region") or "us-east-1"

# Tags for all resources
tags = {
    "Project": "ModelServe",
    "Environment": environment,
    "ManagedBy": "Pulumi",
}

# ============================================
# VPC and Networking
# ============================================

# Get default VPC
default_vpc = aws.ec2.get_vpc(default=True)

# Create security groups
api_security_group = aws.ec2.SecurityGroup(
    "modelserve-api-sg",
    name=f"modelserve-api-{environment}",
    description="Security group for ModelServe API",
    vpc_id=default_vpc.id,
    tags={**tags, "Name": f"modelserve-api-sg-{environment}"},
)

# Allow SSH for debugging
aws.ec2.SecurityGroupRule(
    "ssh-access",
    type="ingress",
    from_port=22,
    to_port=22,
    protocol="tcp",
    cidr_blocks=["0.0.0.0/0"],
    security_group_id=api_security_group.id,
    description="SSH access for debugging",
)

# Allow HTTP/HTTPS
aws.ec2.SecurityGroupRule(
    "http-access",
    type="ingress",
    from_port=80,
    to_port=80,
    protocol="tcp",
    cidr_blocks=["0.0.0.0/0"],
    security_group_id=api_security_group.id,
    description="HTTP access",
)

aws.ec2.SecurityGroupRule(
    "https-access",
    type="ingress",
    from_port=443,
    to_port=443,
    protocol="tcp",
    cidr_blocks=["0.0.0.0/0"],
    security_group_id=api_security_group.id,
    description="HTTPS access",
)

# Allow API port 8000
aws.ec2.SecurityGroupRule(
    "api-access",
    type="ingress",
    from_port=8000,
    to_port=8000,
    protocol="tcp",
    cidr_blocks=["0.0.0.0/0"],
    security_group_id=api_security_group.id,
    description="ModelServe API port",
)

# Allow MLflow port 5000
aws.ec2.SecurityGroupRule(
    "mlflow-access",
    type="ingress",
    from_port=5000,
    to_port=5000,
    protocol="tcp",
    cidr_blocks=["10.0.0.0/16"],
    security_group_id=api_security_group.id,
    description="MLflow tracking server",
)

# Allow Prometheus port 9090
aws.ec2.SecurityGroupRule(
    "prometheus-access",
    type="ingress",
    from_port=9090,
    to_port=9090,
    protocol="tcp",
    cidr_blocks=["10.0.0.0/16"],
    security_group_id=api_security_group.id,
    description="Prometheus monitoring",
)

# Allow Grafana port 3000
aws.ec2.SecurityGroupRule(
    "grafana-access",
    type="ingress",
    from_port=3000,
    to_port=3000,
    protocol="tcp",
    cidr_blocks=["10.0.0.0/16"],
    security_group_id=api_security_group.id,
    description="Grafana dashboard",
)

# Allow Redis port 6379
aws.ec2.SecurityGroupRule(
    "redis-access",
    type="ingress",
    from_port=6379,
    to_port=6379,
    protocol="tcp",
    cidr_blocks=["10.0.0.0/16"],
    security_group_id=api_security_group.id,
    description="Redis feature store",
)

# Allow all egress
aws.ec2.SecurityGroupRule(
    "all-egress",
    type="egress",
    from_port=0,
    to_port=0,
    protocol="-1",
    cidr_blocks=["0.0.0.0/0"],
    security_group_id=api_security_group.id,
    description="Allow all outbound",
)

# ============================================
# IAM Role for EC2
# ============================================

instance_role = aws.iam.Role(
    "modelserve-instance-role",
    name=f"modelserve-instance-role-{environment}",
    assume_role_policy="""{
        "Version": "2012-10-17",
        "Statement": [{
            "Action": "sts:AssumeRole",
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"}
        }]
    }""",
    tags=tags,
)

# Instance profile
instance_profile = aws.iam.InstanceProfile(
    "modelserve-instance-profile",
    name=f"modelserve-instance-profile-{environment}",
    role=instance_role.name,
)

# S3 full access policy attachment
aws.iam.RolePolicyAttachment(
    "s3-access",
    role=instance_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonS3FullAccess",
)

# ============================================
# ECR Repository
# ============================================

ecr_repository = aws.ecr.Repository(
    "modelserve",
    name=f"modelserve-{environment}",
    image_tag_mutability="MUTABLE",
    image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(
        scan_on_push=True,
    ),
    encryption_configuration=aws.ecr.RepositoryEncryptionConfigurationArgs(
        encryption_type="AES256",
    ),
    tags={**tags, "Name": f"modelserve-{environment}"},
)

# Lifecycle policy for image cleanup
aws.ecr.LifecyclePolicy(
    "modelserve-lifecycle",
    repository=ecr_repository.name,
    policy="""{
        "rules": [{
            "rulePriority": 1,
            "description": "Keep last 10 images",
            "selection": {
                "tagStatus": "any",
                "countType": "imageCountMoreThan",
                "countNumber": 10
            },
            "action": {
                "type": "expire"
            }
        }]
    }""",
)

# ============================================
# S3 Bucket for MLflow Artifacts
# ============================================

mlflow_bucket = aws.s3.BucketV2(
    "modelserve-artifacts",
    bucket=f"modelserve-artifacts-{environment}",
    tags={**tags, "Name": f"modelserve-artifacts-{environment}"},
)

# Block public access
aws.s3.BucketPublicAccessBlock(
    "mlflow-artifacts-block",
    bucket=mlflow_bucket.id,
    block_public_acls=True,
    block_public_policy=True,
    ignore_public_acls=True,
    restrict_public_buckets=True,
)

# Bucket versioning
aws.s3.BucketVersioningV2(
    "mlflow-artifacts-versioning",
    bucket=mlflow_bucket.id,
    versioning_configuration=aws.s3.BucketVersioningV2VersioningConfigurationArgs(
        status="Enabled",
    ),
)

# ============================================
# EC2 Instance for ModelServe
# ============================================

# Get latest Amazon Linux 2 AMI
ami = aws.ec2.get_ami(
    most_recent=True,
    owners=["amazon"],
    filters=[aws.ec2.GetAmiFilterArgs(name="name", values=["amzn2-ami-hvm-*-x86_64-gp2"])],
)

# User data script for Docker installation
user_data = """#!/bin/bash
yum update -y
amazon-linux-extras install docker -y
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

# Pull and run ModelServe
docker pull {ecr_uri}:latest
docker run -d --name modelserve \
  -p 8000:8000 \
  -p 5000:5000 \
  -p 9090:9090 \
  -p 3000:3000 \
  --restart unless-stopped \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  -e AWS_DEFAULT_REGION=us-east-1 \
  {ecr_uri}:latest
""".format(ecr_uri=ecr_repository.repository_url)

# Create EC2 instance
ec2_instance = aws.ec2.Instance(
    "modelserve-instance",
    ami=ami.id,
    instance_type="t3.medium",
    iam_instance_profile=instance_profile.name,
    vpc_security_group_ids=[api_security_group.id],
    user_data=user_data,
    tags={**tags, "Name": f"modelserve-{environment}-instance"},
)

# Get instance public IP
instance_ip = ec2_instance.public_ip

# ============================================
# Outputs
# ============================================

pulumi.export("ec2_public_ip", instance_ip)
pulumi.export("ec2_instance_id", ec2_instance.id)
pulumi.export("ecr_repository_url", ecr_repository.repository_url)
pulumi.export("mlflow_bucket_name", mlflow_bucket.id)
pulumi.export("security_group_id", api_security_group.id)