"""
Pulumi Infrastructure for ModelServe AWS Environment.
Region: ap-southeast-1 (Singapore)
"""

import os
import pulumi
import pulumi_aws as aws

# Configuration
config = pulumi.Config()
environment = config.get("environment") or "dev"
aws_region = config.get("aws_region") or "ap-southeast-1"

# Tags for all resources
tags = {
    "Project": "ModelServe",
    "Environment": environment,
    "ManagedBy": "Pulumi",
}

# ============================================
# VPC and Networking
# ============================================

# Try to get default VPC first, create new one if not exists
try:
    default_vpc = aws.ec2.get_vpc(default=True)
    vpc_id = default_vpc.id
    vpc_cidr = default_vpc.cidr_block
    print(f"Using existing VPC: {vpc_id}")
except Exception:
    print("No default VPC found, creating new VPC...")
    main_vpc = aws.ec2.Vpc(
        "modelserve-main-vpc",
        cidr_block="10.0.0.0/16",
        enable_dns_hostnames=True,
        enable_dns_support=True,
        tags={**tags, "Name": f"modelserve-vpc-{environment}"},
    )
    vpc_id = main_vpc.id
    vpc_cidr = main_vpc.cidr_block
    
    # Internet Gateway
    internet_gateway = aws.ec2.InternetGateway(
        "modelserve-igw",
        vpc_id=vpc_id,
        tags={**tags, "Name": f"modelserve-igw-{environment}"},
    )
    
    # Public Subnet
    public_subnet = aws.ec2.Subnet(
        "modelserve-public-subnet",
        vpc_id=vpc_id,
        cidr_block="10.0.1.0/24",
        availability_zone=f"{aws_region}a",
        map_public_ip_on_launch=True,
        tags={**tags, "Name": f"modelserve-public-subnet-{environment}"},
    )
    
    # Route Table
    route_table = aws.ec2.RouteTable(
        "modelserve-route-table",
        vpc_id=vpc_id,
        routes=[
            aws.ec2.RouteTableRouteArgs(
                cidr_block="0.0.0.0/0",
                gateway_id=internet_gateway.id,
            )
        ],
        tags={**tags, "Name": f"modelserve-route-table-{environment}"},
    )
    
    # Associate Route Table
    aws.ec2.RouteTableAssociation(
        "modelserve-rt-association",
        subnet_id=public_subnet.id,
        route_table_id=route_table.id,
    )

# Create security groups
api_security_group = aws.ec2.SecurityGroup(
    "modelserve-api-sg",
    name=f"modelserve-api-{environment}",
    description="Security group for ModelServe API",
    vpc_id=vpc_id,
    tags={**tags, "Name": f"modelserve-api-sg-{environment}"},
)

# Security group rules
aws.ec2.SecurityGroupRule(
    "ssh-access",
    type="ingress",
    from_port=22,
    to_port=22,
    protocol="tcp",
    cidr_blocks=["0.0.0.0/0"],
    security_group_id=api_security_group.id,
)

aws.ec2.SecurityGroupRule(
    "http-access",
    type="ingress",
    from_port=80,
    to_port=80,
    protocol="tcp",
    cidr_blocks=["0.0.0.0/0"],
    security_group_id=api_security_group.id,
)

aws.ec2.SecurityGroupRule(
    "api-access",
    type="ingress",
    from_port=8000,
    to_port=8000,
    protocol="tcp",
    cidr_blocks=["0.0.0.0/0"],
    security_group_id=api_security_group.id,
)

aws.ec2.SecurityGroupRule(
    "mlflow-access",
    type="ingress",
    from_port=5000,
    to_port=5000,
    protocol="tcp",
    cidr_blocks=["0.0.0.0/0"],
    security_group_id=api_security_group.id,
)

aws.ec2.SecurityGroupRule(
    "prometheus-access",
    type="ingress",
    from_port=9090,
    to_port=9090,
    protocol="tcp",
    cidr_blocks=["0.0.0.0/0"],
    security_group_id=api_security_group.id,
)

aws.ec2.SecurityGroupRule(
    "grafana-access",
    type="ingress",
    from_port=3000,
    to_port=3000,
    protocol="tcp",
    cidr_blocks=["0.0.0.0/0"],
    security_group_id=api_security_group.id,
)

aws.ec2.SecurityGroupRule(
    "all-egress",
    type="egress",
    from_port=0,
    to_port=0,
    protocol="-1",
    cidr_blocks=["0.0.0.0/0"],
    security_group_id=api_security_group.id,
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
)

instance_profile = aws.iam.InstanceProfile(
    "modelserve-instance-profile",
    name=f"modelserve-instance-profile-{environment}",
    role=instance_role.name,
)

aws.iam.RolePolicyAttachment(
    "s3-access",
    role=instance_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonS3FullAccess",
)

aws.iam.RolePolicyAttachment(
    "ecr-readonly",
    role=instance_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
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
    bucket=f"modelserve-artifacts-{environment}-{aws.get_caller_identity().account_id}",
    tags={**tags, "Name": f"modelserve-artifacts-{environment}"},
)

aws.s3.BucketPublicAccessBlock(
    "mlflow-artifacts-block",
    bucket=mlflow_bucket.id,
    block_public_acls=True,
    block_public_policy=True,
    ignore_public_acls=True,
    restrict_public_buckets=True,
)

# ============================================
# EC2 Instance
# ============================================

# Get Amazon Linux 2023 AMI
ami = aws.ec2.get_ami(
    most_recent=True,
    owners=["amazon"],
    filters=[aws.ec2.GetAmiFilterArgs(name="name", values=["al2023-ami-*"])],
)

ec2_instance = aws.ec2.Instance(
    "modelserve-ec2",
    ami=ami.id,
    instance_type="t3.medium",
    vpc_security_group_ids=[api_security_group.id],
    iam_instance_profile=instance_profile.name,
    user_data="""#!/bin/bash
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
""",
    tags={**tags, "Name": f"modelserve-{environment}-instance"},
)

# Elastic IP
eip = aws.ec2.Eip(
    "modelserve-eip",
    domain="vpc",
    instance=ec2_instance.id,
    tags={**tags, "Name": f"modelserve-{environment}-eip"},
)

# ============================================
# Outputs
# ============================================

pulumi.export("ec2_public_ip", ec2_instance.public_ip)
pulumi.export("ec2_private_ip", ec2_instance.private_ip)
pulumi.export("ec2_instance_id", ec2_instance.id)
pulumi.export("ecr_repository_url", ecr_repository.repository_url)
pulumi.export("ecr_registry", ecr_repository.registry_id)
pulumi.export("s3_bucket_name", mlflow_bucket.bucket)
pulumi.export("security_group_id", api_security_group.id)
pulumi.export("vpc_id", vpc_id)