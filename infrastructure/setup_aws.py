#!/usr/bin/env python3
"""
AWS Infrastructure Setup using AWS CLI
Region: ap-southeast-1 (Singapore)
"""
import os
import subprocess
import json

# AWS Credentials
os.environ["AWS_ACCESS_KEY_ID"] = "AKIA3NAW5VGYNDEAE6GO"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
os.environ["AWS_REGION"] = "ap-southeast-1"

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_cmd(cmd, description):
    print(f"\n{'='*50}")
    print(f"{description}")
    print(f"{'='*50}")
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    return result

# 1. Create VPC
print("\n[1/7] Creating VPC...")
result = run_cmd([
    "aws", "ec2", "create-vpc",
    "--cidr-block", "10.0.0.0/16",
    "--tag-specifications", 'ResourceType=vpc,Tags=[{Key=Name,Value=modelserve-vpc}]'
], "Create VPC")
vpc_id = json.loads(result.stdout)["Vpc"]["VpcId"]
print(f"VPC ID: {vpc_id}")

# 2. Create Internet Gateway
print("\n[2/7] Creating Internet Gateway...")
result = run_cmd([
    "aws", "ec2", "create-internet-gateway",
    "--tag-specifications", 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=modelserve-igw}]'
], "Create IGW")
igw_id = json.loads(result.stdout)["InternetGateway"]["InternetGatewayId"]
print(f"IGW ID: {igw_id}")

# 3. Attach IGW to VPC
print("\n[3/7] Attaching IGW to VPC...")
run_cmd([
    "aws", "ec2", "attach-internet-gateway",
    "--vpc-id", vpc_id,
    "--internet-gateway-id", igw_id
], "Attach IGW")

# 4. Create Subnet
print("\n[4/7] Creating Subnet...")
result = run_cmd([
    "aws", "ec2", "create-subnet",
    "--vpc-id", vpc_id,
    "--cidr-block", "10.0.1.0/24",
    "--availability-zone", "ap-southeast-1a",
    "--tag-specifications", 'ResourceType=subnet,Tags=[{Key=Name,Value=modelserve-subnet}]'
], "Create Subnet")
subnet_id = json.loads(result.stdout)["Subnet"]["SubnetId"]
print(f"Subnet ID: {subnet_id}")

# 5. Create Route Table
print("\n[5/7] Creating Route Table...")
result = run_cmd([
    "aws", "ec2", "create-route-table",
    "--vpc-id", vpc_id
], "Create Route Table")
rtb_id = json.loads(result.stdout)["RouteTable"]["RouteTableId"]
print(f"RTB ID: {rtb_id}")

# Add route to IGW
run_cmd([
    "aws", "ec2", "create-route",
    "--route-table-id", rtb_id,
    "--destination-cidr-block", "0.0.0.0/0",
    "--gateway-id", igw_id
], "Add route to IGW")

# Associate RT with Subnet
run_cmd([
    "aws", "ec2", "associate-route-table",
    "--subnet-id", subnet_id,
    "--route-table-id", rtb_id
], "Associate RT with Subnet")

# Note: Key pair not created - access via AWS Console browser connect
# Key pair is optional for EC2 instance launch

# 6. Create Security Group
print("\n[6/7] Creating Security Group...")
result = run_cmd([
    "aws", "ec2", "create-security-group",
    "--group-name", "modelserve-sg",
    "--description", "ModelServe Security Group",
    "--vpc-id", vpc_id,
    "--tag-specifications", 'ResourceType=security-group,Tags=[{Key=Name,Value=modelserve-sg}]'
], "Create Security Group")
sg_id = json.loads(result.stdout)["GroupId"]
print(f"Security Group ID: {sg_id}")

# Add security group rules
rules = [
    ("22", "SSH"),
    ("80", "HTTP"),
    ("443", "HTTPS"),
    ("8000", "API"),
    ("5000", "MLflow"),
    ("9090", "Prometheus"),
    ("3000", "Grafana"),
]
for port, name in rules:
    run_cmd([
        "aws", "ec2", "authorize-security-group-ingress",
        "--group-id", sg_id,
        "--protocol", "tcp",
        "--port", port,
        "--cidr", "0.0.0.0/0"
    ], f"Allow {name} (port {port})")

# 7. Create EC2 Instance
print("\n[7/7] Launching EC2 Instance...")
result = run_cmd([
    "aws", "ec2", "run-instances",
    "--image-id", "ami-064ac0bc94e195394",  # Amazon Linux 2023 minimal x86_64 in Singapore
    "--count", "1",
    "--instance-type", "t3.medium",
    "--subnet-id", subnet_id,
    "--security-group-ids", sg_id,
    "--associate-public-ip-address",
    "--tag-specifications", 'ResourceType=instance,Tags=[{Key=Name,Value=modelserve-instance}]',
    "--user-data", """#!/bin/bash
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
"""
], "Launch EC2")

# Get instance ID
instance_id = json.loads(result.stdout)["Instances"][0]["InstanceId"]
print(f"Instance ID: {instance_id}")

# Wait for instance to be running
print("\nWaiting for instance to be running...")
run_cmd([
    "aws", "ec2", "wait", "instance-running",
    "--instance-ids", instance_id
], "Wait for instance")

# Get public IP
result = run_cmd([
    "aws", "ec2", "describe-instances",
    "--instance-ids", instance_id
], "Get instance details")
reservation = json.loads(result.stdout)["Reservations"][0]
instance = reservation["Instances"][0]
public_ip = instance.get("PublicIpAddress", "N/A")
print(f"\n{'='*50}")
print("EC2 Public IP:", public_ip)
print(f"{'='*50}")

# Create ECR Repository
print("\n[EXTRA] Creating ECR Repository...")
run_cmd([
    "aws", "ecr", "create-repository",
    "--repository-name", "modelserve-prod",
    "--region", "ap-southeast-1"
], "Create ECR")

# Summary
print("\n" + "="*50)
print("INFRASTRUCTURE DEPLOYED!")
print("="*50)
print(f"VPC ID:        {vpc_id}")
print(f"Subnet ID:     {subnet_id}")
print(f"Security Group:{sg_id}")
print(f"EC2 Instance:  {instance_id}")
print(f"Public IP:     {public_ip}")
print("\nSave the Public IP for deployment!")