import pulumi
import pulumi_aws as aws

config = pulumi.Config()
project_name = config.require("project_name")
aws_region = config.require("aws_region")

vpc = aws.ec2.Vpc(
    f"{project_name}-vpc",
    cidr_block="10.0.0.0/16",
    enable_dns_hostnames=True,
    enable_dns_support=True,
    tags={
        "Name": f"{project_name}-vpc",
        "Environment": "production"
    }
)

internet_gateway = aws.ec2.InternetGateway(
    f"{project_name}-igw",
    vpc_id=vpc.id,
    tags={
        "Name": f"{project_name}-igw",
        "Environment": "production"
    }
)

public_subnet = aws.ec2.Subnet(
    f"{project_name}-subnet",
    vpc_id=vpc.id,
    cidr_block="10.0.1.0/24",
    availability_zone=f"{aws_region}a",
    map_public_ip_on_launch=True,
    tags={
        "Name": f"{project_name}-subnet",
        "Environment": "production"
    }
)

route_table = aws.ec2.RouteTable(
    f"{project_name}-rt",
    vpc_id=vpc.id,
    routes=[
        aws.ec2.RouteTableRouteArgs(
            cidr_block="0.0.0.0/0",
            gateway_id=internet_gateway.id
        )
    ],
    tags={
        "Name": f"{project_name}-rt",
        "Environment": "production"
    }
)

aws.ec2.RouteTableAssociation(
    f"{project_name}-rta",
    subnet_id=public_subnet.id,
    route_table_id=route_table.id
)

security_group = aws.ec2.SecurityGroup(
    f"{project_name}-sg",
    vpc_id=vpc.id,
    description="Security group for ModelServe",
    ingress=[
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=22,
            to_port=22,
            cidr_blocks=["0.0.0.0/0"]
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=8000,
            to_port=8000,
            cidr_blocks=["0.0.0.0/0"]
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=5000,
            to_port=5000,
            cidr_blocks=["0.0.0.0/0"]
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=5432,
            to_port=5432,
            cidr_blocks=["0.0.0.0/0"]
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=6379,
            to_port=6379,
            cidr_blocks=["0.0.0.0/0"]
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=9090,
            to_port=9090,
            cidr_blocks=["0.0.0.0/0"]
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=3000,
            to_port=3000,
            cidr_blocks=["0.0.0.0/0"]
        ),
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            protocol="-1",
            from_port=0,
            to_port=0,
            cidr_blocks=["0.0.0.0/0"]
        )
    ],
    tags={
        "Name": f"{project_name}-sg",
        "Environment": "production"
    }
)

key_pair = aws.ec2.KeyPair(
    f"{project_name}-key",
    public_key="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFkN7V4sX5r9zP7cJ8x2K9mL3nQ6wR4tY8hG5vB9sK2p0t",
    tags={
        "Name": f"{project_name}-key",
        "Environment": "production"
    }
)

instance = aws.ec2.Instance(
    f"{project_name}-ec2",
    ami="ami-0a8b9e1d9e3e01b7a",
    instance_type="t3.medium",
    subnet_id=public_subnet.id,
    vpc_security_group_ids=[security_group.id],
    key_name=key_pair.key_name,
    root_block_device=aws.ec2.InstanceRootBlockDeviceArgs(
        volume_size=30,
        volume_type="gp3"
    ),
    user_data="""#!/bin/bash
yum update -y
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user
curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
mkdir -p ~/.aws
""",
    tags={
        "Name": f"{project_name}-ec2",
        "Environment": "production"
    }
)

eip = aws.ec2.Eip(
    f"{project_name}-eip",
    instance=instance.id,
    domain="vpc",
    tags={
        "Name": f"{project_name}-eip",
        "Environment": "production"
    }
)

ecr_api = aws.ecr.Repository(
    "modelserve",
    image_tag_mutability="MUTABLE",
    image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(
        scan_on_push=True
    ),
    tags={
        "Environment": "production"
    }
)

ecr_mlflow = aws.ecr.Repository(
    "modelserve-mlflow",
    image_tag_mutability="MUTABLE",
    image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(
        scan_on_push=True
    ),
    tags={
        "Environment": "production"
    }
)

pulumi.export("vpc_id", vpc.id)
pulumi.export("subnet_id", public_subnet.id)
pulumi.export("security_group_id", security_group.id)
pulumi.export("instance_id", instance.id)
pulumi.export("instance_public_ip", instance.public_ip)
pulumi.export("elastic_ip", eip.public_ip)
pulumi.export("ecr_api_repo", ecr_api.repository_url)
pulumi.export("ecr_mlflow_repo", ecr_mlflow.repository_url)