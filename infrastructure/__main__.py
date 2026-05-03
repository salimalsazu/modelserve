import pulumi
import pulumi_aws as aws

server = aws.ec2.Instance(
    "modelserve-server",
    instance_type="t2.micro",
    ami="ami-0c55b159cbfafe1f0",
    tags={
        "Project": "modelserve"
    }
)

pulumi.export("public_ip", server.public_ip)