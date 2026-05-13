#!/bin/bash
# User data script for ModelServe EC2 instance
set -e

echo "=========================================="
echo "Installing Docker and Docker Compose"
echo "=========================================="

# Update and install Docker
sudo dnf update -y
sudo dnf install -y docker

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add ec2-user to docker group
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

echo "=========================================="
echo "Installing GitHub Actions Runner"
echo "=========================================="

# Create runner directory
mkdir -p actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.316.0.tar.gz https://github.com/actions/runner/releases/download/v2.316.0/actions-runner-linux-x64-2.316.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.316.0.tar.gz

echo "=========================================="
echo "Setup complete!"
echo "=========================================="