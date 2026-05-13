#!/usr/bin/env python3
"""Run Pulumi infrastructure provisioning."""
import os
import subprocess
import shutil
import sys

# Use conda python if available
conda_python = shutil.which("python", path=os.path.join(os.environ.get("CONDA_PREFIX", ""), "Scripts"))
python_exe = os.path.join(os.environ.get("CONDA_PREFIX", ""), "python.exe") if os.path.exists(os.path.join(os.environ.get("CONDA_PREFIX", ""), "python.exe")) else sys.executable

# Set environment variables
os.environ["AWS_ACCESS_KEY_ID"] = "AKIA3NAW5VGYNDEAE6GO"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
os.environ["AWS_REGION"] = "ap-southeast-1"
os.environ["PULUMI_ACCESS_TOKEN"] = "pul-28d78f456889e3e9f0b68ffcb512b0c57c730571"

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Find pulumi CLI
pulumi_cmd = shutil.which("pulumi") or os.path.join(os.environ.get("PULUMI_HOME", os.path.expanduser("~/.pulumi")), "bin", "pulumi.exe")

print("=" * 50)
print("ModelServe Infrastructure Provisioning")
print("Region: ap-southeast-1 (Singapore)")
print("=" * 50)

# Login
print("\n[1/5] Logging in...")
subprocess.run([pulumi_cmd, "login", "--non-interactive"], env=os.environ)

# Stack
print("\n[2/5] Selecting stack...")
result = subprocess.run([pulumi_cmd, "stack", "select", "prod"], env=os.environ)
if result.returncode != 0:
    print("   Creating new stack...")
    subprocess.run([pulumi_cmd, "stack", "init", "prod"], env=os.environ)

# Config
print("\n[3/5] Setting config...")
subprocess.run([pulumi_cmd, "config", "set", "aws:region", "ap-southeast-1", "--stack", "prod"], env=os.environ)
subprocess.run([pulumi_cmd, "config", "set", "environment", "prod", "--stack", "prod"], env=os.environ)

# Destroy old resources first (if any)
print("\n[4/5] Checking existing resources...")
subprocess.run([pulumi_cmd, "destroy", "--stack", "prod", "--yes", "--non-interactive"], env=os.environ, capture_output=True)

# Preview
print("\n[5/5] Previewing infrastructure...")
result = subprocess.run([pulumi_cmd, "preview", "--stack", "prod", "--non-interactive"], env=os.environ)

if result.returncode == 0:
    print("\n" + "=" * 50)
    print("Deploying infrastructure...")
    print("=" * 50)
    result = subprocess.run([pulumi_cmd, "up", "--stack", "prod", "--yes"], env=os.environ)
    
    if result.returncode == 0:
        print("\n" + "=" * 50)
        print("Infrastructure deployed! Outputs:")
        print("=" * 50)
        subprocess.run([pulumi_cmd, "stack", "output", "--stack", "prod"], env=os.environ)
else:
    print("\nPreview failed. Check errors above.")
    sys.exit(1)