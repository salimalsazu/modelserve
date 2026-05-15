$env:PULUMI_CONFIG_PASSPHRASE = ""
Set-Location "F:\AI Exam\modelserve\infrastructure"
& .\.venv\Scripts\python.exe -c "import pulumi" 2>&1
pulumi preview
