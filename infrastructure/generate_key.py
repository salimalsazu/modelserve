from paramiko import RSAKey
import os

key_path = os.path.join(os.path.expanduser("~"), ".ssh", "modelserve-key.pem")
k = RSAKey.generate(2048)
k.write_private_key_file(key_path)
print(f"Key generated at: {key_path}")
