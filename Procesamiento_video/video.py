import subprocess

subprocess.run(['dir', '/B', '/A:-D', '/O:GN'], shell=True)
