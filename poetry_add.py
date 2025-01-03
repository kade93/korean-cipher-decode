import subprocess

# Get the list of installed packages
result = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE, text=True)
packages = result.stdout.strip().split("\n")

# Generate poetry add commands
poetry_commands = [f"poetry add {pkg}" for pkg in packages]

# Print the commands
for cmd in poetry_commands:
    print(cmd)