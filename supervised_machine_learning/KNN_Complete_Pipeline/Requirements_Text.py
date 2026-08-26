import ast
import os

# Folder path containing Python files
folder_path = r"C:/Users/Bharani Kumar/Desktop/Data Analytics/Supervised Learning/4.b.KNN"

used_packages = set()
skip_libs = {"__future__", "builtins", "os", "sys"}

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".py"):
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                tree = ast.parse(file.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        if n.name.split('.')[0] not in skip_libs:
                            used_packages.add(n.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] not in skip_libs:
                        used_packages.add(node.module.split('.')[0])
        except Exception as e:
            print(f"⚠️ Skipping {filename} due to error: {e}")

# Save to requirements.txt
with open("requirements.txt", "w") as f:
    for pkg in sorted(used_packages):
        f.write(pkg + "\n")

print("✅ requirements.txt created successfully.")
