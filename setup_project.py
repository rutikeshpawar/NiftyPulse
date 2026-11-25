from pathlib import Path

# List of folders we want to create
folders = [
    "data/raw",
    "data/processed",
    "src",
    "notebooks",
    "models",
    "reports/figures",
    "app",
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)   # create folder (and parents)

# Empty file so Python treats src as a package
Path("src/__init__.py").touch()

# Create empty starter files
for file in [".env", "requirements.txt", "README.md", ".gitignore"]:
    Path(file).touch()

print("Project structure created.")