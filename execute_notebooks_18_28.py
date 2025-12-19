#!/usr/bin/env python3
"""Execute notebooks 18-28 (skipping 16-17 due to timeout issues)."""
import subprocess
from pathlib import Path

def execute_notebook(notebook_path, timeout=600):
    """Execute a single notebook using nbconvert."""
    print(f"Executing: {notebook_path.name}")
    try:
        result = subprocess.run(
            [
                'jupyter', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--inplace',
                f'--ExecutePreprocessor.timeout={timeout}',
                str(notebook_path)
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Success: {notebook_path.name}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {notebook_path.name}:")
        print(e.stderr[-2000:])  # Last 2000 chars to avoid too much output
        print()
        return False

# Find notebooks 18-28
notebooks_dir = Path('/home/user/machine-learning-playground/notebooks')
notebooks = sorted(notebooks_dir.glob('*_improved_v2.ipynb'))

# Filter for notebooks 18-28
target_notebooks = [nb for nb in notebooks if nb.name.startswith(tuple(f'{i:02d}_' for i in range(18, 29)))]

print(f"Found {len(target_notebooks)} notebooks to execute (18-28)\n")

success_count = 0
failed_notebooks = []

for notebook_path in target_notebooks:
    if execute_notebook(notebook_path):
        success_count += 1
    else:
        failed_notebooks.append(notebook_path.name)

print("=" * 60)
print(f"Summary: {success_count}/{len(target_notebooks)} notebooks executed successfully")
if failed_notebooks:
    print(f"\nFailed notebooks:")
    for nb in failed_notebooks:
        print(f"  - {nb}")
else:
    print("\n✓ All notebooks executed successfully!")
