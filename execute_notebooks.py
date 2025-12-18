#!/usr/bin/env python3
"""
Script to execute all Jupyter notebooks in the notebooks directory.
"""
import os
import sys
import subprocess
from pathlib import Path

def execute_notebook(notebook_path):
    """Execute a single notebook using nbconvert."""
    print(f"Executing: {notebook_path.name}")
    try:
        result = subprocess.run(
            [
                'jupyter', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--inplace',
                '--ExecutePreprocessor.timeout=600',
                str(notebook_path)
            ],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Success: {notebook_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {notebook_path.name}:")
        print(e.stderr)
        return False

def main():
    notebooks_dir = Path('/home/user/machine-learning-playground/notebooks')

    # Get all notebook files, sorted by name
    notebooks = sorted(notebooks_dir.glob('*.ipynb'))

    if not notebooks:
        print("No notebooks found!")
        sys.exit(1)

    print(f"Found {len(notebooks)} notebooks to execute\n")

    successful = []
    failed = []

    for notebook in notebooks:
        if execute_notebook(notebook):
            successful.append(notebook.name)
        else:
            failed.append(notebook.name)
        print()

    # Summary
    print("=" * 60)
    print(f"Summary: {len(successful)}/{len(notebooks)} notebooks executed successfully")
    print("=" * 60)

    if failed:
        print("\nFailed notebooks:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("\n✓ All notebooks executed successfully!")

if __name__ == '__main__':
    main()
