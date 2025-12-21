#!/usr/bin/env python3
"""
Execute fixed notebooks 20-28 (excluding 16, 18 which timeout)
"""

import subprocess
import sys
from pathlib import Path

def execute_notebook(notebook_path, timeout=900):
    """Execute a single notebook using nbconvert."""
    print(f"\n{'='*60}")
    print(f"Executing: {notebook_path.name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run([
            'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--inplace',
            f'--ExecutePreprocessor.timeout={timeout}',
            str(notebook_path)
        ], check=True, capture_output=True, text=True, timeout=timeout+60)

        print(f"✅ Success: {notebook_path.name}")
        return True
    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout: {notebook_path.name} (exceeded {timeout}s)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {notebook_path.name}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout[-2000:]}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr[-2000:]}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {notebook_path.name}")
        print(f"Error: {str(e)}")
        return False

def main():
    notebooks_dir = Path('notebooks')

    # Notebooks to execute (excluding 16 and 18 due to timeout issues)
    notebook_numbers = [17, 19, 20, 21, 22, 23, 25, 26, 28]

    results = {}

    for num in notebook_numbers:
        matching_notebooks = list(notebooks_dir.glob(f'{num:02d}_*.ipynb'))

        if not matching_notebooks:
            print(f"⚠️ Notebook {num} not found")
            continue

        notebook_path = matching_notebooks[0]

        # Use longer timeout for notebooks with Optuna (17, 20)
        timeout = 1800 if num in [17, 20] else 900

        success = execute_notebook(notebook_path, timeout=timeout)
        results[num] = 'Success' if success else 'Failed'

    print(f"\n\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")

    for num in notebook_numbers:
        status = results.get(num, 'Not executed')
        emoji = '✅' if status == 'Success' else '❌' if status == 'Failed' else '⚠️'
        print(f"{emoji} Notebook {num:02d}: {status}")

    successful = sum(1 for s in results.values() if s == 'Success')
    total = len(results)
    print(f"\n📊 Total: {successful}/{total} successful")

    return 0 if successful == total else 1

if __name__ == '__main__':
    sys.exit(main())
