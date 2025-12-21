#!/usr/bin/env python3
"""
Script to fix and execute notebooks 14, 16, 18, 21, 22, 23, 26, 28
"""
import json
import sys
import subprocess
from pathlib import Path

notebooks_dir = Path("/Users/ikmx/source/personal/machine-learning-playground/notebooks")

# Fix notebook 14 - replace get_tree_count() with tree_count_
notebook_14 = notebooks_dir / "14_catboost_categorical_improved_v2.ipynb"
print(f"Fixing {notebook_14.name}...")

with open(notebook_14, 'r') as f:
    nb = json.load(f)

# Find and fix the problematic cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'get_tree_count()' in source:
            print(f"  Found problematic code in cell, fixing...")
            # Replace get_tree_count() with tree_count_
            if isinstance(cell['source'], list):
                cell['source'] = [line.replace('get_tree_count()', 'tree_count_') for line in cell['source']]
            else:
                cell['source'] = cell['source'].replace('get_tree_count()', 'tree_count_')
            print(f"  Fixed!")

# Save the fixed notebook
with open(notebook_14, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"✓ Fixed and saved {notebook_14.name}\n")

# List of notebooks to execute
notebooks_to_execute = [
    "14_catboost_categorical_improved_v2.ipynb",
    "16_titanic_gbdt_modeling_improved_v2.ipynb",
    "18_house_prices_regression_improved_v2.ipynb",
    "21_shap_model_interpretation_improved_v2.ipynb",
    "22_stacking_ensemble_meta_learning_improved_v2.ipynb",
    "23_imbalanced_data_handling_improved_v2.ipynb",
    "26_tabular_deep_learning_improved_v2.ipynb",
    "28_comprehensive_project_improved_v2.ipynb",
]

jupyter_nbconvert = "/Users/ikmx/Library/Python/3.9/bin/jupyter-nbconvert"

print("Starting notebook execution...\n")
for nb_name in notebooks_to_execute:
    nb_path = notebooks_dir / nb_name
    print(f"Executing {nb_name}...")
    try:
        result = subprocess.run(
            [jupyter_nbconvert, "--to", "notebook", "--execute", "--inplace",
             str(nb_path), "--ExecutePreprocessor.timeout=600"],
            capture_output=True,
            text=True,
            timeout=700
        )
        if result.returncode == 0:
            print(f"  ✓ Successfully executed {nb_name}\n")
        else:
            print(f"  ✗ Error executing {nb_name}")
            print(f"  STDERR: {result.stderr[:500]}\n")
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout executing {nb_name}\n")
    except Exception as e:
        print(f"  ✗ Exception executing {nb_name}: {str(e)}\n")

print("\nDone!")
