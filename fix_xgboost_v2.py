#!/usr/bin/env python3
"""Fix XGBoost early_stopping_rounds - improved version."""
import json
import re
from pathlib import Path

def fix_xgboost_early_stopping(notebook_path):
    """Fix early_stopping_rounds parameter for XGBoost 3.x."""
    print(f"Fixing XGBoost early_stopping in: {notebook_path.name}")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            if isinstance(source, str):
                source = [source]

            source_text = ''.join(source)

            # Check if this cell uses early_stopping_rounds in fit()
            if 'early_stopping_rounds=' in source_text and '.fit(' in source_text:
                print(f"  Found early_stopping_rounds in fit(), fixing...")

                # Extract the early_stopping_rounds value
                match = re.search(r'early_stopping_rounds\s*=\s*(\d+)', source_text)
                if match:
                    early_stop_value = match.group(1)

                    # More careful replacement - handle different cases
                    # Case 1: ..., early_stopping_rounds=N, ...
                    new_source_text = re.sub(
                        r',\s*early_stopping_rounds\s*=\s*\d+\s*,',
                        ',',
                        source_text
                    )

                    # Case 2: ..., early_stopping_rounds=N\n
                    new_source_text = re.sub(
                        r',\s*early_stopping_rounds\s*=\s*\d+\s*\n',
                        '\n',
                        new_source_text
                    )

                    # Case 3: early_stopping_rounds=N, ...
                    new_source_text = re.sub(
                        r'early_stopping_rounds\s*=\s*\d+\s*,',
                        '',
                        new_source_text
                    )

                    # Find XGBoost model initialization and add parameter there
                    if 'XGBClassifier(' in new_source_text:
                        # Find the constructor
                        constructor_pattern = r'(XGBClassifier\([^)]*?)'
                        def add_param(match):
                            content = match.group(1)
                            # Add early_stopping_rounds before the closing paren
                            if content.rstrip().endswith(','):
                                return content + f'\n    early_stopping_rounds={early_stop_value},'
                            else:
                                return content + f',\n    early_stopping_rounds={early_stop_value}'

                        new_source_text = re.sub(
                            r'XGBClassifier\((.*?)\)',
                            lambda m: f'XGBClassifier({m.group(1)},\n    early_stopping_rounds={early_stop_value})',
                            new_source_text,
                            count=1,
                            flags=re.DOTALL
                        )
                        modified = True

                    elif 'XGBRegressor(' in new_source_text:
                        new_source_text = re.sub(
                            r'XGBRegressor\((.*?)\)',
                            lambda m: f'XGBRegressor({m.group(1)},\n    early_stopping_rounds={early_stop_value})',
                            new_source_text,
                            count=1,
                            flags=re.DOTALL
                        )
                        modified = True

                    if modified:
                        cell['source'] = new_source_text
                        cell['outputs'] = []
                        cell['execution_count'] = None

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  ✓ Fixed and saved")
    else:
        print(f"  No changes needed")

    return modified

# Main execution
notebooks_dir = Path('/home/user/machine-learning-playground/notebooks')

xgboost_notebooks = [
    notebooks_dir / '13_gbdt_introduction_improved_v2.ipynb',
    notebooks_dir / '14_catboost_categorical_improved_v2.ipynb',
    notebooks_dir / '17_titanic_top30_submission_improved_v2.ipynb',
    notebooks_dir / '19_store_demand_timeseries_improved_v2.ipynb'
]

for nb_path in xgboost_notebooks:
    if nb_path.exists():
        fix_xgboost_early_stopping(nb_path)

print("\n✓ All XGBoost fixes applied!")
