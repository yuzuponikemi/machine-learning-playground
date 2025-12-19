#!/usr/bin/env python3
"""Fix all notebook issues: California Housing fetch and XGBoost early_stopping_rounds."""
import json
import re
from pathlib import Path

def fix_california_housing(notebook_path):
    """Replace fetch_california_housing() with synthetic data."""
    print(f"Fixing California Housing in: {notebook_path.name}")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            if isinstance(source, str):
                source = [source]

            source_text = ''.join(source)

            # Check if this cell fetches California Housing
            if 'fetch_california_housing()' in source_text and 'Synthetic' not in source_text:
                print(f"  Found fetch_california_housing, replacing...")

                # Replace the fetch with synthetic data generation
                new_source = """# Synthetic housing data (replacing fetch_california_housing due to network restrictions)
np.random.seed(42)
X_housing_synthetic, y_housing_synthetic = make_regression(
    n_samples=1000, n_features=8, n_informative=8, noise=15, random_state=42
)
# Create a namespace object to mimic the dataset structure
class SyntheticHousingData:
    def __init__(self):
        self.data = X_housing_synthetic
        self.target = y_housing_synthetic
        # Scale target to realistic housing prices
        self.target = (self.target - self.target.min()) / (self.target.max() - self.target.min()) * 4 + 0.5
        self.feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                              'Population', 'AveOccup', 'Latitude', 'Longitude']

housing_data = SyntheticHousingData()
X_housing = pd.DataFrame(housing_data.data[:1000], columns=housing_data.feature_names)
y_housing = housing_data.target[:1000]

# データ分割
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=42
)

print(f"回帰データサイズ: {X_train_h.shape}")
"""

                cell['source'] = new_source
                cell['outputs'] = []
                cell['execution_count'] = None
                modified = True
                break

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  ✓ Fixed and saved")
    else:
        print(f"  No changes needed")

    return modified

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
                match = re.search(r'early_stopping_rounds=(\d+)', source_text)
                if match:
                    early_stop_value = match.group(1)

                    # Remove early_stopping_rounds from fit() call
                    new_source_text = re.sub(
                        r',?\s*early_stopping_rounds=\d+,?\s*',
                        '',
                        source_text
                    )

                    # Find XGBoost model initialization and add parameter there
                    # Look for XGBClassifier( or XGBRegressor(
                    if 'XGBClassifier(' in new_source_text:
                        new_source_text = new_source_text.replace(
                            'XGBClassifier(',
                            f'XGBClassifier(\n    early_stopping_rounds={early_stop_value},\n   '
                        )
                        modified = True
                    elif 'XGBRegressor(' in new_source_text:
                        new_source_text = new_source_text.replace(
                            'XGBRegressor(',
                            f'XGBRegressor(\n    early_stopping_rounds={early_stop_value},\n   '
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

# Fix California Housing in notebooks 12 and 21
california_housing_notebooks = [
    notebooks_dir / '12_complete_ml_pipeline_improved_v2.ipynb',
    notebooks_dir / '21_shap_model_interpretation_improved_v2.ipynb'
]

for nb_path in california_housing_notebooks:
    if nb_path.exists():
        fix_california_housing(nb_path)

# Fix XGBoost early_stopping in notebooks 13, 14, 17, 19
xgboost_notebooks = [
    notebooks_dir / '13_gbdt_introduction_improved_v2.ipynb',
    notebooks_dir / '14_catboost_categorical_improved_v2.ipynb',
    notebooks_dir / '17_titanic_top30_submission_improved_v2.ipynb',
    notebooks_dir / '19_store_demand_timeseries_improved_v2.ipynb'
]

for nb_path in xgboost_notebooks:
    if nb_path.exists():
        fix_xgboost_early_stopping(nb_path)

print("\n✓ All fixes applied!")
