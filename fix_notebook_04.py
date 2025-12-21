#!/usr/bin/env python3
"""Fix notebook 04 to use synthetic data instead of fetching California Housing dataset."""
import json
from pathlib import Path

notebook_path = Path('/home/user/machine-learning-playground/notebooks/04_linear_models_simulation_improved_v2.ipynb')

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and replace cell 22 which fetches California Housing data
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        # Check if this is the cell that fetches California Housing
        if 'fetch_california_housing()' in source:
            print("Found California Housing fetch cell, replacing with synthetic data...")

            # Replace with synthetic data generation
            new_source = """# Synthetic housing price dataset (simulating California Housing)
# Generate realistic housing data with 8 features
from sklearn.datasets import make_regression

np.random.seed(RANDOM_STATE)

# Generate base regression data
X_housing, y_housing_base = make_regression(
    n_samples=20640,  # Same as California Housing
    n_features=8,
    n_informative=8,
    noise=15,
    random_state=RANDOM_STATE
)

# Scale target to realistic housing prices (in $100k units)
y_housing = (y_housing_base - y_housing_base.min()) / (y_housing_base.max() - y_housing_base.min()) * 4 + 0.5

# Feature names matching California Housing
feature_names_housing = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]

# DataFrameに変換
df_housing = pd.DataFrame(X_housing, columns=feature_names_housing)
df_housing['Price'] = y_housing

print("🏠 Synthetic Housing Dataset (simulating California Housing)")
print(f"\\nデータサイズ: {df_housing.shape}")
print(f"\\n特徴量:")
for feat in feature_names_housing:
    print(f"   - {feat}")

print(f"\\n統計情報:")
display(df_housing.describe())

# 相関行列
plt.figure(figsize=(10, 8))
sns.heatmap(df_housing.corr(), annot=True, fmt='.2f', cmap='RdYlGn', center=0)
plt.title('特徴量の相関行列')
plt.show()"""

            cell['source'] = new_source.split('\n')
            cell['outputs'] = []
            cell['execution_count'] = None
            print("✓ Cell replaced successfully")

# Save the modified notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✓ Notebook saved: {notebook_path}")
