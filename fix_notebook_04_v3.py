#!/usr/bin/env python3
"""Fix notebook 04 properly with list format for cell source."""
import json
from pathlib import Path

notebook_path = Path('/home/user/machine-learning-playground/notebooks/04_linear_models_simulation_improved_v2.ipynb')

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and replace cell 22 which has the housing data
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell.get('source', '')

        # Handle both string and list formats
        if isinstance(source, str):
            source_text = source
        else:
            source_text = ''.join(source)

        # Check if this is the housing data cell
        if 'Synthetic housing' in source_text or 'fetch_california_housing()' in source_text:
            print("Found California Housing cell, replacing with proper format...")

            # Create new source as a list of lines
            new_lines = [
                "# Synthetic housing price dataset (simulating California Housing)\n",
                "np.random.seed(RANDOM_STATE)\n",
                "\n",
                "# Generate base regression data\n",
                "X_housing, y_housing_base = make_regression(\n",
                "    n_samples=20640,  # Same as California Housing\n",
                "    n_features=8,\n",
                "    n_informative=8,\n",
                "    noise=15,\n",
                "    random_state=RANDOM_STATE\n",
                ")\n",
                "\n",
                "# Scale target to realistic housing prices (in $100k units)\n",
                "y_housing = (y_housing_base - y_housing_base.min()) / (y_housing_base.max() - y_housing_base.min()) * 4 + 0.5\n",
                "\n",
                "# Feature names matching California Housing\n",
                "feature_names_housing = [\n",
                "    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',\n",
                "    'Population', 'AveOccup', 'Latitude', 'Longitude'\n",
                "]\n",
                "\n",
                "# DataFrameに変換\n",
                "df_housing = pd.DataFrame(X_housing, columns=feature_names_housing)\n",
                "df_housing['Price'] = y_housing\n",
                "\n",
                "print(\"🏠 Synthetic Housing Dataset (simulating California Housing)\")\n",
                "print(f\"\\nデータサイズ: {df_housing.shape}\")\n",
                "print(f\"\\n特徴量:\")\n",
                "for feat in feature_names_housing:\n",
                "    print(f\"   - {feat}\")\n",
                "\n",
                "print(f\"\\n統計情報:\")\n",
                "print(df_housing.describe())\n",
                "\n",
                "# 相関行列\n",
                "plt.figure(figsize=(10, 8))\n",
                "sns.heatmap(df_housing.corr(), annot=True, fmt='.2f', cmap='RdYlGn', center=0)\n",
                "plt.title('特徴量の相関行列')\n",
                "plt.show()"
            ]

            # Set the cell source as a list
            cell['source'] = new_lines
            cell['outputs'] = []
            cell['execution_count'] = None
            print("✓ Cell replaced successfully with proper list format")
            break

# Save the modified notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✓ Notebook saved: {notebook_path}")
