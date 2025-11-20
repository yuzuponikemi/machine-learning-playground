# Machine Learning Learning Plan with Jupyter Notebooks

A hands-on curriculum for learning ML through simulations, parameter exploration, and MLP training using scikit-learn.

---

## Overview

This plan focuses on **learning by doing** - each notebook contains:
- Synthetic data generation with controllable parameters
- Visual waveform/outcome analysis
- Interactive parameter exploration
- Model training and evaluation
- Hyperparameter tuning across parameter space

---

## Phase 1: Foundations (Notebooks 1-3)

### Notebook 1: `01_data_simulation_basics.ipynb`
**Goal**: Understand how data parameters affect outcomes

**Contents**:
- Generate synthetic waveforms (sine, noise, combinations)
- Visualize how parameters (amplitude, frequency, noise) affect signals
- Create regression datasets with `make_regression()`
- Create classification datasets with `make_classification()`, `make_moons()`, `make_circles()`
- Interactive sliders for parameter exploration (ipywidgets)

**Key Skills**:
- numpy for data generation
- matplotlib/seaborn for visualization
- Understanding feature distributions

---

### Notebook 2: `02_preprocessing_and_feature_engineering.ipynb`
**Goal**: Prepare data for ML models

**Contents**:
- Scaling: StandardScaler, MinMaxScaler, RobustScaler
- Encoding: OneHotEncoder, LabelEncoder
- Feature selection techniques
- Train/test/validation splits
- Visualize how preprocessing affects data distribution

**Key Skills**:
- sklearn.preprocessing
- sklearn.model_selection
- Pipeline construction

---

### Notebook 3: `03_model_evaluation_metrics.ipynb`
**Goal**: Understand how to measure model performance

**Contents**:
- Regression metrics: MSE, RMSE, MAE, R²
- Classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrices visualization
- Cross-validation strategies
- Learning curves and validation curves

**Key Skills**:
- sklearn.metrics
- sklearn.model_selection.cross_val_score
- Interpreting model performance

---

## Phase 2: Classical ML Models (Notebooks 4-6)

### Notebook 4: `04_linear_models_simulation.ipynb`
**Goal**: Deep dive into linear models with parameter simulation

**Contents**:
- Linear Regression on synthetic waveforms
- Simulate: noise levels, feature correlations, outliers
- Ridge, Lasso, ElasticNet regularization
- Visualize decision boundaries
- Parameter sweep: alpha values, regularization strength

**Experiments**:
```python
# Example: Sweep regularization parameters
alphas = np.logspace(-4, 4, 50)
for alpha in alphas:
    model = Ridge(alpha=alpha)
    # Train, evaluate, store results
# Plot: alpha vs MSE, alpha vs coefficient magnitude
```

---

### Notebook 5: `05_tree_and_ensemble_models.ipynb`
**Goal**: Understand tree-based models and ensembles

**Contents**:
- Decision Trees: max_depth, min_samples_split effects
- Random Forest: n_estimators, max_features exploration
- Gradient Boosting: learning_rate, n_estimators tuning
- Visualize tree structures
- Feature importance analysis

**Experiments**:
- Simulate overfitting with deep trees
- Compare single tree vs ensemble performance
- Parameter grid visualization (heatmaps)

---

### Notebook 6: `06_svm_and_kernels.ipynb`
**Goal**: Understand SVM and kernel methods

**Contents**:
- Linear SVM on linearly separable data
- Kernel SVM: RBF, polynomial kernels
- Simulate non-linear decision boundaries
- C and gamma parameter exploration
- Visualize support vectors and margins

---

## Phase 3: Neural Networks & MLP (Notebooks 7-9) ⭐

### Notebook 7: `07_mlp_fundamentals.ipynb`
**Goal**: Build intuition for MLP architecture

**Contents**:
- MLP architecture explanation with diagrams
- Forward pass visualization
- Activation functions: ReLU, tanh, sigmoid (visualized)
- sklearn MLPClassifier and MLPRegressor basics
- Simple classification on make_moons data

**Experiments**:
```python
# Visualize activation functions
x = np.linspace(-5, 5, 100)
activations = {
    'relu': np.maximum(0, x),
    'tanh': np.tanh(x),
    'sigmoid': 1 / (1 + np.exp(-x))
}
# Plot all on same axes
```

---

### Notebook 8: `08_mlp_parameter_space_exploration.ipynb` ⭐⭐⭐
**Goal**: Comprehensive MLP hyperparameter tuning

**Contents**:
- **Architecture Search**:
  - hidden_layer_sizes: (10,), (50,), (100,), (50,50), (100,50,25)
  - Visualize how depth/width affects learning

- **Learning Parameters**:
  - learning_rate_init: 0.0001 to 0.1
  - alpha (L2 regularization): 0.0001 to 1.0
  - batch_size effects

- **Optimization**:
  - solver: 'adam', 'sgd', 'lbfgs'
  - momentum for SGD
  - early_stopping effects

**Complete Simulation**:
```python
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, learning_curve
import numpy as np

# Generate synthetic data with known patterns
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=3,
    random_state=42
)

# Define parameter space
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate_init': [0.001, 0.01],
    'solver': ['adam'],
    'max_iter': [500]
}

# Grid search with cross-validation
mlp = MLPClassifier(random_state=42, early_stopping=True)
grid_search = GridSearchCV(
    mlp, param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)
grid_search.fit(X_train, y_train)

# Visualize results
results_df = pd.DataFrame(grid_search.cv_results_)
# Create heatmaps, line plots, etc.
```

**Visualizations**:
- Heatmap: hidden_layer_sizes vs alpha (accuracy)
- Learning curves for best models
- Loss curves during training
- Decision boundary evolution (for 2D data)

---

### Notebook 9: `09_mlp_regression_waveforms.ipynb` ⭐⭐⭐
**Goal**: MLP for waveform/signal prediction

**Contents**:
- Generate complex waveforms (sum of sines, damped oscillations)
- Train MLP to predict waveform values
- Visualize predicted vs actual waveforms
- Parameter effects on waveform reconstruction

**Complete Simulation**:
```python
from sklearn.neural_network import MLPRegressor

# Generate complex waveform
t = np.linspace(0, 4*np.pi, 1000)
y_true = (np.sin(t) +
          0.5*np.sin(3*t) +
          0.25*np.sin(5*t) +
          0.1*np.random.randn(len(t)))

# Create features (e.g., time-delayed values)
def create_features(y, n_lags=10):
    X = np.array([y[i:i+n_lags] for i in range(len(y)-n_lags)])
    y_target = y[n_lags:]
    return X, y_target

X, y = create_features(y_true, n_lags=20)

# Train MLP Regressor
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
mlp.fit(X_train, y_train)

# Plot predicted vs actual waveform
y_pred = mlp.predict(X_test)
plt.figure(figsize=(14, 5))
plt.plot(y_test, label='Actual', alpha=0.7)
plt.plot(y_pred, label='Predicted', alpha=0.7)
plt.legend()
plt.title('MLP Waveform Prediction')
```

---

## Phase 4: Advanced Topics (Notebooks 10-12)

### Notebook 10: `10_automated_hyperparameter_tuning.ipynb`
**Goal**: Systematic parameter optimization

**Contents**:
- GridSearchCV deep dive
- RandomizedSearchCV for large spaces
- Bayesian optimization (sklearn-optimize)
- Visualizing search results
- Best practices for hyperparameter tuning

---

### Notebook 11: `11_model_comparison_and_selection.ipynb`
**Goal**: Compare models systematically

**Contents**:
- Compare Linear, Tree, SVM, MLP on same data
- Statistical significance testing
- Ensemble methods (VotingClassifier, Stacking)
- When to use which model

---

### Notebook 12: `12_complete_ml_pipeline.ipynb`
**Goal**: End-to-end ML workflow

**Contents**:
- Data loading and exploration
- Preprocessing pipeline
- Model selection
- Hyperparameter optimization
- Final evaluation
- Model persistence (joblib)

---

## Recommended Learning Path

### Week 1-2: Foundations
- Complete Notebooks 1-3
- Practice: Generate 5 different synthetic datasets

### Week 3-4: Classical Models
- Complete Notebooks 4-6
- Practice: Parameter sweeps on each model type

### Week 5-6: MLP Deep Dive ⭐
- Complete Notebooks 7-9
- Practice: Full parameter space exploration
- Goal: Achieve >95% accuracy on synthetic classification

### Week 7-8: Advanced & Integration
- Complete Notebooks 10-12
- Practice: Build complete pipeline for a real dataset

---

## Key Libraries to Install

```bash
pip install numpy pandas matplotlib seaborn scikit-learn ipywidgets jupyter
```

For enhanced visualizations:
```bash
pip install plotly yellowbrick
```

---

## Quick Start: Your First Experiment

Create `00_quick_start.ipynb` with this code to get started immediately:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# 1. Generate Data
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# 2. Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 3. Define Parameter Space
param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (20, 10)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01]
}

# 4. Grid Search
mlp = MLPClassifier(max_iter=500, random_state=42)
grid_search = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 5. Results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.4f}")

# 6. Visualize Decision Boundary
def plot_decision_boundary(model, X, y):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.title('MLP Decision Boundary')

plt.figure(figsize=(10, 6))
plot_decision_boundary(grid_search.best_estimator_, X_scaled, y)
plt.show()
```

---

## Success Metrics

By the end of this curriculum, you should be able to:

1. **Generate** synthetic data with controlled parameters
2. **Visualize** how data parameters affect model outcomes
3. **Train** MLP models with various architectures
4. **Explore** parameter spaces systematically
5. **Evaluate** models using appropriate metrics
6. **Select** optimal hyperparameters using grid/random search
7. **Build** complete ML pipelines

---

## Next Steps

After completing this plan:
1. Apply to real datasets (UCI ML Repository, Kaggle)
2. Explore deep learning with TensorFlow/PyTorch
3. Study specific domains (NLP, Computer Vision, Time Series)

---

*Happy Learning!*
