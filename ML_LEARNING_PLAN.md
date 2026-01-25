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

---

## Phase 5: GBDT Master Course (Notebooks 13-16) 🌳

### Notebook 13: `13_gbdt_introduction.ipynb`
**Goal**: Master LightGBM and XGBoost fundamentals

**Contents**:
- GBDT basic concepts (boosting, trees, gradients)
- LightGBM architecture and parameters
- XGBoost implementation
- Key hyperparameters: learning_rate, num_leaves, max_depth
- Overfitting detection and early stopping

**Key Skills**:
- lightgbm, xgboost libraries
- Understanding tree-based boosting
- Feature importance visualization

---

### Notebook 14: `14_catboost_categorical.ipynb`
**Goal**: Master categorical feature handling with CatBoost

**Contents**:
- CatBoost's categorical feature processing
- Comparison: LightGBM vs XGBoost vs CatBoost
- Ordered boosting mechanism
- GPU acceleration
- Handling high-cardinality categoricals

**Key Skills**:
- catboost library
- Categorical encoding strategies
- Performance optimization

---

### Notebook 15: `15_titanic_eda_feature_engineering.ipynb`
**Goal**: Professional EDA and feature engineering

**Contents**:
- Comprehensive Titanic dataset EDA
- Missing value imputation strategies
- Feature creation: Title extraction, Family size, Fare bins
- Feature interaction engineering
- Correlation analysis and feature selection

**Key Skills**:
- pandas profiling
- Advanced feature engineering
- Domain knowledge application

---

### Notebook 16: `16_titanic_gbdt_modeling.ipynb`
**Goal**: GBDT modeling with cross-validation and ensemble

**Contents**:
- Stratified K-Fold cross-validation
- LightGBM, XGBoost, CatBoost training
- Hyperparameter tuning basics
- Model ensemble (averaging, weighted voting)
- Kaggle submission preparation

**Key Skills**:
- Cross-validation strategies
- Ensemble methods
- Kaggle workflow

---

## Phase 6: Kaggle Competition Practice (Notebooks 17-19) 🏆

### Notebook 17: `17_titanic_top30_submission.ipynb` ⭐⭐⭐
**Goal**: Achieve Kaggle Titanic Top 30% (Target: 0.79+ accuracy)

**Contents**:
- **Advanced Feature Engineering**:
  - Ticket prefix extraction (class indicator)
  - Cabin deck analysis
  - Name length and rare names
  - Age imputation with ML models
  - Feature polynomial combinations

- **Advanced Model Tuning**:
  - Optuna for hyperparameter optimization (preview)
  - 10-fold Stratified CV for robust validation
  - Out-of-fold predictions
  - Pseudo-labeling from test set

- **Ensemble Strategy**:
  - 3-model weighted ensemble (LightGBM + XGBoost + CatBoost)
  - Blending vs Stacking comparison
  - Calibration (Platt scaling)

- **Submission Strategy**:
  - Multiple submission versions
  - Leaderboard probing techniques
  - Score variance analysis

**Target Performance**:
- Local CV: 0.85+
- Public LB: 0.79+ (Top 30%)

---

### Notebook 18: `18_house_prices_regression.ipynb` ⭐⭐⭐
**Goal**: Master regression with GBDT on House Prices competition

**Contents**:
- **Regression-Specific Challenges**:
  - Target transformation (log, Box-Cox)
  - Outlier detection and handling (IsolationForest, Z-score)
  - Skewness correction
  - Heavy-tailed distributions

- **Advanced Feature Engineering**:
  - 200+ features → feature selection
  - Polynomial features for numerical variables
  - Neighborhood encoding (target encoding, frequency)
  - Time-based features (YearBuilt, YearRemodeled)

- **GBDT for Regression**:
  - Objective functions: rmse, mae, huber
  - Quantile regression for uncertainty
  - Feature interactions in trees

- **Model Evaluation**:
  - RMSLE (Root Mean Squared Log Error)
  - Residual analysis
  - Prediction interval estimation

**Target Performance**:
- Local CV RMSLE: 0.12-0.13
- Public LB: Top 20%

**Datasets**: Kaggle House Prices Competition

---

### Notebook 19: `19_store_demand_timeseries.ipynb` ⭐⭐⭐
**Goal**: Time series forecasting with GBDT

**Contents**:
- **Time Series Fundamentals**:
  - Trend, seasonality, residuals decomposition
  - Autocorrelation (ACF/PACF)
  - Train/validation temporal split (NO shuffling!)

- **Time Series Features for GBDT**:
  - Lag features (1, 7, 14, 28, 365 days)
  - Rolling window statistics (mean, std, min, max)
  - Expanding window features
  - Day of week, month, year, holidays
  - Fourier features for seasonality

- **GBDT Time Series Modeling**:
  - Walk-forward validation
  - Multi-step forecasting
  - Handling zero-inflated data
  - Store-item hierarchy modeling

- **Advanced Techniques**:
  - Exogenous variables (promotions, events)
  - LGBM with dart booster for time series
  - Residual modeling

**Target Performance**:
- SMAPE < 15%
- Kaggle Store Item Demand: Top 25%

**Datasets**: Kaggle Store Item Demand Forecasting Challenge

---

## Phase 7: Advanced GBDT Techniques (Notebooks 20-22) 🚀

### Notebook 20: `20_optuna_hyperparameter_optimization.ipynb` ⭐⭐⭐
**Goal**: Master automated hyperparameter tuning with Optuna

**Contents**:
- **Optuna Fundamentals**:
  - Tree-structured Parzen Estimator (TPE)
  - Bayesian optimization vs Grid/Random search
  - Study object and trial management
  - Pruning unpromising trials

- **LightGBM + Optuna**:
  - Complete parameter space definition
  - Categorical parameters (boosting_type, objective)
  - Integer parameters (num_leaves, max_depth)
  - Float parameters (learning_rate, feature_fraction)
  - Conditional parameters

- **Advanced Optimization**:
  - Multi-objective optimization (accuracy + inference time)
  - Parallel optimization (multiple workers)
  - Visualization (optimization history, parameter importance)
  - Saving/loading studies

- **Integration with CV**:
  - Cross-validation inside Optuna objective
  - Early stopping with callbacks
  - Best model selection

**Complete Example**:
```python
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }

    model = LGBMClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    return score

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best params: {study.best_params}")
print(f"Best score: {study.best_value}")

# Visualization
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

**Datasets**: Titanic, House Prices (apply to previous competitions)

---

### Notebook 21: `21_shap_model_interpretation.ipynb` ⭐⭐⭐
**Goal**: Master model interpretation with SHAP

**Contents**:
- **SHAP Fundamentals**:
  - Shapley values from game theory
  - TreeSHAP for tree-based models
  - Local vs global interpretability
  - Feature attribution

- **SHAP Visualizations**:
  - Waterfall plots (single prediction explanation)
  - Force plots (interactive explanations)
  - Summary plots (global feature importance)
  - Dependence plots (feature interactions)
  - Decision plots (multi-class classification)

- **Practical Applications**:
  - Debugging model errors
  - Finding spurious correlations
  - Feature selection guided by SHAP
  - Explaining predictions to stakeholders
  - Detecting bias in models

- **Advanced SHAP**:
  - Clustering SHAP values
  - SHAP interaction values
  - Comparing SHAP across models
  - SHAP for regression vs classification

**Complete Example**:
```python
import shap
import lightgbm as lgb

# Train model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global importance
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)

# Local explanation for single prediction
idx = 0
shap.waterfall_plot(shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    data=X_test.iloc[idx],
    feature_names=X_test.columns.tolist()
))

# Feature interactions
shap.dependence_plot("Age", shap_values, X_test, interaction_index="Sex")
```

**Datasets**: Titanic (interpret predictions), Credit Default (bias detection)

---

### Notebook 22: `22_stacking_meta_learning.ipynb` ⭐⭐⭐
**Goal**: Master stacking and meta-learning ensemble

**Contents**:
- **Stacking Fundamentals**:
  - Multi-level ensemble architecture
  - Base models (level 0) and meta-model (level 1)
  - Out-of-fold predictions to avoid overfitting
  - Diversity in base models

- **Implementation**:
  - Manual stacking with cross-validation
  - sklearn StackingClassifier/StackingRegressor
  - mlxtend StackingCVClassifier

- **Base Model Selection**:
  - LightGBM (fast, accurate)
  - XGBoost (robust)
  - CatBoost (categorical handling)
  - Neural networks (MLP)
  - Linear models (for diversity)

- **Meta-Model Selection**:
  - Logistic Regression (simple, fast)
  - LightGBM (capture interactions)
  - Ridge/Lasso (regularized linear)

- **Advanced Stacking**:
  - Multi-level stacking (3+ levels)
  - Feature engineering for meta-model
  - Including original features in meta-model
  - Blending vs Stacking comparison

**Complete Example**:
```python
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

# Define base models
base_models = [
    ('lgb', LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42)),
    ('cat', CatBoostClassifier(iterations=100, learning_rate=0.05, random_state=42, verbose=0)),
]

# Define meta-model
meta_model = LogisticRegression()

# Create stacking ensemble
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,  # Out-of-fold predictions
    stack_method='predict_proba',  # Use probabilities
    n_jobs=-1
)

# Train
stacking_clf.fit(X_train, y_train)

# Evaluate
print(f"Stacking CV Score: {stacking_clf.score(X_val, y_val)}")

# Compare with base models
for name, model in base_models:
    model.fit(X_train, y_train)
    print(f"{name} Score: {model.score(X_val, y_val)}")
```

**Datasets**: Titanic, House Prices (boost scores by 1-2%)

---

## Phase 8: Specialized Topics (Notebooks 23-25) 🎯

### Notebook 23: `23_imbalanced_data_strategies.ipynb`
**Goal**: Handle imbalanced datasets effectively

**Contents**:
- **Imbalance Detection**:
  - Class distribution analysis
  - Stratified sampling importance

- **Sampling Techniques**:
  - Oversampling (SMOTE, ADASYN)
  - Undersampling (RandomUnderSampler, Tomek Links)
  - Hybrid (SMOTEENN, SMOTETomek)

- **GBDT-Specific Techniques**:
  - scale_pos_weight (XGBoost)
  - is_unbalance parameter (LightGBM)
  - class_weights parameter (CatBoost)
  - Focal Loss for extreme imbalance

- **Evaluation**:
  - Precision-Recall curves (NOT ROC for imbalanced)
  - F1-Score, F-beta score
  - Confusion matrix analysis

**Datasets**: Credit Card Fraud Detection (1% fraud rate)

---

### Notebook 24: `24_timeseries_feature_engineering.ipynb`
**Goal**: Advanced time series feature creation

**Contents**:
- **Temporal Features**:
  - Cyclical encoding (sin/cos for month, day)
  - Holiday indicators
  - Business day flags

- **Lag Features**:
  - Multiple lag windows
  - Lag feature selection (PACF)

- **Rolling Statistics**:
  - Moving averages (7-day, 30-day, 90-day)
  - Rolling std, min, max, quantiles
  - Exponentially weighted moving average (EWMA)

- **Domain-Specific Features**:
  - Promotional calendars
  - Event-based features
  - External variables (weather, economic indicators)

**Datasets**: Rossmann Store Sales, M5 Forecasting

---

### Notebook 25: `25_categorical_encoding_advanced.ipynb`
**Goal**: Master categorical variable encoding

**Contents**:
- **Traditional Encoding**:
  - One-Hot Encoding (low cardinality)
  - Label Encoding (ordinal)

- **Advanced Encoding**:
  - Target Encoding (mean encoding)
  - Frequency Encoding
  - Binary Encoding
  - Hash Encoding (high cardinality)

- **Embedding Methods**:
  - Entity Embeddings for categorical features
  - Category Encoding library

- **Handling High Cardinality**:
  - Rare category grouping
  - Hierarchical encoding
  - CatBoost's native handling vs manual encoding

**Datasets**: Avazu Click-Through Rate Prediction (high cardinality)

---

## Phase 9: Final Projects & Integration (Notebooks 26-28) 🎓

### Notebook 26: `26_tabular_deep_learning.ipynb`
**Goal**: Explore deep learning for tabular data

**Contents**:
- **TabNet**:
  - Attention-based architecture
  - Interpretability with masks
  - Comparison with GBDT

- **Neural Oblivious Decision Ensembles (NODE)**:
  - Differentiable trees
  - Hybrid approach

- **When to use DL vs GBDT**:
  - GBDT wins on most tabular data
  - DL advantages (multi-modal, transfer learning)

**Datasets**: Porto Seguro Safe Driver Prediction

---

### Notebook 27: `27_kaggle_competition_workflow.ipynb`
**Goal**: Complete competition workflow from start to finish

**Contents**:
- **Phase 1: Understanding**:
  - Read competition description and rules
  - Understand evaluation metric
  - Review starter kernels

- **Phase 2: EDA & Feature Engineering**:
  - Comprehensive EDA
  - Feature creation strategy
  - Validation strategy

- **Phase 3: Modeling**:
  - Baseline model
  - Hyperparameter tuning
  - Ensemble creation

- **Phase 4: Submission**:
  - Multiple submissions
  - Leaderboard analysis
  - Final model selection

**Practice Competition**: Choose from active Kaggle competitions

---

### Notebook 28: `28_your_project_comprehensive.ipynb` ⭐⭐⭐
**Goal**: Build your own complete ML project

**Contents**:
- **Project Ideas**:
  - Find dataset (UCI, Kaggle Datasets, government data)
  - Define business problem
  - Set evaluation criteria

- **Complete Pipeline**:
  - Data collection and cleaning
  - EDA and insights
  - Feature engineering
  - Model selection and tuning
  - Ensemble creation
  - Model interpretation (SHAP)
  - Deployment preparation

- **Documentation**:
  - README with project description
  - Jupyter notebook with clear sections
  - Model performance report

**Deliverable**: Complete portfolio project showcasing all learned skills

---

## Recommended Learning Path (Phases 5-9)

### Months 1-2: GBDT Foundations (Notebooks 13-16)
- Master LightGBM, XGBoost, CatBoost
- Practice: Titanic competition (baseline)

### Month 3: Kaggle Practice (Notebooks 17-19)
- Achieve Top 30% on Titanic
- Complete House Prices regression
- Tackle time series forecasting

### Month 4: Advanced Techniques (Notebooks 20-22)
- Master Optuna optimization
- Learn SHAP interpretation
- Build stacking ensembles

### Month 5: Specialized Topics (Notebooks 23-25)
- Handle imbalanced data
- Master time series features
- Advanced categorical encoding

### Month 6: Integration & Projects (Notebooks 26-28)
- Explore tabular DL
- Complete full Kaggle workflow
- Build final portfolio project

---

## Success Metrics (GBDT Phase)

By completing this curriculum, you will have:

✅ **Competition Skills**:
- Kaggle Titanic: Top 30% (0.79+ accuracy)
- Kaggle House Prices: Top 20% (RMSLE < 0.13)
- Store Demand: Top 25% (SMAPE < 15%)

✅ **Technical Mastery**:
- LightGBM, XGBoost, CatBoost proficiency
- Optuna hyperparameter optimization
- SHAP model interpretation
- Stacking ensembles
- Time series forecasting with GBDT

✅ **Portfolio**:
- 3+ Kaggle competition submissions
- 1 complete end-to-end project
- Model interpretation reports

---

## Additional Resources

### Books
- "Kaggleで勝つデータ分析の技術" by 門脇大輔 et al.
- "Hands-On Gradient Boosting with XGBoost and scikit-learn"
- "Feature Engineering for Machine Learning" by Alice Zheng

### Communities
- Kaggle Discussions and Kernels
- GitHub: Kaggle Solutions (e.g., Kazuki Onodera's repos)
- Reddit: r/MachineLearning, r/kaggle

### Tools
- Optuna: https://optuna.org/
- SHAP: https://shap.readthedocs.io/
- Category Encoders: https://contrib.scikit-learn.org/category_encoders/

---

## Next Steps

After completing this comprehensive plan:
1. **Compete regularly**: Enter 1-2 active Kaggle competitions per month
2. **Specialize**: Choose a domain (NLP, CV, Time Series, Recommender Systems)
3. **Contribute**: Share kernels, write blog posts, mentor others
4. **Advanced ML**: MLOps, model deployment, production systems
5. **Research**: Read papers, implement state-of-the-art methods

---

## Phase 10: Optimization Methods (H_optimization/) 🎯

この単元では、機械学習で広く使われるブラックボックス最適化手法の理論と実装を学びます。
Notebook 20（Optunaハイパーパラメータ最適化）の前提知識として推奨されます。

> **📁 命名規則（新方式）**
> - 単元ディレクトリ: `H_optimization/`（プレフィックスで単元間の順序を管理）
> - ノートブック: 番号なし、内容を表す名前のみ
> - 順序: このカリキュラムファイルで一元管理

---

### `blackbox_intro.ipynb`
**Goal**: ブラックボックス最適化の全体像を理解する

**Contents**:
- **最適化問題の定式化**:
  - 目的関数、制約条件、探索空間
  - 連続最適化 vs 離散最適化
  - 勾配ベース vs 勾配フリー手法

- **ブラックボックス最適化の特徴**:
  - 目的関数の内部構造が不明
  - 勾配情報が利用できない
  - 評価コストが高い場合の戦略

- **局所解 vs 大域解**:
  - 多峰性関数の可視化
  - 局所解に陥る問題
  - 大域最適化の困難さ

- **手法の分類と比較**:
  - 無情報探索（Grid/Random Search）
  - モデルベース手法（ベイズ最適化）
  - メタヒューリスティクス（進化計算、スウォーム知能）
  - 各手法の長所・短所・適用場面

**ベンチマーク関数**:
- Rastrigin関数、Rosenbrock関数、Ackley関数
- 2D可視化による関数特性の理解

---

### `simulated_annealing.ipynb`
**Goal**: 焼きなまし法の原理と実装をマスターする

**Contents**:
- **物理的アナロジー**:
  - 金属の焼きなまし過程
  - エネルギー最小化と最適化の対応

- **アルゴリズムの詳細**:
  - メトロポリス基準（Metropolis criterion）
  - 温度スケジュール（冷却率）
  - 近傍解の生成方法

- **探索と活用のバランス**:
  - 高温：広範囲探索（exploration）
  - 低温：局所改善（exploitation）
  - 温度減衰による段階的収束

- **実装と可視化**:
  - Pythonによるスクラッチ実装
  - 探索軌跡のアニメーション
  - 温度スケジュールの影響比較

- **パラメータチューニング**:
  - 初期温度の設定
  - 冷却率の選択
  - 停止条件の設計

**応用例**: 巡回セールスマン問題、スケジューリング問題

---

### `genetic_algorithm.ipynb`
**Goal**: 遺伝的アルゴリズムの原理と実装をマスターする

**Contents**:
- **生物学的アナロジー**:
  - 自然選択と適者生存
  - 遺伝子、染色体、個体、集団

- **基本オペレータ**:
  - 選択（Selection）：ルーレット選択、トーナメント選択、エリート選択
  - 交叉（Crossover）：一点交叉、二点交叉、一様交叉
  - 突然変異（Mutation）：ビット反転、ガウシアン変異

- **適応度関数**:
  - 目的関数から適応度への変換
  - スケーリング手法
  - 制約処理（ペナルティ法、修復法）

- **収束性と多様性**:
  - 早熟収束の問題
  - 多様性維持の技法
  - 探索と活用のバランス

- **実装と可視化**:
  - Pythonによるスクラッチ実装
  - 世代ごとの集団分布の可視化
  - 適応度の進化過程

**応用例**: 関数最適化、特徴選択、ニューラルネットワークの構造探索

---

### `particle_swarm.ipynb`
**Goal**: 粒子群最適化の原理と実装をマスターする

**Contents**:
- **群知能のアナロジー**:
  - 鳥の群れ、魚の群れの行動
  - 個体の局所情報と群の大域情報

- **アルゴリズムの詳細**:
  - 粒子の位置と速度
  - 個体最良（pbest）と群最良（gbest）
  - 速度更新式と位置更新式

- **パラメータの役割**:
  - 慣性重み（inertia weight）
  - 認知係数（cognitive coefficient）
  - 社会係数（social coefficient）
  - 速度制限（velocity clamping）

- **バリエーション**:
  - グローバルベスト vs ローカルベスト
  - 適応的パラメータ調整
  - QPSO（量子粒子群最適化）

- **実装と可視化**:
  - Pythonによるスクラッチ実装
  - 粒子の動きのアニメーション
  - 収束過程の可視化

**応用例**: 連続最適化、ニューラルネットワークの重み最適化

---

### `bayesian_optimization_gp.ipynb` ⭐⭐⭐
**Goal**: ガウス過程ベースのベイズ最適化を理解する

**Contents**:
- **ベイズ最適化の原理**:
  - サロゲートモデル（代理モデル）
  - 不確実性の定量化
  - 探索と活用のトレードオフ

- **ガウス過程（Gaussian Process）**:
  - ガウス過程回帰の基礎
  - カーネル関数（RBF、Matérn）
  - 平均関数と共分散関数
  - ハイパーパラメータの最適化

- **獲得関数（Acquisition Function）**:
  - Expected Improvement (EI)
  - Probability of Improvement (PI)
  - Upper Confidence Bound (UCB)
  - 獲得関数の可視化と比較

- **実装と可視化**:
  - scikit-optimizeによる実装
  - GPyOptによる実装
  - 獲得関数の変化のアニメーション
  - サロゲートモデルの更新過程

- **GPベースの限界**:
  - 計算量 O(n³) の問題
  - 高次元での性能劣化
  - → TPEへの動機付け

**前提知識**: 基礎的な確率・統計の知識

---

### `tpe_algorithm.ipynb` ⭐⭐⭐
**Goal**: TPE（Tree-structured Parzen Estimator）とDefine-by-Runを深く理解する

**Contents**:
- **TPEの動機**:
  - ガウス過程の限界を克服
  - Optunaのデフォルトアルゴリズム

- **Parzen推定（カーネル密度推定）**:
  - カーネル密度推定の基礎
  - バンド幅の選択
  - 可視化による直感的理解

- **TPEのアルゴリズム**:
  - 観測値を良い群 l(x) と悪い群 g(x) に分割
  - 分割閾値 γ（上位何%を「良い」とするか）
  - 各群の密度推定
  - 獲得関数: l(x)/g(x) の最大化

  ```
  良いスコアの領域を密に、悪いスコアの領域を疎にサンプリング

  l(x): 上位γ%の観測値から推定した密度
  g(x): 下位(1-γ)%の観測値から推定した密度

  次の探索点 = argmax l(x)/g(x)
  ```

- **GPとの比較**:
  | 観点 | Gaussian Process | TPE |
  |------|------------------|-----|
  | 計算量 | O(n³) | O(n log n) |
  | 高次元 | 苦手 | 比較的強い |
  | カテゴリ変数 | 扱いにくい | 自然に扱える |
  | 不確実性 | 明示的にモデル化 | 暗黙的 |

- **Tree-structuredの意味**:
  - 条件付きパラメータの扱い
  - パラメータ間の依存関係
  - 階層的な探索空間の表現

- **Define-by-Run（動的探索空間定義）**:
  - Define-and-Run vs Define-by-Run
    ```python
    # Define-and-Run（静的）: 事前に全パラメータを定義
    param_grid = {
        'model': ['RF', 'SVM'],
        'RF__n_estimators': [100, 200],  # SVMでも定義が必要
        'SVM__C': [0.1, 1.0],             # RFでも定義が必要
    }

    # Define-by-Run（動的）: 実行時に必要なパラメータだけ定義
    def objective(trial):
        model = trial.suggest_categorical('model', ['RF', 'SVM'])
        if model == 'RF':
            n_estimators = trial.suggest_int('n_estimators', 100, 500)
        else:  # SVM
            C = trial.suggest_float('C', 0.1, 10.0)
    ```
  - 条件分岐による柔軟な探索空間
  - TPEのTree構造との相性の良さ
  - 実行時の探索空間の動的構築

- **実装と可視化**:
  - Pythonによるスクラッチ実装
  - l(x)とg(x)の密度分布の可視化
  - 探索点の選択過程のアニメーション
  - 条件付き探索空間のDAG可視化

---

### `pruning_early_stopping.ipynb` ⭐⭐
**Goal**: 早期終了（Pruning）の理論と実装を理解する

**Contents**:
- **Pruningの動機**:
  - 計算リソースの効率的な配分
  - 見込みのない試行の早期打ち切り
  - バンディット問題との関連

- **Successive Halving**:
  - アルゴリズムの詳細
    ```
    ラウンド0: 全81試行を1エポック学習 → 上位1/3を残す（27試行）
    ラウンド1: 27試行を3エポック学習 → 上位1/3を残す（9試行）
    ラウンド2: 9試行を9エポック学習 → 上位1/3を残す（3試行）
    ラウンド3: 3試行を27エポック学習 → 最良の1試行を選択
    ```
  - 理論的背景（予算配分問題）
  - 計算量の削減効果

- **Hyperband**:
  - Successive Halvingの拡張
  - 複数のブラケット（早期打ち切りの強度）
  - 探索と活用のバランス
  - 理論的保証

- **Median Pruner**:
  - シンプルで実用的なPruner
  - 中央値との比較による打ち切り
  - n_startup_trials、n_warmup_stepsの意味

- **その他のPruner**:
  - Percentile Pruner
  - Threshold Pruner
  - Patient Pruner（改良版）

- **Pruningの実装パターン**:
  ```python
  def objective(trial):
      model = create_model(trial)
      for epoch in range(100):
          train_one_epoch(model)
          val_score = evaluate(model)

          # 中間結果を報告
          trial.report(val_score, epoch)

          # Pruning判定
          if trial.should_prune():
              raise optuna.TrialPruned()

      return val_score
  ```

- **Pruningの注意点**:
  - 学習曲線の性質（早期の性能が最終性能を予測できるか）
  - warmupの重要性
  - Pruneしすぎのリスク

- **可視化**:
  - Pruneされた試行の学習曲線
  - リソース配分の効率性の比較
  - Successive Halvingのアニメーション

**次のステップ**: Notebook 20（Optunaハイパーパラメータ最適化）

---

### `optimization_comparison.ipynb`
**Goal**: 各最適化手法を比較し、問題に応じた手法選択ができるようになる

**Contents**:
- **ベンチマーク実験**:
  - 複数のテスト関数での性能比較
  - 収束速度の比較
  - 解の品質の比較

- **問題特性と手法選択**:
  | 問題特性 | 推奨手法 |
  |----------|----------|
  | 低次元・評価コスト高 | ベイズ最適化（GP） |
  | 中次元・評価コスト高 | TPE |
  | 高次元・評価コスト低 | 粒子群最適化、遺伝的アルゴリズム |
  | 離散最適化 | 遺伝的アルゴリズム、焼きなまし法 |
  | ノイズの多い目的関数 | ベイズ最適化（ノイズ考慮） |
  | 並列評価可能 | 粒子群最適化、遺伝的アルゴリズム |
  | カテゴリ変数が多い | TPE |

- **ハイブリッド手法**:
  - メタヒューリスティクス + 局所探索
  - 複数手法の組み合わせ

- **実践的なガイドライン**:
  - 計算予算に応じた手法選択
  - ハイパーパラメータ最適化への応用
  - AutoML との関係

**まとめ**:
- 各手法の長所・短所の整理
- 実務での選択指針
- 今後の学習リソース

---

### Recommended Learning Path (Optimization Methods)

```
H_optimization/
├── blackbox_intro.ipynb            ← 1. 最適化の全体像
├── simulated_annealing.ipynb       ← 2. メタヒューリスティクス①
├── genetic_algorithm.ipynb         ← 3. メタヒューリスティクス②
├── particle_swarm.ipynb            ← 4. メタヒューリスティクス③
├── bayesian_optimization_gp.ipynb  ← 5. ベイズ最適化（GP）
├── tpe_algorithm.ipynb             ← 6. TPE + Define-by-Run ⭐
├── pruning_early_stopping.ipynb    ← 7. Pruning（早期終了）⭐
└── optimization_comparison.ipynb   ← 8. 手法比較と選択指針
```

**学習の流れ**:
1. **1-4**: 古典的なメタヒューリスティクス手法で最適化の基礎を固める
2. **5-6**: モデルベース最適化（GP → TPE）で現代的手法を学ぶ
3. **7**: 計算効率を上げるPruningの仕組みを理解
4. **8**: 全手法を比較し、問題に応じた選択ができるようになる

**所要時間**: 約3週間

**次のステップ**: Notebook 20（Optunaハイパーパラメータ最適化）で実践的なハイパーパラメータチューニングへ

---

*You now have a complete roadmap from ML basics to Kaggle Grandmaster-level skills. Let's build amazing models! 🚀*
