# Machine Learning Playground

機械学習の基礎から実践まで学べる、包括的な日本語教育用リポジトリです。

## 📚 概要

このリポジトリには、scikit-learnを使った機械学習の学習教材が含まれています。
日本語の教科書フレームワークに基づき、初心者でも体系的に学べるように設計されています。

### 特徴

- ✅ **29個の包括的なノートブック**: 基礎から実践Kaggle競技まで網羅
- ✅ **詳細な日本語説明**: 10,000文字以上の解説
- ✅ **豊富なコード コメント**: 200行以上の詳細な説明
- ✅ **実世界の応用例**: Kaggleコンペティション実践
- ✅ **自己評価クイズ**: 理解度を確認
- ✅ **よくあるエラー解説**: トラブルシューティング
- ✅ **GBDT完全マスター**: LightGBM、XGBoost、CatBoost
- ✅ **Kaggle Top 30%達成**: 実践的なテクニック

## 🗂️ ディレクトリ構造

```
machine-learning-playground/
├── notebooks/                         # 📓 Jupyter Notebooks（学習教材）
│   ├── 00_quick_start_improved_v2.ipynb
│   ├── 01_data_simulation_basics_improved_v2.ipynb
│   ├── 02_preprocessing_and_feature_engineering_improved_v2.ipynb
│   ├── 03_model_evaluation_metrics_improved_v2.ipynb
│   ├── 04_linear_models_simulation_improved_v2.ipynb
│   ├── 05_tree_and_ensemble_models_improved_v2.ipynb
│   ├── 06_svm_and_kernels_improved_v2.ipynb
│   ├── 07_mlp_fundamentals_improved_v2.ipynb
│   ├── 08_mlp_parameter_space_exploration_improved_v2.ipynb
│   ├── 09_mlp_regression_waveforms_improved_v2.ipynb
│   ├── 10_automated_hyperparameter_tuning_improved_v2.ipynb
│   ├── 11_model_comparison_and_selection_improved_v2.ipynb
│   └── 12_complete_ml_pipeline_improved_v2.ipynb
│
├── scripts/                           # 🔧 ユーティリティスクリプト
│   ├── notebook_improvements/        # ノートブック改善用スクリプト
│   │   ├── README.md
│   │   ├── analyze_notebooks.py
│   │   ├── improve_all_notebooks.py
│   │   └── ...
│   │
│   └── examples/                     # サンプルスクリプト
│       ├── README.md
│       ├── linearRegression.py
│       ├── naivebayes.py
│       └── ...
│
├── ML_LEARNING_PLAN.md               # 📖 学習計画ガイド
└── README.md                         # このファイル
```

## 🚀 はじめに

### 前提条件

- Python 3.7以上
- Jupyter Notebook または JupyterLab

### インストール

```bash
# リポジトリのクローン
git clone https://github.com/yuzuponikemi/machine-learning-playground.git
cd machine-learning-playground

# 依存パッケージのインストール（推奨: 仮想環境を使用）
pip install -r requirements.txt

# Jupyter Notebookの起動
jupyter notebook notebooks/
```

## 📖 学習カリキュラム

### 初級コース（推定時間: 10-15時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 00 | クイックスタート | 機械学習の基本フロー | 30-45分 | ★☆☆☆☆ |
| 01 | データシミュレーション | 合成データの生成 | 60-90分 | ★☆☆☆☆ |
| 02 | 前処理と特徴量エンジニアリング | データの準備 | 90-120分 | ★★☆☆☆ |
| 03 | モデル評価指標 | 性能の測定方法 | 90-120分 | ★★☆☆☆ |

### 中級コース（推定時間: 15-20時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 04 | 線形モデル | 線形回帰とロジスティック回帰 | 120-150分 | ★★★☆☆ |
| 05 | 決定木とアンサンブル | ランダムフォレスト、勾配ブースティング | 120-150分 | ★★★☆☆ |
| 06 | SVMとカーネル | サポートベクターマシン | 120-150分 | ★★★☆☆ |
| 07 | MLP基礎 | ニューラルネットワーク入門 | 120-150分 | ★★★☆☆ |

### 上級コース（推定時間: 10-15時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 08 | MLPパラメータ探索 | ハイパーパラメータチューニング | 90-120分 | ★★★★☆ |
| 09 | MLP回帰 | 波形データの回帰問題 | 90-120分 | ★★★★☆ |
| 10 | 自動ハイパーパラメータ調整 | GridSearch、RandomSearch | 90-120分 | ★★★☆☆ |
| 11 | モデル比較と選択 | 複数モデルの比較手法 | 90-120分 | ★★★★☆ |
| 12 | 完全なMLパイプライン | エンドツーエンドの実装 | 120-150分 | ★★★★☆ |

### 🌳 GBDTマスターコース（推定時間: 15-20時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 13 | GBDT入門 | LightGBM、XGBoost基礎 | 120-150分 | ★★★☆☆ |
| 14 | CatBoost | カテゴリカル変数の処理 | 120-150分 | ★★★☆☆ |
| 15 | Titanic EDA | 特徴量エンジニアリング実践 | 120-150分 | ★★★★☆ |
| 16 | Titanic GBDT | モデリングとアンサンブル | 120-150分 | ★★★★☆ |

### 🏆 Kaggle実践コース（推定時間: 25-30時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 17 | Titanic Top 30% | 高度な特徴量とアンサンブル | 180-240分 | ★★★★★ |
| 18 | House Prices回帰 | GBDT回帰問題の実践 | 180-240分 | ★★★★★ |
| 19 | Store Demand | 時系列予測×GBDT | 180-240分 | ★★★★★ |

### 🚀 高度なテクニックコース（推定時間: 20-25時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 20 | Optuna最適化 | 自動ハイパーパラメータ調整 | 150-180分 | ★★★★☆ |
| 21 | SHAPモデル解釈 | モデルの説明可能性 | 150-180分 | ★★★★☆ |
| 22 | Stackingアンサンブル | メタ学習の実践 | 150-180分 | ★★★★★ |

### 🎯 専門トピックコース（推定時間: 15-20時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 23 | 不均衡データ対策 | SMOTE、Focal Loss | 120-150分 | ★★★★☆ |
| 24 | 時系列特徴量 | ラグ、移動平均、周期性 | 120-150分 | ★★★★☆ |
| 25 | カテゴリカル変数 | Target Encoding、Embedding | 120-150分 | ★★★★☆ |

### 🎓 最終プロジェクトコース（推定時間: 20-30時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 26 | Tabularディープラーニング | TabNet、NODE | 150-180分 | ★★★★★ |
| 27 | Kaggle完全ワークフロー | コンペティション攻略法 | 180-240分 | ★★★★★ |
| 28 | 総合演習プロジェクト | 独自のML プロジェクト作成 | 300-400分 | ★★★★★ |

**合計推定時間**: 140-190時間（約3-6ヶ月の学習期間）

## 🎯 学習目標

このカリキュラムを完了すると、以下ができるようになります：

### 基礎スキル（ノートブック 0-12）
- ✅ 機械学習の基本的なワークフローを理解できる
- ✅ データの前処理と特徴量エンジニアリングができる
- ✅ 適切な評価指標を選択し、モデルを評価できる

### 実践スキル（ノートブック 0-12）
- ✅ 問題に応じた適切なアルゴリズムを選択できる
- ✅ ハイパーパラメータを調整して性能を最適化できる
- ✅ 過学習を検出し、対処できる

### 応用スキル（ノートブック 0-12）
- ✅ 複数のモデルを比較し、最適なものを選択できる
- ✅ エンドツーエンドの機械学習パイプラインを構築できる
- ✅ 実務で使える機械学習システムを設計できる

### GBDT専門スキル（ノートブック 13-22）
- ✅ LightGBM、XGBoost、CatBoostを自在に使いこなせる
- ✅ Optunaで自動ハイパーパラメータ最適化ができる
- ✅ SHAPでモデルの解釈と説明ができる
- ✅ Stackingでアンサンブル学習を実装できる

### Kaggle競技スキル（ノートブック 17-19, 27）
- ✅ **Titanic**: Top 30%以上のスコア達成（0.79+）
- ✅ **House Prices**: Top 20%以上のスコア達成（RMSLE 0.13以下）
- ✅ **Store Demand**: Top 25%以上のスコア達成（SMAPE 15%以下）
- ✅ コンペティション全体の戦略立案と実行ができる

### 専門技術スキル（ノートブック 23-28）
- ✅ 不均衡データの処理ができる（SMOTE、Focal Loss）
- ✅ 時系列データの特徴量エンジニアリングができる
- ✅ カテゴリカル変数の高度なエンコーディングができる
- ✅ Tabularデータのディープラーニングモデルを使える
- ✅ 完全なMLプロジェクトをポートフォリオとして作成できる

## 💡 使い方のヒント

### 効果的な学習方法

1. **順番に学習**: ノートブックは番号順に進めることを推奨
2. **手を動かす**: コードを実際に実行して結果を確認
3. **クイズに挑戦**: 各章末の自己評価クイズで理解度を確認
4. **エラーを恐れない**: よくあるエラーとその解決法も記載されています

### コードの実行

```python
# 各セルは Shift + Enter で実行
# または、上部メニューの Cell > Run All で全セル実行
```

### 学習の進め方

```
1. 学習目標を確認 → 何を学ぶか明確にする
2. コードを実行 → 実際に動かしてみる
3. 説明を読む → なぜそうなるのか理解する
4. クイズに挑戦 → 理解度を確認する
5. 次の章へ → 着実にステップアップ
```

## 🛠️ 技術スタック

- **Python**: 3.7+
- **基礎ライブラリ**:
  - scikit-learn: 機械学習アルゴリズム
  - NumPy: 数値計算
  - Pandas: データ操作
  - Matplotlib/Seaborn: データ可視化

- **GBDT専門ライブラリ**:
  - LightGBM: 高速な勾配ブースティング
  - XGBoost: 高精度な勾配ブースティング
  - CatBoost: カテゴリカル変数に強い

- **最適化・解釈ツール**:
  - Optuna: 自動ハイパーパラメータ最適化
  - SHAP: モデル解釈と説明可能性
  - Imbalanced-learn: 不均衡データ処理

## 📚 参考資料

### 推奨書籍（基礎）
- "Hands-On Machine Learning" by Aurélien Géron
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "scikit-learn公式ドキュメント": https://scikit-learn.org

### 推奨書籍（GBDT・Kaggle）
- **「Kaggleで勝つデータ分析の技術」** 門脇大輔ほか（必読！）
- "Hands-On Gradient Boosting with XGBoost and scikit-learn"
- "Feature Engineering for Machine Learning" by Alice Zheng
- "Interpretable Machine Learning" by Christoph Molnar

### オンラインリソース（基礎）
- [Kaggle](https://www.kaggle.com): 実データで練習
- [UCI ML Repository](https://archive.ics.uci.edu/ml/): データセット
- [scikit-learn tutorials](https://scikit-learn.org/stable/tutorial/): 公式チュートリアル

### オンラインリソース（GBDT・Kaggle）
- **[Kaggle Competitions](https://www.kaggle.com/competitions)**: Titanic、House Pricesから始める
- **[Kaggle Notebooks](https://www.kaggle.com/code)**: Grandmaster解法を学ぶ
- [LightGBM公式](https://lightgbm.readthedocs.io/): パラメータ詳細
- [XGBoost公式](https://xgboost.readthedocs.io/): アルゴリズム解説
- [CatBoost公式](https://catboost.ai/): カテゴリカル処理
- [Optuna公式](https://optuna.org/): ハイパーパラメータ最適化
- [SHAP公式](https://shap.readthedocs.io/): モデル解釈

### コミュニティ
- **Reddit**: r/MachineLearning、r/kaggle
- **GitHub**: Kaggle Solutions（例：Kazuki Onodera氏のリポジトリ）
- **Discord/Slack**: Kaggle公式コミュニティ

## 🤝 コントリビューション

改善提案やバグ報告は、Issueまたはプルリクエストでお願いします。

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。

## 📞 お問い合わせ

質問や提案がある場合は、GitHubのIssueをご利用ください。

---

**Happy Learning! 🎓**

機械学習の世界へようこそ。このリポジトリがあなたの学習の助けになれば幸いです。
