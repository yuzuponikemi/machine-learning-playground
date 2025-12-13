# Machine Learning Playground

機械学習の基礎から実践まで学べる、包括的な日本語教育用リポジトリです。

## 📚 概要

このリポジトリには、scikit-learnを使った機械学習の学習教材が含まれています。
日本語の教科書フレームワークに基づき、初心者でも体系的に学べるように設計されています。

### 特徴

- ✅ **13個の包括的なノートブック**: 基礎から応用まで網羅
- ✅ **詳細な日本語説明**: 10,000文字以上の解説
- ✅ **豊富なコード コメント**: 200行以上の詳細な説明
- ✅ **実世界の応用例**: 具体的なユースケース
- ✅ **自己評価クイズ**: 理解度を確認
- ✅ **よくあるエラー解説**: トラブルシューティング

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

**合計推定時間**: 35-50時間

## 🎯 学習目標

このカリキュラムを完了すると、以下ができるようになります：

### 基礎スキル
- ✅ 機械学習の基本的なワークフローを理解できる
- ✅ データの前処理と特徴量エンジニアリングができる
- ✅ 適切な評価指標を選択し、モデルを評価できる

### 実践スキル
- ✅ 問題に応じた適切なアルゴリズムを選択できる
- ✅ ハイパーパラメータを調整して性能を最適化できる
- ✅ 過学習を検出し、対処できる

### 応用スキル
- ✅ 複数のモデルを比較し、最適なものを選択できる
- ✅ エンドツーエンドの機械学習パイプラインを構築できる
- ✅ 実務で使える機械学習システムを設計できる

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
- **主要ライブラリ**:
  - scikit-learn: 機械学習アルゴリズム
  - NumPy: 数値計算
  - Pandas: データ操作
  - Matplotlib/Seaborn: データ可視化

## 📚 参考資料

### 推奨書籍
- "Hands-On Machine Learning" by Aurélien Géron
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "scikit-learn公式ドキュメント": https://scikit-learn.org

### オンラインリソース
- [Kaggle](https://www.kaggle.com): 実データで練習
- [UCI ML Repository](https://archive.ics.uci.edu/ml/): データセット
- [scikit-learn tutorials](https://scikit-learn.org/stable/tutorial/): 公式チュートリアル

## 🤝 コントリビューション

改善提案やバグ報告は、Issueまたはプルリクエストでお願いします。

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。

## 📞 お問い合わせ

質問や提案がある場合は、GitHubのIssueをご利用ください。

---

**Happy Learning! 🎓**

機械学習の世界へようこそ。このリポジトリがあなたの学習の助けになれば幸いです。
