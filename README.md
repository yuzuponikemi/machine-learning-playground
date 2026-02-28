# Machine Learning Playground

機械学習の基礎から最新の生成モデルまで学べる、包括的な日本語教育用リポジトリです。

## 📚 概要

このリポジトリには、機械学習の基礎から拡散モデル・画像生成AIまで、2つの学習コースが含まれています。
日本語の教科書フレームワークに基づき、初心者でも体系的に学べるように設計されています。

### 🎓 コース概要

#### 📊 機械学習基礎コース（Notebooks 00-28）
scikit-learnを使った機械学習の基礎から実践まで

#### 🎨 生成モデルコース（Notebooks 30-45）
PyTorchを使った拡散モデル・画像生成AIの理論と実装

### 特徴

- ✅ **133個の包括的なノートブック**: 基礎から言語モデリングまで網羅
- ✅ **詳細な日本語説明**: 10,000文字以上の解説
- ✅ **豊富なコード コメント**: 200行以上の詳細な説明
- ✅ **実世界の応用例**: Kaggleコンペティション実践
- ✅ **自己評価クイズ**: 理解度を確認
- ✅ **よくあるエラー解説**: トラブルシューティング
- ✅ **GBDT完全マスター**: LightGBM、XGBoost、CatBoost
- ✅ **Kaggle Top 30%達成**: 実践的なテクニック
- ✅ **拡散モデル実装**: DDPM、VAE、Stable Diffusion

## 🗂️ ディレクトリ構造

```
machine-learning-playground/
├── notebooks/                         # 📓 Jupyter Notebooks（学習教材）
│   ├── fundamentals/                 # 00-12: 機械学習基礎
│   │   ├── 00_quick_start_improved_v2.ipynb
│   │   ├── 01_data_simulation_basics_improved_v2.ipynb
│   │   └── ...
│   │
│   ├── gbdt/                         # 13-22: GBDT マスター
│   │   ├── 13_gbdt_introduction_improved_v2.ipynb
│   │   ├── 14_catboost_categorical_improved_v2.ipynb
│   │   └── ...
│   │
│   ├── advanced/                     # 23-28: 高度なトピック
│   │   ├── 23_imbalanced_data_handling_improved_v2.ipynb
│   │   ├── 24_time_series_feature_engineering_improved_v2.ipynb
│   │   └── ...
│   │
│   ├── generative/                   # 30-45: 生成モデル
│   │   ├── 30_probability_and_normal_distribution_v1.ipynb
│   │   ├── 31_maximum_likelihood_estimation_v1.ipynb
│   │   └── ... (拡散モデル、VAE、Stable Diffusion)
│   │
│   ├── 3d-vision/                    # 50-63: 3Dビジョン
│   │   ├── 50_optics_fundamentals_v1.ipynb
│   │   └── ... (カメラモデル、SfM、NeRF)
│   │
│   ├── neural-engine/                # 70-76: ニューラルエンジン
│   │   ├── 70_derivative_intuition_v1.ipynb
│   │   └── ... (逆伝播、計算グラフ)
│   │
│   ├── spatial-cnn/                  # 80-102: 空間CNN
│   │   ├── 80_what_is_convolution_v1.ipynb
│   │   └── ... (CNN、ViT、セグメンテーション)
│   │
│   ├── H_optimization/              # 110-116: 最適化
│   │   ├── 110_gradient_descent_fundamentals_v1.ipynb
│   │   └── ... (Adam、学習率スケジューリング)
│   │
│   ├── sequence-models/             # 120-126: シーケンスモデリング 🆕
│   │   ├── 120_sequence_modeling_intro_v1.ipynb
│   │   ├── 121_vanilla_rnn_v1.ipynb
│   │   └── ... (BPTT, LSTM, GRU, Seq2Seq, Attention)
│   │
│   ├── spatiotemporal/              # 130-136: 時空間モデリング
│   │   ├── 130_temporal_attention_fundamentals_v1.ipynb
│   │   ├── 131_video_diffusion_models_v1.ipynb
│   │   └── ... (DiT、物理動画生成)
│   │
│   ├── world-models/                # 140-146: 世界モデル
│   │   ├── 140_representation_learning_for_prediction_v1.ipynb
│   │   ├── 141_jepa_joint_embedding_predictive_v1.ipynb
│   │   └── ... (DreamerV3、Genie、GridWorld)
│   │
│   └── embeddings/                  # 150-157: 埋め込み 🆕
│       ├── 150_embedding_geometry_v1.ipynb
│       ├── 151_word2vec_static_embeddings_v1.ipynb
│       ├── 152_contextual_embeddings_v1.ipynb
│       └── ... (文埋め込み、可視化、ベクトル検索、距離学習)
│
├── scripts/                           # 🔧 ユーティリティスクリプト
│   ├── notebook_improvements/        # ノートブック改善用スクリプト
│   └── examples/                     # サンプルスクリプト
│
├── ML_LEARNING_PLAN.md               # 📖 機械学習基礎の学習計画
├── DIFFUSION_MODELS_CURRICULUM.md   # 🎨 生成モデルカリキュラム 🆕
├── NOTEBOOK_GUIDELINES.md            # 📋 ノートブック作成ガイドライン
├── requirements.txt                   # 機械学習基礎用パッケージ
├── requirements-generative.txt       # 生成モデル用パッケージ 🆕
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

# 機械学習基礎コース用パッケージのインストール
pip install -r requirements.txt

# 生成モデルコース用パッケージのインストール（オプション）
pip install -r requirements-generative.txt

# Jupyter Notebookの起動
jupyter notebook notebooks/
```

### 推奨される学習順序

1. **機械学習基礎コース**（Notebooks 00-28）から開始
2. PyTorchの基礎を理解したら **生成モデルコース**（Notebooks 30-45）へ進む

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

**機械学習基礎コース合計推定時間**: 140-190時間（約3-6ヶ月）

---

## 🎨 生成モデルコース（Notebooks 30-45）

### 前提知識
- 機械学習基礎コース（Notebooks 00-12）の完了推奨
- Pythonプログラミングの基礎
- 基礎的な数学（微分、行列計算、確率）

### Phase 1: 確率統計の基礎（推定時間: 10-12時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 30 | 正規分布と確率の基礎 | 確率分布、中心極限定理 | 120-150分 | ★★☆☆☆ |
| 31 | 最尤推定 | 生成モデルの基礎 | 120-150分 | ★★★☆☆ |
| 32 | 多次元正規分布 | 共分散行列、可視化 | 120-150分 | ★★★☆☆ |

### Phase 2: 混合モデルと最適化（推定時間: 8-10時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 33 | 混合ガウスモデル（GMM） | 多峰性分布のモデリング | 120-150分 | ★★★☆☆ |
| 34 | EMアルゴリズム | ELBO、KLダイバージェンス | 150-180分 | ★★★★☆ |

### Phase 3: PyTorchとNN（推定時間: 8-10時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 35 | PyTorch基礎と勾配法 | テンソル演算、自動微分 | 120-150分 | ★★☆☆☆ |
| 36 | ニューラルネットワーク | MLP、MNIST分類 | 120-150分 | ★★★☆☆ |

### Phase 4: 変分オートエンコーダ（推定時間: 6-8時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 37 | VAE理論 | エンコーダ・デコーダ、ELBO | 120-150分 | ★★★★☆ |
| 38 | VAE実装 | 潜在空間、画像生成 | 150-180分 | ★★★★☆ |

### Phase 5: 拡散モデル理論（推定時間: 8-10時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 39 | 拡散モデル基礎 | 拡散過程、逆拡散過程 | 150-180分 | ★★★★☆ |
| 40 | 拡散モデルELBO | ELBO導出、数式理解 | 180-240分 | ★★★★★ |

### Phase 6: 拡散モデル実装（推定時間: 10-12時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 41 | U-Netと位置エンコーディング | アーキテクチャ設計 | 150-180分 | ★★★★☆ |
| 42 | 拡散モデル実装 | DDPM、ノイズ予測 | 180-240分 | ★★★★★ |
| 43 | 拡散モデル訓練 | 最適化、評価指標 | 180-240分 | ★★★★★ |

### Phase 7: 応用（推定時間: 8-10時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 44 | 条件付き拡散モデル | クラス条件付き生成 | 150-180分 | ★★★★☆ |
| 45 | Stable Diffusion | CFG、Diffusersライブラリ | 180-240分 | ★★★★★ |

**生成モデルコース合計推定時間**: 58-72時間（約2-3ヶ月）

**全コース合計**: 198-262時間（約5-9ヶ月の学習期間）

詳細は [DIFFUSION_MODELS_CURRICULUM.md](./DIFFUSION_MODELS_CURRICULUM.md) を参照

---

## 🎬 時空間モデリングコース（Notebooks 130-136） 🆕

### 前提知識
- 生成モデルコース（Notebooks 30-45）の完了推奨
- 3Dビジョンコース（Notebooks 50-63）の基礎知識
- ViT/Self-Attentionの理解（Notebook 95）

### Phase 6: 時空間・動画生成（推定時間: 15-20時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 130 | 時間的注意機構の基礎 | Temporal Attention, Causal Mask | 120-150分 | ★★★☆☆ |
| 131 | Video Diffusion Models | U-Net temporal拡張, Moving-MNIST | 150-180分 | ★★★★☆ |
| 132 | Diffusion Transformer (DiT) | パッチ埋め込み, adaLN-Zero | 150-180分 | ★★★★☆ |
| 133 | カメラと物体の運動分離 | Plücker座標, オプティカルフロー | 120-150分 | ★★★★☆ |
| 134 | 時間的一貫性の技術 | Temporal Super-Resolution, FVD | 120-150分 | ★★★★☆ |
| 135 | 物理動画生成 (Capstone) | 物理シミュレーション+DiT | 240-300分 | ★★★★★ |
| 136 | 時空間モデリング総括 | 技術体系整理, Sora解説 | 90-120分 | ★★★★☆ |

---

## 🌍 世界モデルコース（Notebooks 140-146） 🆕

### 前提知識
- 時空間モデリングコース（Notebooks 130-136）の完了推奨
- PyTorch基礎（Notebook 35-36）
- 最適化手法（Notebooks 110-116）

### Phase 7: 世界モデル・行動（推定時間: 18-24時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 140 | 予測のための表現学習 | 対照学習, InfoNCE, t-SNE | 120-150分 | ★★★☆☆ |
| 141 | JEPA | Joint Embedding Predictive, EMA | 150-180分 | ★★★★☆ |
| 142 | モデルベースRL基礎 | Dyna-Q, GridWorld | 150-180分 | ★★★★☆ |
| 143 | DreamerV3 | RSSM, 想像内学習 | 180-240分 | ★★★★★ |
| 144 | Genie | 潜在行動発見, VQ-VAE | 150-180分 | ★★★★★ |
| 145 | GridWorldエージェント (Capstone) | 世界モデル+MPC計画 | 300-360分 | ★★★★★ |
| 146 | 世界モデル総括 | 全Phase統合, AGI展望 | 90-120分 | ★★★★☆ |

**全コース合計**: 約350-440時間（約8-14ヶ月の学習期間）

---

## 🔄 シーケンスモデリングコース（Notebooks 120-126） 🆕

### 前提知識
- ニューラルエンジンコース（Notebooks 70-76）の完了推奨
- PyTorch基礎（Notebook 35-36、125-126で使用）

### Phase: シーケンスモデリング（推定時間: 15-20時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 120 | シーケンスとは何か | 静的vs系列データ、MLPの限界、状態の概念 | 60-90分 | ★★☆☆☆ |
| 121 | バニラRNN | 状態方程式、重み共有、forward実装 | 120-150分 | ★★★☆☆ |
| 122 | BPTT | 時間方向の逆伝播、勾配消失/爆発、Adding Problem | 150-180分 | ★★★★☆ |
| 123 | LSTM | 4ゲート、セル状態の勾配高速道路 | 150-180分 | ★★★★☆ |
| 124 | GRUと時系列予測 | GRU設計、三者比較、多ステップ予測 | 120-150分 | ★★★☆☆ |
| 125 | Seq2Seq | Encoder-Decoder、Teacher Forcing | 150-180分 | ★★★★☆ |
| 126 | Attention | Bahdanau Attention、Transformerへの架け橋 | 150-180分 | ★★★★☆ |

---

## 🔤 埋め込みコース（Notebooks 150-157）

### 前提知識
- ニューラルエンジンコース（Notebooks 70-76）の完了推奨
- PyTorch基礎（Notebook 35-36）
- 対照学習の基礎（Notebook 140）があると望ましい

### Phase 8: 埋め込み（推定時間: 15-20時間）

| # | ノートブック | 内容 | 時間 | 難易度 |
|---|------------|------|------|--------|
| 150 | 埋め込みの幾何学 | コサイン類似度, ユークリッド距離, 高次元の呪い | 90-120分 | ★★☆☆☆ |
| 151 | Word2Vec と静的埋め込み | Skip-gram/CBOW実装, 負例サンプリング, FastText/GloVe比較 | 120-150分 | ★★★☆☆ |
| 152 | 文脈付き埋め込み | BERT層別埋め込み抽出, 多義語可視化, Attention分析 | 120-150分 | ★★★★☆ |
| 153 | 文・文書の埋め込み | sentence-transformers, プーリング戦略比較 | 90-120分 | ★★★☆☆ |
| 154 | 多様体学習と可視化 | PCA → t-SNE → UMAP, パラメータ影響の体感 | 120-150分 | ★★★☆☆ |
| 155 | ベクトル検索とインデックス | FAISS, IVF, HNSW, 速度vs精度 | 90-120分 | ★★★☆☆ |
| 156 | 距離学習とファインチューニング | Triplet Loss, Contrastive Loss, ドメイン特化 | 120-150分 | ★★★★☆ |
| 157 | 埋め込みの応用と統合 | RAG, クラスタリング, バイアス分析, 多言語 | 90-120分 | ★★★☆☆ |

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

### 生成モデルスキル（ノートブック 30-45）
- ✅ 確率統計の基礎から生成モデルまでの流れを理解している
- ✅ PyTorchでニューラルネットワークを実装できる
- ✅ VAEを実装し、潜在空間を探索できる
- ✅ 拡散モデルをスクラッチで実装できる
- ✅ U-Netアーキテクチャを理解し実装できる
- ✅ 条件付き拡散モデルで画像を生成できる
- ✅ Stable Diffusionの仕組みを理解し、使いこなせる
- ✅ ELBO、KLダイバージェンスの数式を導出できる

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

### 機械学習基礎コース
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

### 生成モデルコース 🆕
- **Python**: 3.8+
- **ディープラーニングフレームワーク**:
  - PyTorch: ディープラーニングの基盤
  - torchvision: コンピュータビジョン

- **生成モデルライブラリ**:
  - Diffusers: 拡散モデル（Stable Diffusion等）
  - Transformers: CLIP、テキストエンコーダ
  - accelerate: 訓練の高速化

- **可視化・ユーティリティ**:
  - Matplotlib/Seaborn: データ可視化
  - Pillow/OpenCV: 画像処理
  - tqdm: プログレスバー

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

### オンラインリソース（生成モデル） 🆕
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/): 拡散モデルの公式ドキュメント
- [Lil'Log - Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/): 理論解説
- [Annotated Diffusion](https://huggingface.co/blog/annotated-diffusion): コード付き解説
- [PyTorch Tutorials](https://pytorch.org/tutorials/): 公式チュートリアル

### 推奨論文（生成モデル） 🆕
- **DDPM (2020)**: "Denoising Diffusion Probabilistic Models" - Ho et al.
- **Improved DDPM (2021)**: "Improved Denoising Diffusion Probabilistic Models"
- **Latent Diffusion (2022)**: "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion)
- **Classifier-Free Guidance (2022)**: "Classifier-Free Diffusion Guidance"

### コミュニティ
- **Reddit**: r/MachineLearning、r/kaggle、r/StableDiffusion
- **GitHub**: Kaggle Solutions、Diffusion Models実装例
- **Discord/Slack**: Kaggle公式、Hugging Face コミュニティ

## 🤝 コントリビューション

改善提案やバグ報告は、Issueまたはプルリクエストでお願いします。

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。

## 📞 お問い合わせ

質問や提案がある場合は、GitHubのIssueをご利用ください。

---

**Happy Learning! 🎓**

機械学習の世界へようこそ。このリポジトリがあなたの学習の助けになれば幸いです。
