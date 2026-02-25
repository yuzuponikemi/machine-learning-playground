# 総合シラバス — Machine Learning Playground

> **最終更新**: 2026-02-23
> **総ノートブック数**: 150
> **推定総学習時間**: 375-480時間

---

## 目次

1. [カリキュラム全体像](#カリキュラム全体像)
2. [Unit 1: ML基礎 (00-12)](#unit-1-ml基礎-fundamentals)
3. [Unit 2: GBDT (13-22)](#unit-2-gbdt)
4. [Unit 3: 専門トピック (23-28)](#unit-3-専門トピック-advanced)
5. [Unit 4: 生成モデル (30-45)](#unit-4-生成モデル-generative)
6. [Unit 5: 3Dビジョン (50-63)](#unit-5-3dビジョン-3d-vision)
7. [Unit 6: ニューラルエンジン (70-76)](#unit-6-ニューラルエンジン-neural-engine)
8. [Unit 7: 空間CNN (80-102)](#unit-7-空間cnn-spatial-cnn)
9. [Unit 8: 最適化手法 (110-116)](#unit-8-最適化手法-optimization)
10. [Unit 9: シーケンスモデリング (120-126)](#unit-9-シーケンスモデリング-sequence-models)
11. [Unit 10: 時空間モデリング (130-136)](#unit-10-時空間モデリング-spatiotemporal)
12. [Unit 11: 世界モデル (140-146)](#unit-11-世界モデル-world-models)
13. [Unit 12: 埋め込み (150-157)](#unit-12-埋め込み-embeddings)
14. [Unit 13: 言語モデリング (160-167)](#unit-13-言語モデリング-language-models)
15. [Unit 14: 画像変容と潜在空間 (300-316)](#unit-14-画像変容と潜在空間-image-morphing)
16. [キーワード索引](#キーワード索引)
17. [依存関係マップ](#依存関係マップ)

---

## カリキュラム全体像

```
ML基礎 (00-12)  ──→  GBDT (13-22)  ──→  専門トピック (23-28)
     │
     ├──→  生成モデル (30-45)  ──→  時空間 (130-136)  ──→  世界モデル (140-146)
     │                    │
     │                    └──→  画像変容 (300-316)
     │
     ├──→  3Dビジョン (50-63)
     │
     ├──→  ニューラルエンジン (70-76)  ──→  空間CNN (80-102)
     │                    │                    │
     │                    │                    └──→  最適化 (110-116)
     │                    │
     │                    └──→  シーケンス (120-126)  ──→  時空間 (130-136)
     │                                                ──→  言語モデリング (160-167)
     │
     └──→  埋め込み (150-157)  ──→  言語モデリング (160-167)
```

---

## Unit 1: ML基礎 (fundamentals/)

**対象**: プログラミング初心者〜ML初学者
**推定時間**: 20-30時間
**使用ライブラリ**: scikit-learn, NumPy, Pandas, Matplotlib, Seaborn

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 00 | クイックスタート: 最初のMLP実験 | ★☆☆☆☆ | 30分 | ML基本フロー, make_moons, MLPClassifier, GridSearchCV, 決定境界 |
| 01 | データシミュレーションの基礎 | ★☆☆☆☆ | 60-90分 | 合成データ生成, make_regression/classification, 波形データ, ipywidgets |
| 02 | 前処理と特徴量エンジニアリング | ★★☆☆☆ | 90-120分 | StandardScaler, MinMaxScaler, OneHotEncoder, 特徴量選択, Pipeline |
| 03 | モデル評価指標 | ★★☆☆☆ | 90-120分 | MSE/RMSE/MAE/R², Accuracy/Precision/Recall/F1/ROC-AUC, 混同行列, CV |
| 04 | 線形モデル | ★★★☆☆ | 120-150分 | 線形回帰, Ridge/Lasso/ElasticNet, 正則化パスの可視化, パラメータスイープ |
| 05 | 決定木とアンサンブル | ★★★☆☆ | 120-150分 | 決定木, RandomForest, GradientBoosting, 特徴量重要度, 過学習シミュレーション |
| 06 | SVMとカーネル法 | ★★★☆☆ | 120-150分 | 線形SVM, RBF/多項式カーネル, C/gammaパラメータ, サポートベクター, マージン |
| 07 | MLP基礎 | ★★★☆☆ | 120-150分 | パーセプトロン, 順伝播, 活性化関数(ReLU/tanh/sigmoid), MLPClassifier |
| 08 | MLPパラメータ空間の探索 | ★★★★☆ | 90-120分 | アーキテクチャ探索(深さ/幅), 学習率, Adam/SGD/LBFGS, 早期停止, ヒートマップ |
| 09 | MLP回帰と波形予測 | ★★★★☆ | 90-120分 | 波形合成, 時間遅延特徴量, MLPRegressor, 波形再構成の可視化 |
| 10 | 自動ハイパーパラメータチューニング | ★★★☆☆ | 90-120分 | GridSearchCV, RandomizedSearchCV, ベイズ最適化(sklearn-optimize) |
| 11 | モデル比較と選択 | ★★★★☆ | 90-120分 | 統計的有意差検定, VotingClassifier, StackingClassifier, モデル選択基準 |
| 12 | 完全なMLパイプライン | ★★★★☆ | 120-150分 | Pipeline構築, EDA, 前処理→学習→評価, モデル永続化(joblib), ベストプラクティス |

---

## Unit 2: GBDT (gbdt/)

**対象**: ML基礎を習得した学習者
**推定時間**: 25-35時間
**使用ライブラリ**: LightGBM, XGBoost, CatBoost, Optuna, SHAP

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 13 | GBDT入門 | ★★★☆☆ | 120-150分 | ブースティング原理, LightGBM/XGBoost基本, learning_rate/num_leaves, 早期停止 |
| 14 | CatBoostとカテゴリ変数 | ★★★☆☆ | 120-150分 | CatBoostのOrdered Boosting, LightGBM/XGBoost/CatBoost比較, GPU加速 |
| 15 | Titanic EDA & 特徴量工学 | ★★★★☆ | 120-150分 | 欠損値補完, Title抽出, 家族サイズ, Fare binning, 特徴量交互作用, 相関分析 |
| 16 | Titanic GBDTモデリング | ★★★★☆ | 120-150分 | Stratified K-Fold, 3モデル訓練, アンサンブル(平均/加重投票), Kaggle提出 |
| 17 | Titanic Top 30%達成 | ★★★★★ | 180-240分 | 高度な特徴量工学, Optuna(プレビュー), OOF予測, 疑似ラベリング, Blending vs Stacking |
| 18 | House Prices回帰 | ★★★★★ | 180-240分 | 目的変数変換(log/Box-Cox), 外れ値検出, RMSLE, 量子回帰, 残差分析 |
| 19 | Store Demand時系列予測 | ★★★★★ | 180-240分 | トレンド/季節性分解, ラグ特徴量, Walk-forward検証, 多段階予測, SMAPE |
| 20 | Optuna最適化 | ★★★★☆ | 150-180分 | TPEアルゴリズム, パラメータ空間定義, Pruning, 並列最適化, 可視化 |
| 21 | SHAPモデル解釈 | ★★★★☆ | 150-180分 | Shapley値, TreeSHAP, waterfall/force/summary/dependenceプロット, バイアス検出 |
| 21b | 特徴量重要度の深掘り | ★★★★☆ | - | 特徴量重要度の詳細分析（拡張版） |
| 22 | Stackingアンサンブル | ★★★★★ | 150-180分 | 多層アンサンブル, OOF予測, ベースモデル/メタモデル選択, 多層Stacking, Blending比較 |

---

## Unit 3: 専門トピック (advanced/)

**対象**: GBDT/ML実践者
**推定時間**: 15-25時間
**使用ライブラリ**: imbalanced-learn, TabNet, category_encoders

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 23 | 不均衡データ対策 | ★★★★☆ | 120-150分 | SMOTE/ADASYN, Undersampling, SMOTETomek, scale_pos_weight, Focal Loss, PR曲線 |
| 24 | 時系列特徴量エンジニアリング | ★★★★☆ | 120-150分 | 周期エンコーディング(sin/cos), ラグ選択(PACF), 移動統計, EWMA, 外部変数 |
| 25 | カテゴリカル変数の高度処理 | ★★★★☆ | 120-150分 | Target Encoding, Frequency Encoding, Hash Encoding, Entity Embedding, 高カーディナリティ |
| 26 | Tabularディープラーニング | ★★★★★ | 150-180分 | TabNet(Attention), NODE(微分可能木), GBDT vs DL比較, 適用場面の判断 |
| 27 | Kaggle完全ワークフロー | ★★★★★ | 180-240分 | 理解→EDA→前処理→モデリング→アンサンブル→提出の全フェーズ, LB分析 |
| 28 | 総合演習プロジェクト | ★★★★★ | 300-400分 | エンドツーエンドプロジェクト, データ収集→EDA→モデリング→解釈→デプロイ準備 |

---

## Unit 4: 生成モデル (generative/)

**対象**: PyTorch基礎を習得した学習者
**推定時間**: 60-75時間
**使用ライブラリ**: PyTorch, torchvision, Diffusers, Transformers

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 30 | 正規分布と確率の基礎 | ★★☆☆☆ | 120-150分 | 確率分布, 期待値/分散, 中心極限定理, サンプリング |
| 31 | 最尤推定 | ★★★☆☆ | 120-150分 | 母集団 vs 標本, MLE理論, パラメータ推定, データ生成 |
| 32 | 多次元正規分布 | ★★★☆☆ | 120-150分 | 多変量正規PDF, 共分散行列, 2D可視化, マハラノビス距離 |
| 33 | 混合ガウスモデル(GMM) | ★★★☆☆ | 120-150分 | 多峰性分布, GMMの数式, データ生成, パラメータ推定の課題 |
| 34 | EMアルゴリズム | ★★★★☆ | 150-180分 | KLダイバージェンス, ELBO, E-step/M-step, 責任率, 収束 |
| 35 | PyTorch基礎と勾配降下法 | ★★☆☆☆ | 120-150分 | テンソル, autograd, 勾配降下, 線形回帰, SGD/Adam |
| 36 | ニューラルネットワークとMNIST | ★★★☆☆ | 120-150分 | MLP実装, 活性化関数, DataLoader, バッチ学習, 学習曲線 |
| 37 | VAE理論 | ★★★★☆ | 120-150分 | エンコーダ/デコーダ, ELBO導出, 再パラメータ化トリック, KL項 |
| 38 | VAE実装 | ★★★★☆ | 150-180分 | VAEネットワーク構築, ELBO損失, MNIST学習, 潜在空間可視化, 画像生成 |
| 39 | 拡散モデル理論(基礎) | ★★★★☆ | 150-180分 | 拡散過程, 逆拡散過程, ノイズスケジュール, 直感的理解 |
| 40 | 拡散モデル理論(ELBO) | ★★★★★ | 180-240分 | ELBO計算, q(x_t\|x_0)導出, 簡略化訓練目的関数 |
| 41 | U-Netと位置エンコーディング | ★★★★☆ | 150-180分 | U-Netアーキテクチャ, Skip接続, 正弦波位置エンコーディング |
| 42 | 拡散モデル実装(基礎) | ★★★★★ | 180-240分 | Forward/Reverse diffusion実装, ノイズ予測モデル, MNIST生成 |
| 43 | 拡散モデル学習と評価 | ★★★★★ | 180-240分 | 学習最適化, FID/IS, 大規模データ実験 |
| 44 | 条件付き拡散モデル | ★★★★☆ | 150-180分 | クラス条件付き生成, Classifier-Free Guidance, CLIP, ガイダンス強度 |
| 45 | Diffusersライブラリ実践 | ★★★★★ | 180-240分 | Stable Diffusion構成, テキストエンコーダ, Diffusersライブラリ, CFG |

---

## Unit 5: 3Dビジョン (3d-vision/)

**対象**: 線形代数と基礎的なプログラミング力のある学習者
**推定時間**: 25-35時間
**使用ライブラリ**: NumPy, OpenCV, Matplotlib

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 50 | 光学の基礎 | ★★☆☆☆ | - | 光の性質, レンズ, 画像形成の物理 |
| 51 | ピンホール投影モデル | ★★☆☆☆ | - | カメラモデル, 内部パラメータ, 射影変換 |
| 52 | レンズ歪み補正 | ★★★☆☆ | - | 放射/接線歪み, 歪み係数, 補正アルゴリズム |
| 53 | 座標変換と剛体運動 | ★★★☆☆ | - | 回転行列, 同次座標, 剛体変換 |
| 54 | カメラキャリブレーション | ★★★☆☆ | - | カメラ行列推定, チェッカーボード, Zhang法 |
| 55 | エピポーラ幾何学 | ★★★★☆ | - | 基本行列, エピポーラ線, 8点アルゴリズム |
| 56 | ステレオビジョンと視差 | ★★★★☆ | - | ステレオマッチング, 視差マップ, 深度推定 |
| 57 | 三角測量と3D再構成 | ★★★★☆ | - | 三角測量, 3D点復元 |
| 58 | 特徴検出とマッチング | ★★★★☆ | - | SIFT, SURF, 特徴マッチング, 対応点 |
| 59 | SfMパイプライン基礎 | ★★★★☆ | - | Structure from Motion, インクリメンタルSfM |
| 60 | バンドル調整 | ★★★★☆ | - | バンドル調整, 非線形最適化 |
| 61 | レイキャスティングと座標 | ★★★★☆ | - | レイキャスティング, レンダリング座標系 |
| 62 | 体積レンダリング | ★★★★☆ | - | ボリュームレンダリング, 透過率, 色の積分 |
| 63 | NeRF入門 | ★★★★☆ | - | Neural Radiance Fields, 暗黙的表現, ポジショナルエンコーディング |

---

## Unit 6: ニューラルエンジン (neural-engine/)

**対象**: 高校数学レベルの微分知識がある学習者
**推定時間**: 10-15時間
**使用ライブラリ**: NumPy, Matplotlib

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 70 | 微分の直感 | ★★☆☆☆ | - | 数値微分, 変化率, 関数の傾き |
| 71 | 連鎖律の分解 | ★★☆☆☆ | - | チェーンルール, 合成関数の微分 |
| 72 | 計算グラフ | ★★☆☆☆ | - | 計算グラフ表現, ノード演算, 自動微分の基盤 |
| 73 | 逆伝播（スカラー） | ★★★☆☆ | - | スカラー逆伝播, 勾配の流れ |
| 74 | 逆伝播（行列） | ★★★☆☆ | - | 行列逆伝播, ヤコビアン |
| 75 | 訓練ループとSGD | ★★★☆☆ | - | 訓練ループ構成, 確率的勾配降下法, ミニバッチ |
| 76 | 勾配病理 | ★★★☆☆ | - | 勾配消失/爆発, BatchNorm, 残差接続, 勾配クリッピング |

---

## Unit 7: 空間CNN (spatial-cnn/)

**対象**: ニューラルエンジン基礎を習得した学習者
**推定時間**: 30-40時間
**使用ライブラリ**: NumPy, PyTorch, torchvision

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 80 | 畳み込みとは何か（直感） | ★★☆☆☆ | - | ぼかし, 移動平均, カーネルの直感 |
| 81 | 畳み込みの数学的定義 | ★★☆☆☆ | - | 離散畳み込み式, 相関との違い, パディング, ストライド |
| 82 | NumPy実装（基礎） | ★★★☆☆ | - | ナイーブなループ実装 |
| 83 | NumPy実装（高速化） | ★★★☆☆ | - | im2col, ベクトル化, パフォーマンス比較 |
| 84 | 古典的フィルタ | ★★★☆☆ | - | Sobel, Laplacian, Gaussian, エッジ検出 |
| 85 | カーネルの可視化と解釈 | ★★☆☆☆ | - | 学習済みカーネルの可視化, フィルタの役割 |
| 86 | 受容野入門 | ★★★☆☆ | - | 受容野の概念, 計算方法 |
| 87 | 受容野と深さ | ★★★☆☆ | - | ネットワーク深さと受容野の成長 |
| 88 | ダウンサンプリング | ★★★☆☆ | - | Max/Average Pooling, ストライドによる縮小 |
| 89 | 受容野と3DGS | ★★★☆☆ | - | 3D Gaussian Splatting文脈での受容野 |
| 90 | 帰納バイアス入門 | ★★☆☆☆ | - | CNN固有のバイアス, 局所性, 重み共有 |
| 91 | 重み共有 | ★★☆☆☆ | - | パラメータ共有の仕組み, パラメータ削減 |
| 92 | 並進等変性 | ★★★☆☆ | - | 畳み込みの等変性, 位置不変性 |
| 93 | CNN vs MLP | ★★★☆☆ | - | 全結合との比較, パラメータ効率, 空間構造 |
| 94 | CNNが失敗する場合 | ★★★☆☆ | - | 回転, スケール変化, 長距離依存への限界 |
| 95 | CNNを超えて | ★★★☆☆ | - | Vision Transformer, Self-Attention, パッチ埋め込み |
| 96 | セマンティックセグメンテーション | ★★★★☆ | - | FCN, ピクセル分類, アップサンプリング |
| 97 | U-Netアーキテクチャ | ★★★★☆ | - | エンコーダ-デコーダ, Skip接続, 医療画像応用 |
| 98 | 特徴ピラミッド(FPN) | ★★★★☆ | - | FPN, マルチスケール特徴, トップダウンパスウェイ |
| 99 | スキップ接続 | ★★★★☆ | - | ResNet残差接続, DenseNet, 勾配流の改善 |
| 100 | CNN応用例 | ★★★★☆ | - | 実践的な応用事例 |
| 101 | 空間CNN総括 | ★★★☆☆ | - | シリーズ全体の統合 |
| 102 | 未来の展望 | ★★★☆☆ | - | 今後の方向性, 新しいアーキテクチャ |

---

## Unit 8: 最適化手法 (H_optimization/)

**対象**: 勾配降下法の基礎を理解した学習者
**推定時間**: 12-18時間
**使用ライブラリ**: NumPy, Matplotlib, Optuna

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 110 | 勾配降下法の基礎 | ★★★☆☆ | - | 目的関数, 収束条件, 学習率の影響, バッチ/ミニバッチ/SGD |
| 111 | モーメンタムとネステロフ | ★★★☆☆ | - | 運動量, ネステロフ加速勾配, 振動の抑制, 収束加速 |
| 112 | 適応的学習率 | ★★★☆☆ | - | AdaGrad, RMSprop, Adam, パラメータ別学習率 |
| 113 | 学習率スケジューリング | ★★★☆☆ | - | StepLR, CosineAnnealing, Warmup, OneCycleLR |
| 114 | 正則化と最適化 | ★★★★☆ | - | L1/L2正則化, Weight Decay, Dropout, 最適化との関係 |
| 115 | 高度な最適化手法 | ★★★★☆ | - | 2次手法, LAMB, Lookahead, SWA |
| 116 | オプティマイザ選択ガイド | ★★★☆☆ | - | 問題別の最適化手法選択, 実践レシピ, ハイパーパラメータ設定 |

---

## Unit 9: シーケンスモデリング (sequence-models/)

**対象**: ニューラルエンジン基礎を習得した学習者
**推定時間**: 15-20時間
**使用ライブラリ**: NumPy (120-124), PyTorch (125-126)

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 120 | シーケンスとは何か ― 時間の次元へ | ★★☆☆☆ | 60-90分 | 静的 vs 系列データ, スライディングウィンドウMLPの限界, 状態の概念 |
| 121 | バニラRNN ― 状態を持つネットワーク | ★★★☆☆ | 120-150分 | 状態方程式, 重み共有, 展開計算グラフ, forward実装, gradient check |
| 122 | BPTT ― 時間を遡る逆伝播 | ★★★★☆ | 150-180分 | BPTT完全実装, 固有値と勾配消失, Truncated BPTT, 勾配クリッピング, Adding Problem |
| 123 | LSTM ― 記憶を制御する | ★★★★☆ | 150-180分 | 4ゲート, セル状態の勾配高速道路, forward/backward実装, RNN vs LSTM実証 |
| 124 | GRUと時系列予測 | ★★★☆☆ | 120-150分 | GRU設計, RNN/LSTM/GRU三者比較, 時系列前処理, 多ステップ予測 |
| 125 | Seq2Seq ― シーケンス変換 | ★★★★☆ | 150-180分 | Encoder-Decoder, コンテキストベクトル, Teacher Forcing, ボトルネック実証 |
| 126 | Attention ― Transformerへの架け橋 | ★★★★☆ | 150-180分 | Bahdanau Attention, Attentionマトリクス可視化, Scaled Dot-Product, Self-Attentionへの布石 |

---

## Unit 10: 時空間モデリング (spatiotemporal/)

**対象**: 生成モデルコースと3Dビジョン基礎を習得した学習者
**推定時間**: 15-20時間
**使用ライブラリ**: PyTorch, einops

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 130 | 時間的注意機構の基礎 | ★★★☆☆ | 120-150分 | Temporal Attention, Causal Mask, シーケンスのSelf-Attention |
| 131 | Video Diffusion Models | ★★★★☆ | 150-180分 | U-Net時間拡張, Moving-MNIST, 動画生成 |
| 132 | Diffusion Transformer (DiT) | ★★★★☆ | 150-180分 | パッチ埋め込み, adaLN-Zero, Transformerベース拡散 |
| 133 | カメラと物体の運動分離 | ★★★★☆ | 120-150分 | Plücker座標, オプティカルフロー, 運動分解 |
| 134 | 時間的一貫性の技術 | ★★★★☆ | 120-150分 | Temporal Super-Resolution, FVD指標, 一貫性損失 |
| 135 | 物理動画生成 (Capstone) | ★★★★★ | 240-300分 | 物理シミュレーション+DiT, ダイナミクスモデリング |
| 136 | 時空間モデリング総括 | ★★★★☆ | 90-120分 | 技術体系整理, Sora解説 |

---

## Unit 11: 世界モデル (world-models/)

**対象**: 時空間モデリングと表現学習の基礎を習得した学習者
**推定時間**: 18-24時間
**使用ライブラリ**: PyTorch, gymnasium

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 140 | 予測のための表現学習 | ★★★☆☆ | 120-150分 | 対照学習, InfoNCE, t-SNE, 線形プローブ, ピクセル予測 vs 特徴空間予測 |
| 141 | JEPA | ★★★★☆ | 150-180分 | Joint Embedding Predictive Architecture, EMA, マスク予測 |
| 142 | モデルベースRL基礎 | ★★★★☆ | 150-180分 | Dyna-Q, GridWorld, 潜在空間での計画 |
| 143 | DreamerV3 | ★★★★★ | 180-240分 | RSSM, 想像内学習, 潜在空間計画 |
| 144 | Genie | ★★★★★ | 150-180分 | 潜在行動発見, VQ-VAE, インタラクティブ制御 |
| 145 | GridWorldエージェント (Capstone) | ★★★★★ | 300-360分 | 世界モデル+MPC計画, エージェント制御 |
| 146 | 世界モデル総括 | ★★★★☆ | 90-120分 | 全Phase統合, AGI展望 |

---

## Unit 12: 埋め込み (embeddings/)

**対象**: ニューラルエンジン基礎とPyTorchを習得した学習者
**推定時間**: 15-20時間
**使用ライブラリ**: gensim, transformers, sentence-transformers, faiss-cpu, umap-learn

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 150 | 埋め込みの幾何学 | ★★☆☆☆ | 90-120分 | コサイン類似度, ユークリッド距離, ドット積, GloVe探索, 単語演算(king-man+woman), 高次元の呪い |
| 151 | Word2Vecと静的埋め込み | ★★★☆☆ | 120-150分 | 分布仮説, Skip-gram NumPy実装, 負例サンプリング, CBOW, GloVe, FastText, 多義語問題 |
| 152 | 文脈付き埋め込み | ★★★★☆ | 120-150分 | BERT層別抽出, 多義語("bank")の文脈依存変化, 層別情報(構文vs意味), Attention可視化 |
| 153 | 文・文書の埋め込み | ★★★☆☆ | 90-120分 | Sentence-BERT, Bi-Encoder vs Cross-Encoder, プーリング(CLS/Mean/Max), 意味検索, STS評価 |
| 154 | 多様体学習と可視化 | ★★★☆☆ | 120-150分 | PCA(固有値分解), t-SNE(perplexity), UMAP(n_neighbors/min_dist), Swiss Roll, MNIST digits |
| 155 | ベクトル検索とインデックス | ★★★☆☆ | 90-120分 | Brute Force, FAISS, IVF(nlist/nprobe), PQ(量子化), HNSW(M/efSearch), recall vs 速度 |
| 156 | 距離学習とファインチューニング | ★★★★☆ | 120-150分 | Contrastive Loss, Triplet Loss, Hard Negative Mining, FashionMNIST実験, 評価指標(Recall@K, MAP) |
| 157 | 埋め込みの応用と統合 | ★★★☆☆ | 90-120分 | RAG, K-Meansクラスタリング, WEATバイアス分析, 多言語埋め込み, Capstoneプロジェクト |

---

## Unit 13: 言語モデリング (language-models/)

**対象**: 埋め込みシリーズとTransformer基礎を習得した学習者
**推定時間**: 15-20時間
**使用ライブラリ**: PyTorch, transformers, tokenizers, numpy, matplotlib

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 160 | 言語モデリングの基礎 | ★★☆☆☆ | 90-120分 | N-gram(Unigram/Bigram/Trigram), 連鎖律, マルコフ仮定, スムージング(ラプラス/Kneser-Ney), パープレキシティ |
| 161 | トークナイゼーション | ★★★☆☆ | 90-120分 | BPEスクラッチ実装, WordPiece, SentencePiece, HuggingFace tokenizers比較, 特殊トークン([CLS]/[MASK]/[PAD]) |
| 162 | NLPのためのTransformer | ★★★★☆ | 120-150分 | Multi-Head Attention(テキスト), Sinusoidal/Learned/RoPE位置エンコーディング, Encoder-only/Decoder-only/Enc-Dec分類, Pre-LN/Post-LN |
| 163 | BERTの事前学習 | ★★★★☆ | 120-150分 | MLM(15%マスク, 80/10/10ルール), NSP, ミニBERT事前学習ループ, テキスト分類ファインチューニング |
| 164 | GPTと自己回帰言語モデル | ★★★★☆ | 120-150分 | Causal LM目的関数, ミニGPTスクラッチ実装, next-token prediction, Teacher Forcing vs Autoregressive |
| 165 | デコーディング戦略 | ★★★☆☆ | 90-120分 | Greedy/Beam Search, Top-k/Nucleus(Top-p) Sampling, Temperature, 反復ペナルティ, 全手法比較 |
| 166 | 現代LLMのアーキテクチャ | ★★★★☆ | 90-120分 | Scaling Laws(Kaplan/Chinchilla), RoPE実装, GQA(MHA→MQA→GQA), KV-Cache, FlashAttention概念, RMSNorm, SwiGLU |
| 167 | ファインチューニングとアライメント | ★★★★☆ | 120-150分 | LoRA/QLoRA, Instruction Tuning, RLHF/DPO概要, RAG生成側完成, BLEU/ROUGE/パープレキシティ |

---

## Unit 14: 画像変容と潜在空間 (image-morphing/)

**対象**: 生成モデルコース（特にVAEと拡散モデル）を習得した学習者
**推定時間**: 25-35時間
**使用ライブラリ**: PyTorch, Diffusers, Transformers, torchvision

### Phase 1: 潜在空間の理論と探索

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 300 | 潜在空間とは何か | ★★★☆☆ | 60-90分 | 高次元→低次元の圧縮, マニフォールド仮説, AE/VAEの潜在空間の違い, 潜在コードの直感 |
| 301 | VAE潜在空間の探索 | ★★★☆☆ | 90-120分 | 訓練済みVAEの潜在空間を可視化(t-SNE/PCA), クラスタ構造, デコーダで潜在点→画像復元 |
| 302 | 潜在ベクトル演算 | ★★★☆☆ | 90-120分 | z_smile - z_neutral + z_other = ?, ベクトル加算/減算, 属性ベクトルの発見, CelebAでの実験 |
| 303 | 潜在走査（Latent Traversal） | ★★★☆☆ | 60-90分 | 1次元ずつ動かして変化を観察, 各次元の意味の発見, GIFアニメーション生成 |
| 304 | 解きほぐされた表現 | ★★★★☆ | 90-120分 | Disentanglement, β-VAE, 独立な因子の分離, 定量的評価(DCI/MIG), なぜ重要か |

### Phase 2: GANベースの画像変容

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 305 | GAN潜在空間の構造 | ★★★☆☆ | 90-120分 | GANの潜在空間はなぜ構造的か, Z空間 vs W空間(StyleGAN), 分布の違い, モード崩壊の影響 |
| 306 | 潜在空間の線形補間とSlerp | ★★★☆☆ | 60-90分 | 線形補間(Lerp) vs 球面線形補間(Slerp), なぜ球面か, 補間パスの可視化, 補間品質の比較 |
| 307 | 意味ベクトルと属性編集 | ★★★★☆ | 90-120分 | InterFaceGAN, 属性方向の発見(年齢/性別/眼鏡), 条件付き操作, 編集強度の制御 |
| 308 | GANSpace: 教師なし方向発見 | ★★★★☆ | 90-120分 | PCAで潜在空間の主要方向を発見, 教師あり vs 教師なし, 方向の解釈と可視化 |
| 309 | GANによる加齢シミュレーション | ★★★★☆ | 90-120分 | 加齢ベクトルの構築, 年齢方向の連続的変化, アイデンティティ保持, 実画像への適用(GAN Inversion) |

### Phase 3: Diffusionベースの画像変容

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 310 | DDIMの決定論的サンプリング | ★★★★☆ | 90-120分 | DDPMとDDIMの違い, η=0の決定論性, 少ステップ生成, 同じノイズ→同じ画像の性質 |
| 311 | DDIM Inversion入門 | ★★★★☆ | 90-120分 | 画像→ノイズへの逆算, 決定論的逆過程, 再構成精度の検証, なぜInversionが変容に必要か |
| 312 | ノイズ空間での補間 | ★★★★☆ | 90-120分 | 2画像をInversion→ノイズ空間で補間→デノイズ, 構造のグラデーション, 補間ステップ数の効果 |
| 313 | テキスト特徴量の補間 | ★★★★☆ | 90-120分 | CLIPテキスト埋め込みの補間, プロンプト間のスムーズ遷移, ノイズ+テキスト同時補間の効果 |
| 314 | SDEditと構造ガイド変容 | ★★★★☆ | 90-120分 | SDEdit(途中ノイズからデノイズ), ノイズ強度と忠実度/創造性のトレードオフ, img2img変換 |

### Phase 4: 統合実践

| # | タイトル | 難易度 | 時間 | 主なトピック |
|---|---------|--------|------|------------|
| 315 | GAN vs Diffusion Morphing比較 | ★★★★★ | 120-150分 | 同じペアでGAN/Diffusion比較, 品質評価(FID/LPIPS/SSIM), 得意/不得意の分析, 使い分けガイド |
| 316 | Image Morphing Capstone | ★★★★★ | 150-180分 | 猫→ハンバーガー等の自由変容, Morphing動画生成, パラメータ実験, ポートフォリオ作品制作 |

---

## キーワード索引

新しいコンテンツを追加する際に、既存の内容と重複しないかこの索引で確認してください。

### A-E
| キーワード | 関連ノートブック |
|-----------|--------------|
| Adam / AdaGrad / RMSprop | 35, 112, 116 |
| Adding Problem | 122, 123 |
| Attention / Self-Attention | 92, 95, 126, 130, 132, 152 |
| AutoML / ハイパーパラメータ最適化 | 10, 20, 116 |
| Backpropagation (逆伝播) | 73, 74, 75, 122 |
| BPTT (Backpropagation Through Time) | 122 |
| Batch Normalization | 76 |
| BERT | 152, 153 |
| Bias (バイアス分析) | 157 |
| Bundle Adjustment | 60 |
| CatBoost | 14, 16, 17, 22 |
| Chain Rule (連鎖律) | 71 |
| CLIP | 44, 45 |
| Clustering (クラスタリング) | 157 |
| CNN (畳み込みニューラルネットワーク) | 80-102 |
| Computation Graph (計算グラフ) | 72 |
| Contrastive Learning (対照学習) | 140, 156 |
| Cross-Validation (交差検証) | 03, 11, 16 |
| Curse of Dimensionality (高次元の呪い) | 150 |
| DDIM / DDIM Inversion | 310, 311, 312 |
| Decision Tree (決定木) | 05 |
| Diffusion Models (拡散モデル) | 37-43, 131, 132, 310-314 |
| Disentangled Representation (解きほぐされた表現) | 304 |
| DiT (Diffusion Transformer) | 132 |
| DreamerV3 | 143 |
| Dropout | 114 |
| Dyna-Q | 142 |
| EDA (探索的データ分析) | 15, 27 |
| ELBO | 34, 37, 40 |
| EM Algorithm | 34 |
| Embedding (埋め込み) | 150-157 |
| Encoder-Decoder / Seq2Seq | 125, 126 |
| Epipolar Geometry | 55 |

### F-N
| キーワード | 関連ノートブック |
|-----------|--------------|
| FAISS | 155 |
| FastText | 151 |
| Feature Engineering (特徴量工学) | 02, 15, 24, 25 |
| Feature Importance | 21, 21b |
| Feature Pyramid Network (FPN) | 98 |
| Focal Loss | 23 |
| GAN Inversion | 309 |
| GAN Latent Space (GAN潜在空間) | 305, 306, 307, 308, 309 |
| GANSpace | 308 |
| GloVe | 150, 151, 157 |
| GMM (混合ガウスモデル) | 33, 34 |
| Gradient Descent (勾配降下法) | 35, 70, 110 |
| Gradient Clipping (勾配クリッピング) | 76, 122 |
| Gradient Pathology (勾配病理) | 76, 122 |
| GRU (Gated Recurrent Unit) | 124 |
| GridSearchCV | 08, 10 |
| HNSW | 155 |
| Image Morphing (画像変容) | 315, 316 |
| InfoNCE | 140 |
| InterFaceGAN | 307 |
| IVF (Inverted File Index) | 155 |
| JEPA | 141 |
| Kaggle | 17, 18, 19, 27 |
| K-Means | 157 |
| KL Divergence | 34, 37 |
| Lasso / Ridge / ElasticNet | 04, 114 |
| Latent Space (潜在空間) | 300, 301, 302, 303, 304, 305 |
| Latent Traversal (潜在走査) | 303 |
| Learning Rate Scheduling | 113 |
| LightGBM | 13, 14, 16, 17, 20 |
| Linear Models (線形モデル) | 04 |
| LSTM (Long Short-Term Memory) | 123 |
| MLE (最尤推定) | 31 |
| MLP (多層パーセプトロン) | 07, 08, 09 |
| Momentum / Nesterov | 111 |
| Multilingual (多言語) | 157 |
| NeRF | 63 |
| Normal Distribution (正規分布) | 30, 32 |

### O-Z
| キーワード | 関連ノートブック |
|-----------|--------------|
| Optuna | 20, 116 |
| PCA | 150, 154 |
| Pipeline (MLパイプライン) | 12 |
| Pooling (プーリング) | 88, 153 |
| Product Quantization (PQ) | 155 |
| PyTorch | 35, 36 |
| RAG (検索拡張生成) | 157 |
| Random Forest | 05 |
| Receptive Field (受容野) | 86, 87, 89 |
| Regularization (正則化) | 04, 114 |
| ResNet / Skip Connection | 99 |
| RNN (Recurrent Neural Network) | 121, 122, 123, 124 |
| RSSM | 143 |
| Semantic Segmentation | 96, 97 |
| Sentence-BERT / Sentence-Transformers | 153, 157 |
| SGD | 75, 110 |
| SDEdit | 314 |
| SHAP | 21 |
| Skip-gram / CBOW | 151 |
| Slerp (球面線形補間) | 306, 315 |
| SMOTE | 23 |
| Stable Diffusion | 45 |
| StyleGAN | 305, 307, 308, 309 |
| Sequence Modeling (シーケンスモデリング) | 120-126 |
| Stacking | 12, 22 |
| Stereo Vision | 56 |
| Structure from Motion (SfM) | 59 |
| SVM | 06 |
| t-SNE | 140, 154, 301 |
| TabNet | 26 |
| Teacher Forcing | 125 |
| Time Series (時系列) | 19, 24, 124, 130 |
| Transformer | 95, 126, 132 |
| Triangulation | 57 |
| Triplet Loss | 156 |
| U-Net | 41, 97, 131 |
| UMAP | 154, 157 |
| VAE (変分オートエンコーダ) | 37, 38, 300, 301, 304 |
| Vision Transformer (ViT) | 95 |
| Volume Rendering | 62 |
| VQ-VAE | 144 |
| Word2Vec | 151 |
| XGBoost | 13, 16, 17, 22 |
| β-VAE | 304 |

---

## 依存関係マップ

ノートブック間の前提知識の関係：

```
[00-03 データ基礎] → [04-06 古典モデル] → [07-09 MLP] → [10-12 統合]
                                                              │
                    [13-16 GBDT基礎] ← ──────────────────────┘
                         │
                    [17-19 Kaggle実践]
                         │
                    [20-22 高度GBDT] → [23-28 専門トピック]

[30-34 確率統計] → [35-36 PyTorch] → [37-38 VAE] → [39-43 拡散モデル] → [44-45 応用]
                        │                                                    │
                        │                                               [130-136 時空間]
                        │                                                    │
                        │                                               [140-146 世界モデル]
                        │
                        │            [37-38 VAE] + [44-45 応用] → [300-316 画像変容]
                        │
                   [70-76 ニューラルエンジン] → [80-102 空間CNN]
                        │                           │
                        │                      [110-116 最適化]
                        │
                        ├── [120-126 シーケンス] → [130-136 時空間]
                        │                       → [160-167 言語モデリング]
                        │
                   [150-157 埋め込み] → [160-167 言語モデリング]

[50-63 3Dビジョン] ← [線形代数の基礎]
```

---

> **メンテナンスルール**: 新しいノートブックを追加した場合は、必ずこのシラバスも更新してください。詳細は `CLAUDE.md` を参照。
