# CLAUDE.md - Machine Learning Playground

## プロジェクト概要

機械学習の基礎から応用までを体系的に学ぶための教育用Jupyterノートブック集。
現在118以上のノートブックを収録。

## 重要なルール

### シラバスの更新（必須）

**新しいノートブックやコンテンツを追加・削除・変更した場合は、必ず `SYLLABUS.md` を更新すること。**

更新対象：
- ノートブックの追加・削除・名称変更 → 該当ユニットのテーブルを更新
- 新しいセクション/ユニットの追加 → セクション追加＋カリキュラム概要図の更新
- キーワードの追加 → キーワード索引（A-Z）の更新
- 依存関係の変更 → 依存関係マップの更新
- ノートブック総数の変更 → ヘッダーの総数を更新

### ノートブック作成ガイドライン

新しいノートブックを作成する際は `NOTEBOOK_GUIDELINES.md` に従うこと。
主な要件：
- 日本語での詳細解説
- `# ============================================================` によるセクション区切り
- 学習目標チェックリスト、前提知識、推定時間、難易度の記載
- 豊富な可視化（matplotlib/seaborn、最低3-4個のプロット）
- まとめ、チートシート、よくある間違い、自己評価クイズ
- `np.random.seed(42)` で再現性確保

### 図（matplotlib/seaborn）の言語ルール

- **図のタイトル、軸ラベル、凡例、アノテーション等はすべて英語で記載すること**
- ノートブック本文（Markdownセル）の解説は日本語でOK
- `japanize_matplotlib` は使用禁止（日本語フォントの表示が不安定なため）
- 理由：日本語を図に入れると表示崩れ・ずれが発生しやすい

### ドキュメント整合性

コンテンツ変更時に確認すべきファイル：
- `SYLLABUS.md` — 総合シラバス（最優先で更新）
- `README.md` — ノートブック数・ディレクトリ構成
- `ML_LEARNING_PLAN.md` — 学習計画

## プロジェクト構成

```
notebooks/
├── fundamentals/        # 00-13: ML基礎
├── pytorch-basics/      # 35-36: PyTorch入門
├── generative-models/   # 30-45: 生成モデル
├── architectures/       # 50-65: ニューラルアーキテクチャ
├── neural-engine/       # 70-76: ニューラルネットワーク内部
├── 3d-vision/          # 80-88: 3Dコンピュータビジョン
├── reinforcement-learning/ # 90-97: 強化学習
├── optimization/        # 100-107: 最適化手法
├── advanced-architectures/ # 110-117: 先端アーキテクチャ
├── representation-learning/ # 130-145: 表現学習
└── embeddings/          # 150-157: 埋め込み
```

## 技術スタック

- Python 3.x, PyTorch, NumPy, Matplotlib, Seaborn
- Jupyter Notebook (nbformat 4.4)
- 追加ライブラリ: gensim, transformers, sentence-transformers, faiss-cpu, umap-learn 等
