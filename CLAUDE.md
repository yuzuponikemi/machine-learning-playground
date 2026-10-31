# CLAUDE.md - Machine Learning Playground

## プロジェクト概要

機械学習の基礎から応用までを体系的に学ぶための教育用Jupyterノートブック集。
現在 157 ファイル（番号は156通り。21 のみ `21_shap_model_interpretation` と
`21_feature_importance_deep_dive` の2本）を15ユニットに収録。
ユニットの正本は `SYLLABUS.md`。

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
- ノートブック本文（Markdownセル）の解説は日本語でOK。コード内のコメント・docstring・
  `print` 出力も日本語でよい（規約の対象は**図に描かれるテキスト**だけ）
- `japanize_matplotlib` は使用禁止（日本語フォントの表示が不安定なため）
- **手書きの日本語フォント設定も禁止**（`def setup_japanese_font()` /
  `rcParams['font.family'] = 'Hiragino Sans'` 等）。japanize と同じ問題を持つ
- 理由：日本語を図に入れると表示崩れ・ずれが発生しやすい

**現状（2026-10-25 時点で全ノートブック適合済み）**:
日本語の図テキスト 0件、`japanize_matplotlib` 0件、手書きフォント設定 0件。
既存ノートブックを一括で直すツールが `scripts/notebook_improvements/` にある。

| ツール | 用途 |
|---|---|
| `plot_text_i18n.py` | 図テキストの抽出と英語化（`check` / `extract` / `apply`） |
| `remove_jp_font_setup.py` | 日本語フォント設定の除去（図テキストが英語化済みのものだけ） |

どちらも AST の位置情報で matplotlib のテキスト設定呼び出しだけを対象にするので、
Markdown セルや `print` の日本語は触らない。置換後は各セルを `ast.parse` で検証する。

### ドキュメント整合性

コンテンツ変更時に確認すべきファイル：
- `SYLLABUS.md` — 総合シラバス（最優先で更新）
- `README.md` — ノートブック数・ディレクトリ構成
- `ML_LEARNING_PLAN.md` — 学習計画
- `DIFFUSION_MODELS_CURRICULUM.md` — 拡散モデルカリキュラム（30-45）
- `VLA_CURRICULUM.md` — VLA カリキュラム計画（170-174、実装済み）
- `GENERATIVE_TAXONOMY_CURRICULUM.md` — 生成モデルの分類を完成させる計画
  （46-49, 147, Unit 16-18 = 180-186 / 190-196 / 200-206、未実装）

## プロジェクト構成

```
notebooks/
├── fundamentals/        # 00-12: ML基礎 (13)
├── gbdt/                # 13-22: GBDT・Kaggle実践 (11)
├── advanced/            # 23-28: 専門トピック (6)
├── generative/          # 30-47: 生成モデル・拡散モデル・MDN (17)
├── 3d-vision/           # 50-63: 3Dコンピュータビジョン・NeRF (14)
├── neural-engine/       # 70-76: 誤差逆伝播の内部 (7)
├── spatial-cnn/         # 80-102: 畳み込みと帰納バイアス (23)
├── H_optimization/      # 110-116: 最適化手法 (7)
├── sequence-models/     # 120-126: RNN/LSTM/Attention (7)
├── spatiotemporal/      # 130-136: 時空間・動画生成 (7)
├── world-models/        # 140-146: 世界モデル (7)
├── embeddings/          # 150-157: 埋め込み (8)
├── language-models/     # 160-167: 言語モデリング (8)
├── vla/                 # 170-174: VLA 視覚・言語・行動 (5)
└── image-morphing/      # 300-316: 画像変容と潜在空間 (17)
```

`notebooks/` 直下の `data/` `catboost_info/` `submissions/` はノートブックではなく
データ・学習ログ・提出物の置き場。

## 技術スタック

- Python 3.x, PyTorch, NumPy, Matplotlib, Seaborn
- Jupyter Notebook (nbformat 4.4)
- 追加ライブラリ: gensim, transformers, sentence-transformers, faiss-cpu, umap-learn 等

## ノートブック & プロット規約

- `japanize_matplotlib` は使わない——プロットのテキスト（タイトル、ラベル、凡例）はすべて英語にする
- サブエージェントを並列でノートブック作成する場合、同時実行は 5 本まで。各バッチ後に git commit する
