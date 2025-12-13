# スクリプト集

このディレクトリには、プロジェクトで使用されるユーティリティスクリプトとサンプルコードが含まれています。

## 📁 ディレクトリ構造

```
scripts/
├── notebook_improvements/    # ノートブック改善用スクリプト
│   ├── README.md            # 詳細な説明
│   ├── analyze_notebooks.py
│   ├── improve_all_notebooks.py
│   └── ...
│
└── examples/                # 学習用サンプルスクリプト
    ├── README.md           # 詳細な説明
    ├── linearRegression.py
    ├── naivebayes.py
    └── ...
```

## 📖 各ディレクトリの説明

### 🔧 notebook_improvements/

Jupyter Notebookを教育用フレームワークに基づいて改善するためのスクリプト集。

**主な機能:**
- ノートブックの分析と診断
- 詳細な日本語説明の自動追加
- プレースホルダーの検出と削除
- 品質保証チェック

**詳細**: [notebook_improvements/README.md](./notebook_improvements/README.md)

---

### 💡 examples/

機械学習の基本概念を学ぶためのシンプルなサンプルスクリプト集。

**含まれる内容:**
- 線形回帰
- ナイーブベイズ
- パーセプトロン
- TensorFlow入門
- データ処理例

**詳細**: [examples/README.md](./examples/README.md)

---

## 🎯 推奨される学習順序

1. **初学者向け**:
   ```
   examples/ のサンプルスクリプトで基礎を学ぶ
   ↓
   notebooks/ の改善版ノートブックで体系的に学習
   ```

2. **ノートブック作成者向け**:
   ```
   notebook_improvements/ のスクリプトを使用して
   教育用ノートブックを作成・改善
   ```

## 🔗 関連リソース

- [改善版ノートブック](../notebooks/)
- [プロジェクトルート](../)

---

**最終更新**: 2025年12月13日
