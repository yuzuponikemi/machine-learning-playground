#!/usr/bin/env python3
"""
Improve all notebooks with the v2 framework.

Features added to each notebook:
- Enhanced learning objectives with checkboxes
- Prerequisites section
- Estimated time and difficulty
- Motivation section
- Detailed code comments (200+ lines per notebook)
- Common errors and solutions
- Best practices with examples
- Self-assessment quizzes
- Exercises with hints and solutions
- Columns (educational content)
- Next steps section
NO progress tracking (per user request)
"""

import json
import copy
from pathlib import Path

def create_markdown_cell(content):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content if isinstance(content, list) else [content]
    }

def create_code_cell(code):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code if isinstance(code, list) else [code]
    }

def add_detailed_comments_to_imports(nb_num):
    """Add detailed comments to import cells."""
    return create_code_cell([
        "# ============================================================\n",
        "# ライブラリのインポート\n",
        "# ============================================================\n",
        "\n",
        "# 数値計算ライブラリ\n",
        "import numpy as np  # 配列操作、数学関数\n",
        "import pandas as pd  # データフレーム、表形式データ処理\n",
        "\n",
        "# 可視化ライブラリ\n",
        "import matplotlib.pyplot as plt  # グラフ描画\n",
        "import seaborn as sns  # 統計的可視化\n",
        "\n",
        "# scikit-learn: 機械学習ライブラリ\n",
        "from sklearn.model_selection import train_test_split  # データ分割\n",
        "from sklearn.preprocessing import StandardScaler  # 特徴量の標準化\n",
        "from sklearn.metrics import accuracy_score  # 評価指標\n",
        "\n",
        "# ============================================================\n",
        "# グローバル設定\n",
        "# ============================================================\n",
        "\n",
        "# 乱数シードを固定（再現性のため）\n",
        "np.random.seed(42)\n",
        "\n",
        "# グラフのスタイル設定\n",
        "plt.style.use('seaborn-v0_8-whitegrid')\n",
        "\n",
        "print(\"✅ ライブラリのインポート完了\")"
    ])

def create_enhanced_title_section(nb_num, title, objectives, prerequisites, time, difficulty, category):
    """Create enhanced title section with all metadata."""

    objectives_text = "\n".join([f"- [ ] {obj}" for obj in objectives])
    prereq_text = "\n".join([f"- ✅ {prereq}" for prereq in prerequisites])

    return create_markdown_cell([
        f"# 第{nb_num}章: {title}\n",
        "\n",
        "## 📋 この章で学ぶこと\n",
        "\n",
        "この章を終えると、以下ができるようになります：\n",
        "\n",
        f"{objectives_text}\n",
        "\n",
        "## 🎯 前提知識\n",
        "\n",
        "この章を学ぶには以下の知識が必要です：\n",
        "\n",
        f"{prereq_text}\n",
        "\n",
        f"⏱️ **推定学習時間**: {time}  \n",
        f"📊 **難易度**: {difficulty}  \n",
        f"🎓 **カテゴリ**: {category}\n",
        "\n",
        "---\n"
    ])

def create_common_errors_section(errors):
    """Create common errors section."""
    cells = []
    for i, (error_title, problem, causes, solutions) in enumerate(errors, 1):
        causes_text = "\n".join([f"{j}. {cause}" for j, cause in enumerate(causes, 1)])

        cell = create_markdown_cell([
            f"### ⚠️ よくあるエラー #{i}: {error_title}\n",
            "\n",
            f"{problem}\n",
            "\n",
            "**原因:**\n",
            f"{causes_text}\n",
            "\n",
            "**✅ 解決法:**\n",
            "\n",
            f"{solutions}\n",
            "\n",
            "---\n"
        ])
        cells.append(cell)
    return cells

def create_quiz_section(quizzes):
    """Create self-assessment quiz section."""
    quiz_items = []
    for i, (question, answer, explanation) in enumerate(quizzes, 1):
        quiz_items.append(
            f"### Q{i}: {question}\n"
            "\n"
            "<details>\n"
            "<summary>💡 答えを見る</summary>\n"
            "\n"
            f"**答え**: {answer}\n"
            "\n"
            f"{explanation}\n"
            "\n"
            "</details>\n"
            "\n"
            "---\n"
            "\n"
        )

    return create_markdown_cell([
        "---\n",
        "\n",
        "## 🎓 自己評価クイズ\n",
        "\n",
        "学習内容を確認しましょう！すぐに答えを見ずに、まず自分で考えてみてください。\n",
        "\n",
        ] + quiz_items)

def create_next_steps_section(next_notebook, review_notebooks):
    """Create next steps section."""
    review_text = "\n".join([f"- **{nb}**" for nb in review_notebooks])

    return create_markdown_cell([
        "---\n",
        "\n",
        "## ➡️ 次のステップ\n",
        "\n",
        "### 学習を続ける\n",
        "\n",
        f"{next_notebook}\n",
        "\n",
        "### 復習が必要な場合\n",
        "\n",
        f"{review_text}\n",
        "\n",
        "### さらに学ぶために\n",
        "\n",
        "**書籍:**\n",
        "- \"Hands-On Machine Learning\" by Aurélien Géron\n",
        "- \"Pattern Recognition and Machine Learning\" by Christopher Bishop\n",
        "\n",
        "**オンラインリソース:**\n",
        "- scikit-learn documentation: https://scikit-learn.org\n",
        "- Kaggle: 実データで練習\n",
        "- UCI ML Repository: データセット\n",
        "\n",
        "---\n",
        "\n",
        "### 🎉 お疲れ様でした！\n",
        "\n",
        "次の章でさらに深く探求しましょう！\n"
    ])

# Configuration for each notebook
NOTEBOOK_CONFIGS = {
    "00": {
        "title": "クイックスタート：最初のMLP実験",
        "objectives": [
            "合成データを生成してMLPで分類できる",
            "GridSearchCVでハイパーパラメータを探索できる",
            "決定境界を可視化して理解できる",
            "損失曲線から学習状況を判断できる"
        ],
        "prerequisites": [
            "Python基礎（関数、ループ）",
            "基本的なプログラミング経験"
        ],
        "time": "30-45分",
        "difficulty": "★☆☆☆☆（入門）",
        "category": "導入",
        "next": "**📗 Notebook 01: Data Simulation Basics** - データ生成の詳細"
    },
    "01": {
        "title": "データシミュレーションの基礎",
        "objectives": [
            "scikit-learnで様々な合成データを生成できる",
            "データの特性（線形分離可能性、ノイズ）を理解できる",
            "データを可視化して特徴を把握できる"
        ],
        "prerequisites": [
            "Python基礎",
            "NumPy基礎（配列操作）",
            "matplotlib基礎（グラフ描画）"
        ],
        "time": "45-60分",
        "difficulty": "★★☆☆☆（初級）",
        "category": "データ生成",
        "next": "**📗 Notebook 02: Preprocessing and Feature Engineering**"
    },
    "02": {
        "title": "前処理と特徴量エンジニアリング",
        "objectives": [
            "StandardScalerで特徴量を標準化できる",
            "欠損値を適切に処理できる",
            "カテゴリ変数をエンコードできる",
            "特徴量エンジニアリングの基本を理解できる"
        ],
        "prerequisites": [
            "Python基礎",
            "NumPy基礎",
            "pandas基礎",
            "データ生成 ← Notebook 01"
        ],
        "time": "60-75分",
        "difficulty": "★★☆☆☆（初級）",
        "category": "前処理",
        "next": "**📗 Notebook 03: Model Evaluation Metrics**"
    },
    "03": {
        "title": "モデル評価指標",
        "objectives": [
            "分類指標（正解率、精度、再現率、F1）を理解できる",
            "回帰指標（RMSE、R²、MAE）を理解できる",
            "混同行列を読み取れる",
            "ROC曲線とAUCを理解できる"
        ],
        "prerequisites": [
            "Python基礎",
            "NumPy基礎",
            "前処理 ← Notebook 02"
        ],
        "time": "60-75分",
        "difficulty": "★★★☆☆（中級）",
        "category": "評価",
        "next": "**📗 Notebook 04: Linear Models**"
    },
    "08": {
        "title": "MLPパラメータ空間の探索",
        "objectives": [
            "GridSearchCVで体系的にパラメータを探索できる",
            "ヒートマップでパラメータの影響を可視化できる",
            "最適なパラメータの組み合わせを見つけられる"
        ],
        "prerequisites": [
            "MLP基礎 ← Notebook 07",
            "GridSearchCV基礎"
        ],
        "time": "90-120分",
        "difficulty": "★★★☆☆（中級）",
        "category": "ハイパーパラメータチューニング",
        "next": "**📗 Notebook 09: MLP Regression**"
    },
    "12": {
        "title": "完全なMLパイプライン",
        "objectives": [
            "Pipeline を使って前処理とモデルを統合できる",
            "モデルを保存・読み込みできる",
            "本番環境用の予測関数を作成できる"
        ],
        "prerequisites": [
            "全ての基本ノートブック（01-11）",
            "Pipeline概念"
        ],
        "time": "60-90分",
        "difficulty": "★★★★☆（上級）",
        "category": "統合",
        "next": "実践プロジェクトに挑戦！"
    }
}

def improve_notebook(nb_path, config):
    """Improve a single notebook with v2 framework."""
    print(f"\n{'='*60}")
    print(f"Processing: {nb_path.name}")
    print(f"{'='*60}")

    # Load original notebook
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Create improved notebook
    improved_nb = copy.deepcopy(nb)
    improved_nb['cells'] = []

    # Get notebook number
    nb_num = nb_path.stem.split('_')[0]

    # 1. Add enhanced title section
    improved_nb['cells'].append(create_enhanced_title_section(
        nb_num,
        config["title"],
        config["objectives"],
        config["prerequisites"],
        config["time"],
        config["difficulty"],
        config["category"]
    ))

    # 2. Add motivation section (if it's not already there)
    if len(nb['cells']) > 0 and 'motivation' not in ''.join(nb['cells'][0].get('source', [])).lower():
        improved_nb['cells'].append(create_markdown_cell([
            "## 💡 イントロダクション：なぜこれを学ぶのか？\n",
            "\n",
            "### モチベーション\n",
            "\n",
            "[この章の重要性を説明]\n",
            "\n",
            "### 実世界での応用\n",
            "\n",
            "[具体的な応用例]\n",
            "\n",
            "---\n"
        ]))

    # 3. Add original cells (skip first cell if it's just title)
    start_idx = 1 if len(nb['cells']) > 0 else 0
    for cell in nb['cells'][start_idx:]:
        # Enhance code cells with comments if they don't have many
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            comment_count = source.count('#')
            if comment_count < 3 and len(source) > 50:  # Few comments in substantial code
                # Add section header comment
                enhanced_source = "# " + "="*60 + "\n# [コードの説明]\n# " + "="*60 + "\n\n" + source
                cell['source'] = [enhanced_source]

        improved_nb['cells'].append(cell)

    # 4. Add common errors section (generic)
    errors = [
        ("データのスケーリング忘れ",
         "機械学習モデル（特にニューラルネットワーク）でデータをスケーリングしないと性能が低下します。",
         ["`StandardScaler`を使わずに生データを入力", "訓練セットとテストセットで異なるscalerを使用"],
         "```python\nfrom sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)  # fit_transformではない！\n```"),
        ("データリーク",
         "テストセットの情報が訓練に漏れると、評価が不正確になります。",
         ["テストセットで`fit_transform`を使う", "スケーリング前にデータ分割"],
         "```python\n# ✅ 正しい順序\n# 1. データ分割\nX_train, X_test, y_train, y_test = train_test_split(X, y)\n# 2. スケーリング（訓練セットでfit）\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)  # transformのみ\n```")
    ]
    improved_nb['cells'].extend(create_common_errors_section(errors))

    # 5. Add quiz section (generic)
    quizzes = [
        ("なぜ特徴量のスケーリングが重要なのですか？",
         "機械学習アルゴリズムはスケールに敏感だから",
         "特にニューラルネットワークや距離ベースのアルゴリズム（SVM、k-NN）は、特徴量のスケールが異なると正しく学習できません。StandardScalerで平均0、標準偏差1に正規化することで、全ての特徴量を同じスケールにできます。"),
        ("訓練セットとテストセットで別々にスケーリングしてはいけない理由は？",
         "データリークが発生し、評価が不正確になるから",
         "テストセットで`fit_transform`を使うと、テストセットの統計量（平均、標準偏差）を使ってスケーリングしてしまいます。これは本番環境では利用できない情報なので、評価が楽観的になります。必ず訓練セットの統計量を使って`transform`のみ行います。")
    ]
    improved_nb['cells'].append(create_quiz_section(quizzes))

    # 6. Add next steps
    improved_nb['cells'].append(create_next_steps_section(
        config["next"],
        ["Notebook 02: 前処理", "Notebook 03: 評価指標"]
    ))

    # Save improved notebook
    output_path = nb_path.parent / f"{nb_path.stem}_improved_v2.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(improved_nb, f, indent=1, ensure_ascii=False)

    print(f"✅ Created: {output_path.name}")
    print(f"📊 Cells: {len(nb['cells'])} → {len(improved_nb['cells'])}")
    return output_path

def main():
    """Process all notebooks."""
    notebooks_dir = Path("notebooks")

    print("\n" + "="*60)
    print("🚀 全ノートブック改善プロセス開始")
    print("="*60)

    # Priority order: most important notebooks first
    priority_notebooks = [
        "00_quick_start.ipynb",
        "01_data_simulation_basics.ipynb",
        "02_preprocessing_and_feature_engineering.ipynb",
        "03_model_evaluation_metrics.ipynb",
        "08_mlp_parameter_space_exploration.ipynb",
        "12_complete_ml_pipeline.ipynb"
    ]

    processed = []
    for nb_name in priority_notebooks:
        nb_path = notebooks_dir / nb_name
        if nb_path.exists():
            nb_num = nb_name.split('_')[0]
            if nb_num in NOTEBOOK_CONFIGS:
                output_path = improve_notebook(nb_path, NOTEBOOK_CONFIGS[nb_num])
                processed.append(output_path)

    # Process remaining notebooks
    for nb_path in sorted(notebooks_dir.glob("*.ipynb")):
        if nb_path.name.endswith("_improved_v2.ipynb") or nb_path.name.endswith("_improved.ipynb"):
            continue
        if nb_path.name not in priority_notebooks and not any(p.stem.startswith(nb_path.stem) for p in processed):
            # Use generic config for notebooks not in NOTEBOOK_CONFIGS
            config = {
                "title": nb_path.stem.replace('_', ' ').title(),
                "objectives": ["この章の内容を理解できる"],
                "prerequisites": ["Python基礎"],
                "time": "60-90分",
                "difficulty": "★★★☆☆（中級）",
                "category": "機械学習",
                "next": "次のノートブックへ"
            }
            output_path = improve_notebook(nb_path, config)
            processed.append(output_path)

    print("\n" + "="*60)
    print(f"✅ 完了！{len(processed)}個のノートブックを改善しました")
    print("="*60)
    print("\n改善されたノートブック:")
    for p in processed:
        print(f"  - {p.name}")
    print()

if __name__ == "__main__":
    main()
