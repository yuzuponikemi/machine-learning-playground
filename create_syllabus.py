#!/usr/bin/env python3
"""Create a comprehensive syllabus from all notebooks"""

import json
import re
from pathlib import Path
from datetime import datetime

def extract_notebook_info(notebook_path):
    """Extract title, overview, and learning objectives from notebook."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        cells = nb.get('cells', [])

        title = ""
        overview = ""
        objectives = []

        # Extract from first markdown cell
        for cell in cells[:5]:  # Check first 5 cells
            if cell.get('cell_type') == 'markdown':
                source = ''.join(cell.get('source', []))

                # Extract title (first # heading)
                if not title:
                    title_match = re.search(r'^#\s+(.+?)$', source, re.MULTILINE)
                    if title_match:
                        title = title_match.group(1).strip()

                # Extract overview
                if '概要' in source or 'Overview' in source:
                    overview_match = re.search(r'(?:概要|Overview)[^\n]*\n(.+?)(?=\n##|\Z)', source, re.DOTALL)
                    if overview_match:
                        overview = overview_match.group(1).strip()

                # Extract learning objectives
                if '学習目標' in source or 'Learning' in source:
                    obj_section = re.search(r'(?:学習目標|学習目的)[^\n]*\n(.+?)(?=\n##|\Z)', source, re.DOTALL)
                    if obj_section:
                        obj_text = obj_section.group(1)
                        # Extract bullet points
                        objectives = re.findall(r'[-*]\s+(.+)', obj_text)

        return {
            'title': title,
            'overview': overview[:200] if overview else "",
            'objectives': objectives[:5] if objectives else []
        }

    except Exception as e:
        return {
            'title': notebook_path.name,
            'overview': f"Error: {str(e)}",
            'objectives': []
        }

def create_syllabus():
    """Create comprehensive syllabus document."""
    notebooks_dir = Path('notebooks')
    notebooks = sorted(notebooks_dir.glob('*_improved_v2.ipynb'))

    syllabus = []

    # Header
    syllabus.append("=" * 100)
    syllabus.append("機械学習完全マスター講座 - 総合シラバス")
    syllabus.append("Machine Learning Complete Master Course - Comprehensive Syllabus")
    syllabus.append("=" * 100)
    syllabus.append(f"作成日: {datetime.now().strftime('%Y年%m月%d日')}")
    syllabus.append(f"総ノートブック数: {len(notebooks)}")
    syllabus.append("=" * 100)
    syllabus.append("")

    # Table of Contents
    syllabus.append("📚 目次 (TABLE OF CONTENTS)")
    syllabus.append("=" * 100)
    syllabus.append("")

    # Group notebooks by category
    categories = {
        '基礎編 (Fundamentals)': range(0, 7),
        'ニューラルネットワーク編 (Neural Networks)': range(7, 13),
        'GBDT・実践編 (GBDT & Practice)': range(13, 20),
        '高度な手法編 (Advanced Techniques)': range(20, 29)
    }

    notebook_info_list = []
    for nb_path in notebooks:
        nb_num = int(nb_path.name.split('_')[0])
        info = extract_notebook_info(nb_path)
        info['number'] = nb_num
        info['filename'] = nb_path.name
        notebook_info_list.append(info)

    for category, num_range in categories.items():
        syllabus.append(f"\n【{category}】")
        syllabus.append("-" * 100)

        for info in notebook_info_list:
            if info['number'] in num_range:
                num_str = f"{info['number']:02d}"
                title = info['title'] or info['filename']
                syllabus.append(f"  {num_str}. {title}")
        syllabus.append("")

    syllabus.append("=" * 100)
    syllabus.append("")

    # Detailed content for each notebook
    syllabus.append("📖 詳細内容 (DETAILED CONTENTS)")
    syllabus.append("=" * 100)
    syllabus.append("")

    for info in notebook_info_list:
        num_str = f"{info['number']:02d}"
        title = info['title'] or info['filename']

        syllabus.append(f"\n{'=' * 100}")
        syllabus.append(f"Notebook {num_str}: {title}")
        syllabus.append(f"{'=' * 100}")

        if info['overview']:
            syllabus.append(f"\n概要:")
            syllabus.append(f"  {info['overview']}")

        if info['objectives']:
            syllabus.append(f"\n学習目標:")
            for i, obj in enumerate(info['objectives'], 1):
                syllabus.append(f"  {i}. {obj}")

        syllabus.append("")

    # Index - Key topics
    syllabus.append("\n" + "=" * 100)
    syllabus.append("🔍 キーワード索引 (KEYWORD INDEX)")
    syllabus.append("=" * 100)
    syllabus.append("")

    keywords_map = {
        'データ前処理': [0, 1, 2, 12],
        '特徴量エンジニアリング': [2, 15, 24, 25],
        'モデル評価': [3, 11, 12],
        '線形モデル': [4],
        '決定木・アンサンブル': [5, 13, 14],
        'SVM': [6],
        'ニューラルネットワーク': [7, 8, 9, 26],
        'ハイパーパラメータ最適化': [10, 20],
        'GBDT (XGBoost/LightGBM/CatBoost)': [13, 14, 16, 17, 18, 19],
        'Kaggle実践': [15, 16, 17, 18, 19, 27, 28],
        'SHAP (モデル解釈)': [21],
        'Stacking': [22],
        '不均衡データ': [23],
        '時系列': [19, 24],
        'カテゴリカル変数': [14, 25],
        'TabNet (Deep Learning)': [26],
        'Optuna': [20],
    }

    for keyword, nb_nums in sorted(keywords_map.items()):
        nb_list = ', '.join([f"{n:02d}" for n in nb_nums])
        syllabus.append(f"  • {keyword:<35} → Notebooks: {nb_list}")

    syllabus.append("")
    syllabus.append("=" * 100)
    syllabus.append("END OF SYLLABUS")
    syllabus.append("=" * 100)

    return '\n'.join(syllabus)

if __name__ == '__main__':
    syllabus_text = create_syllabus()

    # Save to file
    output_file = Path('SYLLABUS.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(syllabus_text)

    print(syllabus_text)
    print(f"\n✅ Syllabus saved to: {output_file}")
