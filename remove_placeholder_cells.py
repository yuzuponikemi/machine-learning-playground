#!/usr/bin/env python3
"""
プレースホルダーを含むセルを削除する
"""

import json
from pathlib import Path

def remove_placeholder_cells(nb_path):
    """プレースホルダーセルを削除"""
    print(f"\n処理中: {nb_path.name}")

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    original_count = len(nb['cells'])
    new_cells = []
    removed_count = 0

    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown' and cell.get('source'):
            source = ''.join(cell['source'])

            # プレースホルダーを含むセルをスキップ
            if '[この章の重要性を説明]' in source and '[具体的な応用例]' in source:
                print(f"  削除: プレースホルダーセル")
                removed_count += 1
                continue

        new_cells.append(cell)

    nb['cells'] = new_cells

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"  完了: {original_count} → {len(new_cells)} セル ({removed_count}個削除)")
    return removed_count

def main():
    """メイン処理"""
    print("=" * 70)
    print("🗑️  プレースホルダーセルを削除")
    print("=" * 70)

    notebooks_dir = Path("notebooks")
    notebooks = sorted(notebooks_dir.glob("*_improved_v2.ipynb"))

    total_removed = 0
    for nb_path in notebooks:
        removed = remove_placeholder_cells(nb_path)
        total_removed += removed

    print("\n" + "=" * 70)
    print(f"✅ 完了！合計{total_removed}個のプレースホルダーセルを削除しました")
    print("=" * 70)

if __name__ == "__main__":
    main()
