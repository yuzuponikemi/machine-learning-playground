#!/usr/bin/env python3
"""
全ノートブックでプレースホルダーや不完全な説明を検出
"""

import json
from pathlib import Path
import re

def find_placeholders_in_notebook(nb_path):
    """ノートブック内のプレースホルダーを検出"""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    issues = []

    # プレースホルダーパターン
    placeholder_patterns = [
        r'\[.*?\]',  # [何か]
        r'\.\.\.+',  # ...
        r'TODO',
        r'FIXME',
        r'XXX',
        r'この章の重要性',
        r'具体的な応用例',
        r'ここに.*?を記述',
        r'以下を.*?してください',
    ]

    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell.get('source', []))

            # プレースホルダー検出
            for pattern in placeholder_patterns:
                matches = re.findall(pattern, source, re.IGNORECASE)
                if matches:
                    # コンテキストを取得（前後の文字）
                    for match in matches:
                        # 明らかなMarkdown記法は除外
                        if match in ['[Top]', '[Back]']:
                            continue

                        # 見出しやリンクの一部でない場合のみ
                        context_start = max(0, source.find(match) - 50)
                        context_end = min(len(source), source.find(match) + len(match) + 50)
                        context = source[context_start:context_end]

                        issues.append({
                            'cell_idx': cell_idx,
                            'pattern': pattern,
                            'match': match,
                            'context': context.strip()
                        })

    return issues

def main():
    """メイン処理"""
    print("=" * 70)
    print("🔍 全ノートブックでプレースホルダーを検索")
    print("=" * 70)
    print()

    notebooks_dir = Path("notebooks")
    notebooks = sorted(notebooks_dir.glob("*_improved_v2.ipynb"))

    all_issues = {}

    for nb_path in notebooks:
        issues = find_placeholders_in_notebook(nb_path)
        if issues:
            all_issues[nb_path.name] = issues

    # 結果を表示
    if all_issues:
        print(f"⚠️  {len(all_issues)}個のノートブックで問題を検出しました\n")

        for nb_name, issues in all_issues.items():
            print(f"\n{'='*70}")
            print(f"📓 {nb_name}")
            print(f"{'='*70}")

            for issue in issues:
                print(f"\nセル {issue['cell_idx']}:")
                print(f"  パターン: {issue['pattern']}")
                print(f"  検出: {issue['match']}")
                print(f"  コンテキスト: {issue['context'][:100]}...")
                print()
    else:
        print("✅ プレースホルダーは見つかりませんでした！")

    print("\n" + "=" * 70)
    print(f"検索完了: {len(notebooks)}個のノートブックをチェック")
    print("=" * 70)

if __name__ == "__main__":
    main()
