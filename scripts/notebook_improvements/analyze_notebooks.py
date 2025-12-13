#!/usr/bin/env python3
"""Analyze notebook structure for improvement planning."""

import json
from pathlib import Path

notebooks_dir = Path("notebooks")

print("📚 機械学習教科書 - 現在のノートブック構成\n")
print("=" * 80)

for nb_path in sorted(notebooks_dir.glob("*.ipynb")):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Get first cell (usually title)
    first_cell = nb['cells'][0]
    if first_cell['cell_type'] == 'markdown':
        title_lines = first_cell['source']
        title = ''.join(title_lines).split('\n')[0].replace('#', '').strip()
    else:
        title = "No title"

    # Count cells
    n_code = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    n_markdown = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')

    # Check for learning objectives
    has_objectives = any(
        'learning objectives' in ''.join(c.get('source', [])).lower()
        for c in nb['cells'] if c['cell_type'] == 'markdown'
    )

    # Check for summary
    has_summary = any(
        'summary' in ''.join(c.get('source', [])).lower()
        for c in nb['cells'] if c['cell_type'] == 'markdown'
    )

    print(f"\n📘 {nb_path.name}")
    print(f"   タイトル: {title}")
    print(f"   セル数: {n_code} コード, {n_markdown} マークダウン")
    print(f"   学習目標: {'✅' if has_objectives else '❌'}")
    print(f"   まとめ: {'✅' if has_summary else '❌'}")

print("\n" + "=" * 80)
