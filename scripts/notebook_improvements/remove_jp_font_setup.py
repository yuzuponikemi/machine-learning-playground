#!/usr/bin/env python3
"""remove_jp_font_setup.py — 日本語フォント設定コードを安全に除去する

CLAUDE.md の規約:
  「`japanize_matplotlib` は使用禁止（日本語フォントの表示が不安定なため）」
  「図のタイトル、軸ラベル、凡例、アノテーション等はすべて英語で記載すること」

`japanize_matplotlib` の import とは別に、**手書きの日本語フォント設定**が
多くのノートブックに入っている（`def setup_japanese_font()` / `rcParams['font.family']`）。
図テキストを英語化し終えたノートブックでは、この設定は死にコードであり、
規約の趣旨（日本語フォントに依存しない）にも反するので除去する。

**前提**: そのノートブックの図テキストが英語化済みであること。
`plot_text_i18n.py check` が 0 を返す状態で実行する。日本語の図テキストが
残っているノートブックからフォント設定だけ外すと、図が豆腐表示になって
着手前より悪くなるため、本ツールは既定で拒否する（--force で上書き可）。

除去する対象:
  1. `def setup_japanese_font(): ...` 関数定義まるごと
  2. `font_used = setup_japanese_font()` と、それに続く `if font_used:` ブロック
  3. `import matplotlib.font_manager as fm`（ノートブック内で `fm.` を他で使っていない場合のみ）
  4. `plt.rcParams['font.family'] = ...`（日本語フォント名を含む行）
  5. 上記に付随する `# 日本語フォント設定` などのコメント行

使い方:
  python3 scripts/notebook_improvements/remove_jp_font_setup.py check notebooks/<unit>/*.ipynb
  python3 scripts/notebook_improvements/remove_jp_font_setup.py apply notebooks/<unit>/*.ipynb
"""

from __future__ import annotations

import ast
import json
import re
import sys

JP_FONT_NAMES = ("Hiragino", "Yu Gothic", "MS Gothic", "Noto Sans CJK",
                 "IPAexGothic", "TakaoPGothic", "AppleGothic", "Arial Unicode MS")
FONT_COMMENT = re.compile(r"^\s*#.*(日本語フォント|japanese font|フォント設定)", re.I)


def load(p):
    return json.load(open(p))


def save(nb, p):
    json.dump(nb, open(p, "w"), ensure_ascii=False, indent=1)


def has_jp_font_code(src: str) -> bool:
    return ("setup_japanese_font" in src
            or any(n in src for n in JP_FONT_NAMES))


def _uses_fm_elsewhere(nb, skip_cell: int) -> bool:
    """fm.<something> を他のセルで使っているか（font_manager の import を残すべきか）。"""
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code" or i == skip_cell:
            continue
        if re.search(r"\bfm\.", "".join(c["source"])):
            return True
    return False


def clean_cell(src: str, nb, cell_index: int) -> str:
    """1セルからフォント設定を除去した source を返す。"""
    lines = src.split("\n")
    drop = set()

    # --- 1) def setup_japanese_font(...) を AST で範囲ごと落とす ---
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "setup_japanese_font":
                for ln in range(node.lineno - 1, (node.end_lineno or node.lineno)):
                    drop.add(ln)
            # font_used = setup_japanese_font()
            if isinstance(node, ast.Assign) and "setup_japanese_font" in ast.dump(node):
                for ln in range(node.lineno - 1, (node.end_lineno or node.lineno)):
                    drop.add(ln)
            # if font_used: print(...)
            if isinstance(node, ast.If) and "font_used" in ast.dump(node.test):
                for ln in range(node.lineno - 1, (node.end_lineno or node.lineno)):
                    drop.add(ln)

    # --- 2) 行単位の対象 ---
    uses_fm = _uses_fm_elsewhere(nb, cell_index)
    for i, ln in enumerate(lines):
        if i in drop:
            continue
        if "setup_japanese_font" in ln:
            drop.add(i); continue
        if re.search(r"rcParams\[['\"]font\.family['\"]\]", ln) and (
                any(n in ln for n in JP_FONT_NAMES) or "japanese" in ln.lower()):
            drop.add(i); continue
        if "font_manager" in ln and not uses_fm:
            drop.add(i); continue
        if FONT_COMMENT.match(ln):
            drop.add(i); continue
        if any(n in ln for n in JP_FONT_NAMES) and ln.strip().startswith("#"):
            drop.add(i); continue

    kept = [ln for i, ln in enumerate(lines) if i not in drop]

    # --- 3) 除去跡の空行が3つ以上続いたら2つに詰める ---
    out, blanks = [], 0
    for ln in kept:
        if ln.strip() == "":
            blanks += 1
            if blanks > 2:
                continue
        else:
            blanks = 0
        out.append(ln)
    return "\n".join(out)


def process(path: str, apply: bool, force: bool):
    nb = load(path)
    # 図テキストが英語化済みかを確認する（安全装置）
    if not force:
        from importlib.util import spec_from_file_location, module_from_spec
        import pathlib
        tool = pathlib.Path(__file__).with_name("plot_text_i18n.py")
        spec = spec_from_file_location("i18n", tool)
        mod = module_from_spec(spec); spec.loader.exec_module(mod)
        rows, _ = mod.extract(path)
        if rows:
            return f"SKIP (日本語の図テキストが {len(rows)} 箇所残っている)", 0

    changed = 0
    cleaned_sources = {}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if not has_jp_font_code(src):
            continue
        new = clean_cell(src, nb, i)
        if new == src:
            continue
        try:
            ast.parse(new)
        except SyntaxError as e:
            return f"SYNTAX BREAK cell {i}: {e.msg}", 0
        cleaned_sources[i] = new
        changed += 1
    if apply and changed:
        for i, new in cleaned_sources.items():
            nb["cells"][i]["source"] = new.splitlines(keepends=True)
        save(nb, path)
    # 残存チェックは「掃除後の姿」に対して行う（dry-run でも正しく数える）
    left = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = cleaned_sources.get(i, "".join(c["source"]))
        if has_jp_font_code(src):
            left += 1
    return f"cells cleaned={changed} remaining={left}", changed


if __name__ == "__main__":
    cmd = sys.argv[1]
    force = "--force" in sys.argv
    paths = [a for a in sys.argv[2:] if not a.startswith("--")]
    total = 0
    for p in paths:
        msg, n = process(p, apply=(cmd == "apply"), force=force)
        total += n
        print(f"{p}: {msg}")
    print(f"total cells {'cleaned' if cmd == 'apply' else 'would clean'}: {total}")
