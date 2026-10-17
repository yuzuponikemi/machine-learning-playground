#!/usr/bin/env python3
"""plot_text_i18n.py — 図（matplotlib）のテキストだけを英語化する補助ツール

CLAUDE.md の規約:
  「図のタイトル、軸ラベル、凡例、アノテーション等はすべて英語で記載すること」
  「japanize_matplotlib は使用禁止」
本文（Markdown セル）と print 出力の日本語は規約の対象外なので、**触ってはいけない**。
そのため文字列の素朴な一括置換は使えない。このツールは AST の位置情報を使い、
matplotlib のテキスト設定呼び出し（set_title / set_xlabel / text / annotate /
label= など）に渡されている文字列リテラルだけを書き換える。

使い方:

  # 1. 対象を数える
  python3 scripts/notebook_improvements/plot_text_i18n.py check notebooks/<unit>/*.ipynb

  # 2. 日本語を抽出（1行1件の JSON。翻訳の下書きに使う）
  python3 scripts/notebook_improvements/plot_text_i18n.py extract notebooks/<unit>/*.ipynb

  # 3. {"日本語": "English"} の対応表を書いて適用（japanize の import も削除される）
  python3 scripts/notebook_improvements/plot_text_i18n.py apply map.json notebooks/<unit>/*.ipynb

  # 4. check が 0 になったことを確認する

注意点:
  - f-string の literal 部分も個別に拾う。前後が繋がるので訳語の語順に注意
    （例: f'{x} での接線' → ' - tangent line' のように接尾辞として訳す）。
  - f-string 内の `{{` `}}`（エスケープされた波括弧）を含む文字列は自動置換しない
    （置換すると `{` 1個になり式として解釈されてしまう）。DEFER として報告されるので手で直す。
  - 置換後は必ず各セルを ast.parse で検証し、壊れるセルはスキップする。
  - 既存の出力セル（描画済み画像）は書き換えないので、図が英語になるのは次回実行時。
"""
import json, ast, re, sys, collections

JP = re.compile(r'[぀-ヿ㐀-䶿一-鿿！-｠　-〿]')
TEXT_FUNCS = {
    'set_title','set_xlabel','set_ylabel','set_zlabel','suptitle','title','xlabel','ylabel',
    'text','annotate','set_xticklabels','set_yticklabels','xticks','yticks','legend',
    'set_label','colorbar','bar_label','figtext','set_text','table','set_ylabels','set_xlabels',
}
KW_TEXT = {'label','title','xlabel','ylabel','suptitle','s','text','colLabels','rowLabels'}

def fname(call):
    f=call.func
    return f.attr if isinstance(f,ast.Attribute) else (f.id if isinstance(f,ast.Name) else None)

def targets(src):
    """-> list of (lineno, col, end_lineno, end_col, value, func)"""
    try: tree=ast.parse(src)
    except SyntaxError: return []
    out=[]
    def collect(node, label):
        for n in ast.walk(node):
            if isinstance(n, ast.Constant) and isinstance(n.value,str) and JP.search(n.value):
                out.append((n.lineno,n.col_offset,n.end_lineno,n.end_col_offset,n.value,label))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn=fname(node)
            if fn in TEXT_FUNCS:
                collect(node, fn)
            else:
                for kw in node.keywords:
                    if kw.arg in KW_TEXT: collect(kw, f'{fn}({kw.arg}=)')
    # dedupe identical positions (walk can visit nested calls twice)
    return sorted(set(out))

def load(p): return json.load(open(p))
def save(nb,p): json.dump(nb, open(p,'w'), ensure_ascii=False, indent=1)

def extract(p):
    nb=load(p); rows=[]
    for ci,c in enumerate(nb['cells']):
        if c['cell_type']!='code': continue
        for (l,co,el,eco,val,fn) in targets(''.join(c['source'])):
            rows.append({'cell':ci,'func':fn,'old':val})
    uniq=collections.OrderedDict()
    for r in rows: uniq.setdefault(r['old'], []).append((r['cell'],r['func']))
    return rows, uniq

def _nl(s):
    return (s.replace('\\', '\\\\').replace('\n', '\\n')
             .replace('\t', '\\t').replace('\r', '\\r'))

def variants(s):
    """ソース上でその文字列がどう書かれている可能性があるか（素のまま / \n 等をエスケープ /
    さらにクォートもエスケープ）を順に返す。(変換関数, 文字列) の組。"""
    base = _nl(s)
    return [
        (lambda x: x, s),
        (_nl, base),
        (lambda x: _nl(x).replace("'", "\\'"), base.replace("'", "\\'")),
        (lambda x: _nl(x).replace('"', '\\"'), base.replace('"', '\\"')),
    ]

def apply(p, mapping):
    nb=load(p); changed=0; missing=set()
    for c in nb['cells']:
        if c['cell_type']!='code': continue
        src=''.join(c['source'])
        tg=targets(src)
        if not tg: continue
        lines=src.split('\n')
        blines=[l.encode() for l in lines]
        deferred=[]
        for (l,co,el,eco,val,fn) in sorted(tg, key=lambda t:(-t[0],-t[1])):
            if val not in mapping: missing.add(val); continue
            new=mapping[val]
            if l!=el:
                deferred.append((val,new)); continue
            b=blines[l-1]
            try: segs=b[co:eco].decode()
            except UnicodeDecodeError: continue
            hit=None
            for conv, lit in variants(val):
                if lit in segs: hit=(conv,lit); break
            if hit is None:
                deferred.append((val,new)); continue
            conv, lit = hit
            rep=segs.replace(lit, conv(new))
            blines[l-1]=b[:co]+rep.encode()+b[eco:]
            changed+=1
        newsrc='\n'.join(x.decode() for x in blines)
        # 複数行リテラル・位置特定に失敗したものはセル全体で一意なときだけ置換する
        for val,new in deferred:
            done=False
            for conv, lit in variants(val):
                if newsrc.count(lit)==1:
                    newsrc=newsrc.replace(lit, conv(new)); changed+=1; done=True; break
            if not done: missing.add('DEFER:'+val)
        if newsrc!=src:
            try: ast.parse(newsrc)
            except SyntaxError as e:
                print(f'  !! syntax break, cell skipped: {e}'); continue
            c['source']=newsrc.splitlines(keepends=True)
    save(nb,p)
    return changed, missing

def drop_japanize(p):
    nb=load(p); n=0
    for c in nb['cells']:
        if c['cell_type']!='code': continue
        src=''.join(c['source'])
        if 'japanize_matplotlib' not in src: continue
        keep=[ln for ln in src.split('\n')
              if 'japanize_matplotlib' not in ln]
        new='\n'.join(keep)
        if new!=src:
            ast.parse(new); c['source']=new.splitlines(keepends=True); n+=1
    save(nb,p)
    return n

if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='extract':
        for p in sys.argv[2:]:
            rows,uniq=extract(p)
            print(f'### {p}  ({len(rows)} instances, {len(uniq)} unique)')
            for k,v in uniq.items():
                print(json.dumps({'old':k,'funcs':sorted({f for _,f in v})}, ensure_ascii=False))
    elif cmd=='apply':
        mapping=json.load(open(sys.argv[2]))
        tot=0
        for p in sys.argv[3:]:
            ch,miss=apply(p,mapping); dj=drop_japanize(p)
            rows,_=extract(p)
            print(f'{p}: replaced={ch} japanize_cells_cleaned={dj} remaining_jp={len(rows)}')
            if miss: print('   unmapped:', json.dumps(sorted(miss),ensure_ascii=False)[:600])
            tot+=ch
        print('total replaced', tot)
    elif cmd=='check':
        for p in sys.argv[2:]:
            rows,uniq=extract(p)
            nb=json.load(open(p))
            jz='JZ' if 'japanize_matplotlib' in json.dumps(nb) else '--'
            print(f'{jz} {len(rows):4d} jp-instances  {p}')
