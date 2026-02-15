#!/usr/bin/env python3
"""Build notebook 146 as JSON."""
import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n"), "id": None}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source.split("\n"),
            "execution_count": None, "outputs": [], "id": None}

cells = []

# ── Cell 0: Title ──────────────────────────────────────────
cells.append(md(
r"""# 第146章: 世界モデル総括 — Phase 7の統合と未来展望

## この章で学ぶこと

この章を終えると、以下ができるようになります：

- [ ] Phase 7（Notebook 140-145）で学んだ技術の全体像を説明できる
- [ ] 3つの予測パラダイム（ピクセル/特徴/潜在行動）の違いと特徴を比較できる
- [ ] 世界モデルの統一的な構造（表現→予測→行動）を理解できる
- [ ] Phase 1-7 の全カリキュラムを俯瞰し、知識の繋がりを説明できる
- [ ] AGIに向けた世界モデルの未来展望を議論できる

## 前提知識

- Notebook 140: 表現学習の基礎（対照学習、BYOL）
- Notebook 141: JEPA — Joint Embedding Predictive Architecture
- Notebook 142: モデルベース強化学習の基礎
- Notebook 143: DreamerV3 — 潜在空間での想像による計画
- Notebook 144: Genie — 潜在行動発見と世界モデル生成
- Notebook 145: 世界モデルの応用と発展

---

| 項目 | 内容 |
|------|------|
| 推定学習時間 | 90-120分 |
| 難易度 | ★★★★☆（上級） |
| カテゴリ | Phase 7 総括・統合 |
| 必要ライブラリ | numpy, matplotlib |"""
))

# ── Cell 1: TOC ──────────────────────────────────────────
cells.append(md(
r"""## 目次

1. [Phase 7で学んだ技術体系](#section1)
2. [3つの予測パラダイム比較](#section2)
3. [世界モデルの統一的理解](#section3)
4. [全Phase統合マップ（Phase 1-7）](#section4)
5. [AGIへの道 — 世界モデルの未来](#section5)
6. [総合クイズ（5問）](#section6)
7. [学習の旅の完結](#section7)"""
))

# ── Cell 2: Setup ──────────────────────────────────────────
cells.append(code(
r"""# ============================================================
# 環境設定
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import warnings

warnings.filterwarnings('ignore')

# 日本語フォント設定
import matplotlib.font_manager as fm

def setup_japanese_font():
    japanese_fonts = [
        'Hiragino Sans', 'Hiragino Maru Gothic Pro',
        'Yu Gothic', 'MS Gothic',
        'Noto Sans CJK JP', 'IPAexGothic',
    ]
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    for font in japanese_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            return font
    try:
        import japanize_matplotlib
        return 'japanize_matplotlib'
    except ImportError:
        return None

font_used = setup_japanese_font()
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

np.random.seed(42)

print(f'日本語フォント: {font_used}')
print('環境設定完了')"""
))

# ── Cell 3: Section 1 header ──────────────────────────────
cells.append(md(
r"""<a id="section1"></a>
## 1. Phase 7で学んだ技術体系

Phase 7「世界モデル」では、6つのノートブック（140-145）を通じて、
エージェントが**世界の内部モデル**を構築し、そのモデルを使って**予測・計画・行動**する技術を学びました。

各ノートブックは前のノートブックの知識を土台に、段階的に高度な概念を積み上げています。
まず、その全体像をフロー図で確認しましょう。"""
))

# ── Cell 4: Phase 7 flow diagram ──────────────────────────
cells.append(code(
r"""def visualize_phase7_flow():
    """Phase 7 カリキュラムのフロー図"""
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    notebooks = [
        (0.15, 0.88, '140\n表現学習の基礎', '#4ECDC4',
         ['対照学習 (SimCLR)', 'BYOL / 非対照手法', '表現の質の評価']),
        (0.50, 0.88, '141\nJEPA', '#45B7D1',
         ['Joint Embedding', '予測的アーキテクチャ', 'I-JEPA / V-JEPA']),
        (0.85, 0.88, '142\nモデルベースRL', '#96CEB4',
         ['環境モデル学習', 'Model-based vs Free', '計画と想像']),
        (0.25, 0.52, '143\nDreamerV3', '#FFEAA7',
         ['RSSM', '潜在空間での想像', 'Actor-Critic']),
        (0.60, 0.52, '144\nGenie', '#DDA0DD',
         ['潜在行動発見', 'ST-Transformer', 'ビデオからの学習']),
        (0.42, 0.18, '145\n応用と発展', '#FF6B6B',
         ['UniSim / Genie2', 'ロボティクス応用', '基盤世界モデル']),
    ]

    for x, y, title, color, topics in notebooks:
        box = FancyBboxPatch((x - 0.12, y - 0.12), 0.24, 0.20,
                             boxstyle='round,pad=0.02',
                             facecolor=color, edgecolor='#333333',
                             linewidth=2, alpha=0.85)
        ax.add_patch(box)
        ax.text(x, y + 0.04, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color='#222222')
        topic_str = '\n'.join([f'  {t}' for t in topics])
        ax.text(x, y - 0.06, topic_str, ha='center', va='center',
                fontsize=8, color='#444444')

    arrows = [
        (0.27, 0.88, 0.38, 0.88),
        (0.62, 0.88, 0.73, 0.88),
        (0.20, 0.76, 0.25, 0.64),
        (0.55, 0.76, 0.55, 0.64),
        (0.80, 0.76, 0.65, 0.64),
        (0.30, 0.40, 0.38, 0.30),
        (0.55, 0.40, 0.47, 0.30),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555555',
                                    lw=2.0, connectionstyle='arc3,rad=0.1'))

    ax.text(0.50, 0.98, 'Phase 7: 世界モデル  カリキュラムマップ',
            ha='center', va='center', fontsize=18, fontweight='bold',
            color='#222222')

    ax.text(0.50, 0.94, '表現学習 → 予測アーキテクチャ → 計画・行動 → 応用',
            ha='center', va='center', fontsize=12, color='#666666',
            style='italic')

    plt.tight_layout()
    plt.show()

visualize_phase7_flow()"""
))

# ── Cell 5: Key concepts text ──────────────────────────────
cells.append(code(
r"""def print_phase7_summary():
    """各ノートブックの核心概念をまとめる"""
    print("=" * 70)
    print("Phase 7: 各ノートブックの核心概念")
    print("=" * 70)
    print("""
【Notebook 140: 表現学習の基礎】
  核心: 世界を理解するには、まず良い「表現」が必要
  - 対照学習: 同じ画像の異なるビュー → 近く、異なる画像 → 遠く
  - BYOL: ネガティブサンプル不要の自己教師あり学習
  - 学んだこと: 表現の質がすべての下流タスクの性能を決める

【Notebook 141: JEPA】
  核心: ピクセルを再構成するのではなく、特徴空間で予測する
  - Joint Embedding: 入力と目標を共通の特徴空間に埋め込む
  - Predictive: 一部の特徴から他の部分の特徴を予測する
  - MAEとの違い: ピクセル再構成 vs 特徴予測

【Notebook 142: モデルベース強化学習】
  核心: 環境のモデルを学習し、「想像」の中で計画を立てる
  - Model-based RL: 遷移関数 p(s'|s,a) を学習
  - Model-free RL: 方策や価値関数を直接学習
  - 利点: サンプル効率が高い（実環境での試行が少なくて済む）

【Notebook 143: DreamerV3】
  核心: 潜在空間で「夢を見る」ことで効率的に学習
  - RSSM: 決定的状態 + 確率的状態のハイブリッド世界モデル
  - Imagination: 学習した世界モデル内でロールアウト
  - Actor-Critic: 想像上の経験から方策と価値を学習

【Notebook 144: Genie】
  核心: ラベルなし動画から行動の概念を自動発見
  - Latent Action Model: 連続フレーム間の「行動」を潜在空間で推定
  - ST-Transformer: 時空間的に次フレームを予測
  - 意義: 教師なしで「インタラクティブ環境」を生成

【Notebook 145: 応用と発展】
  核心: 世界モデルは汎用AIの基盤技術になりうる
  - Foundation World Models: 大規模な汎用世界モデル
  - ロボティクス: ビデオから物理法則を学習
  - 課題: 長期予測、物理整合性、スケーラビリティ
    """)

print_phase7_summary()"""
))

# ── Cell 6: Section 2 header ──────────────────────────────
cells.append(md(
r"""<a id="section2"></a>
## 2. 3つの予測パラダイム比較

Phase 7を通じて、世界を「予測する」ための3つの根本的に異なるアプローチを学びました。

| パラダイム | 代表手法 | 予測対象 | Phase 7での位置 |
|-----------|---------|---------|----------------|
| ピクセル予測 | VideoGPT, 自己回帰モデル | 生のピクセル値 | 背景知識 |
| 特徴予測 | JEPA, 対照学習 | 潜在特徴ベクトル | Notebook 140-141 |
| 潜在行動予測 | Genie, DreamerV3 | 行動条件付き状態遷移 | Notebook 142-144 |

これら3つのパラダイムの違いを詳しく比較しましょう。"""
))

# ── Cell 7: Paradigm comparison table ──────────────────────
cells.append(code(
r"""def print_paradigm_comparison():
    """3つの予測パラダイムの詳細比較"""
    print("=" * 75)
    print("3つの予測パラダイム詳細比較")
    print("=" * 75)
    print("""
┌──────────────┬────────────────────┬────────────────────┬────────────────────┐
│              │ ピクセル予測        │ 特徴予測           │ 潜在行動予測        │
├──────────────┼────────────────────┼────────────────────┼────────────────────┤
│ 予測対象      │ 生のピクセル値      │ 抽象的な特徴       │ 行動条件付き状態    │
│              │ (H x W x 3)       │ (d次元ベクトル)    │ (s_{t+1}|s_t, a)  │
├──────────────┼────────────────────┼────────────────────┼────────────────────┤
│ 代表手法      │ VideoGPT           │ JEPA, SimCLR       │ DreamerV3, Genie   │
│              │ Video Diffusion    │ BYOL, MAE          │ World Models       │
├──────────────┼────────────────────┼────────────────────┼────────────────────┤
│ 損失関数      │ MSE / CrossEntropy │ コサイン類似度      │ KL + 再構成        │
│              │ (ピクセル空間)      │ (特徴空間)         │ (潜在空間)         │
├──────────────┼────────────────────┼────────────────────┼────────────────────┤
│ 長所          │ 視覚的に評価可能    │ 意味的な情報を捉える │ 行動・制御が可能    │
│              │ 生成品質が明確      │ ノイズに頑健       │ 計画が立てられる    │
├──────────────┼────────────────────┼────────────────────┼────────────────────┤
│ 短所          │ 計算コストが高い    │ 直接生成できない    │ 報酬設計が必要      │
│              │ ノイズも予測する    │ 評価が間接的       │ モデル誤差の蓄積    │
├──────────────┼────────────────────┼────────────────────┼────────────────────┤
│ 精度          │ ★★★★☆            │ ★★★☆☆            │ ★★★★☆            │
│ 効率          │ ★★☆☆☆            │ ★★★★★            │ ★★★★☆            │
│ 解釈性        │ ★★★★★            │ ★★☆☆☆            │ ★★★☆☆            │
│ 拡張性        │ ★★★☆☆            │ ★★★★★            │ ★★★★☆            │
└──────────────┴────────────────────┴────────────────────┴────────────────────┘
    """)

print_paradigm_comparison()"""
))

# ── Cell 8: Radar chart ──────────────────────────────────
cells.append(code(
r"""def radar_chart_paradigms():
    """3つのパラダイムをレーダーチャートで比較"""
    categories = ['予測精度', '計算効率', '解釈性', 'スケーラビリティ',
                  '行動可能性', '汎用性']
    N = len(categories)

    pixel    = [4.0, 2.0, 5.0, 3.0, 1.5, 3.0]
    feature  = [3.0, 5.0, 2.0, 5.0, 2.0, 4.5]
    latent   = [4.0, 4.0, 3.0, 4.0, 5.0, 4.0]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    pixel   += pixel[:1]
    feature += feature[:1]
    latent  += latent[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    ax.plot(angles, pixel, 'o-', linewidth=2.5, color='#E74C3C', label='ピクセル予測', markersize=8)
    ax.fill(angles, pixel, alpha=0.12, color='#E74C3C')
    ax.plot(angles, feature, 's-', linewidth=2.5, color='#3498DB', label='特徴予測', markersize=8)
    ax.fill(angles, feature, alpha=0.12, color='#3498DB')
    ax.plot(angles, latent, 'D-', linewidth=2.5, color='#2ECC71', label='潜在行動予測', markersize=8)
    ax.fill(angles, latent, alpha=0.12, color='#2ECC71')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13)
    ax.set_ylim(0, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=9, color='gray')
    ax.set_title('3つの予測パラダイム比較', fontsize=16, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

radar_chart_paradigms()"""
))

# ── Cell 9: Grouped bar chart ──────────────────────────────
cells.append(code(
r"""def grouped_bar_paradigms():
    """3つのパラダイムをグループ棒グラフで比較"""
    categories = ['予測精度', '計算効率', '解釈性', 'スケーラビリティ',
                  '行動可能性', '汎用性']
    pixel   = [4.0, 2.0, 5.0, 3.0, 1.5, 3.0]
    feature = [3.0, 5.0, 2.0, 5.0, 2.0, 4.5]
    latent  = [4.0, 4.0, 3.0, 4.0, 5.0, 4.0]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width, pixel,   width, label='ピクセル予測',   color='#E74C3C', alpha=0.85)
    bars2 = ax.bar(x,         feature, width, label='特徴予測',       color='#3498DB', alpha=0.85)
    bars3 = ax.bar(x + width, latent,  width, label='潜在行動予測',   color='#2ECC71', alpha=0.85)

    ax.set_xlabel('評価軸', fontsize=13)
    ax.set_ylabel('スコア (5点満点)', fontsize=13)
    ax.set_title('予測パラダイム別スコア比較', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 6)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

grouped_bar_paradigms()"""
))

# ── Cell 10: Section 3 header ──────────────────────────────
cells.append(md(
r"""<a id="section3"></a>
## 3. 世界モデルの統一的理解

### 表現 → 予測 → 行動: 共通の構造

Phase 7で学んだ全てのアプローチは、根本的に同じ3段階の構造を共有しています:

1. **表現 (Representation)**: 生の観測を意味のある潜在表現に変換する
2. **予測 (Prediction)**: 現在の状態から未来の状態を予測する
3. **行動 (Action)**: 予測に基づいて最適な行動を選択する

手法ごとに各段階の実装は異なりますが、この骨格は共通です。"""
))

# ── Cell 11: Unified structure table ──────────────────────
cells.append(code(
r"""def print_unified_structure():
    """統一構造の比較テーブル"""
    print("=" * 75)
    print("世界モデルの統一構造: 表現 → 予測 → 行動")
    print("=" * 75)
    print("""
┌──────────┬─────────────────┬─────────────────┬─────────────────┐
│ 段階      │ DreamerV3        │ Genie            │ JEPA             │
├──────────┼─────────────────┼─────────────────┼─────────────────┤
│          │ CNN/MLP          │ VQ-VAE           │ ViT エンコーダ    │
│ 表現      │ エンコーダ        │ エンコーダ        │ (コンテキスト)    │
│          │ → 潜在状態 z_t   │ → 離散トークン    │ → 特徴ベクトル    │
├──────────┼─────────────────┼─────────────────┼─────────────────┤
│          │ RSSM             │ Dynamics         │ Predictor        │
│ 予測      │ (GRU + 確率モデル)│ Backbone         │ (特徴空間での     │
│          │ → z_{t+1}予測    │ → 次フレーム予測  │  マスク領域予測)  │
├──────────┼─────────────────┼─────────────────┼─────────────────┤
│          │ Actor-Critic     │ Latent Action    │ (下流タスクで     │
│ 行動      │ (方策 + 価値関数) │ Model            │  線形プローブ等)  │
│          │ → 最適行動選択   │ → 行動を自動発見  │ → 転移学習       │
├──────────┼─────────────────┼─────────────────┼─────────────────┤
│ 学習信号   │ 報酬 + KL正則化  │ 再構成 + VQ損失  │ 特徴予測誤差      │
│ 行動監視   │ あり（明示的）   │ なし（自動発見）  │ なし             │
│ 計画能力   │ あり（想像）     │ あり（生成）     │ なし（表現のみ）  │
└──────────┴─────────────────┴─────────────────┴─────────────────┘

共通点:
  - 全て「潜在空間」で動作する（生のピクセルではない）
  - 全て「自己教師あり」の要素を持つ
  - 全て「予測」が中核にある

相違点:
  - 行動の扱い: 明示的（DreamerV3）/ 自動発見（Genie）/ なし（JEPA）
  - 目的: 制御（DreamerV3）/ 生成（Genie）/ 表現学習（JEPA）
  - 評価: 報酬（DreamerV3）/ 生成品質（Genie）/ 下流タスク（JEPA）
    """)

print_unified_structure()"""
))

# ── Cell 12: Unified structure diagram ──────────────────────
cells.append(code(
r"""def visualize_unified_structure():
    """統一構造の図解"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    methods = [
        ('DreamerV3', '#FFEAA7', [
            ('観測 o_t', 0.5, 0.92, '#FFD93D'),
            ('エンコーダ\n(CNN)', 0.5, 0.75, '#4ECDC4'),
            ('RSSM\n(GRU + 確率)', 0.5, 0.55, '#45B7D1'),
            ('想像\nロールアウト', 0.5, 0.35, '#96CEB4'),
            ('Actor-Critic\n行動選択', 0.5, 0.15, '#FF6B6B'),
        ]),
        ('Genie', '#DDA0DD', [
            ('動画フレーム', 0.5, 0.92, '#FFD93D'),
            ('VQ-VAE\nエンコーダ', 0.5, 0.75, '#4ECDC4'),
            ('ST-Transformer\nDynamics', 0.5, 0.55, '#45B7D1'),
            ('次フレーム\n予測・生成', 0.5, 0.35, '#96CEB4'),
            ('Latent Action\n自動発見', 0.5, 0.15, '#DDA0DD'),
        ]),
        ('JEPA', '#AED6F1', [
            ('画像パッチ', 0.5, 0.92, '#FFD93D'),
            ('ViT\nエンコーダ', 0.5, 0.75, '#4ECDC4'),
            ('Predictor\n(特徴予測)', 0.5, 0.55, '#45B7D1'),
            ('ターゲット\n特徴との整合', 0.5, 0.35, '#96CEB4'),
            ('下流タスク\n転移学習', 0.5, 0.15, '#AED6F1'),
        ]),
    ]

    stage_labels = ['入力', '表現', '予測', '出力', '行動/応用']

    for ax, (method_name, bg_color, components) in zip(axes, methods):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        bg = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                            boxstyle='round,pad=0.02',
                            facecolor=bg_color, alpha=0.15,
                            edgecolor='gray', linewidth=1)
        ax.add_patch(bg)

        for i, (label, x, y, color) in enumerate(components):
            box = FancyBboxPatch((x - 0.2, y - 0.06), 0.4, 0.12,
                                 boxstyle='round,pad=0.02',
                                 facecolor=color, edgecolor='#333333',
                                 linewidth=1.5, alpha=0.8)
            ax.add_patch(box)
            ax.text(x, y, label, ha='center', va='center',
                    fontsize=9, fontweight='bold')

            ax.text(0.05, y, stage_labels[i], ha='left', va='center',
                    fontsize=8, color='#888888', style='italic')

            if i < len(components) - 1:
                next_y = components[i + 1][2]
                ax.annotate('', xy=(0.5, next_y + 0.06), xytext=(0.5, y - 0.06),
                            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

        ax.set_title(method_name, fontsize=15, fontweight='bold', pad=10)

    fig.suptitle('世界モデルの統一構造: 表現 → 予測 → 行動',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

visualize_unified_structure()"""
))

# ── Cell 13: Section 4 header ──────────────────────────────
cells.append(md(
r"""<a id="section4"></a>
## 4. 全Phase統合マップ（Phase 1-7）

### カリキュラム全体の俯瞰

Phase 1から Phase 7まで、私たちは機械学習の基礎から世界モデルまでの長い旅を歩んできました。
ここで全体像を振り返り、各Phaseがどのように繋がっているかを確認しましょう。

| Phase | テーマ | 範囲 | 核心 |
|-------|--------|------|------|
| Phase 1 | 統計・確率の基礎 | Nb 30-34 | データの記述と確率モデル |
| Phase 2 | 最適化の基礎 | Nb 35, 70-76, 110-116 | 勾配降下法、バックプロパゲーション |
| Phase 3 | ニューラルネットワーク | Nb 80-102 | CNN、帰納バイアス、空間理解 |
| Phase 4 | 生成モデル | Nb 36-45 | VAE、拡散モデル、条件付き生成 |
| Phase 5 | 3D視覚理解 | Nb 50-63 | カメラモデル、NeRF、3D再構成 |
| Phase 6 | 時空間モデリング | Nb 130-131 | Temporal Attention、Video Diffusion |
| Phase 7 | 世界モデル | Nb 140-146 | 表現学習、JEPA、DreamerV3、Genie |"""
))

# ── Cell 14: Grand phase map visualization ──────────────────
cells.append(code(
r"""def visualize_grand_phase_map():
    """Phase 1-7 の全体マップを可視化"""
    fig, ax = plt.subplots(figsize=(22, 14))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    phases = [
        (0.08, 0.85, 0.16, 0.10, 'Phase 1\n統計・確率', '#E8D5B7',
         '正規分布\nMLE\nGMM\nEM'),
        (0.30, 0.85, 0.16, 0.10, 'Phase 2\n最適化', '#B7D5E8',
         '勾配降下\nBackprop\nAdam\nスケジューリング'),
        (0.52, 0.85, 0.16, 0.10, 'Phase 3\nCNN・空間', '#D5E8B7',
         '畳み込み\n受容野\n帰納バイアス\nU-Net'),
        (0.74, 0.85, 0.16, 0.10, 'Phase 4\n生成モデル', '#E8B7D5',
         'VAE\n拡散モデル\nLDM\nCLIP'),
        (0.20, 0.55, 0.16, 0.10, 'Phase 5\n3Dビジョン', '#D5B7E8',
         'カメラモデル\nSfM\nNeRF\n3D再構成'),
        (0.50, 0.55, 0.16, 0.10, 'Phase 6\n時空間', '#B7E8D5',
         'Temporal Attn\nVideo Diffusion\nDiT\nST-Block'),
        (0.35, 0.22, 0.30, 0.12, 'Phase 7\n世界モデル', '#FFD700',
         '表現学習 / JEPA / Model-based RL\nDreamerV3 / Genie / 基盤世界モデル'),
    ]

    for x, y, w, h, title, color, desc in phases:
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle='round,pad=0.015',
                             facecolor=color, edgecolor='#333333',
                             linewidth=2.5, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.7, title, ha='center', va='center',
                fontsize=12, fontweight='bold', color='#222222')
        ax.text(x + w / 2, y + h * 0.25, desc, ha='center', va='center',
                fontsize=8, color='#444444')

    flow_arrows = [
        (0.24, 0.88, 0.30, 0.88, 'データ→最適化'),
        (0.46, 0.88, 0.52, 0.88, '最適化→ネットワーク'),
        (0.68, 0.88, 0.74, 0.88, 'ネットワーク→生成'),
        (0.38, 0.82, 0.28, 0.65, 'CNN→3D'),
        (0.68, 0.82, 0.58, 0.65, '生成→時空間'),
        (0.82, 0.82, 0.60, 0.65, '生成→時空間'),
        (0.28, 0.52, 0.40, 0.34, '3D→世界モデル'),
        (0.58, 0.52, 0.52, 0.34, '時空間→世界モデル'),
    ]

    for x1, y1, x2, y2, label in flow_arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#777777',
                                    lw=2, connectionstyle='arc3,rad=0.15'))

    ax.text(0.50, 0.98, '全カリキュラム統合マップ: Phase 1 - 7',
            ha='center', va='center', fontsize=20, fontweight='bold')

    layer_labels = [
        (0.02, 0.88, '基礎層', '#999999'),
        (0.02, 0.58, '応用層', '#999999'),
        (0.02, 0.26, '統合層', '#999999'),
    ]
    for x, y, label, color in layer_labels:
        ax.text(x, y, label, ha='left', va='center',
                fontsize=11, color=color, fontweight='bold',
                rotation=90)

    plt.tight_layout()
    plt.show()

visualize_grand_phase_map()"""
))

# ── Cell 15: Knowledge flow explanation ────────────────────
cells.append(code(
r"""def print_knowledge_flow():
    """Phase間の知識の流れを説明"""
    print("=" * 70)
    print("Phase間の知識の流れ")
    print("=" * 70)
    print("""
【基礎層 → 応用層】

  Phase 1 (統計) → Phase 4 (生成モデル)
    確率分布、最尤推定 → VAEのELBO、拡散モデルのノイズスケジュール

  Phase 2 (最適化) → 全てのPhase
    勾配降下法、Adam → 全てのニューラルネットワークの学習

  Phase 3 (CNN) → Phase 5 (3D), Phase 6 (時空間)
    畳み込み、U-Net → NeRFのMLP、Video Diffusionのアーキテクチャ

  Phase 4 (生成) → Phase 6 (時空間)
    拡散モデル、LDM → Video Diffusion、DiT

【応用層 → 統合層】

  Phase 5 (3D) → Phase 7 (世界モデル)
    空間理解、3D表現 → 世界の物理的理解、シミュレーション

  Phase 6 (時空間) → Phase 7 (世界モデル)
    時間的注意機構 → 動画予測、状態遷移モデル
    Video Diffusion → Genie、基盤世界モデル

【Phase 7が統合するもの】

  世界モデルは、全Phaseの知識を統合する頂点に位置します:
  - 統計的基盤（Phase 1）で不確実性をモデル化
  - 最適化技法（Phase 2）でモデルを学習
  - 空間理解（Phase 3）で視覚世界を処理
  - 生成能力（Phase 4）で未来を想像
  - 3D理解（Phase 5）で物理世界を理解
  - 時間的推論（Phase 6）で動的変化を予測
    """)

print_knowledge_flow()"""
))

# ── Cell 16: Section 5 header ──────────────────────────────
cells.append(md(
r"""<a id="section5"></a>
## 5. AGIへの道 — 世界モデルの未来

### なぜ世界モデルがAGIの鍵と言われるのか

世界モデルは、単なる動画予測を超えた野心的なビジョンを持っています。
Yann LeCunをはじめとする研究者たちは、世界モデルこそがAGI（汎用人工知能）への
道筋であると主張しています。

その理由は明確です：人間の知能の本質は、**世界がどう動くかの内部モデル**を持ち、
それを使って**予測・計画・行動**することにあるからです。"""
))

# ── Cell 17: LeCun's vision ──────────────────────────────
cells.append(code(
r"""def print_lecun_vision():
    """Yann LeCunのビジョンを説明"""
    print("=" * 70)
    print("Yann LeCunのビジョン: JEPAベースの世界モデル")
    print("=" * 70)
    print("""
LeCunが2022年に発表した論文「A Path Towards Autonomous Machine Intelligence」では、
以下のアーキテクチャが提案されました:

┌─────────────────────────────────────────────────────────┐
│                 自律的知能のアーキテクチャ                  │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│  │ World     │←──│ Perception│←──│ 感覚入力  │           │
│  │ Model     │    │ Module   │    │ (観測)    │           │
│  │ (世界モデル)│    │ (知覚)   │    │          │           │
│  └────┬─────┘    └──────────┘    └──────────┘           │
│       │                                                  │
│       ↓                                                  │
│  ┌──────────┐    ┌──────────┐                           │
│  │ Cost      │←──│ Intrinsic│                           │
│  │ Module    │    │ Cost     │                           │
│  │ (コスト)   │    │ (内発的)  │                           │
│  └────┬─────┘    └──────────┘                           │
│       │                                                  │
│       ↓                                                  │
│  ┌──────────┐    ┌──────────┐                           │
│  │ Actor     │───→│ 行動出力  │                           │
│  │ (行動器)   │    │          │                           │
│  └──────────┘    └──────────┘                           │
└─────────────────────────────────────────────────────────┘

核心的な主張:
  1. 世界モデルは特徴空間で動作すべき（JEPA的アプローチ）
  2. ピクセル空間での生成は不要（計算の無駄）
  3. 階層的な計画が可能であるべき
  4. 内発的動機づけ（好奇心）が重要

これはまさにPhase 7で学んだ技術の延長線上にあります:
  - JEPA（Nb 141）→ 知覚モジュール + 世界モデル
  - DreamerV3（Nb 143）→ 想像による計画
  - Genie（Nb 144）→ 行動の自動発見
    """)

print_lecun_vision()"""
))

# ── Cell 18: Future directions ──────────────────────────
cells.append(code(
r"""def print_future_directions():
    """世界モデルの未来の方向性"""
    print("=" * 70)
    print("世界モデルの5つの未来方向")
    print("=" * 70)
    print("""
【1. 基盤世界モデル (Foundation World Models)】
  - Genie 2 (DeepMind): 大規模な汎用世界シミュレータ
  - UniSim (Google): テキスト/行動条件付きの統一シミュレータ
  - 目標: 1つのモデルで多様な環境をシミュレート

【2. 身体化AI (Embodied AI)】
  - 動画から物理法則を学ぶロボット
  - Physical Intelligence: 実世界の物理をモデル化
  - Sim-to-Real: シミュレーション → 実世界への転移
  - 目標: 汎用的なロボット操作能力

【3. 階層的世界モデル (Hierarchical World Models)】
  - 抽象度の異なる複数の世界モデルを階層的に構築
  - 高レベル: 「部屋を出る」「目的地に行く」
  - 低レベル: 「右足を前に出す」「ドアノブを回す」
  - 目標: 長期的な計画と短期的な制御の統合

【4. マルチモーダル世界モデル】
  - 視覚だけでなく、音声・触覚・言語を統合
  - テキスト記述から世界をシミュレート
  - 言語モデルとの統合（LLMが世界モデルの一部を担う？）
  - 目標: 人間のような多感覚的な世界理解

【5. 自己改善する世界モデル】
  - 予測誤差から自動的にモデルを改善
  - 能動的探索: 世界モデルの不確実性が高い領域を優先的に探索
  - 好奇心駆動型学習（Intrinsic Motivation）
  - 目標: 継続的に学習し成長するシステム
    """)

print_future_directions()"""
))

# ── Cell 19: Timeline visualization ──────────────────────
cells.append(code(
r"""def visualize_timeline():
    """世界モデル研究のマイルストーンタイムライン"""
    fig, ax = plt.subplots(figsize=(20, 9))

    milestones = [
        (2015, 'World Models\n(Ha & Schmidhuber)', '#4ECDC4', 'VAE + RNN\n夢の中で学習'),
        (2017, 'SimCLR\n対照学習の基礎', '#45B7D1', '視覚表現学習の\nブレイクスルー'),
        (2019, 'Dreamer v1', '#96CEB4', '潜在空間での\n想像と計画'),
        (2020, 'BYOL\n非対照学習', '#FFD93D', 'ネガティブサンプル\n不要の学習'),
        (2021, 'MAE\nマスク自己符号化', '#FF6B6B', 'ViT + マスク\n予測学習'),
        (2023, 'DreamerV3\n汎用世界モデル', '#FFEAA7', 'ハイパーパラメータ\n自動調整'),
        (2023.5, 'I-JEPA\n(LeCun)', '#DDA0DD', '特徴空間での\n予測学習'),
        (2024, 'Genie\n(DeepMind)', '#AED6F1', '潜在行動の\n自動発見'),
        (2024.5, 'Genie 2 / UniSim\n基盤世界モデル', '#F0E68C', '大規模汎用\nシミュレータ'),
        (2025, '???  AGIへ', '#FF9FF3', '階層的世界モデル\n身体化AI'),
    ]

    y_base = 0.45
    for i, (year, name, color, desc) in enumerate(milestones):
        x = (year - 2014.5) / 11.5
        y_offset = 0.22 if i % 2 == 0 else -0.22

        ax.plot([x, x], [y_base, y_base + y_offset * 0.6], color=color, lw=2.5)
        circle = plt.Circle((x, y_base), 0.015, color=color, ec='#333', lw=1.5, zorder=5)
        ax.add_patch(circle)

        box = FancyBboxPatch((x - 0.06, y_base + y_offset * 0.65 - 0.06), 0.12, 0.12,
                             boxstyle='round,pad=0.01',
                             facecolor=color, edgecolor='#555', linewidth=1.5, alpha=0.85)
        ax.add_patch(box)
        ax.text(x, y_base + y_offset * 0.65 + 0.02, name, ha='center', va='center',
                fontsize=8, fontweight='bold')
        ax.text(x, y_base + y_offset * 0.65 - 0.035, desc, ha='center', va='center',
                fontsize=6.5, color='#444')

    ax.plot([0.02, 0.98], [y_base, y_base], color='#333', lw=3, zorder=1)
    for year in range(2015, 2026):
        x = (year - 2014.5) / 11.5
        ax.text(x, y_base - 0.05, str(year), ha='center', va='top', fontsize=10, color='#555')

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 0.95)
    ax.axis('off')
    ax.set_title('世界モデル研究のマイルストーン', fontsize=18, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.show()

visualize_timeline()"""
))

# ── Cell 20: Limitations ──────────────────────────────────
cells.append(code(
r"""def print_limitations():
    """世界モデルの現在の限界と未解決問題"""
    print("=" * 70)
    print("世界モデルの現在の限界と未解決問題")
    print("=" * 70)
    print("""
【限界1: 長期予測の困難さ】
  - 予測誤差が時間とともに蓄積（compounding error）
  - 数秒〜数十秒先の予測は現実的だが、数分先は困難
  - 対策: 階層的予測、不確実性を考慮した計画

【限界2: 物理的整合性】
  - 生成された動画が物理法則に違反することがある
  - 物体の貫通、突然の消失、重力の無視
  - 対策: 物理シミュレータとの統合、物理インフォームドな損失関数

【限界3: 因果推論の欠如】
  - 現在の世界モデルは相関を学ぶが、因果関係を理解しない
  - 「ボールを蹴った → ボールが動いた」の因果をモデル化できない
  - 対策: 因果推論フレームワーク（SCM）との統合

【限界4: スケーラビリティ】
  - 複雑な環境では膨大な計算リソースが必要
  - リアルタイム制御には推論速度が不足
  - 対策: モデル蒸留、効率的なアーキテクチャ

【限界5: 汎化能力】
  - 訓練環境と異なる環境への転移が困難
  - ドメインシフトに弱い
  - 対策: 大規模事前学習、メタ学習

【限界6: 評価指標の未確立】
  - 「良い世界モデル」の定量的な評価基準がない
  - FIDやFVDは視覚品質のみ測定
  - 対策: タスク依存の評価、物理的整合性メトリクス
    """)

print_limitations()"""
))

# ── Cell 21: Section 6 header ──────────────────────────────
cells.append(md(
r"""<a id="section6"></a>
## 6. 総合クイズ（5問）

Phase 7全体の理解度を確認するクイズです。各問題をじっくり考えてから回答を確認してください。"""
))

# ── Cell 22: Quiz Q1 ──────────────────────────────────────
cells.append(md(
r"""### Q1: 対照学習と再構成ベースの表現学習の根本的な違いは何ですか？

ヒント: SimCLR（対照学習）とMAE（再構成）を比較して考えてください。

<details>
<summary>回答を見る</summary>

**対照学習（Contrastive Learning）:**
- 正例ペア（同じ画像の異なるビュー）を近くに、負例ペア（異なる画像）を遠くに配置する
- 損失関数: InfoNCE等の対照損失
- 特徴: 意味的な類似性を直接最適化する
- 例: SimCLR, MoCo, BYOL

**再構成ベース（Reconstructive）:**
- 入力の一部をマスクし、残りから元の入力を再構成する
- 損失関数: MSE（ピクセル空間での再構成誤差）
- 特徴: 低レベルの詳細も含めて学習する
- 例: MAE, BEiT

**根本的な違い:**
- 対照学習は「何が似ているか」を学ぶ（意味的な不変性）
- 再構成は「何が欠けているか」を学ぶ（局所的な構造）
- LeCunの主張: どちらも完璧ではなく、JEPAのように特徴空間での予測が最も効果的

| 観点 | 対照学習 | 再構成 |
|------|---------|--------|
| 学習目標 | 類似度の最大化/最小化 | 入力の復元 |
| 崩壊リスク | 全て同じ表現に縮退 | 低レベル詳細への過適合 |
| 対策 | 負例、EMA、停止勾配 | マスク比率の調整 |

</details>"""
))

# ── Cell 23: Quiz Q2 ──────────────────────────────────────
cells.append(md(
r"""### Q2: JEPAとMAEの決定的な違いは何ですか？ なぜLeCunはJEPAを推奨するのですか？

ヒント: 予測が行われる空間（ピクセル空間 vs 特徴空間）に注目してください。

<details>
<summary>回答を見る</summary>

**MAE (Masked Autoencoder):**
- マスクされたパッチの**ピクセル値**を再構成する
- 予測空間: ピクセル空間（RGB値）
- デコーダが必要（ピクセルを生成するため）
- 損失: MSE in pixel space

**JEPA (Joint Embedding Predictive Architecture):**
- マスクされたパッチの**特徴表現**を予測する
- 予測空間: 潜在特徴空間（エンコーダの出力）
- デコーダ不要（特徴ベクトルを直接予測）
- 損失: MSE/cosine in feature space

**決定的な違い:**
```
MAE:   [visible patches] → Encoder → Decoder → [pixel reconstruction]
JEPA:  [visible patches] → Encoder → Predictor → [feature prediction]
                                                        ↕ 比較
       [target patches]  → Target Encoder → [target features]
```

**LeCunがJEPAを推奨する理由:**
1. **効率性**: ピクセルの全詳細を再構成する必要がない
2. **意味的**: 特徴空間はノイズやテクスチャの微細な違いを無視できる
3. **スケーラビリティ**: より抽象的な予測は計算コストが低い
4. **哲学的**: 人間も世界をピクセルレベルで予測していない

</details>"""
))

# ── Cell 24: Quiz Q3 ──────────────────────────────────────
cells.append(md(
r"""### Q3: Model-based RL と Model-free RL の主な違いは何ですか？ DreamerV3はなぜModel-basedアプローチを採用したのですか？

<details>
<summary>回答を見る</summary>

**Model-free RL:**
- 環境のモデルを学習しない
- 方策（Policy）や価値関数（Value function）を直接学習
- 実環境で大量のサンプルが必要（サンプル効率が低い）
- 例: PPO, SAC, DQN

**Model-based RL:**
- 環境の遷移モデル p(s'|s,a) を学習
- モデルの中で「想像」してデータを生成
- 少ない実環境データで効率的に学習
- 例: DreamerV3, MuZero, MBPO

**比較:**

| 観点 | Model-free | Model-based |
|------|-----------|-------------|
| サンプル効率 | 低い | 高い |
| 計算コスト | 環境ステップが主 | モデル学習 + 想像が主 |
| 性能上限 | 高い（理論上） | モデル精度に依存 |
| 複雑な環境 | スケールしやすい | モデル誤差が蓄積 |

**DreamerV3がModel-basedを採用した理由:**
1. **サンプル効率**: 実環境での試行を最小限に抑えられる
2. **安全性**: 危険な行動を想像の中で試せる
3. **計画能力**: 複数ステップ先を見据えた意思決定が可能
4. **汎用性**: RSSMによる潜在空間モデルで多様な環境に適応

</details>"""
))

# ── Cell 25: Quiz Q4 ──────────────────────────────────────
cells.append(md(
r"""### Q4: DreamerV3のRSSM（Recurrent State-Space Model）が「決定的状態」と「確率的状態」の両方を持つ理由は何ですか？

<details>
<summary>回答を見る</summary>

**RSSMの構造:**
```
確率的状態 z_t ~ p(z_t | h_t)     ← 不確実性を表現
決定的状態 h_t = f(h_{t-1}, z_{t-1}, a_{t-1})  ← 長期記憶を保持
```

**決定的状態 (h_t) の役割:**
- GRUセルで実装される
- 長期的な依存関係を記憶する
- 時間方向の情報の安定した伝播を担当
- 勾配が安定して流れる（勾配消失を防ぐ）

**確率的状態 (z_t) の役割:**
- 環境の不確実性をモデル化する
- 同じ状況でも異なる結果がありうることを表現
- KLダイバージェンスで正則化される
- 多様な想像ロールアウトを可能にする

**なぜ両方が必要か:**
1. **決定的のみ**: 不確実性を表現できず、1つの未来しか予測できない
2. **確率的のみ**: 長期記憶が保持できず、情報がノイズに埋もれる
3. **両方**: 安定した長期記憶 + 不確実性の表現 = 現実的な世界モデル

**直感的な例え:**
- 決定的状態 = 「今日は晴れで、道路は乾いている」（確定した事実）
- 確率的状態 = 「次の交差点で車が来るかもしれないし、来ないかもしれない」（不確実な未来）

</details>"""
))

# ── Cell 26: Quiz Q5 ──────────────────────────────────────
cells.append(md(
r"""### Q5: Genieの「潜在行動発見（Latent Action Discovery）」はなぜ革新的なのですか？ 従来のアプローチとの違いは何ですか？

<details>
<summary>回答を見る</summary>

**従来のアプローチ:**
- 行動は人間が定義する（上下左右、ボタン押下など）
- ラベル付きデータが必要（状態-行動ペア）
- 環境ごとに行動空間を再定義する必要がある

**Genieの潜在行動発見:**
- ラベルなし動画のみから「行動」の概念を自動的に発見
- 連続する2フレーム間の変化を「潜在行動」として推定
- 行動空間は離散的（VQ-VAEで離散化）

**仕組み:**
```
フレーム t    → Latent Action Model → 潜在行動 a_t
フレーム t+1                           (離散トークン)
```

潜在行動モデルは以下を学習:
- 2フレーム間の「差分」を抽象化
- 意味的に一貫した行動カテゴリを自動形成
- 例: 「左移動」「ジャンプ」「静止」などが自然に分離

**革新性:**
1. **教師なし**: 行動ラベルが一切不要
2. **汎用性**: 任意の動画ドメインに適用可能
3. **制御可能性**: 発見された行動でインタラクティブに環境を操作できる
4. **スケーラビリティ**: インターネット上の大量の動画データを活用可能

**意義:**
Genieは「世界モデル = 環境シミュレータ」を動画データだけから構築できることを示しました。
これは、人間のラベリングなしでAIが世界の「操作方法」を発見できることを意味します。

</details>"""
))

# ── Cell 27: Section 7 header ──────────────────────────────
cells.append(md(
r"""<a id="section7"></a>
## 7. 学習の旅の完結

### Phase 7の完了、おめでとうございます

Phase 7「世界モデル」の全7ノートブック（140-146）を完了しました。
これは、Phase 1から始まった長い学習の旅の集大成です。"""
))

# ── Cell 28: Congratulations ──────────────────────────────
cells.append(code(
r"""def print_congratulations():
    """完了メッセージ"""
    print("=" * 70)
    print("Phase 7 完了 — おめでとうございます！")
    print("=" * 70)
    print("""
Phase 1 から Phase 7 まで、あなたは以下の壮大な旅を歩んできました:

  Phase 1: 統計と確率の基礎を固め
  Phase 2: 最適化の技法を身につけ
  Phase 3: CNNで空間を理解し
  Phase 4: 生成モデルで創造を学び
  Phase 5: 3Dビジョンで世界を立体的に捉え
  Phase 6: 時空間モデリングで動きを理解し
  Phase 7: 世界モデルで予測・計画・行動を統合した

この知識の全てが、世界モデルという1つの大きなテーマに収束しています。
あなたは今、現代のAI研究の最前線を理解する基盤を持っています。
    """)

print_congratulations()"""
))

# ── Cell 29: Learning journey visualization ──────────────
cells.append(code(
r"""def visualize_learning_journey():
    """学習の旅を可視化"""
    fig, ax = plt.subplots(figsize=(16, 8))

    phases = [
        ('Phase 1\n統計', 1, '#E8D5B7'),
        ('Phase 2\n最適化', 2, '#B7D5E8'),
        ('Phase 3\nCNN', 3, '#D5E8B7'),
        ('Phase 4\n生成', 4, '#E8B7D5'),
        ('Phase 5\n3D', 5, '#D5B7E8'),
        ('Phase 6\n時空間', 6, '#B7E8D5'),
        ('Phase 7\n世界モデル', 7, '#FFD700'),
    ]

    complexity = [1.0, 1.8, 3.0, 4.2, 5.0, 6.0, 7.5]
    understanding = [0.8, 1.5, 2.8, 3.8, 4.5, 5.5, 7.0]

    x_vals = np.array([p[1] for p in phases])

    ax.fill_between(x_vals, 0, complexity, alpha=0.2, color='#3498DB', label='複雑さ')
    ax.fill_between(x_vals, 0, understanding, alpha=0.2, color='#2ECC71', label='理解度')
    ax.plot(x_vals, complexity, 'o-', color='#3498DB', lw=2.5, markersize=10, label='複雑さ')
    ax.plot(x_vals, understanding, 's-', color='#2ECC71', lw=2.5, markersize=10, label='理解度')

    for name, x, color in phases:
        ax.axvline(x=x, color='gray', alpha=0.2, linestyle='--')
        ax.text(x, -0.8, name, ha='center', va='top', fontsize=11,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(-1.5, 9)
    ax.set_ylabel('レベル', fontsize=13)
    ax.set_title('学習の旅: 複雑さと理解度の成長', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])

    ax.annotate('あなたはここ！', xy=(7, 7.0), xytext=(6.2, 8.2),
                fontsize=14, fontweight='bold', color='#E74C3C',
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.5))

    plt.tight_layout()
    plt.show()

visualize_learning_journey()"""
))

# ── Cell 30: Next steps ──────────────────────────────────
cells.append(code(
r"""def print_next_steps():
    """次のステップの提案"""
    print("=" * 70)
    print("次に探索すべきテーマ")
    print("=" * 70)
    print("""
【論文を読む】
  1. "A Path Towards Autonomous Machine Intelligence" (LeCun, 2022)
     → 世界モデルベースのAGIアーキテクチャの全体像

  2. "Mastering Diverse Domains through World Models" (DreamerV3, 2023)
     → 汎用世界モデルの設計原則

  3. "Genie: Generative Interactive Environments" (DeepMind, 2024)
     → 潜在行動発見の革新的アプローチ

  4. "V-JEPA: Video Joint Embedding Predictive Architecture" (Meta, 2024)
     → 動画理解のための特徴予測

  5. "Learning Interactive Real-World Simulators" (UniSim, 2024)
     → 基盤世界モデルのビジョン

【オープンソースプロジェクト】
  - DreamerV3: https://github.com/danijar/dreamerv3
  - JEPA実装: https://github.com/facebookresearch/ijepa
  - Video Prediction: https://github.com/wilson1yan/VideoGPT

【発展的なトピック】
  - 因果推論と世界モデルの統合
  - マルチエージェント世界モデル
  - ニューロシンボリック世界モデル
  - 言語グラウンディングと世界モデル
    """)

print_next_steps()"""
))

# ── Cell 31: Final checklist ──────────────────────────────
cells.append(code(
r"""def print_final_checklist():
    """最終学習チェックリスト"""
    print("=" * 70)
    print("Phase 7 最終学習チェックリスト")
    print("=" * 70)

    sections = [
        ("表現学習 (Nb 140)", [
            "対照学習の原理（正例/負例）を説明できる",
            "BYOLの非対照学習の仕組みを説明できる",
            "表現の質を評価する方法を知っている",
        ]),
        ("JEPA (Nb 141)", [
            "Joint Embeddingの意味を説明できる",
            "JEPAとMAEの違いを説明できる",
            "特徴空間での予測の利点を議論できる",
        ]),
        ("モデルベースRL (Nb 142)", [
            "Model-based と Model-free の違いを説明できる",
            "環境モデルの学習方法を理解している",
            "想像による計画の利点を説明できる",
        ]),
        ("DreamerV3 (Nb 143)", [
            "RSSMの構造（決定的+確率的状態）を説明できる",
            "潜在空間でのロールアウトの仕組みを理解している",
            "Actor-Criticの役割を説明できる",
        ]),
        ("Genie (Nb 144)", [
            "潜在行動発見の仕組みを説明できる",
            "ST-Transformerの役割を理解している",
            "ラベルなし動画からの学習の意義を議論できる",
        ]),
        ("応用と発展 (Nb 145)", [
            "基盤世界モデルの概念を説明できる",
            "世界モデルの現在の限界を3つ以上挙げられる",
            "ロボティクスへの応用可能性を議論できる",
        ]),
        ("総括 (Nb 146 - 本章)", [
            "3つの予測パラダイムを比較できる",
            "Phase 1-7の全体像を説明できる",
            "世界モデルの未来展望を議論できる",
        ]),
    ]

    for section, items in sections:
        print(f"\n [{section}]")
        for item in items:
            print(f"  [ ] {item}")

    total = sum(len(items) for _, items in sections)
    print(f"\n{'=' * 70}")
    print(f"全 {total} 項目")
    print("全ての項目にチェックが入れば、Phase 7の学習は完了です！")

print_final_checklist()"""
))

# ── Cell 32: Final message ──────────────────────────────
cells.append(md(
r"""---

## おわりに

Phase 7「世界モデル」の学習、お疲れさまでした。

世界モデルは、現代のAI研究において最も活発で野心的な分野の1つです。
この分野は急速に進化しており、今後も新しいブレイクスルーが続くでしょう。

Phase 1から Phase 7まで積み上げてきた知識は、これらの最先端研究を理解し、
さらには自ら貢献するための確かな土台となります。

**学びは終わりではなく、始まりです。**

ここで得た知識を武器に、次のステップへ進んでください。

---

> "The essence of intelligence is the ability to predict the future."
> -- Yann LeCun

---"""
))

# ── Fix cell sources to use proper line-based format ──────
for i, cell in enumerate(cells):
    # Convert source from single string split to proper list of lines
    raw = cell["source"]
    # raw is a list from split("\n") — join back and re-split with \n kept
    text = "\n".join(raw)
    lines = text.split("\n")
    # Each line except the last should end with \n
    new_source = []
    for j, line in enumerate(lines):
        if j < len(lines) - 1:
            new_source.append(line + "\n")
        else:
            new_source.append(line)
    cell["source"] = new_source
    cell["id"] = f"cell-{i:02d}"

notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

output_path = "/Users/ikmx/source/personal/machine-learning-playground/notebooks/world-models/146_world_models_synthesis_v1.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Wrote {len(cells)} cells to {output_path}")
