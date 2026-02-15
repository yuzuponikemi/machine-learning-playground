#!/usr/bin/env python3
"""Build script for notebook 132_diffusion_transformer_dit_v1.ipynb"""
import json, os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n"), "id": None}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source.split("\n"),
            "outputs": [], "execution_count": None, "id": None}

cells = []

# ── Cell 0: Title ──
cells.append(md(
"""# 第132章: Diffusion Transformer (DiT) — U-NetからTransformerへ

## 📋 この章で学ぶこと

この章を終えると、以下ができるようになります：

- [ ] パッチ埋め込み（PatchEmbed）で画像をトークン列に変換できる
- [ ] adaLN-Zero条件付けの仕組みを説明・実装できる
- [ ] DiTBlock（Transformer + adaLN-Zero）をスクラッチで実装できる
- [ ] DiT全体モデルをMNIST向けに構築・訓練できる
- [ ] Soraの技術的基盤としてのDiTの役割を説明できる

## 🎯 前提知識

- ✅ Notebook 95（Vision Transformer / Self-Attention）
- ✅ Notebook 131（Video Diffusion Models）
- ✅ Notebook 43（Latent Diffusion Models）

⏱️ **推定学習時間**: 150-180分
📊 **難易度**: ★★★★☆（上級）
🎓 **カテゴリ**: 時空間モデリング"""
))

# ── Cell 1: TOC ──
cells.append(md(
"""## 目次

1. [U-NetからDiTへの進化の動機](#section1)
2. [PatchEmbed — 画像をトークン列に変換](#section2)
3. [adaLN-Zero条件付け — タイムステップとクラスの条件付け](#section3)
4. [DiTBlock — Transformer + adaLN-Zero](#section4)
5. [DiT full model（MNIST向け軽量版）](#section5)
6. [MNIST訓練・生成デモ](#section6)
7. [Soraとの技術的関連](#section7)
8. [まとめ・よくあるエラー・自己評価クイズ](#summary)"""
))

# ── Cell 2: Setup ──
cells.append(code(
"""# ============================================================
# 環境設定
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import warnings

warnings.filterwarnings('ignore')

# 日本語フォント設定
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
    return None

font_used = setup_japanese_font()
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 再現性の確保
torch.manual_seed(42)
np.random.seed(42)

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'✅ ライブラリのインポート完了')
print(f'🖥️ デバイス: {device}')
print(f'📝 日本語フォント: {font_used}')"""
))

# ── Cell 3: Section 1 intro ──
cells.append(md(
"""<a id="section1"></a>
## 1. U-NetからDiTへの進化の動機

### 🤔 なぜU-Netを置き換えるのか？

これまで学んできた拡散モデル（Notebook 40-43）では、**U-Net** がノイズ予測の中核を担っていました。
U-Netはスキップ接続と階層的なダウン/アップサンプリングにより優れた生成品質を達成しています。

しかし、NLP分野では**Transformer**がスケーリング則（Scaling Law）を示し、
モデルを大きくすればするほど性能が向上することが明らかになりました。

**Diffusion Transformer（DiT）** は、この知見を拡散モデルに持ち込む試みです。

### 📊 U-Net vs DiT の比較

| 特性 | U-Net | DiT |
|------|-------|-----|
| 基本構造 | CNN (エンコーダ-デコーダ) | Transformer (パッチトークン) |
| 条件付け | Cross-Attention / 加算 | adaLN-Zero |
| スケーリング | 難しい（構造が複雑） | 容易（層を積むだけ） |
| スキップ接続 | あり（U字構造） | なし（直列Transformer） |
| 帰納バイアス | 局所性・階層性 | 弱い（大域的注意） |
| Scaling Law | 不明確 | 明確に従う |

### 💡 DiTの核心アイデア

1. **画像をパッチに分割**してトークン列にする（ViTと同様）
2. **adaLN-Zero** で拡散タイムステップとクラスラベルを条件付け
3. **標準的なTransformerブロック**を積み重ねる
4. **Unpatchify** でトークン列を画像に戻す

```
[DiTアーキテクチャ]
画像 → PatchEmbed → [DiTBlock × N] → Unpatchify → 予測ノイズ
         ↑                ↑
    位置埋め込み      adaLN-Zero条件付け
                   (タイムステップ + クラス)
```"""
))

# ── Cell 4: Architecture comparison vis ──
cells.append(code(
"""# ============================================================
# U-Net vs DiT のアーキテクチャ比較図
# ============================================================

def visualize_unet_vs_dit():
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- U-Net (左) ---
    ax = axes[0]
    # エンコーダブロック
    blocks_enc = [
        (0.1, 0.7, 0.15, 0.15, 'Conv + GN\\n+ SiLU', 'lightblue'),
        (0.1, 0.5, 0.15, 0.12, 'Downsample', 'lightyellow'),
        (0.1, 0.3, 0.15, 0.15, 'Conv + GN\\n+ SiLU', 'lightblue'),
    ]
    for x, y, w, h, label, color in blocks_enc:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color,
                                    edgecolor='black', lw=1.5))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=8)

    # ボトルネック
    ax.add_patch(plt.Rectangle((0.35, 0.35, ), 0.15, 0.15,
                                facecolor='lightcoral', edgecolor='black', lw=1.5))
    ax.text(0.425, 0.425, 'Bottleneck', ha='center', va='center', fontsize=8)

    # デコーダブロック
    blocks_dec = [
        (0.6, 0.3, 0.15, 0.15, 'Conv + GN\\n+ SiLU', 'lightblue'),
        (0.6, 0.5, 0.15, 0.12, 'Upsample', 'lightyellow'),
        (0.6, 0.7, 0.15, 0.15, 'Conv + GN\\n+ SiLU', 'lightblue'),
    ]
    for x, y, w, h, label, color in blocks_dec:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color,
                                    edgecolor='black', lw=1.5))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=8)

    # スキップ接続
    ax.annotate('', xy=(0.6, 0.77), xytext=(0.25, 0.77),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='green',
                               connectionstyle='arc3,rad=-0.4'))
    ax.text(0.42, 0.92, 'Skip Connection', ha='center', fontsize=9, color='green')

    ax.set_xlim(0, 0.85)
    ax.set_ylim(0.15, 1.0)
    ax.set_title('U-Net\\n(階層的 CNN 構造)', fontsize=13, fontweight='bold')
    ax.axis('off')

    # --- DiT (右) ---
    ax = axes[1]
    # パッチ埋め込み
    ax.add_patch(plt.Rectangle((0.2, 0.82), 0.45, 0.1,
                                facecolor='lightgreen', edgecolor='black', lw=1.5))
    ax.text(0.425, 0.87, 'PatchEmbed + Positional Embedding', ha='center', fontsize=9)

    # DiTブロック
    y_positions = [0.68, 0.54, 0.40]
    for i, y in enumerate(y_positions):
        ax.add_patch(plt.Rectangle((0.2, y), 0.45, 0.1,
                                    facecolor='lightyellow', edgecolor='black', lw=1.5))
        ax.text(0.425, y + 0.05, f'DiTBlock {i+1} (adaLN-Zero + MHA + FFN)',
                ha='center', fontsize=8)

    # Unpatchify
    ax.add_patch(plt.Rectangle((0.2, 0.25), 0.45, 0.1,
                                facecolor='lightcoral', edgecolor='black', lw=1.5))
    ax.text(0.425, 0.30, 'Unpatchify → 予測ノイズ', ha='center', fontsize=9)

    # 条件付け矢印
    ax.add_patch(plt.Rectangle((0.72, 0.45), 0.18, 0.25,
                                facecolor='plum', edgecolor='black', lw=1.5, alpha=0.7))
    ax.text(0.81, 0.60, 'adaLN-Zero\\n条件付け\\n(t, class)', ha='center', fontsize=8)
    for y in y_positions:
        ax.annotate('', xy=(0.65, y + 0.05), xytext=(0.72, y + 0.05),
                    arrowprops=dict(arrowstyle='->', lw=1.2, color='purple'))

    # 矢印（フロー）
    for y1, y2 in zip([0.82, 0.68, 0.54, 0.40], [0.78, 0.64, 0.50, 0.35]):
        ax.annotate('', xy=(0.425, y1), xytext=(0.425, y2 + 0.04),
                    arrowprops=dict(arrowstyle='->', lw=1.2))

    ax.set_xlim(0.1, 0.95)
    ax.set_ylim(0.15, 1.0)
    ax.set_title('DiT\\n(直列 Transformer 構造)', fontsize=13, fontweight='bold')
    ax.axis('off')

    plt.suptitle('U-Net vs Diffusion Transformer (DiT)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

visualize_unet_vs_dit()"""
))

# ── Cell 5: Section 2 intro ──
cells.append(md(
"""<a id="section2"></a>
## 2. PatchEmbed — 画像をトークン列に変換

### 📊 パッチ埋め込みの仕組み

ViT（Notebook 95）と同様に、DiTは画像をパッチに分割してトークン列に変換します。

**手順:**
1. 画像 $(C, H, W)$ をパッチサイズ $p$ で分割
2. 各パッチを $C \\times p \\times p$ 次元のベクトルに展開
3. 線形射影で隠れ次元 $d$ に変換

$$
\\text{パッチ数} = \\frac{H}{p} \\times \\frac{W}{p}
$$

**実装のポイント**: `nn.Conv2d(kernel_size=p, stride=p)` を使うと、
パッチ分割と線形射影を1つの操作で実現できます。"""
))

# ── Cell 6: PatchEmbed implementation ──
cells.append(code(
"""# ============================================================
# PatchEmbed の実装
# ============================================================

class PatchEmbed(nn.Module):
    \"\"\"画像をパッチトークン列に変換する

    Conv2dでパッチ分割と線形射影を同時に行う。
    kernel_size=patch_size, stride=patch_size とすることで
    非重複パッチを抽出する。
    \"\"\"

    def __init__(self, img_size=32, patch_size=4, in_channels=1, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Conv2dでパッチ分割+線形射影を同時実行
        # kernel_size=patch_size, stride=patch_size → 非重複パッチ
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        \"\"\"
        Args:
            x: (B, C, H, W) 入力画像
        Returns:
            (B, num_patches, embed_dim) パッチトークン列
        \"\"\"
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \\
            f"入力サイズ {H}x{W} が期待値 {self.img_size}x{self.img_size} と不一致"

        # (B, C, H, W) → (B, embed_dim, H/p, W/p)
        x = self.proj(x)
        # (B, embed_dim, H/p, W/p) → (B, embed_dim, num_patches)
        x = x.flatten(2)
        # (B, embed_dim, num_patches) → (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x

# テスト
patch_embed = PatchEmbed(img_size=32, patch_size=4, in_channels=1, embed_dim=128)
x_test = torch.randn(2, 1, 32, 32)
tokens = patch_embed(x_test)

print("="*60)
print("PatchEmbed テスト")
print("="*60)
print(f"入力画像:       {x_test.shape}  (B, C, H, W)")
print(f"パッチサイズ:   {patch_embed.patch_size}")
print(f"パッチ数:       {patch_embed.num_patches}  ({32//4} x {32//4})")
print(f"出力トークン列: {tokens.shape}  (B, num_patches, embed_dim)")
print(f"パラメータ数:   {sum(p.numel() for p in patch_embed.parameters()):,}")
print("✅ PatchEmbed 動作確認完了")"""
))

# ── Cell 7: PatchEmbed visualization ──
cells.append(code(
"""# ============================================================
# PatchEmbed の動作を可視化
# ============================================================

def visualize_patch_embed():
    \"\"\"パッチ埋め込みの処理を段階的に可視化\"\"\"
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # ダミー画像を生成（グラデーション）
    np.random.seed(42)
    img = np.zeros((32, 32))
    # 円形パターン
    for i in range(32):
        for j in range(32):
            img[i, j] = np.sin(i * 0.3) * np.cos(j * 0.3)

    # 1. 元画像
    axes[0].imshow(img, cmap='viridis')
    axes[0].set_title('① 入力画像\\n(1, 32, 32)', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # 2. パッチ分割
    axes[1].imshow(img, cmap='viridis')
    patch_size = 4
    for i in range(0, 33, patch_size):
        axes[1].axhline(y=i-0.5, color='red', linewidth=1.5)
        axes[1].axvline(x=i-0.5, color='red', linewidth=1.5)
    axes[1].set_title(f'② パッチ分割\\n({32//patch_size}x{32//patch_size} = {(32//patch_size)**2}パッチ)',
                      fontsize=11, fontweight='bold')
    axes[1].axis('off')

    # 3. パッチをトークンに展開
    patches = []
    for i in range(0, 32, patch_size):
        for j in range(0, 32, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size].flatten()
            patches.append(patch)
    patch_matrix = np.array(patches)

    axes[2].imshow(patch_matrix, cmap='viridis', aspect='auto')
    axes[2].set_xlabel('パッチ内ピクセル (16次元)', fontsize=10)
    axes[2].set_ylabel('パッチ番号 (64個)', fontsize=10)
    axes[2].set_title('③ パッチ展開\\n(64, 16)', fontsize=11, fontweight='bold')

    # 4. 線形射影後のトークン
    # 実際のPatchEmbedを通す
    pe = PatchEmbed(img_size=32, patch_size=4, in_channels=1, embed_dim=128)
    with torch.no_grad():
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tokens = pe(img_tensor)  # (1, 64, 128)

    axes[3].imshow(tokens[0].numpy()[:, :32], cmap='RdBu_r', aspect='auto')
    axes[3].set_xlabel('埋め込み次元 (128のうち32表示)', fontsize=10)
    axes[3].set_ylabel('パッチ番号 (64個)', fontsize=10)
    axes[3].set_title('④ 線形射影後\\n(64, 128)', fontsize=11, fontweight='bold')

    plt.suptitle('PatchEmbed: 画像 → トークン列 への変換過程',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

visualize_patch_embed()"""
))

# ── Cell 8: Section 3 intro ──
cells.append(md(
"""<a id="section3"></a>
## 3. adaLN-Zero条件付け — タイムステップとクラスの条件付け

### 🤔 なぜ新しい条件付け方式が必要か？

U-Netでは、タイムステップ条件付けに**加算（add）**や**Cross-Attention**を使っていました。
DiTでは、より効果的な**adaLN-Zero**（Adaptive Layer Normalization with Zero-initialization）を使います。

### 📊 adaLN-Zero の仕組み

通常のLayer Normalizationは：
$$
\\text{LN}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sigma} + \\beta
$$

adaLN-Zeroでは、$\\gamma, \\beta$ を**条件ベクトル**（タイムステップ+クラス）から生成します：
$$
\\text{adaLN}(x, c) = \\gamma(c) \\odot \\frac{x - \\mu}{\\sigma} + \\beta(c)
$$

さらに、ブロック出力に**ゲート $\\alpha(c)$** を掛けます。
初期化時に $\\alpha = 0$ とすることで、訓練初期は恒等写像として振る舞い、学習が安定します。

```
条件ベクトル c (タイムステップ + クラス)
    ↓ MLP
[γ, β, α] を生成
    ↓
LayerNorm(x) × γ + β → Attention/FFN → × α → 出力
```"""
))

# ── Cell 9: adaLN-Zero implementation ──
cells.append(code(
"""# ============================================================
# adaLN-Zero の実装
# ============================================================

class AdaLNZero(nn.Module):
    \"\"\"Adaptive Layer Normalization with Zero-initialization

    条件ベクトルからscale(γ), shift(β), gate(α)を生成し、
    LayerNormの出力を変調する。
    gate(α)は0初期化され、訓練初期は恒等写像として機能する。
    \"\"\"

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

        # 条件ベクトルからγ, β, αを生成するMLP
        # 出力は3*dim: [γ, β, α]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * dim)
        )

        # ゲートαを0で初期化（重要！）
        # これにより訓練初期はブロック出力が0になり、
        # 残差接続のみが活きて恒等写像として機能する
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, cond):
        \"\"\"
        Args:
            x: (B, N, D) 入力トークン列
            cond: (B, cond_dim) 条件ベクトル
        Returns:
            x_modulated: (B, N, D) 変調されたトークン列
            alpha: (B, 1, D) ゲート値（後段で使用）
        \"\"\"
        # 条件ベクトルからγ, β, αを生成
        modulation = self.adaLN_modulation(cond)  # (B, 3*D)
        gamma, beta, alpha = modulation.chunk(3, dim=-1)  # 各 (B, D)

        # γ, β, αを(B, 1, D)にreshapeしてブロードキャスト可能にする
        gamma = gamma.unsqueeze(1)  # (B, 1, D)
        beta = beta.unsqueeze(1)    # (B, 1, D)
        alpha = alpha.unsqueeze(1)  # (B, 1, D)

        # adaLN: LayerNorm(x) * (1 + γ) + β
        x_norm = self.norm(x)
        x_modulated = x_norm * (1 + gamma) + beta

        return x_modulated, alpha

# テスト
B, N, D, cond_dim = 2, 64, 128, 256
x = torch.randn(B, N, D)
cond = torch.randn(B, cond_dim)

adaln = AdaLNZero(dim=D, cond_dim=cond_dim)
x_mod, alpha = adaln(x, cond)

print("="*60)
print("adaLN-Zero テスト")
print("="*60)
print(f"入力トークン:   {x.shape}")
print(f"条件ベクトル:   {cond.shape}")
print(f"変調後トークン: {x_mod.shape}")
print(f"ゲートα:       {alpha.shape}")
print(f"α初期値の平均: {alpha.mean().item():.6f}  (≈0 であるべき)")
print(f"α初期値の最大: {alpha.abs().max().item():.6f}")
print("✅ adaLN-Zero 動作確認完了（αが≈0で初期化されている）")"""
))

# ── Cell 10: adaLN-Zero visualization ──
cells.append(code(
"""# ============================================================
# adaLN-Zero の効果を可視化
# ============================================================

def visualize_adaln_zero():
    \"\"\"adaLN-Zeroの変調効果を可視化\"\"\"
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    torch.manual_seed(42)
    B, N, D, cond_dim = 1, 16, 64, 128
    x = torch.randn(B, N, D)

    adaln = AdaLNZero(dim=D, cond_dim=cond_dim)

    # 異なる条件ベクトルでの変調を比較
    conditions = [
        ('初期状態 (学習前)', torch.zeros(B, cond_dim)),
        ('条件A (ランダム1)', torch.randn(B, cond_dim)),
        ('条件B (ランダム2)', torch.randn(B, cond_dim) * 2),
    ]

    for col, (label, cond) in enumerate(conditions):
        with torch.no_grad():
            x_mod, alpha = adaln(x, cond)

        # 変調前後のトークンを表示
        axes[0, col].imshow(x[0, :, :32].numpy(), cmap='RdBu_r',
                           vmin=-2, vmax=2, aspect='auto')
        axes[0, col].set_title(f'{label}\\n変調前', fontsize=11, fontweight='bold')
        axes[0, col].set_ylabel('トークン番号' if col == 0 else '')

        axes[1, col].imshow(x_mod[0, :, :32].detach().numpy(), cmap='RdBu_r',
                           vmin=-2, vmax=2, aspect='auto')
        axes[1, col].set_title(f'変調後 (α平均={alpha.mean():.3f})', fontsize=11)
        axes[1, col].set_xlabel('埋め込み次元')
        axes[1, col].set_ylabel('トークン番号' if col == 0 else '')

    plt.suptitle('adaLN-Zero: 条件ベクトルによるトークンの変調\\n'
                 '(初期状態ではα≈0のため変調後≈変調前)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

visualize_adaln_zero()"""
))

# ── Cell 11: Section 4 intro ──
cells.append(md(
"""<a id="section4"></a>
## 4. DiTBlock — Transformer + adaLN-Zero

### 📊 DiTBlockの構造

DiTBlockは、標準的なTransformerブロック（Pre-Norm）のLayerNormを**adaLN-Zero**に置き換えたものです。

```
入力 x, 条件 c
  │
  ├─ adaLN-Zero₁(x, c) → γ₁, β₁, α₁
  │     ↓
  │  Multi-Head Attention × α₁
  │     ↓
  ├─ + (残差接続)
  │
  ├─ adaLN-Zero₂(x, c) → γ₂, β₂, α₂
  │     ↓
  │  Feed-Forward Network × α₂
  │     ↓
  └─ + (残差接続)
  │
  出力
```

ポイント:
- 標準的なTransformerのLayerNormをadaLN-Zeroに置換
- Attention出力とFFN出力にそれぞれゲート $\\alpha$ を適用
- 初期化時 $\\alpha = 0$ のため、ブロックは恒等写像からスタート"""
))

# ── Cell 12: DiTBlock implementation ──
cells.append(code(
"""# ============================================================
# DiTBlock の実装
# ============================================================

class DiTBlock(nn.Module):
    \"\"\"Diffusion Transformer ブロック

    Pre-norm Transformer block with adaLN-Zero.
    標準的なTransformerのLayerNormをadaLN-Zeroに置き換え、
    条件付けを行う。
    \"\"\"

    def __init__(self, dim, num_heads, cond_dim, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # adaLN-Zero for Attention
        self.adaln_attn = AdaLNZero(dim, cond_dim)

        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        # adaLN-Zero for FFN
        self.adaln_ffn = AdaLNZero(dim, cond_dim)

        # Feed-Forward Network
        mlp_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x, cond):
        \"\"\"
        Args:
            x: (B, N, D) 入力トークン列
            cond: (B, cond_dim) 条件ベクトル
        Returns:
            (B, N, D) 出力トークン列
        \"\"\"
        # 1. adaLN-Zero + Multi-Head Attention + 残差接続
        x_mod, alpha1 = self.adaln_attn(x, cond)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + alpha1 * attn_out  # α₁でゲーティング

        # 2. adaLN-Zero + FFN + 残差接続
        x_mod2, alpha2 = self.adaln_ffn(x, cond)
        ffn_out = self.ffn(x_mod2)
        x = x + alpha2 * ffn_out  # α₂でゲーティング

        return x

# テスト
B, N, D, cond_dim = 2, 64, 128, 256
x = torch.randn(B, N, D)
cond = torch.randn(B, cond_dim)

dit_block = DiTBlock(dim=D, num_heads=4, cond_dim=cond_dim)
out = dit_block(x, cond)

print("="*60)
print("DiTBlock テスト")
print("="*60)
print(f"入力:  {x.shape}")
print(f"条件:  {cond.shape}")
print(f"出力:  {out.shape}")
print(f"パラメータ数: {sum(p.numel() for p in dit_block.parameters()):,}")

# 初期状態では出力≈入力（恒等写像）であることを確認
diff = (out - x).abs().mean().item()
print(f"初期状態の入出力差: {diff:.6f}  (≈0 であるべき)")
print("✅ DiTBlock 動作確認完了")"""
))

# ── Cell 13: DiTBlock structure vis ──
cells.append(code(
"""# ============================================================
# DiTBlock内部の情報フローを可視化
# ============================================================

def visualize_dit_block_flow():
    \"\"\"DiTBlock内の情報フロー図\"\"\"
    fig, ax = plt.subplots(figsize=(10, 12))

    # ブロック定義: (x, y, w, h, label, color)
    blocks = [
        (0.3, 0.88, 0.4, 0.06, '入力 x  (B, N, D)', 'lightgray'),
        (0.3, 0.78, 0.4, 0.06, 'adaLN-Zero₁ → γ₁, β₁, α₁', 'plum'),
        (0.3, 0.68, 0.4, 0.06, 'Multi-Head Self-Attention', 'lightblue'),
        (0.3, 0.58, 0.4, 0.06, '× α₁ (ゲーティング)', 'lightyellow'),
        (0.3, 0.48, 0.4, 0.06, '+ 残差接続', 'lightgreen'),
        (0.3, 0.38, 0.4, 0.06, 'adaLN-Zero₂ → γ₂, β₂, α₂', 'plum'),
        (0.3, 0.28, 0.4, 0.06, 'Feed-Forward Network', 'lightblue'),
        (0.3, 0.18, 0.4, 0.06, '× α₂ (ゲーティング)', 'lightyellow'),
        (0.3, 0.08, 0.4, 0.06, '+ 残差接続 → 出力', 'lightgreen'),
    ]

    for x, y, w, h, label, color in blocks:
        ax.add_patch(plt.Rectangle((x, y), w, h,
                                    facecolor=color, edgecolor='black', lw=1.5))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10)

    # 矢印
    for i in range(len(blocks) - 1):
        y1 = blocks[i][1]
        y2 = blocks[i+1][1] + blocks[i+1][3]
        ax.annotate('', xy=(0.5, y2), xytext=(0.5, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5))

    # 残差接続の矢印
    ax.annotate('', xy=(0.25, 0.51), xytext=(0.25, 0.91),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='green',
                               connectionstyle='arc3,rad=0.3'))
    ax.text(0.12, 0.7, '残差', fontsize=9, color='green', rotation=90, ha='center')

    ax.annotate('', xy=(0.25, 0.11), xytext=(0.25, 0.51),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='green',
                               connectionstyle='arc3,rad=0.3'))
    ax.text(0.12, 0.3, '残差', fontsize=9, color='green', rotation=90, ha='center')

    # 条件ベクトル
    ax.add_patch(plt.Rectangle((0.78, 0.55), 0.18, 0.35,
                                facecolor='wheat', edgecolor='black', lw=1.5))
    ax.text(0.87, 0.72, '条件 c\\n(t + class)', ha='center', fontsize=10, fontweight='bold')

    ax.annotate('', xy=(0.7, 0.81), xytext=(0.78, 0.72),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='purple'))
    ax.annotate('', xy=(0.7, 0.41), xytext=(0.78, 0.65),
                arrowprops=dict(arrowstyle='->', lw=1.2, color='purple'))

    ax.set_xlim(0.05, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title('DiTBlock 内部の情報フロー', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.show()

    print("ポイント:")
    print("- adaLN-Zeroが標準LayerNormを置き換え、条件付けを実現")
    print("- α₁, α₂のゲーティングにより出力強度を制御")
    print("- 初期化時α=0のため、最初は恒等写像（安定した学習開始）")

visualize_dit_block_flow()"""
))

# ── Cell 14: Section 5 intro ──
cells.append(md(
"""<a id="section5"></a>
## 5. DiT full model（MNIST向け軽量版）

### 📊 全体アーキテクチャ

```
入力画像 (B, 1, 32, 32) + タイムステップ t + クラスラベル y
  ↓
PatchEmbed: (B, 64, 128)  [32/4=8, 8×8=64パッチ]
  ↓
+ 位置埋め込み (学習可能)
  ↓
DiTBlock × 4  [dim=128, heads=4, adaLN-Zero条件付け]
  ↓
Final LayerNorm + Linear
  ↓
Unpatchify: (B, 1, 32, 32)  [予測ノイズ]
```

教育目的のため、以下の軽量設定を使用します:
- パッチサイズ: 4
- 埋め込み次元: 128
- ヘッド数: 4
- レイヤ数: 4
- 画像サイズ: 32×32（MNISTを28→32にパディング）"""
))

# ── Cell 15: Sinusoidal embedding + timestep/label embedding ──
cells.append(code(
"""# ============================================================
# タイムステップ・クラスラベルの埋め込み
# ============================================================

def sinusoidal_embedding(timesteps, dim):
    \"\"\"正弦波位置エンコーディング（拡散タイムステップ用）

    Args:
        timesteps: (B,) 整数テンソル
        dim: 埋め込み次元
    Returns:
        (B, dim) 位置埋め込み
    \"\"\"
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    \"\"\"拡散タイムステップの埋め込み

    Sinusoidal → MLP で条件ベクトルに変換
    \"\"\"
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, t):
        \"\"\"t: (B,) → (B, cond_dim)\"\"\"
        t_emb = sinusoidal_embedding(t, self.dim)
        return self.mlp(t_emb)


class LabelEmbedder(nn.Module):
    \"\"\"クラスラベルの埋め込み

    離散ラベル → 学習可能な埋め込みベクトル
    \"\"\"
    def __init__(self, num_classes, cond_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, cond_dim)

    def forward(self, y):
        \"\"\"y: (B,) 整数ラベル → (B, cond_dim)\"\"\"
        return self.embedding(y)

# テスト
t_embedder = TimestepEmbedder(dim=128, cond_dim=256)
l_embedder = LabelEmbedder(num_classes=10, cond_dim=256)

t = torch.tensor([0, 100, 500, 999])
y = torch.tensor([0, 3, 7, 9])

t_emb = t_embedder(t)
y_emb = l_embedder(y)
cond = t_emb + y_emb  # タイムステップとクラスを加算で結合

print("タイムステップ埋め込み:", t_emb.shape)
print("クラスラベル埋め込み:", y_emb.shape)
print("条件ベクトル (t + y):", cond.shape)"""
))

# ── Cell 16: DiT full model ──
cells.append(code(
"""# ============================================================
# DiT (Diffusion Transformer) 完全モデル
# ============================================================

class DiT(nn.Module):
    \"\"\"Diffusion Transformer — MNIST向け軽量版

    PatchEmbed → 位置埋め込み → DiTBlock × N → Unpatchify

    Args:
        img_size: 入力画像サイズ (正方形)
        patch_size: パッチサイズ
        in_channels: 入力チャンネル数
        dim: 埋め込み次元
        depth: DiTBlockの数
        num_heads: Attentionヘッド数
        num_classes: クラス数
        mlp_ratio: FFNの拡張率
    \"\"\"

    def __init__(self, img_size=32, patch_size=4, in_channels=1,
                 dim=128, depth=4, num_heads=4, num_classes=10, mlp_ratio=4.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.num_patches = (img_size // patch_size) ** 2

        # 条件ベクトルの次元
        cond_dim = dim * 2  # 256

        # 1. パッチ埋め込み
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, dim)

        # 2. 学習可能な位置埋め込み
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        # 3. 条件埋め込み (タイムステップ + クラスラベル)
        self.t_embedder = TimestepEmbedder(dim, cond_dim)
        self.y_embedder = LabelEmbedder(num_classes, cond_dim)

        # 4. DiTBlocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, cond_dim, mlp_ratio)
            for _ in range(depth)
        ])

        # 5. 最終層: LayerNorm + 線形射影でパッチ空間に戻す
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim)
        )
        nn.init.zeros_(self.final_adaln[-1].weight)
        nn.init.zeros_(self.final_adaln[-1].bias)

        # 出力射影: dim → patch_size * patch_size * in_channels
        self.final_proj = nn.Linear(
            dim, patch_size * patch_size * in_channels
        )

    def unpatchify(self, x):
        \"\"\"トークン列を画像に戻す

        Args:
            x: (B, num_patches, patch_size**2 * C)
        Returns:
            (B, C, H, W) 画像
        \"\"\"
        p = self.patch_size
        h = w = self.img_size // p
        c = self.in_channels

        # (B, h*w, p*p*c) → (B, h, w, p, p, c)
        x = x.reshape(-1, h, w, p, p, c)
        # (B, h, w, p, p, c) → (B, c, h*p, w*p)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(-1, c, h * p, w * p)
        return x

    def forward(self, x, t, y):
        \"\"\"
        Args:
            x: (B, C, H, W) ノイズ付き画像
            t: (B,) 拡散タイムステップ
            y: (B,) クラスラベル
        Returns:
            (B, C, H, W) 予測ノイズ
        \"\"\"
        # 1. パッチ埋め込み + 位置埋め込み
        x = self.patch_embed(x)        # (B, N, D)
        x = x + self.pos_embed         # (B, N, D)

        # 2. 条件ベクトルを生成 (タイムステップ + クラスラベル)
        cond = self.t_embedder(t) + self.y_embedder(y)  # (B, cond_dim)

        # 3. DiTBlocks
        for block in self.blocks:
            x = block(x, cond)         # (B, N, D)

        # 4. 最終層: adaLN + 線形射影
        modulation = self.final_adaln(cond)
        gamma, beta = modulation.chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        x = self.final_proj(x)         # (B, N, p*p*C)

        # 5. Unpatchify: トークン列 → 画像
        x = self.unpatchify(x)         # (B, C, H, W)

        return x

# モデル生成とテスト
model = DiT(
    img_size=32, patch_size=4, in_channels=1,
    dim=128, depth=4, num_heads=4, num_classes=10
).to(device)

# テスト入力
x_test = torch.randn(4, 1, 32, 32, device=device)
t_test = torch.randint(0, 1000, (4,), device=device)
y_test = torch.randint(0, 10, (4,), device=device)

# 順伝播
with torch.no_grad():
    out_test = model(x_test, t_test, y_test)

total_params = sum(p.numel() for p in model.parameters())
print("="*60)
print("DiT (Diffusion Transformer) モデルサマリ")
print("="*60)
print(f"画像サイズ:     {model.img_size}x{model.img_size}")
print(f"パッチサイズ:   {model.patch_size}")
print(f"パッチ数:       {model.num_patches}")
print(f"埋め込み次元:   {model.dim}")
print(f"DiTBlock数:     {len(model.blocks)}")
print(f"入力:           {x_test.shape}")
print(f"出力:           {out_test.shape}")
print(f"パラメータ数:   {total_params:,}")
print("✅ DiT モデル動作確認完了")"""
))

# ── Cell 17: Section 6 intro ──
cells.append(md(
"""<a id="section6"></a>
## 6. MNIST訓練・生成デモ

### 📊 訓練設定

教育目的の軽量設定でMNIST上のDiTを訓練します。

- データ: MNIST (28×28 → 32×32にパディング)
- バッチサイズ: 128
- エポック数: 15（教育目的）
- 拡散ステップ数: 500
- ノイズスケジュール: コサインスケジュール"""
))

# ── Cell 18: Dataset prep ──
cells.append(code(
"""# ============================================================
# MNISTデータセットの準備（28×28 → 32×32パディング）
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),                    # [0, 1]
    transforms.Pad(2),                        # 28→32 (2px padding)
    transforms.Normalize((0.5,), (0.5,)),     # [-1, 1]
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

# 教育目的でサブセットを使用（訓練を高速化）
from torch.utils.data import Subset
subset_indices = np.random.choice(len(train_dataset), 10000, replace=False)
train_subset = Subset(train_dataset, subset_indices)

train_loader = DataLoader(
    train_subset, batch_size=128, shuffle=True, drop_last=True
)

# サンプルを確認
sample_batch, sample_labels = next(iter(train_loader))
print(f"バッチ形状:   {sample_batch.shape}  (B, C, H, W)")
print(f"ラベル形状:   {sample_labels.shape}")
print(f"値の範囲:     [{sample_batch.min():.1f}, {sample_batch.max():.1f}]")
print(f"データセット:  {len(train_subset)} 枚")

# サンプル表示
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(16):
    ax = axes[i // 8, i % 8]
    img = sample_batch[i, 0].numpy() * 0.5 + 0.5
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'{sample_labels[i].item()}', fontsize=10)
    ax.axis('off')
plt.suptitle('MNIST サンプル (32×32にパディング済み)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""
))

# ── Cell 19: Diffusion scheduler ──
cells.append(code(
"""# ============================================================
# 拡散スケジューラ（コサインスケジュール）
# ============================================================

class DiffusionScheduler:
    \"\"\"DDPM拡散スケジューラ（コサインスケジュール）

    コサインスケジュールは低ノイズ領域に多くのステップを割き、
    高品質な生成を実現する。
    \"\"\"

    def __init__(self, num_timesteps=500, s=0.008, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device

        # コサインスケジュールでα_barを計算
        steps = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
        alpha_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]

        # βを計算
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        betas = torch.clamp(betas, 0.0001, 0.02)

        # 各種係数を事前計算
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bar = alpha_bar.to(device)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar).to(device)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar).to(device)

    def q_sample(self, x_0, t, noise=None):
        \"\"\"前方拡散: x_0にノイズを加える\"\"\"
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_a = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_1ma = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return sqrt_a * x_0 + sqrt_1ma * noise, noise

# スケジューラを生成
scheduler = DiffusionScheduler(num_timesteps=500, device=device)

# スケジュールの可視化
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

t_axis = np.arange(500)

axes[0].plot(t_axis, scheduler.betas.cpu().numpy(), 'b-', linewidth=2)
axes[0].set_xlabel('タイムステップ t')
axes[0].set_ylabel('β_t')
axes[0].set_title('ノイズスケジュール β_t', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_axis, scheduler.alpha_bar.cpu().numpy(), 'r-', linewidth=2)
axes[1].set_xlabel('タイムステップ t')
axes[1].set_ylabel('ᾱ_t')
axes[1].set_title('累積積 ᾱ_t', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_axis, scheduler.sqrt_one_minus_alpha_bar.cpu().numpy(), 'g-', linewidth=2)
axes[2].set_xlabel('タイムステップ t')
axes[2].set_ylabel('√(1-ᾱ_t)')
axes[2].set_title('ノイズ係数 √(1-ᾱ_t)', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.suptitle('コサインノイズスケジュール', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""
))

# ── Cell 20: Forward diffusion vis ──
cells.append(code(
"""# ============================================================
# 前方拡散の可視化
# ============================================================

def visualize_forward_diffusion():
    \"\"\"MNISTに対する前方拡散過程を可視化\"\"\"
    sample_img = sample_batch[:1].to(device)  # (1, 1, 32, 32)

    timesteps_to_show = [0, 50, 100, 200, 350, 499]
    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(18, 3.5))

    for i, t_val in enumerate(timesteps_to_show):
        t = torch.tensor([t_val], device=device)
        noisy, _ = scheduler.q_sample(sample_img, t)

        img = noisy[0, 0].cpu().numpy() * 0.5 + 0.5
        axes[i].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f't = {t_val}', fontsize=12, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle('前方拡散: クリーン画像 → 完全ノイズ\\n(コサインスケジュール)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

visualize_forward_diffusion()"""
))

# ── Cell 21: Training loop ──
cells.append(code(
"""# ============================================================
# DiTの訓練
# ============================================================

# モデルを再初期化
dit_model = DiT(
    img_size=32, patch_size=4, in_channels=1,
    dim=128, depth=4, num_heads=4, num_classes=10
).to(device)

optimizer = torch.optim.AdamW(dit_model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

NUM_EPOCHS = 15
losses = []

print(f"パラメータ数: {sum(p.numel() for p in dit_model.parameters()):,}")
print(f"デバイス: {device}")
print(f"訓練開始...")
print()

for epoch in range(NUM_EPOCHS):
    dit_model.train()
    epoch_losses = []

    for batch_imgs, batch_labels in train_loader:
        batch_imgs = batch_imgs.to(device)       # (B, 1, 32, 32)
        batch_labels = batch_labels.to(device)    # (B,)
        B = batch_imgs.shape[0]

        # ランダムなタイムステップ
        t = torch.randint(0, scheduler.num_timesteps, (B,), device=device)

        # 前方拡散
        noisy_imgs, noise = scheduler.q_sample(batch_imgs, t)

        # DiTでノイズを予測
        predicted_noise = dit_model(noisy_imgs, t, batch_labels)

        # MSE損失
        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        # 勾配クリッピング（安定化）
        torch.nn.utils.clip_grad_norm_(dit_model.parameters(), 1.0)
        optimizer.step()

        epoch_losses.append(loss.item())

    scheduler_lr.step()
    avg_loss = np.mean(epoch_losses)
    losses.append(avg_loss)

    if (epoch + 1) % 3 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS}  |  Loss: {avg_loss:.4f}  |  LR: {optimizer.param_groups[0]['lr']:.2e}")

print()
print("✅ DiT 訓練完了")"""
))

# ── Cell 22: Loss plot ──
cells.append(code(
"""# ============================================================
# 訓練損失の可視化
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(losses) + 1), losses, 'b-o', linewidth=2, markersize=5)
plt.xlabel('エポック', fontsize=12)
plt.ylabel('損失 (MSE)', fontsize=12)
plt.title('DiT 訓練損失', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, len(losses) + 1))
plt.tight_layout()
plt.show()

print(f"最終損失: {losses[-1]:.4f}")"""
))

# ── Cell 23: Sampling function ──
cells.append(code(
"""# ============================================================
# DDPMサンプリング（逆拡散）
# ============================================================

@torch.no_grad()
def sample_dit(model, scheduler, num_samples, class_labels, device,
               img_size=32, in_channels=1):
    \"\"\"DiTモデルからDDPMサンプリングを実行

    Args:
        model: 訓練済みDiTモデル
        scheduler: DiffusionScheduler
        num_samples: 生成数
        class_labels: (num_samples,) クラスラベル
        device: デバイス
    Returns:
        (num_samples, C, H, W) 生成画像
    \"\"\"
    model.eval()

    # 完全なノイズから開始
    x = torch.randn(num_samples, in_channels, img_size, img_size, device=device)

    T = scheduler.num_timesteps
    for t_val in reversed(range(T)):
        t = torch.full((num_samples,), t_val, device=device, dtype=torch.long)

        # ノイズ予測
        predicted_noise = model(x, t, class_labels)

        # DDPM逆拡散ステップ
        alpha_t = scheduler.alphas[t_val]
        alpha_bar_t = scheduler.alpha_bar[t_val]
        beta_t = scheduler.betas[t_val]

        # 平均を計算
        x = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        )

        # ノイズを追加（t > 0 の場合）
        if t_val > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta_t) * noise

    return x.clamp(-1, 1)

print("✅ サンプリング関数定義完了")"""
))

# ── Cell 24: Generate samples ──
cells.append(code(
"""# ============================================================
# サンプル生成
# ============================================================

print("生成中（500ステップの逆拡散）...")

# 各クラスから2枚ずつ生成（合計20枚）
class_labels = torch.arange(10, device=device).repeat(2)  # [0,1,...,9,0,1,...,9]

generated = sample_dit(
    dit_model, scheduler,
    num_samples=20,
    class_labels=class_labels,
    device=device
)

print(f"生成画像: {generated.shape}")

# 表示
fig, axes = plt.subplots(2, 10, figsize=(18, 4))
for i in range(20):
    row = i // 10
    col = i % 10
    img = generated[i, 0].cpu().numpy() * 0.5 + 0.5
    axes[row, col].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
    axes[row, col].set_title(f'{class_labels[i].item()}', fontsize=10)
    axes[row, col].axis('off')

plt.suptitle('DiTによる条件付き生成 (各数字クラス)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""
))

# ── Cell 25: Reverse diffusion vis ──
cells.append(code(
"""# ============================================================
# 逆拡散過程の可視化
# ============================================================

@torch.no_grad()
def visualize_reverse_process(model, scheduler, class_label, device, steps_show=8):
    \"\"\"逆拡散の途中ステップを可視化\"\"\"
    model.eval()
    x = torch.randn(1, 1, 32, 32, device=device)
    y = torch.tensor([class_label], device=device)

    T = scheduler.num_timesteps
    show_at = np.linspace(T-1, 0, steps_show, dtype=int)
    snapshots = {}

    for t_val in reversed(range(T)):
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        predicted_noise = model(x, t, y)

        alpha_t = scheduler.alphas[t_val]
        alpha_bar_t = scheduler.alpha_bar[t_val]
        beta_t = scheduler.betas[t_val]

        x = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        )
        if t_val > 0:
            x = x + torch.sqrt(beta_t) * torch.randn_like(x)

        if t_val in show_at:
            snapshots[t_val] = x.clone().cpu()

    # 可視化
    fig, axes = plt.subplots(1, steps_show, figsize=(2.5 * steps_show, 3))
    for i, t_val in enumerate(sorted(snapshots.keys(), reverse=True)):
        img = snapshots[t_val][0, 0].numpy() * 0.5 + 0.5
        axes[i].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f't={t_val}', fontsize=11, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle(f'逆拡散過程 (クラス: {class_label})\\nノイズ → 画像',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# 数字 "3" と "7" の生成過程
visualize_reverse_process(dit_model, scheduler, class_label=3, device=device)
visualize_reverse_process(dit_model, scheduler, class_label=7, device=device)"""
))

# ── Cell 26: Compare real vs generated ──
cells.append(code(
"""# ============================================================
# 実画像と生成画像の比較
# ============================================================

def compare_real_generated():
    \"\"\"実画像と生成画像を並べて比較\"\"\"
    fig, axes = plt.subplots(3, 10, figsize=(18, 6))

    # 各クラスの実画像を取得
    real_imgs = {}
    for img, label in train_subset:
        l = label if isinstance(label, int) else label.item()
        if l not in real_imgs:
            real_imgs[l] = img
        if len(real_imgs) == 10:
            break

    # 実画像（上段）
    for i in range(10):
        img = real_imgs[i][0].numpy() * 0.5 + 0.5
        axes[0, i].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'{i}', fontsize=10)
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('実画像', fontsize=12, fontweight='bold')

    # 生成画像（中段・下段：2セット）
    for row in range(1, 3):
        labels = torch.arange(10, device=device)
        gen = sample_dit(dit_model, scheduler, 10, labels, device)
        for i in range(10):
            img = gen[i, 0].cpu().numpy() * 0.5 + 0.5
            axes[row, i].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
            axes[row, i].axis('off')
        axes[row, 0].set_ylabel(f'DiT生成{row}', fontsize=12, fontweight='bold')

    plt.suptitle('実画像 vs DiT生成画像', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

print("実画像とDiT生成画像を比較中...")
compare_real_generated()"""
))

# ── Cell 27: Section 7 intro ──
cells.append(md(
"""<a id="section7"></a>
## 7. Soraとの技術的関連

### 📊 DiTからSoraへ

OpenAIの動画生成モデル**Sora**は、DiTの延長線上にある技術です。
DiTの設計原理がどのようにSoraに活かされているかを整理します。

### DiT → Sora の技術的な進化

| 要素 | DiT (画像) | Sora (動画) |
|------|-----------|------------|
| 入力 | 2Dパッチ $(H/p \\times W/p)$ | **3D時空間パッチ** $(T/p_t \\times H/p \\times W/p)$ |
| トークン | 空間パッチ | **時空間パッチ**（動画の立方体） |
| 位置埋め込み | 2D | **3D**（時間+空間） |
| Attention | 空間のみ | **時空間** (Notebook 130参照) |
| 条件付け | クラスラベル | **テキスト** (CLIP/T5) |
| 潜在空間 | VAE (2D) | **Video VAE** (3D) |
| スケーリング | 〜数億パラメータ | **数十億パラメータ** |

### 💡 Soraの核心技術

1. **Spacetime Patches**: 動画を3D立方体に分割してトークン化
2. **Variable Resolution**: 異なる解像度・アスペクト比に対応
3. **Scaling Law**: DiTで示されたスケーリング則を動画に拡張
4. **Video VAE**: 動画の潜在空間で効率的に拡散"""
))

# ── Cell 28: Sora architecture diagram ──
cells.append(code(
"""# ============================================================
# DiT → Sora のスケーリング比較図
# ============================================================

def visualize_dit_to_sora():
    \"\"\"DiTからSoraへの技術的進化を図示\"\"\"
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- 左: DiT (画像) ---
    ax = axes[0]
    # 2Dパッチの図
    for i in range(4):
        for j in range(4):
            color = plt.cm.Set3(i * 4 + j)
            rect = plt.Rectangle((j, 3-i), 0.9, 0.9,
                                  facecolor=color, edgecolor='gray', lw=1)
            ax.add_patch(rect)

    ax.text(2, -0.8, '2Dパッチ → トークン列', ha='center', fontsize=12)
    ax.text(2, -1.3, f'パッチ数: {4}×{4} = {16}', ha='center', fontsize=11)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-2, 4.5)
    ax.set_aspect('equal')
    ax.set_title('DiT: 画像の2Dパッチ\\n(H/p × W/p)', fontsize=13, fontweight='bold')
    ax.axis('off')

    # --- 右: Sora (動画) ---
    ax = axes[1]
    # 3D時空間パッチの図（擬似3D）
    for t in range(3):
        offset_x = t * 1.5
        offset_y = t * 0.8
        for i in range(3):
            for j in range(3):
                color = plt.cm.Set3(t * 9 + i * 3 + j)
                x = j + offset_x
                y = (2-i) + offset_y
                rect = plt.Rectangle((x, y), 0.85, 0.85,
                                      facecolor=color, edgecolor='gray',
                                      lw=1, alpha=0.8)
                ax.add_patch(rect)
        ax.text(1 + offset_x, -0.3 + offset_y, f't={t}', ha='center', fontsize=10)

    ax.text(4.5, -0.8, '3D時空間パッチ → トークン列', ha='center', fontsize=12)
    ax.text(4.5, -1.3, f'パッチ数: T/p_t × H/p × W/p', ha='center', fontsize=11)
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-2, 5.5)
    ax.set_aspect('equal')
    ax.set_title('Sora: 動画の3D時空間パッチ\\n(T/p_t × H/p × W/p)', fontsize=13, fontweight='bold')
    ax.axis('off')

    plt.suptitle('DiT → Sora: 2Dパッチから3D時空間パッチへ',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

visualize_dit_to_sora()"""
))

# ── Cell 29: Scaling law vis ──
cells.append(code(
"""# ============================================================
# DiTのスケーリング則（概念図）
# ============================================================

def visualize_scaling_law():
    \"\"\"DiTのスケーリング則を概念的に可視化\"\"\"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- 左: モデルサイズ vs FID ---
    model_sizes = [33, 130, 460, 675]  # DiT-S, B, L, XL (Mパラメータ)
    fid_scores = [68.4, 43.5, 23.3, 9.6]  # 概念的なFID
    labels = ['DiT-S/2', 'DiT-B/2', 'DiT-L/2', 'DiT-XL/2']

    axes[0].plot(model_sizes, fid_scores, 'b-o', linewidth=2, markersize=8)
    for i, label in enumerate(labels):
        axes[0].annotate(label, (model_sizes[i], fid_scores[i]),
                         textcoords="offset points", xytext=(10, 10), fontsize=10)
    axes[0].set_xlabel('パラメータ数 (M)', fontsize=12)
    axes[0].set_ylabel('FID (低いほど良い)', fontsize=12)
    axes[0].set_title('DiTのスケーリング則\\nモデルサイズ vs 生成品質', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()

    # --- 右: パッチサイズの影響 ---
    patch_sizes = [2, 4, 8]
    num_tokens = [(256//p)**2 for p in patch_sizes]
    fid_by_patch = [9.6, 23.3, 55.7]  # 概念的

    colors = ['green', 'orange', 'red']
    bars = axes[1].bar(range(len(patch_sizes)), fid_by_patch, color=colors, alpha=0.7,
                       edgecolor='black')
    axes[1].set_xticks(range(len(patch_sizes)))
    axes[1].set_xticklabels([f'p={p}\\n({n} tokens)' for p, n in zip(patch_sizes, num_tokens)],
                            fontsize=10)
    axes[1].set_ylabel('FID (低いほど良い)', fontsize=12)
    axes[1].set_title('パッチサイズの影響\\n(DiT-XL)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    for bar, fid in zip(bars, fid_by_patch):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'FID={fid}', ha='center', fontsize=10)

    plt.suptitle('DiTはTransformerのスケーリング則に従う\\n'
                 '(大きいモデル + 小さいパッチ = 高品質)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("重要な知見:")
    print("1. モデルを大きくするほどFIDが改善 → Scaling Law が成立")
    print("2. パッチサイズを小さくするほど品質向上（ただし計算量は増加）")
    print("3. この特性がSoraのような大規模動画生成モデルの基盤となった")

visualize_scaling_law()"""
))

# ── Cell 30: Parameter comparison table ──
cells.append(code(
"""# ============================================================
# DiT設定の比較表
# ============================================================

def show_dit_variants():
    \"\"\"DiTの各バリアントの仕様を表示\"\"\"
    print("="*70)
    print("DiT バリアントの比較")
    print("="*70)
    print(f"{'モデル':<12} {'レイヤ数':>8} {'隠れ次元':>8} {'ヘッド数':>8} {'パラメータ':>12}")
    print("-"*70)

    variants = [
        ('DiT-S/8',  12,  384,   6,  '33M'),
        ('DiT-B/4',  12,  768,  12, '130M'),
        ('DiT-L/4',  24, 1024,  16, '460M'),
        ('DiT-XL/2', 28, 1152,  16, '675M'),
        ('本NB版',     4,  128,   4, f'{sum(p.numel() for p in dit_model.parameters())/1e3:.0f}K'),
    ]

    for name, layers, dim, heads, params in variants:
        marker = ' ← 教育用' if name == '本NB版' else ''
        print(f"{name:<12} {layers:>8} {dim:>8} {heads:>8} {params:>12}{marker}")

    print()
    print("※ DiT-XL/2 はImageNet 256×256でFID=2.27を達成（当時のSoTA）")
    print("※ /2, /4, /8 はパッチサイズを示す")

show_dit_variants()"""
))

# ── Cell 31: Summary markdown ──
cells.append(md(
"""<a id="summary"></a>
## 8. まとめ

### 🎯 このノートブックで学んだこと

**PatchEmbed（パッチ埋め込み）**
- ✓ Conv2d(kernel_size=p, stride=p) でパッチ分割と線形射影を同時実行
- ✓ 画像 (C, H, W) → トークン列 (N, D) への変換

**adaLN-Zero条件付け**
- ✓ LayerNormのγ, βを条件ベクトルから動的に生成
- ✓ ゲートαを0初期化することで訓練が安定（恒等写像からスタート）
- ✓ タイムステップとクラスラベルを加算で結合

**DiTBlock**
- ✓ Pre-norm Transformer + adaLN-Zero
- ✓ Attention出力とFFN出力にαゲーティング
- ✓ 標準的なTransformerと同じスケーリング特性

**DiTモデル全体**
- ✓ PatchEmbed → 位置埋め込み → DiTBlock×N → Unpatchify
- ✓ MNISTでの条件付き生成を実演

**Soraとの関連**
- ✓ 2Dパッチ → 3D時空間パッチへの拡張
- ✓ Scaling Lawに従うため、大規模化が容易

### 📊 DiTチートシート

| コンポーネント | 役割 | 実装のポイント |
|---------------|------|---------------|
| PatchEmbed | 画像→トークン | Conv2d(k=p, s=p) |
| adaLN-Zero | 条件付きLayerNorm | α=0初期化 |
| DiTBlock | Transformer + 条件付け | Pre-norm + αゲート |
| Unpatchify | トークン→画像 | reshape + permute |"""
))

# ── Cell 32: Common errors ──
cells.append(md(
"""### ⚠️ よくあるエラー

#### エラー #1: パッチサイズと画像サイズの不整合

```python
# ❌ 28×28の画像にpatch_size=4 → 28/4=7（割り切れるが端が不均一）
model = DiT(img_size=28, patch_size=4)  # 28は4で割り切れる

# ✅ 32×32にパディングしてからpatch_size=4
transforms.Pad(2)  # 28→32
model = DiT(img_size=32, patch_size=4)  # 32/4=8 ぴったり
```

**理由**: パッチサイズで画像サイズが割り切れないと、端のピクセルが失われます。

---

#### エラー #2: adaLN-Zeroの初期化忘れ

```python
# ❌ デフォルト初期化 → 訓練初期から大きな出力が出て不安定
self.adaln_modulation = nn.Sequential(
    nn.SiLU(),
    nn.Linear(cond_dim, 3 * dim)
)

# ✅ 0初期化 → 初期は恒等写像で安定
nn.init.zeros_(self.adaln_modulation[-1].weight)
nn.init.zeros_(self.adaln_modulation[-1].bias)
```

**理由**: ゲートαが0でないと、初期状態で大きなランダム出力が加算され、学習が不安定になります。

---

#### エラー #3: Unpatchifyの次元操作ミス

```python
# ❌ reshapeの順序を間違えると画像が壊れる
x = x.reshape(B, C, H, W)  # 直接reshapeは危険

# ✅ 正しい手順: (B, h*w, p*p*C) → (B, h, w, p, p, C) → permute → reshape
x = x.reshape(B, h, w, p, p, C)
x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, h*p, w*p)
```

**理由**: パッチの並び順と画像のピクセル順序は異なるため、正しいpermute操作が必要です。"""
))

# ── Cell 33: Quiz ──
cells.append(md(
"""## 🎓 自己評価クイズ

学習内容を確認しましょう！すぐに答えを見ずに、まず自分で考えてみてください。

### Q1: DiTがU-Netの代わりにTransformerを使う最大のメリットは何ですか？

<details>
<summary>💡 答えを見る</summary>

**答え**: **スケーリング則（Scaling Law）に従う**ことです。

U-Netは構造が複雑で、モデルを大きくする方法が明確ではありません（チャンネル数？層数？スキップ接続？）。
一方、DiTは標準的なTransformerなので、層を積み重ねるだけでモデルサイズを増やせ、
NLPで実証されたスケーリング則がそのまま適用できます。

DiT-XL/2はImageNet 256×256でFID=2.27を達成し、U-Netベースのモデルを上回りました。

</details>

---

### Q2: adaLN-Zeroの「Zero」とは何を意味し、なぜ重要ですか？

<details>
<summary>💡 答えを見る</summary>

**答え**: ゲートパラメータ $\\alpha$ を**0で初期化**することを意味します。

$\\alpha = 0$ のとき、DiTBlockの出力は：
$$x_{\\text{out}} = x + \\alpha \\cdot \\text{Attn}(x) = x + 0 = x$$

つまり、訓練開始時に各DiTBlockは**恒等写像**として機能します。
これにより深いネットワークでも勾配が安定して伝播し、訓練の初期段階が安定します。

ResNetの残差接続と同様の効果ですが、adaLN-Zeroはこれを**学習可能なゲート**で実現しています。

</details>

---

### Q3: PatchEmbedでConv2d(kernel_size=p, stride=p)を使う理由は？

<details>
<summary>💡 答えを見る</summary>

**答え**: パッチ分割と線形射影を**1つの操作で効率的に行う**ためです。

- `kernel_size=p`: 各パッチのサイズ（p×p）を指定
- `stride=p`: パッチ同士が重ならない（非重複）ことを保証

これは以下の2ステップを1つに圧縮しています：
1. 画像をp×pパッチに分割 → 各パッチを $C \\times p \\times p$ ベクトルに展開
2. 線形変換 $W \\in \\mathbb{R}^{d \\times (C \\cdot p^2)}$ でembedding次元に射影

Conv2dの出力特徴マップのサイズは自動的に $(H/p, W/p)$ になります。

</details>

---

### Q4: DiTの条件付けで、タイムステップとクラスラベルを「加算」で結合する設計の利点は？

<details>
<summary>💡 答えを見る</summary>

**答え**: **シンプルさ**と**パラメータ効率**です。

加算で結合すると、1つの条件ベクトル $c = c_t + c_y$ に集約されるため：
- adaLN-Zeroのパラメータが1セット（γ, β, α）で済む
- Cross-Attentionのような追加モジュールが不要
- 計算コストが低い

連結（concatenation）やCross-Attentionに比べてシンプルですが、
DiT論文の実験では十分な性能を達成しています。
テキスト条件付けのような長いシーケンスの場合は、Cross-Attentionが必要になります。

</details>

---

### Q5: DiTからSoraへの最も重要な技術的拡張は何ですか？

<details>
<summary>💡 答えを見る</summary>

**答え**: **2Dパッチから3D時空間パッチ（Spacetime Patches）への拡張**です。

DiTでは画像を2Dパッチに分割しますが、Soraでは動画を3D立方体に分割します：

- DiT: $(H/p \\times W/p)$ パッチ → 空間トークン
- Sora: $(T/p_t \\times H/p \\times W/p)$ パッチ → **時空間トークン**

これにより：
1. 時間的な一貫性を自然に学習
2. 可変長動画・可変解像度に対応
3. DiTで実証されたスケーリング則をそのまま適用

Notebook 130で学んだTemporal AttentionとNotebook 131のVideo Diffusionが、
ここで統合されていることが分かります。

</details>"""
))

# ── Cell 34: Checklist + next steps ──
cells.append(md(
"""---

### ✅ 学習チェックリスト

- [ ] PatchEmbedの仕組みをConv2dの観点から説明できる
- [ ] adaLN-Zeroの3つの出力（γ, β, α）の役割を説明できる
- [ ] DiTBlockの情報フローを図示できる
- [ ] DiTモデル全体のアーキテクチャを説明できる
- [ ] U-NetとDiTの違いを3つ以上挙げられる
- [ ] SoraがDiTをどのように拡張しているか説明できる

---

### 📚 参考文献

- Peebles & Xie, "Scalable Diffusion Models with Transformers" (ICCV 2023)
- Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT, ICLR 2021)
- OpenAI, "Video generation models as world simulators" (Sora Technical Report, 2024)

---

**次のステップ**: DiTの基礎を理解したので、次は動画生成への応用や、
テキスト条件付けとの統合など、より高度なトピックに進みましょう！"""
))

# ── Fix cell IDs ──
for i, cell in enumerate(cells):
    cell["id"] = f"cell-{i}"
    # Source should be a list of strings with newlines
    if isinstance(cell["source"], list):
        # Rejoin and re-split properly
        src = "\n".join(cell["source"])
        lines = src.split("\n")
        cell["source"] = [line + "\n" if j < len(lines) - 1 else line
                          for j, line in enumerate(lines)]

nb = {
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
            "version": "3.11.0"
        }
    },
    "cells": cells
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "132_diffusion_transformer_dit_v1.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Written {out_path}")
print(f"  {len(cells)} cells")
