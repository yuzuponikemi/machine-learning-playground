#!/usr/bin/env python3
"""Build notebook 143 - Part 3: cells 33-48."""
import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n"), "id": None}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source.split("\n"),
            "outputs": [], "execution_count": None, "id": None}

cells = []

# ============================================================
# Cell 33: Policy improvement visualization header
# ============================================================
cells.append(md("""### 7.2 方策の改善過程

想像の中での訓練によって、Actor の方策がどのように改善されるかを確認します。
訓練前と訓練後で、同じ初期状態からの行動を比較してみましょう。"""))

# ============================================================
# Cell 34: Policy improvement visualization
# ============================================================
cells.append(code("""# 方策改善の効果を可視化
# 訓練後の方策で複数エピソードを実行
n_eval_episodes = 10
all_eval_rewards = []
all_eval_positions = []

dreamer.actor.eval()
dreamer.world_model.eval()

for ep in range(n_eval_episodes):
    env_eval = BouncingBall1D(max_steps=50)
    obs = env_eval.reset()
    ep_rewards = []
    ep_positions = [env_eval.pos]

    h, z = dreamer.world_model.rssm.initial_state(1)

    with torch.no_grad():
        for t in range(50):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            obs_embed = dreamer.world_model.encoder(obs_t)
            act_dummy = torch.zeros(1, 2, device=device)
            h, z, _, _ = dreamer.world_model.rssm(h, z, act_dummy, obs_embed)
            state = dreamer.world_model.get_state(h, z)
            action = dreamer.actor(state)
            action_np = action.cpu().numpy()[0]

            obs, reward, done, info = env_eval.step(action_np)
            ep_rewards.append(reward)
            ep_positions.append(info['pos'])
            if done:
                break

    all_eval_rewards.append(sum(ep_rewards))
    all_eval_positions.append(ep_positions)

# ランダム方策でも同様に実行
random_rewards = []
random_positions = []

for ep in range(n_eval_episodes):
    env_rnd = BouncingBall1D(max_steps=50)
    obs = env_rnd.reset()
    ep_rewards = []
    ep_positions = [env_rnd.pos]

    for t in range(50):
        action = np.random.randn(2) * 0.5
        obs, reward, done, info = env_rnd.step(action)
        ep_rewards.append(reward)
        ep_positions.append(info['pos'])
        if done:
            break

    random_rewards.append(sum(ep_rewards))
    random_positions.append(ep_positions)

# --- 可視化 ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. 軌道の比較
for pos_list in random_positions[:3]:
    axes[0].plot(pos_list, color='gray', alpha=0.4, linewidth=1)
axes[0].plot(random_positions[0], color='gray', alpha=0.4, linewidth=1, label='ランダム方策')

for pos_list in all_eval_positions[:3]:
    axes[0].plot(pos_list, color='#3498DB', alpha=0.7, linewidth=1.5)
axes[0].plot(all_eval_positions[0], color='#3498DB', alpha=0.7, linewidth=1.5,
             label='学習済み方策')

axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='目標')
axes[0].set_title('軌道の比較', fontsize=12)
axes[0].set_xlabel('ステップ')
axes[0].set_ylabel('位置')
axes[0].set_ylim(-2.5, 2.5)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# 2. 累積報酬の分布
axes[1].boxplot([random_rewards, all_eval_rewards],
                labels=['ランダム', '学習済み'])
axes[1].set_title('累積報酬の比較', fontsize=12)
axes[1].set_ylabel('累積報酬')
axes[1].grid(True, alpha=0.3)

# 3. 位置の絶対値（目標からの距離）
random_mean_dist = [np.mean(np.abs(pos)) for pos in random_positions]
learned_mean_dist = [np.mean(np.abs(pos)) for pos in all_eval_positions]

axes[2].bar([0, 1], [np.mean(random_mean_dist), np.mean(learned_mean_dist)],
            yerr=[np.std(random_mean_dist), np.std(learned_mean_dist)],
            color=['gray', '#3498DB'], capsize=5)
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['ランダム', '学習済み'])
axes[2].set_title('平均距離（中心からの乖離）', fontsize=12)
axes[2].set_ylabel('平均 |位置|')
axes[2].grid(True, alpha=0.3)

plt.suptitle('方策の改善: ランダム vs 学習済み（夢で訓練）', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print(f"ランダム方策の平均累積報酬: {np.mean(random_rewards):.2f} +/- {np.std(random_rewards):.2f}")
print(f"学習済み方策の平均累積報酬: {np.mean(all_eval_rewards):.2f} +/- {np.std(all_eval_rewards):.2f}")"""))

# ============================================================
# Cell 35: Section 8 Header
# ============================================================
cells.append(md("""---

## 8. DreamerV3の技術的革新

### 8.1 Symlog Predictions

DreamerV3 では、報酬や価値のスケールが大きく異なるタスクに対応するため、
**symlog 変換** を使用します。

$$
\\text{symlog}(x) = \\text{sign}(x) \\cdot \\ln(|x| + 1)
$$

これにより、小さな値も大きな値も均等に扱えるようになります。"""))

# ============================================================
# Cell 36: Symlog
# ============================================================
cells.append(code("""def symlog(x):
    \"\"\"
    Symlog 変換: sign(x) * ln(|x| + 1)

    小さな値には線形に近く、大きな値には対数的に作用する。
    報酬のスケールが異なるタスク間で共通のネットワークを使えるようにする。
    \"\"\"
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x):
    \"\"\"Symlog の逆変換\"\"\"
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# 可視化
x = torch.linspace(-100, 100, 1000)
y_symlog = symlog(x)
y_log = torch.sign(x) * torch.log10(torch.abs(x) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Symlog vs 線形
axes[0].plot(x.numpy(), x.numpy(), 'gray', alpha=0.3, label='線形 y=x')
axes[0].plot(x.numpy(), y_symlog.numpy(), '#E74C3C', linewidth=2, label='symlog(x)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Symlog 変換', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-10, 10)

# 逆変換の検証
x_test = torch.tensor([-50.0, -1.0, 0.0, 1.0, 50.0, 1000.0])
y_test = symlog(x_test)
x_recovered = symexp(y_test)

axes[1].axis('off')
table_data = []
for xi, yi, xr in zip(x_test, y_test, x_recovered):
    table_data.append([f'{xi.item():.1f}', f'{yi.item():.4f}', f'{xr.item():.4f}'])

table = axes[1].table(cellText=table_data,
                      colLabels=['x', 'symlog(x)', 'symexp(symlog(x))'],
                      loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)
axes[1].set_title('Symlog の可逆性の検証', fontsize=12)

plt.tight_layout()
plt.show()

print("【Symlog の利点】")
print("・報酬が 0.01 のタスクも 10000 のタスクも同じネットワークで扱える")
print("・勾配の大きさが安定する")
print("・DreamerV3 がドメイン横断で強い理由の1つ")"""))

# ============================================================
# Cell 37: Discrete Latent States
# ============================================================
cells.append(md("""### 8.2 離散潜在状態（Categorical Latent）

DreamerV3 では、確率的状態 $z_t$ を **離散カテゴリカル分布** で表現します（本ノートブックではガウスで簡略化しました）。

**なぜ離散か？**

| 特性 | ガウス潜在 | カテゴリカル潜在 |
|------|----------|----------------|
| **表現力** | 連続空間の任意の点 | 離散シンボルの組み合わせ |
| **KL 制御** | KL が滑らかに変化 | KL の最小値が明確（離散的） |
| **多峰性** | 単峰（1つの山） | 多峰（複数の選択肢を表現可能） |
| **計算** | Reparameterization trick | Straight-Through + Gumbel-Softmax |

DreamerV3 では 32 個のカテゴリカル変数 x 32 クラス = **1024 次元** の離散潜在空間を使用します。"""))

# ============================================================
# Cell 38: Discrete latent demo
# ============================================================
cells.append(code("""# カテゴリカル潜在状態のデモ
# DreamerV3 の実際の設計: 32カテゴリ x 32クラス

def categorical_latent_demo():
    \"\"\"カテゴリカル潜在状態の動作を示すデモ\"\"\"
    n_categories = 8   # 簡略化: 8カテゴリ
    n_classes = 8      # 各カテゴリのクラス数

    # ロジット（ネットワークの出力を想定）
    logits = torch.randn(1, n_categories, n_classes)

    # Gumbel-Softmax でサンプリング（微分可能な離散サンプリング）
    # temperature が低いほど one-hot に近づく
    temperatures = [5.0, 1.0, 0.1]

    fig, axes = plt.subplots(1, len(temperatures) + 1, figsize=(16, 4))

    # ソフトマックス（温度=1、参考用）
    probs = F.softmax(logits[0], dim=-1).detach().numpy()
    im = axes[0].imshow(probs, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    axes[0].set_title('Softmax 確率', fontsize=11)
    axes[0].set_xlabel('クラス')
    axes[0].set_ylabel('カテゴリ')
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    for i, temp in enumerate(temperatures):
        # Gumbel-Softmax サンプリング
        sample = F.gumbel_softmax(logits[0], tau=temp, hard=False)
        sample_np = sample.detach().numpy()

        im = axes[i+1].imshow(sample_np, cmap='Reds', vmin=0, vmax=1, aspect='auto')
        axes[i+1].set_title(f'Gumbel-Softmax\\n(temp={temp})', fontsize=11)
        axes[i+1].set_xlabel('クラス')
        if i == 0:
            axes[i+1].set_ylabel('カテゴリ')
        plt.colorbar(im, ax=axes[i+1], fraction=0.046)

    plt.suptitle('カテゴリカル潜在状態: 温度によるサンプリングの変化', fontsize=13)
    plt.tight_layout()
    plt.show()

    print("【温度の効果】")
    print("・高温 (temp=5.0): 一様分布に近い → 探索的")
    print("・中温 (temp=1.0): 適度な確率的選択")
    print("・低温 (temp=0.1): one-hot に近い → 決定的")

categorical_latent_demo()"""))

# ============================================================
# Cell 39: Scaling
# ============================================================
cells.append(md("""### 8.3 ドメイン横断のスケーリング

DreamerV3 の最大の革新は、**ハイパーパラメータを固定したまま** 150以上の異なるタスクで高性能を達成した点です。

**スケーリングの鍵**:
1. **Symlog**: 報酬スケールの正規化
2. **カテゴリカル潜在**: 安定した KL 制御
3. **Percentile scaling**: Actor の勾配を正規化
4. **Unimix categorical**: 探索の下限を保証"""))

# ============================================================
# Cell 40: DreamerV1 vs V2 vs V3
# ============================================================
cells.append(code("""# DreamerV1 vs V2 vs V3 の比較表
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')

table_data = [
    ['特徴', 'DreamerV1 (2020)', 'DreamerV2 (2021)', 'DreamerV3 (2023)'],
    ['潜在状態', 'ガウス (連続)', 'カテゴリカル (離散)', 'カテゴリカル (離散)'],
    ['報酬変換', 'なし', 'なし', 'Symlog'],
    ['KL制御', 'KL ペナルティ', 'KL バランシング', 'Free bits + バランシング'],
    ['Actor学習', 'Reinforce + STE', 'Reinforce + STE', 'Percentile scaling'],
    ['Critic', 'MLP', 'MLP', 'MLP + Symlog'],
    ['対象ドメイン', 'Atari, DMC', 'Atari (人間超え)', '150+タスク (固定HP)'],
    ['探索', 'なし', 'Plan2Explore', 'Unimix categorical'],
    ['ネットワーク', '小〜中規模', '中規模', 'XLサイズまでスケール'],
    ['主要な成果', 'Model-based RLの\\n実用化', 'Atariで\\n人間レベル超え', 'ドメイン横断\\n固定HP'],
]

colors = [['#D5DBDB'] * 4]  # ヘッダー
for _ in range(len(table_data) - 1):
    colors.append(['#EBF5FB', '#FDEDEC', '#FEF9E7', '#E8F8F5'])

table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                 cellColours=colors)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 2.0)

# ヘッダーのスタイル
for j in range(4):
    table[0, j].set_text_props(fontweight='bold')

ax.set_title('DreamerV1 → V2 → V3 の進化', fontsize=15, pad=20)
plt.tight_layout()
plt.show()

print("【DreamerV3 の最大の貢献】")
print("ハイパーパラメータを一切チューニングせずに、")
print("Atari、DMControl、Minecraft など 150+ のタスクで")
print("最先端またはそれに匹敵する性能を達成した。")"""))

# ============================================================
# Cell 41: Architecture summary
# ============================================================
cells.append(code("""# DreamerV3 のアーキテクチャ全体図
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(-1, 15)
ax.set_ylim(-1, 11)
ax.axis('off')

# --- 大きな枠: World Model ---
wm_rect = FancyBboxPatch((0, 3.5), 9, 6.5, boxstyle="round,pad=0.3",
                         facecolor='#EBF5FB', edgecolor='#2980B9', linewidth=2.5,
                         linestyle='--')
ax.add_patch(wm_rect)
ax.text(4.5, 9.5, 'World Model', ha='center', fontsize=14,
        fontweight='bold', color='#2980B9')

# RSSM
rssm_rect = FancyBboxPatch((0.5, 5.5), 4, 3, boxstyle="round,pad=0.15",
                           facecolor='#AED6F1', edgecolor='#2980B9', linewidth=2)
ax.add_patch(rssm_rect)
ax.text(2.5, 8, 'RSSM', ha='center', fontsize=13, fontweight='bold')
ax.text(2.5, 7.2, 'GRU ($h_t$)', ha='center', fontsize=10)
ax.text(2.5, 6.5, 'Prior/Posterior ($z_t$)', ha='center', fontsize=10)

# Encoder
enc_rect = FancyBboxPatch((0.5, 4), 2.5, 1.2, boxstyle="round,pad=0.1",
                          facecolor='#ABEBC6', edgecolor='#27AE60', linewidth=1.5)
ax.add_patch(enc_rect)
ax.text(1.75, 4.6, 'Encoder', ha='center', fontsize=11, fontweight='bold')

# Decoder
dec_rect = FancyBboxPatch((5, 7), 3.5, 1.2, boxstyle="round,pad=0.1",
                          facecolor='#FADBD8', edgecolor='#E74C3C', linewidth=1.5)
ax.add_patch(dec_rect)
ax.text(6.75, 7.6, 'Decoder', ha='center', fontsize=11, fontweight='bold')

# Reward Predictor
rew_rect = FancyBboxPatch((5, 5.5), 3.5, 1.2, boxstyle="round,pad=0.1",
                          facecolor='#F9E79F', edgecolor='#F39C12', linewidth=1.5)
ax.add_patch(rew_rect)
ax.text(6.75, 6.1, 'Reward Pred', ha='center', fontsize=11, fontweight='bold')

# Continue Predictor
cont_rect = FancyBboxPatch((5, 4), 3.5, 1.2, boxstyle="round,pad=0.1",
                           facecolor='#D7BDE2', edgecolor='#8E44AD', linewidth=1.5)
ax.add_patch(cont_rect)
ax.text(6.75, 4.6, 'Continue Pred', ha='center', fontsize=11, fontweight='bold')

# --- Actor & Critic（World Model の外） ---
actor_rect = FancyBboxPatch((10, 7), 4, 2, boxstyle="round,pad=0.15",
                            facecolor='#F5B7B1', edgecolor='#E74C3C', linewidth=2)
ax.add_patch(actor_rect)
ax.text(12, 8.3, 'Actor $\\pi_\\theta$', ha='center', fontsize=13, fontweight='bold')
ax.text(12, 7.5, '$s_t \\to a_t$', ha='center', fontsize=11)

critic_rect = FancyBboxPatch((10, 4.5), 4, 2, boxstyle="round,pad=0.15",
                             facecolor='#FAD7A0', edgecolor='#F39C12', linewidth=2)
ax.add_patch(critic_rect)
ax.text(12, 5.8, 'Critic $V_\\psi$', ha='center', fontsize=13, fontweight='bold')
ax.text(12, 5.0, '$s_t \\to v_t$', ha='center', fontsize=11)

# --- 環境 ---
env_rect = FancyBboxPatch((4, 0.5), 5, 2, boxstyle="round,pad=0.2",
                          facecolor='#D5DBDB', edgecolor='#2C3E50', linewidth=2.5)
ax.add_patch(env_rect)
ax.text(6.5, 1.5, 'Environment', ha='center', fontsize=13, fontweight='bold')

# --- 矢印 ---
arrow_kw = dict(arrowstyle='->', lw=1.5, mutation_scale=15)

# Env → Encoder
ax.annotate('$o_t$', xy=(1.75, 4), xytext=(4.5, 2.7),
            fontsize=10, ha='center',
            arrowprops={**arrow_kw, 'color': '#27AE60'})

# RSSM → Heads
ax.annotate('$s_t$', xy=(5, 7.6), xytext=(4.5, 7.0),
            fontsize=10, arrowprops={**arrow_kw, 'color': '#E74C3C'})
ax.annotate('', xy=(5, 6.1), xytext=(4.5, 6.5),
            arrowprops={**arrow_kw, 'color': '#F39C12'})
ax.annotate('', xy=(5, 4.6), xytext=(4.5, 5.5),
            arrowprops={**arrow_kw, 'color': '#8E44AD'})

# State → Actor/Critic
ax.annotate('$s_t$', xy=(10, 8.0), xytext=(9, 7.5),
            fontsize=10, arrowprops={**arrow_kw, 'color': '#E74C3C'})
ax.annotate('$s_t$', xy=(10, 5.5), xytext=(9, 6.0),
            fontsize=10, arrowprops={**arrow_kw, 'color': '#F39C12'})

# Actor → Env
ax.annotate('$a_t$', xy=(8.5, 2.5), xytext=(12, 7.0),
            fontsize=10, ha='center',
            arrowprops={**arrow_kw, 'color': '#E74C3C',
                       'connectionstyle': 'arc3,rad=0.3'})

ax.set_title('DreamerV3 アーキテクチャ全体図', fontsize=16, pad=15)
plt.tight_layout()
plt.show()"""))

# ============================================================
# Cell 42: Section 9 Header
# ============================================================
cells.append(md("""---

## 9. まとめ・よくあるエラー・確認テスト

### 9.1 まとめ

このノートブックで学んだことを振り返ります。"""))

# ============================================================
# Cell 43: Summary
# ============================================================
cells.append(code("""# 学習内容のまとめ
fig, ax = plt.subplots(figsize=(13, 8))
ax.axis('off')

summary_text = \"\"\"
  DreamerV3: 夢の中で数千回試行する世界モデル — 学習のまとめ

  1. DreamerV3 の全体像
     - 3フェーズ学習: 世界モデル学習 → 想像 → データ収集
     - 「心的シミュレーション」をニューラルネットワークで実現
     - サンプル効率の大幅な向上

  2. RSSM（Recurrent State Space Model）
     - 確定的状態 h_t (GRU): 過去の履歴を圧縮
     - 確率的状態 z_t: 環境の不確実性を表現
     - Prior p(z|h) と Posterior q(z|h,o) の学習

  3. WorldModel
     - Encoder + RSSM + Decoder + Reward/Continue Predictor
     - 再構成損失 + 報酬損失 + 継続損失 + KL 損失

  4. Imagination Rollout
     - Prior のみで未来をシミュレーション（夢の生成）
     - 実環境なしで数千ステップの経験を生成可能

  5. Actor-Critic in Imagination
     - Actor: 想像上の累積リターンを最大化
     - Critic: lambda-return で価値推定
     - 世界モデルを固定して方策を更新

  6. DreamerV3 の技術的革新
     - Symlog, カテゴリカル潜在, Percentile scaling
     - 150+ タスクで固定ハイパーパラメータ
\"\"\"

ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#EBF5FB', alpha=0.8))

plt.tight_layout()
plt.show()"""))

# ============================================================
# Cell 44: Common Errors
# ============================================================
cells.append(md("""### 9.2 よくあるエラー 3選

#### エラー 1: KL ダイバージェンスが発散する

```python
# NG: log_std の範囲を制限していない
log_std = self.net(h)  # 値が大きくなりすぎる可能性
std = torch.exp(log_std)  # exp(100) = inf!

# OK: クランプで安全な範囲に制限
log_std = torch.clamp(self.net(h), min=-5.0, max=2.0)
std = torch.exp(log_std)
```

**原因**: ネットワークの出力が制約なしだと、標準偏差が極端な値になり KL が発散する。

---

#### エラー 2: 想像上のロールアウトで勾配が切れている

```python
# NG: 世界モデルとActorの両方を torch.no_grad() で囲んでしまう
with torch.no_grad():
    imagined = imagine_rollout(world_model, actor, h, z, horizon=15)
# Actor の勾配が計算できない!

# OK: 世界モデルは .eval() + detach()、Actor は勾配を通す
world_model.eval()
h_start = h.detach()
z_start = z.detach()
imagined = imagine_rollout(world_model, actor, h_start, z_start, horizon=15)
```

**原因**: Actor の更新には想像軌道を通じた勾配が必要。世界モデルの勾配だけを止める。

---

#### エラー 3: 世界モデルの学習と方策の学習を同時に行う

```python
# NG: 全パラメータを1つのオプティマイザで更新
optimizer = Adam(list(world_model.parameters()) + list(actor.parameters()))

# OK: 別々のオプティマイザで、別々のフェーズで更新
wm_optimizer = Adam(world_model.parameters(), lr=3e-4)
actor_optimizer = Adam(actor.parameters(), lr=1e-4)
```

**原因**: 世界モデルと方策は異なる目的関数を持つ。同時に更新すると世界モデルが方策に都合の良いように歪む。"""))

# ============================================================
# Cell 45: Quiz Header
# ============================================================
cells.append(md("""### 9.3 確認テスト（5問）

以下の問題で理解度を確認しましょう。"""))

# ============================================================
# Cell 46: Quiz
# ============================================================
cells.append(code("""# 確認テスト
quiz = [
    {
        "question": "Q1: RSSM の確定的状態 h_t と確率的状態 z_t は、それぞれ何を表現するか？",
        "choices": [
            "A) h_t = 未来の予測、z_t = 過去の記憶",
            "B) h_t = 過去の履歴の圧縮、z_t = 環境の不確実性",
            "C) h_t = 行動の記憶、z_t = 報酬の予測",
            "D) h_t = 観測の記憶、z_t = アクションの分布"
        ],
        "answer": "B",
        "explanation": "h_t は GRU で過去の履歴を圧縮的に保持し、"
                      "z_t はガウス/カテゴリカル分布で環境の不確実性を表現します。"
    },
    {
        "question": "Q2: DreamerV3 の3フェーズ学習の正しい順序は？",
        "choices": [
            "A) データ収集 → Actor-Critic → 世界モデル",
            "B) 世界モデル → データ収集 → Actor-Critic",
            "C) 世界モデル → 想像（Actor-Critic）→ データ収集",
            "D) Actor-Critic → 世界モデル → データ収集"
        ],
        "answer": "C",
        "explanation": "まず実データで世界モデルを更新し、次に想像上で Actor-Critic を訓練し、"
                      "最後に改善された方策でデータを収集します。"
    },
    {
        "question": "Q3: Imagination Rollout で Prior を使う理由は？",
        "choices": [
            "A) Posterior より精度が高いから",
            "B) 計算が軽いから",
            "C) 想像時には観測がないため、観測なしで予測する Prior を使う必要がある",
            "D) Prior の方が KL が小さいから"
        ],
        "answer": "C",
        "explanation": "Imagination では実環境の観測が得られないため、"
                      "h_t のみから z_t を予測する Prior p(z|h) を使用します。"
    },
    {
        "question": "Q4: Symlog 変換の主な目的は？",
        "choices": [
            "A) 計算速度の向上",
            "B) 異なるスケールの報酬を統一的に扱うこと",
            "C) メモリ使用量の削減",
            "D) 探索性能の向上"
        ],
        "answer": "B",
        "explanation": "Symlog は sign(x)*ln(|x|+1) で、小さな値も大きな値も均等に扱えます。"
                      "これにより報酬スケールが異なるタスクでも同じ HP で動作します。"
    },
    {
        "question": "Q5: DreamerV3 がモデルフリー RL より優れている主な点は？",
        "choices": [
            "A) 必ずより高い最終性能を達成する",
            "B) サンプル効率が高い（少ない実環境データで学習できる）",
            "C) ハイパーパラメータのチューニングが不要",
            "D) 計算コストが低い"
        ],
        "answer": "B",
        "explanation": "世界モデルの中で数千ステップ想像できるため、"
                      "実環境での試行回数を大幅に削減できます（サンプル効率の向上）。"
    }
]

print("=" * 60)
print("  確認テスト: DreamerV3 世界モデル")
print("=" * 60)

for i, q in enumerate(quiz):
    print(f"\\n{q['question']}")
    for choice in q['choices']:
        print(f"  {choice}")

print("\\n" + "=" * 60)
print("  解答")
print("=" * 60)

for i, q in enumerate(quiz):
    print(f"\\nQ{i+1}: 正解は {q['answer']}")
    print(f"   {q['explanation']}")"""))

# ============================================================
# Cell 47: Model summary
# ============================================================
cells.append(code("""# 最終的なモデルサイズの確認
print("=" * 60)
print("  SimpleDreamer モデルサイズまとめ")
print("=" * 60)

components = [
    ("World Model (RSSM)", dreamer.world_model.rssm),
    ("World Model (Encoder)", dreamer.world_model.encoder),
    ("World Model (Decoder)", dreamer.world_model.decoder),
    ("World Model (Reward)", dreamer.world_model.reward_pred),
    ("World Model (Continue)", dreamer.world_model.continue_pred),
    ("Actor", dreamer.actor),
    ("Critic", dreamer.critic),
]

total = 0
for name, module in components:
    n_params = sum(p.numel() for p in module.parameters())
    total += n_params
    print(f"  {name:30s}: {n_params:>8,} パラメータ")

print(f"  {'':30s}  {'':>8s}")
print(f"  {'合計':30s}: {total:>8,} パラメータ")
print()
print("注: これは教育用の小規模モデルです。")
print("   実際の DreamerV3 は数百万パラメータ規模です。")"""))

# ============================================================
# Cell 48: Next Steps
# ============================================================
cells.append(md("""---

### 次のステップ

このノートブックで DreamerV3 の核心的なアイデアを実装しました。
さらに深く学びたい場合は、以下に挑戦してみてください:

1. **画像観測への拡張**: Encoder/Decoder を CNN に置き換え、画像ベースのタスクに適用
2. **カテゴリカル潜在状態**: ガウス分布をカテゴリカル分布に置き換え（Gumbel-Softmax）
3. **Symlog の統合**: 報酬予測と Critic に Symlog 変換を適用
4. **より複雑な環境**: CartPole、MountainCar、Atari などで検証
5. **Plan2Explore**: 報酬なしの探索（内発的動機付け）の実装

### 参考文献

- Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3, 2023)
- Hafner et al., "Mastering Atari with Discrete World Models" (DreamerV2, 2021)
- Hafner et al., "Dream to Control" (DreamerV1, 2020)
- Ha & Schmidhuber, "World Models" (2018)

---

*第143章 終わり*"""))

# Save part 3
with open('/Users/ikmx/source/personal/machine-learning-playground/notebooks/world-models/_cells_part3.json', 'w') as f:
    json.dump(cells, f, ensure_ascii=False)

print(f"Part 3: {len(cells)} cells saved")
