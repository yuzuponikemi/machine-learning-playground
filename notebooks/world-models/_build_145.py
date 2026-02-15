#!/usr/bin/env python3
"""Build notebook 145: Grid World Agent with World Model planning."""
import json, os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n"), "id": None}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source.split("\n"),
            "execution_count": None, "outputs": [], "id": None}

cells = []

# =====================================================================
# Cell 0: Title
# =====================================================================
cells.append(md(
r"""# 第145章: グリッドワールドの知能体 — 学習した世界モデルで計画する (Capstone)

## Grid World Agent: Planning with a Learned World Model

---

### このノートブックの位置づけ

**世界モデル（World Models）シリーズ** の Capstone として、
エージェントが **観測画像からの表現学習 → 遷移・報酬モデルの訓練 → MPC 計画** を
7×7 グリッドワールドで実践します。鍵を拾い、扉を開け、ゴールに到達する一連の
タスクを通じて、Phase 7 の概念を統合します。

### 学習目標

1. **KeyDoorGridWorld** 環境を実装し、RGB 観測を生成できる
2. **ObservationEncoder / Decoder** で画像 ↔ 潜在表現を変換できる
3. **TransitionModel / RewardModel** で世界のダイナミクスを学習できる
4. **Model Predictive Control (MPC)** で潜在空間上の計画を行える
5. **WorldModelAgent** の 4 フェーズパイプラインを統合実行できる
6. **Q-Learning ベースライン** と比較し、世界モデルの利点を定量評価できる

### 前提知識

- Notebook 142: モデルベース RL の基礎（Dyna-Q）
- Notebook 143: DreamerV3 ― 潜在空間での想像
- Notebook 144: Genie ― 潜在行動発見
- Notebook 75: Training loop の設計
- Notebook 112: Adam オプティマイザ

### 難易度: ★★★★★ | 所要時間: 300〜360分

---"""
))

# =====================================================================
# Cell 1: TOC
# =====================================================================
cells.append(md(
r"""## 目次

1. [環境セットアップ](#section1)
2. [KeyDoorGridWorld の実装](#section2)
3. [環境の可視化](#section3)
4. [ランダム探索によるデータ収集](#section4)
5. [世界モデルの構成要素](#section5)
6. [世界モデルの訓練](#section6)
7. [MPC: 潜在空間での計画](#section7)
8. [WorldModelAgent パイプライン](#section8)
9. [Q-Learning ベースライン](#section9)
10. [モデル精度 vs 計画成功率の分析](#section10)
11. [まとめ・よくあるエラー・確認クイズ](#section11)"""
))

# =====================================================================
# Cell 2: Setup
# =====================================================================
cells.append(code(
r"""# ============================================================
# 環境設定
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, deque
import warnings, copy, time
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

np.random.seed(42)

print("環境セットアップ完了")
print(f"NumPy version: {np.__version__}")"""
))

# =====================================================================
# Cell 3: Section 1 header
# =====================================================================
cells.append(md(
r"""<a id="section1"></a>

---

## 1. 環境セットアップ

このノートブックは **NumPy のみ** で動作します。CNN・MLP・Adam をすべてスクラッチで
実装し、GPU 不要で世界モデルの全パイプラインを体験します。

### NumPy ベースのミニフレームワーク

学習に必要な最小限のユーティリティを先に定義します。"""
))

# =====================================================================
# Cell 4: Mini-framework (activations, layers, Adam)
# =====================================================================
cells.append(code(
r"""# ============================================================
# NumPy ミニフレームワーク: 活性化関数・線形層・Adam
# ============================================================

def relu(x):
    # ReLU 活性化関数
    return np.maximum(0, x)

def relu_grad(x):
    # ReLU の勾配
    return (x > 0).astype(np.float64)

def sigmoid(x):
    # シグモイド関数（数値安定版）
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x, axis=-1):
    # ソフトマックス関数
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def mse_loss(pred, target):
    # 平均二乗誤差
    return np.mean((pred - target) ** 2)

def he_init(fan_in, fan_out):
    # He 初期化
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)


class Linear:
    # 全結合層
    def __init__(self, in_dim, out_dim):
        self.W = he_init(in_dim, out_dim)
        self.b = np.zeros(out_dim)
        self.x = None  # forward 時の入力を保存

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        self.grad_W = self.x.T @ grad_out
        self.grad_b = np.sum(grad_out, axis=0)
        return grad_out @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.grad_W, 'W'), (self.b, self.grad_b, 'b')]


class Adam:
    # Adam オプティマイザ
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.states = {}

    def step(self, layers):
        self.t += 1
        for layer in layers:
            for param, grad, name in layer.params_and_grads():
                key = id(param)
                if key not in self.states:
                    self.states[key] = (np.zeros_like(param), np.zeros_like(param))
                m, v = self.states[key]
                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * grad ** 2
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                self.states[key] = (m, v)


print("ミニフレームワーク定義完了:")
print("  - relu, sigmoid, softmax, mse_loss")
print("  - Linear (全結合層)")
print("  - Adam オプティマイザ")"""
))

# =====================================================================
# Cell 5: Section 2 header
# =====================================================================
cells.append(md(
r"""<a id="section2"></a>

---

## 2. KeyDoorGridWorld の実装

### 2.1 環境の設計

7×7 のグリッドワールドを実装します。エージェントは **鍵を拾い → 扉を開け → ゴール** に
到達する必要があります。

```
┌───┬───┬───┬───┬───┬───┬───┐
│ A │   │   │ W │   │   │   │  A: エージェント (0,0)
├───┼───┼───┼───┼───┼───┼───┤
│   │ W │   │ W │   │ W │   │  W: 壁
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │  K: 鍵 (2,5)
├───┼───┼───┼───┼───┼───┼───┤
│ W │   │ W │   │ W │ K │   │  D: 扉 (4,3) ← 鍵が必要
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │ D │   │   │   │  G: ゴール (6,6)
├───┼───┼───┼───┼───┼───┼───┤
│   │ W │   │   │   │ W │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │ W │   │   │ G │
└───┴───┴───┴───┴───┴───┴───┘
```

- **行動**: 上(0), 右(1), 下(2), 左(3)
- **報酬**: ゴール +1.0, 鍵取得 +0.5, ステップ -0.01
- **観測**: 7×7×3 の RGB 画像（各セルを1ピクセルとして色分け）"""
))

# =====================================================================
# Cell 6: KeyDoorGridWorld class
# =====================================================================
cells.append(code(
r"""class KeyDoorGridWorld:
    # 
    7x7 KeyDoor GridWorld 環境

    状態: (row, col, has_key) のタプル
    行動: 0=上, 1=右, 2=下, 3=左
    報酬: +1.0(ゴール), +0.5(鍵取得), -0.01(ステップ)
    観測: 7x7x3 RGB 画像 (float32, 0-1)
    

    # セルタイプ定数
    EMPTY = 0
    WALL = 1
    KEY = 2
    DOOR = 3
    GOAL = 4

    # 行動
    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    ACTION_NAMES = ['上', '右', '下', '左']

    # 色定義 (RGB, 0-1)
    COLORS = {
        'empty':    np.array([0.95, 0.95, 0.95]),
        'wall':     np.array([0.20, 0.20, 0.20]),
        'key':      np.array([1.00, 0.85, 0.00]),
        'door':     np.array([0.60, 0.40, 0.20]),
        'door_open':np.array([0.85, 0.75, 0.60]),
        'goal':     np.array([0.20, 0.80, 0.20]),
        'agent':    np.array([0.90, 0.20, 0.20]),
    }

    def __init__(self):
        self.grid_size = 7
        self.n_actions = 4
        self.max_steps = 100

        # グリッド配置
        self.walls = {
            (0, 3), (1, 1), (1, 3), (1, 5),
            (3, 0), (3, 2), (3, 4),
            (5, 1), (5, 5), (6, 3),
        }
        self.key_pos = (3, 5)
        self.door_pos = (4, 3)
        self.goal_pos = (6, 6)
        self.start_pos = (0, 0)

        self.reset()

    def reset(self):
        # 環境をリセット
        self.agent_pos = self.start_pos
        self.has_key = False
        self.door_open = False
        self.steps = 0
        self.done = False
        return self.render()

    def step(self, action):
        # 行動を実行し (obs, reward, done, info) を返す
        if self.done:
            return self.render(), 0.0, True, {}

        self.steps += 1
        reward = -0.01  # ステップペナルティ

        dr, dc = self.ACTIONS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        # 範囲チェック
        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
            new_pos = (nr, nc)
            # 壁チェック
            if new_pos not in self.walls:
                # 扉チェック（鍵がなければ通れない）
                if new_pos == self.door_pos and not self.door_open:
                    if self.has_key:
                        self.door_open = True
                        self.agent_pos = new_pos
                    # 鍵がなければ移動できない
                else:
                    self.agent_pos = new_pos

        # 鍵の取得
        if self.agent_pos == self.key_pos and not self.has_key:
            self.has_key = True
            reward += 0.5

        # ゴール判定
        if self.agent_pos == self.goal_pos:
            reward += 1.0
            self.done = True

        # 最大ステップ
        if self.steps >= self.max_steps:
            self.done = True

        info = {'has_key': self.has_key, 'door_open': self.door_open}
        return self.render(), reward, self.done, info

    def render(self):
        # 7x7x3 RGB 観測を返す
        img = np.zeros((self.grid_size, self.grid_size, 3))

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                pos = (r, c)
                if pos in self.walls:
                    img[r, c] = self.COLORS['wall']
                elif pos == self.key_pos and not self.has_key:
                    img[r, c] = self.COLORS['key']
                elif pos == self.door_pos:
                    if self.door_open:
                        img[r, c] = self.COLORS['door_open']
                    else:
                        img[r, c] = self.COLORS['door']
                elif pos == self.goal_pos:
                    img[r, c] = self.COLORS['goal']
                else:
                    img[r, c] = self.COLORS['empty']

        # エージェントを上書き
        img[self.agent_pos[0], self.agent_pos[1]] = self.COLORS['agent']
        return img.astype(np.float32)

    def get_state(self):
        # 内部状態をタプルで返す
        return (self.agent_pos[0], self.agent_pos[1], int(self.has_key))

    def get_flat_state_index(self):
        # 状態を整数インデックスに変換（Q-Learning 用）
        r, c = self.agent_pos
        k = int(self.has_key)
        return r * self.grid_size * 2 + c * 2 + k

    @property
    def n_flat_states(self):
        return self.grid_size * self.grid_size * 2


# テスト
env = KeyDoorGridWorld()
obs = env.reset()
print(f"観測の形状: {obs.shape}")
print(f"初期位置: {env.agent_pos}")
print(f"鍵所持: {env.has_key}")
print(f"行動空間: {env.n_actions} ({env.ACTION_NAMES})")
print(f"状態空間 (flat): {env.n_flat_states}")"""
))

# =====================================================================
# Cell 7: Section 3 header
# =====================================================================
cells.append(md(
r"""<a id="section3"></a>

---

## 3. 環境の可視化

### 3.1 グリッドの表示

環境の初期状態と、鍵取得後の状態を並べて表示します。"""
))

# =====================================================================
# Cell 8: Visualization
# =====================================================================
cells.append(code(
r"""def render_large(env, ax, title=''):
    # 環境を拡大して可視化
    obs = env.render()
    n = env.grid_size

    for r in range(n):
        for c in range(n):
            color = obs[r, c]
            rect = plt.Rectangle((c, n - 1 - r), 1, 1,
                                  facecolor=color, edgecolor='#BDBDBD', linewidth=1.5)
            ax.add_patch(rect)

            pos = (r, c)
            label = ''
            if pos in env.walls:
                label = 'W'
            elif pos == env.key_pos and not env.has_key:
                label = 'K'
            elif pos == env.door_pos:
                label = 'D' if not env.door_open else 'd'
            elif pos == env.goal_pos:
                label = 'G'
            if pos == env.agent_pos:
                label = 'A'

            if label:
                fc = 'white' if label in ('W', 'A') else 'black'
                ax.text(c + 0.5, n - 1 - r + 0.5, label,
                        ha='center', va='center', fontsize=13,
                        fontweight='bold', color=fc)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n))
    ax.set_yticklabels(range(n - 1, -1, -1))


# 初期状態と鍵取得後の比較
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

env1 = KeyDoorGridWorld()
env1.reset()
render_large(env1, axes[0], '初期状態')

env2 = KeyDoorGridWorld()
env2.reset()
# 手動で鍵取得状態にする
env2.agent_pos = (3, 5)
env2.has_key = True
render_large(env2, axes[1], '鍵取得後')

legend_elements = [
    mpatches.Patch(facecolor=[0.90, 0.20, 0.20], label='A: エージェント'),
    mpatches.Patch(facecolor=[1.00, 0.85, 0.00], label='K: 鍵'),
    mpatches.Patch(facecolor=[0.60, 0.40, 0.20], label='D: 扉（閉）'),
    mpatches.Patch(facecolor=[0.20, 0.80, 0.20], label='G: ゴール'),
    mpatches.Patch(facecolor=[0.20, 0.20, 0.20], label='W: 壁'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5,
           fontsize=10, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.show()

print("【タスクの流れ】")
print("  1. エージェント(A) が鍵(K) まで移動して取得 (+0.5)")
print("  2. 扉(D) に到達して開錠")
print("  3. ゴール(G) に到達 (+1.0)")"""
))

# =====================================================================
# Cell 9: Test episode
# =====================================================================
cells.append(code(
r"""# テストエピソード: ランダム行動
env = KeyDoorGridWorld()
obs = env.reset()
total_reward = 0
trajectory = [env.agent_pos]

for step in range(20):
    action = np.random.randint(4)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    trajectory.append(env.agent_pos)
    if done:
        break

print(f"ランダム行動 {step+1} ステップ")
print(f"  累積報酬: {total_reward:.3f}")
print(f"  鍵取得: {info['has_key']}, 扉開: {info['door_open']}")
print(f"  軌跡（先頭10）: {trajectory[:10]}")"""
))

# =====================================================================
# Cell 10: Section 4 header
# =====================================================================
cells.append(md(
r"""<a id="section4"></a>

---

## 4. ランダム探索によるデータ収集

### 4.1 トランジションバッファ

世界モデルの訓練には、環境との相互作用データが必要です。
まずランダム方策で 500 件の遷移 $(o_t, a_t, r_t, o_{t+1})$ を収集します。

このデータが世界モデル学習の **Phase 1: データ収集** に対応します。"""
))

# =====================================================================
# Cell 11: Data collection
# =====================================================================
cells.append(code(
r"""class TransitionBuffer:
    # 遷移データのバッファ

    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.dones = []

    def add(self, obs, action, reward, next_obs, done):
        if len(self.obs) >= self.capacity:
            self.obs.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_obs.pop(0)
            self.dones.pop(0)
        self.obs.append(obs.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_obs.append(next_obs.copy())
        self.dones.append(done)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.obs), size=batch_size, replace=False)
        return (
            np.array([self.obs[i] for i in indices]),
            np.array([self.actions[i] for i in indices]),
            np.array([self.rewards[i] for i in indices]),
            np.array([self.next_obs[i] for i in indices]),
            np.array([self.dones[i] for i in indices]),
        )

    def __len__(self):
        return len(self.obs)


def collect_random_data(env, buffer, n_transitions=500):
    # ランダム方策で遷移データを収集
    collected = 0
    episodes = 0

    while collected < n_transitions:
        obs = env.reset()
        episodes += 1
        while collected < n_transitions:
            action = np.random.randint(env.n_actions)
            next_obs, reward, done, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, done)
            collected += 1
            obs = next_obs
            if done:
                break

    return episodes


# データ収集
np.random.seed(42)
env = KeyDoorGridWorld()
buffer = TransitionBuffer(capacity=5000)
n_episodes = collect_random_data(env, buffer, n_transitions=500)

print(f"データ収集完了:")
print(f"  遷移数: {len(buffer)}")
print(f"  エピソード数: {n_episodes}")

# 統計
rewards_arr = np.array(buffer.rewards)
print(f"  報酬統計: mean={rewards_arr.mean():.4f}, "
      f"min={rewards_arr.min():.2f}, max={rewards_arr.max():.2f}")
print(f"  正の報酬の割合: {(rewards_arr > 0).mean():.3f}")"""
))

# =====================================================================
# Cell 12: Data distribution visualization
# =====================================================================
cells.append(code(
r"""# 収集データの分布を可視化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (1) 報酬分布
axes[0].hist(buffer.rewards, bins=30, color='#2196F3', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('報酬', fontsize=11)
axes[0].set_ylabel('頻度', fontsize=11)
axes[0].set_title('報酬分布（ランダム探索）', fontsize=13)
axes[0].grid(True, alpha=0.3)

# (2) 行動分布
action_counts = np.bincount(buffer.actions, minlength=4)
axes[1].bar(range(4), action_counts, color=['#E91E63', '#4CAF50', '#FF9800', '#9C27B0'],
            alpha=0.8, edgecolor='white')
axes[1].set_xticks(range(4))
axes[1].set_xticklabels(KeyDoorGridWorld.ACTION_NAMES)
axes[1].set_ylabel('頻度', fontsize=11)
axes[1].set_title('行動分布', fontsize=13)
axes[1].grid(True, alpha=0.3)

# (3) 訪問ヒートマップ
visit_map = np.zeros((7, 7))
for obs in buffer.obs:
    # エージェント位置を観測から復元（赤色のピクセルを検出）
    for r in range(7):
        for c in range(7):
            if obs[r, c, 0] > 0.8 and obs[r, c, 1] < 0.3:
                visit_map[r, c] += 1
im = axes[2].imshow(visit_map, cmap='YlOrRd', interpolation='nearest')
axes[2].set_title('訪問頻度マップ', fontsize=13)
plt.colorbar(im, ax=axes[2], shrink=0.8)

plt.tight_layout()
plt.show()

print("【観察】")
print("  ランダム方策ではスタート付近の訪問が多く、遠方は少ない")
print("  鍵やゴールに到達する確率は低い")"""
))

# =====================================================================
# Cell 13: Section 5 header
# =====================================================================
cells.append(md(
r"""<a id="section5"></a>

---

## 5. 世界モデルの構成要素

### 5.1 アーキテクチャ概要

世界モデルは4つのコンポーネントで構成されます:

```
  観測 o_t (7x7x3)
      │
      ▼
 ┌──────────────────┐
 │ ObservationEncoder│  → 潜在ベクトル z_t (64次元)
 └──────┬───────────┘
        │
   ┌────┴────┐
   │         │
   ▼         ▼
┌────────┐ ┌────────────┐
│Transition│ │RewardModel │  → 予測報酬 r̂_t
│Model    │ └────────────┘
│         │
│ z_t,a_t │
│  → ẑ_{t+1}│
└────┬────┘
     │
     ▼
┌──────────────────┐
│ObservationDecoder │  → 再構成画像 ô_{t+1}
└──────────────────┘
```

### 5.2 簡略化

CPU で効率的に動作させるため、**CNN の代わりに flatten + MLP** で
エンコーダ/デコーダを実装します。7×7×3 = 147 次元なので MLP で十分です。"""
))

# =====================================================================
# Cell 14: ObservationEncoder
# =====================================================================
cells.append(code(
r"""class ObservationEncoder:
    # 
    観測エンコーダ: 7x7x3 画像 → 64次元潜在ベクトル

    構造: flatten(147) → Linear(147, 128) → ReLU
                       → Linear(128, 64) → ReLU
                       → Linear(64, 64)
    

    def __init__(self, obs_dim=147, latent_dim=64):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.fc1 = Linear(obs_dim, 128)
        self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64, latent_dim)

        # 中間値の保存
        self.h1 = None
        self.h2 = None

    def forward(self, obs):
        # 
        Args:
            obs: (batch, 7, 7, 3) or (batch, 147)
        Returns:
            z: (batch, 64) 潜在ベクトル
        
        if obs.ndim > 2:
            x = obs.reshape(obs.shape[0], -1)
        else:
            x = obs
        self.h1 = self.fc1.forward(x)
        h1_act = relu(self.h1)
        self.h2 = self.fc2.forward(h1_act)
        h2_act = relu(self.h2)
        z = self.fc3.forward(h2_act)
        return z

    def backward(self, grad_z):
        # 逆伝播
        grad = self.fc3.backward(grad_z)
        grad = grad * relu_grad(self.h2)
        grad = self.fc2.backward(grad)
        grad = grad * relu_grad(self.h1)
        grad = self.fc1.backward(grad)
        return grad

    def get_layers(self):
        return [self.fc1, self.fc2, self.fc3]


# テスト
encoder = ObservationEncoder()
test_obs = np.random.rand(4, 7, 7, 3).astype(np.float32)
z = encoder.forward(test_obs)
print(f"ObservationEncoder:")
print(f"  入力: {test_obs.shape} → 出力: {z.shape}")
print(f"  パラメータ数: {sum(l.W.size + l.b.size for l in encoder.get_layers())}")"""
))

# =====================================================================
# Cell 15: TransitionModel
# =====================================================================
cells.append(code(
r"""class TransitionModel:
    # 
    遷移モデル: (z_t, a_t) → z_{t+1} の予測

    行動は one-hot (4次元) に変換して潜在ベクトルと結合。
    構造: Linear(64+4, 128) → ReLU → Linear(128, 64)
    

    def __init__(self, latent_dim=64, n_actions=4):
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        self.fc1 = Linear(latent_dim + n_actions, 128)
        self.fc2 = Linear(128, latent_dim)
        self.h1 = None

    def forward(self, z, action):
        # 
        Args:
            z: (batch, 64) 現在の潜在ベクトル
            action: (batch,) 行動インデックス
        Returns:
            z_next: (batch, 64) 予測された次の潜在ベクトル
        
        # 行動を one-hot に変換
        batch_size = z.shape[0]
        a_onehot = np.zeros((batch_size, self.n_actions))
        a_onehot[np.arange(batch_size), action.astype(int)] = 1.0

        # 結合
        x = np.concatenate([z, a_onehot], axis=1)
        self.h1 = self.fc1.forward(x)
        h1_act = relu(self.h1)
        z_next = self.fc2.forward(h1_act)
        return z_next

    def backward(self, grad_z_next):
        # 逆伝播
        grad = self.fc2.backward(grad_z_next)
        grad = grad * relu_grad(self.h1)
        grad = self.fc1.backward(grad)
        return grad

    def get_layers(self):
        return [self.fc1, self.fc2]


# テスト
trans_model = TransitionModel()
z_test = np.random.randn(4, 64)
a_test = np.array([0, 1, 2, 3])
z_next = trans_model.forward(z_test, a_test)
print(f"TransitionModel:")
print(f"  入力: z{z_test.shape} + action{a_test.shape}")
print(f"  出力: z_next{z_next.shape}")"""
))

# =====================================================================
# Cell 16: RewardModel
# =====================================================================
cells.append(code(
r"""class RewardModel:
    # 
    報酬モデル: (z_t, a_t) → r̂_t の予測

    構造: Linear(64+4, 64) → ReLU → Linear(64, 1)
    

    def __init__(self, latent_dim=64, n_actions=4):
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        self.fc1 = Linear(latent_dim + n_actions, 64)
        self.fc2 = Linear(64, 1)
        self.h1 = None

    def forward(self, z, action):
        batch_size = z.shape[0]
        a_onehot = np.zeros((batch_size, self.n_actions))
        a_onehot[np.arange(batch_size), action.astype(int)] = 1.0

        x = np.concatenate([z, a_onehot], axis=1)
        self.h1 = self.fc1.forward(x)
        h1_act = relu(self.h1)
        r_pred = self.fc2.forward(h1_act)
        return r_pred.squeeze(-1)

    def backward(self, grad_r):
        if grad_r.ndim == 1:
            grad_r = grad_r[:, None]
        grad = self.fc2.backward(grad_r)
        grad = grad * relu_grad(self.h1)
        grad = self.fc1.backward(grad)
        return grad

    def get_layers(self):
        return [self.fc1, self.fc2]


# テスト
reward_model = RewardModel()
r_pred = reward_model.forward(z_test, a_test)
print(f"RewardModel:")
print(f"  入力: z{z_test.shape} + action{a_test.shape}")
print(f"  出力: r_pred{r_pred.shape}")"""
))

# =====================================================================
# Cell 17: ObservationDecoder
# =====================================================================
cells.append(code(
r"""class ObservationDecoder:
    # 
    観測デコーダ: 64次元潜在ベクトル → 7x7x3 再構成画像

    構造: Linear(64, 64) → ReLU → Linear(64, 128) → ReLU → Linear(128, 147) → Sigmoid
    

    def __init__(self, latent_dim=64, obs_dim=147):
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.fc1 = Linear(latent_dim, 64)
        self.fc2 = Linear(64, 128)
        self.fc3 = Linear(128, obs_dim)
        self.h1 = None
        self.h2 = None
        self.h3 = None

    def forward(self, z):
        self.h1 = self.fc1.forward(z)
        h1_act = relu(self.h1)
        self.h2 = self.fc2.forward(h1_act)
        h2_act = relu(self.h2)
        self.h3 = self.fc3.forward(h2_act)
        return sigmoid(self.h3)

    def backward(self, grad_out):
        # sigmoid の勾配
        sig_out = sigmoid(self.h3)
        grad = grad_out * sig_out * (1 - sig_out)
        grad = self.fc3.backward(grad)
        grad = grad * relu_grad(self.h2)
        grad = self.fc2.backward(grad)
        grad = grad * relu_grad(self.h1)
        grad = self.fc1.backward(grad)
        return grad

    def get_layers(self):
        return [self.fc1, self.fc2, self.fc3]


# テスト
decoder = ObservationDecoder()
z_test2 = np.random.randn(4, 64)
recon = decoder.forward(z_test2)
print(f"ObservationDecoder:")
print(f"  入力: z{z_test2.shape}")
print(f"  出力: recon{recon.shape}")
print(f"  出力範囲: [{recon.min():.3f}, {recon.max():.3f}]")"""
))

# =====================================================================
# Cell 18: Section 6 header
# =====================================================================
cells.append(md(
r"""<a id="section6"></a>

---

## 6. 世界モデルの訓練

### 6.1 訓練ループ

世界モデル全体を **3つの損失関数** で同時に訓練します:

1. **再構成損失**: $\mathcal{L}_{recon} = \text{MSE}(o_t, \hat{o}_t)$ — エンコーダ + デコーダ
2. **遷移損失**: $\mathcal{L}_{trans} = \text{MSE}(z_{t+1}, \hat{z}_{t+1})$ — 遷移モデル
3. **報酬損失**: $\mathcal{L}_{reward} = \text{MSE}(r_t, \hat{r}_t)$ — 報酬モデル

$$\mathcal{L}_{total} = \mathcal{L}_{recon} + \mathcal{L}_{trans} + \mathcal{L}_{reward}$$

### 6.2 訓練の流れ

```
obs_t  → Encoder → z_t ──┬──→ Decoder → ô_t   (再構成損失)
                          │
obs_t+1 → Encoder → z_t+1│   (ターゲット)
                          │
               z_t, a_t ──┼──→ TransitionModel → ẑ_{t+1} (遷移損失)
                          │
               z_t, a_t ──┴──→ RewardModel → r̂_t        (報酬損失)
```"""
))

# =====================================================================
# Cell 19: Training loop
# =====================================================================
cells.append(code(
r"""def train_world_model(encoder, decoder, trans_model, reward_model,
                       buffer, n_epochs=20, batch_size=32, lr=1e-3):
    # 
    世界モデルの訓練ループ

    Returns:
        losses: dict with 'recon', 'trans', 'reward', 'total' lists
    
    all_layers = (encoder.get_layers() + decoder.get_layers() +
                  trans_model.get_layers() + reward_model.get_layers())
    optimizer = Adam(lr=lr)

    losses = {'recon': [], 'trans': [], 'reward': [], 'total': []}
    n_batches = max(1, len(buffer) // batch_size)

    for epoch in range(n_epochs):
        epoch_losses = {'recon': 0, 'trans': 0, 'reward': 0, 'total': 0}

        for _ in range(n_batches):
            obs_batch, act_batch, rew_batch, next_obs_batch, _ = buffer.sample(batch_size)

            # ---- Forward ----
            # エンコード
            z = encoder.forward(obs_batch)
            z_next_target = encoder.forward(next_obs_batch)

            # デコード（再構成）
            recon = decoder.forward(z)
            obs_flat = obs_batch.reshape(batch_size, -1)

            # 遷移予測
            z_next_pred = trans_model.forward(z, act_batch)

            # 報酬予測
            r_pred = reward_model.forward(z, act_batch)

            # ---- 損失計算 ----
            L_recon = mse_loss(recon, obs_flat)
            L_trans = mse_loss(z_next_pred, z_next_target.copy())
            L_reward = mse_loss(r_pred, rew_batch)
            L_total = L_recon + L_trans + L_reward

            # ---- Backward ----
            # 再構成損失の勾配
            grad_recon = 2.0 * (recon - obs_flat) / obs_flat.size
            grad_z_from_decoder = decoder.backward(grad_recon)

            # 遷移損失の勾配
            grad_z_next_pred = 2.0 * (z_next_pred - z_next_target) / z_next_target.size
            trans_model.backward(grad_z_next_pred)

            # 報酬損失の勾配
            grad_r = 2.0 * (r_pred - rew_batch) / rew_batch.size
            reward_model.backward(grad_r)

            # エンコーダへの勾配（デコーダからの勾配のみ、簡略化）
            encoder.backward(grad_z_from_decoder)

            # ---- パラメータ更新 ----
            optimizer.step(all_layers)

            epoch_losses['recon'] += L_recon
            epoch_losses['trans'] += L_trans
            epoch_losses['reward'] += L_reward
            epoch_losses['total'] += L_total

        for key in epoch_losses:
            epoch_losses[key] /= n_batches
            losses[key].append(epoch_losses[key])

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:>3d} | "
                  f"Total: {epoch_losses['total']:.5f} | "
                  f"Recon: {epoch_losses['recon']:.5f} | "
                  f"Trans: {epoch_losses['trans']:.5f} | "
                  f"Reward: {epoch_losses['reward']:.6f}")

    return losses


# 世界モデルの構築と訓練
print("=" * 60)
print("世界モデルの訓練開始（20 エポック）")
print("=" * 60)

np.random.seed(42)
encoder = ObservationEncoder()
decoder = ObservationDecoder()
trans_model = TransitionModel()
reward_model = RewardModel()

losses = train_world_model(encoder, decoder, trans_model, reward_model,
                            buffer, n_epochs=20, batch_size=32, lr=1e-3)

print("\n訓練完了!")"""
))

# =====================================================================
# Cell 20: Loss curves
# =====================================================================
cells.append(code(
r"""# 損失曲線の可視化
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

titles = ['Total Loss', 'Reconstruction Loss', 'Transition Loss', 'Reward Loss']
keys = ['total', 'recon', 'trans', 'reward']
colors = ['#333333', '#2196F3', '#4CAF50', '#FF9800']

for ax, title, key, color in zip(axes, titles, keys, colors):
    ax.plot(losses[key], 'o-', color=color, linewidth=2, markersize=4)
    ax.set_xlabel('エポック', fontsize=11)
    ax.set_ylabel('損失', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"【最終損失】")
for key in keys:
    print(f"  {key}: {losses[key][-1]:.6f}")"""
))

# =====================================================================
# Cell 21: Reconstruction visualization
# =====================================================================
cells.append(code(
r"""# 再構成品質の確認
obs_sample, _, _, _, _ = buffer.sample(5)
z_sample = encoder.forward(obs_sample)
recon_sample = decoder.forward(z_sample).reshape(-1, 7, 7, 3)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(5):
    axes[0, i].imshow(obs_sample[i], interpolation='nearest')
    axes[0, i].set_title(f'元画像 {i+1}', fontsize=10)
    axes[0, i].axis('off')

    axes[1, i].imshow(np.clip(recon_sample[i], 0, 1), interpolation='nearest')
    axes[1, i].set_title(f'再構成 {i+1}', fontsize=10)
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=12)
axes[1, 0].set_ylabel('Reconstructed', fontsize=12)
plt.suptitle('観測の再構成品質', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 再構成誤差
recon_error = np.mean((obs_sample.reshape(-1, 147) - recon_sample.reshape(-1, 147))**2)
print(f"平均再構成誤差 (MSE): {recon_error:.6f}")"""
))

# =====================================================================
# Cell 22: Section 7 header
# =====================================================================
cells.append(md(
r"""<a id="section7"></a>

---

## 7. MPC: 潜在空間での計画

### 7.1 Model Predictive Control (MPC)

学習した世界モデルを使って、**潜在空間上で計画** を行います。

**アルゴリズム:**

1. 現在の観測 $o_t$ をエンコードして $z_t$ を得る
2. $K=50$ 個のランダム行動列 $(a_0, a_1, \ldots, a_{H-1})$ をサンプル（$H=5$）
3. 各行動列について遷移モデルでロールアウトし、報酬モデルで累積報酬を予測
4. 最大累積報酬の行動列を選び、**最初の行動** $a_0$ を実行

```
z_t → [a0, a1, ..., a4] → TransitionModel → [ẑ1, ẑ2, ..., ẑ5]
                           RewardModel     → [r̂0, r̂1, ..., r̂4]
                                             sum(r̂) = score
                                             最大 score の a0 を選択
```

これは DreamerV3 の想像ロールアウト（Notebook 143）の簡略版です。"""
))

# =====================================================================
# Cell 23: MPC implementation
# =====================================================================
cells.append(code(
r"""class ModelPredictiveControl:
    # 
    MPC: 潜在空間での計画

    ランダムシューティング法:
      K 個のランダム行動列をサンプルし、
      学習済み世界モデルでロールアウトして最良の行動列を選ぶ。
    

    def __init__(self, encoder, trans_model, reward_model,
                 n_actions=4, K=50, horizon=5):
        # 
        Args:
            encoder: ObservationEncoder
            trans_model: TransitionModel
            reward_model: RewardModel
            n_actions: 行動数
            K: サンプルする行動列の数
            horizon: 計画のホライズン（何ステップ先まで見るか）
        
        self.encoder = encoder
        self.trans_model = trans_model
        self.reward_model = reward_model
        self.n_actions = n_actions
        self.K = K
        self.horizon = horizon

    def plan(self, obs):
        # 
        現在の観測から最良の行動を計画する

        Args:
            obs: (7, 7, 3) 現在の観測
        Returns:
            best_action: int 最良の最初の行動
            best_score: float 最良の累積報酬予測
        
        # 現在の観測をエンコード
        obs_batch = obs[np.newaxis, ...]  # (1, 7, 7, 3)
        z_current = self.encoder.forward(obs_batch)  # (1, 64)

        # K 個の z_current を複製
        z_start = np.tile(z_current, (self.K, 1))  # (K, 64)

        # K 個のランダム行動列をサンプル
        action_seqs = np.random.randint(0, self.n_actions,
                                         size=(self.K, self.horizon))

        # 各行動列をロールアウトして累積報酬を計算
        total_rewards = np.zeros(self.K)
        z = z_start.copy()

        for t in range(self.horizon):
            actions = action_seqs[:, t]
            # 報酬予測
            r_pred = self.reward_model.forward(z, actions)
            total_rewards += r_pred
            # 遷移予測
            z = self.trans_model.forward(z, actions)

        # 最良の行動列を選択
        best_idx = np.argmax(total_rewards)
        best_action = action_seqs[best_idx, 0]
        best_score = total_rewards[best_idx]

        return int(best_action), float(best_score)


# MPC のテスト
mpc = ModelPredictiveControl(encoder, trans_model, reward_model,
                              n_actions=4, K=50, horizon=5)

env_test = KeyDoorGridWorld()
obs_test = env_test.reset()
action, score = mpc.plan(obs_test)
print(f"MPC テスト:")
print(f"  選択された行動: {action} ({KeyDoorGridWorld.ACTION_NAMES[action]})")
print(f"  予測累積報酬: {score:.4f}")"""
))

# =====================================================================
# Cell 24: MPC visualization
# =====================================================================
cells.append(code(
r"""# MPC の計画を可視化
def visualize_mpc_planning(env, mpc, n_steps=30):
    # MPC で計画した軌跡を可視化
    obs = env.reset()
    trajectory = [env.agent_pos]
    rewards = []
    actions_taken = []

    for step in range(n_steps):
        action, score = mpc.plan(obs)
        obs, reward, done, info = env.step(action)
        trajectory.append(env.agent_pos)
        rewards.append(reward)
        actions_taken.append(action)
        if done:
            break

    return trajectory, rewards, actions_taken, info

np.random.seed(123)
env_mpc = KeyDoorGridWorld()
traj, rews, acts, final_info = visualize_mpc_planning(env_mpc, mpc, n_steps=50)

print(f"MPC エージェントの結果:")
print(f"  ステップ数: {len(rews)}")
print(f"  累積報酬: {sum(rews):.3f}")
print(f"  鍵取得: {final_info['has_key']}")
print(f"  扉開放: {final_info['door_open']}")
print(f"  最終位置: {traj[-1]}")

# 軌跡の可視化
fig, ax = plt.subplots(figsize=(7, 7))
render_large(env_mpc, ax, f'MPC の軌跡（{len(rews)} ステップ）')

# 軌跡を線で描画
n = 7
for i in range(len(traj) - 1):
    r1, c1 = traj[i]
    r2, c2 = traj[i + 1]
    ax.plot([c1 + 0.5, c2 + 0.5],
            [n - 1 - r1 + 0.5, n - 1 - r2 + 0.5],
            'b-', alpha=0.3 + 0.7 * i / len(traj), linewidth=2)

plt.tight_layout()
plt.show()"""
))

# =====================================================================
# Cell 25: Section 8 header
# =====================================================================
cells.append(md(
r"""<a id="section8"></a>

---

## 8. WorldModelAgent パイプライン

### 8.1 4 フェーズの統合

WorldModelAgent は以下の4フェーズを繰り返します:

| フェーズ | 内容 | 対応 |
|---------|------|------|
| Phase 1 | ランダム探索でデータ収集 | `collect_random_data()` |
| Phase 2 | 世界モデルの訓練 | `train_world_model()` |
| Phase 3 | MPC で計画 | `mpc.plan()` |
| Phase 4 | 計画した行動を実環境で実行 | `env.step()` |

```
┌──────────────┐      ┌──────────────┐
│ Phase 1:      │──→  │ Phase 2:      │
│ ランダム探索   │      │ モデル訓練    │
└──────────────┘      └──────┬───────┘
       ▲                      │
       │                      ▼
┌──────┴───────┐      ┌──────────────┐
│ Phase 4:      │←──  │ Phase 3:      │
│ 実環境で実行   │      │ MPC 計画     │
└──────────────┘      └──────────────┘
```

### 8.2 イテレーション

各イテレーションで Phase 3-4 を実行し、新しいデータを Phase 1 のバッファに追加します。
一定回数ごとに Phase 2 でモデルを再訓練し、探索能力を向上させます。"""
))

# =====================================================================
# Cell 26: WorldModelAgent
# =====================================================================
cells.append(code(
r"""class WorldModelAgent:
    # 
    世界モデルエージェント: 4 フェーズパイプライン

    Phase 1: ランダム探索（初期データ収集）
    Phase 2: 世界モデル訓練
    Phase 3: MPC 計画
    Phase 4: 実環境で行動実行 → バッファに追加
    

    def __init__(self, env, initial_collect=200, retrain_interval=100,
                 retrain_epochs=10, K=50, horizon=5, lr=1e-3):
        self.env = env
        self.initial_collect = initial_collect
        self.retrain_interval = retrain_interval
        self.retrain_epochs = retrain_epochs
        self.K = K
        self.horizon = horizon
        self.lr = lr

        # バッファ
        self.buffer = TransitionBuffer(capacity=5000)

        # 世界モデル
        self.encoder = ObservationEncoder()
        self.decoder = ObservationDecoder()
        self.trans_model = TransitionModel()
        self.reward_model = RewardModel()

        # MPC
        self.mpc = ModelPredictiveControl(
            self.encoder, self.trans_model, self.reward_model,
            n_actions=env.n_actions, K=K, horizon=horizon
        )

    def run(self, n_episodes=50, verbose=True):
        # 
        エージェントを実行

        Returns:
            episode_rewards: 各エピソードの累積報酬
            episode_steps: 各エピソードのステップ数
            episode_keys: 各エピソードで鍵を取得したか
            episode_goals: 各エピソードでゴールに到達したか
        
        # ---- Phase 1: 初期データ収集 ----
        if verbose:
            print("Phase 1: 初期ランダム探索...")
        collect_random_data(self.env, self.buffer, self.initial_collect)
        if verbose:
            print(f"  収集データ: {len(self.buffer)} 遷移")

        # ---- Phase 2: 初期モデル訓練 ----
        if verbose:
            print("Phase 2: 初期モデル訓練...")
        train_world_model(
            self.encoder, self.decoder,
            self.trans_model, self.reward_model,
            self.buffer, n_epochs=self.retrain_epochs,
            batch_size=32, lr=self.lr
        )

        # ---- Phase 3 & 4: MPC + 実行ループ ----
        episode_rewards = []
        episode_steps = []
        episode_keys = []
        episode_goals = []
        total_steps = 0

        for ep in range(n_episodes):
            obs = self.env.reset()
            ep_reward = 0
            ep_steps = 0
            got_key = False
            reached_goal = False

            while not self.env.done:
                # Phase 3: MPC 計画
                action, _ = self.mpc.plan(obs)

                # Phase 4: 実環境で実行
                next_obs, reward, done, info = self.env.step(action)
                self.buffer.add(obs, action, reward, next_obs, done)

                ep_reward += reward
                ep_steps += 1
                total_steps += 1
                obs = next_obs
                got_key = got_key or info['has_key']
                reached_goal = reached_goal or (reward >= 1.0)

            episode_rewards.append(ep_reward)
            episode_steps.append(ep_steps)
            episode_keys.append(got_key)
            episode_goals.append(reached_goal)

            # 定期的にモデル再訓練
            if (ep + 1) % self.retrain_interval == 0 and ep + 1 < n_episodes:
                if verbose:
                    print(f"  モデル再訓練（Episode {ep+1}）...")
                train_world_model(
                    self.encoder, self.decoder,
                    self.trans_model, self.reward_model,
                    self.buffer, n_epochs=self.retrain_epochs,
                    batch_size=32, lr=self.lr
                )

            if verbose and (ep + 1) % 10 == 0:
                avg_r = np.mean(episode_rewards[-10:])
                key_rate = np.mean(episode_keys[-10:])
                goal_rate = np.mean(episode_goals[-10:])
                print(f"  Episode {ep+1:>3d} | "
                      f"報酬: {avg_r:>7.3f} | "
                      f"鍵率: {key_rate:.1%} | "
                      f"ゴール率: {goal_rate:.1%}")

        return episode_rewards, episode_steps, episode_keys, episode_goals


print("WorldModelAgent 定義完了")
print("  Phase 1: ランダム探索 → Phase 2: モデル訓練")
print("  Phase 3: MPC 計画 → Phase 4: 実行 (ループ)")"""
))

# =====================================================================
# Cell 27: Run WorldModelAgent
# =====================================================================
cells.append(code(
r"""# WorldModelAgent の実行
print("=" * 60)
print("WorldModelAgent 実行開始")
print("=" * 60)

np.random.seed(42)
env_wm = KeyDoorGridWorld()
agent_wm = WorldModelAgent(
    env_wm,
    initial_collect=200,
    retrain_interval=25,
    retrain_epochs=10,
    K=50,
    horizon=5,
    lr=1e-3,
)

wm_rewards, wm_steps, wm_keys, wm_goals = agent_wm.run(n_episodes=50, verbose=True)

print("\n実行完了!")
print(f"  最終10ep 平均報酬: {np.mean(wm_rewards[-10:]):.3f}")
print(f"  最終10ep 鍵取得率: {np.mean(wm_keys[-10:]):.1%}")
print(f"  最終10ep ゴール率: {np.mean(wm_goals[-10:]):.1%}")"""
))

# =====================================================================
# Cell 28: Section 9 header
# =====================================================================
cells.append(md(
r"""<a id="section9"></a>

---

## 9. Q-Learning ベースライン

### 9.1 テーブルベース Q-Learning

世界モデルの効果を評価するため、同じ環境で Q-Learning を実行します。
Q-Learning はモデルフリー手法であり、環境のモデルを一切使いません（Notebook 142 参照）。"""
))

# =====================================================================
# Cell 29: Q-Learning baseline
# =====================================================================
cells.append(code(
r"""class QLearningBaseline:
    # 
    Q-Learning ベースライン（モデルフリー）

    状態: (row, col, has_key) をフラットインデックスに変換
    

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.15):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state_idx):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self.q_table[state_idx]
        return np.random.choice(np.where(q == q.max())[0])

    def update(self, s, a, r, s_next, done):
        target = r if done else r + self.gamma * np.max(self.q_table[s_next])
        self.q_table[s, a] += self.alpha * (target - self.q_table[s, a])


def run_q_learning(env, n_episodes=200, alpha=0.1, gamma=0.95, epsilon=0.15):
    # Q-Learning の実行
    agent = QLearningBaseline(env.n_flat_states, env.n_actions,
                               alpha=alpha, gamma=gamma, epsilon=epsilon)

    episode_rewards = []
    episode_steps = []
    episode_keys = []
    episode_goals = []

    for ep in range(n_episodes):
        env.reset()
        s = env.get_flat_state_index()
        ep_reward = 0
        got_key = False
        reached_goal = False

        while not env.done:
            a = agent.choose_action(s)
            _, reward, done, info = env.step(a)
            s_next = env.get_flat_state_index()
            agent.update(s, a, reward, s_next, done)
            s = s_next
            ep_reward += reward
            got_key = got_key or info['has_key']
            reached_goal = reached_goal or (reward >= 1.0)

        episode_rewards.append(ep_reward)
        episode_steps.append(env.steps)
        episode_keys.append(got_key)
        episode_goals.append(reached_goal)

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(episode_rewards[-50:])
            key_r = np.mean(episode_keys[-50:])
            goal_r = np.mean(episode_goals[-50:])
            print(f"  Episode {ep+1:>4d} | "
                  f"報酬: {avg_r:>7.3f} | "
                  f"鍵率: {key_r:.1%} | "
                  f"ゴール率: {goal_r:.1%}")

    return episode_rewards, episode_steps, episode_keys, episode_goals


# Q-Learning 実行
print("=" * 60)
print("Q-Learning ベースライン実行（200 エピソード）")
print("=" * 60)

np.random.seed(42)
env_ql = KeyDoorGridWorld()
ql_rewards, ql_steps, ql_keys, ql_goals = run_q_learning(
    env_ql, n_episodes=200, alpha=0.1, gamma=0.95, epsilon=0.15
)

print(f"\n最終50ep 平均報酬: {np.mean(ql_rewards[-50:]):.3f}")
print(f"最終50ep 鍵取得率: {np.mean(ql_keys[-50:]):.1%}")
print(f"最終50ep ゴール率: {np.mean(ql_goals[-50:]):.1%}")"""
))

# =====================================================================
# Cell 30: Comparison plot
# =====================================================================
cells.append(code(
r"""# WorldModel vs Q-Learning の比較
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

window = 5

# (1) 累積報酬
ax = axes[0, 0]
if len(wm_rewards) > window:
    wm_smooth = np.convolve(wm_rewards, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(wm_rewards)), wm_smooth,
            color='#E91E63', linewidth=2, label='WorldModel Agent')
ql_smooth = np.convolve(ql_rewards, np.ones(window)/window, mode='valid')
ax.plot(range(window-1, len(ql_rewards)), ql_smooth,
        color='#2196F3', linewidth=2, label='Q-Learning')
ax.set_xlabel('エピソード', fontsize=11)
ax.set_ylabel('報酬（移動平均）', fontsize=11)
ax.set_title('報酬の推移', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (2) ステップ数
ax = axes[0, 1]
if len(wm_steps) > window:
    wm_s = np.convolve(wm_steps, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(wm_steps)), wm_s,
            color='#E91E63', linewidth=2, label='WorldModel Agent')
ql_s = np.convolve(ql_steps, np.ones(window)/window, mode='valid')
ax.plot(range(window-1, len(ql_steps)), ql_s,
        color='#2196F3', linewidth=2, label='Q-Learning')
ax.set_xlabel('エピソード', fontsize=11)
ax.set_ylabel('ステップ数', fontsize=11)
ax.set_title('ステップ数の推移', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (3) 鍵取得率
ax = axes[1, 0]
wm_key_cum = np.cumsum(wm_keys) / (np.arange(len(wm_keys)) + 1)
ql_key_cum = np.cumsum(ql_keys) / (np.arange(len(ql_keys)) + 1)
ax.plot(wm_key_cum, color='#E91E63', linewidth=2, label='WorldModel Agent')
ax.plot(ql_key_cum, color='#2196F3', linewidth=2, label='Q-Learning')
ax.set_xlabel('エピソード', fontsize=11)
ax.set_ylabel('累積鍵取得率', fontsize=11)
ax.set_title('鍵取得率の推移', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (4) ゴール到達率
ax = axes[1, 1]
wm_goal_cum = np.cumsum(wm_goals) / (np.arange(len(wm_goals)) + 1)
ql_goal_cum = np.cumsum(ql_goals) / (np.arange(len(ql_goals)) + 1)
ax.plot(wm_goal_cum, color='#E91E63', linewidth=2, label='WorldModel Agent')
ax.plot(ql_goal_cum, color='#2196F3', linewidth=2, label='Q-Learning')
ax.set_xlabel('エピソード', fontsize=11)
ax.set_ylabel('累積ゴール到達率', fontsize=11)
ax.set_title('ゴール到達率の推移', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle('WorldModel Agent vs Q-Learning ベースライン',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

print("【比較まとめ】")
print(f"  WorldModel: 報酬={np.mean(wm_rewards[-10:]):.3f}, "
      f"ゴール率={np.mean(wm_goals[-10:]):.1%} ({len(wm_rewards)} episodes)")
print(f"  Q-Learning: 報酬={np.mean(ql_rewards[-50:]):.3f}, "
      f"ゴール率={np.mean(ql_goals[-50:]):.1%} ({len(ql_rewards)} episodes)")"""
))

# =====================================================================
# Cell 31: Section 10 header
# =====================================================================
cells.append(md(
r"""<a id="section10"></a>

---

## 10. モデル精度 vs 計画成功率の分析

### 10.1 世界モデルの精度評価

世界モデルの各コンポーネントの精度が計画の成功にどう影響するかを分析します。"""
))

# =====================================================================
# Cell 32: Model accuracy analysis
# =====================================================================
cells.append(code(
r"""# 世界モデルの精度評価
def evaluate_model_accuracy(encoder, trans_model, reward_model, buffer, n_samples=200):
    # 世界モデルの精度を評価
    obs, actions, rewards, next_obs, _ = buffer.sample(min(n_samples, len(buffer)))

    # エンコード
    z = encoder.forward(obs)
    z_next_true = encoder.forward(next_obs)

    # 遷移予測
    z_next_pred = trans_model.forward(z, actions)
    trans_error = np.mean((z_next_pred - z_next_true) ** 2)

    # 報酬予測
    r_pred = reward_model.forward(z, actions)
    reward_error = np.mean((r_pred - rewards) ** 2)

    # 報酬の符号一致率
    sign_match = np.mean(np.sign(r_pred) == np.sign(rewards))

    return {
        'trans_mse': trans_error,
        'reward_mse': reward_error,
        'reward_sign_accuracy': sign_match,
    }


# 訓練済みモデルの精度
metrics = evaluate_model_accuracy(
    agent_wm.encoder, agent_wm.trans_model,
    agent_wm.reward_model, agent_wm.buffer
)

print("世界モデル精度評価:")
print(f"  遷移予測 MSE: {metrics['trans_mse']:.6f}")
print(f"  報酬予測 MSE: {metrics['reward_mse']:.6f}")
print(f"  報酬符号一致率: {metrics['reward_sign_accuracy']:.1%}")"""
))

# =====================================================================
# Cell 33: Model accuracy vs planning success
# =====================================================================
cells.append(code(
r"""# モデル精度とゴール達成率の関係を可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (1) 遷移予測精度の分析
obs_eval, act_eval, rew_eval, nobs_eval, _ = agent_wm.buffer.sample(
    min(100, len(agent_wm.buffer)))
z_eval = agent_wm.encoder.forward(obs_eval)
z_next_true = agent_wm.encoder.forward(nobs_eval)
z_next_pred = agent_wm.trans_model.forward(z_eval, act_eval)

# 各次元の誤差
dim_errors = np.mean((z_next_pred - z_next_true) ** 2, axis=0)

axes[0].bar(range(len(dim_errors)), dim_errors, color='#4CAF50', alpha=0.7)
axes[0].set_xlabel('潜在次元', fontsize=11)
axes[0].set_ylabel('MSE', fontsize=11)
axes[0].set_title('遷移モデル: 次元ごとの予測誤差', fontsize=13)
axes[0].grid(True, alpha=0.3)

# (2) 報酬予測の散布図
r_pred_eval = agent_wm.reward_model.forward(z_eval, act_eval)
axes[1].scatter(rew_eval, r_pred_eval, alpha=0.5, color='#E91E63', s=30)
r_range = [min(rew_eval.min(), r_pred_eval.min()),
           max(rew_eval.max(), r_pred_eval.max())]
axes[1].plot(r_range, r_range, 'k--', linewidth=1, label='完全予測')
axes[1].set_xlabel('実際の報酬', fontsize=11)
axes[1].set_ylabel('予測報酬', fontsize=11)
axes[1].set_title('報酬モデル: 実際 vs 予測', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("【分析結果】")
print("  遷移モデル: 一部の潜在次元で誤差が大きい場合がある")
print("  報酬モデル: ステップペナルティ(-0.01)は比較的正確だが")
print("             鍵報酬(+0.5)やゴール報酬(+1.0)のような稀な事象は")
print("             データが少ないため予測が困難")"""
))

# =====================================================================
# Cell 34: Summary of comparison
# =====================================================================
cells.append(code(
r"""# 手法間比較の定量的まとめ
def print_comparison_summary(wm_rewards, wm_goals, wm_keys,
                              ql_rewards, ql_goals, ql_keys):
    print("=" * 70)
    print("WorldModel Agent vs Q-Learning: 定量比較")
    print("=" * 70)
    print(f"""
┌──────────────────┬───────────────────┬───────────────────┐
│ 指標              │ WorldModel Agent  │ Q-Learning        │
├──────────────────┼───────────────────┼───────────────────┤
│ エピソード数      │ {len(wm_rewards):>17d} │ {len(ql_rewards):>17d} │
│ 平均報酬 (全体)   │ {np.mean(wm_rewards):>17.3f} │ {np.mean(ql_rewards):>17.3f} │
│ 鍵取得率 (全体)   │ {np.mean(wm_keys):>16.1%} │ {np.mean(ql_keys):>16.1%} │
│ ゴール率 (全体)   │ {np.mean(wm_goals):>16.1%} │ {np.mean(ql_goals):>16.1%} │
│ 環境データ量      │ {len(agent_wm.buffer):>17d} │            N/A    │
│ 学習方式          │ MPC (Model-based) │ Q-table (Free)    │
└──────────────────┴───────────────────┴───────────────────┘
    """)

    print("【考察】")
    print("  1. Q-Learning は十分なエピソードで安定した方策を学習する")
    print("  2. WorldModel Agent は少ないエピソードでも計画的に動ける")
    print("  3. ただし世界モデルの精度がボトルネックになりうる")
    print("  4. 世界モデルは「想像」で計画するため、未知の状況に弱い")


print_comparison_summary(wm_rewards, wm_goals, wm_keys,
                         ql_rewards, ql_goals, ql_keys)"""
))

# =====================================================================
# Cell 35: Section 11 header
# =====================================================================
cells.append(md(
r"""<a id="section11"></a>

---

## 11. まとめ・よくあるエラー・確認クイズ

### 11.1 このノートブックで学んだこと

| 項目 | 内容 |
|------|------|
| **KeyDoorGridWorld** | 鍵→扉→ゴールの逐次タスクを持つ 7×7 環境 |
| **ObservationEncoder** | 観測画像を 64 次元潜在ベクトルに変換 |
| **TransitionModel** | (z_t, a_t) → z_{t+1} を予測する遷移モデル |
| **RewardModel** | (z_t, a_t) → r̂_t を予測する報酬モデル |
| **ObservationDecoder** | 潜在ベクトルから観測画像を再構成 |
| **MPC** | K 個のランダム行動列をロールアウトして最良を選ぶ計画手法 |
| **WorldModelAgent** | データ収集→訓練→計画→実行の 4 フェーズパイプライン |
| **Q-Learning ベースライン** | モデルフリーの比較対象 |

### 11.2 Phase 7 との接続

| 本ノートブックの要素 | Phase 7 での対応 |
|---------------------|------------------|
| ObservationEncoder | Nb 140: 表現学習、Nb 141: JEPA のエンコーダ |
| TransitionModel | Nb 142: Dyna-Q の遷移モデル、Nb 143: RSSM |
| MPC (ランダムシューティング) | Nb 143: DreamerV3 の Actor-Critic（より洗練された計画） |
| 潜在行動空間 | Nb 144: Genie の潜在行動発見 |"""
))

# =====================================================================
# Cell 36: Common errors
# =====================================================================
cells.append(md(
r"""### 11.3 よくあるエラー 3選

#### エラー1: エンコーダの勾配がデコーダからのみ流れる

```python
# 問題: 遷移損失・報酬損失のエンコーダへの勾配を無視
encoder.backward(grad_z_from_decoder)  # デコーダからの勾配のみ
```

**解説**: 厳密には遷移損失と報酬損失からもエンコーダへの勾配が流れるべきです。
本実装では簡略化していますが、実際の DreamerV3 では全損失がエンコーダに伝播します。

#### エラー2: MPC で最初の行動だけでなく全行動列を実行してしまう

```python
# NG: 全ホライズンの行動を一度に実行
for t in range(horizon):
    obs, r, done, _ = env.step(best_actions[t])
```

**対策**: MPC は **各ステップで再計画** するのが正しい使い方です。
最初の行動 $a_0$ だけを実行し、次の観測で再度計画します（receding horizon）。

#### エラー3: バッファに古いデータが溜まりすぎる

```python
# 問題: 初期のランダムデータが大量に残り、モデルが古い経験に引きずられる
buffer = TransitionBuffer(capacity=100000)  # 大きすぎる
```

**対策**: バッファサイズを適切に設定し、古いデータを自然に押し出す。
または優先度付きリプレイを使って重要な経験を優先的に利用します。"""
))

# =====================================================================
# Cell 37: Quiz
# =====================================================================
cells.append(md(
r"""### 11.4 確認クイズ（5問）

---

**Q1**: MPC のランダムシューティング法で、ホライズン H=5、サンプル数 K=50 の場合、
遷移モデルは合計何回呼び出されますか？

<details>
<summary>回答を表示</summary>

**250 回** です。K=50 個の行動列それぞれについて H=5 ステップのロールアウトを行うため、
$50 \times 5 = 250$ 回の遷移予測が必要です。

</details>

---

**Q2**: 世界モデルの訓練で使う3つの損失関数を挙げ、それぞれの役割を説明してください。

<details>
<summary>回答を表示</summary>

1. **再構成損失** $\mathcal{L}_{recon}$: エンコーダ + デコーダが観測を正確に復元できるかを評価。
   表現の品質を保証する。

2. **遷移損失** $\mathcal{L}_{trans}$: 予測された次の潜在ベクトルと実際の次の潜在ベクトルの差。
   環境のダイナミクスを正確にモデル化する。

3. **報酬損失** $\mathcal{L}_{reward}$: 予測報酬と実際の報酬の差。
   MPC での計画に必要な報酬予測の精度を確保する。

</details>

---

**Q3**: KeyDoorGridWorld で Q-Learning の状態空間が `7 × 7 × 2 = 98` である理由を説明してください。

<details>
<summary>回答を表示</summary>

状態は `(row, col, has_key)` の3要素で構成されます:
- `row`: 0-6 の 7 通り
- `col`: 0-6 の 7 通り
- `has_key`: 0 or 1 の 2 通り

鍵の所持状態によって同じ位置でも異なる状態として扱う必要があるため
（扉を通れるかどうかが変わる）、$7 \times 7 \times 2 = 98$ 状態になります。

</details>

---

**Q4**: WorldModelAgent で「モデル再訓練」を行う理由は何ですか？
初期訓練だけでは不十分な理由を説明してください。

<details>
<summary>回答を表示</summary>

初期訓練はランダム方策で収集したデータに基づくため:

1. **データの偏り**: ランダム探索ではスタート付近のデータが多く、
   ゴール付近のデータが少ない。モデルの精度に偏りが生じる。

2. **分布シフト**: MPC で行動すると、ランダム方策とは異なる状態を訪問する。
   初期モデルが経験したことのない領域では予測が不正確になる。

3. **段階的改善**: 新しいデータでモデルを再訓練することで、
   エージェントが実際に訪問する領域でのモデル精度が向上し、
   より良い計画が可能になる（好循環）。

</details>

---

**Q5**: DreamerV3（Notebook 143）と本ノートブックの MPC の計画手法の違いを
1つ挙げてください。

<details>
<summary>回答を表示</summary>

**MPC（本ノートブック）:**
- ランダムシューティング法: K 個のランダム行動列から最良を選ぶ
- 行動列を **直接探索** する（計画時に方策ネットワークを使わない）
- 各ステップで再計画が必要

**DreamerV3:**
- Actor-Critic: 方策ネットワーク（Actor）と価値ネットワーク（Critic）を学習
- 想像の中で **方策を勾配ベースで最適化** する
- 学習済みの方策を直接使って行動できる（計画不要）

つまり、MPC は「計画時に探索」、DreamerV3 は「訓練時に方策を学習」という
根本的な違いがあります。DreamerV3 の方が推論時に高速です。

</details>"""
))

# =====================================================================
# Cell 38: Final checklist
# =====================================================================
cells.append(code(
r"""def print_checklist():
    # 学習チェックリスト
    print("=" * 60)
    print("第145章 学習チェックリスト")
    print("=" * 60)
    items = [
        "KeyDoorGridWorld の状態・行動・報酬設計を説明できる",
        "7x7x3 RGB 観測の生成方法を理解している",
        "ObservationEncoder の構造と役割を説明できる",
        "TransitionModel が (z, a) → z' を予測する仕組みを理解している",
        "RewardModel が (z, a) → r を予測する仕組みを理解している",
        "ObservationDecoder による再構成の目的を説明できる",
        "MPC ランダムシューティングのアルゴリズムを説明できる",
        "WorldModelAgent の 4 フェーズパイプラインを説明できる",
        "モデル再訓練が必要な理由を説明できる",
        "Q-Learning との比較で世界モデルの利点と限界を議論できる",
        "遷移予測誤差が計画品質に与える影響を理解している",
        "DreamerV3 (Nb 143) との手法の違いを説明できる",
    ]
    for i, item in enumerate(items, 1):
        print(f"  [ ] {i:>2d}. {item}")
    print(f"\n全 {len(items)} 項目")

print_checklist()"""
))

# =====================================================================
# Cell 39: Final markdown
# =====================================================================
cells.append(md(
r"""---

## おわりに

本ノートブックでは、Phase 7「世界モデル」シリーズの **Capstone** として、
学習した世界モデルによる計画の全パイプラインを実装しました。

### 本章の核心

> **世界モデル = 環境を心の中にシミュレーションする能力**

エージェントが環境のダイナミクスを学習し（遷移モデル）、
その「想像」の中で複数の行動計画を評価し（MPC）、
最良の計画を実環境で実行する — これは人間が日常的に行っている意思決定プロセスそのものです。

### 参考文献

1. Ha, D., & Schmidhuber, J. (2018). World Models. *arXiv:1803.10122*.
2. Hafner, D., et al. (2023). Mastering Diverse Domains through World Models. *arXiv:2301.04104*.
3. Sutton, R. S. (1991). Dyna, an Integrated Architecture for Learning, Planning, and Reacting.
4. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence.

---"""
))


# =====================================================================
# Fix cell sources to proper line-based format & assign IDs
# =====================================================================
for i, cell in enumerate(cells):
    raw = cell["source"]
    text = "\n".join(raw)
    lines = text.split("\n")
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

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "145_grid_world_agent_v1.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Wrote {len(cells)} cells to {output_path}")

# Count code lines
code_lines = 0
for cell in cells:
    if cell["cell_type"] == "code":
        code_lines += len(cell["source"])
print(f"Code cells: {sum(1 for c in cells if c['cell_type'] == 'code')}")
print(f"Markdown cells: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"Approximate code lines: {code_lines}")
