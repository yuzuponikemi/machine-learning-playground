"""ArmWorld — Unit 15 (VLA) 共通の実験基盤

2リンク平面アーム + 色付きブロックの最小環境。Notebook 170-174 で使い回す。

設計方針（既存150本と揃える）:
  - 外部シミュレータに依存しない（gymnasium / MuJoCo を使わない）。`reset()` / `step()`
    の API だけ Gym 互換にしてある。
  - CPU で完結する。画像は NumPy で直接ラスタライズする（matplotlib を経由しない）。
  - `seed` で完全に再現する。

VLA に必要で、既存150本のどこにも無かった要素をここで一度に供給する:
  - 連続行動（関節角速度指令）
  - 画像観測（64x64 RGB）
  - 固有感覚（関節角・エンドエフェクタ位置・グリッパ）
  - 言語指示（テンプレート生成）
  - 順運動学（SE(2)）

Notebook 170 で ArmWorld と ScriptedExpert を導入し、171 以降はここから import する。
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# 定数
# ----------------------------------------------------------------------------

LINK1, LINK2 = 1.0, 0.8          # リンク長
WORLD_HALF = 2.0                 # 描画範囲 [-2, 2] x [-2, 2]
IMG_SIZE = 64                    # 観測画像の一辺
MAX_DELTA = 0.20                 # 1ステップで動かせる関節角の上限 [rad]
MAX_EEF_STEP = 0.12              # action_space="eef" のとき 1ステップで動かす距離の上限
SUCCESS_RADIUS = 0.15            # 目標に到達したと見なす距離

# ブロックの色（名前 -> RGB）。言語指示でこの名前を使う。
COLORS = {
    "red":   (220,  50,  50),
    "blue":  ( 50,  90, 220),
    "green": ( 40, 170,  70),
}
COLOR_NAMES = list(COLORS.keys())

# 目標領域の名前 -> 中心座標。言語指示でこの名前を使う。
REGIONS = {
    "upper left":  (-0.80,  1.25),
    "upper right": ( 0.80,  1.25),
    "left":        (-1.40,  0.30),
    "right":       ( 1.40,  0.30),
}
REGION_NAMES = list(REGIONS.keys())


# ----------------------------------------------------------------------------
# 順運動学（SE(2)）
# ----------------------------------------------------------------------------

def forward_kinematics(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """関節角 theta=(th1, th2) から 肘 と エンドエフェクタ(EEF) の位置を返す。

    Notebook 53 の剛体変換の2次元版。th1 は基部からの絶対角、th2 は肘の相対角。
    """
    th1, th2 = float(theta[0]), float(theta[1])
    elbow = np.array([LINK1 * np.cos(th1), LINK1 * np.sin(th1)])
    eef = elbow + np.array([LINK2 * np.cos(th1 + th2), LINK2 * np.sin(th1 + th2)])
    return elbow, eef


def _jacobian(theta: np.ndarray) -> np.ndarray:
    """EEF 位置の関節角に関するヤコビアン（2x2）。"""
    th1, th2 = float(theta[0]), float(theta[1])
    s1, c1 = np.sin(th1), np.cos(th1)
    s12, c12 = np.sin(th1 + th2), np.cos(th1 + th2)
    return np.array([[-LINK1 * s1 - LINK2 * s12, -LINK2 * s12],
                     [ LINK1 * c1 + LINK2 * c12,  LINK2 * c12]])


def _draw_disc(img: np.ndarray, xy, radius: float, rgb) -> None:
    """ワールド座標 xy を中心とする円を img に描く（NumPy だけで塗る）。"""
    n = img.shape[0]
    gy, gx = np.mgrid[0:n, 0:n]
    # ピクセル中心のワールド座標（y は画像上方向が +）
    wx = (gx + 0.5) / n * (2 * WORLD_HALF) - WORLD_HALF
    wy = WORLD_HALF - (gy + 0.5) / n * (2 * WORLD_HALF)
    mask = (wx - xy[0]) ** 2 + (wy - xy[1]) ** 2 <= radius ** 2
    img[mask] = rgb


def _draw_segment(img: np.ndarray, p0, p1, width: float, rgb) -> None:
    """線分 p0-p1 を太さ width で描く（点と線分の距離で塗る）。"""
    n = img.shape[0]
    gy, gx = np.mgrid[0:n, 0:n]
    wx = (gx + 0.5) / n * (2 * WORLD_HALF) - WORLD_HALF
    wy = WORLD_HALF - (gy + 0.5) / n * (2 * WORLD_HALF)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    denom = float(d @ d) + 1e-9
    t = ((wx - p0[0]) * d[0] + (wy - p0[1]) * d[1]) / denom
    t = np.clip(t, 0.0, 1.0)
    cx = p0[0] + t * d[0]
    cy = p0[1] + t * d[1]
    mask = (wx - cx) ** 2 + (wy - cy) ** 2 <= width ** 2
    img[mask] = rgb


# ----------------------------------------------------------------------------
# 環境
# ----------------------------------------------------------------------------

class ArmWorld:
    """2リンク平面アームの到達・運搬タスク。

    行動 (continuous, 2次元 + グリッパ1次元):
        a[0], a[1] : 関節角の増分。**[-1, 1] に正規化された値**で渡す。
                     環境側で clip して MAX_DELTA 倍する（a=1 で +0.2 rad）。
                     正規化しておくと方針の出力分布（標準偏差 0.3 程度）が
                     飽和しないので、連続制御の学習が安定する。
        a[2]       : グリッパ（>0 で閉じる）。task="reach" では使わない。

    観測 (dict):
        proprio : (7,) [cos th1, sin th1, cos th2, sin th2, eef_x, eef_y, gripper]
        target  : (2,) 目標座標（task="reach" では目標点、"transport" ではブロック位置）
        image   : (64, 64, 3) uint8
        instruction : str  テンプレート生成の自然文

    報酬: -(EEF と目標の距離)。到達したステップで +1 のボーナス。
    """

    def __init__(self, task: str = "reach", n_blocks: int = 3,
                 max_steps: int = 50, seed: int | None = None,
                 image_obs: bool = True, action_space: str = "eef",
                 success_radius: float = SUCCESS_RADIUS):
        assert task in ("reach", "transport")
        assert action_space in ("eef", "joint")
        self.task = task
        self.action_space = action_space
        # 許容誤差。小さくすると「精度が要るタスク」になる（171 で使う）
        self.success_radius = success_radius
        self.n_blocks = n_blocks
        self.max_steps = max_steps
        self.image_obs = image_obs
        self.rng = np.random.default_rng(seed)
        self.action_dim = 3
        self.proprio_dim = 7
        self.reset()

    # -- 内部状態 ----------------------------------------------------------
    def _sample_blocks(self) -> None:
        self.block_pos = []
        self.block_colors = []
        names = list(COLOR_NAMES)
        self.rng.shuffle(names)
        for i in range(self.n_blocks):
            r = self.rng.uniform(0.85, 1.50)
            a = self.rng.uniform(0.15, np.pi - 0.15)
            self.block_pos.append(np.array([r * np.cos(a), r * np.sin(a)]))
            self.block_colors.append(names[i % len(names)])

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.theta = np.array([self.rng.uniform(0.2, 1.4),
                               self.rng.uniform(-1.0, 1.0)])
        self.gripper = 0.0
        self.t = 0
        self.success = False
        self._sample_blocks()

        if self.task == "reach":
            # 目標はブロックのうち1つ。指示文はその色を指す。
            self.target_idx = int(self.rng.integers(self.n_blocks))
            self.target_color = self.block_colors[self.target_idx]
            self.region = None
            self.instruction = f"touch the {self.target_color} block"
        else:
            self.target_idx = int(self.rng.integers(self.n_blocks))
            self.target_color = self.block_colors[self.target_idx]
            self.region = REGION_NAMES[int(self.rng.integers(len(REGION_NAMES)))]
            self.instruction = (f"move the {self.target_color} block "
                                f"to the {self.region}")
        return self._obs()

    @property
    def target(self) -> np.ndarray:
        return self.block_pos[self.target_idx]

    def _obs(self) -> dict:
        _, eef = forward_kinematics(self.theta)
        proprio = np.array([np.cos(self.theta[0]), np.sin(self.theta[0]),
                            np.cos(self.theta[1]), np.sin(self.theta[1]),
                            eef[0], eef[1], self.gripper], dtype=np.float32)
        obs = {"proprio": proprio,
               "target": self.target.astype(np.float32),
               "instruction": self.instruction}
        if self.image_obs:
            obs["image"] = self.render()
        return obs

    def state_vec(self) -> np.ndarray:
        """方針ネットワークに渡す低次元状態（proprio + 目標の相対位置）。"""
        o = self._obs()
        _, eef = forward_kinematics(self.theta)
        rel = self.target - eef
        return np.concatenate([o["proprio"], rel]).astype(np.float32)

    def step(self, action) -> tuple[dict, float, bool, dict]:
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.size == 2:
            a = np.concatenate([a, [0.0]])
        # 行動は [-1, 1] 正規化。環境側でスケールする。
        u = np.clip(a[:2], -1.0, 1.0)
        if self.action_space == "joint":
            # 関節空間：そのまま関節角の増分になる
            d = u * MAX_DELTA
        else:
            # 作業空間（EEF）：EEF をこの方向に動かしたい、という指令。
            # 減衰付き最小二乗で関節角の増分に変換する（operational space control）。
            J = _jacobian(self.theta)
            dx = u * MAX_EEF_STEP
            d = J.T @ np.linalg.solve(J @ J.T + 0.05 * np.eye(2), dx)
            d = np.clip(d, -MAX_DELTA, MAX_DELTA)
        self.theta = self.theta + d
        # 関節の可動範囲（物理的に折り返さないように制限する）
        self.theta[0] = np.clip(self.theta[0], -0.8, np.pi + 0.8)
        self.theta[1] = np.clip(self.theta[1], -2.8, 2.8)
        self.gripper = 1.0 if a[2] > 0 else 0.0

        _, eef = forward_kinematics(self.theta)
        if self.task == "transport" and self.gripper > 0.5:
            # 掴んでいる間はブロックが EEF に追従する
            if np.linalg.norm(eef - self.block_pos[self.target_idx]) < 0.35:
                self.block_pos[self.target_idx] = eef.copy()

        if self.task == "reach":
            goal = self.target
        else:
            goal = np.array(REGIONS[self.region])
            if self.gripper < 0.5:
                goal = self.block_pos[self.target_idx]
        dist = float(np.linalg.norm(eef - goal))
        reward = -dist
        reached = dist < self.success_radius
        if reached and not self.success:
            self.success = True
            reward += 1.0

        self.t += 1
        done = bool(self.t >= self.max_steps or reached)
        return self._obs(), reward, done, {"distance": dist,
                                           "success": self.success}

    # -- 描画 --------------------------------------------------------------
    def render(self, size: int = IMG_SIZE) -> np.ndarray:
        img = np.full((size, size, 3), 240, dtype=np.uint8)
        # 目標領域（transport のときだけ薄く塗る）
        if self.task == "transport" and self.region is not None:
            _draw_disc(img, REGIONS[self.region], 0.30, (205, 205, 225))
        for pos, cname in zip(self.block_pos, self.block_colors):
            _draw_disc(img, pos, 0.16, COLORS[cname])
        elbow, eef = forward_kinematics(self.theta)
        _draw_segment(img, (0.0, 0.0), elbow, 0.07, (90, 90, 90))
        _draw_segment(img, elbow, eef, 0.055, (130, 130, 130))
        _draw_disc(img, (0.0, 0.0), 0.10, (40, 40, 40))
        _draw_disc(img, eef, 0.08, (20, 20, 20) if self.gripper > 0.5
                   else (250, 180, 40))
        return img


# ----------------------------------------------------------------------------
# スクリプト方針（デモ生成用。Notebook 171 以降で使う）
# ----------------------------------------------------------------------------

class ScriptedExpert:
    """ヤコビアンの擬似逆行列で EEF を目標へ寄せる手書き方針。

    学習器ではなく「デモを作る人間の代わり」。171 の模倣学習の教師データになる。

    転置 J^T ではなく減衰付き最小二乗 J^T (J J^T + lam I)^{-1} を使う。
    転置は向きだけ合っていて大きさが関節角に依存するため、2リンクでも
    姿勢によって収束が極端に遅くなる（実測: reach 成功率 0.72 → 0.95）。
    """

    def __init__(self, gain: float = 1.2, damping: float = 0.1,
                 noise: float = 0.0, seed: int | None = None):
        self.gain = gain
        self.damping = damping
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def jacobian(theta: np.ndarray) -> np.ndarray:
        return _jacobian(theta)

    def act(self, env: ArmWorld) -> np.ndarray:
        _, eef = forward_kinematics(env.theta)
        if env.task == "reach":
            goal = env.target
            grip = 0.0
        else:
            holding = (env.gripper > 0.5)
            near = np.linalg.norm(eef - env.block_pos[env.target_idx]) < 0.3
            if holding or near:
                goal = np.array(REGIONS[env.region])
                grip = 1.0
            else:
                goal = env.block_pos[env.target_idx]
                grip = 0.0
        err = goal - eef
        if env.action_space == "eef":
            # 作業空間指令：EEF を誤差の方向に動かせばよい（幾何の反転が要らない）
            a_norm = np.clip(self.gain * err / MAX_EEF_STEP, -1.0, 1.0)
        else:
            # 関節空間指令：減衰付き最小二乗で誤差を関節角に落とす（特異姿勢でも解ける）
            J = self.jacobian(env.theta)
            d = self.gain * (J.T @ np.linalg.solve(
                J @ J.T + self.damping * np.eye(2), err))
            a_norm = np.clip(d / MAX_DELTA, -1.0, 1.0)
        if self.noise > 0:
            a_norm = np.clip(a_norm + self.rng.normal(0, self.noise, size=2), -1.0, 1.0)
        return np.concatenate([a_norm, [grip]])


def rollout(env: ArmWorld, policy, max_steps: int | None = None,
            collect_images: bool = False) -> dict:
    """1エピソード回して軌跡を返す。policy は act(env) か callable(state)。"""
    obs = env.reset()
    states, actions, rewards, images = [], [], [], []
    steps = max_steps or env.max_steps
    for _ in range(steps):
        if hasattr(policy, "act"):
            a = policy.act(env)
        else:
            a = policy(env.state_vec())
        states.append(env.state_vec())
        actions.append(np.asarray(a, dtype=np.float32).reshape(-1)[:3])
        if collect_images:
            images.append(obs["image"])
        obs, r, done, info = env.step(a)
        rewards.append(r)
        if done:
            break
    return {"states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "images": np.array(images) if collect_images else None,
            "success": bool(info["success"]),
            "instruction": env.instruction}
