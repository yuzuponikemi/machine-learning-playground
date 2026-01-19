"""
Camera Utilities for 3D Computer Vision
=======================================

カメラモデル、射影、光線生成のユーティリティ関数群
"""

import numpy as np
from typing import Tuple, Optional, Union


# ============================================================
# 内部パラメータ
# ============================================================

def create_intrinsic_matrix(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    skew: float = 0.0
) -> np.ndarray:
    """
    カメラ内部パラメータ行列を生成

    K = | fx  skew  cx |
        | 0   fy    cy |
        | 0   0     1  |

    Parameters:
    -----------
    fx, fy : float
        焦点距離（ピクセル単位）
    cx, cy : float
        主点座標（ピクセル単位）
    skew : float
        スキュー係数（通常0）

    Returns:
    --------
    K : np.ndarray (3, 3)
        内部パラメータ行列
    """
    return np.array([
        [fx, skew, cx],
        [0,  fy,   cy],
        [0,  0,    1]
    ], dtype=np.float64)


def create_intrinsic_from_fov(
    fov_x: float,
    width: int,
    height: int,
    fov_y: Optional[float] = None
) -> np.ndarray:
    """
    画角（FOV）から内部パラメータ行列を生成

    Parameters:
    -----------
    fov_x : float
        水平方向の画角（ラジアン）
    width : int
        画像幅（ピクセル）
    height : int
        画像高さ（ピクセル）
    fov_y : float, optional
        垂直方向の画角（Noneの場合、fov_xから計算）

    Returns:
    --------
    K : np.ndarray (3, 3)
        内部パラメータ行列
    """
    fx = width / (2 * np.tan(fov_x / 2))

    if fov_y is None:
        fy = fx  # 正方形ピクセルを仮定
    else:
        fy = height / (2 * np.tan(fov_y / 2))

    cx = width / 2
    cy = height / 2

    return create_intrinsic_matrix(fx, fy, cx, cy)


# ============================================================
# カメラポーズ
# ============================================================

def create_camera_pose(
    R: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """
    カメラポーズ行列（ワールド→カメラ変換）を生成

    [R|t] = | r11 r12 r13 tx |
            | r21 r22 r23 ty |
            | r31 r32 r33 tz |

    Parameters:
    -----------
    R : np.ndarray (3, 3)
        回転行列
    t : np.ndarray (3,)
        並進ベクトル

    Returns:
    --------
    pose : np.ndarray (3, 4)
        カメラポーズ行列
    """
    t = np.asarray(t).flatten()
    pose = np.zeros((3, 4), dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def get_camera_center(pose: np.ndarray) -> np.ndarray:
    """
    カメラポーズからカメラ中心（ワールド座標）を計算

    C = -R^T @ t

    Parameters:
    -----------
    pose : np.ndarray (3, 4)
        カメラポーズ行列 [R|t]

    Returns:
    --------
    C : np.ndarray (3,)
        カメラ中心のワールド座標
    """
    R = pose[:3, :3]
    t = pose[:3, 3]
    return -R.T @ t


def look_at(
    camera_position: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = np.array([0, 1, 0])
) -> np.ndarray:
    """
    カメラ位置とターゲットからカメラポーズを生成（look-at変換）

    Parameters:
    -----------
    camera_position : np.ndarray (3,)
        カメラの位置（ワールド座標）
    target : np.ndarray (3,)
        注視点（ワールド座標）
    up : np.ndarray (3,)
        上方向ベクトル

    Returns:
    --------
    pose : np.ndarray (3, 4)
        カメラポーズ行列
    """
    camera_position = np.asarray(camera_position).flatten()
    target = np.asarray(target).flatten()
    up = np.asarray(up).flatten()

    # カメラのz軸（視線方向の逆）
    z = camera_position - target
    z = z / np.linalg.norm(z)

    # カメラのx軸
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)

    # カメラのy軸
    y = np.cross(z, x)

    # 回転行列（ワールド→カメラ）
    R = np.stack([x, y, z], axis=0)

    # 並進ベクトル
    t = -R @ camera_position

    return create_camera_pose(R, t)


# ============================================================
# 射影
# ============================================================

def project_points(
    points_3d: np.ndarray,
    K: np.ndarray,
    pose: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
    t: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    3D点を2D画像座標に射影

    Parameters:
    -----------
    points_3d : np.ndarray (N, 3) or (3,)
        3D点（ワールド座標）
    K : np.ndarray (3, 3)
        内部パラメータ行列
    pose : np.ndarray (3, 4), optional
        カメラポーズ行列 [R|t]
    R : np.ndarray (3, 3), optional
        回転行列（poseがNoneの場合）
    t : np.ndarray (3,), optional
        並進ベクトル（poseがNoneの場合）

    Returns:
    --------
    points_2d : np.ndarray (N, 2) or (2,)
        2D画像座標
    """
    points_3d = np.asarray(points_3d)
    single_point = points_3d.ndim == 1
    if single_point:
        points_3d = points_3d.reshape(1, -1)

    # カメラ座標系への変換
    if pose is not None:
        R = pose[:3, :3]
        t = pose[:3, 3]
    elif R is None:
        R = np.eye(3)
        t = np.zeros(3)

    t = np.asarray(t).flatten()
    points_cam = (R @ points_3d.T).T + t  # (N, 3)

    # 正規化画像座標
    points_norm = points_cam[:, :2] / points_cam[:, 2:3]  # (N, 2)

    # ピクセル座標への変換
    points_2d = (K[:2, :2] @ points_norm.T).T + K[:2, 2]  # (N, 2)

    if single_point:
        return points_2d[0]
    return points_2d


def project_points_homogeneous(
    points_3d: np.ndarray,
    P: np.ndarray
) -> np.ndarray:
    """
    投影行列を使って3D点を2D画像座標に射影

    Parameters:
    -----------
    points_3d : np.ndarray (N, 3)
        3D点（ワールド座標）
    P : np.ndarray (3, 4)
        投影行列 P = K @ [R|t]

    Returns:
    --------
    points_2d : np.ndarray (N, 2)
        2D画像座標
    """
    points_3d = np.asarray(points_3d)
    if points_3d.ndim == 1:
        points_3d = points_3d.reshape(1, -1)

    # 同次座標に変換
    points_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])

    # 射影
    points_proj = (P @ points_h.T).T  # (N, 3)

    # 正規化
    points_2d = points_proj[:, :2] / points_proj[:, 2:3]

    return points_2d


# ============================================================
# 光線生成（Ray Casting）
# ============================================================

def pixel_to_ray(
    pixels: np.ndarray,
    K: np.ndarray,
    pose: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ピクセル座標から光線（原点と方向）を生成

    r(t) = origin + t * direction

    Parameters:
    -----------
    pixels : np.ndarray (N, 2) or (2,)
        ピクセル座標 (u, v)
    K : np.ndarray (3, 3)
        内部パラメータ行列
    pose : np.ndarray (3, 4), optional
        カメラポーズ行列 [R|t]（Noneの場合、カメラ座標系）

    Returns:
    --------
    origins : np.ndarray (N, 3) or (3,)
        光線の原点（カメラ中心）
    directions : np.ndarray (N, 3) or (3,)
        光線の方向（正規化済み）
    """
    pixels = np.asarray(pixels)
    single_pixel = pixels.ndim == 1
    if single_pixel:
        pixels = pixels.reshape(1, -1)

    N = pixels.shape[0]
    K_inv = np.linalg.inv(K)

    # ピクセル座標を同次座標に
    pixels_h = np.hstack([pixels, np.ones((N, 1))])  # (N, 3)

    # カメラ座標系での方向
    dirs_cam = (K_inv @ pixels_h.T).T  # (N, 3)
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=1, keepdims=True)

    if pose is not None:
        R = pose[:3, :3]
        t = pose[:3, 3]
        # ワールド座標系へ変換
        origins = np.tile(-R.T @ t, (N, 1))  # カメラ中心
        directions = (R.T @ dirs_cam.T).T  # 方向
    else:
        origins = np.zeros((N, 3))
        directions = dirs_cam

    if single_pixel:
        return origins[0], directions[0]
    return origins, directions


def sample_along_ray(
    origin: np.ndarray,
    direction: np.ndarray,
    near: float,
    far: float,
    num_samples: int,
    randomize: bool = False
) -> np.ndarray:
    """
    光線上の点をサンプリング

    Parameters:
    -----------
    origin : np.ndarray (3,)
        光線の原点
    direction : np.ndarray (3,)
        光線の方向
    near : float
        サンプリング開始距離
    far : float
        サンプリング終了距離
    num_samples : int
        サンプル数
    randomize : bool
        ランダムな摂動を加えるか

    Returns:
    --------
    points : np.ndarray (num_samples, 3)
        サンプリングされた3D点
    t_vals : np.ndarray (num_samples,)
        光線パラメータ t の値
    """
    origin = np.asarray(origin).flatten()
    direction = np.asarray(direction).flatten()

    # 等間隔のt値を生成
    t_vals = np.linspace(near, far, num_samples)

    if randomize:
        # 各区間内でランダムにサンプル
        mids = (t_vals[:-1] + t_vals[1:]) / 2
        upper = np.concatenate([mids, t_vals[-1:]])
        lower = np.concatenate([t_vals[:1], mids])
        t_vals = lower + (upper - lower) * np.random.rand(num_samples)

    # 点を計算
    points = origin[None, :] + t_vals[:, None] * direction[None, :]

    return points, t_vals


def get_all_rays(
    height: int,
    width: int,
    K: np.ndarray,
    pose: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    画像の全ピクセルに対する光線を生成

    Parameters:
    -----------
    height : int
        画像の高さ
    width : int
        画像の幅
    K : np.ndarray (3, 3)
        内部パラメータ行列
    pose : np.ndarray (3, 4), optional
        カメラポーズ行列

    Returns:
    --------
    origins : np.ndarray (H*W, 3)
        光線の原点
    directions : np.ndarray (H*W, 3)
        光線の方向
    """
    # ピクセルグリッドを生成
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)
    pixels = np.stack([u.flatten(), v.flatten()], axis=1)  # (H*W, 2)

    return pixel_to_ray(pixels, K, pose)


# ============================================================
# 歪み補正
# ============================================================

def undistort_points(
    points: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray
) -> np.ndarray:
    """
    レンズ歪みを補正した点座標を計算

    Parameters:
    -----------
    points : np.ndarray (N, 2)
        歪んだ画像座標
    K : np.ndarray (3, 3)
        内部パラメータ行列
    dist_coeffs : np.ndarray (4,) or (5,)
        歪み係数 [k1, k2, p1, p2, (k3)]

    Returns:
    --------
    undistorted : np.ndarray (N, 2)
        歪み補正後の画像座標
    """
    # OpenCVを使用する場合はcv2.undistortPointsを推奨
    # ここでは簡易実装

    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    # 正規化座標に変換
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (points[:, 0] - cx) / fx
    y = (points[:, 1] - cy) / fy

    # 歪み係数
    k1, k2, p1, p2 = dist_coeffs[:4]
    k3 = dist_coeffs[4] if len(dist_coeffs) > 4 else 0

    # 反復法で歪みを除去
    x_undist, y_undist = x.copy(), y.copy()

    for _ in range(10):  # 反復回数
        r2 = x_undist**2 + y_undist**2
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

        x_undist = (x - 2 * p1 * x_undist * y_undist - p2 * (r2 + 2 * x_undist**2)) / radial
        y_undist = (y - p1 * (r2 + 2 * y_undist**2) - 2 * p2 * x_undist * y_undist) / radial

    # ピクセル座標に戻す
    undistorted = np.stack([
        x_undist * fx + cx,
        y_undist * fy + cy
    ], axis=1)

    return undistorted
