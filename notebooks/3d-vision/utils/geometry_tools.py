"""
Geometry Tools for 3D Computer Vision
=====================================

回転行列、座標変換、幾何計算のユーティリティ関数群
"""

import numpy as np
from typing import Union, Tuple


# ============================================================
# 回転行列
# ============================================================

def rotation_matrix_x(theta: float) -> np.ndarray:
    """
    X軸周りの回転行列を生成

    Parameters:
    -----------
    theta : float
        回転角度（ラジアン）

    Returns:
    --------
    R : np.ndarray (3, 3)
        回転行列
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])


def rotation_matrix_y(theta: float) -> np.ndarray:
    """
    Y軸周りの回転行列を生成

    Parameters:
    -----------
    theta : float
        回転角度（ラジアン）

    Returns:
    --------
    R : np.ndarray (3, 3)
        回転行列
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_z(theta: float) -> np.ndarray:
    """
    Z軸周りの回転行列を生成

    Parameters:
    -----------
    theta : float
        回転角度（ラジアン）

    Returns:
    --------
    R : np.ndarray (3, 3)
        回転行列
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float,
                             order: str = 'xyz') -> np.ndarray:
    """
    オイラー角から回転行列を生成

    Parameters:
    -----------
    roll : float
        X軸周りの回転（ラジアン）
    pitch : float
        Y軸周りの回転（ラジアン）
    yaw : float
        Z軸周りの回転（ラジアン）
    order : str
        回転の適用順序 ('xyz', 'zyx', etc.)

    Returns:
    --------
    R : np.ndarray (3, 3)
        回転行列
    """
    Rx = rotation_matrix_x(roll)
    Ry = rotation_matrix_y(pitch)
    Rz = rotation_matrix_z(yaw)

    rotation_map = {'x': Rx, 'y': Ry, 'z': Rz}

    R = np.eye(3)
    for axis in order:
        R = rotation_map[axis] @ R

    return R


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    ベクトルから歪対称行列（反対称行列）を生成

    [v]× = | 0   -v_z  v_y |
           | v_z  0   -v_x |
           |-v_y  v_x  0   |

    Parameters:
    -----------
    v : np.ndarray (3,)
        3次元ベクトル

    Returns:
    --------
    skew : np.ndarray (3, 3)
        歪対称行列
    """
    v = np.asarray(v).flatten()
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    """
    ロドリゲスの公式による回転行列

    R = I + sin(θ)[k]× + (1-cos(θ))[k]×²

    Parameters:
    -----------
    axis : np.ndarray (3,)
        回転軸（正規化される）
    theta : float
        回転角度（ラジアン）

    Returns:
    --------
    R : np.ndarray (3, 3)
        回転行列
    """
    axis = np.asarray(axis).flatten()
    axis = axis / np.linalg.norm(axis)  # 正規化

    K = skew_symmetric(axis)
    I = np.eye(3)

    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def rodrigues_vector(rvec: np.ndarray) -> np.ndarray:
    """
    回転ベクトル（軸・角度表現）から回転行列を生成

    回転ベクトル: rvec = θ * k （ノルムが角度、方向が軸）

    Parameters:
    -----------
    rvec : np.ndarray (3,)
        回転ベクトル

    Returns:
    --------
    R : np.ndarray (3, 3)
        回転行列
    """
    rvec = np.asarray(rvec).flatten()
    theta = np.linalg.norm(rvec)

    if theta < 1e-10:
        return np.eye(3)

    axis = rvec / theta
    return rodrigues(axis, theta)


def rotation_matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """
    回転行列から回転ベクトルを抽出

    Parameters:
    -----------
    R : np.ndarray (3, 3)
        回転行列

    Returns:
    --------
    rvec : np.ndarray (3,)
        回転ベクトル
    """
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    if theta < 1e-10:
        return np.zeros(3)

    if np.abs(theta - np.pi) < 1e-6:
        # θ ≈ π の特殊ケース
        eigvals, eigvecs = np.linalg.eig(R)
        idx = np.argmin(np.abs(eigvals - 1))
        axis = np.real(eigvecs[:, idx])
    else:
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(theta))

    return theta * axis


# ============================================================
# 同次座標変換
# ============================================================

def to_homogeneous(points: np.ndarray) -> np.ndarray:
    """
    点を同次座標に変換

    Parameters:
    -----------
    points : np.ndarray (N, D) or (D,)
        D次元の点（群）

    Returns:
    --------
    points_h : np.ndarray (N, D+1) or (D+1,)
        同次座標
    """
    points = np.asarray(points)
    if points.ndim == 1:
        return np.append(points, 1)
    else:
        ones = np.ones((points.shape[0], 1))
        return np.hstack([points, ones])


def from_homogeneous(points_h: np.ndarray) -> np.ndarray:
    """
    同次座標から通常座標に変換

    Parameters:
    -----------
    points_h : np.ndarray (N, D+1) or (D+1,)
        同次座標

    Returns:
    --------
    points : np.ndarray (N, D) or (D,)
        通常座標
    """
    points_h = np.asarray(points_h)
    if points_h.ndim == 1:
        return points_h[:-1] / points_h[-1]
    else:
        return points_h[:, :-1] / points_h[:, -1:]


# ============================================================
# 変換行列
# ============================================================

def create_transformation_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    回転行列と並進ベクトルから4x4変換行列を生成

    T = | R  t |
        | 0  1 |

    Parameters:
    -----------
    R : np.ndarray (3, 3)
        回転行列
    t : np.ndarray (3,) or (3, 1)
        並進ベクトル

    Returns:
    --------
    T : np.ndarray (4, 4)
        変換行列
    """
    t = np.asarray(t).flatten()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def decompose_transformation_matrix(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    4x4変換行列から回転行列と並進ベクトルを抽出

    Parameters:
    -----------
    T : np.ndarray (4, 4)
        変換行列

    Returns:
    --------
    R : np.ndarray (3, 3)
        回転行列
    t : np.ndarray (3,)
        並進ベクトル
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def inverse_transformation(T: np.ndarray) -> np.ndarray:
    """
    変換行列の逆行列を計算

    T^{-1} = | R^T  -R^T t |
             | 0       1   |

    Parameters:
    -----------
    T : np.ndarray (4, 4)
        変換行列

    Returns:
    --------
    T_inv : np.ndarray (4, 4)
        逆変換行列
    """
    R, t = decompose_transformation_matrix(T)
    R_inv = R.T
    t_inv = -R_inv @ t
    return create_transformation_matrix(R_inv, t_inv)
