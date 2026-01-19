"""
幾何変換ツール

3D Computer Visionで使用する基本的な幾何変換関数を提供します。
- 回転行列の生成（オイラー角、ロドリゲス変換）
- 座標変換（同次座標系、射影変換）
- 点の正規化
"""

import numpy as np
from typing import Tuple, Union


# ============================================================
# 回転行列の生成
# ============================================================

def rotation_matrix_x(angle: float) -> np.ndarray:
    """
    X軸周りの回転行列を生成

    Parameters
    ----------
    angle : float
        回転角度（ラジアン）

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        回転行列

    Examples
    --------
    >>> R = rotation_matrix_x(np.pi / 2)  # 90度回転
    >>> print(R)
    """
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ])

    return R


def rotation_matrix_y(angle: float) -> np.ndarray:
    """
    Y軸周りの回転行列を生成

    Parameters
    ----------
    angle : float
        回転角度（ラジアン）

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        回転行列
    """
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

    return R


def rotation_matrix_z(angle: float) -> np.ndarray:
    """
    Z軸周りの回転行列を生成

    Parameters
    ----------
    angle : float
        回転角度（ラジアン）

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        回転行列
    """
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

    return R


def euler_to_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
    order: str = 'xyz'
) -> np.ndarray:
    """
    オイラー角から回転行列を生成

    Parameters
    ----------
    roll : float
        X軸周りの回転角度（ラジアン）
    pitch : float
        Y軸周りの回転角度（ラジアン）
    yaw : float
        Z軸周りの回転角度（ラジアン）
    order : str, default='xyz'
        回転の順序（'xyz', 'zyx' など）

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        回転行列

    Examples
    --------
    >>> R = euler_to_rotation_matrix(0, np.pi/4, np.pi/2)
    >>> print(R.shape)
    (3, 3)
    """
    Rx = rotation_matrix_x(roll)
    Ry = rotation_matrix_y(pitch)
    Rz = rotation_matrix_z(yaw)

    if order == 'xyz':
        R = Rz @ Ry @ Rx
    elif order == 'zyx':
        R = Rx @ Ry @ Rz
    else:
        raise ValueError(f"Unsupported rotation order: {order}")

    return R


def rodrigues_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """
    ロドリゲスベクトルから回転行列を生成

    Parameters
    ----------
    rvec : np.ndarray, shape (3,) or (3, 1)
        ロドリゲスベクトル（回転軸 × 回転角度）

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        回転行列

    Notes
    -----
    ロドリゲスの公式:
    R = I + sin(θ) * K + (1 - cos(θ)) * K^2

    ここで、θ = ||rvec||, K は rvec/θ の歪対称行列

    Examples
    --------
    >>> rvec = np.array([0, 0, np.pi/2])  # Z軸周り90度
    >>> R = rodrigues_to_rotation_matrix(rvec)
    >>> print(R)
    """
    rvec = rvec.flatten()

    # 回転角度
    theta = np.linalg.norm(rvec)

    if theta < 1e-10:
        # 回転角度が非常に小さい場合は単位行列
        return np.eye(3)

    # 正規化された回転軸
    k = rvec / theta

    # 歪対称行列（skew-symmetric matrix）
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    # ロドリゲスの公式
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    return R


def rotation_matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """
    回転行列からロドリゲスベクトルを生成

    Parameters
    ----------
    R : np.ndarray, shape (3, 3)
        回転行列

    Returns
    -------
    rvec : np.ndarray, shape (3,)
        ロドリゲスベクトル

    Examples
    --------
    >>> R = rotation_matrix_z(np.pi/2)
    >>> rvec = rotation_matrix_to_rodrigues(R)
    >>> print(rvec)
    """
    # 回転角度
    trace = np.trace(R)
    theta = np.arccos((trace - 1) / 2)

    if theta < 1e-10:
        # 回転角度が非常に小さい場合はゼロベクトル
        return np.zeros(3)

    # 回転軸
    k = 1 / (2 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])

    # ロドリゲスベクトル
    rvec = theta * k

    return rvec


# ============================================================
# 座標変換
# ============================================================

def homogeneous_transform(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """
    3D点を回転・並進変換（同次座標系）

    Parameters
    ----------
    points : np.ndarray, shape (N, 3) or (3, N)
        3D点群
    R : np.ndarray, shape (3, 3)
        回転行列
    t : np.ndarray, shape (3,) or (3, 1)
        並進ベクトル

    Returns
    -------
    transformed_points : np.ndarray, same shape as points
        変換後の3D点群

    Examples
    --------
    >>> points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> R = rotation_matrix_z(np.pi/2)
    >>> t = np.array([1, 2, 3])
    >>> transformed = homogeneous_transform(points, R, t)
    """
    # 入力の形状を確認
    if points.shape[0] == 3 and points.shape[1] != 3:
        # (3, N) 形式
        points_T = points
        transpose_output = True
    else:
        # (N, 3) 形式
        points_T = points.T
        transpose_output = False

    t = t.reshape(3, 1)

    # 変換: P' = R @ P + t
    transformed = R @ points_T + t

    if transpose_output:
        return transformed
    else:
        return transformed.T


def project_points(
    points_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray = None,
    t: np.ndarray = None,
    distortion: np.ndarray = None
) -> np.ndarray:
    """
    3D点を2D画像座標に射影

    Parameters
    ----------
    points_3d : np.ndarray, shape (N, 3)
        3D点群（世界座標系）
    K : np.ndarray, shape (3, 3)
        カメラ内部パラメータ行列
    R : np.ndarray, shape (3, 3), optional
        回転行列（世界→カメラ座標系）
    t : np.ndarray, shape (3,), optional
        並進ベクトル
    distortion : np.ndarray, optional
        歪み係数（未実装）

    Returns
    -------
    points_2d : np.ndarray, shape (N, 2)
        2D画像座標

    Examples
    --------
    >>> points_3d = np.array([[0, 0, 5], [1, 1, 5]])
    >>> K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    >>> points_2d = project_points(points_3d, K)
    """
    # 外部パラメータが指定されていない場合は単位行列・ゼロベクトル
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # 世界座標系からカメラ座標系への変換
    points_cam = homogeneous_transform(points_3d, R, t)

    # 射影（同次座標系）
    points_homogeneous = K @ points_cam.T  # (3, N)

    # 正規化して2D座標へ
    points_2d = points_homogeneous[:2, :] / points_homogeneous[2, :]

    return points_2d.T


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    点群を正規化（平均0、標準偏差√2）

    Parameters
    ----------
    points : np.ndarray, shape (N, 2) or (N, 3)
        2D or 3D点群

    Returns
    -------
    normalized_points : np.ndarray, same shape as points
        正規化された点群
    T : np.ndarray, shape (3, 3) or (4, 4)
        正規化変換行列

    Notes
    -----
    Hartleyの正規化：8点アルゴリズムなどで使用
    数値安定性を向上させる

    Examples
    --------
    >>> points = np.random.rand(100, 2) * 100
    >>> normalized, T = normalize_points(points)
    >>> print(normalized.mean(axis=0))  # ほぼ [0, 0]
    """
    dim = points.shape[1]

    # 重心
    centroid = points.mean(axis=0)

    # 重心からの距離
    distances = np.linalg.norm(points - centroid, axis=1)

    # スケール（平均距離を√2にする）
    scale = np.sqrt(dim) / distances.mean()

    # 正規化
    normalized_points = (points - centroid) * scale

    # 変換行列
    if dim == 2:
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
    else:  # dim == 3
        T = np.array([
            [scale, 0, 0, -scale * centroid[0]],
            [0, scale, 0, -scale * centroid[1]],
            [0, 0, scale, -scale * centroid[2]],
            [0, 0, 0, 1]
        ])

    return normalized_points, T


# ============================================================
# テストとデモ
# ============================================================

def _test_rotation_matrices():
    """回転行列の動作確認"""
    print("=" * 60)
    print("回転行列のテスト")
    print("=" * 60)

    # Z軸周り90度回転
    R = rotation_matrix_z(np.pi / 2)
    print("\nZ軸周り90度回転:")
    print(R)

    # 点 (1, 0, 0) を回転 → (0, 1, 0) になるはず
    p = np.array([1, 0, 0])
    p_rotated = R @ p
    print(f"\n点 {p} を回転 → {p_rotated}")

    # オイラー角からの変換
    R_euler = euler_to_rotation_matrix(0, 0, np.pi / 2)
    print("\nオイラー角から生成した回転行列:")
    print(R_euler)
    print(f"2つの行列は等しい: {np.allclose(R, R_euler)}")

    # ロドリゲス変換
    rvec = np.array([0, 0, np.pi / 2])
    R_rod = rodrigues_to_rotation_matrix(rvec)
    print("\nロドリゲスベクトルから生成した回転行列:")
    print(R_rod)
    print(f"2つの行列は等しい: {np.allclose(R, R_rod)}")

    # 逆変換
    rvec_back = rotation_matrix_to_rodrigues(R)
    print(f"\n回転行列からロドリゲスベクトルに変換: {rvec_back}")
    print(f"元のベクトルと等しい: {np.allclose(rvec, rvec_back)}")


def _test_projection():
    """射影変換の動作確認"""
    print("\n" + "=" * 60)
    print("射影変換のテスト")
    print("=" * 60)

    # カメラ内部パラメータ
    K = np.array([
        [500, 0, 320],  # fx, 0, cx
        [0, 500, 240],   # 0, fy, cy
        [0, 0, 1]
    ])

    # 3D点（カメラ前方5m）
    points_3d = np.array([
        [0, 0, 5],     # 中心
        [1, 0, 5],     # 右
        [0, 1, 5],     # 上
        [-1, 0, 5],    # 左
        [0, -1, 5]     # 下
    ])

    # 射影
    points_2d = project_points(points_3d, K)

    print("\n3D点:")
    print(points_3d)
    print("\n2D投影:")
    print(points_2d)

    # 中心点は画像中心 (320, 240) に投影されるはず
    print(f"\n中心点の投影: {points_2d[0]}")
    print(f"画像中心と一致: {np.allclose(points_2d[0], [320, 240])}")


if __name__ == "__main__":
    _test_rotation_matrices()
    _test_projection()
    print("\n" + "=" * 60)
    print("✅ すべてのテストが完了しました")
    print("=" * 60)
