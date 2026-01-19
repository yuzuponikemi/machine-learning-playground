"""
Visualization Tools for 3D Computer Vision
==========================================

カメラ、点群、エピポーラ線などの可視化ユーティリティ
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Tuple, Optional, Union


# ============================================================
# カメラの可視化
# ============================================================

def plot_camera(
    ax: Axes3D,
    pose: np.ndarray,
    scale: float = 1.0,
    color: str = 'blue',
    label: Optional[str] = None
) -> None:
    """
    3Dプロットにカメラを描画

    Parameters:
    -----------
    ax : Axes3D
        matplotlibの3D軸
    pose : np.ndarray (3, 4)
        カメラポーズ行列 [R|t]
    scale : float
        カメラの描画サイズ
    color : str
        カメラの色
    label : str, optional
        ラベル
    """
    R = pose[:3, :3]
    t = pose[:3, 3]

    # カメラ中心
    C = -R.T @ t

    # カメラの軸を描画
    axes_length = scale * 0.5
    for i, (axis_color, axis_name) in enumerate(zip(['red', 'green', 'blue'], ['X', 'Y', 'Z'])):
        axis_dir = R.T[:, i] * axes_length
        ax.quiver(C[0], C[1], C[2], axis_dir[0], axis_dir[1], axis_dir[2],
                  color=axis_color, arrow_length_ratio=0.2, linewidth=1.5)

    # カメラの視錐台（Frustum）を描画
    # 簡略化: ピラミッド形状で表現
    fov = np.pi / 4  # 45度
    aspect = 4 / 3
    near = scale * 0.3

    # 視錐台の4つの角
    half_h = near * np.tan(fov / 2)
    half_w = half_h * aspect

    corners_cam = np.array([
        [-half_w, -half_h, near],
        [half_w, -half_h, near],
        [half_w, half_h, near],
        [-half_w, half_h, near]
    ])

    # ワールド座標に変換
    corners_world = (R.T @ corners_cam.T).T + C

    # カメラ中心から各角への線
    for corner in corners_world:
        ax.plot([C[0], corner[0]], [C[1], corner[1]], [C[2], corner[2]],
                color=color, linewidth=1, alpha=0.7)

    # 視錐台の前面
    verts = [list(corners_world)]
    face = Poly3DCollection(verts, alpha=0.2, facecolor=color, edgecolor=color)
    ax.add_collection3d(face)

    # カメラ中心にマーカー
    ax.scatter([C[0]], [C[1]], [C[2]], c=color, s=50, marker='o', label=label)


def plot_cameras_and_points(
    poses: List[np.ndarray],
    points_3d: Optional[np.ndarray] = None,
    colors: Optional[Union[str, List[str]]] = None,
    point_colors: Optional[np.ndarray] = None,
    camera_scale: float = 1.0,
    figsize: Tuple[int, int] = (12, 10),
    title: str = 'Cameras and 3D Points',
    elev: float = 20,
    azim: float = 45
) -> Tuple[plt.Figure, Axes3D]:
    """
    複数のカメラと3D点群を可視化

    Parameters:
    -----------
    poses : List[np.ndarray]
        カメラポーズのリスト
    points_3d : np.ndarray (N, 3), optional
        3D点群
    colors : str or List[str], optional
        カメラの色
    point_colors : np.ndarray (N, 3), optional
        点の色（RGB、0-1）
    camera_scale : float
        カメラの描画サイズ
    figsize : Tuple[int, int]
        図のサイズ
    title : str
        タイトル
    elev, azim : float
        視点の角度

    Returns:
    --------
    fig : plt.Figure
    ax : Axes3D
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # カメラを描画
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(poses)))
    elif isinstance(colors, str):
        colors = [colors] * len(poses)

    for i, (pose, color) in enumerate(zip(poses, colors)):
        plot_camera(ax, pose, scale=camera_scale, color=color, label=f'Camera {i}')

    # 点群を描画
    if points_3d is not None:
        if point_colors is None:
            ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                      c='gray', s=5, alpha=0.5, label='3D Points')
        else:
            ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                      c=point_colors, s=5, alpha=0.5, label='3D Points')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.view_init(elev=elev, azim=azim)

    # 軸のスケールを揃える
    _set_axes_equal(ax)

    ax.legend()
    plt.tight_layout()

    return fig, ax


def _set_axes_equal(ax: Axes3D) -> None:
    """3Dプロットの軸スケールを等しくする"""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


# ============================================================
# エピポーラ幾何の可視化
# ============================================================

def plot_epipolar_lines(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
    figsize: Tuple[int, int] = (16, 8)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    2つの画像とエピポーラ線を可視化

    Parameters:
    -----------
    img1 : np.ndarray (H, W) or (H, W, 3)
        1枚目の画像
    img2 : np.ndarray (H, W) or (H, W, 3)
        2枚目の画像
    pts1 : np.ndarray (N, 2)
        1枚目の画像上の点
    pts2 : np.ndarray (N, 2)
        2枚目の画像上の点
    F : np.ndarray (3, 3)
        基礎行列
    figsize : Tuple[int, int]
        図のサイズ

    Returns:
    --------
    fig : plt.Figure
    axes : np.ndarray of Axes
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.jet(np.linspace(0, 1, len(pts1)))

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 1枚目の画像
    ax = axes[0]
    ax.imshow(img1, cmap='gray' if img1.ndim == 2 else None)

    # pts2に対応するエピポーラ線を描画（img1上）
    for pt2, color in zip(pts2, colors):
        pt2_h = np.array([pt2[0], pt2[1], 1])
        line = F.T @ pt2_h  # ax + by + c = 0

        # 線の描画（画像境界内）
        x0, x1 = 0, w1
        y0 = (-line[2] - line[0] * x0) / (line[1] + 1e-10)
        y1 = (-line[2] - line[0] * x1) / (line[1] + 1e-10)

        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1, alpha=0.7)

    # 対応点をプロット
    ax.scatter(pts1[:, 0], pts1[:, 1], c=colors, s=50, marker='o', edgecolors='white')
    ax.set_title('Image 1 with Epipolar Lines', fontsize=12, fontweight='bold')
    ax.axis('off')

    # 2枚目の画像
    ax = axes[1]
    ax.imshow(img2, cmap='gray' if img2.ndim == 2 else None)

    # pts1に対応するエピポーラ線を描画（img2上）
    for pt1, color in zip(pts1, colors):
        pt1_h = np.array([pt1[0], pt1[1], 1])
        line = F @ pt1_h  # ax + by + c = 0

        x0, x1 = 0, w2
        y0 = (-line[2] - line[0] * x0) / (line[1] + 1e-10)
        y1 = (-line[2] - line[0] * x1) / (line[1] + 1e-10)

        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1, alpha=0.7)

    ax.scatter(pts2[:, 0], pts2[:, 1], c=colors, s=50, marker='o', edgecolors='white')
    ax.set_title('Image 2 with Epipolar Lines', fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    return fig, axes


def plot_correspondences(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    inliers: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (16, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    2つの画像間の対応点を可視化

    Parameters:
    -----------
    img1, img2 : np.ndarray
        画像
    pts1, pts2 : np.ndarray (N, 2)
        対応点
    inliers : np.ndarray (N,), optional
        インライアのマスク
    figsize : Tuple[int, int]
        図のサイズ
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 画像を横に並べる
    h = max(h1, h2)
    combined = np.zeros((h, w1 + w2, 3), dtype=np.uint8)

    if img1.ndim == 2:
        combined[:h1, :w1] = np.stack([img1] * 3, axis=-1)
    else:
        combined[:h1, :w1] = img1

    if img2.ndim == 2:
        combined[:h2, w1:] = np.stack([img2] * 3, axis=-1)
    else:
        combined[:h2, w1:] = img2

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(combined)

    # 対応線を描画
    if inliers is None:
        inliers = np.ones(len(pts1), dtype=bool)

    for i, (pt1, pt2, is_inlier) in enumerate(zip(pts1, pts2, inliers)):
        color = 'green' if is_inlier else 'red'
        alpha = 0.8 if is_inlier else 0.3

        ax.plot([pt1[0], pt2[0] + w1], [pt1[1], pt2[1]],
                color=color, linewidth=1, alpha=alpha)
        ax.scatter([pt1[0]], [pt1[1]], c=color, s=20, alpha=alpha)
        ax.scatter([pt2[0] + w1], [pt2[1]], c=color, s=20, alpha=alpha)

    ax.set_title(f'Correspondences: {inliers.sum()} inliers / {len(inliers)} total',
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    return fig, ax


# ============================================================
# 深度マップ・視差マップの可視化
# ============================================================

def plot_depth_map(
    depth: np.ndarray,
    image: Optional[np.ndarray] = None,
    colormap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (14, 6),
    title: str = 'Depth Map'
) -> Tuple[plt.Figure, np.ndarray]:
    """
    深度マップを可視化

    Parameters:
    -----------
    depth : np.ndarray (H, W)
        深度マップ
    image : np.ndarray (H, W) or (H, W, 3), optional
        元画像
    colormap : str
        カラーマップ
    vmin, vmax : float, optional
        表示範囲
    figsize : Tuple[int, int]
        図のサイズ
    title : str
        タイトル

    Returns:
    --------
    fig : plt.Figure
    axes : np.ndarray of Axes
    """
    if image is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].imshow(image, cmap='gray' if image.ndim == 2 else None)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        im = axes[1].imshow(depth, cmap=colormap, vmin=vmin, vmax=vmax)
        axes[1].set_title(title, fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        fig, ax = plt.subplots(figsize=(figsize[0]//2, figsize[1]))
        axes = np.array([ax])

        im = ax.imshow(depth, cmap=colormap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    return fig, axes


# ============================================================
# 特徴点の可視化
# ============================================================

def plot_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    figsize: Tuple[int, int] = (10, 8),
    marker_size: int = 10,
    color: str = 'red',
    title: str = 'Keypoints'
) -> Tuple[plt.Figure, plt.Axes]:
    """
    画像上の特徴点を可視化

    Parameters:
    -----------
    image : np.ndarray
        画像
    keypoints : np.ndarray (N, 2)
        特徴点座標
    figsize : Tuple[int, int]
        図のサイズ
    marker_size : int
        マーカーサイズ
    color : str
        マーカーの色
    title : str
        タイトル

    Returns:
    --------
    fig : plt.Figure
    ax : plt.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
    ax.scatter(keypoints[:, 0], keypoints[:, 1],
               c=color, s=marker_size, marker='o', alpha=0.7)
    ax.set_title(f'{title} ({len(keypoints)} points)', fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    return fig, ax


# ============================================================
# 光線の可視化
# ============================================================

def plot_rays(
    ax: Axes3D,
    origins: np.ndarray,
    directions: np.ndarray,
    length: float = 1.0,
    color: str = 'blue',
    alpha: float = 0.3,
    subsample: int = 1
) -> None:
    """
    3Dプロットに光線を描画

    Parameters:
    -----------
    ax : Axes3D
        matplotlibの3D軸
    origins : np.ndarray (N, 3)
        光線の原点
    directions : np.ndarray (N, 3)
        光線の方向
    length : float
        光線の長さ
    color : str
        光線の色
    alpha : float
        透明度
    subsample : int
        サブサンプリング間隔
    """
    origins = origins[::subsample]
    directions = directions[::subsample]

    for o, d in zip(origins, directions):
        endpoint = o + length * d
        ax.plot([o[0], endpoint[0]], [o[1], endpoint[1]], [o[2], endpoint[2]],
                color=color, alpha=alpha, linewidth=0.5)


def plot_sampled_points_on_ray(
    ax: Axes3D,
    origin: np.ndarray,
    direction: np.ndarray,
    near: float,
    far: float,
    num_samples: int,
    color: str = 'red',
    ray_color: str = 'blue'
) -> None:
    """
    光線上のサンプル点を可視化

    Parameters:
    -----------
    ax : Axes3D
        matplotlibの3D軸
    origin : np.ndarray (3,)
        光線の原点
    direction : np.ndarray (3,)
        光線の方向
    near, far : float
        サンプリング範囲
    num_samples : int
        サンプル数
    color : str
        サンプル点の色
    ray_color : str
        光線の色
    """
    # 光線を描画
    endpoint = origin + far * direction
    ax.plot([origin[0], endpoint[0]], [origin[1], endpoint[1]], [origin[2], endpoint[2]],
            color=ray_color, linewidth=2, alpha=0.5)

    # サンプル点を生成・描画
    t_vals = np.linspace(near, far, num_samples)
    points = origin[None, :] + t_vals[:, None] * direction[None, :]

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=color, s=30, marker='o', alpha=0.8)

    # near/farの境界を表示
    near_point = origin + near * direction
    far_point = origin + far * direction
    ax.scatter([near_point[0]], [near_point[1]], [near_point[2]],
               c='green', s=100, marker='^', label='near')
    ax.scatter([far_point[0]], [far_point[1]], [far_point[2]],
               c='purple', s=100, marker='v', label='far')
