"""
3D可視化ツール

カメラ、点群、エピポーラ線などの3D可視化機能を提供します。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Tuple

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠️ Open3Dがインストールされていません。一部の機能が制限されます。")


class Visualizer3D:
    """
    3D可視化クラス（Open3Dベース）

    カメラ、点群、メッシュなどを3Dで可視化します。
    """

    def __init__(self):
        """Open3D可視化ウィンドウの初期化"""
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3Dがインストールされていません")

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.geometries = []

    def add_camera(
        self,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        scale: float = 0.5,
        color: Tuple[float, float, float] = (1, 0, 0)
    ):
        """
        カメラを追加

        Parameters
        ----------
        K : np.ndarray, shape (3, 3)
            内部パラメータ行列
        R : np.ndarray, shape (3, 3)
            回転行列
        t : np.ndarray, shape (3,)
            並進ベクトル
        scale : float, default=0.5
            カメラの表示サイズ
        color : tuple, default=(1, 0, 0)
            カメラの色（RGB）
        """
        # カメラ中心
        C = -R.T @ t

        # カメラの向き（Z軸方向）
        z_axis = R.T @ np.array([0, 0, 1])

        # カメラフレームの作成
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=scale, origin=C
        )

        # カメラの向きに回転
        camera_frame.rotate(R.T, center=C)

        # 視錐台（frustum）の作成（簡略版）
        # TODO: より詳細な視錐台の実装

        self.vis.add_geometry(camera_frame)
        self.geometries.append(camera_frame)

    def add_points(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        size: float = 1.0
    ):
        """
        点群を追加

        Parameters
        ----------
        points : np.ndarray, shape (N, 3)
            3D点群
        colors : np.ndarray, shape (N, 3), optional
            各点の色（RGB, 0-1の範囲）
        size : float, default=1.0
            点のサイズ（未使用）
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        self.vis.add_geometry(pcd)
        self.geometries.append(pcd)

    def add_line_set(
        self,
        points: np.ndarray,
        lines: np.ndarray,
        colors: Optional[np.ndarray] = None
    ):
        """
        線分集合を追加

        Parameters
        ----------
        points : np.ndarray, shape (N, 3)
            端点の座標
        lines : np.ndarray, shape (M, 2)
            線分のインデックス（各行が [始点index, 終点index]）
        colors : np.ndarray, shape (M, 3), optional
            各線分の色（RGB）
        """
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        if colors is not None:
            line_set.colors = o3d.utility.Vector3dVector(colors)

        self.vis.add_geometry(line_set)
        self.geometries.append(line_set)

    def show(self):
        """可視化ウィンドウを表示"""
        self.vis.run()
        self.vis.destroy_window()


# ============================================================
# Matplotlibベースの可視化関数
# ============================================================

def plot_camera(
    ax: Axes3D,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray = None,
    scale: float = 0.5,
    color: str = 'r',
    label: Optional[str] = None
):
    """
    Matplotlibでカメラを描画

    Parameters
    ----------
    ax : Axes3D
        3Dプロット用のaxes
    R : np.ndarray, shape (3, 3)
        回転行列
    t : np.ndarray, shape (3,)
        並進ベクトル
    K : np.ndarray, optional
        内部パラメータ行列（未使用）
    scale : float, default=0.5
        カメラのサイズ
    color : str, default='r'
        カメラの色
    label : str, optional
        ラベル

    Examples
    --------
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> R = np.eye(3)
    >>> t = np.array([0, 0, 0])
    >>> plot_camera(ax, R, t)
    >>> plt.show()
    """
    # カメラ中心
    C = -R.T @ t

    # カメラの座標軸
    axes = R.T * scale
    x_axis = axes[:, 0]
    y_axis = axes[:, 1]
    z_axis = axes[:, 2]

    # 座標軸の描画
    ax.quiver(C[0], C[1], C[2], x_axis[0], x_axis[1], x_axis[2],
              color='red', arrow_length_ratio=0.1)
    ax.quiver(C[0], C[1], C[2], y_axis[0], y_axis[1], y_axis[2],
              color='green', arrow_length_ratio=0.1)
    ax.quiver(C[0], C[1], C[2], z_axis[0], z_axis[1], z_axis[2],
              color='blue', arrow_length_ratio=0.1)

    # カメラ中心をプロット
    ax.scatter(C[0], C[1], C[2], c=color, s=100, marker='o', label=label)


def plot_points_3d(
    ax: Axes3D,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    size: float = 20,
    alpha: float = 0.6,
    label: Optional[str] = None
):
    """
    3D点群を描画

    Parameters
    ----------
    ax : Axes3D
        3Dプロット用のaxes
    points : np.ndarray, shape (N, 3)
        3D点群
    colors : np.ndarray or str, optional
        点の色
    size : float, default=20
        点のサイズ
    alpha : float, default=0.6
        透明度
    label : str, optional
        ラベル
    """
    if colors is None:
        colors = 'blue'

    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors, s=size, alpha=alpha, label=label
    )


def plot_epipolar_lines(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
    n_lines: int = 10
) -> Tuple[plt.Figure, plt.Axes]:
    """
    エピポーラ線を描画

    Parameters
    ----------
    img1 : np.ndarray
        画像1
    img2 : np.ndarray
        画像2
    pts1 : np.ndarray, shape (N, 2)
        画像1の点
    pts2 : np.ndarray, shape (N, 2)
        画像2の対応点
    F : np.ndarray, shape (3, 3)
        基礎行列
    n_lines : int, default=10
        描画するエピポーラ線の数

    Returns
    -------
    fig : Figure
        matplotlib Figure
    axes : array of Axes
        matplotlib Axes（2つ）

    Examples
    --------
    >>> fig, axes = plot_epipolar_lines(img1, img2, pts1, pts2, F, n_lines=5)
    >>> plt.show()
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ランダムにn_lines個の点を選択
    n_pts = len(pts1)
    indices = np.random.choice(n_pts, min(n_lines, n_pts), replace=False)

    # 画像の表示
    axes[0].imshow(img1)
    axes[0].set_title('Image 1', fontsize=14, fontweight='bold')

    axes[1].imshow(img2)
    axes[1].set_title('Image 2', fontsize=14, fontweight='bold')

    # 各点に対してエピポーラ線を描画
    colors = plt.cm.rainbow(np.linspace(0, 1, len(indices)))

    for i, idx in enumerate(indices):
        color = colors[i]

        # 画像1の点
        pt1 = pts1[idx]
        axes[0].scatter(pt1[0], pt1[1], c=[color], s=100, marker='o')

        # 画像2のエピポーラ線: l' = F @ p
        pt1_homogeneous = np.array([pt1[0], pt1[1], 1])
        epiline = F @ pt1_homogeneous

        # エピポーラ線を画像2に描画: ax + by + c = 0
        # y = -(ax + c) / b
        h, w = img2.shape[:2]
        x = np.array([0, w])
        y = -(epiline[0] * x + epiline[2]) / epiline[1]

        axes[1].plot(x, y, color=color, linewidth=2, alpha=0.7)

        # 画像2の対応点
        pt2 = pts2[idx]
        axes[1].scatter(pt2[0], pt2[1], c=[color], s=100, marker='o')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    return fig, axes


def setup_3d_plot(
    figsize: Tuple[int, int] = (10, 8),
    title: str = "3D Visualization"
) -> Tuple[plt.Figure, Axes3D]:
    """
    3Dプロット用のFigureとAxesを準備

    Parameters
    ----------
    figsize : tuple, default=(10, 8)
        図のサイズ
    title : str, default="3D Visualization"
        タイトル

    Returns
    -------
    fig : Figure
        matplotlib Figure
    ax : Axes3D
        3D Axes

    Examples
    --------
    >>> fig, ax = setup_3d_plot()
    >>> ax.scatter([0, 1], [0, 1], [0, 1])
    >>> plt.show()
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)

    # 等しいアスペクト比（できる範囲で）
    ax.set_box_aspect([1, 1, 1])

    return fig, ax


# ============================================================
# テストとデモ
# ============================================================

def _test_matplotlib_visualization():
    """Matplotlib可視化のテスト"""
    print("=" * 60)
    print("Matplotlib可視化のテスト")
    print("=" * 60)

    fig, ax = setup_3d_plot(title="カメラと点群の可視化")

    # カメラ1（原点）
    R1 = np.eye(3)
    t1 = np.zeros(3)
    plot_camera(ax, R1, t1, scale=1.0, color='red', label='Camera 1')

    # カメラ2（右に移動）
    R2 = np.eye(3)
    t2 = np.array([2, 0, 0])
    plot_camera(ax, R2, t2, scale=1.0, color='blue', label='Camera 2')

    # 3D点群
    points = np.random.randn(50, 3) * 0.5 + np.array([0, 0, 5])
    plot_points_3d(ax, points, colors='green', size=30, label='Points')

    ax.legend()
    ax.set_xlim([-2, 4])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 8])

    plt.savefig('/tmp/test_3d_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✅ 3D可視化を /tmp/test_3d_visualization.png に保存しました")
    plt.close()


def _test_open3d_visualization():
    """Open3D可視化のテスト"""
    if not OPEN3D_AVAILABLE:
        print("\n⚠️ Open3Dが利用できないため、このテストはスキップされます")
        return

    print("\n" + "=" * 60)
    print("Open3D可視化のテスト")
    print("=" * 60)

    vis = Visualizer3D()

    # カメラ1
    K1 = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    R1 = np.eye(3)
    t1 = np.zeros(3)
    vis.add_camera(K1, R1, t1, scale=0.5, color=(1, 0, 0))

    # カメラ2
    R2 = np.eye(3)
    t2 = np.array([2, 0, 0])
    vis.add_camera(K1, R2, t2, scale=0.5, color=(0, 0, 1))

    # 点群
    points = np.random.randn(100, 3) * 0.5 + np.array([0, 0, 5])
    colors = np.random.rand(100, 3)
    vis.add_points(points, colors=colors)

    print("\n✅ Open3Dウィンドウを開きます（閉じるまで待機）")
    vis.show()


if __name__ == "__main__":
    _test_matplotlib_visualization()
    # _test_open3d_visualization()  # インタラクティブなので手動実行推奨

    print("\n" + "=" * 60)
    print("✅ すべてのテストが完了しました")
    print("=" * 60)
