"""
カメラモデル

ピンホールカメラモデルの実装と関連ユーティリティを提供します。
"""

import numpy as np
from typing import Tuple, Optional
from .geometry_tools import project_points, rodrigues_to_rotation_matrix


def build_intrinsic_matrix(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    skew: float = 0.0
) -> np.ndarray:
    """
    カメラ内部パラメータ行列を構築

    Parameters
    ----------
    fx : float
        X方向の焦点距離（ピクセル単位）
    fy : float
        Y方向の焦点距離（ピクセル単位）
    cx : float
        主点のX座標（ピクセル単位）
    cy : float
        主点のY座標（ピクセル単位）
    skew : float, default=0.0
        歪み係数（通常は0）

    Returns
    -------
    K : np.ndarray, shape (3, 3)
        内部パラメータ行列

    Examples
    --------
    >>> K = build_intrinsic_matrix(500, 500, 320, 240)
    >>> print(K)
    [[500.   0. 320.]
     [  0. 500. 240.]
     [  0.   0.   1.]]
    """
    K = np.array([
        [fx, skew, cx],
        [0,  fy,   cy],
        [0,  0,    1]
    ])
    return K


def build_projection_matrix(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """
    射影行列 P = K[R|t] を構築

    Parameters
    ----------
    K : np.ndarray, shape (3, 3)
        内部パラメータ行列
    R : np.ndarray, shape (3, 3)
        回転行列
    t : np.ndarray, shape (3,) or (3, 1)
        並進ベクトル

    Returns
    -------
    P : np.ndarray, shape (3, 4)
        射影行列

    Examples
    --------
    >>> K = build_intrinsic_matrix(500, 500, 320, 240)
    >>> R = np.eye(3)
    >>> t = np.array([0, 0, 0])
    >>> P = build_projection_matrix(K, R, t)
    >>> print(P.shape)
    (3, 4)
    """
    t = t.reshape(3, 1)
    Rt = np.hstack([R, t])
    P = K @ Rt
    return P


class Camera:
    """
    カメラの基本クラス

    Attributes
    ----------
    K : np.ndarray, shape (3, 3)
        内部パラメータ行列
    R : np.ndarray, shape (3, 3)
        回転行列（世界→カメラ座標系）
    t : np.ndarray, shape (3,)
        並進ベクトル
    width : int
        画像幅（ピクセル）
    height : int
        画像高さ（ピクセル）
    """

    def __init__(
        self,
        K: np.ndarray,
        R: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
        width: int = 640,
        height: int = 480
    ):
        """
        Parameters
        ----------
        K : np.ndarray, shape (3, 3)
            内部パラメータ行列
        R : np.ndarray, shape (3, 3), optional
            回転行列（デフォルト: 単位行列）
        t : np.ndarray, shape (3,), optional
            並進ベクトル（デフォルト: ゼロベクトル）
        width : int, default=640
            画像幅
        height : int, default=480
            画像高さ
        """
        self.K = K
        self.R = R if R is not None else np.eye(3)
        self.t = t if t is not None else np.zeros(3)
        self.width = width
        self.height = height

    @property
    def fx(self) -> float:
        """X方向の焦点距離"""
        return self.K[0, 0]

    @property
    def fy(self) -> float:
        """Y方向の焦点距離"""
        return self.K[1, 1]

    @property
    def cx(self) -> float:
        """主点のX座標"""
        return self.K[0, 2]

    @property
    def cy(self) -> float:
        """主点のY座標"""
        return self.K[1, 2]

    @property
    def center(self) -> np.ndarray:
        """
        カメラ中心の3D座標（世界座標系）

        Returns
        -------
        C : np.ndarray, shape (3,)
            カメラ中心
        """
        # C = -R^T @ t
        return -self.R.T @ self.t

    @property
    def projection_matrix(self) -> np.ndarray:
        """
        射影行列 P = K[R|t]

        Returns
        -------
        P : np.ndarray, shape (3, 4)
            射影行列
        """
        return build_projection_matrix(self.K, self.R, self.t)

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """
        3D点を2D画像座標に射影

        Parameters
        ----------
        points_3d : np.ndarray, shape (N, 3)
            3D点群（世界座標系）

        Returns
        -------
        points_2d : np.ndarray, shape (N, 2)
            2D画像座標
        """
        return project_points(points_3d, self.K, self.R, self.t)

    def set_pose(self, R: np.ndarray, t: np.ndarray):
        """
        カメラポーズを設定

        Parameters
        ----------
        R : np.ndarray, shape (3, 3)
            回転行列
        t : np.ndarray, shape (3,)
            並進ベクトル
        """
        self.R = R
        self.t = t

    def set_pose_from_rodrigues(self, rvec: np.ndarray, tvec: np.ndarray):
        """
        ロドリゲスベクトルからカメラポーズを設定

        Parameters
        ----------
        rvec : np.ndarray, shape (3,)
            ロドリゲスベクトル
        tvec : np.ndarray, shape (3,)
            並進ベクトル
        """
        R = rodrigues_to_rotation_matrix(rvec)
        self.set_pose(R, tvec)

    def __repr__(self) -> str:
        return (
            f"Camera(\n"
            f"  fx={self.fx:.2f}, fy={self.fy:.2f},\n"
            f"  cx={self.cx:.2f}, cy={self.cy:.2f},\n"
            f"  size=({self.width}x{self.height}),\n"
            f"  center={self.center}\n"
            f")"
        )


class PinholeCamera(Camera):
    """
    ピンホールカメラモデル

    Cameraクラスを継承し、より便利なコンストラクタを提供
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int = 640,
        height: int = 480,
        R: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None
    ):
        """
        Parameters
        ----------
        fx : float
            X方向の焦点距離
        fy : float
            Y方向の焦点距離
        cx : float
            主点のX座標
        cy : float
            主点のY座標
        width : int, default=640
            画像幅
        height : int, default=480
            画像高さ
        R : np.ndarray, shape (3, 3), optional
            回転行列
        t : np.ndarray, shape (3,), optional
            並進ベクトル
        """
        K = build_intrinsic_matrix(fx, fy, cx, cy)
        super().__init__(K, R, t, width, height)

    @classmethod
    def from_fov(
        cls,
        fov_degrees: float,
        width: int,
        height: int,
        R: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None
    ) -> 'PinholeCamera':
        """
        視野角（Field of View）からカメラを生成

        Parameters
        ----------
        fov_degrees : float
            水平視野角（度）
        width : int
            画像幅
        height : int
            画像高さ
        R : np.ndarray, optional
            回転行列
        t : np.ndarray, optional
            並進ベクトル

        Returns
        -------
        camera : PinholeCamera
            カメラインスタンス

        Examples
        --------
        >>> cam = PinholeCamera.from_fov(60, 640, 480)
        >>> print(cam)
        """
        # 視野角から焦点距離を計算
        # fx = (width / 2) / tan(fov / 2)
        fov_rad = np.radians(fov_degrees)
        fx = (width / 2) / np.tan(fov_rad / 2)
        fy = fx  # アスペクト比1:1を仮定

        # 主点は画像中心
        cx = width / 2
        cy = height / 2

        return cls(fx, fy, cx, cy, width, height, R, t)


# ============================================================
# テストとデモ
# ============================================================

def _test_camera():
    """カメラクラスの動作確認"""
    print("=" * 60)
    print("カメラクラスのテスト")
    print("=" * 60)

    # カメラの作成
    cam = PinholeCamera(fx=500, fy=500, cx=320, cy=240)
    print("\n作成したカメラ:")
    print(cam)

    # 3D点の射影
    points_3d = np.array([
        [0, 0, 5],
        [1, 0, 5],
        [0, 1, 5]
    ])

    points_2d = cam.project(points_3d)
    print("\n3D点:")
    print(points_3d)
    print("\n2D投影:")
    print(points_2d)

    # 視野角からのカメラ生成
    cam_fov = PinholeCamera.from_fov(60, 640, 480)
    print("\n視野角60度のカメラ:")
    print(cam_fov)

    # カメラポーズの設定
    R = np.eye(3)
    t = np.array([1, 2, 3])
    cam.set_pose(R, t)
    print(f"\nカメラ中心（ポーズ設定後）: {cam.center}")


if __name__ == "__main__":
    _test_camera()
    print("\n" + "=" * 60)
    print("✅ すべてのテストが完了しました")
    print("=" * 60)
