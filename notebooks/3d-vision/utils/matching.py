"""
特徴点マッチングツール

SIFT/ORBなどの特徴点検出・マッチングと、RANSACによる外れ値除去を提供します。
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional


def detect_and_compute(
    image: np.ndarray,
    method: str = 'sift',
    **kwargs
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    特徴点の検出と記述子の計算

    Parameters
    ----------
    image : np.ndarray
        入力画像（グレースケールまたはカラー）
    method : str, default='sift'
        使用する特徴量検出器（'sift', 'orb', 'akaze'）
    **kwargs
        各検出器の追加パラメータ

    Returns
    -------
    keypoints : list of cv2.KeyPoint
        検出された特徴点
    descriptors : np.ndarray, shape (N, D)
        特徴量記述子

    Examples
    --------
    >>> img = cv2.imread('image.jpg')
    >>> kp, desc = detect_and_compute(img, method='sift')
    >>> print(f"検出された特徴点数: {len(kp)}")
    """
    # グレースケール変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 特徴量検出器の選択
    if method.lower() == 'sift':
        detector = cv2.SIFT_create(**kwargs)
    elif method.lower() == 'orb':
        detector = cv2.ORB_create(**kwargs)
    elif method.lower() == 'akaze':
        detector = cv2.AKAZE_create(**kwargs)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # 検出と記述
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    return keypoints, descriptors


def match_features(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    method: str = 'bf',
    ratio_test: float = 0.75,
    cross_check: bool = False,
    **kwargs
) -> List[cv2.DMatch]:
    """
    特徴量記述子のマッチング

    Parameters
    ----------
    descriptors1 : np.ndarray
        画像1の記述子
    descriptors2 : np.ndarray
        画像2の記述子
    method : str, default='bf'
        マッチング手法（'bf': ブルートフォース, 'flann': FLANN）
    ratio_test : float, default=0.75
        Lowe's ratio testの閾値（0-1）
    cross_check : bool, default=False
        クロスチェックを有効にするか
    **kwargs
        マッチャーの追加パラメータ

    Returns
    -------
    good_matches : list of cv2.DMatch
        良好なマッチ

    Examples
    --------
    >>> kp1, desc1 = detect_and_compute(img1)
    >>> kp2, desc2 = detect_and_compute(img2)
    >>> matches = match_features(desc1, desc2, ratio_test=0.75)
    >>> print(f"マッチ数: {len(matches)}")
    """
    # マッチャーの選択
    if method.lower() == 'bf':
        # ブルートフォースマッチング
        # SIFT/SURFはL2ノルム、ORB/AKAZEはハミング距離
        if descriptors1.dtype == np.uint8:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)

    elif method.lower() == 'flann':
        # FLANNマッチング
        if descriptors1.dtype == np.uint8:
            # ORB/AKAZE用
            index_params = dict(
                algorithm=6,  # FLANN_INDEX_LSH
                table_number=6,
                key_size=12,
                multi_probe_level=1
            )
        else:
            # SIFT/SURF用
            index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE

        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # マッチング
    if cross_check:
        # クロスチェック有効の場合、1対1マッチング
        matches = matcher.match(descriptors1, descriptors2)
        good_matches = matches
    else:
        # k-NNマッチング（k=2）でLowe's ratio test
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_test * n.distance:
                    good_matches.append(m)

    return good_matches


def ransac_fundamental_matrix(
    pts1: np.ndarray,
    pts2: np.ndarray,
    threshold: float = 3.0,
    confidence: float = 0.99,
    max_iters: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSACによる基礎行列の推定と外れ値除去

    Parameters
    ----------
    pts1 : np.ndarray, shape (N, 2)
        画像1の点
    pts2 : np.ndarray, shape (N, 2)
        画像2の対応点
    threshold : float, default=3.0
        インライア判定の閾値（ピクセル単位）
    confidence : float, default=0.99
        信頼度（0-1）
    max_iters : int, default=2000
        最大反復回数

    Returns
    -------
    F : np.ndarray, shape (3, 3)
        基礎行列
    mask : np.ndarray, shape (N,)
        インライアマスク（1: インライア, 0: 外れ値）

    Examples
    --------
    >>> F, mask = ransac_fundamental_matrix(pts1, pts2, threshold=1.0)
    >>> inliers1 = pts1[mask.ravel() == 1]
    >>> inliers2 = pts2[mask.ravel() == 1]
    """
    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        cv2.FM_RANSAC,
        ransacReprojThreshold=threshold,
        confidence=confidence,
        maxIters=max_iters
    )

    return F, mask


def ransac_homography(
    pts1: np.ndarray,
    pts2: np.ndarray,
    threshold: float = 5.0,
    confidence: float = 0.99,
    max_iters: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSACによるホモグラフィ行列の推定と外れ値除去

    Parameters
    ----------
    pts1 : np.ndarray, shape (N, 2)
        画像1の点
    pts2 : np.ndarray, shape (N, 2)
        画像2の対応点
    threshold : float, default=5.0
        インライア判定の閾値（ピクセル単位）
    confidence : float, default=0.99
        信頼度（0-1）
    max_iters : int, default=2000
        最大反復回数

    Returns
    -------
    H : np.ndarray, shape (3, 3)
        ホモグラフィ行列
    mask : np.ndarray, shape (N,)
        インライアマスク（1: インライア, 0: 外れ値）

    Examples
    --------
    >>> H, mask = ransac_homography(pts1, pts2, threshold=5.0)
    >>> inliers1 = pts1[mask.ravel() == 1]
    """
    H, mask = cv2.findHomography(
        pts1, pts2,
        cv2.RANSAC,
        ransacReprojThreshold=threshold,
        confidence=confidence,
        maxIters=max_iters
    )

    return H, mask


def draw_matches(
    img1: np.ndarray,
    kp1: List[cv2.KeyPoint],
    img2: np.ndarray,
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    max_matches: int = 50,
    **kwargs
) -> np.ndarray:
    """
    マッチング結果を描画

    Parameters
    ----------
    img1 : np.ndarray
        画像1
    kp1 : list of cv2.KeyPoint
        画像1の特徴点
    img2 : np.ndarray
        画像2
    kp2 : list of cv2.KeyPoint
        画像2の特徴点
    matches : list of cv2.DMatch
        マッチ
    max_matches : int, default=50
        描画する最大マッチ数
    **kwargs
        cv2.drawMatchesの追加パラメータ

    Returns
    -------
    img_matches : np.ndarray
        マッチング結果画像

    Examples
    --------
    >>> img_matches = draw_matches(img1, kp1, img2, kp2, matches)
    >>> cv2.imshow('Matches', img_matches)
    >>> cv2.waitKey(0)
    """
    # 距離でソートして上位max_matches個を選択
    matches_sorted = sorted(matches, key=lambda x: x.distance)
    matches_to_draw = matches_sorted[:max_matches]

    # デフォルトパラメータ
    default_kwargs = {
        'matchColor': (0, 255, 0),
        'singlePointColor': (255, 0, 0),
        'flags': cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    }
    default_kwargs.update(kwargs)

    # マッチング結果の描画
    img_matches = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches_to_draw,
        None,
        **default_kwargs
    )

    return img_matches


def extract_matched_points(
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    マッチング結果から対応点座標を抽出

    Parameters
    ----------
    kp1 : list of cv2.KeyPoint
        画像1の特徴点
    kp2 : list of cv2.KeyPoint
        画像2の特徴点
    matches : list of cv2.DMatch
        マッチ

    Returns
    -------
    pts1 : np.ndarray, shape (N, 2)
        画像1の対応点座標
    pts2 : np.ndarray, shape (N, 2)
        画像2の対応点座標

    Examples
    --------
    >>> pts1, pts2 = extract_matched_points(kp1, kp2, matches)
    >>> print(f"対応点数: {len(pts1)}")
    """
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    return pts1, pts2


# ============================================================
# テストとデモ
# ============================================================

def _test_matching():
    """マッチングのテスト（合成データ）"""
    print("=" * 60)
    print("特徴点マッチングのテスト")
    print("=" * 60)

    # 合成画像の作成（チェスボード）
    img1 = np.zeros((480, 640), dtype=np.uint8)
    for i in range(0, 480, 60):
        for j in range(0, 640, 80):
            if (i // 60 + j // 80) % 2 == 0:
                img1[i:i+60, j:j+80] = 255

    # 画像2は画像1を少し移動
    img2 = np.zeros_like(img1)
    img2[20:, 30:] = img1[:-20, :-30]

    print("\n合成画像を作成しました")
    print(f"画像1のサイズ: {img1.shape}")
    print(f"画像2のサイズ: {img2.shape}")

    # 特徴点検出
    kp1, desc1 = detect_and_compute(img1, method='orb', nfeatures=500)
    kp2, desc2 = detect_and_compute(img2, method='orb', nfeatures=500)

    print(f"\n特徴点数:")
    print(f"  画像1: {len(kp1)}")
    print(f"  画像2: {len(kp2)}")

    # マッチング
    matches = match_features(desc1, desc2, method='bf', ratio_test=0.75)
    print(f"\nマッチ数: {len(matches)}")

    # 対応点の抽出
    pts1, pts2 = extract_matched_points(kp1, kp2, matches)

    # RANSAC
    if len(pts1) >= 8:
        F, mask = ransac_fundamental_matrix(pts1, pts2, threshold=3.0)
        n_inliers = mask.sum()
        print(f"\nRANSAC後のインライア数: {n_inliers}")
        print(f"インライア率: {n_inliers / len(pts1) * 100:.1f}%")
    else:
        print("\n⚠️ マッチ数が不足しているため、RANSACをスキップしました")


if __name__ == "__main__":
    _test_matching()
    print("\n" + "=" * 60)
    print("✅ すべてのテストが完了しました")
    print("=" * 60)
