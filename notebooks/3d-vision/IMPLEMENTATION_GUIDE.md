# 3D Computer Vision 実装ガイド

このドキュメントは、3D Computer Visionカリキュラムの実装を開始するためのステップバイステップガイドです。

---

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# プロジェクトルートに移動
cd /path/to/machine-learning-playground

# 3D Vision用ライブラリのインストール
pip install -r requirements-3dvision.txt

# または、uvを使用している場合
uv pip install -r requirements-3dvision.txt
```

### 2. ユーティリティのテスト

```bash
# 3d-visionディレクトリに移動
cd notebooks/3d-vision

# geometry_tools.pyのテスト
python utils/geometry_tools.py

# camera.pyのテスト
python utils/camera.py

# visualizer.pyのテスト
python utils/visualizer.py

# matching.pyのテスト
python utils/matching.py
```

すべてのテストが成功すれば、環境セットアップは完了です！

---

## 📚 学習の進め方

### Phase 1: カメラモデルと射影幾何

#### Notebook 50: ピンホールカメラモデルの基礎

**目標**: カメラの基本原理を理解し、3D→2D射影を実装する

**重要な概念**:
- 焦点距離（focal length）
- 主点（principal point）
- 内部パラメータ行列 K

**実装の流れ**:
1. カメラ内部パラメータ行列 K の構築
2. 3D点から2D画像座標への射影
3. 焦点距離の影響の可視化
4. レンズ歪みのシミュレーション

**コード例**:
```python
import numpy as np
from utils.camera import PinholeCamera
from utils.visualizer import setup_3d_plot, plot_points_3d
import matplotlib.pyplot as plt

# カメラの作成
camera = PinholeCamera(fx=500, fy=500, cx=320, cy=240)

# 3D点（カメラ前方5m）
points_3d = np.array([
    [0, 0, 5],
    [1, 0, 5],
    [0, 1, 5]
])

# 2D投影
points_2d = camera.project(points_3d)
print("2D投影:")
print(points_2d)
```

---

#### Notebook 51: カメラ外部パラメータと座標変換

**目標**: 回転・並進変換を理解し、異なる座標系間の変換を実装する

**重要な概念**:
- 回転行列 R
- 並進ベクトル t
- ロドリゲスの公式

**実装の流れ**:
1. 回転行列の生成（X, Y, Z軸周り）
2. ロドリゲス変換の実装
3. 座標変換の可視化
4. 複数カメラの相対姿勢

**コード例**:
```python
from utils.geometry_tools import (
    rotation_matrix_z,
    rodrigues_to_rotation_matrix,
    homogeneous_transform
)

# Z軸周り90度回転
R = rotation_matrix_z(np.pi / 2)
t = np.array([1, 2, 3])

# 3D点の変換
points = np.array([[1, 0, 0], [0, 1, 0]])
transformed = homogeneous_transform(points, R, t)
print("変換後の点:")
print(transformed)
```

---

#### Notebook 52: カメラキャリブレーション

**目標**: 実カメラの内部・外部パラメータを推定する

**重要な概念**:
- Zhang's method
- チェスボードパターン
- 再投影誤差

**実装の流れ**:
1. チェスボード画像の準備
2. コーナー検出
3. カメラ行列と歪み係数の推定
4. 歪み補正の適用

**コード例**:
```python
import cv2

# チェスボードのサイズ
pattern_size = (9, 6)  # 内部コーナーの数

# 複数の画像からキャリブレーション
objpoints = []  # 3D点
imgpoints = []  # 2D点

for image_path in image_paths:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # コーナー検出
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# カメラキャリブレーション
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("カメラ行列 K:")
print(K)
```

---

### Phase 2: エピポーラ幾何とステレオ視

#### Notebook 53: エピポーラ幾何の基礎

**目標**: 2視点幾何学を理解し、基礎行列・本質行列を推定する

**重要な概念**:
- エピポーラ線、エピポール
- 基礎行列 F
- 本質行列 E
- 8点アルゴリズム

**実装の流れ**:
1. 対応点の検出
2. 基礎行列 F の推定（8点アルゴリズム）
3. エピポーラ線の描画
4. 本質行列 E の計算

**コード例**:
```python
from utils.matching import detect_and_compute, match_features, extract_matched_points
from utils.visualizer import plot_epipolar_lines

# 特徴点検出とマッチング
kp1, desc1 = detect_and_compute(img1, method='sift')
kp2, desc2 = detect_and_compute(img2, method='sift')
matches = match_features(desc1, desc2, ratio_test=0.75)

# 対応点の抽出
pts1, pts2 = extract_matched_points(kp1, kp2, matches)

# 基礎行列の推定（RANSAC）
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)

# エピポーラ線の描画
fig, axes = plot_epipolar_lines(img1, img2, pts1, pts2, F, n_lines=10)
plt.show()
```

---

#### Notebook 54: ステレオビジョンと深度推定

**目標**: ステレオ画像ペアから深度マップを生成する

**重要な概念**:
- 視差（disparity）
- ブロックマッチング
- Semi-Global Matching（SGM）
- 深度と視差の関係

**実装の流れ**:
1. ステレオ画像の平行化
2. ブロックマッチングによる視差計算
3. 深度マップの生成
4. 3D点群への変換

**コード例**:
```python
# ステレオマッチング
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
disparity = stereo.compute(img_left_gray, img_right_gray)

# 視差から深度への変換
# Z = (f * B) / d
# f: 焦点距離, B: ベースライン, d: 視差
baseline = 0.1  # メートル
focal_length = K[0, 0]  # ピクセル
depth = (focal_length * baseline) / (disparity + 1e-10)

# 深度マップの可視化
plt.imshow(depth, cmap='viridis')
plt.colorbar(label='Depth (m)')
plt.show()
```

---

### Phase 3: Structure from Motion

#### Notebook 55: 特徴点検出とマッチング

**目標**: SIFT/ORBなどの特徴量を使いこなす

**重要な概念**:
- SIFT, ORB, AKAZE
- 特徴量記述子
- Lowe's ratio test
- RANSAC

**実装の流れ**:
1. 異なる特徴量検出器の比較
2. マッチング手法の比較
3. RANSACによる外れ値除去
4. マッチング結果の可視化

**コード例**:
```python
from utils.matching import (
    detect_and_compute,
    match_features,
    extract_matched_points,
    ransac_homography,
    draw_matches
)

# SIFT特徴量
kp_sift, desc_sift = detect_and_compute(img1, method='sift')

# ORB特徴量
kp_orb, desc_orb = detect_and_compute(img1, method='orb', nfeatures=1000)

# マッチング
matches_sift = match_features(desc_sift1, desc_sift2, ratio_test=0.75)
matches_orb = match_features(desc_orb1, desc_orb2, ratio_test=0.75)

print(f"SIFT matches: {len(matches_sift)}")
print(f"ORB matches: {len(matches_orb)}")

# RANSAC
pts1, pts2 = extract_matched_points(kp1, kp2, matches)
H, mask = ransac_homography(pts1, pts2, threshold=5.0)
print(f"Inliers: {mask.sum()} / {len(pts1)}")
```

---

#### Notebook 56: 三角測量と点群再構成

**目標**: 2視点から3D点を復元する

**重要な概念**:
- 三角測量（Triangulation）
- DLT（Direct Linear Transform）
- 再投影誤差

**実装の流れ**:
1. 2つのカメラの相対姿勢を推定
2. 対応点から3D点を三角測量
3. 疎な点群の生成
4. Open3Dでの3D可視化

**コード例**:
```python
# カメラ行列
P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])  # カメラ1
P2 = K @ np.hstack([R, t.reshape(3, 1)])           # カメラ2

# 三角測量
points_4d_homogeneous = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

# 同次座標から3D座標へ
points_3d = points_4d_homogeneous[:3, :] / points_4d_homogeneous[3, :]
points_3d = points_3d.T

# Open3Dで可視化
import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)

o3d.visualization.draw_geometries([pcd])
```

---

#### Notebook 57: Structure from Motion パイプライン

**目標**: 複数画像からカメラ軌跡と3D構造を同時復元する

**重要な概念**:
- インクリメンタルSfM
- PnP問題
- バンドル調整（Bundle Adjustment）

**実装の流れ**:
1. 画像シーケンスの読み込み
2. 全ペア画像のマッチング
3. 初期2視点の選択と復元
4. 新規カメラの追加（PnP）
5. バンドル調整

**コード例**:
```python
# 簡易的なインクリメンタルSfMの疑似コード

# 1. 初期2視点の選択
img1, img2 = select_initial_pair(images)

# 2. 特徴点マッチング
kp1, desc1 = detect_and_compute(img1)
kp2, desc2 = detect_and_compute(img2)
matches = match_features(desc1, desc2)
pts1, pts2 = extract_matched_points(kp1, kp2, matches)

# 3. 本質行列の推定とR, tの復元
E, mask = cv2.findEssentialMat(pts1, pts2, K)
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

# 4. 初期点群の生成
P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
P2 = K @ np.hstack([R, t.reshape(3, 1)])
points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
points_3d = (points_4d[:3, :] / points_4d[3, :]).T

# 5. 新規画像の追加（PnPで姿勢推定）
for img_new in remaining_images:
    # 新規画像と既存3D点の対応を見つける
    # ...

    # PnPで新規カメラの姿勢を推定
    success, rvec, tvec = cv2.solvePnP(
        points_3d, points_2d_new, K, None, flags=cv2.SOLVEPNP_ITERATIVE
    )

    # 新規3D点を追加
    # ...

    # バンドル調整（オプション）
    # ...
```

---

### Phase 4: NeRF/3DGSへの橋渡し

#### Notebook 58: Ray Castingとボリュームレンダリング

**目標**: カメラから3D空間への光線を生成し、ボリュームレンダリングを実装する

**重要な概念**:
- Ray Casting（光線投射）
- ボリュームレンダリング
- アルファ合成

**実装の流れ**:
1. カメラパラメータから光線の生成
2. 光線上の点のサンプリング
3. ボクセルグリッドの作成
4. ボリュームレンダリングの実装

**コード例**:
```python
def generate_rays(H, W, K, R, t):
    """
    カメラから光線を生成

    Returns
    -------
    rays_o : np.ndarray, shape (H*W, 3)
        光線の原点（カメラ中心）
    rays_d : np.ndarray, shape (H*W, 3)
        光線の方向ベクトル
    """
    # 画像座標のメッシュグリッド
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing='xy'
    )

    # カメラ座標系での方向
    dirs = np.stack([
        (i - K[0, 2]) / K[0, 0],
        (j - K[1, 2]) / K[1, 1],
        np.ones_like(i)
    ], axis=-1)

    # 世界座標系での方向
    rays_d = np.sum(dirs[..., None, :] * R.T, axis=-1)

    # カメラ中心（世界座標系）
    rays_o = -R.T @ t
    rays_o = np.broadcast_to(rays_o, rays_d.shape)

    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    return rays_o, rays_d


# ボリュームレンダリング
def volume_rendering(rays_o, rays_d, density_fn, color_fn, t_near, t_far, n_samples):
    """
    ボリュームレンダリング方程式

    C(r) = Σ T_i * (1 - exp(-σ_i * δ_i)) * c_i
    T_i = exp(-Σ_{j<i} σ_j * δ_j)
    """
    # 光線上のサンプリング点
    t = np.linspace(t_near, t_far, n_samples)
    points = rays_o[:, None, :] + rays_d[:, None, :] * t[None, :, None]

    # 各点での密度と色を取得
    density = density_fn(points)  # (n_rays, n_samples)
    colors = color_fn(points)     # (n_rays, n_samples, 3)

    # 距離
    delta = t[1:] - t[:-1]
    delta = np.concatenate([delta, np.array([1e10])])

    # アルファ値（不透明度）
    alpha = 1.0 - np.exp(-density * delta)

    # 透過率
    transmittance = np.cumprod(1.0 - alpha + 1e-10, axis=-1)
    transmittance = np.concatenate([
        np.ones_like(transmittance[:, :1]),
        transmittance[:, :-1]
    ], axis=-1)

    # 最終的な色
    weights = alpha * transmittance
    rgb = np.sum(weights[..., None] * colors, axis=1)

    return rgb
```

---

#### Notebook 59: 3D Vision から NeRF/3DGS への橋渡し

**目標**: 古典的3D CVと最新の3D生成技術の関係を理解する

**重要な概念**:
- COLMAPとNeRFの関係
- カメラパラメータの正規化
- transforms.jsonフォーマット

**実装の流れ**:
1. SfMで得られたカメラポーズの読み込み
2. NeRF用データフォーマットへの変換
3. transforms.jsonの生成
4. カメラパラメータの誤差シミュレーション

**コード例**:
```python
def create_nerf_transforms(images, cameras, points_3d, output_path):
    """
    NeRF用のtransforms.jsonを生成
    """
    transforms = {
        "camera_angle_x": 2 * np.arctan(cameras[0].width / (2 * cameras[0].fx)),
        "frames": []
    }

    for i, (img_path, camera) in enumerate(zip(images, cameras)):
        # カメラポーズ（OpenGL座標系への変換が必要）
        R = camera.R
        t = camera.t

        # 変換行列（4x4）
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R.T
        transform_matrix[:3, 3] = -R.T @ t

        frame = {
            "file_path": img_path,
            "transform_matrix": transform_matrix.tolist()
        }

        transforms["frames"].append(frame)

    # JSONファイルとして保存
    import json
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=2)

    print(f"✅ transforms.json を {output_path} に保存しました")


# 使用例
create_nerf_transforms(
    images=image_paths,
    cameras=camera_list,
    points_3d=reconstructed_points,
    output_path="transforms.json"
)
```

---

## 🔧 よくある問題と解決法

### 1. OpenCVのSIFTが使えない

**問題**:
```python
AttributeError: module 'cv2' has no attribute 'SIFT_create'
```

**解決法**:
```bash
# opencv-contrib-pythonをインストール
pip install opencv-contrib-python
```

### 2. Open3Dのインポートエラー

**問題**:
```python
ImportError: No module named 'open3d'
```

**解決法**:
```bash
pip install open3d
```

### 3. カメラキャリブレーションでコーナーが検出されない

**問題**: `cv2.findChessboardCorners` が False を返す

**解決法**:
- チェスボードが画像全体に明瞭に写っているか確認
- 照明条件を改善
- `cv2.CALIB_CB_ADAPTIVE_THRESH` フラグを試す

```python
ret, corners = cv2.findChessboardCorners(
    gray, pattern_size,
    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
)
```

### 4. マッチング数が少ない

**問題**: 特徴点マッチングで十分なマッチが得られない

**解決法**:
- 特徴点の検出数を増やす: `nfeatures=2000`
- ratio testの閾値を緩める: `ratio_test=0.8`
- 異なる特徴量検出器を試す（SIFT, ORB, AKAZE）

---

## 📊 デバッグのコツ

### 1. 射影の確認

```python
# 既知の3D点を射影して、期待通りの2D座標になるか確認
points_3d = np.array([[0, 0, 5]])  # カメラ前方5m
points_2d = camera.project(points_3d)
print(f"射影結果: {points_2d}")
print(f"期待値: [cx, cy] = [{camera.cx}, {camera.cy}]")
```

### 2. 回転行列の確認

```python
# 回転行列は直交行列（R @ R.T = I）
R = rotation_matrix_z(np.pi / 2)
print(f"R @ R.T =\n{R @ R.T}")
print(f"det(R) = {np.linalg.det(R)}")  # det(R) = 1 のはず
```

### 3. エピポーラ制約の確認

```python
# エピポーラ制約: x'^T F x = 0
for i in range(len(pts1)):
    pt1_homogeneous = np.array([pts1[i, 0], pts1[i, 1], 1])
    pt2_homogeneous = np.array([pts2[i, 0], pts2[i, 1], 1])

    error = pt2_homogeneous.T @ F @ pt1_homogeneous
    print(f"Point {i}: error = {error:.6f}")  # ≈ 0 のはず
```

---

## 🎯 次のステップ

このガイドを完了したら、以下に挑戦してみましょう：

1. **COLMAPの使用**: 実用的なSfMツールを試す
2. **Nerfstudio**: NeRFの学習と新規視点合成
3. **自分のプロジェクト**: 部屋や物体の3D再構成
4. **論文の実装**: 最新の3D技術を自分で実装

---

**Happy Coding! 🚀**
