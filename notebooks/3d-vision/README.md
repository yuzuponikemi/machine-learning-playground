# 3D Computer Vision 基礎実装プロジェクト

## 📁 プロジェクト構成

```
notebooks/3d-vision/
├── README.md                                           # このファイル
├── utils/                                              # 共通ユーティリティ
│   ├── __init__.py
│   ├── geometry_tools.py                               # 幾何変換関数
│   ├── camera.py                                       # カメラモデルクラス
│   ├── visualizer.py                                   # 3D可視化ツール
│   └── matching.py                                     # 特徴点マッチングツール
│
├── data/                                               # データセット
│   ├── calibration/                                    # キャリブレーション用画像
│   │   └── checkerboard/
│   ├── stereo/                                         # ステレオ画像ペア
│   │   ├── left/
│   │   └── right/
│   └── sfm/                                            # SfM用画像シーケンス
│       └── sequence1/
│
├── 50_pinhole_camera_model_v1.ipynb                    # Phase 1
├── 51_camera_extrinsics_transforms_v1.ipynb
├── 52_camera_calibration_v1.ipynb
│
├── 53_epipolar_geometry_fundamentals_v1.ipynb          # Phase 2
├── 54_stereo_vision_depth_estimation_v1.ipynb
│
├── 55_feature_detection_matching_v1.ipynb              # Phase 3
├── 56_triangulation_point_cloud_v1.ipynb
├── 57_structure_from_motion_pipeline_v1.ipynb
│
├── 58_ray_casting_volume_rendering_v1.ipynb            # Phase 4
└── 59_bridge_to_nerf_3dgs_v1.ipynb
```

---

## 🛠️ 環境セットアップ

### 必須ライブラリのインストール

```bash
# 基本ライブラリ
pip install numpy scipy matplotlib pillow

# コンピュータビジョン
pip install opencv-python opencv-contrib-python

# 3D処理・可視化
pip install open3d plotly

# オプション（高度な機能）
pip install pycolmap trimesh
```

または、プロジェクトルートから：

```bash
# uvを使用している場合
uv pip install numpy scipy matplotlib pillow opencv-python opencv-contrib-python open3d plotly

# 通常のpip
pip install -r requirements-3dvision.txt
```

---

## 📚 ノートブック一覧

### Phase 1: カメラモデルと射影幾何（6-8時間）

| No. | タイトル | 難易度 | 時間 | 主な内容 |
|-----|---------|--------|------|---------|
| 50 | ピンホールカメラモデルの基礎 | ★★☆☆☆ | 90-120分 | 焦点距離、主点、内部パラメータ行列K |
| 51 | カメラ外部パラメータと座標変換 | ★★★☆☆ | 90-120分 | 回転行列R、並進ベクトルt、座標変換 |
| 52 | カメラキャリブレーション | ★★★☆☆ | 120-150分 | Zhang's method、歪み補正 |

### Phase 2: エピポーラ幾何とステレオ視（8-10時間）

| No. | タイトル | 難易度 | 時間 | 主な内容 |
|-----|---------|--------|------|---------|
| 53 | エピポーラ幾何の基礎 | ★★★☆☆ | 120-150分 | 基礎行列F、本質行列E、8点アルゴリズム |
| 54 | ステレオビジョンと深度推定 | ★★★★☆ | 150-180分 | 視差マップ、ブロックマッチング、SGM |

### Phase 3: Structure from Motion（10-12時間）

| No. | タイトル | 難易度 | 時間 | 主な内容 |
|-----|---------|--------|------|---------|
| 55 | 特徴点検出とマッチング | ★★★☆☆ | 120-150分 | SIFT/ORB、マッチング、RANSAC |
| 56 | 三角測量と点群再構成 | ★★★★☆ | 120-150分 | DLT法、三角測量、疎な点群生成 |
| 57 | Structure from Motion パイプライン | ★★★★★ | 180-240分 | インクリメンタルSfM、バンドル調整 |

### Phase 4: NeRF/3DGSへの橋渡し（8-10時間）

| No. | タイトル | 難易度 | 時間 | 主な内容 |
|-----|---------|--------|------|---------|
| 58 | Ray Castingとボリュームレンダリング | ★★★★☆ | 150-180分 | 光線生成、ボリュームレンダリング |
| 59 | 3D Vision から NeRF/3DGS への橋渡し | ★★★★☆ | 120-150分 | NeRF/3DGSの基礎、カメラパラメータ |

---

## 🔧 ユーティリティモジュール

### `utils/geometry_tools.py`
幾何変換関数（回転行列、ロドリゲス変換、射影変換など）

### `utils/camera.py`
カメラモデルクラス（内部・外部パラメータの管理、射影計算）

### `utils/visualizer.py`
3D可視化ツール（Open3D, Matplotlibベース）

### `utils/matching.py`
特徴点マッチングツール（SIFT/ORB、RANSAC）

---

## 📊 データセット

### 提供データ
- チェスボードパターン画像（キャリブレーション用）
- ステレオ画像ペア（深度推定用）
- 画像シーケンス（SfM用）

### 独自データの追加
`data/` ディレクトリ配下に、自分で撮影した画像を配置できます。

---

## 🎯 学習の進め方

1. **Phase 1から順番に進める**
   - カメラモデルの理解は全ての基礎です

2. **コードを実際に動かす**
   - セルを1つずつ実行し、結果を確認

3. **パラメータを変えて実験**
   - 焦点距離、視差範囲など、パラメータを変えて挙動を観察

4. **自分のデータで試す**
   - スマホで撮影した画像で実験

5. **練習問題に挑戦**
   - 各ノートブック末尾の練習問題で理解を深める

---

## 🚀 発展的な学習

このプロジェクトを完了後、以下のトピックに挑戦できます：

- **COLMAPの使用**: 実用的なSfMツールの活用
- **Nerfstudio**: NeRFの実装と学習
- **3D Gaussian Splatting**: 最新の3D表現技術
- **Visual SLAM**: リアルタイムカメラトラッキング

---

## 📞 サポート

質問や提案がある場合は、GitHubのIssueをご利用ください。

---

**Happy 3D Vision Learning! 🎓✨**
