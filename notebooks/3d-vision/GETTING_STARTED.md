# 3D Computer Vision 基礎 - Getting Started

このドキュメントは、3D Computer Visionカリキュラムを始めるための最初のステップです。

---

## 🎯 このカリキュラムで学べること

**「空間の文法」**を習得し、2D画像から3D世界を理解する力を身につけます。

### 学習内容
- ✅ ピンホールカメラモデルと射影幾何
- ✅ エピポーラ幾何とステレオ視
- ✅ Structure from Motion（SfM）
- ✅ NeRF/3DGaussian Splattingへの橋渡し

### 学習成果
- カメラの内部・外部パラメータを理解し、推定できる
- ステレオ画像から深度マップを生成できる
- 複数画像から3D点群とカメラ軌跡を復元できる
- NeRF/3DGSで使われるカメラパラメータの意味を理解できる

---

## 📚 前提知識

このカリキュラムを始める前に、以下の知識があることが推奨されます：

- ✅ Pythonプログラミングの基礎
- ✅ NumPy、Matplotlibの基本的な使い方
- ✅ 線形代数の基礎（行列演算、固有値、特異値分解）
- ✅ 基礎的な数学（微分、最適化、確率の基本）

**推奨**: このリポジトリのノートブック00-12（機械学習基礎コース）を完了していること

---

## 🚀 セットアップ

### 1. 環境の準備

```bash
# プロジェクトルートに移動
cd /path/to/machine-learning-playground

# 3D Vision用ライブラリのインストール
pip install -r requirements-3dvision.txt
```

### 2. ユーティリティのテスト

```bash
# 3d-visionディレクトリに移動
cd notebooks/3d-vision

# すべてのユーティリティをテスト
python utils/geometry_tools.py
python utils/camera.py
python utils/visualizer.py
python utils/matching.py
```

すべてのテストが「✅ すべてのテストが完了しました」と表示されれば成功です！

---

## 📖 カリキュラムの進め方

### Phase 1: カメラモデルと射影幾何（6-8時間）

| ノートブック | タイトル | 時間 |
|-------------|---------|------|
| 50 | ピンホールカメラモデルの基礎 | 90-120分 |
| 51 | カメラ外部パラメータと座標変換 | 90-120分 |
| 52 | カメラキャリブレーション | 120-150分 |

**学ぶこと**: カメラの基本原理、3D→2D射影、座標変換

### Phase 2: エピポーラ幾何とステレオ視（8-10時間）

| ノートブック | タイトル | 時間 |
|-------------|---------|------|
| 53 | エピポーラ幾何の基礎 | 120-150分 |
| 54 | ステレオビジョンと深度推定 | 150-180分 |

**学ぶこと**: 2視点幾何学、基礎行列、深度推定

### Phase 3: Structure from Motion（10-12時間）

| ノートブック | タイトル | 時間 |
|-------------|---------|------|
| 55 | 特徴点検出とマッチング | 120-150分 |
| 56 | 三角測量と点群再構成 | 120-150分 |
| 57 | SfMパイプライン | 180-240分 |

**学ぶこと**: 特徴点マッチング、三角測量、インクリメンタルSfM

### Phase 4: NeRF/3DGSへの橋渡し（8-10時間）

| ノートブック | タイトル | 時間 |
|-------------|---------|------|
| 58 | Ray Castingとボリュームレンダリング | 150-180分 |
| 59 | 3D Vision から NeRF/3DGS への橋渡し | 120-150分 |

**学ぶこと**: Ray Casting、ボリュームレンダリング、NeRF/3DGSの基礎

**合計学習時間**: 32-40時間（4-5週間）

---

## 📁 プロジェクト構成

```
notebooks/3d-vision/
├── README.md                                    # プロジェクト概要
├── GETTING_STARTED.md                           # このファイル
├── IMPLEMENTATION_GUIDE.md                      # 実装ガイド
├── utils/                                       # ユーティリティモジュール
│   ├── __init__.py
│   ├── geometry_tools.py                        # 幾何変換
│   ├── camera.py                                # カメラモデル
│   ├── visualizer.py                            # 3D可視化
│   └── matching.py                              # 特徴点マッチング
├── data/                                        # データセット
└── 50_pinhole_camera_model_v1.ipynb             # ノートブック（Phase 1）
    ...
```

---

## 🎓 学習のヒント

### 1. コードを実際に動かす
- セルを1つずつ実行し、結果を確認
- パラメータを変えて挙動を観察

### 2. 視覚化を活用する
- 3Dプロットで直感を養う
- カメラと点群の関係を可視化

### 3. 数式を恐れない
- 数式は「空間の文法」を記述する道具
- まずコードで動かし、後から数式を理解

### 4. 自分のデータで実験
- スマホで撮影した画像で試す
- 実データでの挙動を確認

### 5. つまずいたら
- `IMPLEMENTATION_GUIDE.md` の「よくある問題と解決法」を確認
- ユーティリティのテストコードを参考にする

---

## 📚 推奨リソース

### 書籍
- **「Multiple View Geometry in Computer Vision」** Richard Hartley, Andrew Zisserman
- **「Computer Vision: Algorithms and Applications」** Richard Szeliski

### オンライン
- **OpenCV公式ドキュメント（Camera Calibration and 3D Reconstruction）**
  https://docs.opencv.org/4.x/d9/db7/tutorial_py_table_of_contents_calib3d.html
- **First Principles of Computer Vision (YouTube)**
  コンピュータビジョンの基礎を丁寧に解説

### 論文（古典）
- **Photo Tourism** (Snavely et al., 2006) - SfMの基礎
- **SIFT** (Lowe, 2004) - 特徴点検出の金字塔

### 論文（最新）
- **NeRF** (Mildenhall et al., 2020) - Neural Radiance Fields
- **3D Gaussian Splatting** (Kerbl et al., 2023) - リアルタイム3D表現

---

## 🎯 最初の一歩

さあ、始めましょう！

```bash
# Jupyter Notebookを起動
cd notebooks/3d-vision
jupyter notebook 50_pinhole_camera_model_v1.ipynb
```

**または**、まずはユーティリティを眺めてみる：

```python
# Pythonインタラクティブシェルで
from utils.camera import PinholeCamera
from utils.visualizer import setup_3d_plot, plot_camera
import numpy as np
import matplotlib.pyplot as plt

# カメラを作成
cam = PinholeCamera.from_fov(60, 640, 480)
print(cam)

# 3D点を射影
points_3d = np.array([[0, 0, 5], [1, 0, 5]])
points_2d = cam.project(points_3d)
print("2D投影:", points_2d)
```

---

## 💬 サポート

質問や提案がある場合は、GitHubのIssueをご利用ください。

---

## 🌟 この先に広がる世界

このカリキュラムを完了すると、以下のような次世代技術への扉が開かれます：

- **NeRF（Neural Radiance Fields）**: 画像から3Dシーンを学習
- **3D Gaussian Splatting**: リアルタイム3D再構成
- **Visual SLAM**: リアルタイムカメラトラッキング
- **NeoVerse**: 4D時空間表現

**視点という主観**と**空間という客観**の関係を数学的に記述できるようになることで、あなたの認知は拡張されます。

---

**Happy Learning! 🎓✨**

3D Computer Visionの世界へようこそ。
