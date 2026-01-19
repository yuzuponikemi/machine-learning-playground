"""
3D Computer Vision Utilities

このパッケージは、3D Computer Visionの基礎実装に必要な
ユーティリティ関数とクラスを提供します。
"""

from .geometry_tools import (
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    euler_to_rotation_matrix,
    rodrigues_to_rotation_matrix,
    rotation_matrix_to_rodrigues,
    homogeneous_transform,
    project_points,
    normalize_points,
)

from .camera import (
    Camera,
    PinholeCamera,
    build_intrinsic_matrix,
    build_projection_matrix,
)

from .visualizer import (
    Visualizer3D,
    plot_camera,
    plot_points_3d,
    plot_epipolar_lines,
)

__all__ = [
    # geometry_tools
    'rotation_matrix_x',
    'rotation_matrix_y',
    'rotation_matrix_z',
    'euler_to_rotation_matrix',
    'rodrigues_to_rotation_matrix',
    'rotation_matrix_to_rodrigues',
    'homogeneous_transform',
    'project_points',
    'normalize_points',
    # camera
    'Camera',
    'PinholeCamera',
    'build_intrinsic_matrix',
    'build_projection_matrix',
    # visualizer
    'Visualizer3D',
    'plot_camera',
    'plot_points_3d',
    'plot_epipolar_lines',
]

__version__ = '1.0.0'
