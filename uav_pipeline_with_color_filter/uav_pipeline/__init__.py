"""Top-level package for the UAV RANSAC pipeline.

This package provides utilities for reading Pix4D calibration files,
loading point clouds, computing rays via collinearity equations,
estimating ground planes, clustering 3D points and filtering them
according to observation counts. See individual modules for details.

한국어 설명
------------
이 패키지는 Pix4D 보정 파일을 읽고, 포인트 클라우드를 로드하고,
공선 조건식을 사용하여 레이를 계산하며, 지면 평면을 추정하고,
3차원 점들을 군집화한 뒤 관측 횟수에 따라 필터링하는 등 UAV
분석 파이프라인에 필요한 여러 도구를 제공합니다. 자세한 내용은
각 모듈의 문서를 참고하세요.
"""

from .camera_parser import CameraIntrinsics, CameraExtrinsics, parse_internal_parameters, parse_external_parameters
from .pointcloud_loader import PointCloud, load_las, build_kdtree
from .ray_casting import (
    rotation_matrix_from_omega_phi_kappa,
    pixel_to_normalized,
    undistort_point,
    ray_from_pixel,
    intersect_ray_with_plane,
    estimate_ground_plane,
)
from .wire_filter import cluster_points, filter_by_count

__all__ = [
    "CameraIntrinsics",
    "CameraExtrinsics",
    "parse_internal_parameters",
    "parse_external_parameters",
    "PointCloud",
    "load_las",
    "build_kdtree",
    "rotation_matrix_from_omega_phi_kappa",
    "pixel_to_normalized",
    "undistort_point",
    "ray_from_pixel",
    "intersect_ray_with_plane",
    "estimate_ground_plane",
    "cluster_points",
    "filter_by_count",
]