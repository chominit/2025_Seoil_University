"""
ray_casting.py
---------------

Functions for projecting 2D image coordinates into 3D space using
collinearity equations and intersecting those rays with a point
cloud or a ground plane. These utilities form the core of a
photogrammetric measurement pipeline for extracting world coordinates
from pixel positions given calibrated camera parameters.

Overview
--------
To map a point from image coordinates (xi, yi) to a ray in world
space, we use the collinearity equations:

::

    [X]   [X0]             [m11 m12 m13] [x]
    [Y] = [Y0] + (Zc / f) * [m21 m22 m23] [y]
    [Z]   [Z0]             [m31 m32 m33] [f]

where (X0, Y0, Z0) is the camera centre, (x, y) are the image
coordinates centred on the principal point and scaled by the focal
length, f is the focal length and [mij] is the rotation matrix
derived from the exterior orientation angles (Omega, Phi, Kappa).
Since the object point is at an unknown depth along the ray, we can
parameterise the ray and intersect it with a surface such as a plane
or a point cloud.

This module implements:

* Conversion from Omega/Phi/Kappa angles to a rotation matrix.
* Distortion correction for observed image coordinates.
* Ray parameterisation from pixel coordinates.
* Intersection of a ray with an estimated ground plane using analytic
  geometry.
* Estimation of a dominant plane (ground) from a point cloud via
  RANSAC.

Dependencies
------------
* numpy
* scipy (optional for RANSAC implementation; fallback provided)
* sklearn (optional for RANSAC if available)

한국어 설명
------------
이 모듈은 보정된 카메라 파라미터를 사용하여 이미지상의 2차원 좌표를 3차원 공간상의
레이로 변환하고, 그 레이를 포인트 클라우드나 평면과 교차시키는 기능을 제공합니다.
사진측량에서 공선 조건식을 이용해 픽셀 위치로부터 세계 좌표를 계산할 때 필수적인
연산을 수행합니다. 오메가/파이/카파(회전각)를 회전 행렬로 변환하고, 렌즈 왜곡을
보정한 뒤, 픽셀 좌표를 정규화하여 레이를 만들고, 그 레이와 평면(예: 지면) 또는
포인트 클라우드와의 교점을 찾습니다. 또한 RANSAC을 이용해 점군에서 우세한 평면을
추정하는 함수도 포함되어 있습니다.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import RANSACRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
except ImportError:
    RANSACRegressor = None  # type: ignore
    PolynomialFeatures = None  # type: ignore
    make_pipeline = None  # type: ignore


def rotation_matrix_from_omega_phi_kappa(omega: float, phi: float, kappa: float) -> np.ndarray:
    """Compute the rotation matrix from Omega, Phi, Kappa angles.

    Parameters
    ----------
    omega, phi, kappa : float
        Rotation angles in degrees following the photogrammetric
        convention. Omega rotates around X, phi around Y and kappa
        around Z.

    Returns
    -------
    ndarray of shape (3, 3)
        Rotation matrix transforming from camera coordinates to world
        coordinates.

    한국어 설명
    --------------
    오메가(Omega), 파이(Phi), 카파(Kappa) 회전각으로부터 회전 행렬을 계산합니다.
    각 각도는 photogrammetry 규약에 따라 도(degree) 단위로 주어지며, 오메가는 X축
    주위 회전, 파이는 Y축 주위, 카파는 Z축 주위 회전을 의미합니다. 반환되는 행렬은
    카메라 좌표계를 세계 좌표계로 변환하는 3×3 회전 행렬입니다.
    """
    # Convert degrees to radians
    o = math.radians(omega)
    p = math.radians(phi)
    k = math.radians(kappa)
    # Rotation matrices around X, Y, Z axes
    r_x = np.array([
        [1, 0, 0],
        [0, math.cos(o), -math.sin(o)],
        [0, math.sin(o), math.cos(o)],
    ])
    r_y = np.array([
        [math.cos(p), 0, math.sin(p)],
        [0, 1, 0],
        [-math.sin(p), 0, math.cos(p)],
    ])
    r_z = np.array([
        [math.cos(k), -math.sin(k), 0],
        [math.sin(k), math.cos(k), 0],
        [0, 0, 1],
    ])
    # Combined rotation matrix (Z * Y * X)
    return r_z @ r_y @ r_x


def undistort_point(x: float, y: float, intrinsics) -> Tuple[float, float]:
    """Apply radial and tangential distortion correction to normalized coordinates.

    Parameters
    ----------
    x, y : float
        Distorted normalized image coordinates (with origin at the
        principal point and scaled by focal length).
    intrinsics : Any
        Camera intrinsics object with attributes ``radial_distortion``
        and ``tangential_distortion``.

    Returns
    -------
    (float, float)
        Undistorted normalized coordinates.

    Notes
    -----
    This uses the Brown–Conrady distortion model with radial
    coefficients (k1, k2, k3) and tangential coefficients (p1, p2).
    See photogrammetry references for details.

    한국어 설명
    --------------
    이 함수는 정규화된 이미지 좌표에 대해 방사 및 접선 왜곡을 보정합니다. Brown–Conrady
    모델을 사용하며, 방사 왜곡 계수(k1, k2, k3)와 접선 왜곡 계수(p1, p2)를 적용하여
    왜곡된 좌표를 보정합니다. 자세한 내용은 사진측량 참고자료를 참고하세요.
    """
    k1, k2, k3 = intrinsics.radial_distortion
    p1, p2 = intrinsics.tangential_distortion
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_distorted = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_distorted = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return x_distorted, y_distorted


def pixel_to_normalized(image_x: float, image_y: float, intrinsics) -> Tuple[float, float]:
    """Convert pixel coordinates to normalized camera coordinates.

    This subtracts the principal point, divides by the focal length and
    applies lens distortion correction.

    Parameters
    ----------
    image_x, image_y : float
        Pixel coordinates in the image reference frame (origin at
        top-left). The y axis increases downwards.
    intrinsics : CameraIntrinsics
        Intrinsic parameters including focal length and principal
        point.

    Returns
    -------
    (float, float)
        Undistorted normalized coordinates in the camera frame.

    한국어 설명
    --------------
    픽셀 좌표를 카메라 좌표계의 정규화 좌표로 변환합니다. 주점을 원점으로 이동하고
    초점거리로 나누어 크기를 정규화한 후, 렌즈 왜곡을 보정합니다. 반환값은 왜곡이
    제거된 정규화 좌표(x, y)입니다.
    """
    cx_px, cy_px = intrinsics.principal_point_px
    f_px = intrinsics.focal_length_px
    # Shift origin to principal point and normalize by focal length
    x = (image_x - cx_px) / f_px
    y = (image_y - cy_px) / f_px
    # Apply distortion correction
    x_u, y_u = undistort_point(x, y, intrinsics)
    return x_u, y_u


def ray_from_pixel(
    image_x: float,
    image_y: float,
    intrinsics,
    extrinsics,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a 3D ray from a pixel coordinate.

    Parameters
    ----------
    image_x, image_y : float
        Pixel coordinates.
    intrinsics : CameraIntrinsics
        Intrinsic parameters for the camera model.
    extrinsics : CameraExtrinsics
        External orientation for the specific image.

    Returns
    -------
    (ndarray, ndarray)
        Tuple `(origin, direction)` where origin is the camera centre
        (3-element vector) and direction is a unit vector pointing
        outward along the ray in world coordinates.

    한국어 설명
    --------------
    이미지 픽셀 좌표에서 출발하는 3차원 레이를 계산합니다. 픽셀을 정규화한 후
    카메라 좌표계에서 벡터를 만들고, 주어진 외부 파라미터(오메가, 파이, 카파)를
    사용해 세계 좌표계로 변환합니다. 반환값은 카메라 중심 좌표와 정규화된 레이
    방향 벡터입니다.
    """
    # Convert pixel to normalized camera coordinates (undistorted)
    x_u, y_u = pixel_to_normalized(image_x, image_y, intrinsics)
    # Coordinates in camera coordinate system (z pointing forward)
    cam_vec = np.array([x_u, y_u, 1.0])
    # Compute rotation matrix for this image
    R = rotation_matrix_from_omega_phi_kappa(*extrinsics.angles)
    # Transform direction to world coordinates and normalise
    world_dir = R @ cam_vec
    world_dir /= np.linalg.norm(world_dir)
    # Camera centre in world coordinates
    origin = np.array(extrinsics.position)
    return origin, world_dir


def intersect_ray_with_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute the intersection of a ray with a plane.

    Parameters
    ----------
    origin : ndarray
        Starting point of the ray.
    direction : ndarray
        Unit direction vector of the ray.
    plane_point : ndarray
        A point lying on the plane.
    plane_normal : ndarray
        Normal vector of the plane.

    Returns
    -------
    Optional[ndarray]
        The intersection point as a 3D coordinate, or ``None`` if the
        ray is parallel to the plane or the intersection lies behind
        the ray origin.

    한국어 설명
    --------------
    주어진 레이(origin과 direction)와 평면(plane_point와 plane_normal)의 교점을
    계산합니다. 레이가 평면과 평행하거나, 교점이 카메라 뒤쪽에 존재하면 ``None``을
    반환합니다. 그렇지 않으면 교점의 3D 좌표를 numpy 배열로 반환합니다.
    """
    denom = np.dot(direction, plane_normal)
    if abs(denom) < 1e-8:
        # Ray is parallel to the plane
        return None
    t = np.dot(plane_point - origin, plane_normal) / denom
    if t < 0:
        # Intersection is behind the camera
        return None
    return origin + t * direction


def estimate_ground_plane(points: np.ndarray, max_trials: int = 1000, residual_threshold: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a dominant horizontal plane (ground) from a point cloud.

    This uses a RANSAC-based approach to fit a plane of the form
    ax + by + cz + d = 0 to the input points. The returned plane is
    represented by a point on the plane and a normal vector.

    Parameters
    ----------
    points : ndarray
        Nx3 array of point coordinates.
    max_trials : int, optional
        Maximum number of RANSAC iterations. Default is 1000.
    residual_threshold : float, optional
        Threshold distance for a point to be considered an inlier.
        Default is 0.05 units (depends on coordinate system).

    Returns
    -------
    (ndarray, ndarray)
        Tuple `(plane_point, plane_normal)`. The plane point is the
        centroid of inlier points; the normal is a unit vector.

    Notes
    -----
    If scikit-learn is not available, a minimal custom RANSAC is
    implemented. This algorithm may be slower but avoids external
    dependencies.

    한국어 설명
    --------------
    주어진 포인트 클라우드에서 우세한 평면(주로 지면)을 RANSAC을 사용해 추정합니다.
    입력은 (N x 3) 형태의 점 배열이며, 반환값은 평면 위의 한 점과 단위 노멀 벡터입니다.
    scikit‑learn이 설치되어 있지 않은 경우 간단한 맞춤형 RANSAC을 사용하여 느릴 수
    있지만 외부 의존성을 피합니다.
    """
    if RANSACRegressor is not None:
        # Fit plane in the form z = ax + by + c
        X = points[:, :2]
        y = points[:, 2]
        # Use polynomial features of degree 1 to fit a plane
        model = make_pipeline(PolynomialFeatures(degree=1), RANSACRegressor(max_trials=max_trials, residual_threshold=residual_threshold))
        model.fit(X, y)
        # Coefficients correspond to [c, a, b]
        coef = model.named_steps['ransacregressor'].estimator_.coef_
        intercept = model.named_steps['ransacregressor'].estimator_.intercept_
        # Plane normal: (-a, -b, 1), normalised
        a = coef[1]
        b = coef[2]
        c = -1.0
        normal = np.array([-a, -b, c])
        normal /= np.linalg.norm(normal)
        # Compute a point on the plane as the centroid of inliers
        inlier_mask = model.named_steps['ransacregressor'].inlier_mask_
        plane_point = points[inlier_mask].mean(axis=0)
        return plane_point, normal
    else:
        # Minimal custom RANSAC: sample three random points to define a plane
        best_inliers: List[int] = []
        best_normal = np.array([0.0, 0.0, 1.0])
        best_point = np.zeros(3)
        rng = np.random.default_rng()
        n_points = points.shape[0]
        for _ in range(max_trials):
            idx = rng.choice(n_points, size=3, replace=False)
            p0, p1, p2 = points[idx]
            # Compute normal of the candidate plane
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            if np.linalg.norm(normal) < 1e-8:
                continue
            normal /= np.linalg.norm(normal)
            # Compute distances of all points to the plane
            dists = np.abs((points - p0).dot(normal))
            inliers = np.where(dists < residual_threshold)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers.tolist()
                best_normal = normal
                best_point = p0
        if not best_inliers:
            # Fallback: use mean and vertical normal
            return points.mean(axis=0), np.array([0.0, 0.0, 1.0])
        inlier_points = points[best_inliers]
        plane_point = inlier_points.mean(axis=0)
        return plane_point, best_normal


__all__ = [
    "rotation_matrix_from_omega_phi_kappa",
    "undistort_point",
    "pixel_to_normalized",
    "ray_from_pixel",
    "intersect_ray_with_plane",
    "estimate_ground_plane",
]