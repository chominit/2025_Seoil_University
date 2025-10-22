"""
pointcloud_loader.py
--------------------

This module wraps the loading of airborne lidar or photogrammetry
point clouds from LAS/LAZ files. It uses the `laspy` library
(version >= 2.0) to read LAS/LAZ files and convert them into numpy
arrays. The loader also offers convenience functions for building a
KD-tree using scipy for efficient nearest neighbour queries, which
will be helpful for ray–point cloud intersection tests.

Example usage::

    from uav_pipeline.pointcloud_loader import load_las, build_kdtree
    points, attributes = load_las("path/to/pointcloud.las")
    kdtree = build_kdtree(points)
    dists, idxs = kdtree.query([[0.0, 0.0, 0.0]], k=1)

Requirements
------------
* laspy >= 2.0
* numpy
* scipy (for KD-tree). If scipy is not available, a simple brute-force
  implementation could be substituted, but performance will suffer on
  large point clouds.

한국어 설명
------------
이 모듈은 항공 라이다나 사진측량으로 얻은 LAS/LAZ 포인트 클라우드를 읽어
numpy 배열로 변환하는 기능을 제공합니다. `laspy` 라이브러리를 사용하여
포인트 클라우드 파일을 읽고, 점 좌표와 추가 속성을 `PointCloud` 객체로
반환합니다. 또한 scipy를 이용해 KD‑트리를 생성하는 편의 함수도 제공하여,
레이와 점군의 최근접 교차를 효율적으로 계산할 수 있습니다.

예시::

    from uav_pipeline.pointcloud_loader import load_las, build_kdtree
    pc = load_las("path/to/pointcloud.las")
    kdtree = build_kdtree(pc.points)
    dists, idxs = kdtree.query([[0.0, 0.0, 0.0]], k=1)

필수 라이브러리
---------------
* laspy >= 2.0
* numpy
* scipy – KD-트리를 만들기 위해 필요합니다. scipy가 없을 경우 간단한
  브루트포스 구현으로 대체할 수 있으나, 대규모 점군에서는 성능이 떨어집니다.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import laspy
except ImportError as e:
    raise ImportError(
        "laspy is required to read LAS/LAZ files. Install it with `pip install laspy`."
    ) from e

try:
    from scipy.spatial import cKDTree as KDTree
except ImportError as e:
    KDTree = None  # type: ignore
    # We allow the module to load even if scipy is missing. A fallback
    # KD-tree implementation could be provided here if desired.


@dataclass
class PointCloud:
    """Simple container to hold point cloud coordinates and attributes.

    한국어 설명
    --------------
    이 데이터클래스는 포인트 클라우드의 좌표와 각 점에 대한 추가 속성을 저장합니다.
    ``points``는 (N, 3) 형태의 numpy 배열로, 각 행은 X, Y, Z 좌표를 의미합니다.
    ``attributes``는 강도(intensity), 반사 번호(return number), 분류(classification)
    등 포인트에 대한 추가 정보를 키-값 쌍으로 담습니다.
    """

    points: np.ndarray  # shape (N, 3)
    attributes: Dict[str, np.ndarray]  # additional per-point attributes


def load_las(path: str) -> PointCloud:
    """Load a LAS/LAZ file into memory.

    Parameters
    ----------
    path : str
        Path to the LAS or LAZ file.

    Returns
    -------
    PointCloud
        A container with the point coordinates and available
        attributes. Coordinates are returned in a Nx3 numpy array
        (columns X, Y, Z). Additional attributes (e.g. intensity,
        classification) are stored in a dictionary keyed by attribute
        name.

    Notes
    -----
    LAS files may contain billions of points, which will not fit in
    memory on typical machines. In such cases consider streaming the
    points in chunks using laspy's iterator interface instead of
    loading everything at once. This helper is designed for
    convenience, not scalability.

    한국어 설명
    --------------
    주어진 LAS/LAZ 파일을 메모리로 읽어 들여 포인트 클라우드와 속성을 반환합니다.
    좌표는 (N × 3) numpy 배열로 제공되며, 추가적인 속성(예: intensity,
    classification)은 사전 형태로 저장됩니다. LAS 파일에 수억 개의 점이 들어 있을
    수 있으므로 메모리 한계를 고려해야 합니다. 이러한 경우 laspy의 스트리밍
    기능을 사용하여 분할 처리하는 것이 좋습니다. 이 함수는 편의를 위한 것으로,
    대규모 데이터셋에 최적화되어 있지는 않습니다.
    """
    with laspy.open(path) as las:
        header = las.header
        # Read all points. The returned object supports numpy-style
        # slicing. This will load all points into memory.
        points_data = las.read()
        xyz = np.vstack((points_data.x, points_data.y, points_data.z)).T
        attributes: Dict[str, np.ndarray] = {}
        # Add some common attributes if present
        for attr_name in ["intensity", "return_number", "classification"]:
            if hasattr(points_data, attr_name):
                attributes[attr_name] = getattr(points_data, attr_name)
    return PointCloud(points=xyz, attributes=attributes)


def build_kdtree(points: np.ndarray) -> KDTree:
    """Build a KD-tree for fast nearest neighbour queries.

    Parameters
    ----------
    points : numpy.ndarray
        Nx3 array of point coordinates.

    Returns
    -------
    scipy.spatial.KDTree
        A KD-tree object for queries. Raises ImportError if SciPy is
        not installed.

    한국어 설명
    --------------
    주어진 점 좌표 배열로부터 KD‑트리를 생성하여 최근접 이웃 쿼리를 빠르게 수행할
    수 있게 합니다. SciPy가 설치되어 있지 않으면 ImportError가 발생하며, 그 경우
    다른 KD‑트리 구현을 제공해야 합니다.
    """
    if KDTree is None:
        raise ImportError(
            "scipy is required for KD-tree functionality. Install scipy or provide your own KD-tree implementation."
        )
    return KDTree(points)


__all__ = [
    "PointCloud",
    "load_las",
    "build_kdtree",
]