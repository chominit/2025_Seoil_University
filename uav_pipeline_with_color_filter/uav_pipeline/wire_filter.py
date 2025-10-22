"""
wire_filter.py
---------------

This module implements simple filtering of candidate 3D points
representing wires or other elongated objects. The core idea is that
features corresponding to wires will be visible in many images
because the wires protrude above the ground and are seldom occluded.
Conversely, spurious intersections created by clutter in a single
image will typically have a low observation count. By counting how
many times a world coordinate appears in projections across the
dataset and applying a threshold, we retain only points with strong
support.

Functions in this module assume that you have already computed a
collection of candidate 3D points for each image (e.g. via ray
casting) and wish to fuse them across images. A simple approach is
to cluster points within a small spatial tolerance and count the
members in each cluster. Clusters with fewer than a user-defined
threshold are discarded.

Dependencies
------------
* numpy
* sklearn (optional for DBSCAN clustering; a fallback is provided)

한국어 설명
------------
이 모듈은 전선과 같이 길게 늘어선 물체를 나타내는 후보 3차원 점들을 간단히
필터링하는 기능을 구현합니다. 전선과 같은 특징은 여러 이미지에서 관찰되는 반면,
단일 이미지의 잡음은 관측 횟수가 적기 때문에, 동일한 세계 좌표가 여러 이미지에서
몇 번 나타나는지를 세어 임계값 이상인 점만 남깁니다. 이를 위해 작은 거리
임계값으로 군집화하여 각 클러스터의 포인트 개수를 세고, 사용자가 설정한 최소
개수보다 적은 클러스터는 버립니다. DBSCAN이 설치되어 있으면 이를 사용하고,
없을 경우 간단한 거리 기반 방법으로 대체합니다.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None  # type: ignore


def cluster_points(points: np.ndarray, eps: float = 0.1, min_samples: int = 1) -> Tuple[np.ndarray, int]:
    """Cluster 3D points using DBSCAN or a simple distance-based grouping.

    Parameters
    ----------
    points : ndarray
        Nx3 array of candidate point coordinates.
    eps : float, optional
        Maximum distance between points in a cluster. Default is 0.1
        units (choose based on the scale of your coordinate system).
    min_samples : int, optional
        Minimum number of points in a cluster to be considered
        significant. Default is 1. Note that this parameter only
        applies when DBSCAN is available.

    Returns
    -------
    labels : ndarray
        Cluster labels for each input point. Noise points are labelled
        as -1.
    n_clusters : int
        The number of clusters found (excluding noise).

    한국어 설명
    --------------
    3D 점들을 DBSCAN 알고리즘 또는 단순한 거리 기반 그룹핑으로 군집화합니다. ``eps``는
    클러스터로 묶을 최대 거리를, ``min_samples``는 DBSCAN에서 클러스터를 형성하기
    위해 필요한 최소 포인트 수를 의미합니다. 반환값은 각 점에 대한 클러스터
    레이블과 찾아낸 클러스터 수입니다.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points array must be of shape (N, 3)")
    if DBSCAN is not None:
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(points)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return labels, n_clusters
    else:
        # Simple agglomerative clustering based on euclidean distance
        # without any fancy optimisation. O(N^2) complexity. Only use
        # this on small point sets.
        N = points.shape[0]
        labels = -np.ones(N, dtype=int)
        cluster_id = 0
        for i in range(N):
            if labels[i] != -1:
                continue
            # Start new cluster
            labels[i] = cluster_id
            for j in range(i + 1, N):
                if labels[j] == -1 and np.linalg.norm(points[j] - points[i]) < eps:
                    labels[j] = cluster_id
            cluster_id += 1
        return labels, cluster_id


def filter_by_count(points: np.ndarray, min_count: int = 3, eps: float = 0.1) -> np.ndarray:
    """Filter points based on observation count across images.

    This clusters the input points and retains only those clusters
    containing at least ``min_count`` members. The cluster centroid is
    returned as the representative location of the wire.

    Parameters
    ----------
    points : ndarray
        Nx3 array of 3D points from all images.
    min_count : int, optional
        Minimum number of points required in a cluster to be kept.
    eps : float, optional
        Distance threshold for clustering. Tune this value to merge
        nearby points belonging to the same physical object. Units are
        the same as your coordinate system.

    Returns
    -------
    ndarray
        Mx3 array of cluster centroids passing the count threshold.

    한국어 설명
    --------------
    모든 이미지에서 얻은 3D 점들을 군집화하여 각 클러스터의 포인트 개수를 계산한 뒤,
    ``min_count`` 이상인 클러스터만 선택합니다. 이때 ``eps``는 같은 객체로 간주할
    최대 거리이며, 반환값은 선택된 클러스터들의 중심 좌표(M x 3 배열)입니다.
    """
    if points.size == 0:
        return np.empty((0, 3))
    labels, n_clusters = cluster_points(points, eps=eps, min_samples=1)
    unique_labels = set(labels)
    results = []
    for lbl in unique_labels:
        if lbl == -1:
            continue  # skip noise
        indices = np.where(labels == lbl)[0]
        if len(indices) >= min_count:
            cluster_points = points[indices]
            centroid = cluster_points.mean(axis=0)
            results.append(centroid)
    return np.array(results)


__all__ = [
    "cluster_points",
    "filter_by_count",
]