"""
pipeline.py
-----------

High-level orchestration script for extracting wires (or other
foreground objects) from UAV imagery and point clouds using collinearity
equations and simple statistical filtering. The pipeline proceeds in
five stages:

1. **Load camera parameters** – parse Pix4D internal and external
   parameters.
2. **Load point cloud** – read the densified point cloud from a LAS
   file.
3. **Estimate ground plane** – fit a plane to the ground points
   (e.g. class 2 in LAS classification) or entire point cloud.
4. **Compute world coordinates of detections** – for each detection
   pixel in each image, back-project the pixel to a ray and intersect
   with the ground plane. Accumulate all intersection points.
5. **Filter by observation count** – cluster the 3D points and keep
   clusters with at least `min_count` members as candidate wires.

The script is designed to be flexible: you can plug in your own
detection loader and adjust clustering thresholds. By keeping
components modular, you can swap out or extend individual stages
without rewriting the entire pipeline.

Usage
-----
Run this script from the command line::

    python -m uav_pipeline.pipeline \
        --intrinsics path/to/calibrated_camera_parameters.txt \
        --extrinsics path/to/calibrated_external_camera_parameters.txt \
        --pointcloud path/to/densified_point_cloud.las \
        --detections path/to/detections.csv \
        --output wires.las \
        --min-count 3 

The detections file is expected to be a CSV with columns:
``image_name, x_px, y_px`` representing the pixel coordinates of
detected wire points in each image. You can modify the `load_detections`
function to match your detection output format.

Note
----
This pipeline performs all computations on the CPU by default. If you
have a large number of detections and wish to accelerate the ray
intersection, consider vectorising operations with NumPy or porting
parts to PyTorch to leverage GPU acceleration. The modular design
makes such changes straightforward.

한국어 설명
-----------
이 스크립트는 UAV 이미지와 포인트 클라우드에서 전선(또는 다른 전경 객체)을
추출하는 고수준 파이프라인을 구현합니다. 파이프라인은 다음과 같은 다섯 단계를
거칩니다.

1. **카메라 파라미터 불러오기** – Pix4D 내부/외부 파라미터 파일을 파싱합니다.
2. **포인트 클라우드 로딩** – 밀집 포인트 클라우드(LAS 파일)를 읽습니다.
3. **지면 평면 추정** – 분류 코드 2(지면)나 전체 점을 이용해 RANSAC으로 평면을 맞춥니다.
4. **검출 픽셀 역투영** – 각 이미지의 검출 픽셀을 레이로 변환하여 지면 평면과 교차시킵니다.
   얻어진 모든 교점들을 누적합니다.
5. **관측 횟수 기반 필터링** – 3D 점들을 군집화하여 최소 관측 횟수 이상인 클러스터만
   전선 후보로 채택합니다.

각 단계는 모듈형으로 구성되어 있어 사용자 정의 로더를 추가하거나 클러스터링
임계값을 조정하는 등 유연한 확장이 가능합니다. 기본적으로 모든 연산은 CPU에서
수행되지만, NumPy 벡터화나 PyTorch를 사용하여 GPU 가속을 적용할 수 있습니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .camera_parser import (
    CameraIntrinsics,
    CameraExtrinsics,
    parse_internal_parameters,
    parse_external_parameters,
)
from .pointcloud_loader import load_las, build_kdtree
from .ray_casting import (
    estimate_ground_plane,
    intersect_ray_with_plane,
    ray_from_pixel,
)
from .wire_filter import filter_by_count

# Try to import PIL for image loading. If Pillow is not installed,
# filtering detections by color will be unavailable.
try:
    from PIL import Image  # type: ignore
except ImportError:
    Image = None  # type: ignore


def load_detections(path: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load detection pixel coordinates from a CSV file.

    The expected CSV format is ``image_name,x_px,y_px`` per line. Blank
    lines and comment lines starting with ``#`` are ignored.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    Dict[str, List[Tuple[float, float]]]
        Mapping from image filename to a list of (x, y) pixel
        coordinates of detections in that image.

    한국어 설명
    --------------
    CSV 파일에서 검출된 픽셀 좌표를 읽어옵니다. CSV 형식은 한 줄에
    ``image_name,x_px,y_px`` 형식의 세 값이 포함되어 있다고 가정합니다. 빈 줄과
    ``#``로 시작하는 주석 줄은 무시됩니다. 반환값은 이미지 이름을 키로,
    (x, y) 픽셀 좌표의 리스트를 값으로 갖는 딕셔너리입니다.
    """
    detections: Dict[str, List[Tuple[float, float]]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            image_name, x_str, y_str = parts[:3]
            try:
                x_px = float(x_str)
                y_px = float(y_str)
            except ValueError:
                continue
            detections.setdefault(image_name, []).append((x_px, y_px))
    return detections


def filter_detections_by_color(
    detections: Dict[str, List[Tuple[float, float]]],
    image_dir: str,
    r_min: int = 230,
    g_min: int = 0,
    g_max: int = 25,
    b_min: int = 178,
    b_max: int = 204,
) -> Dict[str, List[Tuple[float, float]]]:
    """Filter detection pixels based on RGB color thresholds.

    For each detection, the corresponding image is loaded from
    ``image_dir`` and the pixel value at the detection coordinate is
    inspected. Only detections whose RGB values satisfy
    ``r >= r_min``, ``g_min <= g <= g_max`` and ``b_min <= b <= b_max``
    are retained.

    Parameters
    ----------
    detections : Dict[str, List[Tuple[float, float]]]
        Mapping from image names to lists of pixel coordinates.
    image_dir : str
        Directory containing the image files.
    r_min, g_min, g_max, b_min, b_max : int, optional
        Thresholds for the red, green and blue channels.

    Returns
    -------
    Dict[str, List[Tuple[float, float]]]
        Filtered detections dictionary containing only pixels that
        satisfy the color criteria.

    한국어 설명
    --------------
    검출된 픽셀 좌표의 RGB 색상값을 기준으로 필터링합니다. 각 이미지 파일을
    ``image_dir``에서 찾아 열고, 해당 픽셀의 (R, G, B) 값을 확인하여
    ``r >= r_min``, ``g_min <= g <= g_max``, ``b_min <= b <= b_max`` 조건을 만족하는
    경우에만 그 픽셀을 유지합니다.
    """
    if Image is None:
        raise ImportError(
            "Pillow (PIL) is required for color-based filtering. Install it with `pip install pillow`."
        )
    import os

    filtered: Dict[str, List[Tuple[float, float]]] = {}
    for image_name, pts in detections.items():
        image_path = os.path.join(image_dir, image_name)
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            # Skip this image if it cannot be opened
            continue
        width, height = img.size
        for (x, y) in pts:
            xi = int(round(x))
            yi = int(round(y))
            # Ensure coordinates are within image bounds
            if xi < 0 or yi < 0 or xi >= width or yi >= height:
                continue
            r, g, b = img.getpixel((xi, yi))
            if r >= r_min and g_min <= g <= g_max and b_min <= b <= b_max:
                filtered.setdefault(image_name, []).append((x, y))
    return filtered


def main(args: argparse.Namespace) -> None:
    # 1. Parse camera parameters
    print(f"Loading intrinsics from {args.intrinsics}...")
    intrinsics_dict = parse_internal_parameters(args.intrinsics)
    if not intrinsics_dict:
        raise RuntimeError("No camera intrinsics found in the provided file.")
    # Assume single camera model for simplicity. If multiple models are
    # present, choose the first one. You can modify this to select
    # appropriate model per image based on EXIF ID.
    intrinsics: CameraIntrinsics = next(iter(intrinsics_dict.values()))

    print(f"Loading extrinsics from {args.extrinsics}...")
    extrinsics_list = parse_external_parameters(args.extrinsics)
    extrinsics_by_name: Dict[str, CameraExtrinsics] = {e.image_name: e for e in extrinsics_list}

    # 2. Load point cloud
    print(f"Loading point cloud from {args.pointcloud}...")
    pc = load_las(args.pointcloud)
    # We could filter ground points using classification codes (2 for ground)
    ground_indices = None
    if pc.attributes.get("classification") is not None:
        ground_indices = np.where(pc.attributes["classification"] == 2)[0]
    ground_points = pc.points[ground_indices] if ground_indices is not None else pc.points

    # 3. Estimate ground plane
    print("Estimating ground plane...")
    plane_point, plane_normal = estimate_ground_plane(ground_points)
    print(f"Plane point: {plane_point}, normal: {plane_normal}")

    # 4. Load detections
    print(f"Loading detections from {args.detections}...")
    detections = load_detections(args.detections)
    # Optionally filter detections by RGB color thresholds if image_dir is provided
    if getattr(args, "image_dir", None):
        print(f"Filtering detections by color using images in {args.image_dir}...")
        detections = filter_detections_by_color(
            detections,
            args.image_dir,
            r_min=230,
            g_min=0,
            g_max=25,
            b_min=178,
            b_max=204,
        )

    # 5. Back-project detections
    all_world_points: List[np.ndarray] = []
    for image_name, pixel_list in detections.items():
        extrinsics = extrinsics_by_name.get(image_name)
        if extrinsics is None:
            # Skip detections for images without pose
            continue
        for (x_px, y_px) in pixel_list:
            origin, direction = ray_from_pixel(x_px, y_px, intrinsics, extrinsics)
            intersection = intersect_ray_with_plane(origin, direction, plane_point, plane_normal)
            if intersection is not None:
                all_world_points.append(intersection)
    if not all_world_points:
        print("No intersections computed; check detection inputs.")
        return
    all_world_points_arr = np.vstack(all_world_points)

    # 6. Filter by count
    print(f"Clustering {len(all_world_points_arr)} points and filtering with min_count={args.min_count}...")
    filtered_points = filter_by_count(all_world_points_arr, min_count=args.min_count, eps=args.cluster_eps)
    print(f"Retained {len(filtered_points)} cluster centroids.")

    # 7. Export results as LAS
    if args.output:
        try:
            import laspy
        except ImportError:
            raise ImportError(
                "laspy is required to write output LAS files. Install it with `pip install laspy`."
            )
        # Create a simple LAS file with only XYZ coordinates and no attributes
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = [0, 0, 0]
        header.scales = [0.001, 0.001, 0.001]  # adjust scale to suit your data
        las = laspy.LasData(header)
        las.x = filtered_points[:, 0]
        las.y = filtered_points[:, 1]
        las.z = filtered_points[:, 2]
        # Optionally set classification for wires (e.g. code 14 – wire guard or similar)
        try:
            las.classification = np.full(filtered_points.shape[0], 14, dtype=np.uint8)
        except Exception:
            pass
        las.write(args.output)
        print(f"Wrote {filtered_points.shape[0]} points to {args.output}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Extract wires from UAV imagery and point cloud.")
    parser.add_argument("--intrinsics", required=True, help="Path to calibrated_camera_parameters.txt")
    parser.add_argument("--extrinsics", required=True, help="Path to calibrated_external_camera_parameters.txt")
    parser.add_argument("--pointcloud", required=True, help="Path to densified point cloud LAS file")
    parser.add_argument("--detections", required=True, help="Path to CSV file with detection pixel coordinates")
    parser.add_argument("--min-count", type=int, default=3, help="Minimum number of detections to keep a cluster")
    parser.add_argument("--cluster-eps", type=float, default=0.1, help="Distance threshold for clustering in world units")
    parser.add_argument("--output", help="Output LAS file for filtered 3D points")
    parser.add_argument(
        "--image-dir",
        help=(
            "Directory containing source images. If provided, detections will be filtered based on RGB color "
            "thresholds (R >= 230, G between 0 and 25, B between 178 and 204)."
        ),
    )
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()