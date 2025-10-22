"""
multi_site_pipeline.py
----------------------

이 스크립트는 세 개의 사이트(Site A/B/C)에 대해 전선 등 선형 객체를 추출하는
파이프라인을 실행합니다. 각 사이트는 별도의 이미지 폴더, 카메라 파라미터
파일(내부/외부), 포인트클라우드 파일을 가지며, Site A는 P4R 카메라, Site B와
Site C는 Zenmuse 카메라를 사용합니다. 스크립트는 다음과 같은 단계로
작동합니다.

1. **색상 필터링** – 각 이미지에서 RGB 임계조건(R ≥ 230, G 0~25,
   B 178~204)을 만족하는 픽셀만 전선 후보로 추출합니다.
2. **광선 계산** – 추출된 각 픽셀에 대해 카메라 내부/외부 파라미터를
   이용하여 월드 좌표계의 광선을 계산합니다.
3. **지면 평면 추정 및 교차** – 포인트클라우드에서 지면 평면을 추정하고,
   각 광선과 지면 평면의 교점을 계산합니다. (보다 정밀한
   포인트클라우드-광선 교차 계산을 구현할 수도 있습니다.)
4. **다중 영상 교차검증** – 모든 이미지에서 얻은 3D 교점들을
   cluster하여 최소 N개 이상의 관측을 가진 점만 유지합니다.
5. **LAS 저장** – 필터링된 3D 점들을 LAS 파일로 저장하고
   classification 값을 14(전선)로 설정합니다.

사용자는 `site_configs` 딕셔너리에서 각 사이트의 절대 경로를 수정하여
자신의 환경에 맞게 조정할 수 있습니다. 파이프라인 실행 시
Pillow(PIL), laspy, numpy, uav_pipeline 모듈이 필요합니다.
"""

import os
from typing import Dict, List, Tuple

import numpy as np

try:
    from PIL import Image  # Pillow for image loading
except ImportError as e:
    raise ImportError(
        "Pillow가 설치되어 있지 않습니다. 색상 필터링을 위해 `pip install pillow`로 설치해 주세요."
    ) from e

import laspy

from uav_pipeline.camera_parser import parse_internal_parameters, parse_external_parameters
from uav_pipeline.pointcloud_loader import load_las
from uav_pipeline.ray_casting import (
    ray_from_pixel,
    estimate_ground_plane,
    intersect_ray_with_plane,
)
from uav_pipeline.wire_filter import filter_by_count


def detect_colored_pixels(image_path: str) -> List[Tuple[float, float]]:
    """주어진 이미지에서 색상 조건을 만족하는 픽셀 좌표를 반환합니다.

    R ≥ 230, G 0~25, B 178~204 범위에 해당하는 픽셀을 전선 후보로 간주합니다.

    Parameters
    ----------
    image_path : str
        RGB 이미지 파일의 절대 경로.

    Returns
    -------
    List[Tuple[float, float]]
        조건을 만족하는 (x, y) 픽셀 좌표 리스트. x는 열 인덱스, y는 행 인덱스입니다.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    # boolean mask for desired RGB range
    mask = (
        (arr[:, :, 0] >= 230)
        & (arr[:, :, 1] >= 0)
        & (arr[:, :, 1] <= 25)
        & (arr[:, :, 2] >= 178)
        & (arr[:, :, 2] <= 204)
    )
    y_idxs, x_idxs = np.where(mask)
    return [(float(x), float(y)) for x, y in zip(x_idxs, y_idxs)]


def process_site(
    site_name: str,
    cfg: Dict[str, str],
    min_count: int = 3,
    cluster_eps: float = 0.2,
) -> None:
    """단일 사이트에 대해 전체 파이프라인을 실행합니다.

    Parameters
    ----------
    site_name : str
        사이트 이름(예: "Site_A"). 출력 파일명에 사용됩니다.
    cfg : Dict[str, str]
        이미지 디렉터리, 내부/외부 파라미터 파일, 포인트클라우드 경로를 포함하는 설정 딕셔너리.
        키: ``image_dir``, ``intrinsics``, ``extrinsics``, ``pointcloud``, ``output``.
    min_count : int, optional
        교차검증 시 최소 관측 횟수. 기본값은 3.
    cluster_eps : float, optional
        클러스터링 시 거리 임계값. 기본값은 0.2.
    """
    print(f"\n=== {site_name} 처리 시작 ===")

    # 1. 카메라 파라미터 로딩
    intrinsics_dict = parse_internal_parameters(cfg["intrinsics"])
    if not intrinsics_dict:
        raise RuntimeError(f"카메라 내부 파라미터를 찾을 수 없습니다: {cfg['intrinsics']}")
    # 단일 카메라 모델 가정; 여러 모델이 존재하는 경우 필요에 따라 선택
    intrinsics = next(iter(intrinsics_dict.values()))

    extrinsics_list = parse_external_parameters(cfg["extrinsics"])
    extrinsics_by_name = {e.image_name: e for e in extrinsics_list}

    # 2. 포인트클라우드 로딩 및 지면 평면 추정
    pc = load_las(cfg["pointcloud"])
    ground_indices = None
    if pc.attributes.get("classification") is not None:
        # LAS 분류 코드 2는 지면
        ground_indices = np.where(pc.attributes["classification"] == 2)[0]
    ground_points = pc.points[ground_indices] if ground_indices is not None else pc.points
    plane_point, plane_normal = estimate_ground_plane(ground_points)
    print(f"지면 평면 추정 완료: plane_point={plane_point}, plane_normal={plane_normal}")

    # 3. 이미지에서 색상 조건 픽셀 추출
    detections: Dict[str, List[Tuple[float, float]]] = {}
    image_dir = cfg["image_dir"]
    for image_name in extrinsics_by_name.keys():
        image_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(image_path):
            # 이미지가 존재하지 않으면 건너뜀
            continue
        pixels = detect_colored_pixels(image_path)
        if pixels:
            detections[image_name] = pixels
    if not detections:
        print("해당 사이트에서 조건에 맞는 픽셀을 찾지 못했습니다.")
        return

    # 4. 픽셀 -> 광선 -> 지면 교점 계산
    world_points: List[np.ndarray] = []
    for image_name, pix_list in detections.items():
        extr = extrinsics_by_name.get(image_name)
        if extr is None:
            continue
        for (x_px, y_px) in pix_list:
            origin, direction = ray_from_pixel(x_px, y_px, intrinsics, extr)
            intersection = intersect_ray_with_plane(origin, direction, plane_point, plane_normal)
            if intersection is not None:
                world_points.append(intersection)

    if not world_points:
        print("교점이 생성되지 않았습니다. 파라미터를 확인하세요.")
        return

    world_points_arr = np.vstack(world_points)

    # 5. 다중 영상 교차검증 및 필터링
    filtered_points = filter_by_count(world_points_arr, min_count=min_count, eps=cluster_eps)
    print(f"필터링 후 {len(filtered_points)}개의 클러스터 중심점 유지")

    # 6. LAS 파일 저장
    output_path = cfg.get("output")
    if not output_path:
        output_path = os.path.join(os.getcwd(), f"{site_name}_wires.las")
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = [0.0, 0.0, 0.0]
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x = filtered_points[:, 0]
    las.y = filtered_points[:, 1]
    las.z = filtered_points[:, 2]
    # classification 코드 14는 전선(도체)에 해당함
    try:
        las.classification = np.full(filtered_points.shape[0], 14, dtype=np.uint8)
    except Exception:
        pass
    las.write(output_path)
    print(f"{output_path}에 결과 저장 완료")


def main() -> None:
    """여러 사이트를 처리하기 위한 메인 함수."""
    # 사이트별 절대 경로 설정 (사용자 환경에 맞게 수정 필요)
    site_configs = {
        "Site_A": {
            "image_dir": r"G:\\UAV_RANSAC\\25.09.22_Process_By_C_Drive\\P4R_AI_Model_X\\Site_A",
            "intrinsics": r"G:\\UAV_RANSAC\\Pix4d\\P4R Site A Only Original\\1_initial\\params\\P4R Site A Only Original_calibrated_camera_parameters.txt",
            "extrinsics": r"G:\\UAV_RANSAC\\Pix4d\\P4R Site A Only Original\\1_initial\\params\\P4R Site A Only Original_calibrated_external_camera_parameters.txt",
            "pointcloud": r"G:\\UAV_RANSAC\\Pix4d\\P4R Site A Only Original\\3_densification\\point_cloud.las",
            "output": r"G:\\UAV_RANSAC\\Output\\Site_A_wires.las",
        },
        "Site_B": {
            "image_dir": r"G:\\UAV_RANSAC\\25.09.22_Process_By_C_Drive\\P4R_AI_Model_X\\Site_B",
            "intrinsics": r"G:\\UAV_RANSAC\\Pix4d\\Zenmuse Site B Original\\1_initial\\params\\Zenmuse Site B Original_calibrated_camera_parameters.txt",
            "extrinsics": r"G:\\UAV_RANSAC\\Pix4d\\Zenmuse Site B Original\\1_initial\\params\\Zenmuse Site B Original_calibrated_external_camera_parameters.txt",
            "pointcloud": r"G:\\UAV_RANSAC\\Pix4d\\Zenmuse Site B Original\\3_densification\\point_cloud.las",
            "output": r"G:\\UAV_RANSAC\\Output\\Site_B_wires.las",
        },
        "Site_C": {
            "image_dir": r"G:\\UAV_RANSAC\\25.09.22_Process_By_C_Drive\\P4R_AI_Model_X\\Site_C",
            # Site C는 Zenmuse 카메라를 사용하지만 별도의 파라미터 파일 경로를 지정해야 합니다.
            # 필요에 따라 아래 경로를 수정하세요.
            "intrinsics": r"G:\\UAV_RANSAC\\Pix4d\\Zenmuse Site C Original\\1_initial\\params\\Zenmuse Site C Original_calibrated_camera_parameters.txt",
            "extrinsics": r"G:\\UAV_RANSAC\\Pix4d\\Zenmuse Site C Original\\1_initial\\params\\Zenmuse Site C Original_calibrated_external_camera_parameters.txt",
            "pointcloud": r"G:\\UAV_RANSAC\\Pix4d\\Zenmuse Site C Original\\3_densification\\point_cloud.las",
            "output": r"G:\\UAV_RANSAC\\Output\\Site_C_wires.las",
        },
    }

    for site_name, cfg in site_configs.items():
        try:
            process_site(site_name, cfg, min_count=3, cluster_eps=0.2)
        except Exception as exc:
            print(f"{site_name} 처리 중 오류 발생: {exc}")


if __name__ == "__main__":
    main()