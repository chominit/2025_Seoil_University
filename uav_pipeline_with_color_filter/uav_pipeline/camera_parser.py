"""
camera_parser.py
------------------

This module contains utility functions for parsing camera calibration
parameters from Pix4D output files. Pix4D writes a number of text
files during the initial processing stage of a project. Two of the
most important are:

``calibrated_camera_parameters.txt`` – internal parameters for each
camera model (focal length, principal point, distortion coefficients).

``calibrated_external_camera_parameters.txt`` – external pose for
every image (camera position and orientation angles).

The functions here load those files into Python data structures so
other parts of the pipeline can access them conveniently. The goal is
to shield the rest of the code from the exact formatting of the
Pix4D text outputs.

Example usage::

    from uav_pipeline.camera_parser import (
        parse_internal_parameters,
        parse_external_parameters,
    )
    intrinsics = parse_internal_parameters("path/to/Zenmuse Site B Original_calibrated_camera_parameters.txt")
    extrinsics = parse_external_parameters("path/to/Zenmuse Site B Original_calibrated_external_camera_parameters.txt")
    first_pose = extrinsics[0]
    print(first_pose["image_name"], first_pose["position"], first_pose["angles"])

Note
----
These parsers assume the files follow the standard Pix4D format
documented in the Pix4D knowledge base. If your files deviate from
this structure, you may need to adjust the parsing logic accordingly.

한국어 설명
-----------
이 모듈은 Pix4D에서 내보낸 카메라 보정 파일을 파싱하는 유틸리티 함수들을 제공합니다.
Pix4D는 프로젝트의 초기 처리 단계에서 여러 텍스트 파일을 생성합니다. 그중 핵심적인 두
파일은 다음과 같습니다.

* ``calibrated_camera_parameters.txt`` – 각 카메라 모델에 대한 내부 파라미터를 포함합니다.
  초점거리, 주점, 왜곡 계수 등이 기록되어 있습니다.
* ``calibrated_external_camera_parameters.txt`` – 각 이미지의 외부 위치와 자세 정보를 담습니다.
  카메라의 세계 좌표 위치(X, Y, Z)와 회전각(오메가, 파이, 카파)이 포함됩니다.

이 모듈의 함수들은 이러한 파일을 읽어 파이썬 데이터 구조로 변환합니다. 파이프라인의 다른
부분에서는 Pix4D의 파일 포맷을 신경 쓰지 않고 파라미터에 접근할 수 있도록 하기 위함입니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable


@dataclass
class CameraIntrinsics:
    """Simple container for camera intrinsic parameters.

    Attributes
    ----------
    focal_length_px : float
        Focal length expressed in pixels (optimised value from Pix4D).
    principal_point_px : Tuple[float, float]
        (cx, cy) principal point coordinates in pixels.
    radial_distortion : Tuple[float, float, float]
        Radial distortion coefficients (R1, R2, R3).
    tangential_distortion : Tuple[float, float]
        Tangential distortion coefficients (T1, T2).
    sensor_size_mm : Tuple[float, float] | None
        Physical sensor size in millimetres (width, height) if available.
    focal_length_mm : float | None
        Focal length expressed in millimetres. May be ``None`` when not
        provided in the input file.
    principal_point_mm : Tuple[float, float] | None
        (cx, cy) principal point coordinates in millimetres. May be
        ``None`` when not provided in the input file.

    한국어 설명
    ------------
    이 데이터클래스는 카메라 내부 파라미터를 저장하기 위한 단순한 컨테이너입니다. Pix4D의
    결과 파일에서 얻은 최적화된 값들을 보관하며, 초점거리와 주점은 픽셀 단위와
    밀리미터 단위로 모두 저장됩니다. 또한 방사 왜곡과 접선 왜곡 계수, 센서 크기 등도
    포함됩니다.
    """

    focal_length_px: float
    principal_point_px: Tuple[float, float]
    radial_distortion: Tuple[float, float, float]
    tangential_distortion: Tuple[float, float]
    sensor_size_mm: Tuple[float, float] | None = None
    focal_length_mm: float | None = None
    principal_point_mm: Tuple[float, float] | None = None


@dataclass
class CameraExtrinsics:
    """Container for a single image's exterior orientation parameters.

    Attributes
    ----------
    image_name : str
        Filename of the image corresponding to this pose.
    position : Tuple[float, float, float]
        The X, Y, Z coordinates of the camera centre, expressed in the
        project coordinate system.
    angles : Tuple[float, float, float]
        The orientation angles (Omega, Phi, Kappa) in degrees.
        These follow the Pix4D convention. Omega is rotation around
        the X axis, Phi around Y, and Kappa around Z.

    한국어 설명
    ------------
    이 데이터클래스는 이미지 한 장에 대한 외부 자세 파라미터를 담습니다. ``image_name``은
    해당 이미지 파일명을, ``position``은 카메라 중심의 공간 좌표(X, Y, Z)를 의미합니다.
    ``angles``는 오메가·파이·카파로 표현되는 회전각으로, 각각 X축, Y축, Z축 주위를 회전하는
    값을 도(degree) 단위로 보관합니다.
    """

    image_name: str
    position: Tuple[float, float, float]
    angles: Tuple[float, float, float]


def parse_internal_parameters(path: str) -> Dict[str, CameraIntrinsics]:
    """Parse a Pix4D ``*_calibrated_camera_parameters.txt`` file.

    Pix4D writes a single file listing internal camera parameters for
    every camera model used in a project. The file is a free-form
    textual report. This function searches the file for sections
    beginning with the model name and extracts the optimised camera
    parameters.

    Parameters
    ----------
    path : str
        Path to the calibrated camera parameters file.

    Returns
    -------
    Dict[str, CameraIntrinsics]
        Mapping from camera model names to a :class:`CameraIntrinsics`
        instance containing the optimised parameters.

    Notes
    -----
    This parser relies on regular expressions to locate the focal
    length, principal point and distortion coefficients. It may not
    recover every value present in the file. If you need additional
    parameters (e.g. skew, pixel aspect ratio), extend the patterns
    accordingly.

    한국어 설명
    --------------
    이 함수는 Pix4D가 생성한 ``*_calibrated_camera_parameters.txt`` 파일을 읽어 각
    카메라 모델의 내부 파라미터를 추출합니다. 텍스트 보고서 형식으로 되어 있어서 정규식을
    사용하여 초점거리, 주점, 왜곡 계수를 찾습니다. 반환값은 모델명과 `CameraIntrinsics`
    인스턴스를 매핑한 딕셔너리입니다. 필요한 경우 정규식 패턴을 수정하여 추가적인
    파라미터를 추출할 수 있습니다.
    """
    content = Path(path).read_text(encoding="utf-8", errors="ignore")

    # Regular expression patterns for each parameter. Pix4D reports
    # often follow lines like "Optimized Values" then list values on
    # separate lines. We'll attempt to capture three floats per line.
    model_pattern = re.compile(r"^Camera Model Name\(s\)\s+(.+)$", re.MULTILINE)
    focal_px_pattern = re.compile(r"Optimized Values\s*\n\s*([0-9.]+) \[pixel\]", re.MULTILINE)
    focal_mm_pattern = re.compile(r"Optimized Values\s*\n\s*[0-9.]+ \[pixel\]\n\s*([0-9.]+) \[mm\]", re.MULTILINE)
    principal_x_px_pattern = re.compile(r"Optimized Values\s*\n(?:.+\n){2}\s*([0-9.]+) \[pixel\]", re.MULTILINE)
    principal_x_mm_pattern = re.compile(r"Optimized Values\s*\n(?:.+\n){2}\s*[0-9.]+ \[pixel\]\n\s*([0-9.]+) \[mm\]", re.MULTILINE)
    principal_y_px_pattern = re.compile(r"Optimized Values\s*\n(?:.+\n){4}\s*([0-9.]+) \[pixel\]", re.MULTILINE)
    principal_y_mm_pattern = re.compile(r"Optimized Values\s*\n(?:.+\n){4}\s*[0-9.]+ \[pixel\]\n\s*([0-9.]+) \[mm\]", re.MULTILINE)
    radial_pattern = re.compile(r"Optimized Values(?:.+\n){5}\s*([\-0-9.eE]+)\s+([\-0-9.eE]+)\s+([\-0-9.eE]+)")
    tangential_pattern = re.compile(r"Optimized Values(?:.+\n){5}\s*[\-0-9.eE]+\s+[\-0-9.eE]+\s+[\-0-9.eE]+\s+([\-0-9.eE]+)\s+([\-0-9.eE]+)")
    sensor_pattern = re.compile(r"Sensor Dimensions: ([0-9.]+) \[mm\] x ([0-9.]+) \[mm\]")

    models = {}
    model_names = model_pattern.findall(content)
    if not model_names:
        # Fallback: search for "EXIF ID" lines if model name header missing.
        model_names = re.findall(r"EXIF ID: ([^\n]+)", content)

    # Iterate through each camera model section
    for model_name in model_names:
        # Build a dictionary for capturing values
        focal_px_match = focal_px_pattern.search(content)
        focal_px = float(focal_px_match.group(1)) if focal_px_match else None
        focal_mm_match = focal_mm_pattern.search(content)
        focal_mm = float(focal_mm_match.group(1)) if focal_mm_match else None
        px_x_match = principal_x_px_pattern.search(content)
        cx_px = float(px_x_match.group(1)) if px_x_match else None
        mm_x_match = principal_x_mm_pattern.search(content)
        cx_mm = float(mm_x_match.group(1)) if mm_x_match else None
        py_y_match = principal_y_px_pattern.search(content)
        cy_px = float(py_y_match.group(1)) if py_y_match else None
        mm_y_match = principal_y_mm_pattern.search(content)
        cy_mm = float(mm_y_match.group(1)) if mm_y_match else None
        radial_match = radial_pattern.search(content)
        radial = tuple(float(radial_match.group(i)) for i in range(1, 4)) if radial_match else (0.0, 0.0, 0.0)
        tangential_match = tangential_pattern.search(content)
        tangential = tuple(float(tangential_match.group(i)) for i in range(1, 3)) if tangential_match else (0.0, 0.0)
        sensor_match = sensor_pattern.search(content)
        sensor_size = (float(sensor_match.group(1)), float(sensor_match.group(2))) if sensor_match else None
        # Only add if we have at least focal length and principal point.
        if focal_px is not None and cx_px is not None and cy_px is not None:
            models[model_name.strip()] = CameraIntrinsics(
                focal_length_px=focal_px,
                principal_point_px=(cx_px, cy_px),
                radial_distortion=radial,
                tangential_distortion=tangential,
                sensor_size_mm=sensor_size,
                focal_length_mm=focal_mm,
                principal_point_mm=(cx_mm, cy_mm) if cx_mm is not None and cy_mm is not None else None,
            )
    return models


def parse_external_parameters(path: str) -> List[CameraExtrinsics]:
    """Parse a Pix4D ``*_calibrated_external_camera_parameters.txt`` file.

    The external parameters file lists one line per image. Each line
    contains the image filename followed by six numbers: X, Y, Z,
    Omega, Phi and Kappa. This parser reads the file and returns a
    list of :class:`CameraExtrinsics` instances.

    Parameters
    ----------
    path : str
        Path to the external parameters file.

    Returns
    -------
    List[CameraExtrinsics]
        A list of extrinsic parameters in the order they appear in the
        file.

    한국어 설명
    --------------
    이 함수는 ``*_calibrated_external_camera_parameters.txt`` 파일을 파싱하여
    각 이미지의 외부 파라미터를 읽어옵니다. 각 행에는 이미지 이름과
    카메라 중심 좌표(X, Y, Z), 회전각(Omega, Phi, Kappa)가 포함되어 있습니다. 반환값은
    이러한 정보를 담은 `CameraExtrinsics` 객체의 리스트입니다.
    """
    extrinsics: List[CameraExtrinsics] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("imageName"):
                continue
            parts = line.split()
            if len(parts) != 7:
                # Some lines might be malformed; skip them with a warning.
                continue
            image_name = parts[0]
            try:
                x, y, z = map(float, parts[1:4])
                omega, phi, kappa = map(float, parts[4:7])
            except ValueError:
                # Could not parse floats; skip the entry.
                continue
            extrinsics.append(
                CameraExtrinsics(
                    image_name=image_name,
                    position=(x, y, z),
                    angles=(omega, phi, kappa),
                )
            )
    return extrinsics


__all__ = [
    "CameraIntrinsics",
    "CameraExtrinsics",
    "parse_internal_parameters",
    "parse_external_parameters",
]