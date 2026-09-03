#!/usr/bin/env python3
"""Depth-aware alignment of two or more references to one target camera.

A checkerboard is *not* required in the scene. A nominal multi-camera
calibration may come from measured intrinsics or bounded self-calibration. For
each real scene this program:

1. builds a calibration-based coarse reference-to-target view;
2. uses RoMa v2 only to propose cross-modal correspondences;
3. rejects matches that violate forward/backward and calibrated epipolar
   geometry;
4. triangulates target-view depth independently with every reference camera;
5. robustly fuses the available depths; and
6. reprojects each original reference image once into the target coordinate
   system, with visibility and confidence
   masks.

It deliberately does not apply RoMa's raw dense flow directly to the image.
That avoids the rubber-sheet deformation of doors, walls and other straight
structures.  Strict pixels without reliable geometry remain invalid.  An
optional, separately labelled visual completion uses single-view geometry,
inverse-depth planes and rigid-edge constraints before one final reprojection.

Calibration pose convention
---------------------------
The current calibrator writes ``camera_poses["anchor_to_camera"]``, mapping a
point from the anchor frame into the named camera frame:
``X_camera = R X_anchor + T``.  The reader also accepts legacy reversed key
labels, records which key was used, and checks any supplied camera centre
before matching starts.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


REFERENCE_NAMES = ("reference_a", "reference_b")
PROGRAM_VERSION = "4.5-guarded-frame-pose"


class DepthAlignmentError(RuntimeError):
    pass


def configure_rig(reference_cameras: Sequence[str], target_camera: str, anchor_camera: str) -> None:
    global REFERENCE_NAMES
    references = tuple(reference_cameras)
    if len(references) < 2 or len(set(references)) != len(references):
        raise DepthAlignmentError("--reference-cameras requires at least two unique names")
    if target_camera in references:
        raise DepthAlignmentError("--target-camera must not be a reference camera")
    if anchor_camera not in references:
        raise DepthAlignmentError("--anchor-camera must be a reference camera")
    REFERENCE_NAMES = references


@dataclass(frozen=True)
class CameraModel:
    name: str
    K: np.ndarray
    dist: np.ndarray
    image_size: tuple[int, int]
    pose_from_master: np.ndarray


@dataclass
class CameraCandidate:
    name: str
    depth: np.ndarray
    weight: np.ndarray
    valid: np.ndarray
    reference_match_xy: np.ndarray
    coarse_image: np.ndarray
    coarse_map_xy: np.ndarray
    coarse_mask: np.ndarray
    reference_from_target: np.ndarray
    report: dict[str, Any]


@dataclass
class DepthCompletionResult:
    depth: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    source: np.ndarray
    rigid_edges: np.ndarray
    plane_prior_mask: np.ndarray
    report: dict[str, Any]


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DepthAlignmentError(f"JSON不存在：{path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DepthAlignmentError(f"无法读取JSON：{path} ({exc})") from exc
    if not isinstance(value, dict):
        raise DepthAlignmentError(f"JSON顶层必须是对象：{path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    if not path.is_file():
        raise DepthAlignmentError(f"图像不存在：{path}")
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(encoded, flags)
    except Exception as exc:
        raise DepthAlignmentError(f"无法读取图像：{path} ({exc})") from exc
    if image is None:
        raise DepthAlignmentError(f"OpenCV无法解码：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        raise DepthAlignmentError(f"不支持的图像扩展名：{path}")
    params: list[int] = []
    if suffix in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 96]
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise DepthAlignmentError(f"OpenCV无法编码：{path}")
    encoded.tofile(str(path))


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise DepthAlignmentError(f"不支持的图像形状：{image.shape}")


def robust_uint8(image: np.ndarray, clahe: bool = True) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise DepthAlignmentError("图像全部是NaN/Inf")
    low, high = np.percentile(values[finite], (1.0, 99.0))
    low, high = float(low), float(high)
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low = float(np.min(values[finite]))
        high = float(np.max(values[finite]))
    output = np.zeros(values.shape, dtype=np.uint8)
    if high > low:
        output = np.round(np.clip((values - low) / (high - low), 0.0, 1.0) * 255.0).astype(np.uint8)
    if clahe:
        output = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(output)
    return output


def gradient_feature(gray8: np.ndarray) -> np.ndarray:
    image = np.asarray(gray8, dtype=np.float32) / 255.0
    gx = cv2.Scharr(image, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(image, cv2.CV_32F, 0, 1)
    magnitude = np.log1p(4.0 * cv2.magnitude(gx, gy))
    return cv2.GaussianBlur(magnitude, (3, 3), 0.0)


def cosine_similarity(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = (mask > 0) & np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(valid)) < 32:
        return -1.0
    av = np.asarray(a[valid], dtype=np.float64)
    bv = np.asarray(b[valid], dtype=np.float64)
    av -= float(np.mean(av))
    bv -= float(np.mean(bv))
    denominator = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / denominator) if denominator > 1e-12 else -1.0


def model_representation(color_bgr: np.ndarray, gray8: np.ndarray, mode: str) -> np.ndarray:
    if mode == "rgb":
        return np.ascontiguousarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
    if mode == "gray":
        return np.ascontiguousarray(np.repeat(gray8[..., None], 3, axis=2))
    if mode == "structure":
        grad = robust_uint8(gradient_feature(gray8), clahe=False)
        edges = cv2.Canny(gray8, 50, 130)
        return np.ascontiguousarray(np.stack((gray8, grad, edges), axis=2))
    raise DepthAlignmentError(f"未知representation：{mode}")


def camera_arrays(item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    key = "K" if "K" in item else "intrinsic"
    if key not in item or "dist" not in item or "image_size" not in item:
        raise DepthAlignmentError("相机参数缺少K/intrinsic、dist或image_size")
    K = np.asarray(item[key], dtype=np.float64)
    dist = np.asarray(item["dist"], dtype=np.float64).reshape(-1)
    image_size = tuple(int(value) for value in item["image_size"])
    if K.shape != (3, 3) or len(image_size) != 2:
        raise DepthAlignmentError(f"相机参数形状错误：K={K.shape}, image_size={image_size}")
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(dist)):
        raise DepthAlignmentError("相机K或dist包含NaN/Inf")
    return K, dist, image_size


def pose_matrix(item: dict[str, Any]) -> np.ndarray:
    R = np.asarray(item["R"], dtype=np.float64)
    t = np.asarray(item["T"], dtype=np.float64).reshape(3)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
        raise DepthAlignmentError("相机外参R/T无效")
    orthogonal_error = float(np.linalg.norm(R.T @ R - np.eye(3)))
    determinant = float(np.linalg.det(R))
    if orthogonal_error > 1e-2 or abs(determinant - 1.0) > 1e-2:
        raise DepthAlignmentError(
            f"外参R不是有效旋转矩阵：orthogonal_error={orthogonal_error:.3g}, det={determinant:.6g}"
        )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = R
    matrix[:3, 3] = t
    return matrix


def camera_center_from_direct_pose(matrix: np.ndarray) -> np.ndarray:
    """Camera centre in master coordinates for X_camera = R X_master + T."""
    R = matrix[:3, :3]
    T = matrix[:3, 3]
    return -R.T @ T


def convention_requests_inverse(
    item: dict[str, Any], camera_name: str, master: str
) -> bool | None:
    """Interpret an explicit pose equation when one is present."""
    text = str(item.get("convention", "")).lower().replace(" ", "")
    equation = text.split(";")[0]
    if "=" not in equation:
        return None
    lhs, rhs = equation.split("=", 1)
    camera_tokens = (f"x_{camera_name.lower()}", "x_camera")
    master_tokens = (f"x_{master.lower()}", "x_master")
    lhs_camera = any(token in lhs for token in camera_tokens)
    lhs_master = any(token in lhs for token in master_tokens)
    rhs_camera = any(token in rhs for token in camera_tokens)
    rhs_master = any(token in rhs for token in master_tokens)
    if lhs_camera and rhs_master:
        return False
    if lhs_master and rhs_camera:
        return True
    return None


def pose_from_master(
    poses: dict[str, Any],
    camera_name: str,
    master: str,
    diagnostics: dict[str, Any],
) -> np.ndarray:
    def center_key_for(item: dict[str, Any]) -> str | None:
        for key in (
            "camera_center_in_anchor",
            f"camera_center_in_{master}",
            "camera_center_in_master",
            "camera_center_in_main",
        ):
            if key in item:
                return key
        return None

    if camera_name == master:
        matrix = pose_matrix(poses[master]) if master in poses else np.eye(4, dtype=np.float64)
        if np.linalg.norm(matrix - np.eye(4)) > 1e-5:
            raise DepthAlignmentError(f"主相机{master}的外参不是单位变换")
        diagnostics[camera_name] = {
            "source_key": master if master in poses else "implicit_identity",
            "inverted": False,
            "baseline_from_master": 0.0,
            "center_consistency_error": 0.0,
        }
        return matrix

    direct_key = f"{master}_to_{camera_name}"
    legacy_key = f"{camera_name}_to_{master}"
    if direct_key in poses:
        source_key = direct_key
        item = poses[source_key]
        invert = False
    elif legacy_key in poses:
        source_key = legacy_key
        item = poses[source_key]
        # Historical files produced by this project used camera_to_anchor as a
        # label while storing anchor-to-camera R/T. Preserve that behaviour
        # unless an explicit equation or centre field proves it is a true
        # camera-to-anchor transform.
        invert = False
        convention_choice = convention_requests_inverse(item, camera_name, master)
        if convention_choice is not None:
            invert = convention_choice
        elif center_key_for(item) is not None:
            raw = pose_matrix(item)
            inverse = np.linalg.inv(raw)
            expected = np.asarray(
                item[str(center_key_for(item))], dtype=np.float64
            ).reshape(3)
            direct_error = float(np.linalg.norm(camera_center_from_direct_pose(raw) - expected))
            inverse_error = float(np.linalg.norm(camera_center_from_direct_pose(inverse) - expected))
            if inverse_error + 1e-10 < direct_error:
                invert = True
    elif camera_name in poses:
        source_key = camera_name
        item = poses[source_key]
        invert = bool(convention_requests_inverse(item, camera_name, master) or False)
    else:
        raise DepthAlignmentError(
            f"camera_poses中找不到{direct_key!r}、{legacy_key!r}或"
            f"{camera_name!r}；现有键：{list(poses)}"
        )

    if not isinstance(item, dict):
        raise DepthAlignmentError(f"camera_poses[{source_key!r}]必须是对象")
    raw_matrix = pose_matrix(item)
    matrix = np.linalg.inv(raw_matrix) if invert else raw_matrix
    center = camera_center_from_direct_pose(matrix)
    center_error: float | None = None
    center_key = center_key_for(item)
    if center_key is not None:
        expected = np.asarray(item[center_key], dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(expected)):
            raise DepthAlignmentError(f"{source_key}.{center_key} contains NaN/Inf")
        center_error = float(np.linalg.norm(center - expected))
        tolerance = 1e-5 * max(1.0, float(np.linalg.norm(expected)))
        if center_error > tolerance:
            raise DepthAlignmentError(
                f"{source_key} R/T disagrees with {center_key}: "
                f"误差={center_error:.6g}，可能把外参方向读反"
            )
    diagnostics[camera_name] = {
        "source_key": source_key,
        "inverted": invert,
        "explicit_convention": item.get("convention"),
        "baseline_from_master": float(np.linalg.norm(center)),
        "camera_center_in_master": center,
        "center_consistency_error": center_error,
    }
    return matrix


def load_camera_models(
    calibration: dict[str, Any], master: str, target_name: str, target_size: tuple[int, int]
) -> tuple[dict[str, CameraModel], CameraModel, list[str], dict[str, Any]]:
    cameras = calibration.get("cameras")
    poses = calibration.get("camera_poses")
    if not isinstance(cameras, dict) or not isinstance(poses, dict):
        raise DepthAlignmentError("校准JSON必须包含cameras和camera_poses")
    required = (*REFERENCE_NAMES, target_name)
    missing = [name for name in required if name not in cameras]
    if missing:
        raise DepthAlignmentError(f"校准JSON缺少相机：{missing}")

    warnings_out: list[str] = []
    pose_diagnostics: dict[str, Any] = {}
    models: dict[str, CameraModel] = {}
    for name in REFERENCE_NAMES:
        K, dist, size = camera_arrays(cameras[name])
        models[name] = CameraModel(
            name, K, dist, size,
            pose_from_master(poses, name, master, pose_diagnostics),
        )
        if np.max(np.abs(dist)) < 1e-12:
            warnings_out.append(f"{name}: 畸变系数全部为0；若这是EXIF初值而非实测内参，像素级结果会受限")

    K_s_calib, dist_s, size_s_calib = camera_arrays(cameras[target_name])
    sx = target_size[0] / size_s_calib[0]
    sy = target_size[1] / size_s_calib[1]
    if not math.isclose(sx, sy, rel_tol=0.01, abs_tol=0.01):
        raise DepthAlignmentError(
            f"目标相机标定尺寸{size_s_calib}与目标尺寸{target_size}不是等比例缩放"
        )
    scale = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
    target = CameraModel(
        target_name,
        scale @ K_s_calib,
        dist_s,
        target_size,
        pose_from_master(poses, target_name, master, pose_diagnostics),
    )
    if np.max(np.abs(dist_s)) < 1e-12:
        warnings_out.append("target: 畸变系数全部为0；请确认这是有意固定而不是缺少标定")
    positive_baselines = [
        float(pose_diagnostics[name]["baseline_from_master"])
        for name in (*REFERENCE_NAMES, target_name)
        if name != master and float(pose_diagnostics[name]["baseline_from_master"]) > 1e-12
    ]
    if positive_baselines:
        baseline_ratio = max(positive_baselines) / min(positive_baselines)
        pose_diagnostics["summary"] = {
            "translation_unit": calibration.get("translation_unit", "unspecified"),
            "absolute_scale_observable": calibration.get("absolute_scale_observable"),
            "maximum_minimum_baseline_ratio": float(baseline_ratio),
        }
        if baseline_ratio > 2.5:
            warnings_out.append(
                f"主摄基线最大/最小={baseline_ratio:.3f}，外参尺度可能漂移"
            )
    return models, target, warnings_out, pose_diagnostics


def resize_target(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    target_w, target_h = target_size
    if (w, h) == target_size:
        return image
    if not math.isclose(w / h, target_w / target_h, rel_tol=0.01, abs_tol=0.01):
        raise DepthAlignmentError(
            f"目标相机图片尺寸{w}x{h}与目标{target_w}x{target_h}长宽比不同"
        )
    interpolation = cv2.INTER_AREA if w > target_w else cv2.INTER_CUBIC
    return cv2.resize(image, target_size, interpolation=interpolation)


def identity_grid(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    return np.stack((xx, yy), axis=2)


def normalized_target_rays(model: CameraModel) -> tuple[np.ndarray, np.ndarray]:
    width, height = model.image_size
    pixels = identity_grid((height, width))
    normalized = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2).astype(np.float64), model.K, model.dist
    ).reshape(height, width, 2)
    rays = np.concatenate(
        (normalized, np.ones((height, width, 1), dtype=np.float64)), axis=2
    )
    return normalized.astype(np.float32), rays.astype(np.float32)


def relative_pose(target: CameraModel, source: CameraModel) -> np.ndarray:
    """Return target_from_source: X_target = T @ X_source."""
    return target.pose_from_master @ np.linalg.inv(source.pose_from_master)


def project_points(
    points_source: np.ndarray,
    target_from_source: np.ndarray,
    target_camera: CameraModel,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_source, dtype=np.float64).reshape(-1, 3)
    R = target_from_source[:3, :3]
    t = target_from_source[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    projected, _ = cv2.projectPoints(points, rvec, t, target_camera.K, target_camera.dist)
    transformed = (R @ points.T).T + t
    return projected.reshape(-1, 2).astype(np.float32), transformed[:, 2].astype(np.float32)


def remap_scalar(values: np.ndarray, map_xy: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.remap(
        np.asarray(values),
        map_xy[..., 0],
        map_xy[..., 1],
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def build_coarse_view(
    reference: np.ndarray,
    reference_camera: CameraModel,
    target_camera: CameraModel,
    target_rays: np.ndarray,
    depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_to_reference = relative_pose(reference_camera, target_camera)
    rays = np.asarray(target_rays, dtype=np.float64)
    if math.isinf(depth):
        points = rays.reshape(-1, 3)
        rotation_only = target_to_reference.copy()
        rotation_only[:3, 3] = 0.0
        projected, z = project_points(points, rotation_only, reference_camera)
    else:
        points = (rays * float(depth)).reshape(-1, 3)
        projected, z = project_points(points, target_to_reference, reference_camera)
    height, width = target_rays.shape[:2]
    map_xy = projected.reshape(height, width, 2)
    source_w, source_h = reference_camera.image_size
    valid = (
        (z.reshape(height, width) > 1e-8)
        & (map_xy[..., 0] >= -0.5)
        & (map_xy[..., 0] <= source_w - 0.5)
        & (map_xy[..., 1] >= -0.5)
        & (map_xy[..., 1] <= source_h - 0.5)
    )
    coarse = cv2.remap(
        reference,
        map_xy[..., 0],
        map_xy[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return coarse, map_xy.astype(np.float32), (valid.astype(np.uint8) * 255)


def parse_reference_depth(text: str) -> float | str:
    lowered = text.strip().lower()
    if lowered in ("auto", "infinity", "inf"):
        return "infinity" if lowered in ("infinity", "inf") else "auto"
    try:
        value = float(text)
    except ValueError as exc:
        raise DepthAlignmentError("--reference-depth必须是auto、infinity或正数") from exc
    if not math.isfinite(value) or value <= 0:
        raise DepthAlignmentError("--reference-depth数值必须大于0")
    return value


def select_coarse_view(
    reference: np.ndarray,
    reference_camera: CameraModel,
    target_camera: CameraModel,
    target_rays: np.ndarray,
    target_gradient: np.ndarray,
    reference_depth: float | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[dict[str, float | str]]]:
    reference_from_target = relative_pose(reference_camera, target_camera)
    baseline = float(np.linalg.norm(reference_from_target[:3, 3]))
    if reference_depth == "infinity":
        depths = [math.inf]
    elif isinstance(reference_depth, float):
        depths = [reference_depth]
    else:
        depths = [math.inf]
        if baseline > 1e-9:
            depths.extend(baseline * ratio for ratio in (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 32.0, 64.0))

    best_score = -float("inf")
    best: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    best_depth = math.inf
    score_rows: list[dict[str, float | str]] = []
    for depth in depths:
        coarse, map_xy, mask = build_coarse_view(
            reference, reference_camera, target_camera, target_rays, depth
        )
        coarse_gray = robust_uint8(to_gray(coarse), clahe=True)
        score = cosine_similarity(target_gradient, gradient_feature(coarse_gray), mask)
        score_rows.append({
            "depth": "infinity" if math.isinf(depth) else float(depth),
            "gradient_cosine": float(score),
            "valid_ratio": float(np.count_nonzero(mask) / mask.size),
        })
        if score > best_score:
            best_score = score
            best = (coarse, map_xy, mask)
            best_depth = depth
    if best is None:
        raise DepthAlignmentError(f"{reference_camera.name}: 无法生成粗对齐视图")
    return (*best, best_depth, score_rows)


class RomaRunner:
    def __init__(self, setting: str, allow_cpu: bool):
        try:
            import torch
            from romav2 import RoMaV2
            from romav2.device import device
        except Exception as exc:
            raise DepthAlignmentError(
                "无法导入RoMa v2；请在.venv_roma环境运行。原始错误：" + str(exc)
            ) from exc
        if device.type == "cpu" and not allow_cpu:
            raise DepthAlignmentError("没有检测到CUDA；若确认使用CPU，请添加--allow-cpu")
        self.torch = torch
        self.device = device
        self.setting = setting
        torch.set_float32_matmul_precision("highest")
        device_name = str(device)
        if device.type == "cuda":
            device_name = f"cuda: {torch.cuda.get_device_name(device)}"
        print(f"RoMa v2设备：{device_name}")
        print(f"RoMa v2设置：{setting}")
        print("torch.compile：关闭（Windows无需Triton）")
        try:
            self.model = RoMaV2(RoMaV2.Cfg(compile=False))
            self.model.apply_setting(setting)
        except Exception as exc:
            raise DepthAlignmentError(f"RoMa v2初始化失败：{exc}") from exc

    @staticmethod
    def _extract(prediction: dict[str, Any], key: str) -> np.ndarray | None:
        value = prediction.get(key)
        if value is None:
            return None
        return value[0].detach().float().cpu().numpy()

    def match(self, target_rgb: np.ndarray, coarse_reference_rgb: np.ndarray) -> dict[str, np.ndarray]:
        try:
            prediction_ab = self.model.match(
                np.ascontiguousarray(target_rgb), np.ascontiguousarray(coarse_reference_rgb)
            )
            warp_ab = self._extract(prediction_ab, "warp_AB")
            overlap_ab = self._extract(prediction_ab, "overlap_AB")
            warp_ba = self._extract(prediction_ab, "warp_BA")
            overlap_ba = self._extract(prediction_ab, "overlap_BA")
            del prediction_ab
            if self.device.type == "cuda":
                self.torch.cuda.empty_cache()
            if warp_ab is None or overlap_ab is None:
                raise DepthAlignmentError("RoMa v2没有输出warp_AB/overlap_AB")
            if warp_ba is None or overlap_ba is None:
                print("  当前setting为单向；交换图像执行第二次推理")
                prediction_ba = self.model.match(
                    np.ascontiguousarray(coarse_reference_rgb), np.ascontiguousarray(target_rgb)
                )
                warp_ba = self._extract(prediction_ba, "warp_AB")
                overlap_ba = self._extract(prediction_ba, "overlap_AB")
                del prediction_ba
            if warp_ba is None or overlap_ba is None:
                raise DepthAlignmentError("RoMa v2无法产生反向对应")
            if self.device.type == "cuda":
                self.torch.cuda.empty_cache()
            return {
                "warp_AB_norm": np.asarray(warp_ab, dtype=np.float32),
                "warp_BA_norm": np.asarray(warp_ba, dtype=np.float32),
                "overlap_AB": np.asarray(overlap_ab[..., 0] if overlap_ab.ndim == 3 else overlap_ab, dtype=np.float32),
                "overlap_BA": np.asarray(overlap_ba[..., 0] if overlap_ba.ndim == 3 else overlap_ba, dtype=np.float32),
            }
        except RuntimeError as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                raise DepthAlignmentError(
                    "RoMa v2显存不足；请改用--roma-setting turbo，或关闭占用显存的软件"
                ) from exc
            if "triton" in message.lower():
                raise DepthAlignmentError("程序不应调用Triton；请确认使用的是本文件最新版") from exc
            raise DepthAlignmentError(f"RoMa v2推理失败：{exc}") from exc

    def close(self) -> None:
        try:
            del self.model
            if self.device.type == "cuda":
                self.torch.cuda.empty_cache()
        except Exception:
            pass


def resize_field(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    values = np.asarray(field, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 2:
        raise DepthAlignmentError(f"RoMa warp形状错误：{values.shape}")
    if values.shape[:2] == shape:
        return values.copy()
    return cv2.resize(values, (width, height), interpolation=cv2.INTER_LINEAR)


def resize_scalar(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    values = np.asarray(field, dtype=np.float32).squeeze()
    if values.ndim != 2:
        raise DepthAlignmentError(f"RoMa overlap形状错误：{values.shape}")
    if values.shape == shape:
        return values.copy()
    return cv2.resize(values, (width, height), interpolation=cv2.INTER_LINEAR)


def normalized_grid_to_pixel(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    values = np.asarray(grid, dtype=np.float32)
    output = np.empty_like(values)
    output[..., 0] = ((values[..., 0] + 1.0) * width - 1.0) * 0.5
    output[..., 1] = ((values[..., 1] + 1.0) * height - 1.0) * 0.5
    return output


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def epipolar_error_target_pixels(
    target_xy_normalized: np.ndarray,
    reference_xy_normalized: np.ndarray,
    reference_from_target: np.ndarray,
    target_focal: float,
) -> np.ndarray:
    xs = np.concatenate(
        (np.asarray(target_xy_normalized, dtype=np.float64), np.ones((len(target_xy_normalized), 1))),
        axis=1,
    )
    xp = np.concatenate(
        (np.asarray(reference_xy_normalized, dtype=np.float64), np.ones((len(reference_xy_normalized), 1))),
        axis=1,
    )
    R = reference_from_target[:3, :3]
    t = reference_from_target[:3, 3]
    E = skew(t) @ R
    line_reference = (E @ xs.T).T
    line_target = (E.T @ xp.T).T
    numerator = np.abs(np.sum(xp * line_reference, axis=1))
    d_reference = numerator / np.maximum(np.linalg.norm(line_reference[:, :2], axis=1), 1e-12)
    d_target = numerator / np.maximum(np.linalg.norm(line_target[:, :2], axis=1), 1e-12)
    return (np.sqrt(0.5 * (d_reference * d_reference + d_target * d_target)) * target_focal).astype(np.float32)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (float(np.trace(rotation)) - 1.0) * 0.5
    return math.degrees(math.acos(float(np.clip(value, -1.0, 1.0))))


def vector_angle_deg(vector0: np.ndarray, vector1: np.ndarray) -> float:
    first = np.asarray(vector0, dtype=np.float64).reshape(3)
    second = np.asarray(vector1, dtype=np.float64).reshape(3)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return math.inf
    cosine = float(np.dot(first, second) / denominator)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def balanced_pose_samples(
    target_xy: np.ndarray,
    quality: np.ndarray,
    image_size: tuple[int, int],
    maximum: int,
    grid_shape: tuple[int, int] = (24, 18),
) -> np.ndarray:
    """Select deterministic high-quality matches across the whole target grid."""
    count = len(target_xy)
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    width, height = image_size
    columns, rows = grid_shape
    points = np.asarray(target_xy, dtype=np.float64)
    scores = np.asarray(quality, dtype=np.float64)
    cell_x = np.clip(
        (points[:, 0] * columns / max(width, 1)).astype(int), 0, columns - 1
    )
    cell_y = np.clip(
        (points[:, 1] * rows / max(height, 1)).astype(int), 0, rows - 1
    )
    cells = cell_y * columns + cell_x
    per_cell = max(1, int(math.ceil(maximum / (columns * rows))))
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
    selected: list[int] = []
    cell_counts = np.zeros(columns * rows, dtype=np.int32)
    for raw_index in order:
        index = int(raw_index)
        cell = int(cells[index])
        if cell_counts[cell] >= per_cell:
            continue
        selected.append(index)
        cell_counts[cell] += 1
        if len(selected) >= maximum:
            break
    if len(selected) < maximum:
        used = set(selected)
        for raw_index in order:
            index = int(raw_index)
            if index not in used:
                selected.append(index)
            if len(selected) >= maximum:
                break
    return np.asarray(selected, dtype=np.int64)


def _essential_candidates(value: np.ndarray) -> list[np.ndarray]:
    essential = np.asarray(value, dtype=np.float64)
    if essential.shape == (3, 3):
        return [essential]
    if (
        essential.ndim == 2
        and essential.shape[1] == 3
        and essential.shape[0] % 3 == 0
    ):
        return [
            essential[index : index + 3]
            for index in range(0, essential.shape[0], 3)
        ]
    return []


def _blend_direction(
    first: np.ndarray, second: np.ndarray, strength: float
) -> np.ndarray:
    # Copy because ``first`` is commonly a writable view of prior_pose[:3, 3].
    # In-place normalization must never mutate the calibration transform.
    a = np.asarray(first, dtype=np.float64).reshape(3).copy()
    b = np.asarray(second, dtype=np.float64).reshape(3).copy()
    a /= max(float(np.linalg.norm(a)), 1e-12)
    b /= max(float(np.linalg.norm(b)), 1e-12)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    theta = math.acos(cosine)
    if theta <= 1e-8:
        return a
    sine = math.sin(theta)
    if abs(sine) <= 1e-8:
        mixed = (1.0 - strength) * a + strength * b
    else:
        mixed = (
            math.sin((1.0 - strength) * theta) / sine * a
            + math.sin(strength * theta) / sine * b
        )
    return mixed / max(float(np.linalg.norm(mixed)), 1e-12)


def refine_reference_from_target_pose(
    target_xy_normalized: np.ndarray,
    reference_xy_normalized: np.ndarray,
    target_xy_pixels: np.ndarray,
    quality: np.ndarray,
    prior_pose: np.ndarray,
    target_size: tuple[int, int],
    target_focal: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate a bounded per-frame 5-DoF essential pose around the rig pose.

    Essential geometry cannot recover translation magnitude. The calibrated
    baseline is therefore retained exactly; only rotation and translation
    direction may change, and only after explicit quality and drift gates.
    """
    report: dict[str, Any] = {
        "mode": args.pose_refinement,
        "accepted": False,
        "status": "disabled" if args.pose_refinement == "off" else "not_attempted",
        "baseline_scale_policy": "retain_global_calibration_magnitude",
    }
    if args.pose_refinement == "off":
        return prior_pose.copy(), report
    if len(target_xy_normalized) < 8:
        report["status"] = "too_few_matches"
        return prior_pose.copy(), report

    selected = balanced_pose_samples(
        target_xy_pixels,
        quality,
        target_size,
        args.pose_refine_max_samples,
    )
    points0 = np.asarray(target_xy_normalized, dtype=np.float64)[selected]
    points1 = np.asarray(reference_xy_normalized, dtype=np.float64)[selected]
    threshold_normalized = args.pose_refine_ransac_threshold / max(target_focal, 1.0)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    essential_kwargs = {
        "focal": 1.0,
        "pp": (0.0, 0.0),
        "method": method,
        "prob": 0.999,
        "threshold": threshold_normalized,
    }
    try:
        essential, ransac_mask = cv2.findEssentialMat(
            points0,
            points1,
            maxIters=args.pose_refine_ransac_max_iters,
            **essential_kwargs,
        )
    except (cv2.error, TypeError) as first_error:
        # Some OpenCV Python builds expose an overload without ``maxIters``.
        try:
            essential, ransac_mask = cv2.findEssentialMat(
                points0,
                points1,
                **essential_kwargs,
            )
        except (cv2.error, TypeError) as exc:
            report.update(
                status="essential_estimation_failed",
                error=f"{first_error}; fallback: {exc}",
            )
            return prior_pose.copy(), report
    if essential is None or ransac_mask is None:
        report["status"] = "essential_estimation_failed"
        return prior_pose.copy(), report
    homography_mask = None
    try:
        _homography, homography_mask = cv2.findHomography(
            points0 * target_focal,
            points1 * target_focal,
            method,
            args.pose_refine_homography_threshold,
            maxIters=args.pose_refine_ransac_max_iters,
            confidence=0.999,
        )
    except (cv2.error, TypeError):
        try:
            _homography, homography_mask = cv2.findHomography(
                points0 * target_focal,
                points1 * target_focal,
                cv2.RANSAC,
                args.pose_refine_homography_threshold,
            )
        except (cv2.error, TypeError):
            homography_mask = None
    if homography_mask is None:
        report.update(
            status="homography_degeneracy_check_failed",
            sample_count=int(len(points0)),
            ransac_inliers=int(np.count_nonzero(ransac_mask)),
        )
        return prior_pose.copy(), report
    homography_inliers = int(np.count_nonzero(homography_mask))

    prior_rotation = np.asarray(prior_pose[:3, :3], dtype=np.float64)
    prior_translation = np.asarray(prior_pose[:3, 3], dtype=np.float64)
    baseline = float(np.linalg.norm(prior_translation))
    if baseline <= 1e-12:
        report["status"] = "zero_prior_baseline"
        return prior_pose.copy(), report

    candidate_reports: list[dict[str, Any]] = []
    best: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None = None
    for candidate in _essential_candidates(essential):
        try:
            recovered, rotation, translation, pose_mask = cv2.recoverPose(
                candidate,
                points0,
                points1,
                np.eye(3, dtype=np.float64),
                mask=np.asarray(ransac_mask, dtype=np.uint8).copy(),
            )
        except cv2.error:
            continue
        inliers = np.asarray(pose_mask).reshape(-1) > 0
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / max(len(points0), 1)
        rotation_delta = rotation_angle_deg(rotation @ prior_rotation.T)
        translation_delta = vector_angle_deg(
            translation.reshape(3), prior_translation
        )
        item = {
            "inliers": inlier_count,
            "inlier_ratio": inlier_ratio,
            "recover_pose_inliers": int(recovered),
            "rotation_delta_deg": rotation_delta,
            "translation_direction_delta_deg": translation_delta,
            "homography_to_essential_inlier_ratio": (
                homography_inliers / max(inlier_count, 1)
            ),
        }
        candidate_reports.append(item)
        if (
            inlier_count < args.pose_refine_min_inliers
            or inlier_ratio < args.pose_refine_min_inlier_ratio
            or rotation_delta > args.pose_refine_max_rotation_deg
            or translation_delta > args.pose_refine_max_translation_deg
            or item["homography_to_essential_inlier_ratio"]
            >= args.pose_refine_max_homography_dominance
        ):
            continue
        if best is None or (inlier_count, -rotation_delta, -translation_delta) > (
            best[3]["inliers"],
            -best[3]["rotation_delta_deg"],
            -best[3]["translation_direction_delta_deg"],
        ):
            best = (rotation, translation.reshape(3), inliers, item)

    report.update(
        status="quality_gate_rejected",
        sample_count=int(len(points0)),
        ransac_inliers=int(np.count_nonzero(ransac_mask)),
        homography_inliers=homography_inliers,
        candidates=candidate_reports,
    )
    if best is None:
        return prior_pose.copy(), report

    rotation, translation, inliers, selected_report = best
    delta_rvec, _ = cv2.Rodrigues(rotation @ prior_rotation.T)
    blended_delta, _ = cv2.Rodrigues(
        delta_rvec.reshape(3) * args.pose_refine_strength
    )
    refined_rotation = blended_delta @ prior_rotation
    refined_direction = _blend_direction(
        prior_translation,
        translation,
        args.pose_refine_strength,
    )
    refined_pose = np.eye(4, dtype=np.float64)
    refined_pose[:3, :3] = refined_rotation
    refined_pose[:3, 3] = refined_direction * baseline

    before = epipolar_error_target_pixels(
        points0[inliers], points1[inliers], prior_pose, target_focal
    )
    after = epipolar_error_target_pixels(
        points0[inliers], points1[inliers], refined_pose, target_focal
    )
    before_p50 = float(np.median(before)) if len(before) else math.inf
    before_p95 = float(np.quantile(before, 0.95)) if len(before) else math.inf
    after_p50 = float(np.median(after)) if len(after) else math.inf
    after_p95 = float(np.quantile(after, 0.95)) if len(after) else math.inf
    improvement = (before_p50 - after_p50) / max(before_p50, 1e-6)
    report.update(
        selected=selected_report,
        prior_epipolar_p50_px=before_p50,
        prior_epipolar_p95_px=before_p95,
        refined_epipolar_p50_px=after_p50,
        refined_epipolar_p95_px=after_p95,
        median_improvement_fraction=improvement,
        strength=float(args.pose_refine_strength),
    )
    if (
        not math.isfinite(after_p50)
        or improvement < args.pose_refine_min_improvement
        or after_p95 > before_p95 * 1.05
    ):
        report["status"] = "improvement_gate_rejected"
        return prior_pose.copy(), report
    report.update(status="accepted", accepted=True)
    return refined_pose, report


def triangulate_matches(
    target_xy_normalized: np.ndarray,
    reference_xy_normalized: np.ndarray,
    reference_from_target: np.ndarray,
    target_focal: float,
) -> dict[str, np.ndarray]:
    xs = np.asarray(target_xy_normalized, dtype=np.float64)
    xp = np.asarray(reference_xy_normalized, dtype=np.float64)
    P_s = np.hstack((np.eye(3), np.zeros((3, 1))))
    P_p = reference_from_target[:3, :]
    homogeneous = cv2.triangulatePoints(P_s, P_p, xs.T, xp.T).T
    denominator = homogeneous[:, 3]
    safe = np.abs(denominator) > 1e-12
    Xs = np.full((len(xs), 3), np.nan, dtype=np.float64)
    Xs[safe] = homogeneous[safe, :3] / denominator[safe, None]
    R = reference_from_target[:3, :3]
    t = reference_from_target[:3, 3]
    Xp = (R @ Xs.T).T + t

    projected_s = Xs[:, :2] / Xs[:, 2:3]
    projected_p = Xp[:, :2] / Xp[:, 2:3]
    error_s = np.linalg.norm(projected_s - xs, axis=1) * target_focal
    error_p = np.linalg.norm(projected_p - xp, axis=1) * target_focal
    reprojection = np.sqrt(0.5 * (error_s * error_s + error_p * error_p))

    ray_s = np.concatenate((xs, np.ones((len(xs), 1))), axis=1)
    ray_p = np.concatenate((xp, np.ones((len(xp), 1))), axis=1)
    ray_s /= np.maximum(np.linalg.norm(ray_s, axis=1, keepdims=True), 1e-12)
    ray_p /= np.maximum(np.linalg.norm(ray_p, axis=1, keepdims=True), 1e-12)
    ray_p_in_target = (R.T @ ray_p.T).T
    dot = np.sum(ray_s * ray_p_in_target, axis=1)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return {
        "depth_target_z": Xs[:, 2].astype(np.float32),
        "depth_reference_z": Xp[:, 2].astype(np.float32),
        "reprojection_error_px": reprojection.astype(np.float32),
        "triangulation_angle_deg": np.degrees(angle).astype(np.float32),
    }


def quantiles(values: np.ndarray) -> list[float] | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return [float(v) for v in np.quantile(array, (0.0, 0.25, 0.5, 0.75, 0.95, 1.0))]


def colorize_scalar(values: np.ndarray, valid: np.ndarray, logarithmic: bool = False) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(data)
    output = np.zeros((*data.shape, 3), dtype=np.uint8)
    if not np.any(mask):
        return output
    working = np.log(np.maximum(data, 1e-12)) if logarithmic else data
    low, high = np.quantile(working[mask], (0.02, 0.98))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        low, high = float(np.min(working[mask])), float(np.max(working[mask]))
    normalized = np.zeros(data.shape, dtype=np.uint8)
    if high > low:
        normalized[mask] = np.round(
            np.clip((working[mask] - low) / (high - low), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~mask] = 0
    return colored


def process_camera_candidate(
    name: str,
    reference: np.ndarray,
    reference_camera: CameraModel,
    target_camera: CameraModel,
    target_color: np.ndarray,
    target_gray8: np.ndarray,
    target_normalized: np.ndarray,
    target_rays: np.ndarray,
    target_gradient: np.ndarray,
    runner: RomaRunner,
    args: argparse.Namespace,
    output_dir: Path,
) -> CameraCandidate:
    print(f"\n[{name}] 生成无棋盘粗对齐……")
    coarse, coarse_map, coarse_mask, selected_depth, depth_scores = select_coarse_view(
        reference,
        reference_camera,
        target_camera,
        target_rays,
        target_gradient,
        args.reference_depth_value,
    )
    depth_text = "infinity" if math.isinf(selected_depth) else f"{selected_depth:.6g}"
    print(f"[{name}] 自动参考深度：{depth_text}（标定平移单位）")
    write_image(output_dir / f"{name}_coarse.jpg", coarse)
    write_image(output_dir / f"{name}_coarse_mask.png", coarse_mask)

    coarse_gray8 = robust_uint8(to_gray(coarse), clahe=True)
    target_model = model_representation(target_color, target_gray8, args.representation)
    coarse_model = model_representation(coarse, coarse_gray8, args.representation)
    print(f"[{name}] RoMa全图匹配……")
    raw_prediction = runner.match(target_model, coarse_model)
    shape = target_gray8.shape
    prediction = {
        "warp_AB_norm": resize_field(raw_prediction["warp_AB_norm"], shape),
        "warp_BA_norm": resize_field(raw_prediction["warp_BA_norm"], shape),
        "overlap_AB": np.clip(resize_scalar(raw_prediction["overlap_AB"], shape), 0.0, 1.0),
        "overlap_BA": np.clip(resize_scalar(raw_prediction["overlap_BA"], shape), 0.0, 1.0),
    }
    if args.save_predictions:
        np.savez_compressed(
            output_dir / f"{name}_roma_predictions.npz",
            **prediction,
            setting=np.asarray(args.roma_setting),
            representation=np.asarray(args.representation),
        )

    map_ab = normalized_grid_to_pixel(prediction["warp_AB_norm"], shape)
    map_ba = normalized_grid_to_pixel(prediction["warp_BA_norm"], shape)
    identity = identity_grid(shape)
    sampled_ba_x = remap_scalar(map_ba[..., 0], map_ab)
    sampled_ba_y = remap_scalar(map_ba[..., 1], map_ab)
    fb_error = np.sqrt(
        (sampled_ba_x - identity[..., 0]) ** 2 + (sampled_ba_y - identity[..., 1]) ** 2
    ).astype(np.float32)
    overlap_ab = prediction["overlap_AB"]
    sampled_overlap_ba = remap_scalar(prediction["overlap_BA"], map_ab)
    overlap_combined = np.sqrt(
        np.clip(overlap_ab, 0.0, 1.0) * np.clip(sampled_overlap_ba, 0.0, 1.0)
    ).astype(np.float32)

    coarse_h, coarse_w = shape
    inside_coarse = (
        (map_ab[..., 0] >= -0.5)
        & (map_ab[..., 0] <= coarse_w - 0.5)
        & (map_ab[..., 1] >= -0.5)
        & (map_ab[..., 1] <= coarse_h - 0.5)
    )
    sampled_coarse_valid = remap_scalar(coarse_mask, map_ab, cv2.INTER_NEAREST) > 0
    reference_match = np.stack(
        (
            remap_scalar(coarse_map[..., 0], map_ab),
            remap_scalar(coarse_map[..., 1], map_ab),
        ),
        axis=2,
    ).astype(np.float32)
    reference_w, reference_h = reference_camera.image_size
    inside_reference = (
        (reference_match[..., 0] >= -0.5)
        & (reference_match[..., 0] <= reference_w - 0.5)
        & (reference_match[..., 1] >= -0.5)
        & (reference_match[..., 1] <= reference_h - 0.5)
    )
    initial = (
        inside_coarse
        & sampled_coarse_valid
        & inside_reference
        & (overlap_combined >= args.overlap_threshold)
        & (fb_error <= args.fb_threshold)
    )

    reference_normalized = cv2.undistortPoints(
        reference_match.reshape(-1, 1, 2).astype(np.float64), reference_camera.K, reference_camera.dist
    ).reshape(*shape, 2).astype(np.float32)
    flat_indices = np.flatnonzero(initial.reshape(-1))
    if flat_indices.size < args.minimum_matches:
        raise DepthAlignmentError(
            f"{name}: 初筛只剩{flat_indices.size}个匹配，少于--minimum-matches={args.minimum_matches}"
        )
    xs = target_normalized.reshape(-1, 2)[flat_indices]
    xp = reference_normalized.reshape(-1, 2)[flat_indices]
    target_focal = float(math.sqrt(target_camera.K[0, 0] * target_camera.K[1, 1]))
    prior_reference_from_target = relative_pose(reference_camera, target_camera)
    pose_quality = (
        overlap_combined.reshape(-1)[flat_indices]
        * np.exp(
            -0.5
            * (
                fb_error.reshape(-1)[flat_indices]
                / max(args.fb_threshold, 1e-6)
            )
            ** 2
        )
    )
    reference_from_target, pose_refinement = refine_reference_from_target_pose(
        xs,
        xp,
        identity.reshape(-1, 2)[flat_indices],
        pose_quality,
        prior_reference_from_target,
        target_camera.image_size,
        target_focal,
        args,
    )
    if args.pose_refinement != "off":
        print(
            f"[{name}] 单帧外参修正：{pose_refinement['status']}"
            + (
                "; rotation="
                f"{pose_refinement['selected']['rotation_delta_deg']:.3f}deg, "
                "translation direction="
                f"{pose_refinement['selected']['translation_direction_delta_deg']:.3f}deg"
                if pose_refinement.get("accepted")
                else ""
            )
        )
    epipolar = epipolar_error_target_pixels(xs, xp, reference_from_target, target_focal)
    geometry = triangulate_matches(xs, xp, reference_from_target, target_focal)

    depth_values = geometry["depth_target_z"]
    reprojection = geometry["reprojection_error_px"]
    angle_deg = geometry["triangulation_angle_deg"]
    geometric_valid = (
        np.isfinite(depth_values)
        & (depth_values > 0.0)
        & np.isfinite(geometry["depth_reference_z"])
        & (geometry["depth_reference_z"] > 0.0)
        & np.isfinite(epipolar)
        & (epipolar <= args.epipolar_threshold)
        & np.isfinite(reprojection)
        & (reprojection <= args.reprojection_threshold)
        & np.isfinite(angle_deg)
        & (angle_deg >= args.minimum_angle_deg)
    )
    if args.minimum_depth is not None:
        geometric_valid &= depth_values >= args.minimum_depth
    if args.maximum_depth is not None:
        geometric_valid &= depth_values <= args.maximum_depth

    full_valid = np.zeros(shape[0] * shape[1], dtype=bool)
    full_valid[flat_indices] = geometric_valid
    full_valid = full_valid.reshape(shape)
    depth_map = np.full(shape, np.nan, dtype=np.float32)
    depth_flat = depth_map.reshape(-1)
    accepted_indices = flat_indices[geometric_valid]
    depth_flat[accepted_indices] = depth_values[geometric_valid]

    fb_values = fb_error.reshape(-1)[flat_indices][geometric_valid]
    overlap_values = overlap_combined.reshape(-1)[flat_indices][geometric_valid]
    epi_values = epipolar[geometric_valid]
    reproj_values = reprojection[geometric_valid]
    angle_values = angle_deg[geometric_valid]
    confidence_values = (
        overlap_values
        * np.exp(-0.5 * (fb_values / max(args.fb_threshold, 1e-6)) ** 2)
        * np.exp(-0.5 * (epi_values / max(args.epipolar_threshold, 1e-6)) ** 2)
        * np.exp(-0.5 * (reproj_values / max(args.reprojection_threshold, 1e-6)) ** 2)
        * np.clip(angle_values / max(args.angle_full_weight_deg, 1e-6), 0.05, 1.0)
    ).astype(np.float32)
    weight = np.zeros(shape, dtype=np.float32)
    weight.reshape(-1)[accepted_indices] = confidence_values

    valid_ratio = float(np.count_nonzero(full_valid) / full_valid.size)
    print(
        f"[{name}] 几何有效：{np.count_nonzero(full_valid)}像素 "
        f"({100.0 * valid_ratio:.2f}%)；极线误差中位数="
        f"{float(np.median(epi_values)) if epi_values.size else float('nan'):.3f}px"
    )
    write_image(output_dir / f"{name}_geometry_valid.png", full_valid.astype(np.uint8) * 255)
    write_image(output_dir / f"{name}_candidate_depth.png", colorize_scalar(depth_map, full_valid, logarithmic=True))

    report = {
        "camera": name,
        "reference_image_size": list(reference_camera.image_size),
        "baseline_from_target_calibration_units": float(np.linalg.norm(reference_from_target[:3, 3])),
        "coarse_reference_depth": "infinity" if math.isinf(selected_depth) else float(selected_depth),
        "coarse_depth_scores": depth_scores,
        "initial_match_count": int(flat_indices.size),
        "geometric_valid_count": int(np.count_nonzero(full_valid)),
        "geometric_valid_ratio": valid_ratio,
        "fb_error_px_quantiles": quantiles(fb_error[initial]),
        "epipolar_error_target_px_quantiles": quantiles(epi_values),
        "reprojection_error_target_px_quantiles": quantiles(reproj_values),
        "triangulation_angle_deg_quantiles": quantiles(angle_values),
        "depth_target_z_quantiles": quantiles(depth_map[full_valid]),
        "confidence_quantiles": quantiles(weight[full_valid]),
        "pose_refinement": pose_refinement,
    }
    return CameraCandidate(
        name=name,
        depth=depth_map,
        weight=weight,
        valid=full_valid,
        reference_match_xy=reference_match,
        coarse_image=coarse,
        coarse_map_xy=coarse_map,
        coarse_mask=coarse_mask,
        reference_from_target=reference_from_target,
        report=report,
    )


def fuse_depth_candidates(
    candidates: list[CameraCandidate],
    consistency_ratio: float,
    minimum_views: int,
    minimum_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    depths = np.stack([item.depth for item in candidates], axis=0).astype(np.float32)
    weights = np.stack([item.weight for item in candidates], axis=0).astype(np.float32)
    valid = np.isfinite(depths) & (depths > 0.0) & (weights > 0.0)
    # np.where evaluates both branches.  Replace invalid samples before taking
    # the logarithm so NaN/Inf candidates do not produce runtime warnings.
    safe_depths = np.where(valid, depths, 1.0)
    log_depth = np.where(valid, np.log(safe_depths), np.inf)
    finite_log_depth = np.where(valid, log_depth, 0.0)
    count = np.sum(valid, axis=0).astype(np.uint8)
    # Select the candidate depth whose local consensus has the largest total
    # confidence (a weighted medoid).  In particular, two mutually inconsistent
    # cameras do not get averaged into a plausible-looking but false surface.
    log_tolerance = math.log1p(consistency_ratio)
    center_scores = np.zeros_like(weights, dtype=np.float32)
    center_support = np.zeros_like(weights, dtype=np.uint8)
    for center_index in range(len(candidates)):
        center_valid = valid[center_index]
        within = (
            valid
            & center_valid[None]
            & (
                np.abs(finite_log_depth - finite_log_depth[center_index][None])
                <= log_tolerance
            )
        )
        center_scores[center_index] = np.sum(np.where(within, weights, 0.0), axis=0)
        center_support[center_index] = np.sum(within, axis=0).astype(np.uint8)
    # A tiny support-count tie breaker prefers a two-camera consensus over one
    # unusually confident isolated candidate without changing real weights.
    best_center = np.argmax(center_scores + 1e-6 * center_support, axis=0)
    selected_center_log = np.take_along_axis(
        finite_log_depth, best_center[None], axis=0
    )[0]
    selected_center_valid = np.take_along_axis(
        valid, best_center[None], axis=0
    )[0]
    consistent = (
        valid
        & selected_center_valid[None]
        & (np.abs(finite_log_depth - selected_center_log[None]) <= log_tolerance)
    )
    consistent_weight = np.where(consistent, weights, 0.0)
    weight_sum = np.sum(consistent_weight, axis=0)
    support = np.sum(consistent, axis=0).astype(np.uint8)
    weighted_log_sum = np.sum(consistent_weight * finite_log_depth, axis=0)
    fused = np.full(count.shape, np.nan, dtype=np.float32)
    nonzero = weight_sum > 1e-12
    fused[nonzero] = np.exp(weighted_log_sum[nonzero] / weight_sum[nonzero]).astype(np.float32)
    reliable = nonzero & (support >= minimum_views) & (weight_sum >= minimum_weight)
    fused[~reliable] = np.nan
    confidence = np.zeros(count.shape, dtype=np.float32)
    confidence[reliable] = np.clip(
        weight_sum[reliable] / max(float(len(candidates)), 1.0), 0.0, 1.0
    )
    report = {
        "camera_candidates": [item.name for item in candidates],
        "minimum_views": int(minimum_views),
        "depth_consistency_ratio": float(consistency_ratio),
        "candidate_count_quantiles": quantiles(count),
        "support_count_quantiles_reliable": quantiles(support[reliable]),
        "reliable_pixels": int(np.count_nonzero(reliable)),
        "reliable_ratio": float(np.count_nonzero(reliable) / reliable.size),
        "fused_depth_quantiles": quantiles(fused[reliable]),
        "fused_confidence_quantiles": quantiles(confidence[reliable]),
    }
    return fused, confidence, reliable, support, report


def shifted(array: np.ndarray, dy: int, dx: int, fill: float | bool) -> np.ndarray:
    output = np.full_like(array, fill)
    h, w = array.shape
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_y0 = src_y0 + dy
    dst_y1 = src_y1 + dy
    dst_x0 = src_x0 + dx
    dst_x1 = src_x1 + dx
    if src_y1 > src_y0 and src_x1 > src_x0:
        output[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return output


def fill_small_depth_holes(
    depth: np.ndarray,
    reliable: np.ndarray,
    confidence: np.ndarray,
    guide_gray8: np.ndarray,
    radius: int,
    edge_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filled_depth = np.asarray(depth, dtype=np.float32).copy()
    valid = np.asarray(reliable, dtype=bool).copy()
    filled_confidence = np.asarray(confidence, dtype=np.float32).copy()
    if radius <= 0 or not np.any(valid):
        return filled_depth, valid, filled_confidence
    log_depth = np.zeros(depth.shape, dtype=np.float32)
    log_depth[valid] = np.log(np.maximum(filled_depth[valid], 1e-12))
    guide = np.asarray(guide_gray8, dtype=np.float32)
    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    sigma = max(edge_threshold * 0.5, 1.0)
    for _ in range(radius):
        accumulator = np.zeros(depth.shape, dtype=np.float32)
        weight_sum = np.zeros(depth.shape, dtype=np.float32)
        confidence_sum = np.zeros(depth.shape, dtype=np.float32)
        for dy, dx in neighbors:
            neighbor_valid = shifted(valid, dy, dx, False)
            neighbor_log = shifted(log_depth, dy, dx, 0.0)
            neighbor_guide = shifted(guide, dy, dx, 0.0)
            neighbor_confidence = shifted(filled_confidence, dy, dx, 0.0)
            difference = np.abs(guide - neighbor_guide)
            permitted = neighbor_valid & (difference <= edge_threshold)
            weight = np.where(permitted, np.exp(-difference / sigma), 0.0).astype(np.float32)
            accumulator += weight * neighbor_log
            confidence_sum += weight * neighbor_confidence
            weight_sum += weight
        new_pixels = (~valid) & (weight_sum > 1e-6)
        if not np.any(new_pixels):
            break
        log_depth[new_pixels] = accumulator[new_pixels] / weight_sum[new_pixels]
        filled_confidence[new_pixels] = 0.85 * confidence_sum[new_pixels] / weight_sum[new_pixels]
        valid[new_pixels] = True
    filled_depth[valid] = np.exp(log_depth[valid]).astype(np.float32)
    filled_depth[~valid] = np.nan
    return filled_depth, valid, filled_confidence


def detect_rigid_edge_constraints(
    guide_gray8: np.ndarray,
    minimum_line_length: float,
) -> np.ndarray:
    """Detect strong intensity boundaries and long rigid-looking line segments.

    The result is not used as a segmentation label.  It only lowers the
    smoothness coupling across those pixels during depth completion, so an
    incomplete line cannot permanently isolate a region with no depth seed.
    """
    guide = np.asarray(guide_gray8, dtype=np.uint8)
    filtered = cv2.bilateralFilter(guide, 5, 24.0, 3.0)
    canny = cv2.Canny(filtered, 60, 150, L2gradient=True) > 0
    gx = cv2.Scharr(filtered, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(filtered, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    nonzero = magnitude[magnitude > 0]
    strong_threshold = float(np.percentile(nonzero, 82.0)) if nonzero.size else float("inf")
    strong_edges = canny & (magnitude >= strong_threshold)

    line_mask = np.zeros(guide.shape, dtype=np.uint8)
    try:
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        detected = detector.detect(filtered)
        lines = None if detected is None else detected[0]
        if lines is not None:
            for raw_line in lines.reshape(-1, 4):
                x0, y0, x1, y1 = (float(value) for value in raw_line)
                if math.hypot(x1 - x0, y1 - y0) < minimum_line_length:
                    continue
                cv2.line(
                    line_mask,
                    (int(round(x0)), int(round(y0))),
                    (int(round(x1)), int(round(y1))),
                    255,
                    1,
                    cv2.LINE_8,
                )
    except (AttributeError, cv2.error):
        # Canny still supplies useful curved/object boundaries on minimal
        # OpenCV builds without the line-segment detector.
        pass
    return strong_edges | (line_mask > 0)


def fit_inverse_depth_plane_irls(
    xy: np.ndarray,
    inverse_depth: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    """Fit q=1/Z=a*x+b*y+c, the exact image form of a 3-D plane."""
    coordinates = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    values = np.asarray(inverse_depth, dtype=np.float64).reshape(-1)
    base_weight = np.clip(np.asarray(weight, dtype=np.float64).reshape(-1), 0.02, 1.0)
    finite = (
        np.all(np.isfinite(coordinates), axis=1)
        & np.isfinite(values)
        & (values > 0.0)
        & np.isfinite(base_weight)
    )
    coordinates = coordinates[finite]
    values = values[finite]
    base_weight = base_weight[finite]
    if values.size < 6:
        return None
    design = np.column_stack((coordinates, np.ones(values.size, dtype=np.float64)))
    robust_weight = np.ones(values.size, dtype=np.float64)
    coefficients = np.zeros(3, dtype=np.float64)
    condition = float("inf")
    for _ in range(4):
        total_weight = np.maximum(base_weight * robust_weight, 1e-8)
        weighted_design = design * np.sqrt(total_weight)[:, None]
        weighted_value = values * np.sqrt(total_weight)
        normal = weighted_design.T @ weighted_design
        condition = float(np.linalg.cond(normal))
        if not math.isfinite(condition) or condition > 1e8:
            return None
        coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_value, rcond=None)
        residual = values - design @ coefficients
        median = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - median))) + 1e-9
        normalized = np.abs(residual - median) / (2.5 * scale)
        robust_weight = 1.0 / np.sqrt(1.0 + normalized * normalized)
    residual = values - design @ coefficients
    relative_error = float(
        np.median(np.abs(residual)) / max(float(np.median(values)), 1e-12)
    )
    return coefficients.astype(np.float32), relative_error, condition


def edge_planar_depth_completion(
    hard_depth: np.ndarray,
    hard_valid: np.ndarray,
    hard_confidence: np.ndarray,
    soft_depth: np.ndarray,
    soft_valid: np.ndarray,
    soft_confidence: np.ndarray,
    normalized_xy: np.ndarray,
    guide_gray8: np.ndarray,
    iterations: int,
    edge_sigma: float,
    minimum_line_length: float,
    plane_minimum_points: int,
    plane_maximum_relative_error: float,
    confidence_decay: float,
) -> DepthCompletionResult:
    """Complete inverse depth once, with hard geometry, soft matches and edges.

    Source labels are: 1=hard multi-view/local fill, 2=single-view geometry,
    3=accepted rigid-plane prior, 4=edge-aware extrapolation.  Labels 2--4
    are deliberately kept separate from the quantitative reliable mask.
    """
    depth = np.asarray(hard_depth, dtype=np.float32)
    hard = (
        np.asarray(hard_valid, dtype=bool)
        & np.isfinite(depth)
        & (depth > 0.0)
    )
    if int(np.count_nonzero(hard)) < max(plane_minimum_points, 16):
        raise DepthAlignmentError("可靠深度太少，不能进行受约束全幅补全")
    soft_depth_array = np.asarray(soft_depth, dtype=np.float32)
    soft = (
        np.asarray(soft_valid, dtype=bool)
        & ~hard
        & np.isfinite(soft_depth_array)
        & (soft_depth_array > 0.0)
    )
    hard_q = np.zeros(depth.shape, dtype=np.float32)
    hard_q[hard] = 1.0 / depth[hard]
    soft_q = np.zeros(depth.shape, dtype=np.float32)
    soft_q[soft] = 1.0 / soft_depth_array[soft]

    hard_values = hard_q[hard]
    q_low, q_high = np.percentile(hard_values, (0.5, 99.5))
    q_min = max(float(q_low) * 0.50, 1e-12)
    q_max = max(float(q_high) * 2.00, q_min * 1.01)

    # Navier-Stokes inpainting supplies a finite initialization only.  It is
    # subsequently constrained by the data terms, planes and edge-aware
    # optimization; it is never labelled as measured depth.
    initial = np.zeros(depth.shape, dtype=np.float32)
    initial[hard] = hard_q[hard]
    inpaint_mask = (~hard).astype(np.uint8) * 255
    try:
        initial = cv2.inpaint(initial, inpaint_mask, 5.0, cv2.INPAINT_NS)
    except cv2.error:
        initial = cv2.inpaint(initial, inpaint_mask, 5.0, cv2.INPAINT_TELEA)
    invalid_initial = ~np.isfinite(initial) | (initial <= 0.0)
    initial[invalid_initial] = float(np.median(hard_values))
    initial = np.clip(initial, q_min, q_max)

    soft_conf = np.clip(np.asarray(soft_confidence, dtype=np.float32), 0.0, 1.0)
    soft_blend = np.clip(0.15 + 0.70 * soft_conf, 0.15, 0.85)
    initial[soft] = (
        (1.0 - soft_blend[soft]) * initial[soft]
        + soft_blend[soft] * np.clip(soft_q[soft], q_min, q_max)
    )

    guide = np.asarray(guide_gray8, dtype=np.uint8)
    rigid_edges = detect_rigid_edge_constraints(guide, minimum_line_length)
    free_space = (~rigid_edges).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        free_space, connectivity=8
    )
    plane_prior = np.zeros(depth.shape, dtype=np.float32)
    plane_weight = np.zeros(depth.shape, dtype=np.float32)
    plane_prior_mask = np.zeros(depth.shape, dtype=bool)
    accepted_planes = 0
    rejected_planes = 0
    maximum_fit_samples = 6000
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < plane_minimum_points:
            continue
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        roi = np.s_[top : top + height, left : left + width]
        component_roi = labels[roi] == label
        sample_roi = component_roi & hard[roi]
        sample_positions = np.argwhere(sample_roi)
        if sample_positions.shape[0] < plane_minimum_points:
            continue
        if sample_positions.shape[0] > maximum_fit_samples:
            positions = np.linspace(
                0,
                sample_positions.shape[0] - 1,
                maximum_fit_samples,
                dtype=np.int64,
            )
            sample_positions = sample_positions[positions]
        sample_y = sample_positions[:, 0] + top
        sample_x = sample_positions[:, 1] + left
        xy_samples = np.asarray(normalized_xy, dtype=np.float32)[sample_y, sample_x]
        q_samples = hard_q[sample_y, sample_x]
        confidence_samples = np.clip(
            np.asarray(hard_confidence, dtype=np.float32)[sample_y, sample_x],
            0.02,
            1.0,
        )
        fitted = fit_inverse_depth_plane_irls(
            xy_samples, q_samples, confidence_samples
        )
        if fitted is None:
            rejected_planes += 1
            continue
        coefficients, relative_error, _condition = fitted
        if relative_error > plane_maximum_relative_error:
            rejected_planes += 1
            continue
        component_xy = np.asarray(normalized_xy, dtype=np.float32)[roi][component_roi]
        prediction = (
            component_xy @ coefficients[:2] + float(coefficients[2])
        ).astype(np.float32)
        prediction = np.clip(prediction, q_min, q_max)
        plane_prior_roi = plane_prior[roi]
        plane_prior_roi[component_roi] = prediction
        score = float(
            np.clip(
                1.0 - relative_error / max(plane_maximum_relative_error, 1e-9),
                0.05,
                1.0,
            )
        )
        plane_weight_roi = plane_weight[roi]
        plane_weight_roi[component_roi] = score
        plane_prior_mask_roi = plane_prior_mask[roi]
        plane_prior_mask_roi[component_roi] = True
        accepted_planes += 1

    plane_only = plane_prior_mask & ~hard
    plane_blend = np.clip(0.35 + 0.55 * plane_weight, 0.35, 0.90)
    initial[plane_only] = (
        (1.0 - plane_blend[plane_only]) * initial[plane_only]
        + plane_blend[plane_only] * plane_prior[plane_only]
    )
    initial[hard] = hard_q[hard]

    # Pairwise weights operate on the original grayscale guide.  Long rigid
    # edges are a strong (but not absolute) barrier so an edge component with
    # no seed can still receive a low-confidence estimate.
    guide_float = guide.astype(np.float32)
    sigma = max(float(edge_sigma), 1e-3)
    weight_x = np.exp(
        -np.abs(guide_float[:, 1:] - guide_float[:, :-1]) / sigma
    ).astype(np.float32)
    weight_y = np.exp(
        -np.abs(guide_float[1:, :] - guide_float[:-1, :]) / sigma
    ).astype(np.float32)
    barrier_x = rigid_edges[:, 1:] | rigid_edges[:, :-1]
    barrier_y = rigid_edges[1:, :] | rigid_edges[:-1, :]
    weight_x[barrier_x] *= 0.02
    weight_y[barrier_y] *= 0.02
    weight_x = np.maximum(weight_x, 1e-5)
    weight_y = np.maximum(weight_y, 1e-5)

    unary_target = initial.copy()
    unary_weight = np.full(depth.shape, 0.025, dtype=np.float32)
    unary_weight[plane_prior_mask] = 0.30 + 0.90 * plane_weight[plane_prior_mask]
    unary_weight[soft] = np.maximum(
        unary_weight[soft], 0.20 + 0.80 * soft_conf[soft]
    )
    q = initial.copy()
    for _ in range(iterations):
        accumulator = unary_weight * unary_target
        weight_sum = unary_weight.copy()
        accumulator[:, 1:] += weight_x * q[:, :-1]
        weight_sum[:, 1:] += weight_x
        accumulator[:, :-1] += weight_x * q[:, 1:]
        weight_sum[:, :-1] += weight_x
        accumulator[1:, :] += weight_y * q[:-1, :]
        weight_sum[1:, :] += weight_y
        accumulator[:-1, :] += weight_y * q[1:, :]
        weight_sum[:-1, :] += weight_y
        proposal = accumulator / np.maximum(weight_sum, 1e-8)
        update = ~hard
        q[update] = 0.20 * q[update] + 0.80 * proposal[update]
        q[hard] = hard_q[hard]
        q = np.clip(q, q_min, q_max)

    completed_depth = (1.0 / np.maximum(q, 1e-12)).astype(np.float32)
    completed_valid = np.isfinite(completed_depth) & (completed_depth > 0.0)
    distance = cv2.distanceTransform((~hard).astype(np.uint8), cv2.DIST_L2, 5)
    distance_confidence = np.exp(
        -distance / max(float(confidence_decay), 1e-3)
    ).astype(np.float32)
    completed_confidence = 0.05 + 0.20 * distance_confidence
    completed_confidence[plane_prior_mask] = np.maximum(
        completed_confidence[plane_prior_mask],
        (0.30 + 0.50 * plane_weight[plane_prior_mask])
        * distance_confidence[plane_prior_mask],
    )
    completed_confidence[soft] = np.maximum(
        completed_confidence[soft], 0.20 + 0.55 * soft_conf[soft]
    )
    completed_confidence[hard] = np.clip(
        np.asarray(hard_confidence, dtype=np.float32)[hard], 0.0, 1.0
    )
    completed_confidence = np.clip(completed_confidence, 0.0, 1.0)

    source = np.full(depth.shape, 4, dtype=np.uint8)
    source[plane_prior_mask] = 3
    source[soft] = 2
    source[hard] = 1
    report = {
        "mode": "edge-planar",
        "source_labels": {
            "1": "hard_multiview_or_local_fill",
            "2": "single_view_geometry",
            "3": "accepted_rigid_plane_prior",
            "4": "edge_aware_extrapolation",
        },
        "hard_pixels": int(np.count_nonzero(hard)),
        "single_view_seed_pixels": int(np.count_nonzero(soft)),
        "plane_prior_pixels": int(np.count_nonzero(plane_prior_mask & ~hard & ~soft)),
        "edge_extrapolated_pixels": int(np.count_nonzero(source == 4)),
        "completed_pixels": int(np.count_nonzero(completed_valid)),
        "completed_ratio": float(np.count_nonzero(completed_valid) / completed_valid.size),
        "rigid_edge_pixels": int(np.count_nonzero(rigid_edges)),
        "connected_components": int(component_count - 1),
        "accepted_plane_components": int(accepted_planes),
        "rejected_plane_components": int(rejected_planes),
        "iterations": int(iterations),
        "edge_sigma": float(edge_sigma),
        "minimum_line_length": float(minimum_line_length),
        "plane_minimum_points": int(plane_minimum_points),
        "plane_maximum_relative_error": float(plane_maximum_relative_error),
        "confidence_decay": float(confidence_decay),
        "depth_quantiles": quantiles(completed_depth[completed_valid]),
        "confidence_quantiles": quantiles(completed_confidence[completed_valid]),
    }
    return DepthCompletionResult(
        depth=completed_depth,
        valid=completed_valid,
        confidence=completed_confidence,
        source=source,
        rigid_edges=rigid_edges,
        plane_prior_mask=plane_prior_mask,
        report=report,
    )


def zbuffer_visibility(
    map_xy: np.ndarray,
    depth_in_reference: np.ndarray,
    valid: np.ndarray,
    reference_size: tuple[int, int],
    relative_tolerance: float,
) -> np.ndarray:
    width, height = reference_size
    target_height, target_width = map_xy.shape[:2]
    x = np.asarray(map_xy[..., 0], dtype=np.float32)
    y = np.asarray(map_xy[..., 1], dtype=np.float32)
    z = np.asarray(depth_in_reference, dtype=np.float32)
    base_valid = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & (z > 0.0)
        & (x >= -0.5)
        & (x <= width - 0.5)
        & (y >= -0.5)
        & (y <= height - 0.5)
    )
    # Keep integer conversion well-defined even where the geometric map is
    # invalid; base_valid excludes these replacement coordinates afterwards.
    safe_x = np.where(np.isfinite(x), x, 0.0)
    safe_y = np.where(np.isfinite(y), y, 0.0)
    # Rasterize visibility at the target sampling density.  A 4096x3072 source
    # buffer would be sparse at the target resolution and could miss depth
    # collisions along occlusion boundaries; this conservative buffer also uses
    # much less memory.
    z_width = min(width, target_width)
    z_height = min(height, target_height)
    z_x = (safe_x + 0.5) * (z_width / width) - 0.5
    z_y = (safe_y + 0.5) * (z_height / height) - 0.5
    zbuffer = np.full(z_height * z_width, np.inf, dtype=np.float32)
    for x_values in (np.floor(z_x), np.ceil(z_x)):
        for y_values in (np.floor(z_y), np.ceil(z_y)):
            xi = np.clip(x_values.astype(np.int64), 0, z_width - 1)
            yi = np.clip(y_values.astype(np.int64), 0, z_height - 1)
            selected = base_valid
            indices = (yi[selected] * z_width + xi[selected]).reshape(-1)
            np.minimum.at(zbuffer, indices, z[selected].reshape(-1))
    zbuffer = zbuffer.reshape(z_height, z_width)
    nearest_x = np.clip(np.round(z_x).astype(np.int64), 0, z_width - 1)
    nearest_y = np.clip(np.round(z_y).astype(np.int64), 0, z_height - 1)
    nearest_depth = zbuffer[nearest_y, nearest_x]
    tolerance = np.maximum(np.abs(z) * relative_tolerance, 1e-6)
    return base_valid & (z <= nearest_depth + tolerance)


def render_reference_from_target_depth(
    reference: np.ndarray,
    reference_camera: CameraModel,
    target_camera: CameraModel,
    target_rays: np.ndarray,
    depth: np.ndarray,
    depth_valid: np.ndarray,
    occlusion_tolerance: float,
    reference_from_target_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth_array = np.asarray(depth, dtype=np.float64)
    safe_depth = np.where(depth_valid & np.isfinite(depth_array), depth_array, 1.0)
    Xs = np.asarray(target_rays, dtype=np.float64) * safe_depth[..., None]
    reference_from_target = (
        np.asarray(reference_from_target_override, dtype=np.float64)
        if reference_from_target_override is not None
        else relative_pose(reference_camera, target_camera)
    )
    projected, z_reference = project_points(Xs.reshape(-1, 3), reference_from_target, reference_camera)
    map_xy = projected.reshape(*depth.shape, 2).astype(np.float32)
    z_reference_map = z_reference.reshape(depth.shape)
    visible = zbuffer_visibility(
        map_xy, z_reference_map, depth_valid, reference_camera.image_size, occlusion_tolerance
    )
    aligned = cv2.remap(
        reference,
        map_xy[..., 0],
        map_xy[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    aligned[~visible] = 0
    return aligned, visible.astype(np.uint8) * 255, map_xy, z_reference_map


def edge_overlay(moving_bgr: np.ndarray, fixed_gray8: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    moving_gray = robust_uint8(to_gray(moving_bgr), clahe=True)
    fixed = robust_uint8(fixed_gray8, clahe=True)
    moving_edge = cv2.Canny(moving_gray, 50, 130) > 0
    fixed_edge = cv2.Canny(fixed, 50, 130) > 0
    valid = valid_mask > 0
    moving_edge &= valid
    fixed_edge &= valid
    output = np.zeros((*moving_gray.shape, 3), dtype=np.uint8)
    only_moving = moving_edge & ~fixed_edge
    only_fixed = fixed_edge & ~moving_edge
    both = moving_edge & fixed_edge
    output[only_moving] = (0, 0, 255)
    output[only_fixed] = (255, 255, 0)
    output[both] = (255, 255, 255)
    return output


def alpha_overlay(
    moving_bgr: np.ndarray,
    fixed_bgr: np.ndarray,
    valid_mask: np.ndarray,
    moving_alpha: float,
) -> np.ndarray:
    """Blend aligned reference and target views only where geometry is valid."""
    if moving_bgr.shape != fixed_bgr.shape:
        raise DepthAlignmentError(
            f"透明叠加尺寸不一致：reference={moving_bgr.shape}, target={fixed_bgr.shape}"
        )
    valid = np.asarray(valid_mask) > 0
    blended = cv2.addWeighted(
        moving_bgr, float(moving_alpha), fixed_bgr, 1.0 - float(moving_alpha), 0.0
    )
    output = fixed_bgr.copy()
    output[valid] = blended[valid]
    return output


def confidence_alpha_overlay(
    moving_bgr: np.ndarray,
    fixed_bgr: np.ndarray,
    valid_mask: np.ndarray,
    confidence: np.ndarray,
    moving_alpha: float,
    minimum_alpha_fraction: float,
) -> np.ndarray:
    """Blend continuously; low-confidence completion fades toward target."""
    if moving_bgr.shape != fixed_bgr.shape:
        raise DepthAlignmentError(
            f"置信度叠加尺寸不一致：reference={moving_bgr.shape}, target={fixed_bgr.shape}"
        )
    valid = np.asarray(valid_mask) > 0
    confidence_array = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
    alpha_fraction = minimum_alpha_fraction + (
        1.0 - minimum_alpha_fraction
    ) * confidence_array
    alpha = np.clip(float(moving_alpha) * alpha_fraction, 0.0, 1.0)
    alpha[~valid] = 0.0
    output = (
        moving_bgr.astype(np.float32) * alpha[..., None]
        + fixed_bgr.astype(np.float32) * (1.0 - alpha[..., None])
    )
    return np.round(np.clip(output, 0.0, 255.0)).astype(np.uint8)


def completion_source_visual(source: np.ndarray) -> np.ndarray:
    """BGR legend: green=hard, yellow=single, blue=plane, magenta=extrapolated."""
    labels = np.asarray(source, dtype=np.uint8)
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    output[labels == 1] = (0, 200, 0)
    output[labels == 2] = (0, 220, 255)
    output[labels == 3] = (255, 120, 0)
    output[labels == 4] = (200, 0, 200)
    return output


def checkerboard_overlay(
    moving_bgr: np.ndarray,
    fixed_bgr: np.ndarray,
    valid_mask: np.ndarray,
    tile_size: int,
) -> np.ndarray:
    """Alternate reference/target tiles; invalid areas remain target."""
    height, width = moving_bgr.shape[:2]
    yy, xx = np.indices((height, width))
    choose_reference = ((xx // tile_size + yy // tile_size) % 2 == 0)
    choose_reference &= np.asarray(valid_mask) > 0
    output = fixed_bgr.copy()
    output[choose_reference] = moving_bgr[choose_reference]
    return output


def highres_visual_preview(
    reference: np.ndarray,
    map_xy: np.ndarray,
    valid_mask: np.ndarray,
    target_bgr: np.ndarray,
    scale: int,
    moving_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Upsample the trusted map for a sharper, display-only reference preview.

    This preserves more samples from a high-resolution reference source, but it
    does not add target information or improve the native target-grid geometric
    accuracy. The
    result is therefore explicitly labelled visual_only.
    """
    height, width = map_xy.shape[:2]
    output_size = (width * scale, height * scale)
    map_high = cv2.resize(map_xy, output_size, interpolation=cv2.INTER_LINEAR)
    valid_high = cv2.resize(
        (np.asarray(valid_mask) > 0).astype(np.uint8),
        output_size,
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    # Remove one high-resolution pixel around validity boundaries so linear
    # map interpolation never drags foreground and background across a hole.
    if scale > 1:
        valid_high = cv2.erode(
            valid_high.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
        ) > 0
    aligned = cv2.remap(
        reference,
        map_high[..., 0],
        map_high[..., 1],
        cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    aligned[~valid_high] = 0
    target_high = cv2.resize(target_bgr, output_size, interpolation=cv2.INTER_CUBIC)
    overlay = alpha_overlay(
        aligned, target_high, valid_high.astype(np.uint8) * 255, moving_alpha
    )
    return aligned, overlay


def alignment_score(moving_bgr: np.ndarray, fixed_gray8: np.ndarray, mask: np.ndarray) -> float:
    moving = robust_uint8(to_gray(moving_bgr), clahe=True)
    return cosine_similarity(gradient_feature(moving), gradient_feature(fixed_gray8), mask)


def parse_named_paths(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise DepthAlignmentError(
                f"invalid --reference-image {value!r}; expected CAMERA=PATH"
            )
        if name in output:
            raise DepthAlignmentError(f"duplicate --reference-image for {name}")
        output[name] = Path(raw_path)
    return output


def resolve_scene_paths(args: argparse.Namespace) -> tuple[dict[str, Path], Path]:
    explicit = parse_named_paths(args.reference_image)
    unknown = set(explicit).difference(REFERENCE_NAMES)
    if unknown:
        raise DepthAlignmentError(
            "--reference-image contains unknown cameras: " + ", ".join(sorted(unknown))
        )
    target = args.target_image
    if args.image_root is not None:
        root = args.image_root.resolve()
        for name in REFERENCE_NAMES:
            if name not in explicit:
                explicit[name] = root / name / args.frame
        if target is None:
            target = root / args.target_camera / args.frame
    missing_arguments = [name for name in REFERENCE_NAMES if name not in explicit]
    if missing_arguments or target is None:
        raise DepthAlignmentError(
            "provide --image-root/--frame or explicit image paths; missing: "
            + ", ".join(
                missing_arguments + ([args.target_camera] if target is None else [])
            )
        )
    reference_paths = {name: path.resolve() for name, path in explicit.items()}
    return reference_paths, Path(target).resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="无需场景棋盘格：RoMa对应 + 多相机极线约束 + 多视角深度重投影"
    )
    parser.add_argument("--version", action="version", version=PROGRAM_VERSION)
    parser.add_argument("--calibration", required=True, type=Path, help="多相机内外参JSON")
    parser.add_argument("--reference-cameras", nargs="+", required=True)
    parser.add_argument("--target-camera", required=True)
    parser.add_argument("--anchor-camera", required=True)
    parser.add_argument("--image-root", type=Path, help="contains one directory per camera")
    parser.add_argument("--frame", default="frame_000.jpg", help="--image-root模式下的共同文件名")
    parser.add_argument("--reference-image", action="append", default=[], metavar="CAMERA=PATH")
    parser.add_argument("--target-image", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-width", type=int)
    parser.add_argument("--target-height", type=int)
    parser.add_argument(
        "--roma-setting",
        choices=("precise", "mega1500", "scannet1500", "wxbs", "satast", "base", "fast", "turbo"),
        default="fast",
    )
    parser.add_argument("--representation", choices=("gray", "rgb", "structure"), default="gray")
    parser.add_argument(
        "--pose-refinement",
        choices=("off", "essential"),
        default="off",
        help="guarded per-frame rotation/translation-direction refinement around calibration",
    )
    parser.add_argument("--pose-refine-ransac-threshold", type=float, default=1.5)
    parser.add_argument("--pose-refine-ransac-max-iters", type=int, default=10000)
    parser.add_argument("--pose-refine-homography-threshold", type=float, default=2.0)
    parser.add_argument(
        "--pose-refine-max-homography-dominance",
        type=float,
        default=0.95,
        help="reject pose updates explainable almost entirely by one homography",
    )
    parser.add_argument("--pose-refine-max-samples", type=int, default=5000)
    parser.add_argument("--pose-refine-min-inliers", type=int, default=80)
    parser.add_argument("--pose-refine-min-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--pose-refine-max-rotation-deg", type=float, default=3.0)
    parser.add_argument("--pose-refine-max-translation-deg", type=float, default=8.0)
    parser.add_argument("--pose-refine-min-improvement", type=float, default=0.05)
    parser.add_argument("--pose-refine-strength", type=float, default=1.0)
    parser.add_argument("--reference-depth", default="auto", help="auto、infinity或标定平移单位的正数")
    parser.add_argument("--overlap-threshold", type=float, default=0.10)
    parser.add_argument("--fb-threshold", type=float, default=2.0, help="RoMa前后向误差，目标相机像素")
    parser.add_argument("--epipolar-threshold", type=float, default=1.5, help="极线误差，目标相机像素等效值")
    parser.add_argument("--reprojection-threshold", type=float, default=1.5, help="三角化重投影误差")
    parser.add_argument("--minimum-angle-deg", type=float, default=0.02, help="最小三角化夹角")
    parser.add_argument("--angle-full-weight-deg", type=float, default=0.5)
    parser.add_argument("--minimum-depth", type=float)
    parser.add_argument("--maximum-depth", type=float)
    parser.add_argument("--minimum-matches", type=int, default=200)
    parser.add_argument("--depth-consistency", type=float, default=0.20, help="多相机深度相对容差")
    parser.add_argument("--minimum-views", type=int, default=2)
    parser.add_argument("--minimum-fused-weight", type=float, default=0.05)
    parser.add_argument("--fill-radius", type=int, default=0, help="沿目标相机边缘约束填补的小孔半径；首轮验证保持0")
    parser.add_argument("--fill-edge-threshold", type=float, default=18.0, help="填补不得跨越的灰度边缘阈值")
    parser.add_argument(
        "--completion-mode",
        choices=("off", "edge-planar"),
        default="off",
        help="可视化补全：用单视角几何、刚体平面与边缘约束补充严格mask外区域",
    )
    parser.add_argument("--completion-iterations", type=int, default=80)
    parser.add_argument(
        "--completion-edge-sigma",
        type=float,
        default=12.0,
        help="越小越不容易跨越目标相机强边缘",
    )
    parser.add_argument(
        "--completion-line-min-length",
        type=float,
        default=28.0,
        help="作为刚体边缘约束的最短线段，单位为目标相机像素",
    )
    parser.add_argument("--completion-plane-min-points", type=int, default=40)
    parser.add_argument(
        "--completion-plane-residual",
        type=float,
        default=0.035,
        help="接受局部刚体平面的最大逆深度相对中位残差",
    )
    parser.add_argument("--completion-confidence-decay", type=float, default=45.0)
    parser.add_argument(
        "--completion-min-alpha",
        type=float,
        default=0.25,
        help="全幅置信度叠加中最低参考相机透明度占目标alpha的比例",
    )
    parser.add_argument("--occlusion-tolerance", type=float, default=0.01, help="Z-buffer相对深度容差")
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.50,
        help="透明叠加中参考相机图权重；0为纯目标相机，1为纯参考相机",
    )
    parser.add_argument(
        "--checker-tile-size",
        type=int,
        default=32,
        help="参考相机/目标相机棋盘叠加的格宽（目标相机像素）",
    )
    parser.add_argument(
        "--preview-scale",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="额外高分辨率查看图倍数；只改善参考相机显示细节，不提高标定精度",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--allow-unaccepted-calibration",
        action="store_true",
        help="仅诊断：允许读取accepted_for_use=false或forced_output=true的校准",
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    configure_rig(args.reference_cameras, args.target_camera, args.anchor_camera)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if (args.target_width is None) != (args.target_height is None):
        raise DepthAlignmentError("set both --target-width and --target-height, or neither")
    if args.target_width is not None and min(args.target_width, args.target_height) <= 0:
        raise DepthAlignmentError("target dimensions must be positive")
    if not 1 <= args.minimum_views <= len(REFERENCE_NAMES):
        raise DepthAlignmentError(
            f"--minimum-views must be between 1 and {len(REFERENCE_NAMES)}"
        )
    for name in (
        "overlap_threshold",
        "fb_threshold",
        "epipolar_threshold",
        "reprojection_threshold",
        "minimum_angle_deg",
        "depth_consistency",
        "minimum_fused_weight",
        "occlusion_tolerance",
    ):
        if float(getattr(args, name)) < 0:
            raise DepthAlignmentError(f"--{name.replace('_', '-')}不能小于0")
    if args.fill_radius < 0 or args.minimum_matches < 8:
        raise DepthAlignmentError("--fill-radius必须非负，--minimum-matches至少为8")
    if args.pose_refine_max_samples < 8 or args.pose_refine_min_inliers < 8:
        raise DepthAlignmentError("单帧外参修正的样本数和最少内点数必须至少为8")
    if args.pose_refine_min_inliers > args.pose_refine_max_samples:
        raise DepthAlignmentError("单帧外参修正的最少内点数不能超过最大样本数")
    if args.pose_refine_ransac_max_iters < 1:
        raise DepthAlignmentError("--pose-refine-ransac-max-iters必须大于0")
    if not 0.0 < args.pose_refine_ransac_threshold:
        raise DepthAlignmentError("--pose-refine-ransac-threshold必须大于0")
    if not 0.0 < args.pose_refine_homography_threshold:
        raise DepthAlignmentError("--pose-refine-homography-threshold必须大于0")
    if not 0.0 < args.pose_refine_max_homography_dominance <= 1.5:
        raise DepthAlignmentError(
            "--pose-refine-max-homography-dominance必须在(0,1.5]内"
        )
    if not 0.0 <= args.pose_refine_min_inlier_ratio <= 1.0:
        raise DepthAlignmentError("--pose-refine-min-inlier-ratio必须在[0,1]内")
    if not 0.0 <= args.pose_refine_min_improvement < 1.0:
        raise DepthAlignmentError("--pose-refine-min-improvement必须在[0,1)内")
    if not 0.0 < args.pose_refine_strength <= 1.0:
        raise DepthAlignmentError("--pose-refine-strength必须在(0,1]内")
    if (
        args.pose_refine_max_rotation_deg <= 0
        or args.pose_refine_max_translation_deg <= 0
    ):
        raise DepthAlignmentError("单帧外参修正的最大漂移角必须大于0")
    if args.completion_iterations < 1:
        raise DepthAlignmentError("--completion-iterations至少为1")
    if args.completion_edge_sigma <= 0 or args.completion_line_min_length <= 0:
        raise DepthAlignmentError("补全边缘sigma和最短线段长度必须大于0")
    if args.completion_plane_min_points < 6:
        raise DepthAlignmentError("--completion-plane-min-points至少为6")
    if args.completion_plane_residual <= 0 or args.completion_confidence_decay <= 0:
        raise DepthAlignmentError("补全平面残差和置信度衰减必须大于0")
    if not 0.0 <= args.completion_min_alpha <= 1.0:
        raise DepthAlignmentError("--completion-min-alpha必须在[0,1]内")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise DepthAlignmentError("--overlay-alpha必须在[0,1]内")
    if args.checker_tile_size <= 0:
        raise DepthAlignmentError("--checker-tile-size必须大于0")
    if args.minimum_depth is not None and args.minimum_depth <= 0:
        raise DepthAlignmentError("--minimum-depth必须大于0")
    if args.maximum_depth is not None and args.maximum_depth <= 0:
        raise DepthAlignmentError("--maximum-depth必须大于0")
    if (
        args.minimum_depth is not None
        and args.maximum_depth is not None
        and args.minimum_depth >= args.maximum_depth
    ):
        raise DepthAlignmentError("--minimum-depth必须小于--maximum-depth")
    args.reference_depth_value = parse_reference_depth(args.reference_depth)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    calibration_path = args.calibration.resolve()
    output_dir = args.output_dir.resolve()
    reference_paths, target_path = resolve_scene_paths(args)
    target_input = read_image(target_path, cv2.IMREAD_COLOR)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise DepthAlignmentError(f"输出目录非空：{output_dir}；确认后添加--overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = read_json(calibration_path)
    if tuple(calibration.get("reference_cameras", ())) != REFERENCE_NAMES:
        raise DepthAlignmentError("calibration reference camera roles do not match CLI roles")
    if calibration.get("target_camera") != args.target_camera:
        raise DepthAlignmentError("calibration target camera does not match --target-camera")
    if calibration.get("anchor_camera") != args.anchor_camera:
        raise DepthAlignmentError("calibration anchor camera does not match --anchor-camera")
    accepted_status = calibration.get("accepted_for_use")
    forced_status = bool(calibration.get("forced_output", False))
    if (accepted_status is False or forced_status) and not args.allow_unaccepted_calibration:
        raise DepthAlignmentError(
            "校准JSON未通过正式质量门或来自--force-accept；"
            "请改用 runs/calibration/calibration/rig_calibration.json。"
            "仅排错时可添加--allow-unaccepted-calibration"
        )
    target_size = (
        (args.target_width, args.target_height)
        if args.target_width is not None
        else (target_input.shape[1], target_input.shape[0])
    )
    reference_models, target_model, calibration_warnings, pose_diagnostics = load_camera_models(
        calibration, args.anchor_camera, args.target_camera, target_size
    )
    if accepted_status is None:
        calibration_warnings.append("校准JSON没有accepted_for_use字段，无法自动确认质量门")
    print("外参解析：")
    for name in (*REFERENCE_NAMES, args.target_camera):
        item = pose_diagnostics[name]
        print(
            f"  {name:10s} key={item['source_key']} "
            f"inverted={item['inverted']} "
            f"baseline={item['baseline_from_master']:.6g}"
        )
    summary = pose_diagnostics.get("summary", {})
    if "maximum_minimum_baseline_ratio" in summary:
        print(
            "  基线最大/最小="
            f"{summary['maximum_minimum_baseline_ratio']:.4f}；"
            f"单位={summary.get('translation_unit')}"
        )
    for warning_text in calibration_warnings:
        print("警告：" + warning_text)

    references: dict[str, np.ndarray] = {}
    for name in REFERENCE_NAMES:
        image = read_image(reference_paths[name], cv2.IMREAD_COLOR)
        actual_size = (image.shape[1], image.shape[0])
        if actual_size != reference_models[name].image_size:
            raise DepthAlignmentError(
                f"{name}原图尺寸{actual_size}与内参尺寸{reference_models[name].image_size}不一致"
            )
        references[name] = image
    target_color = resize_target(target_input, target_size)
    target_gray8 = robust_uint8(to_gray(target_color), clahe=True)
    target_gradient = gradient_feature(target_gray8)
    target_normalized, target_rays = normalized_target_rays(target_model)
    write_image(output_dir / "target_reference.jpg", target_color)

    runner = RomaRunner(args.roma_setting, args.allow_cpu)
    candidates: list[CameraCandidate] = []
    failures: dict[str, str] = {}
    try:
        for name in REFERENCE_NAMES:
            try:
                candidate = process_camera_candidate(
                    name,
                    references[name],
                    reference_models[name],
                    target_model,
                    target_color,
                    target_gray8,
                    target_normalized,
                    target_rays,
                    target_gradient,
                    runner,
                    args,
                    output_dir,
                )
                candidates.append(candidate)
            except DepthAlignmentError as exc:
                failures[name] = str(exc)
                print(f"警告：[{name}] 跳过：{exc}")
    finally:
        runner.close()

    if len(candidates) < args.minimum_views:
        raise DepthAlignmentError(
            f"只有{len(candidates)}个相机产生深度，少于--minimum-views={args.minimum_views}；失败={failures}"
        )

    fused_depth, fused_confidence, reliable, support, fusion_report = fuse_depth_candidates(
        candidates,
        args.depth_consistency,
        args.minimum_views,
        args.minimum_fused_weight,
    )
    reliable_ratio = float(np.count_nonzero(reliable) / reliable.size)
    print(f"\n多视角可靠深度：{100.0 * reliable_ratio:.2f}%")
    if reliable_ratio < 0.005:
        raise DepthAlignmentError(
            "多视角可靠深度不足0.5%；通常表示内外参方向/数值不准，或跨模态匹配失败"
        )

    render_depth, render_valid, render_confidence = fill_small_depth_holes(
        fused_depth,
        reliable,
        fused_confidence,
        target_gray8,
        args.fill_radius,
        args.fill_edge_threshold,
    )
    filled_only = render_valid & ~reliable
    print(
        f"边缘约束小孔填补后：{100.0 * np.count_nonzero(render_valid) / render_valid.size:.2f}% "
        f"（新增{100.0 * np.count_nonzero(filled_only) / filled_only.size:.2f}%）"
    )

    completion_result: DepthCompletionResult | None = None
    completion_single_view_report: dict[str, Any] | None = None
    if args.completion_mode == "edge-planar":
        (
            single_view_depth,
            single_view_confidence,
            single_view_valid,
            _single_view_support,
            completion_single_view_report,
        ) = fuse_depth_candidates(
            candidates,
            args.depth_consistency,
            1,
            min(args.minimum_fused_weight, 0.01),
        )
        completion_guide = robust_uint8(to_gray(target_color), clahe=False)
        completion_result = edge_planar_depth_completion(
            render_depth,
            render_valid,
            render_confidence,
            single_view_depth,
            single_view_valid,
            single_view_confidence,
            target_normalized,
            completion_guide,
            args.completion_iterations,
            args.completion_edge_sigma,
            args.completion_line_min_length,
            args.completion_plane_min_points,
            args.completion_plane_residual,
            args.completion_confidence_decay,
        )
        completion_result.report["single_view_fusion"] = completion_single_view_report
        print(
            "受约束深度补全："
            f"{100.0 * np.count_nonzero(completion_result.valid) / completion_result.valid.size:.2f}%；"
            f"单视角种子={completion_result.report['single_view_seed_pixels']}；"
            f"接受刚体平面={completion_result.report['accepted_plane_components']}"
        )

    write_image(output_dir / "depth_reliable_mask.png", reliable.astype(np.uint8) * 255)
    write_image(output_dir / "depth_filled_only_mask.png", filled_only.astype(np.uint8) * 255)
    write_image(output_dir / "depth_render_mask.png", render_valid.astype(np.uint8) * 255)
    write_image(output_dir / "depth_target_color.png", colorize_scalar(render_depth, render_valid, logarithmic=True))
    write_image(
        output_dir / "depth_confidence.png",
        np.round(np.clip(render_confidence, 0.0, 1.0) * 255.0).astype(np.uint8),
    )
    support_visual = np.round(
        np.clip(support.astype(np.float32) / len(REFERENCE_NAMES), 0.0, 1.0)
        * 255.0
    ).astype(np.uint8)
    write_image(output_dir / "depth_support_count.png", support_visual)
    if completion_result is not None:
        write_image(
            output_dir / "depth_completed_edge_planar.png",
            colorize_scalar(
                completion_result.depth,
                completion_result.valid,
                logarithmic=True,
            ),
        )
        write_image(
            output_dir / "depth_completion_confidence.png",
            np.round(np.clip(completion_result.confidence, 0.0, 1.0) * 255.0).astype(np.uint8),
        )
        write_image(
            output_dir / "depth_completion_source.png",
            completion_source_visual(completion_result.source),
        )
        write_image(
            output_dir / "depth_completion_source_labels.png",
            completion_result.source.astype(np.uint8),
        )
        write_image(
            output_dir / "depth_rigid_edge_constraints.png",
            completion_result.rigid_edges.astype(np.uint8) * 255,
        )
        write_image(
            output_dir / "depth_plane_prior_mask.png",
            completion_result.plane_prior_mask.astype(np.uint8) * 255,
        )

    map_payload: dict[str, np.ndarray] = {
        "target_depth_z": render_depth.astype(np.float32),
        "target_depth_reliable_mask": reliable.astype(np.uint8),
        "target_depth_render_mask": render_valid.astype(np.uint8),
        "target_depth_confidence": render_confidence.astype(np.float32),
        "target_depth_support_count": support.astype(np.uint8),
    }
    if completion_result is not None:
        map_payload.update(
            {
                "target_depth_completed_z_visual_only": completion_result.depth.astype(np.float32),
                "target_depth_completed_valid_mask": completion_result.valid.astype(np.uint8),
                "target_depth_completion_confidence": completion_result.confidence.astype(np.float32),
                "target_depth_completion_source": completion_result.source.astype(np.uint8),
                "target_depth_rigid_edge_constraints": completion_result.rigid_edges.astype(np.uint8),
                "target_depth_plane_prior_mask": completion_result.plane_prior_mask.astype(np.uint8),
            }
        )
    rendering_reports: dict[str, Any] = {}
    for name in REFERENCE_NAMES:
        candidate_lookup = next((item for item in candidates if item.name == name), None)
        frame_reference_from_target = (
            candidate_lookup.reference_from_target
            if candidate_lookup is not None
            else relative_pose(reference_models[name], target_model)
        )
        map_payload[f"reference_from_target__{name}"] = (
            frame_reference_from_target.astype(np.float64)
        )
        aligned, visible_mask, map_xy, z_reference = render_reference_from_target_depth(
            references[name],
            reference_models[name],
            target_model,
            target_rays,
            render_depth,
            render_valid,
            args.occlusion_tolerance,
            frame_reference_from_target,
        )
        reliable_visible = (visible_mask > 0) & reliable
        preview = aligned.copy()
        coarse_valid = np.zeros_like(visible_mask, dtype=bool)
        if candidate_lookup is not None:
            invalid = visible_mask == 0
            preview[invalid] = candidate_lookup.coarse_image[invalid]
            coarse_valid = candidate_lookup.coarse_mask > 0
        fallback_region = coarse_valid & (visible_mask == 0)
        preview_valid = (visible_mask > 0) | coarse_valid
        overlay = edge_overlay(aligned, target_gray8, visible_mask)
        transparent = alpha_overlay(
            aligned, target_color, visible_mask, args.overlay_alpha
        )
        transparent_reliable = alpha_overlay(
            aligned,
            target_color,
            reliable_visible.astype(np.uint8) * 255,
            args.overlay_alpha,
        )
        checker = checkerboard_overlay(
            aligned,
            target_color,
            reliable_visible.astype(np.uint8) * 255,
            args.checker_tile_size,
        )
        transparent_with_fallback = alpha_overlay(
            preview,
            target_color,
            preview_valid.astype(np.uint8) * 255,
            args.overlay_alpha,
        )
        score = alignment_score(aligned, target_gray8, visible_mask)
        # PNG is the authoritative lossless image.  Keep JPG for compatibility
        # with earlier runs and quick Windows previews.
        write_image(output_dir / f"{name}_aligned.png", aligned)
        write_image(output_dir / f"{name}_aligned.jpg", aligned)
        write_image(output_dir / f"{name}_valid_mask.png", visible_mask)
        write_image(output_dir / f"{name}_reliable_mask.png", reliable_visible.astype(np.uint8) * 255)
        write_image(output_dir / f"{name}_preview_with_coarse_fallback.jpg", preview)
        write_image(
            output_dir / f"{name}_preview_with_coarse_fallback.png", preview
        )
        write_image(output_dir / f"{name}_edge_overlay.png", overlay)
        write_image(output_dir / f"{name}_alpha_overlay.png", transparent)
        write_image(
            output_dir / f"{name}_alpha_overlay_reliable.png",
            transparent_reliable,
        )
        write_image(output_dir / f"{name}_checker_overlay.png", checker)
        write_image(
            output_dir
            / f"{name}_alpha_overlay_with_coarse_fallback_visual_only.png",
            transparent_with_fallback,
        )
        write_image(
            output_dir / f"{name}_coarse_fallback_region_mask.png",
            fallback_region.astype(np.uint8) * 255,
        )
        highres_outputs: dict[str, Any] | None = None
        if args.preview_scale > 1:
            high_aligned, high_overlay = highres_visual_preview(
                references[name],
                map_xy,
                reliable_visible.astype(np.uint8) * 255,
                target_color,
                args.preview_scale,
                args.overlay_alpha,
            )
            high_aligned_path = output_dir / (
                f"{name}_aligned_preview_{args.preview_scale}x_visual_only.png"
            )
            high_overlay_path = output_dir / (
                f"{name}_alpha_overlay_{args.preview_scale}x_visual_only.png"
            )
            write_image(high_aligned_path, high_aligned)
            write_image(high_overlay_path, high_overlay)
            highres_outputs = {
                "scale": args.preview_scale,
                "aligned_visual_only": str(high_aligned_path),
                "alpha_overlay_visual_only": str(high_overlay_path),
                "warning": (
                    "upsampled target-grid geometry for display only; not a higher-accuracy map"
                ),
            }
        completed_rendering: dict[str, Any] | None = None
        if completion_result is not None:
            (
                completed_aligned,
                completed_visible_mask,
                completed_map_xy,
                completed_z_reference,
            ) = render_reference_from_target_depth(
                references[name],
                reference_models[name],
                target_model,
                target_rays,
                completion_result.depth,
                completion_result.valid,
                args.occlusion_tolerance,
                frame_reference_from_target,
            )
            completed_visible = completed_visible_mask > 0
            completed_added = completed_visible & ~(visible_mask > 0)
            completed_confidence = completion_result.confidence * completed_visible.astype(
                np.float32
            )
            completed_overlay = alpha_overlay(
                completed_aligned,
                target_color,
                completed_visible_mask,
                args.overlay_alpha,
            )
            completed_confidence_overlay = confidence_alpha_overlay(
                completed_aligned,
                target_color,
                completed_visible_mask,
                completed_confidence,
                args.overlay_alpha,
                args.completion_min_alpha,
            )
            completed_edges = edge_overlay(
                completed_aligned, target_gray8, completed_visible_mask
            )
            write_image(
                output_dir / f"{name}_aligned_completed_visual_only.png",
                completed_aligned,
            )
            write_image(
                output_dir / f"{name}_valid_completed_visual_only.png",
                completed_visible_mask,
            )
            write_image(
                output_dir / f"{name}_completion_added_mask.png",
                completed_added.astype(np.uint8) * 255,
            )
            write_image(
                output_dir / f"{name}_alpha_overlay_completed_visual_only.png",
                completed_overlay,
            )
            write_image(
                output_dir
                / f"{name}_alpha_overlay_completed_confidence_visual_only.png",
                completed_confidence_overlay,
            )
            write_image(
                output_dir / f"{name}_edge_overlay_completed_visual_only.png",
                completed_edges,
            )
            map_payload[
                f"map_target_to_{name}_completed_visual_only_xy"
            ] = completed_map_xy.astype(np.float32)
            map_payload[f"{name}_completed_visible_mask"] = completed_visible.astype(
                np.uint8
            )
            completed_score = alignment_score(
                completed_aligned, target_gray8, completed_visible_mask
            )
            completed_rendering = {
                "visible_pixels": int(np.count_nonzero(completed_visible)),
                "visible_ratio": float(
                    np.count_nonzero(completed_visible) / completed_visible.size
                ),
                "new_pixels_beyond_strict_render": int(np.count_nonzero(completed_added)),
                "new_ratio_beyond_strict_render": float(
                    np.count_nonzero(completed_added) / completed_added.size
                ),
                "gradient_cosine_on_completed_visible": float(completed_score),
                "reference_depth_quantiles": quantiles(
                    completed_z_reference[completed_visible]
                ),
                "aligned_visual_only": str(
                    output_dir / f"{name}_aligned_completed_visual_only.png"
                ),
                "confidence_overlay_visual_only": str(
                    output_dir
                    / f"{name}_alpha_overlay_completed_confidence_visual_only.png"
                ),
                "warning": (
                    "single-view/plane/extrapolated pixels are approximate; "
                    "use completion source and confidence masks"
                ),
            }
        map_payload[f"map_target_to_{name}_raw_xy"] = map_xy.astype(np.float32)
        map_payload[f"{name}_visible_mask"] = (visible_mask > 0).astype(np.uint8)
        rendering_reports[name] = {
            "visible_pixels": int(np.count_nonzero(visible_mask)),
            "visible_ratio": float(np.count_nonzero(visible_mask) / visible_mask.size),
            "reliable_visible_pixels": int(np.count_nonzero(reliable_visible)),
            "gradient_cosine_on_visible": float(score),
            "reference_depth_quantiles": quantiles(z_reference[visible_mask > 0]),
            "lossless_aligned_png": str(output_dir / f"{name}_aligned.png"),
            "alpha_overlay": str(output_dir / f"{name}_alpha_overlay.png"),
            "alpha_overlay_reliable": str(
                output_dir / f"{name}_alpha_overlay_reliable.png"
            ),
            "checker_overlay": str(output_dir / f"{name}_checker_overlay.png"),
            "alpha_overlay_with_coarse_fallback_visual_only": str(
                output_dir
                / f"{name}_alpha_overlay_with_coarse_fallback_visual_only.png"
            ),
            "coarse_fallback_region_mask": str(
                output_dir / f"{name}_coarse_fallback_region_mask.png"
            ),
            "coarse_fallback_pixels": int(np.count_nonzero(fallback_region)),
            "coarse_fallback_ratio": float(
                np.count_nonzero(fallback_region) / fallback_region.size
            ),
            "high_resolution_preview": highres_outputs,
            "edge_planar_completion": completed_rendering,
        }
        print(
            f"[{name}] 输出有效区域：{100.0 * np.count_nonzero(visible_mask) / visible_mask.size:.2f}%；"
            f"梯度余弦={score:.4f}"
        )
        if completed_rendering is not None:
            print(
                f"[{name}] 补全后共同视野："
                f"{100.0 * completed_rendering['visible_ratio']:.2f}%；"
                f"新增{100.0 * completed_rendering['new_ratio_beyond_strict_render']:.2f}%"
            )

    np.savez_compressed(output_dir / "depth_alignment_maps.npz", **map_payload)
    report = {
        "program_version": PROGRAM_VERSION,
        "method": "calibrated_multiview_depth_reprojection",
        "important": (
            "RoMa is used only for correspondences. Images are rendered by calibrated 3D reprojection; "
            "black/invalid pixels have no trusted common-visible geometry and must not be treated as aligned."
        ),
        "inputs": {
            "calibration": str(calibration_path),
            "calibration_accepted_for_use": accepted_status,
            "calibration_forced_output": forced_status,
            "reference_images": {name: str(path) for name, path in reference_paths.items()},
            "target_image": str(target_path),
        },
        "pose_resolution": pose_diagnostics,
        "target": {
            "camera": args.target_camera,
            "size": list(target_size),
            "coordinate_system": "raw_distorted_target_pixel_grid",
        },
        "settings": {
            "roma_setting": args.roma_setting,
            "representation": args.representation,
            "pose_refinement": args.pose_refinement,
            "pose_refine_ransac_threshold_px": args.pose_refine_ransac_threshold,
            "pose_refine_max_samples": args.pose_refine_max_samples,
            "pose_refine_homography_threshold_px": args.pose_refine_homography_threshold,
            "pose_refine_max_homography_dominance": args.pose_refine_max_homography_dominance,
            "pose_refine_min_inliers": args.pose_refine_min_inliers,
            "pose_refine_min_inlier_ratio": args.pose_refine_min_inlier_ratio,
            "pose_refine_max_rotation_deg": args.pose_refine_max_rotation_deg,
            "pose_refine_max_translation_deg": args.pose_refine_max_translation_deg,
            "pose_refine_min_improvement": args.pose_refine_min_improvement,
            "pose_refine_strength": args.pose_refine_strength,
            "reference_depth": args.reference_depth,
            "overlap_threshold": args.overlap_threshold,
            "fb_threshold": args.fb_threshold,
            "epipolar_threshold": args.epipolar_threshold,
            "reprojection_threshold": args.reprojection_threshold,
            "minimum_angle_deg": args.minimum_angle_deg,
            "depth_consistency": args.depth_consistency,
            "minimum_views": args.minimum_views,
            "fill_radius": args.fill_radius,
            "fill_edge_threshold": args.fill_edge_threshold,
            "completion_mode": args.completion_mode,
            "completion_iterations": args.completion_iterations,
            "completion_edge_sigma": args.completion_edge_sigma,
            "completion_line_min_length": args.completion_line_min_length,
            "completion_plane_min_points": args.completion_plane_min_points,
            "completion_plane_residual": args.completion_plane_residual,
            "completion_confidence_decay": args.completion_confidence_decay,
            "completion_min_alpha": args.completion_min_alpha,
            "occlusion_tolerance": args.occlusion_tolerance,
            "overlay_alpha": args.overlay_alpha,
            "checker_tile_size": args.checker_tile_size,
            "preview_scale": args.preview_scale,
        },
        "calibration_warnings": calibration_warnings,
        "camera_matching": {item.name: item.report for item in candidates},
        "camera_failures": failures,
        "fusion": fusion_report,
        "edge_planar_completion": (
            completion_result.report if completion_result is not None else None
        ),
        "render_depth_valid_ratio": float(np.count_nonzero(render_valid) / render_valid.size),
        "rendering": rendering_reports,
        "outputs": {
            "maps_npz": str(output_dir / "depth_alignment_maps.npz"),
            "depth_reliable_mask": str(output_dir / "depth_reliable_mask.png"),
            "depth_render_mask": str(output_dir / "depth_render_mask.png"),
            "depth_completion_source": (
                str(output_dir / "depth_completion_source.png")
                if completion_result is not None
                else None
            ),
            "depth_completion_confidence": (
                str(output_dir / "depth_completion_confidence.png")
                if completion_result is not None
                else None
            ),
        },
    }
    write_json(output_dir / "alignment_report.json", report)
    print(f"\n完成：{output_dir}")
    print(f"深度与映射：{output_dir / 'depth_alignment_maps.npz'}")
    print(f"报告：{output_dir / 'alignment_report.json'}")
    print("注意：严格*_aligned.png仍是定量结果；*_completed_visual_only.png必须结合补全来源与置信度掩膜使用。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DepthAlignmentError as exc:
        print(f"错误：{exc}")
        raise SystemExit(2)
