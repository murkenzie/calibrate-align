#!/usr/bin/env python3
"""Fuse monocular depth priors with calibrated multi-camera geometry.

This stage consumes the sparse/multi-view geometry produced by
:mod:`multialign.stages.geometry` and expands it with Depth Anything V2
predictions. The learned prediction is never accepted as metric depth directly:
for every selected reference camera an affine map in inverse-depth space is
robustly fitted to calibrated triangulation anchors.

Calibration convention
----------------------
Internally every camera stores ``camera_from_master``:

    X_camera = R @ X_master + T

The loader accepts generic ``anchor_to_camera`` pose keys and the legacy pose
keys emitted by earlier versions of the project.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import heapq
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


REFERENCE_NAMES = ("reference_a", "reference_b")


class PriorAlignmentError(RuntimeError):
    pass


def configure_rig(reference_cameras: Sequence[str], target_camera: str, anchor_camera: str) -> None:
    global REFERENCE_NAMES
    references = tuple(reference_cameras)
    if len(references) < 2 or len(set(references)) != len(references):
        raise PriorAlignmentError("--reference-cameras requires at least two unique names")
    if target_camera in references:
        raise PriorAlignmentError("--target-camera must not be a reference camera")
    if anchor_camera not in references:
        raise PriorAlignmentError("--anchor-camera must be a reference camera")
    REFERENCE_NAMES = references


@dataclass(frozen=True)
class CameraModel:
    name: str
    K: np.ndarray
    dist: np.ndarray
    image_size: tuple[int, int]
    pose_from_master: np.ndarray


@dataclass
class InverseDepthFit:
    camera: str
    x_center: float
    x_scale: float
    slope: float
    intercept: float
    anchors_total: int
    anchors_inlier: int
    inlier_ratio: float
    train_median_relative_error: float
    validation_median_relative_error: float
    validation_p95_relative_error: float
    depth_p01: float
    depth_p99: float
    raw_p01: float
    raw_p99: float
    global_confidence: float

    def inverse_depth(self, raw: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(raw, dtype=np.float32) - self.x_center) / self.x_scale
        return self.slope * normalized + self.intercept


@dataclass
class DepthCandidate:
    camera: str
    depth_target: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    boundary: np.ndarray
    fit: InverseDepthFit
    report: dict[str, Any]


@dataclass
class ReferenceRender:
    """Both conservative and visually complete reference-to-target renders."""

    aligned_complete: np.ndarray
    aligned_surface_copy: np.ndarray
    aligned_raw_sampleable: np.ndarray
    aligned_zbuffer: np.ndarray
    sampleable: np.ndarray
    zbuffer_visible: np.ndarray
    visual_mask: np.ndarray
    occlusion_filled: np.ndarray
    occlusion_relaxed_filled: np.ndarray
    occlusion_relaxed_eligible: np.ndarray
    display_filled: np.ndarray
    texture_structure_refined: np.ndarray
    texture_inpaint_solver_mask: np.ndarray
    edge_nearest_mask: np.ndarray
    map_xy: np.ndarray


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PriorAlignmentError(f"JSON不存在：{path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PriorAlignmentError(f"无法读取JSON：{path} ({exc})") from exc
    if not isinstance(value, dict):
        raise PriorAlignmentError(f"JSON顶层必须是对象：{path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    if not path.is_file():
        raise PriorAlignmentError(f"图像不存在：{path}")
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(encoded, flags)
    except Exception as exc:
        raise PriorAlignmentError(f"无法读取图像：{path} ({exc})") from exc
    if image is None:
        raise PriorAlignmentError(f"OpenCV无法解码：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower()
    params: list[int] = []
    if suffix in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 96]
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise PriorAlignmentError(f"OpenCV无法编码：{path}")
    encoded.tofile(str(path))


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise PriorAlignmentError(f"不支持的图像形状：{image.shape}")


def robust_uint8(image: np.ndarray, clahe: bool = False) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(values)
    output = np.zeros(values.shape, dtype=np.uint8)
    if not np.any(finite):
        return output
    low, high = np.percentile(values[finite], (1.0, 99.0))
    if high <= low:
        low, high = float(np.min(values[finite])), float(np.max(values[finite]))
    if high > low:
        output = np.round(np.clip((values - low) / (high - low), 0.0, 1.0) * 255.0).astype(np.uint8)
    if clahe:
        output = cv2.createCLAHE(2.0, (8, 8)).apply(output)
    return output


def colorize_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values) & (values > 0)
    gray = np.zeros(values.shape, dtype=np.uint8)
    if np.any(mask):
        log_values = np.log(values[mask])
        lo, hi = np.percentile(log_values, (2.0, 98.0))
        if hi > lo:
            gray[mask] = np.round(np.clip((np.log(values[mask]) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(255 - gray, cv2.COLORMAP_TURBO)
    colored[~mask] = 0
    return colored


def quantiles(values: np.ndarray) -> list[float] | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return [float(v) for v in np.quantile(array, (0.0, 0.25, 0.5, 0.75, 1.0))]


def camera_arrays(item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    key = "K" if "K" in item else "intrinsic"
    if key not in item or "image_size" not in item:
        raise PriorAlignmentError("相机参数缺少K/intrinsic或image_size")
    K = np.asarray(item[key], dtype=np.float64)
    dist = np.asarray(item.get("dist", np.zeros(5)), dtype=np.float64).reshape(-1)
    size = tuple(int(v) for v in item["image_size"])
    if K.shape != (3, 3) or len(size) != 2 or min(size) <= 0:
        raise PriorAlignmentError(f"相机参数形状错误：K={K.shape}, image_size={size}")
    return K, dist, size


def pose_matrix(item: dict[str, Any]) -> np.ndarray:
    R = np.asarray(item["R"], dtype=np.float64)
    T = np.asarray(item["T"], dtype=np.float64).reshape(3)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)) or not np.all(np.isfinite(T)):
        raise PriorAlignmentError("外参R/T无效")
    if np.linalg.norm(R.T @ R - np.eye(3)) > 1e-2 or abs(np.linalg.det(R) - 1.0) > 1e-2:
        raise PriorAlignmentError("外参R不是有效旋转矩阵")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = R
    matrix[:3, 3] = T
    return matrix


def pose_from_master(
    poses: dict[str, Any], camera_name: str, master: str, invert_legacy: bool
) -> np.ndarray:
    if camera_name == master:
        return pose_matrix(poses[master]) if master in poses else np.eye(4, dtype=np.float64)

    direct_key = f"{master}_to_{camera_name}"
    if direct_key in poses:
        return pose_matrix(poses[direct_key])

    legacy_key = f"{camera_name}_to_{master}"
    if legacy_key in poses:
        matrix = pose_matrix(poses[legacy_key])
        convention = str(poses[legacy_key].get("convention", "")).lower().replace(" ", "")
        explicitly_direct = "x_camera=r@x_main+t" in convention or "x_camera=r@x_master+t" in convention
        explicitly_inverse = "x_main=r@x_camera+t" in convention or "x_master=r@x_camera+t" in convention
        if explicitly_inverse or (invert_legacy and not explicitly_direct):
            return np.linalg.inv(matrix)
        return matrix

    if camera_name in poses:
        return pose_matrix(poses[camera_name])
    raise PriorAlignmentError(
        f"camera_poses中找不到{direct_key!r}、{legacy_key!r}或{camera_name!r}"
    )


def load_camera_models(
    calibration: dict[str, Any],
    master: str,
    target_name: str,
    target_size: tuple[int, int],
    invert_legacy: bool,
) -> tuple[dict[str, CameraModel], CameraModel]:
    cameras = calibration.get("cameras")
    poses = calibration.get("camera_poses")
    if not isinstance(cameras, dict) or not isinstance(poses, dict):
        raise PriorAlignmentError("校准JSON必须包含cameras和camera_poses")
    missing = [name for name in (*REFERENCE_NAMES, target_name) if name not in cameras]
    if missing:
        raise PriorAlignmentError(f"校准JSON缺少相机：{missing}")

    declared_references = calibration.get("reference_cameras")
    declared_target = calibration.get("target_camera")
    declared_anchor = calibration.get("anchor_camera")
    if declared_references is not None and tuple(declared_references) != REFERENCE_NAMES:
        raise PriorAlignmentError(
            "校准JSON中的reference_cameras与命令行配置不一致："
            f"{declared_references} != {list(REFERENCE_NAMES)}"
        )
    if declared_target is not None and declared_target != target_name:
        raise PriorAlignmentError(
            f"校准JSON中的target_camera={declared_target!r}，但命令行指定{target_name!r}"
        )
    if declared_anchor is not None and declared_anchor != master:
        raise PriorAlignmentError(
            f"校准JSON中的anchor_camera={declared_anchor!r}，但命令行指定{master!r}"
        )

    references: dict[str, CameraModel] = {}
    for name in REFERENCE_NAMES:
        K, dist, size = camera_arrays(cameras[name])
        references[name] = CameraModel(
            name, K, dist, size, pose_from_master(poses, name, master, invert_legacy)
        )

    K, dist, original_size = camera_arrays(cameras[target_name])
    sx = target_size[0] / original_size[0]
    sy = target_size[1] / original_size[1]
    if not math.isclose(sx, sy, rel_tol=0.01, abs_tol=0.01):
        raise PriorAlignmentError(
            f"目标相机标定尺寸{original_size}与输出尺寸{target_size}不是等比例缩放"
        )
    scale = np.diag([sx, sy, 1.0])
    target = CameraModel(
        target_name,
        scale @ K,
        dist,
        target_size,
        pose_from_master(poses, target_name, master, invert_legacy),
    )
    return references, target


def relative_pose(target: CameraModel, source: CameraModel) -> np.ndarray:
    return target.pose_from_master @ np.linalg.inv(source.pose_from_master)


def resize_camera_model(camera: CameraModel, size: tuple[int, int]) -> CameraModel:
    sx = size[0] / camera.image_size[0]
    sy = size[1] / camera.image_size[1]
    scale = np.diag([sx, sy, 1.0])
    return CameraModel(camera.name, scale @ camera.K, camera.dist, size, camera.pose_from_master)


def make_work_image(image: np.ndarray, maximum_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    if max(width, height) <= maximum_side:
        return image.copy()
    scale = maximum_side / max(width, height)
    size = (max(2, int(round(width * scale))), max(2, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def resize_target(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (image.shape[1], image.shape[0]) == size:
        return image
    if not math.isclose(image.shape[1] / image.shape[0], size[0] / size[1], rel_tol=0.01):
        raise PriorAlignmentError(
            f"目标图像{image.shape[1]}x{image.shape[0]}与输出尺寸{size}长宽比不同"
        )
    interpolation = cv2.INTER_AREA if image.shape[1] > size[0] else cv2.INTER_CUBIC
    return cv2.resize(image, size, interpolation=interpolation)


def prepare_segmentation_guide(image: np.ndarray) -> np.ndarray:
    """Create a contrast-stable 3-channel guide without inventing edges."""
    bgr = np.asarray(image, dtype=np.uint8)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    # A demosaiced/averaged target JPG is often effectively monochrome.
    # Repeating a robust CLAHE intensity is safer than a false-colour map,
    # whose artificial colour boundaries could become segmentation cues.
    chroma = np.mean(np.max(bgr, axis=2).astype(np.float32) - np.min(bgr, axis=2))
    if chroma < 3.0:
        gray = robust_uint8(to_gray(bgr), clahe=True)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(1.6, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def normalized_rays(camera: CameraModel) -> np.ndarray:
    width, height = camera.image_size
    yy, xx = np.indices((height, width), dtype=np.float32)
    pixels = np.stack((xx, yy), axis=-1)
    normalized = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2).astype(np.float64), camera.K, camera.dist
    ).reshape(height, width, 2)
    return np.concatenate(
        (normalized, np.ones((height, width, 1), dtype=np.float64)), axis=2
    ).astype(np.float32)


def project_points(
    points_source: np.ndarray, target_from_source: np.ndarray, target: CameraModel
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_source, dtype=np.float64).reshape(-1, 3)
    R = target_from_source[:3, :3]
    T = target_from_source[:3, 3]
    transformed = (R @ points.T).T + T
    rvec, _ = cv2.Rodrigues(R)
    pixels, _ = cv2.projectPoints(points, rvec, T, target.K, target.dist)
    return pixels.reshape(-1, 2).astype(np.float32), transformed[:, 2].astype(np.float32)


def load_geometry_npz(
    path: Path, target_size: tuple[int, int], mask_mode: str, erode: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise PriorAlignmentError(f"几何NPZ不存在：{path}")
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        depth_key = next(
            (k for k in ("target_depth_z", "depth_target_z", "fused_depth") if k in keys),
            None,
        )
        if depth_key is None:
            raise PriorAlignmentError(f"NPZ缺少目标相机深度；现有键：{sorted(keys)}")
        depth = np.asarray(data[depth_key], dtype=np.float32)

        reliable_key = next(
            (k for k in ("target_depth_reliable_mask", "depth_reliable_mask", "reliable_mask") if k in keys),
            None,
        )
        render_key = next(
            (k for k in ("target_depth_render_mask", "depth_render_mask", "render_mask") if k in keys),
            None,
        )
        selected_key = render_key if mask_mode == "render" and render_key else reliable_key
        mask = np.asarray(data[selected_key] > 0, dtype=bool) if selected_key else np.isfinite(depth) & (depth > 0)
        confidence_key = next(
            (k for k in ("target_depth_confidence", "depth_confidence", "confidence") if k in keys),
            None,
        )
        support_key = next(
            (k for k in ("target_depth_support_count", "depth_support_count", "support") if k in keys),
            None,
        )
        confidence = (
            np.asarray(data[confidence_key], dtype=np.float32)
            if confidence_key
            else np.ones(depth.shape, dtype=np.float32)
        )
        support = (
            np.asarray(data[support_key], dtype=np.float32)
            if support_key
            else np.ones(depth.shape, dtype=np.float32)
        )

    target_shape = (target_size[1], target_size[0])
    if depth.shape != target_shape:
        if not math.isclose(depth.shape[1] / depth.shape[0], target_size[0] / target_size[1], rel_tol=0.01):
            raise PriorAlignmentError(
                f"几何深度尺寸{depth.shape[::-1]}与目标输出尺寸{target_size}不兼容"
            )
        depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST) > 0
        confidence = cv2.resize(confidence, target_size, interpolation=cv2.INTER_LINEAR)
        support = cv2.resize(support, target_size, interpolation=cv2.INTER_NEAREST)

    mask &= np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence > 0)
    if erode > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
        mask = cv2.erode(mask.astype(np.uint8), kernel) > 0
    confidence = np.clip(confidence, 0.0, 1.0)
    confidence[~mask] = 0.0
    report = {
        "path": str(path),
        "depth_key": depth_key,
        "mask_key": selected_key,
        "confidence_key": confidence_key,
        "support_key": support_key,
        "anchor_pixels": int(np.count_nonzero(mask)),
        "depth_quantiles": quantiles(depth[mask]),
    }
    return depth, mask, confidence, support, report


class DepthAnythingRunner:
    CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    def __init__(
        self,
        repository: Path,
        checkpoint: Path,
        encoder: str,
        device_name: str,
        precision: str,
        allow_cpu: bool,
    ) -> None:
        if not repository.is_dir():
            raise PriorAlignmentError(f"Depth Anything V2目录不存在：{repository}")
        if not checkpoint.is_file():
            raise PriorAlignmentError(f"Depth Anything V2权重不存在：{checkpoint}")
        sys.path.insert(0, str(repository.resolve()))
        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2
        except Exception as exc:
            raise PriorAlignmentError(
                "无法导入Depth Anything V2；请确认--depth-anything-root指向官方仓库根目录"
            ) from exc

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise PriorAlignmentError("指定了CUDA，但torch.cuda.is_available()为False")
        if device_name == "cpu" and not allow_cpu:
            raise PriorAlignmentError("未检测到CUDA；CPU会很慢，确认后添加--allow-cpu")
        if precision == "auto":
            precision = "fp16" if device_name == "cuda" else "fp32"
        if precision == "fp16" and device_name != "cuda":
            raise PriorAlignmentError("FP16仅在CUDA模式使用")

        self.torch = torch
        self.device_name = device_name
        self.precision = precision
        self.input_size = 518
        self.model = DepthAnythingV2(**self.CONFIGS[encoder])
        try:
            state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(str(checkpoint), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if not isinstance(state, dict):
            raise PriorAlignmentError("Depth Anything权重不是state_dict")
        if state and all(str(k).startswith("module.") for k in state):
            state = {str(k)[7:]: v for k, v in state.items()}
        try:
            self.model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise PriorAlignmentError(f"权重与encoder={encoder}不匹配：{exc}") from exc
        self.model = self.model.to(device_name).eval()
        print(f"Depth Anything V2设备：{device_name}；encoder={encoder}；precision={precision}")

    def infer(self, bgr: np.ndarray, input_size: int) -> np.ndarray:
        self.input_size = input_size
        autocast = (
            self.torch.autocast(device_type="cuda", dtype=self.torch.float16)
            if self.device_name == "cuda" and self.precision == "fp16"
            else contextlib.nullcontext()
        )
        with self.torch.inference_mode(), autocast:
            depth = self.model.infer_image(bgr, input_size)
        result = np.asarray(depth, dtype=np.float32)
        if result.shape != bgr.shape[:2]:
            result = cv2.resize(result, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        if not np.all(np.isfinite(result)):
            raise PriorAlignmentError("Depth Anything输出包含NaN/Inf")
        if self.device_name == "cuda":
            self.torch.cuda.empty_cache()
        return result


class PromptSegmentationRunner:
    """Thin common wrapper for MobileSAM and the original Segment Anything.

    Only prompt-guided prediction is exposed deliberately.  Running an
    automatic mask generator on the whole target frame tends to turn text,
    wall texture and foliage into unrelated instances; the depth stage below
    supplies a box plus positive/negative points for each ambiguous boundary.
    """

    def __init__(
        self,
        backend: str,
        repository: Path | None,
        checkpoint: Path,
        model_type: str,
        device_name: str,
        allow_cpu: bool,
    ) -> None:
        if backend not in {"mobilesam", "sam"}:
            raise PriorAlignmentError(f"未知物体分割后端：{backend}")
        if repository is not None:
            if not repository.is_dir():
                raise PriorAlignmentError(f"物体分割仓库目录不存在：{repository}")
            sys.path.insert(0, str(repository.resolve()))
        if not checkpoint.is_file():
            raise PriorAlignmentError(f"物体分割权重不存在：{checkpoint}")

        try:
            import torch
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=r"Overwriting tiny_vit_.*in registry.*"
                )
                if backend == "mobilesam":
                    from mobile_sam import SamPredictor, sam_model_registry
                else:
                    from segment_anything import SamPredictor, sam_model_registry
        except Exception as exc:
            package = "mobile_sam" if backend == "mobilesam" else "segment_anything"
            raise PriorAlignmentError(
                f"无法导入{package}；{type(exc).__name__}: {exc}。"
                "请检查--segmentation-root和依赖；MobileSAM还需要timm"
            ) from exc

        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise PriorAlignmentError("物体分割指定了CUDA，但torch.cuda.is_available()为False")
        if device_name == "cpu" and not allow_cpu:
            raise PriorAlignmentError("物体分割未检测到CUDA；确认使用CPU后添加--allow-cpu")
        if model_type not in sam_model_registry:
            available = ", ".join(sorted(str(key) for key in sam_model_registry))
            raise PriorAlignmentError(
                f"{backend}不支持model_type={model_type}；可用：{available}"
            )
        try:
            model = sam_model_registry[model_type](checkpoint=str(checkpoint.resolve()))
            model = model.to(device=device_name).eval()
        except Exception as exc:
            raise PriorAlignmentError(
                f"无法载入{backend}权重；请确认model_type={model_type}与权重匹配：{exc}"
            ) from exc

        self.backend = backend
        self.device_name = device_name
        self.torch = torch
        self.predictor = SamPredictor(model)
        print(f"物体分割：{backend}；model_type={model_type}；设备={device_name}")

    def set_image(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(np.asarray(bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        with self.torch.inference_mode():
            self.predictor.set_image(rgb)

    def predict(
        self,
        points_xy: np.ndarray,
        point_labels: np.ndarray,
        box_xyxy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        with self.torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=np.asarray(points_xy, dtype=np.float32),
                point_labels=np.asarray(point_labels, dtype=np.int32),
                box=np.asarray(box_xyxy, dtype=np.float32),
                multimask_output=True,
            )
        masks = np.asarray(masks, dtype=bool)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if masks.ndim != 3 or masks.shape[0] != scores.size:
            raise PriorAlignmentError("物体分割模型返回的mask/score尺寸异常")
        return masks, scores


def bilinear_sample(
    image: np.ndarray, xy: np.ndarray, maximum_remap_side: int = 30000
) -> np.ndarray:
    """Sample arbitrary points without exceeding OpenCV's SHRT_MAX limit.

    ``cv2.remap`` internally requires every source and destination dimension to
    be smaller than 32767.  A point list shaped as N x 1 therefore fails when a
    dense geometry mask contains more than 32766 anchors.  Keep the temporary
    destination one row high and process the point list in bounded chunks.
    """
    points = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    if points.size == 0:
        return np.empty(0, dtype=np.float32)
    if maximum_remap_side <= 0 or maximum_remap_side >= 32767:
        raise PriorAlignmentError("maximum_remap_side必须在1到32766之间")
    source = np.asarray(image, dtype=np.float32)
    output = np.empty(points.shape[0], dtype=np.float32)
    for start in range(0, points.shape[0], maximum_remap_side):
        stop = min(points.shape[0], start + maximum_remap_side)
        chunk = points[start:stop]
        sampled = cv2.remap(
            source,
            chunk[:, 0].reshape(1, -1),
            chunk[:, 1].reshape(1, -1),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
        output[start:stop] = sampled.reshape(-1)
    return output


def spatially_balance(
    xy: np.ndarray,
    weight: np.ndarray,
    image_size: tuple[int, int],
    maximum: int,
    grid: tuple[int, int] = (20, 15),
) -> np.ndarray:
    width, height = image_size
    cell_x = np.clip((xy[:, 0] / max(width, 1) * grid[0]).astype(np.int32), 0, grid[0] - 1)
    cell_y = np.clip((xy[:, 1] / max(height, 1) * grid[1]).astype(np.int32), 0, grid[1] - 1)
    cell = cell_y * grid[0] + cell_x
    per_cell = max(2, int(math.ceil(maximum / (grid[0] * grid[1]))))
    selected: list[np.ndarray] = []
    for value in np.unique(cell):
        indices = np.flatnonzero(cell == value)
        if indices.size > per_cell:
            local = np.argpartition(weight[indices], -per_cell)[-per_cell:]
            indices = indices[local]
        selected.append(indices)
    output = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    if output.size > maximum:
        best = np.argpartition(weight[output], -maximum)[-maximum:]
        output = output[best]
    return output


def weighted_affine(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    A = np.column_stack((x, np.ones_like(x)))
    root = np.sqrt(np.maximum(weight, 1e-8))
    solution, *_ = np.linalg.lstsq(A * root[:, None], y * root, rcond=None)
    return float(solution[0]), float(solution[1])


def relative_inverse_error(predicted_q: np.ndarray, true_q: np.ndarray) -> np.ndarray:
    return np.abs(predicted_q - true_q) / np.maximum(np.abs(true_q), 1e-12)


def robust_inverse_depth_fit(
    camera: str,
    raw: np.ndarray,
    true_depth: np.ndarray,
    weight: np.ndarray,
    ransac_iterations: int,
    inlier_relative: float,
    seed: int,
) -> tuple[InverseDepthFit, np.ndarray]:
    x = np.asarray(raw, dtype=np.float64)
    z = np.asarray(true_depth, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(z) & np.isfinite(w) & (z > 0) & (w > 0)
    x, z, w = x[valid], z[valid], w[valid]
    if x.size < 40:
        raise PriorAlignmentError(f"[{camera}] 可用几何锚点不足40：{x.size}")
    q = 1.0 / z
    center = float(np.median(x))
    scale = float(np.percentile(x, 75) - np.percentile(x, 25))
    if not math.isfinite(scale) or scale < 1e-8:
        raise PriorAlignmentError(f"[{camera}] 单目深度输出几乎为常数")
    xn = (x - center) / scale

    rng = np.random.default_rng(seed)
    order = rng.permutation(x.size)
    validation_count = max(8, int(round(0.2 * x.size)))
    validation = order[:validation_count]
    train = order[validation_count:]
    if train.size < 20:
        train, validation = order, order[:0]

    best_inlier: np.ndarray | None = None
    best_score = -1.0
    best_error = math.inf
    for _ in range(ransac_iterations):
        pair = rng.choice(train, size=2, replace=False)
        dx = xn[pair[1]] - xn[pair[0]]
        if abs(dx) < 1e-4:
            continue
        slope = (q[pair[1]] - q[pair[0]]) / dx
        intercept = q[pair[0]] - slope * xn[pair[0]]
        predicted = slope * xn[train] + intercept
        error = relative_inverse_error(predicted, q[train])
        inlier = np.isfinite(predicted) & (predicted > 0) & (error <= inlier_relative)
        if np.count_nonzero(inlier) < 12:
            continue
        score = float(np.sum(w[train][inlier]))
        median_error = float(np.median(error[inlier]))
        if score > best_score or (math.isclose(score, best_score) and median_error < best_error):
            best_score, best_error, best_inlier = score, median_error, train[inlier]
    if best_inlier is None:
        raise PriorAlignmentError(f"[{camera}] 逆深度RANSAC无法建立稳定映射")

    active = best_inlier
    slope, intercept = weighted_affine(xn[active], q[active], w[active])
    for _ in range(10):
        residual = slope * xn[active] + intercept - q[active]
        normalized = residual / np.maximum(q[active], 1e-12)
        mad = float(np.median(np.abs(normalized - np.median(normalized))))
        delta = max(1.5 * 1.4826 * mad, 0.01)
        huber = np.ones_like(normalized)
        large = np.abs(normalized) > delta
        huber[large] = delta / np.maximum(np.abs(normalized[large]), 1e-12)
        new_slope, new_intercept = weighted_affine(xn[active], q[active], w[active] * huber)
        if abs(new_slope - slope) + abs(new_intercept - intercept) < 1e-12:
            break
        slope, intercept = new_slope, new_intercept

    predicted_all = slope * xn + intercept
    error_all = relative_inverse_error(predicted_all, q)
    inlier_all = np.isfinite(predicted_all) & (predicted_all > 0) & (error_all <= inlier_relative)
    if np.count_nonzero(inlier_all) >= 20:
        slope, intercept = weighted_affine(xn[inlier_all], q[inlier_all], w[inlier_all])
        predicted_all = slope * xn + intercept
        error_all = relative_inverse_error(predicted_all, q)
        inlier_all = np.isfinite(predicted_all) & (predicted_all > 0) & (error_all <= inlier_relative)

    train_error = error_all[train]
    validation_error = error_all[validation] if validation.size else error_all[inlier_all]
    validation_error = validation_error[np.isfinite(validation_error)]
    inlier_ratio = float(np.count_nonzero(inlier_all) / x.size)
    validation_median = float(np.median(validation_error)) if validation_error.size else math.inf
    validation_p95 = float(np.percentile(validation_error, 95)) if validation_error.size else math.inf
    global_confidence = float(np.clip(inlier_ratio * math.exp(-validation_median / 0.20), 0.02, 1.0))
    fit = InverseDepthFit(
        camera=camera,
        x_center=center,
        x_scale=scale,
        slope=float(slope),
        intercept=float(intercept),
        anchors_total=int(x.size),
        anchors_inlier=int(np.count_nonzero(inlier_all)),
        inlier_ratio=inlier_ratio,
        train_median_relative_error=float(np.median(train_error[np.isfinite(train_error)])),
        validation_median_relative_error=validation_median,
        validation_p95_relative_error=validation_p95,
        depth_p01=float(np.percentile(z[inlier_all], 1)),
        depth_p99=float(np.percentile(z[inlier_all], 99)),
        raw_p01=float(np.percentile(x[inlier_all], 1)),
        raw_p99=float(np.percentile(x[inlier_all], 99)),
        global_confidence=global_confidence,
    )
    return fit, inlier_all


def extract_reference_anchors(
    camera: CameraModel,
    work_camera: CameraModel,
    target: CameraModel,
    target_rays: np.ndarray,
    geometry_depth: np.ndarray,
    geometry_mask: np.ndarray,
    geometry_confidence: np.ndarray,
    geometry_support: np.ndarray,
    raw_prediction: np.ndarray,
    maximum_anchors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat_valid = np.flatnonzero(geometry_mask.reshape(-1))
    if flat_valid.size == 0:
        raise PriorAlignmentError(f"[{camera.name}] 几何锚点为空")
    points_s = target_rays.reshape(-1, 3)[flat_valid] * geometry_depth.reshape(-1)[flat_valid, None]
    reference_from_target = relative_pose(camera, target)
    pixels, depth_reference = project_points(points_s, reference_from_target, work_camera)
    width, height = work_camera.image_size
    inside = (
        np.isfinite(depth_reference)
        & (depth_reference > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] <= width - 1)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] <= height - 1)
    )
    pixels, depth_reference, selected = pixels[inside], depth_reference[inside], flat_valid[inside]
    confidence = geometry_confidence.reshape(-1)[selected]
    support = geometry_support.reshape(-1)[selected]
    confidence = confidence * np.clip(support / 2.0, 0.25, 1.0)
    valid = np.isfinite(depth_reference) & np.isfinite(confidence) & (confidence > 0)
    pixels, depth_reference, confidence = pixels[valid], depth_reference[valid], confidence[valid]

    # Balance first, then sample the learned depth.  Apart from avoiding an
    # unnecessarily large remap this prevents a single textured/planar region
    # from dominating the scale-and-shift fit.
    keep = spatially_balance(pixels, confidence, work_camera.image_size, maximum_anchors)
    pixels, depth_reference, confidence = pixels[keep], depth_reference[keep], confidence[keep]
    raw = bilinear_sample(raw_prediction, pixels)
    valid = np.isfinite(raw)
    return pixels[valid], raw[valid], depth_reference[valid], confidence[valid]


def depth_boundaries(depth: np.ndarray, image: np.ndarray, relative_threshold: float, width: int) -> np.ndarray:
    z = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(z) & (z > 0)
    safe = np.where(valid, z, 1.0)
    logz = np.log(safe)
    threshold = math.log1p(relative_threshold)
    edge = np.zeros(z.shape, dtype=bool)
    dx = np.abs(logz[:, 1:] - logz[:, :-1]) > threshold
    dy = np.abs(logz[1:, :] - logz[:-1, :]) > threshold
    edge[:, 1:] |= dx
    edge[:, :-1] |= dx
    edge[1:, :] |= dy
    edge[:-1, :] |= dy
    gray = robust_uint8(to_gray(image), clahe=True)
    strong_image_edge = cv2.Canny(gray, 100, 220) > 0
    edge |= strong_image_edge & cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    edge |= ~valid
    if width > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
        edge = cv2.dilate(edge.astype(np.uint8), kernel) > 0
    return edge


def build_dense_reference_depth(
    raw: np.ndarray,
    fit: InverseDepthFit,
    anchor_xy: np.ndarray,
    work_image: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverse = fit.inverse_depth(raw)
    valid = np.isfinite(inverse) & (inverse > 1e-12)
    depth = np.full(raw.shape, np.nan, dtype=np.float32)
    depth[valid] = 1.0 / inverse[valid]
    minimum = fit.depth_p01 / args.depth_range_expand
    maximum = fit.depth_p99 * args.depth_range_expand
    valid &= (depth >= minimum) & (depth <= maximum)
    depth[~valid] = np.nan

    seed = np.zeros(raw.shape, dtype=np.uint8)
    rounded = np.round(anchor_xy).astype(np.int32)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, raw.shape[1] - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, raw.shape[0] - 1)
    seed[rounded[:, 1], rounded[:, 0]] = 1
    distance = cv2.distanceTransform(1 - seed, cv2.DIST_L2, 3)
    confidence = args.anchor_confidence_floor + (1.0 - args.anchor_confidence_floor) * np.exp(
        -distance / args.anchor_distance_scale
    )
    confidence *= fit.global_confidence

    raw_span = max(fit.raw_p99 - fit.raw_p01, 1e-6)
    extrapolation = np.maximum(fit.raw_p01 - raw, 0) + np.maximum(raw - fit.raw_p99, 0)
    confidence *= np.exp(-extrapolation / raw_span)
    boundary = depth_boundaries(depth, work_image, args.depth_edge_relative, args.boundary_width)
    confidence[boundary] *= args.boundary_confidence_factor
    confidence[~valid] = 0.0
    return depth, np.clip(confidence, 0.0, 1.0).astype(np.float32), boundary


def rasterize_reference_depth_camera(
    camera: CameraModel,
    work_camera: CameraModel,
    target: CameraModel,
    depth_reference: np.ndarray,
    confidence_reference: np.ndarray,
    boundary_reference: np.ndarray,
    row_chunk: int,
    z_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_width, target_height = target.image_size
    zbuffer = np.full(target_width * target_height, np.inf, dtype=np.float32)
    confbuffer = np.zeros(target_width * target_height, dtype=np.float32)
    boundarybuffer = np.zeros(target_width * target_height, dtype=np.uint8)
    target_from_reference = relative_pose(target, camera)
    R = target_from_reference[:3, :3]
    T = target_from_reference[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    width, height = work_camera.image_size

    for y0 in range(0, height, row_chunk):
        y1 = min(height, y0 + row_chunk)
        yy, xx = np.indices((y1 - y0, width), dtype=np.float32)
        yy += y0
        pixels = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
        rays2 = cv2.undistortPoints(pixels.astype(np.float64), work_camera.K, work_camera.dist).reshape(-1, 2)
        rays = np.column_stack((rays2, np.ones(rays2.shape[0], dtype=np.float64)))
        local_depth = depth_reference[y0:y1].reshape(-1).astype(np.float64)
        local_conf = confidence_reference[y0:y1].reshape(-1)
        local_boundary = boundary_reference[y0:y1].reshape(-1)
        valid = np.isfinite(local_depth) & (local_depth > 0) & (local_conf > 0)
        if not np.any(valid):
            continue
        points = rays[valid] * local_depth[valid, None]
        transformed = (R @ points.T).T + T
        projected, _ = cv2.projectPoints(points, rvec, T, target.K, target.dist)
        projected = projected.reshape(-1, 2)
        z = transformed[:, 2]
        conf = local_conf[valid]
        boundary = local_boundary[valid]
        inside = (
            np.isfinite(z)
            & (z > 0)
            & np.all(np.isfinite(projected), axis=1)
            & (projected[:, 0] >= -0.5)
            & (projected[:, 0] <= target_width - 0.5)
            & (projected[:, 1] >= -0.5)
            & (projected[:, 1] <= target_height - 0.5)
        )
        if not np.any(inside):
            continue
        projected, z, conf, boundary = projected[inside], z[inside], conf[inside], boundary[inside]
        xi = np.clip(np.round(projected[:, 0]).astype(np.int32), 0, target_width - 1)
        yi = np.clip(np.round(projected[:, 1]).astype(np.int32), 0, target_height - 1)
        index = yi.astype(np.int64) * target_width + xi

        order = np.lexsort((z, index))
        sorted_index = index[order]
        first = np.r_[True, sorted_index[1:] != sorted_index[:-1]]
        chosen = order[first]
        index, z, conf, boundary = index[chosen], z[chosen], conf[chosen], boundary[chosen]
        previous = zbuffer[index]
        better = z < previous
        near = np.abs(z - previous) <= np.maximum(np.abs(previous) * z_tolerance, 1e-6)
        if np.any(better):
            selected = index[better]
            zbuffer[selected] = z[better]
            confbuffer[selected] = conf[better]
            boundarybuffer[selected] = boundary[better].astype(np.uint8)
        if np.any(near & ~better):
            selected = index[near & ~better]
            confbuffer[selected] = np.maximum(confbuffer[selected], conf[near & ~better])
            boundarybuffer[selected] |= boundary[near & ~better].astype(np.uint8)

    shape = (target_height, target_width)
    depth = zbuffer.reshape(shape)
    valid = np.isfinite(depth)
    depth[~valid] = np.nan
    confidence = confbuffer.reshape(shape)
    confidence[~valid] = 0
    boundary = boundarybuffer.reshape(shape) > 0
    return depth, confidence, boundary


def fuse_candidates(
    candidates: list[DepthCandidate],
    camera_weights: dict[str, float],
    relative_tolerance: float,
    strict_min_views: int,
    strict_min_confidence: float,
    complete_min_confidence: float,
    reference_camera: str,
    geometry_depth: np.ndarray,
    geometry_mask: np.ndarray,
    geometry_confidence: np.ndarray,
    geometry_support: np.ndarray,
    geometry_agreement_relative: float,
    geometry_selection_margin: float,
    geometry_min_confidence: float,
    geometry_min_support: int,
    geometry_conflict_confidence_factor: float,
    model_conflict_confidence_factor: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Select complete surfaces from the learned prior; use geometry as judge.

    The returned depth is always copied from one of the learned candidates.  A
    triangulated geometry value may choose between disagreeing model surfaces,
    boost/reduce confidence, and make a pixel strict, but it never overwrites
    the model's local depth shape.
    """
    names = [candidate.camera for candidate in candidates]
    if reference_camera not in names:
        raise PriorAlignmentError(
            f"参考深度相机{reference_camera!r}没有产生可用先验；可用={names}"
        )
    reference_index = names.index(reference_camera)
    depths = np.stack([c.depth_target for c in candidates], axis=0).astype(np.float32)
    confidences = np.stack([c.confidence * camera_weights[c.camera] for c in candidates], axis=0)
    valid = np.stack([c.valid for c in candidates], axis=0) & np.isfinite(depths) & (depths > 0)
    confidences = np.where(valid, confidences, 0.0)
    logdepth = np.where(valid, np.log(np.maximum(depths, 1e-12)), 0.0)
    tolerance = math.log1p(relative_tolerance)

    # Reference model determines the surface wherever it is visible.  Other
    # models fill only its missing areas, selecting the highest-confidence
    # complete model surface rather than averaging depths across objects.
    selected_index = np.argmax(confidences, axis=0).astype(np.int16)
    reference_valid = valid[reference_index]
    selected_index[reference_valid] = reference_index
    any_valid = np.any(valid, axis=0)

    # Reliable calibrated geometry is allowed to select *among* model
    # candidates when it clearly distinguishes them.  It is never copied into
    # the final depth map, preserving model-derived boundaries and variations.
    geometry_valid = (
        np.asarray(geometry_mask, dtype=bool)
        & np.isfinite(geometry_depth)
        & (geometry_depth > 0)
        & np.isfinite(geometry_confidence)
        & (geometry_confidence >= geometry_min_confidence)
        & (geometry_support >= geometry_min_support)
    )
    geometry_log = np.log(np.maximum(geometry_depth, 1e-12))
    geometry_error = np.where(
        valid,
        np.abs(logdepth - geometry_log[None]),
        np.inf,
    )
    closest_to_geometry = np.argmin(geometry_error, axis=0).astype(np.int16)
    closest_error = np.take_along_axis(
        geometry_error, closest_to_geometry[None], axis=0
    )[0]
    initially_selected_error = np.take_along_axis(
        geometry_error, selected_index[None], axis=0
    )[0]
    geometry_tolerance = math.log1p(geometry_agreement_relative)
    selection_margin = math.log1p(geometry_selection_margin)
    finite_geometry_errors = np.isfinite(initially_selected_error) & np.isfinite(closest_error)
    geometry_improvement = np.full(initially_selected_error.shape, -np.inf, dtype=np.float32)
    np.subtract(
        initially_selected_error,
        closest_error,
        out=geometry_improvement,
        where=finite_geometry_errors,
    )
    geometry_switch = (
        geometry_valid
        & any_valid
        & np.isfinite(closest_error)
        & (closest_error <= geometry_tolerance)
        & (geometry_improvement >= selection_margin)
    )
    selected_index[geometry_switch] = closest_to_geometry[geometry_switch]

    selected_log = np.take_along_axis(logdepth, selected_index[None], axis=0)[0]
    selected_depth = np.take_along_axis(depths, selected_index[None], axis=0)[0]
    selected_confidence = np.take_along_axis(
        confidences, selected_index[None], axis=0
    )[0]
    consistent = valid & (np.abs(logdepth - selected_log[None]) <= tolerance)
    support = np.sum(consistent, axis=0).astype(np.uint8)

    model_conflict = any_valid & np.any(
        valid & (np.abs(logdepth - selected_log[None]) > tolerance), axis=0
    )
    selected_geometry_error = np.abs(selected_log - geometry_log)
    geometry_agreement = (
        geometry_valid & any_valid & (selected_geometry_error <= geometry_tolerance)
    )
    geometry_conflict = geometry_valid & any_valid & ~geometry_agreement

    confidence = np.clip(selected_confidence, 0.0, 1.0).astype(np.float32)
    # Agreement from another model boosts confidence without modifying depth.
    confidence *= np.clip(1.0 + 0.10 * np.maximum(support.astype(np.float32) - 1.0, 0.0), 1.0, 1.25)
    confidence[model_conflict] *= model_conflict_confidence_factor
    confidence[geometry_agreement] = np.maximum(
        confidence[geometry_agreement],
        np.clip(geometry_confidence[geometry_agreement], 0.0, 1.0),
    )
    confidence[geometry_conflict] *= geometry_conflict_confidence_factor
    confidence = np.clip(confidence, 0.0, 1.0)

    fused = selected_depth.astype(np.float32)
    fused[~any_valid] = np.nan
    strict = (
        any_valid
        & ((support >= strict_min_views) | geometry_agreement)
        & (confidence >= strict_min_confidence)
    )
    # A low confidence value means "do not measure here", not "paint this
    # pixel black".  Keep every finite learned surface in the complete map;
    # strict remains the quality-controlled mask used for measurements.
    complete = any_valid
    low_confidence = any_valid & (confidence < complete_min_confidence)

    boundaries = np.stack([c.boundary for c in candidates], axis=0)
    selected_boundary = np.take_along_axis(
        boundaries, selected_index[None], axis=0
    )[0]
    uncertain_boundary = selected_boundary | model_conflict | geometry_conflict
    selected_camera = np.where(any_valid, selected_index + 1, 0).astype(np.uint8)
    report = {
        "policy": "model_first_geometry_as_scale_and_conflict_judge",
        "reference_camera": reference_camera,
        "camera_index": {str(index + 1): name for index, name in enumerate(names)},
        "candidate_count": len(candidates),
        "strict_valid_ratio": float(np.count_nonzero(strict) / strict.size),
        "complete_valid_ratio": float(np.count_nonzero(complete) / complete.size),
        "low_confidence_but_renderable_ratio": float(
            np.count_nonzero(low_confidence) / low_confidence.size
        ),
        "model_conflict_ratio": float(np.count_nonzero(model_conflict) / model_conflict.size),
        "geometry_agreement_ratio": float(np.count_nonzero(geometry_agreement) / geometry_agreement.size),
        "geometry_conflict_ratio": float(np.count_nonzero(geometry_conflict) / geometry_conflict.size),
        "geometry_selected_alternate_ratio": float(np.count_nonzero(geometry_switch) / geometry_switch.size),
        "selected_camera_pixels": {
            name: int(np.count_nonzero(selected_camera == index + 1))
            for index, name in enumerate(names)
        },
        "support_quantiles": quantiles(support[complete]),
        "confidence_quantiles": quantiles(confidence[complete]),
        "depth_quantiles": quantiles(fused[complete]),
    }
    return (
        fused,
        confidence,
        support,
        strict,
        complete,
        uncertain_boundary,
        geometry_agreement,
        geometry_conflict,
        model_conflict,
        selected_camera,
        report,
    )


def lock_reference_surface(
    fused_depth: np.ndarray,
    fused_confidence: np.ndarray,
    fused_complete: np.ndarray,
    reference: DepthCandidate,
    reference_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Restore the reference learned surface wherever that projection exists.

    Fusion and geometric conflict selection remain useful outside the
    reference projection.  Inside it, however, switching candidates at single
    pixels can turn one rigid boundary into alternating near/far fragments.
    This hard lock makes other cameras true hole-fillers rather than competing
    surface owners.
    """
    depth = np.asarray(fused_depth, dtype=np.float32).copy()
    confidence = np.asarray(fused_confidence, dtype=np.float32).copy()
    complete = np.asarray(fused_complete, dtype=bool).copy()
    reference_depth = np.asarray(reference.depth_target, dtype=np.float32)
    reference_confidence = np.asarray(reference.confidence, dtype=np.float32)
    reference_valid = (
        np.asarray(reference.valid, dtype=bool)
        & np.isfinite(reference_depth)
        & (reference_depth > 0)
    )
    if depth.shape != reference_depth.shape:
        raise PriorAlignmentError(
            f"参考深度尺寸{reference_depth.shape}与融合深度{depth.shape}不一致"
        )
    prior_finite = np.isfinite(depth) & (depth > 0)
    overridden = reference_valid & ~prior_finite
    comparable = reference_valid & prior_finite
    overridden[comparable] |= (
        np.abs(reference_depth[comparable] - depth[comparable])
        > np.maximum(np.abs(depth[comparable]) * 1e-6, 1e-6)
    )
    depth[reference_valid] = reference_depth[reference_valid]
    confidence[reference_valid] = np.clip(
        reference_confidence[reference_valid] * float(reference_weight),
        0.0,
        1.0,
    )
    complete[reference_valid] = True
    return depth, confidence, complete, overridden


def fill_tiny_holes(
    depth: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray,
    barrier: np.ndarray,
    iterations: int,
    relative_spread: float,
    edge_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill only a thin band of rasterisation holes.

    Smooth-neighbour holes receive a confidence-weighted log-depth mean.  At
    a strong discontinuity, averaging the two surfaces would create a third,
    physically false plane.  Instead we extend one real surface by at most
    ``iterations`` pixels; foreground is the safest visual default because it
    prevents a background halo around people and rigid objects.
    """
    output = np.asarray(depth, dtype=np.float32).copy()
    mask = np.asarray(valid, dtype=bool).copy()
    conf = np.asarray(confidence, dtype=np.float32).copy()
    barrier = np.asarray(barrier, dtype=bool)
    filled = np.zeros(mask.shape, dtype=bool)
    edge_filled = np.zeros(mask.shape, dtype=bool)
    log_tolerance = math.log1p(relative_spread)
    shifts = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    for _ in range(iterations):
        # The iteration count is the maximum growth distance, so this cannot
        # flood a large unknown region even though boundary pixels are allowed.
        target = ~mask
        if not np.any(target):
            break
        values = []
        weights = []
        for dy, dx in shifts:
            shifted_depth = np.roll(output, (dy, dx), axis=(0, 1))
            shifted_mask = np.roll(mask, (dy, dx), axis=(0, 1))
            shifted_conf = np.roll(conf, (dy, dx), axis=(0, 1))
            if dy < 0:
                shifted_mask[dy:] = False
            elif dy > 0:
                shifted_mask[:dy] = False
            if dx < 0:
                shifted_mask[:, dx:] = False
            elif dx > 0:
                shifted_mask[:, :dx] = False
            values.append(np.where(shifted_mask, np.log(np.maximum(shifted_depth, 1e-12)), np.nan))
            weights.append(np.where(shifted_mask, shifted_conf, 0.0))
        values_array = np.stack(values, axis=0)
        weights_array = np.stack(weights, axis=0)
        count = np.sum(np.isfinite(values_array), axis=0)
        finite_values = np.isfinite(values_array)
        minimum = np.min(np.where(finite_values, values_array, np.inf), axis=0)
        maximum = np.max(np.where(finite_values, values_array, -np.inf), axis=0)
        accepted = target & (count >= 3) & np.isfinite(minimum)
        if not np.any(accepted):
            break
        safe_values = np.nan_to_num(values_array, nan=0.0)
        weight_sum = np.sum(weights_array, axis=0)
        mean_log = np.sum(safe_values * weights_array, axis=0) / np.maximum(weight_sum, 1e-12)
        is_edge = accepted & (((maximum - minimum) > log_tolerance) | barrier)
        is_smooth = accepted & ~is_edge

        chosen_log = mean_log
        if edge_policy == "foreground":
            chosen_log = minimum
        elif edge_policy == "background":
            chosen_log = maximum
        elif edge_policy == "adaptive":
            midpoint = np.zeros(minimum.shape, dtype=np.float32)
            midpoint[accepted] = 0.5 * (minimum[accepted] + maximum[accepted])
            near_members = finite_values & (values_array <= midpoint[None])
            far_members = finite_values & ~near_members
            near_support = np.sum(np.where(near_members, weights_array, 0.0), axis=0)
            far_support = np.sum(np.where(far_members, weights_array, 0.0), axis=0)
            # Select the locally better-supported surface; unlike a global
            # foreground policy this does not grow every pole/person outward.
            chosen_log = np.where(near_support >= far_support, minimum, maximum)
        elif edge_policy == "highest-confidence":
            selectable_weight = np.where(finite_values, weights_array, -1.0)
            selected = np.argmax(selectable_weight, axis=0)
            chosen_log = np.take_along_axis(values_array, selected[None], axis=0)[0]
        else:  # guarded by argparse, retained for direct function callers
            raise PriorAlignmentError(f"未知边缘裂缝策略：{edge_policy}")

        output[is_smooth] = np.exp(mean_log[is_smooth])
        output[is_edge] = np.exp(chosen_log[is_edge])
        mean_confidence = weight_sum / np.maximum(count, 1)
        conf[is_smooth] = 0.75 * mean_confidence[is_smooth]
        conf[is_edge] = 0.20 * np.max(weights_array, axis=0)[is_edge]
        mask[accepted] = True
        filled[accepted] = True
        edge_filled[is_edge] = True
    output[~mask] = np.nan
    conf[~mask] = 0
    return output, mask, conf, filled, edge_filled


def sharpen_depth_discontinuities(
    depth: np.ndarray,
    valid: np.ndarray,
    boundary_guide: np.ndarray,
    radius: int,
    jump_relative: float,
    minimum_change_relative: float = 0.015,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a narrow learned transition band to two real depth surfaces.

    Monocular networks often represent an occlusion edge as several ordered
    intermediate depths.  Reprojection then stretches those bands into a
    coloured/texture contour.  Around an already detected discontinuity this
    routine estimates the local near/far surfaces and assigns every transition
    pixel to exactly one of them.  Smooth depth gradients away from a boundary
    are untouched.
    """
    output = np.asarray(depth, dtype=np.float32).copy()
    mask = np.asarray(valid, dtype=bool) & np.isfinite(output) & (output > 0)
    snapped = np.zeros(mask.shape, dtype=bool)
    if radius <= 0 or not np.any(mask):
        output[~mask] = np.nan
        return output, snapped

    log_depth = np.zeros(output.shape, dtype=np.float32)
    log_depth[mask] = np.log(output[mask])
    fill_value = float(np.median(log_depth[mask]))
    working = np.where(mask, log_depth, fill_value).astype(np.float32)
    # Remove isolated one-pixel prediction noise before estimating the two
    # surface endpoints; this does not directly smooth the returned depth.
    working = cv2.medianBlur(working, 3)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    local_near = cv2.erode(working, kernel)
    local_far = cv2.dilate(working, kernel)
    full_support = cv2.erode(mask.astype(np.uint8), kernel) > 0
    guide = cv2.dilate(np.asarray(boundary_guide, dtype=np.uint8), kernel) > 0
    spread = local_far - local_near
    candidate = (
        mask
        & full_support
        & guide
        & (spread >= math.log1p(jump_relative))
    )
    if not np.any(candidate):
        output[~mask] = np.nan
        return output, snapped

    midpoint = 0.5 * (local_near + local_far)
    choose_near = log_depth <= midpoint
    # Majority filtering removes alternating one-pixel assignments along a
    # straight rigid edge while retaining one crisp near/far seam.
    choose_near = cv2.medianBlur(choose_near.astype(np.uint8) * 255, 3) > 0
    chosen = np.where(choose_near, local_near, local_far)
    snapped = candidate & (
        np.abs(chosen - log_depth) >= math.log1p(minimum_change_relative)
    )
    output[snapped] = np.exp(chosen[snapped])
    output[~mask] = np.nan
    return output, snapped


def target_guided_depth_propagation(
    depth: np.ndarray,
    valid: np.ndarray,
    target_gray: np.ndarray,
    target_edges: np.ndarray,
    snap_radius: int,
    depth_step_relative: float,
    jump_relative: float,
    gradient_weight: float,
    hard_edge_penalty: float,
    maximum_band_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Move a wide/offset depth transition onto target target edges.

    A broad band around every strong depth step is erased conceptually, then
    refilled from stable depths just outside that band.  Multi-source geodesic
    propagation is cheap in smooth target regions but expensive across a
    target edge, so the winning near/far surface changes at the target-image
    structure rather than at the monocular model's offset parallel contours.
    """
    output = np.asarray(depth, dtype=np.float32).copy()
    mask = np.asarray(valid, dtype=bool) & np.isfinite(output) & (output > 0)
    empty = np.zeros(mask.shape, dtype=bool)
    report: dict[str, Any] = {
        "enabled": snap_radius > 0,
        "requested_radius": int(snap_radius),
        "used_radius": 0,
        "band_ratio": 0.0,
        "assigned_ratio": 0.0,
        "changed_ratio": 0.0,
        "status": "disabled" if snap_radius <= 0 else "no_valid_depth",
    }
    if snap_radius <= 0 or not np.any(mask):
        output[~mask] = np.nan
        return output, empty, empty, report

    log_depth = np.zeros(output.shape, dtype=np.float32)
    log_depth[mask] = np.log(output[mask])
    step_threshold = math.log1p(depth_step_relative)
    depth_edge = np.zeros(mask.shape, dtype=bool)
    horizontal = (
        mask[:, 1:] & mask[:, :-1]
        & (np.abs(log_depth[:, 1:] - log_depth[:, :-1]) >= step_threshold)
    )
    vertical = (
        mask[1:, :] & mask[:-1, :]
        & (np.abs(log_depth[1:, :] - log_depth[:-1, :]) >= step_threshold)
    )
    depth_edge[:, 1:] |= horizontal
    depth_edge[:, :-1] |= horizontal
    depth_edge[1:, :] |= vertical
    depth_edge[:-1, :] |= vertical
    if not np.any(depth_edge):
        report["status"] = "no_depth_steps"
        output[~mask] = np.nan
        return output, empty, empty, report

    used_radius = int(snap_radius)
    band = empty
    while used_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * used_radius + 1, 2 * used_radius + 1)
        )
        candidate_band = cv2.dilate(depth_edge.astype(np.uint8), kernel) > 0
        # Do not touch a broad smooth slope merely because one noisy step was
        # found: the neighbourhood must contain a genuine near/far jump.
        fill_value = float(np.median(log_depth[mask]))
        working = np.where(mask, log_depth, fill_value).astype(np.float32)
        local_near = cv2.erode(working, kernel)
        local_far = cv2.dilate(working, kernel)
        candidate_band &= mask & ((local_far - local_near) >= math.log1p(jump_relative))
        fraction = float(np.count_nonzero(candidate_band) / candidate_band.size)
        if fraction <= maximum_band_fraction:
            band = candidate_band
            break
        used_radius //= 2

    if used_radius <= 0 or not np.any(band):
        report.update({"status": "band_too_large_or_empty", "used_radius": used_radius})
        output[~mask] = np.nan
        return output, empty, empty, report

    # Only the one-pixel outer ring is seeded.  Intermediate learned contours
    # inside the wide band cannot become sources and therefore cannot survive.
    ring = cv2.dilate(band.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    sources = ring & ~band & mask
    if not np.any(sources):
        report.update({"status": "no_stable_ring", "used_radius": used_radius})
        output[~mask] = np.nan
        return output, band, empty, report

    structure_array = np.asarray(target_gray)
    structure = (
        structure_array.copy()
        if structure_array.dtype == np.uint8
        else robust_uint8(structure_array, clahe=True)
    )
    gx = cv2.Sobel(structure, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(structure, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    finite_magnitude = magnitude[np.isfinite(magnitude)]
    gradient_scale = float(np.percentile(finite_magnitude, 95.0)) if finite_magnitude.size else 1.0
    gradient_scale = max(gradient_scale, 1e-6)
    gradient = np.clip(magnitude / gradient_scale, 0.0, 1.0).astype(np.float32)
    hard = cv2.morphologyEx(
        np.asarray(target_edges, dtype=np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    ) > 0
    distance_to_depth_edge = cv2.distanceTransform(
        (~depth_edge).astype(np.uint8), cv2.DIST_L2, 3
    )
    association_scale = max(float(used_radius) * 0.5, 1.0)
    hard_strength = hard.astype(np.float32) * np.exp(
        -distance_to_depth_edge / association_scale
    )

    height, width = mask.shape
    flat_band = band.reshape(-1)
    flat_gradient = gradient.reshape(-1)
    flat_hard_strength = hard_strength.reshape(-1)
    flat_input_depth = output.reshape(-1)
    distance = np.full(height * width, np.inf, dtype=np.float64)
    assigned_depth = np.full(height * width, np.nan, dtype=np.float32)
    source_indices = np.flatnonzero(sources.reshape(-1))
    distance[source_indices] = 0.0
    assigned_depth[source_indices] = flat_input_depth[source_indices]
    heap: list[tuple[float, int]] = [(0.0, int(index)) for index in source_indices]
    heapq.heapify(heap)
    neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))

    while heap:
        current_distance, index = heapq.heappop(heap)
        if current_distance > distance[index]:
            continue
        y, x = divmod(index, width)
        for dy, dx in neighbours:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            neighbour = ny * width + nx
            if not flat_band[neighbour]:
                continue
            edge_strength = max(float(flat_gradient[index]), float(flat_gradient[neighbour]))
            hard_crossing = max(
                float(flat_hard_strength[index]),
                float(flat_hard_strength[neighbour]),
            )
            step_cost = 1.0 + gradient_weight * edge_strength
            step_cost += hard_edge_penalty * hard_crossing
            proposal = current_distance + step_cost
            if proposal < distance[neighbour]:
                distance[neighbour] = proposal
                assigned_depth[neighbour] = assigned_depth[index]
                heapq.heappush(heap, (proposal, neighbour))

    assigned = band & np.isfinite(assigned_depth.reshape(mask.shape))
    propagated = assigned_depth.reshape(mask.shape)
    changed = assigned & (
        np.abs(np.log(np.maximum(propagated, 1e-12)) - log_depth)
        >= math.log1p(0.015)
    )
    output[assigned] = propagated[assigned]
    output[~mask] = np.nan
    report.update({
        "status": "applied",
        "used_radius": int(used_radius),
        "depth_step_ratio": float(np.count_nonzero(depth_edge) / depth_edge.size),
        "band_ratio": float(np.count_nonzero(band) / band.size),
        "source_ratio": float(np.count_nonzero(sources) / sources.size),
        "assigned_ratio": float(np.count_nonzero(assigned) / assigned.size),
        "changed_ratio": float(np.count_nonzero(changed) / changed.size),
        "target_hard_edge_ratio": float(np.count_nonzero(hard) / hard.size),
        "hard_edge_association_scale": association_scale,
        "associated_hard_strength_quantiles": quantiles(hard_strength[hard]),
        "gradient_p95": gradient_scale,
    })
    return output, band, changed, report


def _sample_prompt_points(mask: np.ndarray, maximum: int, minimum_spacing: int) -> np.ndarray:
    """Choose deterministic, well-inside and spatially spread prompt points."""
    binary = np.asarray(mask, dtype=np.uint8)
    if maximum <= 0 or not np.any(binary):
        return np.empty((0, 2), dtype=np.float32)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    maximum_distance = float(np.max(distance))
    if maximum_distance <= 0:
        return np.empty((0, 2), dtype=np.float32)
    # Keep the stable core, not one-pixel fringe values.  On a long pole the
    # distance map has a vertical plateau; farthest-point sampling then places
    # prompts along the full pole instead of clustering them at the top row.
    core = distance >= max(1.0, 0.60 * maximum_distance)
    coordinates_yx = np.argwhere(core)
    if coordinates_yx.size == 0:
        coordinates_yx = np.argwhere(binary > 0)
    centroid = np.mean(np.argwhere(binary > 0), axis=0)
    first = int(np.argmin(np.sum((coordinates_yx - centroid) ** 2, axis=1)))
    selected = [first]
    spacing2 = float(max(int(minimum_spacing), 1) ** 2)
    while len(selected) < maximum:
        chosen = coordinates_yx[np.asarray(selected)]
        delta = coordinates_yx[:, None, :] - chosen[None, :, :]
        minimum_distance2 = np.min(np.sum(delta * delta, axis=2), axis=1)
        minimum_distance2[np.asarray(selected)] = -1.0
        next_index = int(np.argmax(minimum_distance2))
        if float(minimum_distance2[next_index]) < spacing2:
            break
        selected.append(next_index)
    chosen_yx = coordinates_yx[np.asarray(selected)]
    return chosen_yx[:, ::-1].astype(np.float32)


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8))
    return (binary > 0) & (eroded == 0)


def _robust_two_depth_levels(values: np.ndarray) -> tuple[float, float] | None:
    """Deterministic 1-D two-cluster fit that keeps a narrow foreground."""
    data = np.asarray(values, dtype=np.float32)
    data = data[np.isfinite(data)]
    if data.size < 12:
        return None
    # A pole or cable may occupy far below 10% of an expanded prompt box.
    # Initialise from trimmed extremes, then use medians so isolated depth
    # speckles cannot keep an artificial cluster alive.
    near = float(np.percentile(data, 2.0))
    far = float(np.percentile(data, 98.0))
    minimum_cluster = max(4, int(round(0.01 * data.size)))
    for _ in range(12):
        midpoint = 0.5 * (near + far)
        lower = data[data <= midpoint]
        upper = data[data > midpoint]
        if lower.size < minimum_cluster or upper.size < minimum_cluster:
            return None
        new_near = float(np.median(lower))
        new_far = float(np.median(upper))
        if abs(new_near - near) + abs(new_far - far) < 1e-6:
            near, far = new_near, new_far
            break
        near, far = new_near, new_far
    return (near, far) if far > near else None


def _thin_depth_step_edges(
    depth: np.ndarray, valid: np.ndarray, step_relative: float
) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values) & (values > 0)
    log_depth = np.zeros(values.shape, dtype=np.float32)
    log_depth[mask] = np.log(values[mask])
    threshold = math.log1p(step_relative)
    edge = np.zeros(mask.shape, dtype=bool)
    horizontal = (
        mask[:, 1:] & mask[:, :-1]
        & (np.abs(log_depth[:, 1:] - log_depth[:, :-1]) >= threshold)
    )
    vertical = (
        mask[1:, :] & mask[:-1, :]
        & (np.abs(log_depth[1:, :] - log_depth[:-1, :]) >= threshold)
    )
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical
    return edge


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    yy, xx = np.nonzero(mask)
    if xx.size == 0:
        return None
    return int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 0.0


def _clean_prompt_mask(
    candidate_mask: np.ndarray,
    roi: np.ndarray,
    positive_points: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Remove detached islands and one-pixel tendrils from a prompted mask.

    MobileSAM occasionally returns the correct pole/person plus a thin connected
    stripe along a nearby sign or curb.  A 3x3 opening removes only sub-pixel/
    one-pixel bridges at the target resolution. We then retain components that
    contain a positive prompt, so unrelated mask islands cannot alter depth.
    """
    raw = np.asarray(candidate_mask, dtype=bool) & np.asarray(roi, dtype=bool)
    if not np.any(raw):
        return raw, {"raw_pixels": 0, "clean_pixels": 0, "removed_pixels": 0}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    # Do not erase a genuinely tiny instance: fall back when opening removes
    # every positive seed.
    points = np.round(positive_points).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, raw.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, raw.shape[0] - 1)
    if not np.any(opened[points[:, 1], points[:, 0]]):
        opened = raw
    count, labels = cv2.connectedComponents(opened.astype(np.uint8), connectivity=8)
    keep_ids = set(int(v) for v in labels[points[:, 1], points[:, 0]] if int(v) > 0)
    if keep_ids:
        cleaned = np.isin(labels, np.fromiter(keep_ids, dtype=np.int32))
    else:
        cleaned = opened
    raw_pixels = int(np.count_nonzero(raw))
    clean_pixels = int(np.count_nonzero(cleaned))
    return cleaned, {
        "raw_pixels": raw_pixels,
        "clean_pixels": clean_pixels,
        "removed_pixels": raw_pixels - clean_pixels,
    }


def _build_local_segmentation_proposals(
    depth: np.ndarray,
    valid: np.ndarray,
    transition_band: np.ndarray,
    surface_radius: int,
    depth_step_relative: float,
    minimum_component_area: int,
    maximum_components: int,
    minimum_depth_separation: float,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    """Split a globally merged wide band into local object proposals.

    Dilating all depth contours by 20--30 pixels often joins a pole, car,
    pavement and background into one component.  Instead, first use the thin
    depth steps as barriers.  Connected surface regions on the near side become
    object proposals; small thin-edge components provide a fallback for broken
    contours.  Only each proposal's local intersection with the original wide
    band may later be modified.
    """
    source = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(source) & (source > 0)
    band = np.asarray(transition_band, dtype=bool) & mask
    thin_edge = _thin_depth_step_edges(source, mask, depth_step_relative)
    thin_edge = cv2.morphologyEx(
        thin_edge.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ) > 0
    # One-pixel barrier preserves narrow poles while closing tiny contour gaps.
    barrier = cv2.dilate(thin_edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    free_surface = mask & ~barrier
    log_depth = np.zeros(source.shape, dtype=np.float32)
    log_depth[mask] = np.log(source[mask])
    minimum_log_separation = math.log1p(minimum_depth_separation)
    radius = max(int(surface_radius), 1)
    proposal_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    ring_radius = max(4, min(radius // 2, 12))
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * ring_radius + 1, 2 * ring_radius + 1)
    )
    proposals: list[dict[str, Any]] = []

    region_count, region_labels, region_stats, _ = cv2.connectedComponentsWithStats(
        free_surface.astype(np.uint8), connectivity=8
    )
    surface_candidates = 0
    near_surface_candidates = 0
    far_surface_candidates = 0
    for index in range(1, region_count):
        area = int(region_stats[index, cv2.CC_STAT_AREA])
        if area < minimum_component_area:
            continue
        region = region_labels == index
        local_band = (
            cv2.dilate(region.astype(np.uint8), proposal_kernel) > 0
        ) & band
        if np.count_nonzero(local_band) < minimum_component_area:
            continue
        ring = (
            cv2.dilate(region.astype(np.uint8), ring_kernel) > 0
        ) & ~region & mask
        inside_values = log_depth[region]
        outside_values = log_depth[ring]
        if inside_values.size < 4 or outside_values.size < 4:
            continue
        signed_separation = float(np.median(outside_values) - np.median(inside_values))
        # A bounded region may be either a foreground object (inside=near) or a
        # background opening between foreground structures (inside=far).  Keep
        # both; forcing every accepted SAM mask to foreground makes poles grow
        # and turns distant signs/doors into duplicated coloured strips.
        if abs(signed_separation) < 0.35 * minimum_log_separation:
            continue
        inside_assignment = "near" if signed_separation > 0 else "far"
        surface_candidates += 1
        if inside_assignment == "near":
            near_surface_candidates += 1
        else:
            far_surface_candidates += 1
        proposals.append({
            "kind": f"{inside_assignment}_surface_region",
            "mask": local_band,
            "positive_hint": region,
            "negative_hint": ring,
            "inside_assignment": inside_assignment,
            "priority": float(np.count_nonzero(local_band) * abs(signed_separation)),
            "depth_separation": abs(signed_separation),
        })

    # Broken/open contours may not enclose a surface.  Add local thin-edge
    # proposals as fallback, but keep them below enclosed near surfaces.
    edge_count, edge_labels, edge_stats, _ = cv2.connectedComponentsWithStats(
        thin_edge.astype(np.uint8), connectivity=8
    )
    edge_candidates = 0
    edge_minimum = max(6, minimum_component_area // 4)
    for index in range(1, edge_count):
        edge_area = int(edge_stats[index, cv2.CC_STAT_AREA])
        if edge_area < edge_minimum:
            continue
        edge_component = edge_labels == index
        local_band = (
            cv2.dilate(edge_component.astype(np.uint8), proposal_kernel) > 0
        ) & band
        if np.count_nonzero(local_band) < minimum_component_area:
            continue
        # Do not add an almost identical proposal already explained by a near
        # surface region.  Partial overlap is retained because it may represent
        # the opposite side of a thick pole.
        if any(_mask_iou(local_band, item["mask"]) >= 0.85 for item in proposals):
            continue
        edge_candidates += 1
        proposals.append({
            "kind": "thin_edge_fallback",
            "mask": local_band,
            "positive_hint": None,
            "negative_hint": None,
            "inside_assignment": "near",
            "priority": float(0.20 * np.count_nonzero(local_band)),
            "depth_separation": None,
        })

    proposals.sort(key=lambda item: float(item["priority"]), reverse=True)
    proposals = proposals[:maximum_components]
    global_count, _, _, _ = cv2.connectedComponentsWithStats(
        band.astype(np.uint8), connectivity=8
    )
    report = {
        "global_band_components": int(max(global_count - 1, 0)),
        "thin_edge_ratio": float(np.count_nonzero(thin_edge) / thin_edge.size),
        "surface_regions_total": int(max(region_count - 1, 0)),
        "surface_region_proposals": int(surface_candidates),
        "near_surface_region_proposals": int(near_surface_candidates),
        "far_surface_region_proposals": int(far_surface_candidates),
        "thin_edge_components_total": int(max(edge_count - 1, 0)),
        "thin_edge_fallback_proposals": int(edge_candidates),
        "local_proposals_after_limit": len(proposals),
    }
    return proposals, thin_edge, barrier, report


def segmentation_guided_depth_partition(
    fallback_depth: np.ndarray,
    surface_source_depth: np.ndarray,
    valid: np.ndarray,
    transition_band: np.ndarray,
    target_image: np.ndarray,
    target_edges: np.ndarray,
    predictor: PromptSegmentationRunner,
    surface_radius: int,
    depth_step_relative: float,
    prompt_margin: int,
    points_per_side: int,
    minimum_component_area: int,
    maximum_components: int,
    minimum_depth_separation: float,
    minimum_score: float,
    minimum_edge_alignment: float,
    minimum_mask_fraction: float,
    maximum_mask_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Use promptable instance masks as hard near/far depth partitions.

    The input transition band only proposes *where* an object boundary needs
    help.  For every connected proposal, depth quantiles create foreground
    (near) and background (far) prompts.  A SAM mask is accepted only when it
    honours those prompts, separates the two depth populations and follows
    target-image structure.  Accepted masks replace only the ambiguous band;
    stable interior/exterior depth is never flattened or overwritten.
    """
    output = np.asarray(fallback_depth, dtype=np.float32).copy()
    source = np.asarray(surface_source_depth, dtype=np.float32)
    mask = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(source) & (source > 0)
        & np.isfinite(output) & (output > 0)
    )
    band = np.asarray(transition_band, dtype=bool) & mask
    object_union = np.zeros(mask.shape, dtype=bool)
    applied_union = np.zeros(mask.shape, dtype=bool)
    changed_union = np.zeros(mask.shape, dtype=bool)
    prompt_visual = np.asarray(target_image, dtype=np.uint8).copy()
    mask_overlay = prompt_visual.copy()
    report: dict[str, Any] = {
        "enabled": True,
        "status": "no_transition_band",
        "components_total": 0,
        "components_tested": 0,
        "components_accepted": 0,
        "object_mask_ratio": 0.0,
        "applied_ratio": 0.0,
        "changed_ratio": 0.0,
        "components": [],
    }
    if not np.any(band):
        output[~mask] = np.nan
        return output, object_union, applied_union, prompt_visual, mask_overlay, report

    log_depth = np.zeros(source.shape, dtype=np.float32)
    log_depth[mask] = np.log(source[mask])
    fill_value = float(np.median(log_depth[mask]))
    working = np.where(mask, log_depth, fill_value).astype(np.float32)
    radius = max(int(surface_radius), 1)
    surface_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    local_near = cv2.erode(working, surface_kernel)
    local_far = cv2.dilate(working, surface_kernel)
    local_spread = local_far - local_near
    minimum_log_separation = math.log1p(minimum_depth_separation)

    proposals, thin_edge, proposal_barrier, proposal_report = _build_local_segmentation_proposals(
        source,
        mask,
        band,
        radius,
        depth_step_relative,
        minimum_component_area,
        maximum_components,
        minimum_depth_separation,
    )
    report.update(proposal_report)
    report["components_total"] = len(proposals)
    if not proposals:
        report["status"] = "no_local_object_proposals"
        report["thin_edge_ratio"] = float(np.count_nonzero(thin_edge) / thin_edge.size)
        output[~mask] = np.nan
        return output, object_union, applied_union, prompt_visual, mask_overlay, report
    # Magenta is diagnostic only: these are the one-pixel depth discontinuities
    # used to split the old globally merged wide transition band.
    prompt_visual[thin_edge] = (255, 0, 255)
    predictor.set_image(target_image)
    edge_near = cv2.dilate(
        np.asarray(target_edges, dtype=np.uint8), np.ones((5, 5), np.uint8)
    ) > 0
    height, width = mask.shape
    accepted_score = np.full(mask.shape, -np.inf, dtype=np.float32)

    for rank, proposal in enumerate(proposals, start=1):
        component = np.asarray(proposal["mask"], dtype=bool) & band
        component_box = _mask_box(component)
        if component_box is None:
            continue
        x, y, component_x1, component_y1 = component_box
        w = component_x1 - x
        h = component_y1 - y
        area = int(np.count_nonzero(component))
        x0 = max(0, x - prompt_margin)
        y0 = max(0, y - prompt_margin)
        x1 = min(width, x + w + prompt_margin)
        y1 = min(height, y + h + prompt_margin)
        roi = np.zeros(mask.shape, dtype=bool)
        roi[y0:y1, x0:x1] = True
        valid_roi = roi & mask
        component_report: dict[str, Any] = {
            "rank": rank,
            "proposal_kind": str(proposal["kind"]),
            "inside_assignment": str(proposal.get("inside_assignment", "near")),
            "component_area": area,
            "component_box": [x, y, x + w - 1, y + h - 1],
            "prompt_box": [x0, y0, x1 - 1, y1 - 1],
            "accepted": False,
        }
        if np.count_nonzero(valid_roi) < max(4 * points_per_side, 20):
            component_report["reason"] = "insufficient_valid_depth"
            report["components"].append(component_report)
            continue

        values = log_depth[valid_roi]
        depth_levels = _robust_two_depth_levels(values)
        if depth_levels is None:
            component_report["reason"] = "unable_to_fit_two_depth_levels"
            report["components"].append(component_report)
            continue
        near_level, far_level = depth_levels
        level_separation = far_level - near_level
        component_report["proposal_depth_separation"] = float(math.expm1(level_separation))
        if level_separation < minimum_log_separation:
            component_report["reason"] = "insufficient_depth_separation"
            report["components"].append(component_report)
            continue

        # Prefer prompts close to the ambiguous transition, while allowing
        # them to sit in the stable core of a wide pole/person/vehicle.
        prompt_kernel_radius = max(6, min(prompt_margin, radius + 12))
        prompt_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * prompt_kernel_radius + 1, 2 * prompt_kernel_radius + 1),
        )
        prompt_zone = cv2.dilate(component.astype(np.uint8), prompt_kernel) > 0
        prompt_zone &= valid_roi
        near_prompt_mask = prompt_zone & (
            log_depth <= near_level + 0.25 * level_separation
        )
        far_prompt_mask = prompt_zone & (
            log_depth >= far_level - 0.25 * level_separation
        )
        inside_assignment = str(proposal.get("inside_assignment", "near"))
        if inside_assignment not in {"near", "far"}:
            inside_assignment = "near"
        positive_prompt_mask = near_prompt_mask if inside_assignment == "near" else far_prompt_mask
        negative_prompt_mask = far_prompt_mask if inside_assignment == "near" else near_prompt_mask
        positive_hint = proposal.get("positive_hint")
        negative_hint = proposal.get("negative_hint")
        if positive_hint is not None:
            positive_depth_membership = (
                log_depth <= near_level + 0.35 * level_separation
                if inside_assignment == "near"
                else log_depth >= far_level - 0.35 * level_separation
            )
            hinted = np.asarray(positive_hint, dtype=bool) & valid_roi & positive_depth_membership
            if np.count_nonzero(hinted) >= points_per_side:
                positive_prompt_mask = hinted
        if negative_hint is not None:
            negative_depth_membership = (
                log_depth >= far_level - 0.35 * level_separation
                if inside_assignment == "near"
                else log_depth <= near_level + 0.35 * level_separation
            )
            hinted = np.asarray(negative_hint, dtype=bool) & valid_roi & negative_depth_membership
            if np.count_nonzero(hinted) >= points_per_side:
                negative_prompt_mask = hinted
        spacing = max(5, min(w, h, prompt_margin) // 3)
        positive = _sample_prompt_points(positive_prompt_mask, points_per_side, spacing)
        negative = _sample_prompt_points(negative_prompt_mask, points_per_side + 1, spacing)
        if positive.shape[0] == 0 or negative.shape[0] == 0:
            component_report["reason"] = "unable_to_place_positive_and_negative_prompts"
            report["components"].append(component_report)
            continue

        points = np.concatenate((positive, negative), axis=0)
        point_labels = np.concatenate((
            np.ones(positive.shape[0], dtype=np.int32),
            np.zeros(negative.shape[0], dtype=np.int32),
        ))
        component_report["positive_points"] = positive.tolist()
        component_report["negative_points"] = negative.tolist()
        report["components_tested"] += 1
        try:
            candidate_masks, predicted_scores = predictor.predict(
                points, point_labels, np.asarray([x0, y0, x1 - 1, y1 - 1], dtype=np.float32)
            )
        except Exception as exc:
            component_report["reason"] = f"inference_failed: {exc}"
            report["components"].append(component_report)
            continue

        dilated_component = cv2.dilate(
            component.astype(np.uint8), np.ones((7, 7), np.uint8)
        ) > 0
        best_mask: np.ndarray | None = None
        best_metrics: dict[str, Any] | None = None
        best_score = -math.inf
        best_rejected_metrics: dict[str, Any] | None = None
        best_rejected_score = -math.inf
        candidate_reports: list[dict[str, Any]] = []
        positive_index = np.round(positive).astype(np.int32)
        negative_index = np.round(negative).astype(np.int32)

        for candidate_mask, predicted_score in zip(candidate_masks, predicted_scores):
            candidate_mask, cleanup = _clean_prompt_mask(candidate_mask, roi, positive)
            positive_recall = float(np.mean(candidate_mask[positive_index[:, 1], positive_index[:, 0]]))
            negative_exclusion = float(np.mean(~candidate_mask[negative_index[:, 1], negative_index[:, 0]]))
            mask_in_roi = candidate_mask & roi
            mask_fraction = float(np.count_nonzero(mask_in_roi) / max(np.count_nonzero(roi), 1))
            boundary = _binary_boundary(candidate_mask) & roi
            boundary_count = int(np.count_nonzero(boundary))
            edge_alignment = float(
                np.count_nonzero(boundary & edge_near) / max(boundary_count, 1)
            )
            band_alignment = float(
                np.count_nonzero(boundary & dilated_component) / max(boundary_count, 1)
            )
            # The ambiguity band can cover an entire thin pole/person.  Do not
            # require an unbanded foreground core here: the positive/negative
            # prompts were already selected from the near/far depth tails.
            inside_depth = log_depth[candidate_mask & valid_roi]
            outside_depth = log_depth[(~candidate_mask) & valid_roi]
            if inside_depth.size and outside_depth.size:
                signed_depth_separation = float(
                    np.median(outside_depth) - np.median(inside_depth)
                )
            else:
                signed_depth_separation = -math.inf
            oriented_depth_separation = (
                signed_depth_separation
                if inside_assignment == "near"
                else -signed_depth_separation
            )
            depth_score = float(np.clip(
                oriented_depth_separation / max(minimum_log_separation, 1e-8), 0, 1
            ))
            midpoint = 0.5 * (near_level + far_level)
            if inside_assignment == "near":
                inside_consistency = float(np.mean(inside_depth <= midpoint)) if inside_depth.size else 0.0
                outside_consistency = float(np.mean(outside_depth >= midpoint)) if outside_depth.size else 0.0
            else:
                inside_consistency = float(np.mean(inside_depth >= midpoint)) if inside_depth.size else 0.0
                outside_consistency = float(np.mean(outside_depth <= midpoint)) if outside_depth.size else 0.0
            area_ok = minimum_mask_fraction <= mask_fraction <= maximum_mask_fraction
            score = (
                0.30 * float(np.clip(predicted_score, 0, 1))
                + 0.15 * positive_recall
                + 0.15 * negative_exclusion
                + 0.20 * depth_score
                + 0.10 * min(edge_alignment / max(minimum_edge_alignment, 1e-6), 1.0)
                + 0.10 * min(band_alignment / 0.10, 1.0)
            )
            gates = {
                "mask_area": bool(area_ok),
                "positive_prompts": bool(positive_recall >= 0.99),
                "negative_prompts": bool(negative_exclusion >= 0.75),
                "depth_order": bool(oriented_depth_separation >= 0.5 * minimum_log_separation),
                "inside_depth_consistency": bool(inside_consistency >= 0.55),
                "outside_depth_consistency": bool(outside_consistency >= 0.50),
                "target_edge": bool(edge_alignment >= minimum_edge_alignment),
                "transition_contact": bool(band_alignment >= 0.02),
            }
            metrics: dict[str, Any] = {
                "predicted_iou": float(predicted_score),
                "positive_recall": positive_recall,
                "negative_exclusion": negative_exclusion,
                "mask_fraction_in_prompt_box": mask_fraction,
                "target_edge_alignment": edge_alignment,
                "transition_band_alignment": band_alignment,
                "inside_assignment": inside_assignment,
                "signed_outside_minus_inside_log_depth": signed_depth_separation,
                "oriented_log_depth_separation": oriented_depth_separation,
                "inside_depth_consistency": inside_consistency,
                "outside_depth_consistency": outside_consistency,
                "mask_cleanup": cleanup,
                "score": score,
                "gates": gates,
            }
            candidate_reports.append(metrics)
            hard_ok = all(gates.values())
            if score > best_rejected_score:
                best_rejected_score = score
                best_rejected_metrics = metrics
            if hard_ok and score > best_score:
                best_score = score
                best_mask = candidate_mask
                best_metrics = metrics

        component_report["candidates"] = candidate_reports
        if best_mask is None or best_metrics is None:
            component_report["reason"] = "no_mask_passed_quality_gates"
            if best_rejected_metrics is not None:
                component_report["best_rejected"] = best_rejected_metrics
                component_report["failed_gates"] = [
                    name
                    for name, passed in best_rejected_metrics.get("gates", {}).items()
                    if not passed
                ]
            report["components"].append(component_report)
            box_color = (0, 0, 255)
        elif best_score < minimum_score:
            component_report["reason"] = "best_score_below_threshold"
            component_report["best_score"] = float(best_score)
            report["components"].append(component_report)
            box_color = (0, 165, 255)
        else:
            eligible = (
                component
                & (local_spread >= minimum_log_separation)
                & np.isfinite(local_near) & np.isfinite(local_far)
            )
            # Local proposals may overlap.  A higher-confidence object proposal
            # owns the overlap; a later weak proposal must not undo its crisp
            # near/far decision.
            update = eligible & (best_score > accepted_score)
            proposed = (
                np.where(best_mask, local_near, local_far)
                if inside_assignment == "near"
                else np.where(best_mask, local_far, local_near)
            )
            changed = update & (
                np.abs(proposed - log_depth) >= math.log1p(0.015)
            )
            output[update] = np.exp(proposed[update])
            accepted_score[update] = float(best_score)
            accepted_object = best_mask & roi
            object_union |= accepted_object
            applied_union |= update
            changed_union |= changed
            component_report.update({
                "accepted": True,
                "best": best_metrics,
                "applied_pixels": int(np.count_nonzero(update)),
                "changed_pixels": int(np.count_nonzero(changed)),
            })
            report["components"].append(component_report)
            report["components_accepted"] += 1
            box_color = (0, 255, 0)

            tint = np.zeros_like(mask_overlay)
            tint[accepted_object] = (
                (0, 210, 60) if inside_assignment == "near" else (210, 110, 0)
            )
            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, tint, 0.35, 0)
            boundary = _binary_boundary(accepted_object)
            mask_overlay[boundary] = (
                (0, 255, 255) if inside_assignment == "near" else (255, 180, 0)
            )

        cv2.rectangle(prompt_visual, (x0, y0), (x1 - 1, y1 - 1), box_color, 2)
        for px, py in positive.astype(np.int32):
            cv2.circle(prompt_visual, (int(px), int(py)), 4, (0, 255, 0), -1)
            cv2.circle(prompt_visual, (int(px), int(py)), 5, (0, 0, 0), 1)
        for px, py in negative.astype(np.int32):
            cv2.circle(prompt_visual, (int(px), int(py)), 4, (0, 0, 255), -1)
            cv2.circle(prompt_visual, (int(px), int(py)), 5, (255, 255, 255), 1)

    output[~mask] = np.nan
    rejection_reasons: dict[str, int] = {}
    failed_gates: dict[str, int] = {}
    for component_report in report["components"]:
        if component_report.get("accepted"):
            continue
        reason = str(component_report.get("reason", "unknown"))
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        for gate in component_report.get("failed_gates", []):
            gate = str(gate)
            failed_gates[gate] = failed_gates.get(gate, 0) + 1
    accepted_near = sum(
        1 for item in report["components"]
        if item.get("accepted") and item.get("inside_assignment") == "near"
    )
    accepted_far = sum(
        1 for item in report["components"]
        if item.get("accepted") and item.get("inside_assignment") == "far"
    )
    report.update({
        "status": "applied" if report["components_accepted"] else "no_mask_accepted",
        "object_mask_ratio": float(np.count_nonzero(object_union) / object_union.size),
        "applied_ratio": float(np.count_nonzero(applied_union) / applied_union.size),
        "changed_ratio": float(np.count_nonzero(changed_union) / changed_union.size),
        "rejection_reasons": rejection_reasons,
        "failed_quality_gates": failed_gates,
        "accepted_near_inside": int(accepted_near),
        "accepted_far_inside": int(accepted_far),
    })
    return output, object_union, changed_union, prompt_visual, mask_overlay, report


def projection_sampleable(
    map_xy: np.ndarray,
    depth_in_reference: np.ndarray,
    valid: np.ndarray,
    reference_size: tuple[int, int],
) -> np.ndarray:
    width, height = reference_size
    x, y = map_xy[..., 0], map_xy[..., 1]
    z = np.asarray(depth_in_reference, dtype=np.float32)
    return (
        np.asarray(valid, dtype=bool)
        & np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 0)
        & (x >= -0.5) & (x <= width - 0.5)
        & (y >= -0.5) & (y <= height - 0.5)
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
    x, y = map_xy[..., 0], map_xy[..., 1]
    z = np.asarray(depth_in_reference, dtype=np.float32)
    base = projection_sampleable(map_xy, z, valid, reference_size)
    safe_x, safe_y = np.where(np.isfinite(x), x, 0), np.where(np.isfinite(y), y, 0)
    z_width, z_height = min(width, target_width), min(height, target_height)
    zx = (safe_x + 0.5) * z_width / width - 0.5
    zy = (safe_y + 0.5) * z_height / height - 0.5
    buffer = np.full(z_width * z_height, np.inf, dtype=np.float32)
    for xv in (np.floor(zx), np.ceil(zx)):
        for yv in (np.floor(zy), np.ceil(zy)):
            xi = np.clip(xv.astype(np.int64), 0, z_width - 1)
            yi = np.clip(yv.astype(np.int64), 0, z_height - 1)
            index = yi[base] * z_width + xi[base]
            np.minimum.at(buffer, index, z[base])
    buffer = buffer.reshape(z_height, z_width)
    nearest = buffer[
        np.clip(np.round(zy).astype(np.int64), 0, z_height - 1),
        np.clip(np.round(zx).astype(np.int64), 0, z_width - 1),
    ]
    return base & (z <= nearest + np.maximum(np.abs(z) * relative_tolerance, 1e-6))


def fill_occlusion_by_target_depth(
    remapped: np.ndarray,
    sampleable: np.ndarray,
    visible: np.ndarray,
    target_depth: np.ndarray,
    maximum_radius: int,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill narrow disocclusions from the matching target-depth surface.

    Backward remapping alone marks a pixel sampleable even when that source
    coordinate is occupied by a nearer object.  Keeping such samples makes a
    pole wider and duplicates a coloured sign as a stripe.  We first reject
    them with the z-buffer, then propagate colour only from neighbours whose
    *target* depth belongs to the same surface.  Thus a background hole receives
    background colour, while a foreground raster crack receives foreground
    colour; no global near/far policy is needed.
    """
    output = np.asarray(remapped, dtype=np.uint8).copy()
    base_visible = np.asarray(visible, dtype=bool)
    target = np.asarray(sampleable, dtype=bool) & ~base_visible
    output[~base_visible] = 0
    filled = np.zeros(base_visible.shape, dtype=bool)
    if maximum_radius <= 0 or not np.any(target):
        return output, filled

    depth = np.asarray(target_depth, dtype=np.float32)
    depth_valid = np.isfinite(depth) & (depth > 0)
    log_depth = np.zeros(depth.shape, dtype=np.float32)
    log_depth[depth_valid] = np.log(depth[depth_valid])
    tolerance = math.log1p(relative_tolerance)
    available = base_visible & depth_valid
    remaining = target & depth_valid
    height, width = depth.shape

    def shifted(array: np.ndarray, dy: int, dx: int, fill_value: Any) -> np.ndarray:
        result = np.full(array.shape, fill_value, dtype=array.dtype)
        if dy >= 0:
            source_y, target_y = slice(dy, height), slice(0, height - dy)
        else:
            source_y, target_y = slice(0, height + dy), slice(-dy, height)
        if dx >= 0:
            source_x, target_x = slice(dx, width), slice(0, width - dx)
        else:
            source_x, target_x = slice(0, width + dx), slice(-dx, width)
        result[target_y, target_x] = array[source_y, source_x]
        return result

    neighbours = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )
    for _ in range(int(maximum_radius)):
        if not np.any(remaining):
            break
        best_cost = np.full(depth.shape, np.inf, dtype=np.float32)
        best_color = np.zeros_like(output)
        for dy, dx in neighbours:
            neighbour_valid = shifted(available, dy, dx, False)
            candidate = remaining & neighbour_valid
            if not np.any(candidate):
                continue
            neighbour_log_depth = shifted(log_depth, dy, dx, np.nan)
            cost = np.abs(neighbour_log_depth - log_depth)
            # Prefer axial propagation when depth agreement is otherwise equal;
            # it produces fewer staircase artefacts along straight rigid edges.
            if dy != 0 and dx != 0:
                cost = cost + 1e-4
            better = candidate & (cost <= tolerance) & (cost < best_cost)
            if not np.any(better):
                continue
            neighbour_color = shifted(output, dy, dx, 0)
            best_cost[better] = cost[better]
            best_color[better] = neighbour_color[better]
        newly = remaining & np.isfinite(best_cost)
        if not np.any(newly):
            break
        output[newly] = best_color[newly]
        available[newly] = True
        filled[newly] = True
        remaining[newly] = False
    return output, filled


def fill_small_unresolved_surface_components(
    image: np.ndarray,
    available: np.ndarray,
    unresolved: np.ndarray,
    surface_guide_depth: np.ndarray,
    maximum_radius: int,
    relaxed_relative_tolerance: float,
    maximum_component_area: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill only small unresolved components with a relaxed surface test.

    Increasing propagation radius cannot help a component whose first neighbour
    fails the strict same-depth tolerance.  This second pass selects only small,
    non-border unresolved components, then permits gradual propagation within a
    wider depth band.  Large disocclusions and field-of-view losses remain
    explicit; a pole/background jump still exceeds the relaxed threshold.
    """
    output = np.asarray(image, dtype=np.uint8).copy()
    base_available = np.asarray(available, dtype=bool).copy()
    candidate_mask = np.asarray(unresolved, dtype=bool).copy()
    filled = np.zeros(candidate_mask.shape, dtype=bool)
    eligible = np.zeros(candidate_mask.shape, dtype=bool)
    if (
        maximum_radius <= 0
        or maximum_component_area <= 0
        or not np.any(candidate_mask)
    ):
        return output, filled, eligible

    depth = np.asarray(surface_guide_depth, dtype=np.float32)
    depth_valid = np.isfinite(depth) & (depth > 0)
    candidate_mask &= depth_valid
    if not np.any(candidate_mask):
        return output, filled, eligible

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8),
        connectivity=8,
    )
    height, width = candidate_mask.shape
    for component in range(1, component_count):
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        component_width = int(stats[component, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
        area = int(stats[component, cv2.CC_STAT_AREA])
        touches_border = (
            x <= 0 or y <= 0
            or x + component_width >= width
            or y + component_height >= height
        )
        if area <= maximum_component_area and not touches_border:
            eligible[labels == component] = True
    if not np.any(eligible):
        return output, filled, eligible

    log_depth = np.zeros(depth.shape, dtype=np.float32)
    log_depth[depth_valid] = np.log(depth[depth_valid])
    tolerance = math.log1p(float(relaxed_relative_tolerance))
    available_now = base_available & depth_valid
    remaining = eligible.copy()

    def shifted(array: np.ndarray, dy: int, dx: int, fill_value: Any) -> np.ndarray:
        result = np.full(array.shape, fill_value, dtype=array.dtype)
        if dy >= 0:
            source_y, target_y = slice(dy, height), slice(0, height - dy)
        else:
            source_y, target_y = slice(0, height + dy), slice(-dy, height)
        if dx >= 0:
            source_x, target_x = slice(dx, width), slice(0, width - dx)
        else:
            source_x, target_x = slice(0, width + dx), slice(-dx, width)
        result[target_y, target_x] = array[source_y, source_x]
        return result

    neighbours = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )
    for _ in range(int(maximum_radius)):
        if not np.any(remaining):
            break
        best_cost = np.full(depth.shape, np.inf, dtype=np.float32)
        best_color = np.zeros_like(output)
        for dy, dx in neighbours:
            neighbour_available = shifted(available_now, dy, dx, False)
            candidate = remaining & neighbour_available
            if not np.any(candidate):
                continue
            neighbour_log_depth = shifted(log_depth, dy, dx, np.nan)
            cost = np.abs(neighbour_log_depth - log_depth)
            if dy != 0 and dx != 0:
                cost = cost + 1e-4
            better = candidate & (cost <= tolerance) & (cost < best_cost)
            if not np.any(better):
                continue
            neighbour_color = shifted(output, dy, dx, 0)
            best_cost[better] = cost[better]
            best_color[better] = neighbour_color[better]
        newly = remaining & np.isfinite(best_cost)
        if not np.any(newly):
            break
        output[newly] = best_color[newly]
        available_now[newly] = True
        filled[newly] = True
        remaining[newly] = False
    return output, filled, eligible


def fill_display_cracks(
    image: np.ndarray,
    sampleable: np.ndarray,
    depth: np.ndarray,
    maximum_radius: int,
    relative_spread: float,
    edge_policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill a narrow display gap by copying one real neighbouring surface.

    Unlike texture inpainting, this never synthesizes a blended line between
    foreground and background.  At a depth jump it copies the selected side;
    in a smooth area it copies the spatially nearest valid pixel.
    """
    output = image.copy()
    filled = np.zeros(sampleable.shape, dtype=bool)
    if maximum_radius <= 0 or np.all(sampleable):
        return output, filled

    valid = np.asarray(sampleable, dtype=bool)
    left = np.zeros_like(valid)
    right = np.zeros_like(valid)
    up = np.zeros_like(valid)
    down = np.zeros_like(valid)
    up_left = np.zeros_like(valid)
    down_right = np.zeros_like(valid)
    up_right = np.zeros_like(valid)
    down_left = np.zeros_like(valid)
    # radius=2 fills roughly 1--4 pixel cracks.  Requiring valid samples on
    # opposite sides prevents a wide FOV-loss region from being fabricated.
    reach = maximum_radius + 1
    for distance in range(1, reach + 1):
        left[:, distance:] |= valid[:, :-distance]
        right[:, :-distance] |= valid[:, distance:]
        up[distance:, :] |= valid[:-distance, :]
        down[:-distance, :] |= valid[distance:, :]
        up_left[distance:, distance:] |= valid[:-distance, :-distance]
        down_right[:-distance, :-distance] |= valid[distance:, distance:]
        up_right[distance:, :-distance] |= valid[:-distance, distance:]
        down_left[:-distance, distance:] |= valid[distance:, :-distance]
    bracketed = (
        (left & right)
        | (up & down)
        | (up_left & down_right)
        | (up_right & down_left)
    )
    filled = ~valid & bracketed

    if np.any(filled):
        depth = np.asarray(depth, dtype=np.float32)
        nearest_distance = np.full(valid.shape, np.inf, dtype=np.float32)
        nearest_color = np.zeros_like(output)
        near_depth = np.full(valid.shape, np.inf, dtype=np.float32)
        far_depth = np.full(valid.shape, -np.inf, dtype=np.float32)
        near_color = np.zeros_like(output)
        far_color = np.zeros_like(output)
        height, width = valid.shape

        def shifted(array: np.ndarray, dy: int, dx: int, fill_value: Any) -> np.ndarray:
            result = np.full(array.shape, fill_value, dtype=array.dtype)
            if dy >= 0:
                source_y, target_y = slice(dy, height), slice(0, height - dy)
            else:
                source_y, target_y = slice(0, height + dy), slice(-dy, height)
            if dx >= 0:
                source_x, target_x = slice(dx, width), slice(0, width - dx)
            else:
                source_x, target_x = slice(0, width + dx), slice(-dx, width)
            result[target_y, target_x] = array[source_y, source_x]
            return result

        offsets = sorted(
            (
                (dy * dy + dx * dx, dy, dx)
                for dy in range(-reach, reach + 1)
                for dx in range(-reach, reach + 1)
                if dy != 0 or dx != 0
            ),
            key=lambda item: item[0],
        )
        for distance_squared, dy, dx in offsets:
            neighbour_valid = shifted(valid, dy, dx, False)
            candidate = filled & neighbour_valid
            if not np.any(candidate):
                continue
            neighbour_depth = shifted(depth, dy, dx, np.nan)
            neighbour_color = shifted(output, dy, dx, 0)

            closer = candidate & (distance_squared < nearest_distance)
            nearest_distance[closer] = distance_squared
            nearest_color[closer] = neighbour_color[closer]

            nearer = candidate & np.isfinite(neighbour_depth) & (neighbour_depth < near_depth)
            near_depth[nearer] = neighbour_depth[nearer]
            near_color[nearer] = neighbour_color[nearer]
            farther = candidate & np.isfinite(neighbour_depth) & (neighbour_depth > far_depth)
            far_depth[farther] = neighbour_depth[farther]
            far_color[farther] = neighbour_color[farther]

        finite_surfaces = (
            np.isfinite(near_depth) & np.isfinite(far_depth)
            & (near_depth > 0) & (far_depth > 0)
        )
        log_surface_span = np.zeros(valid.shape, dtype=np.float32)
        log_surface_span[finite_surfaces] = (
            np.log(far_depth[finite_surfaces]) - np.log(near_depth[finite_surfaces])
        )
        discontinuity = (
            filled
            & finite_surfaces
            & (log_surface_span > math.log1p(relative_spread))
        )
        chosen_color = nearest_color
        if edge_policy == "foreground":
            chosen_color = np.where(discontinuity[..., None], near_color, nearest_color)
        elif edge_policy == "background":
            chosen_color = np.where(discontinuity[..., None], far_color, nearest_color)
        elif edge_policy == "adaptive":
            target_valid = np.isfinite(depth) & (depth > 0) & finite_surfaces
            target_log = np.zeros(depth.shape, dtype=np.float32)
            target_log[target_valid] = np.log(depth[target_valid])
            near_difference = np.full(depth.shape, np.inf, dtype=np.float32)
            far_difference = np.full(depth.shape, np.inf, dtype=np.float32)
            near_difference[target_valid] = np.abs(
                np.log(near_depth[target_valid]) - target_log[target_valid]
            )
            far_difference[target_valid] = np.abs(
                np.log(far_depth[target_valid]) - target_log[target_valid]
            )
            choose_near = near_difference <= far_difference
            surface_color = np.where(choose_near[..., None], near_color, far_color)
            chosen_color = np.where(discontinuity[..., None], surface_color, nearest_color)
        elif edge_policy != "highest-confidence":
            raise PriorAlignmentError(f"未知边缘裂缝策略：{edge_policy}")
        output[filled] = chosen_color[filled]
    return output, filled


def refine_filled_texture_structure(
    image: np.ndarray,
    authorized_fill: np.ndarray,
    unavailable: np.ndarray,
    method: str,
    inpaint_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continue image structure only inside geometry-authorized fill pixels.

    The preceding render stages decide *which surface owns a missing pixel* and
    produce a safe nearest/same-surface colour copy.  A copy is robust but it
    ends slanted edges, text strokes and other isophotes abruptly.  OpenCV's
    Navier--Stokes inpainting transports those directions through a thin hole.

    Pixels outside ``authorized_fill`` are copied back verbatim.  Nearby
    unavailable pixels are included in the solver mask only so their black
    placeholder cannot leak into the solution; they are never committed to the
    returned image.  Thus this stage changes neither depth, visibility nor the
    set of pixels that the geometric quality gates allowed us to fill.
    """
    output = np.asarray(image).copy()
    fill_mask = np.asarray(authorized_fill, dtype=bool)
    unavailable_mask = np.asarray(unavailable, dtype=bool)
    refined = np.zeros(fill_mask.shape, dtype=bool)
    solver_mask = np.zeros(fill_mask.shape, dtype=bool)

    if method == "copy" or inpaint_radius <= 0 or not np.any(fill_mask):
        return output, refined, solver_mask
    if method not in ("navier-stokes", "telea"):
        raise PriorAlignmentError(f"未知填补纹理恢复方式：{method}")
    if output.dtype != np.uint8 or output.ndim not in (2, 3):
        raise PriorAlignmentError(
            "结构延续修复要求uint8灰度或彩色图像；"
            f"当前dtype={output.dtype}, shape={output.shape}"
        )

    # Give the PDE a small unknown halo around the authorized fill.  This keeps
    # neighbouring black placeholders from being treated as real observations,
    # while the final commit remains strictly limited to ``fill_mask``.
    halo_radius = max(1, int(math.ceil(2.0 * float(inpaint_radius))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * halo_radius + 1, 2 * halo_radius + 1),
    )
    near_fill = cv2.dilate(fill_mask.astype(np.uint8), kernel) > 0
    solver_mask = fill_mask | (unavailable_mask & near_fill)
    flag = cv2.INPAINT_NS if method == "navier-stokes" else cv2.INPAINT_TELEA
    restored = cv2.inpaint(
        np.ascontiguousarray(output),
        solver_mask.astype(np.uint8) * 255,
        float(inpaint_radius),
        flag,
    )
    output[fill_mask] = restored[fill_mask]
    refined[fill_mask] = True
    return output, refined, solver_mask


def remap_reference_texture(
    reference: np.ndarray,
    map_xy: np.ndarray,
    surface_guide_depth: np.ndarray,
    target_valid: np.ndarray,
    interpolation: str,
    edge_depth_step: float,
    edge_nearest_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Remap texture without blending colours across a depth discontinuity.

    Bilinear interpolation is desirable inside one surface, but at a pole,
    person, sign, or car boundary its four source taps can belong to two
    different objects.  The resulting average is a soft coloured halo.  The
    edge-aware mode retains bilinear sampling in smooth regions and switches
    only a narrow target-depth boundary band to nearest-neighbour sampling.
    """
    map_x = np.asarray(map_xy[..., 0], dtype=np.float32)
    map_y = np.asarray(map_xy[..., 1], dtype=np.float32)
    if interpolation == "nearest":
        remapped = cv2.remap(
            reference,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return remapped, np.asarray(target_valid, dtype=bool).copy()

    linear = cv2.remap(
        reference,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if interpolation == "linear":
        return linear, np.zeros(target_valid.shape, dtype=bool)
    if interpolation != "edge-aware":
        raise PriorAlignmentError(f"未知纹理插值方式：{interpolation}")

    edge_mask = _thin_depth_step_edges(
        surface_guide_depth,
        target_valid,
        edge_depth_step,
    )
    if edge_nearest_radius > 0 and np.any(edge_mask):
        radius = int(edge_nearest_radius)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel) > 0
    edge_mask &= np.asarray(target_valid, dtype=bool)
    if not np.any(edge_mask):
        return linear, edge_mask

    nearest = cv2.remap(
        reference,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    output = linear
    output[edge_mask] = nearest[edge_mask]
    return output, edge_mask


def _median_blur_float_radius(values: np.ndarray, radius: int) -> np.ndarray:
    """Float32-safe median filtering for arbitrary integer radii.

    OpenCV accepts float32 input only for 3x3 and 5x5 median kernels; 7x7 and
    larger require uint8 in several OpenCV 4/5 builds.  Chaining 5x5 (radius 2)
    and 3x3 (radius 1) passes gives the requested effective neighbourhood while
    retaining metric log-depth precision and avoiding version-specific asserts.
    """
    output = np.asarray(values, dtype=np.float32).copy()
    remaining = int(radius)
    if remaining < 0:
        raise PriorAlignmentError("中值滤波半径不能为负数")
    while remaining >= 2:
        output = cv2.medianBlur(output, 5)
        remaining -= 2
    if remaining == 1:
        output = cv2.medianBlur(output, 3)
    return output


def build_render_surface_guide(
    depth: np.ndarray,
    valid: np.ndarray,
    fill_radius: int,
    median_radius: int,
    relative_spread: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a continuous render-only surface guide without changing metric depth.

    Forward projection of a monocular depth map can leave small target-space
    holes and one-pixel depth fragments.  Those fragments break same-surface
    colour propagation even when the underlying pole/person is coherent.  We
    fill only a bounded hole radius, then median-filter log depth for surface
    *classification*.  Original valid depths remain unchanged for projection;
    the cleaned guide is used only to decide which pixels belong to one render
    surface and where texture interpolation must stay sharp.
    """
    original_depth = np.asarray(depth, dtype=np.float32)
    original_valid = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(original_depth)
        & (original_depth > 0)
    )
    render_depth = original_depth.copy()
    render_valid = original_valid.copy()
    guide_filled = np.zeros(original_valid.shape, dtype=bool)

    if fill_radius > 0 and np.any(original_valid):
        raw_barrier = _thin_depth_step_edges(
            original_depth,
            original_valid,
            max(float(relative_spread), 0.02),
        )
        confidence = original_valid.astype(np.float32)
        (
            filled_depth,
            filled_valid,
            _,
            guide_filled,
            _,
        ) = fill_tiny_holes(
            original_depth,
            original_valid,
            confidence,
            raw_barrier,
            int(fill_radius),
            float(relative_spread),
            "adaptive",
        )
        # Never smooth or alter a metric depth that already existed.  Only the
        # newly completed pixels receive render-only projection depth.
        render_depth[guide_filled] = filled_depth[guide_filled]
        render_valid |= filled_valid

    guide = render_depth.copy()
    if median_radius > 0 and np.any(render_valid):
        log_depth = np.zeros(guide.shape, dtype=np.float32)
        log_depth[render_valid] = np.log(np.maximum(guide[render_valid], 1e-12))
        neutral = float(np.median(log_depth[render_valid]))
        working = np.where(render_valid, log_depth, neutral).astype(np.float32)
        working = _median_blur_float_radius(working, int(median_radius))
        guide[render_valid] = np.exp(working[render_valid])
    guide[~render_valid] = np.nan
    render_depth[~render_valid] = np.nan
    return render_depth, render_valid, guide, guide_filled


def render_reference(
    reference: np.ndarray,
    camera: CameraModel,
    target: CameraModel,
    target_rays: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    surface_guide_depth: np.ndarray,
    texture_interpolation: str,
    edge_depth_step: float,
    edge_nearest_radius: int,
    occlusion_tolerance: float,
    occlusion_fill_radius: int,
    occlusion_fill_relative_tolerance: float,
    unresolved_relaxed_depth_tolerance: float,
    unresolved_max_component_area: int,
    display_crack_radius: int,
    display_crack_relative_spread: float,
    edge_policy: str,
    fill_texture_method: str,
    inpaint_radius: float,
) -> ReferenceRender:
    safe_depth = np.where(valid & np.isfinite(depth), depth, 1.0)
    points = target_rays * safe_depth[..., None]
    pixels, z_reference = project_points(points.reshape(-1, 3), relative_pose(camera, target), camera)
    map_xy = pixels.reshape(*depth.shape, 2)
    z_reference = z_reference.reshape(depth.shape)
    sampleable = projection_sampleable(map_xy, z_reference, valid, camera.image_size)
    visible = zbuffer_visibility(map_xy, z_reference, valid, camera.image_size, occlusion_tolerance)
    remapped, edge_nearest_mask = remap_reference_texture(
        reference,
        map_xy,
        surface_guide_depth,
        valid,
        texture_interpolation,
        edge_depth_step,
        edge_nearest_radius,
    )
    aligned_raw_sampleable = remapped.copy()
    aligned_raw_sampleable[~sampleable] = 0
    aligned_zbuffer = remapped.copy()
    aligned_zbuffer[~visible] = 0
    aligned_complete, occlusion_filled = fill_occlusion_by_target_depth(
        remapped,
        sampleable,
        visible,
        surface_guide_depth,
        occlusion_fill_radius,
        occlusion_fill_relative_tolerance,
    )
    first_pass_visible = visible | occlusion_filled
    first_pass_unresolved = sampleable & ~first_pass_visible
    (
        aligned_complete,
        occlusion_relaxed_filled,
        occlusion_relaxed_eligible,
    ) = fill_small_unresolved_surface_components(
        aligned_complete,
        first_pass_visible,
        first_pass_unresolved,
        surface_guide_depth,
        occlusion_fill_radius,
        unresolved_relaxed_depth_tolerance,
        unresolved_max_component_area,
    )
    safe_visible = first_pass_visible | occlusion_relaxed_filled
    aligned_complete, display_filled = fill_display_cracks(
        aligned_complete, safe_visible, surface_guide_depth, display_crack_radius,
        display_crack_relative_spread, edge_policy,
    )
    # Never let the generic crack filler reintroduce a z-buffer-rejected raw
    # sample.  Such pixels must be filled from the matching target-depth layer
    # above or remain explicitly unavailable.
    unresolved_occlusion = (
        sampleable
        & ~visible
        & ~occlusion_filled
        & ~occlusion_relaxed_filled
    )
    aligned_complete[unresolved_occlusion] = 0
    display_filled[unresolved_occlusion] = False
    visual_mask = safe_visible | display_filled
    aligned_surface_copy = aligned_complete.copy()
    authorized_fill = (
        occlusion_filled
        | occlusion_relaxed_filled
        | display_filled
    )
    unavailable = ~visual_mask
    (
        aligned_complete,
        texture_structure_refined,
        texture_inpaint_solver_mask,
    ) = refine_filled_texture_structure(
        aligned_complete,
        authorized_fill,
        unavailable,
        fill_texture_method,
        inpaint_radius,
    )
    # Solver-only halo pixels and all unresolved disocclusions remain explicit
    # black.  Only geometry-authorized fill pixels receive the restored texture.
    aligned_complete[~visual_mask] = 0
    return ReferenceRender(
        aligned_complete=aligned_complete,
        aligned_surface_copy=aligned_surface_copy,
        aligned_raw_sampleable=aligned_raw_sampleable,
        aligned_zbuffer=aligned_zbuffer,
        sampleable=sampleable,
        zbuffer_visible=visible,
        visual_mask=visual_mask,
        occlusion_filled=occlusion_filled,
        occlusion_relaxed_filled=occlusion_relaxed_filled,
        occlusion_relaxed_eligible=occlusion_relaxed_eligible,
        display_filled=display_filled,
        texture_structure_refined=texture_structure_refined,
        texture_inpaint_solver_mask=texture_inpaint_solver_mask,
        edge_nearest_mask=edge_nearest_mask,
        map_xy=map_xy,
    )


def alpha_overlay(aligned: np.ndarray, target: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    output = target.copy()
    blended = cv2.addWeighted(aligned, alpha, target, 1.0 - alpha, 0)
    output[valid] = blended[valid]
    return output


def edge_overlay(aligned: np.ndarray, target: np.ndarray, valid: np.ndarray) -> np.ndarray:
    moving = cv2.Canny(robust_uint8(to_gray(aligned), True), 50, 130) > 0
    fixed = cv2.Canny(robust_uint8(to_gray(target), True), 50, 130) > 0
    moving &= valid
    fixed &= valid
    output = np.zeros((*valid.shape, 3), dtype=np.uint8)
    output[moving & ~fixed] = (0, 0, 255)
    output[fixed & ~moving] = (255, 255, 0)
    output[moving & fixed] = (255, 255, 255)
    return output


def parse_named_paths(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise PriorAlignmentError(
                f"invalid --reference-image {value!r}; expected CAMERA=PATH"
            )
        if name in output:
            raise PriorAlignmentError(f"duplicate --reference-image for {name}")
        output[name] = Path(raw_path)
    return output


def resolve_paths(args: argparse.Namespace) -> tuple[dict[str, Path], Path]:
    explicit = parse_named_paths(args.reference_image)
    unknown = set(explicit).difference(REFERENCE_NAMES)
    if unknown:
        raise PriorAlignmentError(
            "--reference-image contains unknown cameras: " + ", ".join(sorted(unknown))
        )
    if args.image_root is not None:
        root = args.image_root.resolve()
        for name in REFERENCE_NAMES:
            if name not in explicit:
                explicit[name] = root / name / args.frame
        if args.target_image is None:
            args.target_image = root / args.target_camera / args.frame
    missing = [name for name in REFERENCE_NAMES if name not in explicit]
    if missing or args.target_image is None:
        raise PriorAlignmentError(f"缺少图像路径：{missing + ([] if args.target_image else ['target'])}")
    return {name: path.resolve() for name, path in explicit.items()}, Path(args.target_image).resolve()


def parse_camera_weights(text: str) -> dict[str, float]:
    output = {name: 1.0 for name in REFERENCE_NAMES}
    if not text.strip():
        return output
    for item in text.split(","):
        if "=" not in item:
            raise PriorAlignmentError(
                "--camera-weights must be CAMERA=WEIGHT pairs separated by commas"
            )
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in REFERENCE_NAMES:
            raise PriorAlignmentError(f"未知相机权重：{name}")
        output[name] = float(value)
    if any(not math.isfinite(v) or v <= 0 for v in output.values()):
        raise PriorAlignmentError("所有相机权重必须为正数")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Depth Anything V2 dense priors + calibrated multi-camera geometry "
            "+ reprojection into an unknown-intrinsics target camera"
        )
    )
    parser.add_argument("--reference-cameras", nargs="+", required=True)
    parser.add_argument("--target-camera", required=True)
    parser.add_argument("--anchor-camera", required=True)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--geometry-npz", required=True, type=Path, help="严格几何阶段输出的depth_alignment_maps.npz")
    parser.add_argument("--depth-anything-root", required=True, type=Path, help="Depth-Anything-V2官方仓库根目录")
    parser.add_argument("--checkpoint", required=True, type=Path, help="depth_anything_v2_vits.pth等权重")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--frame", default="frame_000.jpg")
    parser.add_argument(
        "--reference-image",
        action="append",
        default=[],
        metavar="CAMERA=PATH",
        help="repeat once per reference camera when --image-root is not used",
    )
    parser.add_argument("--target-image", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-width", type=int)
    parser.add_argument("--target-height", type=int)
    parser.add_argument("--encoder", choices=("vits", "vitb", "vitl", "vitg"), default="vits")
    parser.add_argument("--model-input-size", type=int, default=518)
    parser.add_argument("--projection-max-side", type=int, default=1600)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp16", "fp32"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--geometry-mask", choices=("reliable", "render"), default="reliable")
    parser.add_argument(
        "--prior-cameras", nargs="+",
        help="references used for dense-prior fusion; defaults to every reference camera",
    )
    parser.add_argument(
        "--reference-depth-camera",
        help="reference whose learned surface is preserved where valid; defaults to the anchor",
    )
    parser.add_argument(
        "--final-depth-source",
        choices=("model-raw", "refined"),
        default="model-raw",
        help="model-raw preserves reference-model boundaries; refined enables target/SAM refinement",
    )
    parser.add_argument("--anchor-erode", type=int, default=1)
    parser.add_argument("--maximum-anchors", type=int, default=6000)
    parser.add_argument("--minimum-anchors", type=int, default=80)
    parser.add_argument("--ransac-iterations", type=int, default=400)
    parser.add_argument("--anchor-inlier-relative", type=float, default=0.20)
    parser.add_argument("--maximum-fit-median-error", type=float, default=0.30)
    parser.add_argument("--depth-range-expand", type=float, default=4.0)
    parser.add_argument("--anchor-distance-scale", type=float, default=280.0)
    parser.add_argument("--anchor-confidence-floor", type=float, default=0.25)
    parser.add_argument("--depth-edge-relative", type=float, default=0.08)
    parser.add_argument("--boundary-width", type=int, default=1)
    parser.add_argument("--boundary-confidence-factor", type=float, default=0.20)
    parser.add_argument("--prior-consistency", type=float, default=0.15)
    parser.add_argument("--geometry-agreement-relative", type=float, default=0.20)
    parser.add_argument(
        "--geometry-selection-margin", type=float, default=0.05,
        help="几何选择另一模型时，误差至少需要改善的相对幅度",
    )
    parser.add_argument("--geometry-min-confidence", type=float, default=0.40)
    parser.add_argument("--geometry-min-support", type=int, default=2)
    parser.add_argument("--geometry-conflict-confidence-factor", type=float, default=0.50)
    parser.add_argument("--model-conflict-confidence-factor", type=float, default=0.50)
    parser.add_argument("--strict-min-views", type=int)
    parser.add_argument("--strict-min-confidence", type=float, default=0.20)
    parser.add_argument(
        "--complete-min-confidence", type=float, default=0.04,
        help="只统计低置信可渲染像素；不会再把这些像素清黑",
    )
    parser.add_argument(
        "--camera-weights", default="",
        help="comma-separated CAMERA=WEIGHT values; unspecified references default to 1",
    )
    parser.add_argument("--row-chunk", type=int, default=64)
    parser.add_argument("--zbuffer-tolerance", type=float, default=0.005)
    parser.add_argument("--occlusion-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--render-occlusion-fill-radius", type=int, default=24,
        help="Z-buffer拒绝后，沿相同目标深度表面补全窄遮挡带的最大半径；0关闭",
    )
    parser.add_argument(
        "--render-surface-depth-tolerance", type=float, default=0.06,
        help="遮挡补全允许的同一表面相对深度差；越小越不容易跨前后景",
    )
    parser.add_argument(
        "--render-surface-guide-fill-radius", type=int, default=8,
        help="仅在显示层补全深度引导孔洞的最大半径；不写回最终/严格深度",
    )
    parser.add_argument(
        "--render-surface-guide-median-radius", type=int, default=2,
        help="显示层表面标签的中值半径；默认2即5x5，去除柱内零碎深度线",
    )
    parser.add_argument(
        "--render-unresolved-relaxed-depth-tolerance", type=float, default=0.25,
        help="仅对小型unresolved组件使用的第二级同表面深度容差；大深度边界仍不可跨越",
    )
    parser.add_argument(
        "--render-unresolved-max-component-area", type=int, default=512,
        help="第二级补色允许处理的单个unresolved组件最大像素数；0关闭",
    )
    parser.add_argument(
        "--render-fill-texture-method",
        choices=("navier-stokes", "telea", "copy"),
        default="navier-stokes",
        help=(
            "几何补全后的纹理恢复；navier-stokes沿边缘/等照度线方向延续，"
            "telea更快，copy保留旧版邻色复制"
        ),
    )
    parser.add_argument(
        "--render-inpaint-radius", type=float, default=3.0,
        help=(
            "仅在几何允许填补的像素内进行结构延续修复的半径；默认3，"
            "建议2到4，0等同copy"
        ),
    )
    parser.add_argument(
        "--render-interpolation",
        choices=("edge-aware", "linear", "nearest"),
        default="edge-aware",
        help="纹理重映射插值；edge-aware仅在深度边界改用最近邻，避免前后景混色",
    )
    parser.add_argument(
        "--render-edge-depth-step", type=float, default=0.04,
        help="相邻目标深度相差至少该比例时，视为需要锐利采样的物体边界",
    )
    parser.add_argument(
        "--render-edge-nearest-radius", type=int, default=1,
        help="深度边界两侧改用最近邻的半径；默认1像素，0只改边界像素",
    )
    parser.add_argument(
        "--fill-radius", type=int, default=2,
        help="深度栅格裂缝的最大补全宽度；默认2像素，不扩散大块区域",
    )
    parser.add_argument("--fill-relative-spread", type=float, default=0.05)
    parser.add_argument(
        "--edge-crack-policy",
        choices=("adaptive", "foreground", "background", "highest-confidence"),
        default="adaptive",
        help="深度断层裂缝选哪一侧表面；adaptive按目标深度或局部支持自动选择",
    )
    parser.add_argument(
        "--edge-sharpen-radius", type=int, default=4,
        help="把单目模型的多层过渡带压成明确前/后景边缘；0关闭，建议3到6",
    )
    parser.add_argument(
        "--edge-sharpen-passes", type=int, choices=(1, 2, 3), default=2,
        help="过渡带锐化轮数；默认2轮可消除较宽的多层等深线",
    )
    parser.add_argument(
        "--edge-sharpen-relative", type=float, default=0.10,
        help="局部前后景深度至少相差该比例才执行边缘锐化",
    )
    parser.add_argument(
        "--edge-sharpen-confidence-factor", type=float, default=0.35,
        help="锐化边缘置信度折减；严格输出仍使用未锐化深度",
    )
    parser.add_argument(
        "--target-edge-snap-radius", type=int, default=24,
        help="允许模型深度轮廓向目标图像结构边缘移动的最大带宽；0关闭，建议18到30",
    )
    parser.add_argument(
        "--target-edge-depth-step", type=float, default=0.025,
        help="检测错误平行深度轮廓的最小相邻深度变化比例",
    )
    parser.add_argument(
        "--target-edge-gradient-weight", type=float, default=12.0,
        help="目标图像梯度对深度传播的阻力权重",
    )
    parser.add_argument(
        "--target-edge-hard-penalty", type=float, default=120.0,
        help="跨越目标图像Canny边缘的额外代价；越大越强制在目标边缘分界",
    )
    parser.add_argument(
        "--target-edge-max-band-fraction", type=float, default=0.65,
        help="安全门：目标图像引导最多允许修改的图像面积比例",
    )
    parser.add_argument(
        "--segmentation-backend", choices=("off", "mobilesam", "sam"), default="off",
        help="使用提示式实例分割作为深度硬边界；显存有限时推荐mobilesam",
    )
    parser.add_argument(
        "--segmentation-root", type=Path,
        help="MobileSAM或segment-anything官方仓库根目录；已pip安装时可省略",
    )
    parser.add_argument(
        "--segmentation-checkpoint", type=Path,
        help="MobileSAM的mobile_sam.pt或原版SAM权重",
    )
    parser.add_argument(
        "--segmentation-model-type",
        help="默认MobileSAM=vit_t，原版SAM=vit_b",
    )
    parser.add_argument(
        "--segmentation-guide-image", type=Path,
        help="可选：已处于目标坐标的RGB/结构图；未指定则直接使用目标图像",
    )
    parser.add_argument("--segmentation-prompt-margin", type=int, default=40)
    parser.add_argument("--segmentation-points-per-side", type=int, default=3)
    parser.add_argument("--segmentation-min-component-area", type=int, default=30)
    parser.add_argument("--segmentation-max-components", type=int, default=24)
    parser.add_argument(
        "--segmentation-min-depth-separation", type=float, default=0.08,
        help="候选物体内外深度至少相差该比例",
    )
    parser.add_argument("--segmentation-min-score", type=float, default=0.55)
    parser.add_argument(
        "--segmentation-min-edge-alignment", type=float, default=0.05,
        help="分割轮廓落在目标图像边缘±2px内的最低比例",
    )
    parser.add_argument("--segmentation-min-mask-fraction", type=float, default=0.005)
    parser.add_argument("--segmentation-max-mask-fraction", type=float, default=0.90)
    parser.add_argument(
        "--render-crack-radius", type=int, default=2,
        help="只对封闭的细小显示裂缝做补色；0关闭，严格图和几何掩膜不受影响",
    )
    parser.add_argument("--target-edge-low", type=int, default=70)
    parser.add_argument("--target-edge-high", type=int, default=170)
    parser.add_argument("--overlay-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--invert-legacy-poses", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "model_input_size", "projection_max_side",
        "maximum_anchors", "minimum_anchors", "ransac_iterations", "depth_range_expand",
        "anchor_distance_scale", "row_chunk",
    )
    for name in positive:
        if float(getattr(args, name)) <= 0:
            raise PriorAlignmentError(f"--{name.replace('_', '-')}必须大于0")
    fractions = (
        "anchor_inlier_relative", "maximum_fit_median_error", "anchor_confidence_floor",
        "depth_edge_relative", "boundary_confidence_factor", "prior_consistency",
        "geometry_agreement_relative", "geometry_selection_margin", "geometry_min_confidence",
        "geometry_conflict_confidence_factor", "model_conflict_confidence_factor",
        "strict_min_confidence", "complete_min_confidence", "zbuffer_tolerance",
        "occlusion_tolerance", "render_surface_depth_tolerance", "render_edge_depth_step",
        "render_unresolved_relaxed_depth_tolerance",
        "fill_relative_spread", "edge_sharpen_relative",
        "edge_sharpen_confidence_factor", "target_edge_depth_step",
        "target_edge_max_band_fraction", "segmentation_min_depth_separation",
        "segmentation_min_score", "segmentation_min_edge_alignment",
        "segmentation_min_mask_fraction", "segmentation_max_mask_fraction",
        "overlay_alpha",
    )
    for name in fractions:
        value = float(getattr(args, name))
        if value < 0 or value > 1:
            raise PriorAlignmentError(f"--{name.replace('_', '-')}必须在0到1之间")
    if (
        args.fill_radius < 0 or args.render_crack_radius < 0
        or args.render_occlusion_fill_radius < 0
        or args.render_edge_nearest_radius < 0
        or args.render_surface_guide_fill_radius < 0
        or args.render_surface_guide_median_radius < 0
        or args.render_unresolved_max_component_area < 0
        or args.render_inpaint_radius < 0
        or args.edge_sharpen_radius < 0
        or args.target_edge_snap_radius < 0
        or args.segmentation_prompt_margin < 0
        or args.boundary_width < 0 or args.anchor_erode < 0
    ):
        raise PriorAlignmentError("半径/腐蚀参数不能为负数")
    if args.minimum_anchors > args.maximum_anchors:
        raise PriorAlignmentError("--minimum-anchors不能大于--maximum-anchors")
    if (args.target_width is None) != (args.target_height is None):
        raise PriorAlignmentError("--target-width和--target-height必须同时指定或同时省略")
    if args.target_width is not None and min(args.target_width, args.target_height) <= 0:
        raise PriorAlignmentError("目标输出尺寸必须大于0")
    if not args.prior_cameras:
        raise PriorAlignmentError("--prior-cameras至少需要一台参考相机")
    unknown_priors = set(args.prior_cameras).difference(REFERENCE_NAMES)
    if unknown_priors:
        raise PriorAlignmentError(
            "--prior-cameras包含未知参考相机：" + ", ".join(sorted(unknown_priors))
        )
    if len(set(args.prior_cameras)) != len(args.prior_cameras):
        raise PriorAlignmentError("--prior-cameras中有重复相机")
    if args.reference_depth_camera not in args.prior_cameras:
        raise PriorAlignmentError("--reference-depth-camera必须包含在--prior-cameras中")
    if args.strict_min_views < 1 or args.strict_min_views > len(args.prior_cameras):
        raise PriorAlignmentError(
            "--strict-min-views必须在1到参与深度融合的参考相机数量之间"
        )
    if args.geometry_min_support < 1:
        raise PriorAlignmentError("--geometry-min-support至少为1")
    if args.target_edge_gradient_weight < 0 or args.target_edge_hard_penalty < 0:
        raise PriorAlignmentError("目标边缘传播权重不能为负数")
    if args.target_edge_depth_step <= 0:
        raise PriorAlignmentError("--target-edge-depth-step必须大于0")
    if args.render_edge_depth_step <= 0:
        raise PriorAlignmentError("--render-edge-depth-step必须大于0")
    if args.target_edge_max_band_fraction <= 0:
        raise PriorAlignmentError("--target-edge-max-band-fraction必须大于0")
    if args.segmentation_backend != "off" and args.segmentation_checkpoint is None:
        raise PriorAlignmentError(
            "启用物体分割时必须提供--segmentation-checkpoint"
        )
    if args.segmentation_points_per_side < 1:
        raise PriorAlignmentError("--segmentation-points-per-side至少为1")
    if args.segmentation_min_component_area < 1 or args.segmentation_max_components < 1:
        raise PriorAlignmentError("物体分割的组件面积和最大组件数必须大于0")
    if args.segmentation_min_mask_fraction >= args.segmentation_max_mask_fraction:
        raise PriorAlignmentError(
            "--segmentation-min-mask-fraction必须小于--segmentation-max-mask-fraction"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_rig(args.reference_cameras, args.target_camera, args.anchor_camera)
    if args.prior_cameras is None:
        args.prior_cameras = list(REFERENCE_NAMES)
    if args.reference_depth_camera is None:
        args.reference_depth_camera = args.anchor_camera
    if args.strict_min_views is None:
        args.strict_min_views = min(2, len(args.prior_cameras))
    validate_args(args)
    camera_weights = parse_camera_weights(args.camera_weights)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise PriorAlignmentError(f"输出目录非空：{output_dir}；确认后添加--overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_paths, target_path = resolve_paths(args)
    target_native = read_image(target_path)
    target_size = (
        (args.target_width, args.target_height)
        if args.target_width is not None
        else (target_native.shape[1], target_native.shape[0])
    )
    calibration = read_json(args.calibration.resolve())
    reference_models, target_model = load_camera_models(
        calibration,
        args.anchor_camera,
        args.target_camera,
        target_size,
        args.invert_legacy_poses,
    )
    geometry_depth, geometry_mask, geometry_confidence, geometry_support, geometry_report = load_geometry_npz(
        args.geometry_npz.resolve(), target_size, args.geometry_mask, args.anchor_erode
    )

    target = resize_target(target_native, target_size)
    target_gray = robust_uint8(to_gray(target), clahe=True)
    target_edge = cv2.Canny(target_gray, args.target_edge_low, args.target_edge_high) > 0
    segmentation_guide = target
    if args.segmentation_guide_image is not None:
        segmentation_guide = resize_target(
            read_image(args.segmentation_guide_image.resolve()), target_size
        )
    segmentation_guide = prepare_segmentation_guide(segmentation_guide)
    target_rays = normalized_rays(target_model)

    references: dict[str, np.ndarray] = {}
    for name in REFERENCE_NAMES:
        image = read_image(reference_paths[name])
        actual = (image.shape[1], image.shape[0])
        if actual != reference_models[name].image_size:
            raise PriorAlignmentError(
                f"{name}图像尺寸{actual}与校准尺寸{reference_models[name].image_size}不一致"
            )
        references[name] = image

    runner = DepthAnythingRunner(
        args.depth_anything_root.resolve(), args.checkpoint.resolve(), args.encoder,
        args.device, args.precision, args.allow_cpu,
    )
    candidates: list[DepthCandidate] = []
    failures: dict[str, str] = {}

    for camera_index, name in enumerate(args.prior_cameras):
        print(f"\n[{name}] 单目深度推理……")
        try:
            work_image = make_work_image(references[name], args.projection_max_side)
            work_camera = resize_camera_model(
                reference_models[name], (work_image.shape[1], work_image.shape[0])
            )
            raw = runner.infer(work_image, args.model_input_size)
            anchor_xy, anchor_raw, anchor_depth, anchor_weight = extract_reference_anchors(
                reference_models[name], work_camera, target_model, target_rays,
                geometry_depth, geometry_mask, geometry_confidence, geometry_support,
                raw, args.maximum_anchors,
            )
            if anchor_xy.shape[0] < args.minimum_anchors:
                raise PriorAlignmentError(
                    f"[{name}] 投影后锚点{anchor_xy.shape[0]}少于{args.minimum_anchors}"
                )
            fit, inliers = robust_inverse_depth_fit(
                name, anchor_raw, anchor_depth, anchor_weight,
                args.ransac_iterations, args.anchor_inlier_relative, args.seed + camera_index,
            )
            print(
                f"[{name}] 锚点={fit.anchors_total}，内点={fit.anchors_inlier} "
                f"({100*fit.inlier_ratio:.1f}%)，验证中位相对误差={100*fit.validation_median_relative_error:.2f}%"
            )
            if fit.validation_median_relative_error > args.maximum_fit_median_error:
                raise PriorAlignmentError(
                    f"[{name}] 深度先验锚定误差{fit.validation_median_relative_error:.3f} "
                    f"> {args.maximum_fit_median_error:.3f}"
                )

            dense_depth, dense_confidence, dense_boundary = build_dense_reference_depth(
                raw, fit, anchor_xy[inliers], work_image, args
            )
            projected_depth, projected_confidence, projected_boundary = rasterize_reference_depth_camera(
                reference_models[name], work_camera, target_model,
                dense_depth, dense_confidence, dense_boundary,
                args.row_chunk, args.zbuffer_tolerance,
            )
            valid = np.isfinite(projected_depth) & (projected_depth > 0) & (projected_confidence > 0)
            report = {
                "reference_image": str(reference_paths[name]),
                "work_size": list(work_camera.image_size),
                "fit": asdict(fit),
                "projected_valid_ratio": float(np.count_nonzero(valid) / valid.size),
                "projected_depth_quantiles": quantiles(projected_depth[valid]),
                "projected_confidence_quantiles": quantiles(projected_confidence[valid]),
            }
            candidates.append(
                DepthCandidate(name, projected_depth, projected_confidence, valid, projected_boundary, fit, report)
            )
            write_image(output_dir / f"{name}_model_raw.png", robust_uint8(raw))
            write_image(output_dir / f"{name}_model_depth.png", colorize_depth(dense_depth, np.isfinite(dense_depth)))
            write_image(output_dir / f"{name}_prior_target.png", colorize_depth(projected_depth, valid))
            write_image(output_dir / f"{name}_prior_confidence.png", np.round(projected_confidence * 255).astype(np.uint8))
            write_image(output_dir / f"{name}_prior_boundary.png", projected_boundary.astype(np.uint8) * 255)
        except PriorAlignmentError as exc:
            failures[name] = str(exc)
            print(f"警告：{exc}；跳过该相机深度先验")

    if not candidates:
        raise PriorAlignmentError(f"所有参考相机都未产生可用深度先验：{failures}")
    if len(candidates) < args.strict_min_views:
        print(f"警告：只有{len(candidates)}台相机可用；严格掩膜可能很少")

    # Depth Anything has finished.  Release it before an optional SAM encoder
    # is loaded so memory-constrained GPUs never hold both networks at once.
    torch_runtime = runner.torch
    depth_device = runner.device_name
    del runner
    gc.collect()
    if depth_device == "cuda":
        torch_runtime.cuda.empty_cache()

    (
        prior_depth, prior_confidence, support, prior_strict, prior_complete,
        uncertain_boundary, geometry_agreement, geometry_conflict,
        model_conflict, selected_camera, fusion_report,
    ) = fuse_candidates(
        candidates, camera_weights, args.prior_consistency, args.strict_min_views,
        args.strict_min_confidence, args.complete_min_confidence,
        args.reference_depth_camera,
        geometry_depth, geometry_mask, geometry_confidence, geometry_support,
        args.geometry_agreement_relative, args.geometry_selection_margin,
        args.geometry_min_confidence, args.geometry_min_support,
        args.geometry_conflict_confidence_factor,
        args.model_conflict_confidence_factor,
    )

    # Re-assert the reference model after fusion.  ``fuse_candidates`` may use
    # reliable geometry to select another learned surface at conflicts, which
    # is useful for the refined candidate but can make a thin rigid object
    # alternate between foreground and background.  The model-raw branch keeps
    # the reference model exactly wherever it exists; other references fill only
    # genuine holes in that reference projection.
    reference_candidate = next(
        candidate for candidate in candidates
        if candidate.camera == args.reference_depth_camera
    )
    (
        reference_locked_depth,
        reference_locked_confidence,
        reference_locked_complete,
        reference_surface_override,
    ) = lock_reference_surface(
        prior_depth,
        prior_confidence,
        prior_complete,
        reference_candidate,
        camera_weights[args.reference_depth_camera],
    )
    reference_valid_locked = (
        np.asarray(reference_candidate.valid, dtype=bool)
        & np.isfinite(reference_candidate.depth_target)
        & (reference_candidate.depth_target > 0)
    )
    reference_camera_code = next(
        index + 1 for index, candidate in enumerate(candidates)
        if candidate.camera == args.reference_depth_camera
    )
    selected_camera = selected_camera.copy()
    selected_camera[reference_valid_locked] = reference_camera_code
    fusion_report["post_lock_selected_camera_pixels"] = {
        candidate.camera: int(np.count_nonzero(selected_camera == index + 1))
        for index, candidate in enumerate(candidates)
    }
    fusion_report["reference_surface_override_ratio"] = float(
        np.count_nonzero(reference_surface_override) / reference_surface_override.size
    )

    # Both branches start from the same reference-locked learned surface.
    # Calibrated geometry has already supplied metric scale and confidence; no
    # sparse triangulation value is copied into this depth map.
    final_depth = reference_locked_depth.copy()
    final_confidence = reference_locked_confidence.copy()
    final_complete = reference_locked_complete.copy()
    final_strict = prior_strict.copy()

    barrier = uncertain_boundary | target_edge
    final_depth, final_complete, final_confidence, filled, edge_filled = fill_tiny_holes(
        final_depth, final_complete, final_confidence, barrier,
        args.fill_radius, args.fill_relative_spread, args.edge_crack_policy,
    )
    # The model-raw candidate permits only the small raster-hole completion
    # above.  It is the conservative choice when the reference monocular model
    # already separates a pole/person/sign correctly.
    model_raw_depth = final_depth.copy()
    model_raw_confidence = final_confidence.copy()
    model_raw_complete = final_complete.copy()
    strict_source_depth = model_raw_depth.copy()
    edge_sharpened = np.zeros(final_complete.shape, dtype=bool)
    for _ in range(args.edge_sharpen_passes):
        final_depth, sharpened_this_pass = sharpen_depth_discontinuities(
            final_depth,
            final_complete,
            uncertain_boundary,
            args.edge_sharpen_radius,
            args.edge_sharpen_relative,
        )
        edge_sharpened |= sharpened_this_pass
        if not np.any(sharpened_this_pass):
            break
    final_confidence[edge_sharpened] *= args.edge_sharpen_confidence_factor
    local_sharpened_depth = final_depth.copy()
    (
        final_depth,
        target_snap_band,
        target_edge_snapped,
        target_snap_report,
    ) = target_guided_depth_propagation(
        final_depth,
        final_complete,
        target_gray,
        target_edge,
        args.target_edge_snap_radius,
        args.target_edge_depth_step,
        args.edge_sharpen_relative,
        args.target_edge_gradient_weight,
        args.target_edge_hard_penalty,
        args.target_edge_max_band_fraction,
    )
    final_confidence[target_edge_snapped] *= args.edge_sharpen_confidence_factor
    target_only_depth = final_depth.copy()
    segmentation_object_mask = np.zeros(final_complete.shape, dtype=bool)
    segmentation_changed = np.zeros(final_complete.shape, dtype=bool)
    segmentation_thin_depth_edges = np.zeros(final_complete.shape, dtype=bool)
    segmentation_prompt_visual = segmentation_guide.copy()
    segmentation_mask_overlay = segmentation_guide.copy()
    segmentation_report: dict[str, Any] = {
        "enabled": False,
        "status": "disabled",
        "backend": args.segmentation_backend,
        "components_accepted": 0,
        "changed_ratio": 0.0,
    }
    if args.segmentation_backend != "off":
        model_type = args.segmentation_model_type
        if model_type is None:
            model_type = "vit_t" if args.segmentation_backend == "mobilesam" else "vit_b"
        segmentation_runner = PromptSegmentationRunner(
            args.segmentation_backend,
            args.segmentation_root.resolve() if args.segmentation_root is not None else None,
            args.segmentation_checkpoint.resolve(),
            model_type,
            args.device,
            args.allow_cpu,
        )
        (
            final_depth,
            segmentation_object_mask,
            segmentation_changed,
            segmentation_prompt_visual,
            segmentation_mask_overlay,
            segmentation_report,
        ) = segmentation_guided_depth_partition(
            final_depth,
            local_sharpened_depth,
            final_complete,
            target_snap_band,
            segmentation_guide,
            target_edge,
            segmentation_runner,
            max(int(target_snap_report.get("used_radius", 0)), args.edge_sharpen_radius, 1),
            args.target_edge_depth_step,
            args.segmentation_prompt_margin,
            args.segmentation_points_per_side,
            args.segmentation_min_component_area,
            args.segmentation_max_components,
            args.segmentation_min_depth_separation,
            args.segmentation_min_score,
            args.segmentation_min_edge_alignment,
            args.segmentation_min_mask_fraction,
            args.segmentation_max_mask_fraction,
        )
        segmentation_report.update({
            "backend": args.segmentation_backend,
            "model_type": model_type,
            "checkpoint": str(args.segmentation_checkpoint.resolve()),
            "guide_image": (
                str(args.segmentation_guide_image.resolve())
                if args.segmentation_guide_image is not None else str(target_path)
            ),
        })
        segmentation_thin_depth_edges = _thin_depth_step_edges(
            local_sharpened_depth,
            final_complete,
            args.target_edge_depth_step,
        )
        segmentation_thin_depth_edges = cv2.morphologyEx(
            segmentation_thin_depth_edges.astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
        ) > 0
        final_confidence[segmentation_changed] *= args.edge_sharpen_confidence_factor
        del segmentation_runner
        gc.collect()
        if depth_device == "cuda":
            torch_runtime.cuda.empty_cache()

    # Preserve both candidates in every run.  Selecting model-raw changes only
    # the final/rendered branch; all target/SAM diagnostics remain available
    # for an A/B comparison without a second neural-network inference.
    refined_candidate_depth = final_depth.copy()
    refined_candidate_confidence = final_confidence.copy()
    if args.final_depth_source == "model-raw":
        final_depth = model_raw_depth.copy()
        final_confidence = model_raw_confidence.copy()
        final_complete = model_raw_complete.copy()
    segmentation_report["applied_to_final"] = bool(
        args.final_depth_source == "refined" and np.any(segmentation_changed)
    )
    segmentation_report["final_depth_source"] = args.final_depth_source
    (
        render_depth,
        render_complete,
        render_surface_guide_depth,
        render_surface_guide_filled,
    ) = build_render_surface_guide(
        final_depth,
        final_complete,
        args.render_surface_guide_fill_radius,
        args.render_surface_guide_median_radius,
        args.render_surface_depth_tolerance,
    )
    print(
        f"最终深度来源={args.final_depth_source}；"
        + (
            "参考模型表面已锁定，目标图像/SAM精修仅保留为候选对照"
            if args.final_depth_source == "model-raw"
            else "使用目标图像/SAM精修候选"
        )
    )
    print(
        f"渲染表面引导补孔="
        f"{100*np.count_nonzero(render_surface_guide_filled)/render_surface_guide_filled.size:.2f}%；"
        f"中值半径={args.render_surface_guide_median_radius}px；"
        "只影响完整显示，不改最终/严格深度"
    )
    print(
        f"\n严格深度覆盖={100*np.count_nonzero(final_strict)/final_strict.size:.2f}%；"
        f"完整先验覆盖={100*np.count_nonzero(final_complete)/final_complete.size:.2f}%；"
        f"微裂缝填补={100*np.count_nonzero(filled)/filled.size:.2f}%；"
        f"过渡带二表面锐化={100*np.count_nonzero(edge_sharpened)/edge_sharpened.size:.2f}%；"
        f"目标边缘重定位={100*np.count_nonzero(target_edge_snapped)/target_edge_snapped.size:.2f}%；"
        f"实例蒙版重判={100*np.count_nonzero(segmentation_changed)/segmentation_changed.size:.2f}%"
    )
    print(
        f"模型冲突={100*np.count_nonzero(model_conflict)/model_conflict.size:.2f}%；"
        f"几何一致={100*np.count_nonzero(geometry_agreement)/geometry_agreement.size:.2f}%；"
        f"几何冲突={100*np.count_nonzero(geometry_conflict)/geometry_conflict.size:.2f}%"
    )
    print(
        f"目标图像引导状态={target_snap_report['status']}；"
        f"使用半径={target_snap_report['used_radius']}px；"
        f"重判宽带={100*float(target_snap_report['band_ratio']):.2f}%"
    )
    if args.segmentation_backend != "off":
        print(
            f"物体分割状态={segmentation_report['status']}；"
            f"接受实例={segmentation_report['components_accepted']}/"
            f"{segmentation_report.get('components_tested', 0)}；"
            f"蒙版内近景={segmentation_report.get('accepted_near_inside', 0)}；"
            f"蒙版内远景={segmentation_report.get('accepted_far_inside', 0)}"
        )
        print(
            "物体候选："
            f"原宽带连通域={segmentation_report.get('global_band_components', 0)}；"
            f"细轮廓={segmentation_report.get('thin_edge_components_total', 0)}；"
            f"局部候选={segmentation_report.get('local_proposals_after_limit', 0)}"
        )
        if segmentation_report.get("rejection_reasons"):
            print(f"候选拒绝原因={segmentation_report['rejection_reasons']}")
        if segmentation_report.get("failed_quality_gates"):
            print(f"未通过质量门={segmentation_report['failed_quality_gates']}")

    write_image(output_dir / "target_reference.jpg", target)
    write_image(output_dir / "target_reference.png", target)
    write_image(output_dir / "geometry_anchor_mask.png", geometry_mask.astype(np.uint8) * 255)
    write_image(output_dir / "geometry_agreement_mask.png", geometry_agreement.astype(np.uint8) * 255)
    write_image(output_dir / "geometry_conflict_mask.png", geometry_conflict.astype(np.uint8) * 255)
    write_image(output_dir / "model_conflict_mask.png", model_conflict.astype(np.uint8) * 255)
    write_image(
        output_dir / "reference_surface_override_mask.png",
        reference_surface_override.astype(np.uint8) * 255,
    )
    selected_visual = np.round(selected_camera.astype(np.float32) / max(len(candidates), 1) * 255).astype(np.uint8)
    write_image(output_dir / "depth_selected_camera.png", selected_visual)
    write_image(output_dir / "depth_prior_fused.png", colorize_depth(prior_depth, prior_complete))
    write_image(
        output_dir / "depth_final_unsharpened.png",
        colorize_depth(strict_source_depth, final_complete),
    )
    write_image(
        output_dir / "depth_final_local_sharpened.png",
        colorize_depth(local_sharpened_depth, final_complete),
    )
    write_image(
        output_dir / "depth_final_target_only.png",
        colorize_depth(target_only_depth, final_complete),
    )
    write_image(
        output_dir / "depth_final_model_raw.png",
        colorize_depth(model_raw_depth, model_raw_complete),
    )
    write_image(
        output_dir / "depth_final_refined_candidate.png",
        colorize_depth(refined_candidate_depth, final_complete),
    )
    write_image(output_dir / "depth_final.png", colorize_depth(final_depth, final_complete))
    write_image(
        output_dir / "depth_render_projection.png",
        colorize_depth(render_depth, render_complete),
    )
    write_image(
        output_dir / "depth_render_surface_guide.png",
        colorize_depth(render_surface_guide_depth, render_complete),
    )
    write_image(output_dir / "depth_strict_mask.png", final_strict.astype(np.uint8) * 255)
    write_image(output_dir / "depth_complete_mask.png", final_complete.astype(np.uint8) * 255)
    write_image(output_dir / "depth_render_complete_mask.png", render_complete.astype(np.uint8) * 255)
    write_image(
        output_dir / "depth_render_surface_guide_filled_mask.png",
        render_surface_guide_filled.astype(np.uint8) * 255,
    )
    write_image(output_dir / "depth_filled_only_mask.png", filled.astype(np.uint8) * 255)
    write_image(output_dir / "depth_edge_crack_filled_mask.png", edge_filled.astype(np.uint8) * 255)
    write_image(output_dir / "depth_edge_sharpened_mask.png", edge_sharpened.astype(np.uint8) * 255)
    write_image(output_dir / "depth_target_snap_band.png", target_snap_band.astype(np.uint8) * 255)
    write_image(output_dir / "depth_target_edge_snapped_mask.png", target_edge_snapped.astype(np.uint8) * 255)
    write_image(output_dir / "target_structure_edges.png", target_edge.astype(np.uint8) * 255)
    write_image(output_dir / "segmentation_guide.jpg", segmentation_guide)
    write_image(
        output_dir / "segmentation_thin_depth_edges.png",
        segmentation_thin_depth_edges.astype(np.uint8) * 255,
    )
    write_image(output_dir / "segmentation_prompts.jpg", segmentation_prompt_visual)
    write_image(output_dir / "segmentation_object_overlay.jpg", segmentation_mask_overlay)
    write_image(
        output_dir / "segmentation_object_mask.png",
        segmentation_object_mask.astype(np.uint8) * 255,
    )
    write_image(
        output_dir / "depth_segmentation_changed_mask.png",
        segmentation_changed.astype(np.uint8) * 255,
    )
    write_image(output_dir / "depth_uncertain_boundary.png", uncertain_boundary.astype(np.uint8) * 255)
    write_image(output_dir / "depth_confidence.png", np.round(np.clip(final_confidence, 0, 1) * 255).astype(np.uint8))
    write_image(
        output_dir / "depth_support_count.png",
        np.round(support.astype(np.float32) / max(len(candidates), 1) * 255).astype(np.uint8),
    )

    maps: dict[str, np.ndarray] = {
        "target_depth_z": final_depth.astype(np.float32),
        "target_render_projection_depth_z": render_depth.astype(np.float32),
        "target_render_surface_guide_depth_z": render_surface_guide_depth.astype(np.float32),
        "target_render_complete_mask": render_complete.astype(np.uint8),
        "target_render_surface_guide_filled_mask": render_surface_guide_filled.astype(np.uint8),
        "target_depth_model_raw_z": model_raw_depth.astype(np.float32),
        "target_depth_refined_candidate_z": refined_candidate_depth.astype(np.float32),
        "target_depth_unsharpened_z": strict_source_depth.astype(np.float32),
        "target_depth_local_sharpened_z": local_sharpened_depth.astype(np.float32),
        "target_depth_target_only_z": target_only_depth.astype(np.float32),
        "target_depth_strict_mask": final_strict.astype(np.uint8),
        "target_depth_complete_mask": final_complete.astype(np.uint8),
        "target_depth_filled_only_mask": filled.astype(np.uint8),
        "target_depth_edge_crack_filled_mask": edge_filled.astype(np.uint8),
        "target_depth_edge_sharpened_mask": edge_sharpened.astype(np.uint8),
        "target_depth_snap_band": target_snap_band.astype(np.uint8),
        "target_depth_target_edge_snapped_mask": target_edge_snapped.astype(np.uint8),
        "target_depth_segmentation_object_mask": segmentation_object_mask.astype(np.uint8),
        "target_depth_segmentation_changed_mask": segmentation_changed.astype(np.uint8),
        "target_depth_segmentation_thin_edges": segmentation_thin_depth_edges.astype(np.uint8),
        "target_depth_uncertain_boundary": uncertain_boundary.astype(np.uint8),
        "target_depth_geometry_agreement_mask": geometry_agreement.astype(np.uint8),
        "target_depth_geometry_conflict_mask": geometry_conflict.astype(np.uint8),
        "target_depth_model_conflict_mask": model_conflict.astype(np.uint8),
        "target_depth_reference_surface_override_mask": reference_surface_override.astype(np.uint8),
        "target_depth_selected_camera": selected_camera.astype(np.uint8),
        "target_depth_confidence": final_confidence.astype(np.float32),
        "target_depth_model_raw_confidence": model_raw_confidence.astype(np.float32),
        "target_depth_refined_candidate_confidence": refined_candidate_confidence.astype(np.float32),
        "target_depth_support_count": support.astype(np.uint8),
    }
    rendering_report: dict[str, Any] = {}
    for name in REFERENCE_NAMES:
        rendered = render_reference(
            references[name], reference_models[name], target_model, target_rays,
            render_depth, render_complete, render_surface_guide_depth,
            args.render_interpolation,
            args.render_edge_depth_step,
            args.render_edge_nearest_radius,
            args.occlusion_tolerance,
            args.render_occlusion_fill_radius,
            args.render_surface_depth_tolerance,
            args.render_unresolved_relaxed_depth_tolerance,
            args.render_unresolved_max_component_area,
            args.render_crack_radius, args.fill_relative_spread,
            args.edge_crack_policy,
            args.render_fill_texture_method,
            args.render_inpaint_radius,
        )
        strict_rendered = render_reference(
            references[name], reference_models[name], target_model, target_rays,
            strict_source_depth, final_complete, strict_source_depth,
            args.render_interpolation,
            args.render_edge_depth_step,
            args.render_edge_nearest_radius,
            args.occlusion_tolerance,
            0,
            args.render_surface_depth_tolerance,
            args.render_unresolved_relaxed_depth_tolerance,
            0,
            0, args.fill_relative_spread, args.edge_crack_policy,
            "copy",
            0.0,
        )
        strict_visible = strict_rendered.zbuffer_visible & final_strict
        aligned_strict = strict_rendered.aligned_zbuffer.copy()
        aligned_strict[~strict_visible] = 0
        occlusion_ambiguous = rendered.sampleable & ~rendered.zbuffer_visible
        occlusion_unresolved = (
            occlusion_ambiguous
            & ~rendered.occlusion_filled
            & ~rendered.occlusion_relaxed_filled
        )
        overlay = alpha_overlay(
            rendered.aligned_complete, target, rendered.visual_mask, args.overlay_alpha
        )
        edges = edge_overlay(rendered.aligned_complete, target, rendered.visual_mask)
        write_image(output_dir / f"{name}_aligned.jpg", rendered.aligned_complete)
        write_image(output_dir / f"{name}_aligned.png", rendered.aligned_complete)
        write_image(
            output_dir / f"{name}_aligned_surface_copy.png",
            rendered.aligned_surface_copy,
        )
        write_image(
            output_dir / f"{name}_aligned_raw_sampleable.jpg",
            rendered.aligned_raw_sampleable,
        )
        write_image(output_dir / f"{name}_aligned_strict.jpg", aligned_strict)
        write_image(output_dir / f"{name}_valid_mask.png", rendered.sampleable.astype(np.uint8) * 255)
        write_image(output_dir / f"{name}_strict_mask.png", strict_visible.astype(np.uint8) * 255)
        write_image(
            output_dir / f"{name}_zbuffer_visible_mask.png",
            rendered.zbuffer_visible.astype(np.uint8) * 255,
        )
        write_image(
            output_dir / f"{name}_occlusion_ambiguous_mask.png",
            occlusion_ambiguous.astype(np.uint8) * 255,
        )
        write_image(
            output_dir / f"{name}_occlusion_surface_filled_mask.png",
            rendered.occlusion_filled.astype(np.uint8) * 255,
        )
        write_image(
            output_dir / f"{name}_occlusion_relaxed_eligible_mask.png",
            rendered.occlusion_relaxed_eligible.astype(np.uint8) * 255,
        )
        write_image(
            output_dir / f"{name}_occlusion_relaxed_filled_mask.png",
            rendered.occlusion_relaxed_filled.astype(np.uint8) * 255,
        )
        write_image(
            output_dir / f"{name}_occlusion_unresolved_mask.png",
            occlusion_unresolved.astype(np.uint8) * 255,
        )
        # Compatibility name retained, but now it means genuinely recovered by
        # same-target-depth propagation rather than raw z-buffer rejection.
        write_image(
            output_dir / f"{name}_occlusion_recovered_mask.png",
            (rendered.occlusion_filled | rendered.occlusion_relaxed_filled).astype(np.uint8) * 255,
        )
        write_image(
            output_dir / f"{name}_display_filled_mask.png",
            rendered.display_filled.astype(np.uint8) * 255,
        )
        write_image(
            output_dir / f"{name}_texture_structure_refined_mask.png",
            rendered.texture_structure_refined.astype(np.uint8) * 255,
        )
        write_image(
            output_dir / f"{name}_texture_inpaint_solver_mask.png",
            rendered.texture_inpaint_solver_mask.astype(np.uint8) * 255,
        )
        write_image(
            output_dir / f"{name}_edge_nearest_mask.png",
            rendered.edge_nearest_mask.astype(np.uint8) * 255,
        )
        write_image(output_dir / f"{name}_overlay_50.jpg", overlay)
        write_image(output_dir / f"{name}_edge_overlay.png", edges)
        maps[f"map_target_to_{name}_raw_xy"] = rendered.map_xy.astype(np.float32)
        maps[f"map_target_to_{name}_strict_raw_xy"] = strict_rendered.map_xy.astype(np.float32)
        maps[f"{name}_sampleable_mask"] = rendered.sampleable.astype(np.uint8)
        maps[f"{name}_zbuffer_visible_mask"] = rendered.zbuffer_visible.astype(np.uint8)
        maps[f"{name}_occlusion_ambiguous_mask"] = occlusion_ambiguous.astype(np.uint8)
        maps[f"{name}_occlusion_surface_filled_mask"] = rendered.occlusion_filled.astype(np.uint8)
        maps[f"{name}_occlusion_relaxed_eligible_mask"] = rendered.occlusion_relaxed_eligible.astype(np.uint8)
        maps[f"{name}_occlusion_relaxed_filled_mask"] = rendered.occlusion_relaxed_filled.astype(np.uint8)
        maps[f"{name}_occlusion_unresolved_mask"] = occlusion_unresolved.astype(np.uint8)
        maps[f"{name}_visible_mask"] = strict_visible.astype(np.uint8)
        maps[f"{name}_display_filled_mask"] = rendered.display_filled.astype(np.uint8)
        maps[f"{name}_texture_structure_refined_mask"] = (
            rendered.texture_structure_refined.astype(np.uint8)
        )
        maps[f"{name}_texture_inpaint_solver_mask"] = (
            rendered.texture_inpaint_solver_mask.astype(np.uint8)
        )
        maps[f"{name}_edge_nearest_mask"] = rendered.edge_nearest_mask.astype(np.uint8)
        rendering_report[name] = {
            "sampleable_ratio": float(
                np.count_nonzero(rendered.sampleable) / rendered.sampleable.size
            ),
            "zbuffer_visible_ratio": float(
                np.count_nonzero(rendered.zbuffer_visible) / rendered.zbuffer_visible.size
            ),
            "strict_visible_ratio": float(
                np.count_nonzero(strict_visible) / strict_visible.size
            ),
            "occlusion_ambiguous_ratio": float(
                np.count_nonzero(occlusion_ambiguous) / occlusion_ambiguous.size
            ),
            "occlusion_surface_filled_ratio": float(
                np.count_nonzero(rendered.occlusion_filled) / rendered.occlusion_filled.size
            ),
            "occlusion_relaxed_eligible_ratio": float(
                np.count_nonzero(rendered.occlusion_relaxed_eligible)
                / rendered.occlusion_relaxed_eligible.size
            ),
            "occlusion_relaxed_filled_ratio": float(
                np.count_nonzero(rendered.occlusion_relaxed_filled)
                / rendered.occlusion_relaxed_filled.size
            ),
            "occlusion_unresolved_ratio": float(
                np.count_nonzero(occlusion_unresolved) / occlusion_unresolved.size
            ),
            "display_filled_ratio": float(
                np.count_nonzero(rendered.display_filled) / rendered.display_filled.size
            ),
            "texture_structure_refined_ratio": float(
                np.count_nonzero(rendered.texture_structure_refined)
                / rendered.texture_structure_refined.size
            ),
            "texture_inpaint_solver_ratio": float(
                np.count_nonzero(rendered.texture_inpaint_solver_mask)
                / rendered.texture_inpaint_solver_mask.size
            ),
            "edge_nearest_ratio": float(
                np.count_nonzero(rendered.edge_nearest_mask) / rendered.edge_nearest_mask.size
            ),
        }
        print(
            f"[{name}] 完整输出覆盖="
            f"{100*np.count_nonzero(rendered.visual_mask)/rendered.visual_mask.size:.2f}%；"
            f"严格可见={100*np.count_nonzero(strict_visible)/strict_visible.size:.2f}%；"
            f"遮挡带同层补全={100*np.count_nonzero(rendered.occlusion_filled)/rendered.occlusion_filled.size:.2f}%；"
            f"小组件放宽补全={100*np.count_nonzero(rendered.occlusion_relaxed_filled)/rendered.occlusion_relaxed_filled.size:.2f}%；"
            f"纹理方向修复={100*np.count_nonzero(rendered.texture_structure_refined)/rendered.texture_structure_refined.size:.2f}%；"
            f"锐利边界采样={100*np.count_nonzero(rendered.edge_nearest_mask)/rendered.edge_nearest_mask.size:.2f}%；"
            f"仍未解决={100*np.count_nonzero(occlusion_unresolved)/occlusion_unresolved.size:.2f}%"
        )

    np.savez_compressed(output_dir / "depth_prior_alignment_maps.npz", **maps)
    report = {
        "schema": "multialign_model_first_calibrated_depth_v14_structure_continuation_inpaint",
        "method": "Depth Anything V2 surfaces -> calibrated scale/shift -> hard reference-surface lock -> bounded render-only depth-hole completion -> median-cleaned surface guide -> edge-aware texture remap -> z-buffer rejection -> strict same-surface fill -> relaxed fill for small non-border unresolved components -> geometry-authorized structure-continuation texture restoration -> separate strict render",
        "important": (
            "The learned model defines local depth variation and object boundaries. Calibrated triangulation only "
            "fits model scale/shift, selects among conflicting learned surfaces, and changes confidence; it never "
            "overwrites final model depth values. In model-raw mode the reference model is reasserted at every valid "
            "reference pixel, so another reference, geometry, target snapping, or SAM cannot alternate a thin object "
            "between near and far surfaces; other references fill only missing reference pixels. The old raw backward "
            "warp is saved as *_aligned_raw_sampleable.jpg "
            "for diagnosis only because it can widen a pole or duplicate a coloured sign at occlusions. The default "
            "texture remap uses bilinear interpolation inside surfaces but nearest-neighbour sampling in a narrow "
            "target-depth edge band, preventing foreground/background colour averaging. *_aligned.jpg then rejects "
            "z-buffer conflicts and fills only from neighbours on the same target-depth "
            "surface. The edge-nearest mask is an interpolation diagnostic, not a hole-fill mask. Small holes and "
            "fragmented depth lines are repaired in a separate render-only surface guide and never written back to "
            "the metric or strict depth. Remaining small non-border occlusion components receive a second pass with "
            "a relaxed depth tolerance; large or border-touching disocclusions remain unresolved rather than being "
            "fabricated. The default Navier-Stokes texture stage then continues isophote/edge direction only inside "
            "pixels already authorized by one of the geometric fill masks; original valid pixels and unresolved "
            "regions are copied back unchanged. *_aligned_surface_copy.png preserves the pre-inpaint result for A/B "
            "diagnosis, while *_texture_structure_refined_mask.png shows the exact committed region. Prompted masks "
            "may represent either a near object or a far opening; "
            "rejected proposals fall back safely."
        ),
        "inputs": {
            "calibration": str(args.calibration.resolve()),
            "geometry_npz": str(args.geometry_npz.resolve()),
            "reference_images": {name: str(path) for name, path in reference_paths.items()},
            "target_image": str(target_path),
            "depth_anything_root": str(args.depth_anything_root.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "segmentation_checkpoint": (
                str(args.segmentation_checkpoint.resolve())
                if args.segmentation_checkpoint is not None else None
            ),
        },
        "camera_roles": {
            "reference_cameras": list(REFERENCE_NAMES),
            "target_camera": args.target_camera,
            "anchor_camera": args.anchor_camera,
        },
        "target": {"camera": args.target_camera, "size": list(target_size)},
        "geometry": geometry_report,
        "camera_weights": camera_weights,
        "camera_priors": {candidate.camera: candidate.report for candidate in candidates},
        "camera_failures": failures,
        "fusion": fusion_report,
        "target_edge_snap": target_snap_report,
        "object_segmentation": segmentation_report,
        "final": {
            "depth_source": args.final_depth_source,
            "reference_camera": args.reference_depth_camera,
            "reference_surface_override_ratio": float(
                np.count_nonzero(reference_surface_override) / reference_surface_override.size
            ),
            "strict_ratio": float(np.count_nonzero(final_strict) / final_strict.size),
            "complete_ratio": float(np.count_nonzero(final_complete) / final_complete.size),
            "render_complete_ratio": float(np.count_nonzero(render_complete) / render_complete.size),
            "render_surface_guide_filled_ratio": float(
                np.count_nonzero(render_surface_guide_filled) / render_surface_guide_filled.size
            ),
            "filled_only_ratio": float(np.count_nonzero(filled) / filled.size),
            "edge_crack_filled_ratio": float(np.count_nonzero(edge_filled) / edge_filled.size),
            "edge_sharpened_ratio": float(np.count_nonzero(edge_sharpened) / edge_sharpened.size),
            "target_edge_snapped_ratio": float(
                np.count_nonzero(target_edge_snapped) / target_edge_snapped.size
            ),
            "segmentation_object_ratio": float(
                np.count_nonzero(segmentation_object_mask) / segmentation_object_mask.size
            ),
            "segmentation_changed_ratio": float(
                np.count_nonzero(segmentation_changed) / segmentation_changed.size
            ),
            "model_conflict_ratio": float(np.count_nonzero(model_conflict) / model_conflict.size),
            "geometry_agreement_ratio": float(np.count_nonzero(geometry_agreement) / geometry_agreement.size),
            "geometry_conflict_ratio": float(np.count_nonzero(geometry_conflict) / geometry_conflict.size),
            "depth_quantiles": quantiles(final_depth[final_complete]),
            "model_raw_depth_quantiles": quantiles(model_raw_depth[model_raw_complete]),
            "refined_candidate_depth_quantiles": quantiles(
                refined_candidate_depth[final_complete]
            ),
        },
        "rendering": rendering_report,
        "settings": vars(args),
    }
    write_json(output_dir / "depth_prior_report.json", report)
    example_reference = REFERENCE_NAMES[0]
    print(f"\n完成：{output_dir}")
    print(f"最终深度与映射：{output_dir / 'depth_prior_alignment_maps.npz'}")
    print(f"透明叠加：{output_dir / f'{example_reference}_overlay_50.jpg'} 等")
    print(
        "纹理补全A/B："
        f"{output_dir / f'{example_reference}_aligned_surface_copy.png'}（旧式复制）与 "
        f"{output_dir / f'{example_reference}_aligned.png'}（结构延续）"
    )
    print(
        "深度A/B："
        f"{output_dir / 'depth_final_model_raw.png'} 与 "
        f"{output_dir / 'depth_final_refined_candidate.png'}"
    )
    print("完整观看使用*_aligned.jpg；定量分析使用*_aligned_strict.jpg和*_strict_mask.png。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PriorAlignmentError as exc:
        print(f"错误：{exc}")
        raise SystemExit(2)
