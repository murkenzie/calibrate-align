#!/usr/bin/env python3
r"""
Extract correspondence points from a fixed rig containing two or more
reference cameras and one target camera, then diagnose whether one fundamental
matrix per pair is shared by all natural scenes.

This is deliberately a geometry program, not an image-warping program:

* Every reference/reference and reference/target pair uses RoMa v2.
* Reference/reference input remains RGB; reference/target input uses the selected
  cross-modal representation.  Only sparse sampled correspondences are saved.
* Every pair is matched on the complete source images, so foreground,
  background and intermediate depths can all contribute constraints.
* A robust F is fitted per capture to reject matcher outliers.
* A single fixed-rig F is fitted across captures and evaluated on held-out
  captures.
* A per-scene homography-dominance diagnostic warns when apparently abundant
  matches still come mostly from one plane (or from very low parallax).
* Images are never resized on disk and no dense/free-form warp is saved.

Expected input is a compatible prepared dataset. The unified ``multialign
calibrate`` command creates this adapter automatically:

    prepared_root/
      images/{reference_a,reference_b,...,target}/frame_XXX.jpg
      target_gray/frame_XXX.jpg               # optional
      target_edges/frame_XXX.png              # optional
      metadata/dataset.json
      splits/{all,custom}.txt

The matching scope is always ``full_image``.

The program is resumable. Existing match NPZ files are reused unless
--overwrite-matches is supplied. Use --diagnose-only to recompute F reports
from existing NPZ files without loading the neural matcher.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from itertools import combinations
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


REFERENCE_CAMERAS = ("reference_a", "reference_b")
TARGET_CAMERA = "target"
ALL_CAMERAS = (*REFERENCE_CAMERAS, TARGET_CAMERA)
DEFAULT_PAIRS = tuple(f"{a}-{b}" for a, b in combinations(ALL_CAMERAS, 2))
PROGRAM_VERSION = "6.0-generic-fullframe"


class RigMatchError(RuntimeError):
    pass


def configure_rig(reference_cameras: Sequence[str], target_camera: str) -> None:
    global REFERENCE_CAMERAS, TARGET_CAMERA, ALL_CAMERAS, DEFAULT_PAIRS
    references = tuple(reference_cameras)
    if len(references) < 2 or len(set(references)) != len(references):
        raise RigMatchError("--reference-cameras requires at least two unique names")
    if target_camera in references:
        raise RigMatchError("--target-camera must not be a reference camera")
    REFERENCE_CAMERAS = references
    TARGET_CAMERA = target_camera
    ALL_CAMERAS = (*references, target_camera)
    DEFAULT_PAIRS = tuple(f"{a}-{b}" for a, b in combinations(ALL_CAMERAS, 2))


@dataclass(frozen=True)
class PairSpec:
    camera0: str
    camera1: str

    @property
    def name(self) -> str:
        return f"{self.camera0}__{self.camera1}"

    @property
    def label(self) -> str:
        return f"{self.camera0}-{self.camera1}"

    @property
    def is_cross_modal(self) -> bool:
        return TARGET_CAMERA in (self.camera0, self.camera1)


@dataclass
class FrameRecord:
    frame: str
    scene_id: str
    variant: str
    group: dict[str, Any]
    paths: dict[str, Path]


@dataclass
class MatchRecord:
    points0: np.ndarray
    points1: np.ndarray
    scores: np.ndarray
    size0_wh: tuple[int, int]
    size1_wh: tuple[int, int]
    metadata: dict[str, Any]


@dataclass
class GroupGeometry:
    frame: FrameRecord
    pair: PairSpec
    match_path: Path
    matches: MatchRecord
    F_common: np.ndarray | None
    group_inlier_mask: np.ndarray
    group_error: np.ndarray
    metrics: dict[str, Any]


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def imread_checked(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    if not path.is_file():
        raise RigMatchError(f"图像不存在：{path}")
    # imdecode is reliable for non-ASCII Windows paths.
    try:
        payload = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(payload, flags)
    except (OSError, ValueError) as exc:
        raise RigMatchError(f"无法读取图像：{path} ({exc})") from exc
    if image is None:
        raise RigMatchError(f"OpenCV无法解码图像：{path}")
    return image


def imwrite_checked(path: Path, image: np.ndarray, params: Sequence[int] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, payload = cv2.imencode(suffix, image, list(params))
    if not ok:
        raise RigMatchError(f"OpenCV无法编码图像：{path}")
    try:
        payload.tofile(str(path))
    except OSError as exc:
        raise RigMatchError(f"无法写入图像：{path} ({exc})") from exc


def parse_pair(text: str) -> PairSpec:
    normalized = text.strip().replace("__", "-")
    parts = normalized.split("-")
    if len(parts) != 2 or any(part not in ALL_CAMERAS for part in parts):
        raise argparse.ArgumentTypeError(
            f"invalid camera pair {text!r}; expected camera_a-camera_b"
        )
    if parts[0] == parts[1]:
        raise argparse.ArgumentTypeError("相机对的两个相机不能相同")
    return PairSpec(parts[0], parts[1])


def resolve_dataset_json(dataset_arg: Path) -> tuple[Path, Path]:
    path = dataset_arg.expanduser().resolve()
    if path.is_file():
        if path.name.casefold() != "dataset.json":
            raise RigMatchError(f"--dataset 文件应为 dataset.json：{path}")
        if path.parent.name.casefold() == "metadata":
            root = path.parent.parent
        else:
            root = path.parent
        return root, path
    candidate = path / "metadata" / "dataset.json"
    if candidate.is_file():
        return path, candidate
    raise RigMatchError(
        f"找不到数据清单：{candidate}\n"
        "--dataset 应指向包含 metadata/dataset.json 的已准备数据目录"
    )


def read_split(root: Path, split_arg: str) -> set[str] | None:
    if split_arg.casefold() in {"all", "*"}:
        candidate = root / "splits" / "all.txt"
        if not candidate.is_file():
            return None
    else:
        raw = Path(split_arg)
        candidate = raw if raw.is_file() else root / "splits" / f"{split_arg}.txt"
    if not candidate.is_file():
        raise RigMatchError(f"找不到split文件：{candidate}")
    values = {
        line.strip()
        for line in candidate.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not values:
        raise RigMatchError(f"split文件为空：{candidate}")
    return values


def first_existing(candidates: Iterable[Path | None]) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    return None


def recorded_path(value: Any, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = root / path
        if candidate.is_file():
            return candidate
    return None


def resolve_frame_paths(
    root: Path,
    group: dict[str, Any],
    target_source: str,
) -> dict[str, Path]:
    frame = str(group["frame"])
    paths: dict[str, Path] = {}
    references = group.get("references", {})
    for camera in REFERENCE_CAMERAS:
        record = references.get(camera, {})
        paths[camera] = first_existing(
            (
                root / "images" / camera / f"{frame}.jpg",
                root / "images" / camera / f"{frame}.jpeg",
                root / "images" / camera / f"{frame}.png",
                recorded_path(record.get("prepared_jpg"), root),
                recorded_path(record.get("source_jpg"), root),
            )
        ) or Path("__missing__")

    target = group.get("target", {})
    if target_source == "composite":
        candidates = (
            root / "images" / TARGET_CAMERA / f"{frame}.jpg",
            root / "images" / TARGET_CAMERA / f"{frame}.png",
            recorded_path(target.get("prepared_composite"), root),
        )
    elif target_source == "gray":
        candidates = (
            root / "target_gray" / f"{frame}.jpg",
            root / "target_gray" / f"{frame}.png",
            recorded_path(target.get("prepared_gray"), root),
        )
    else:
        candidates = (
            root / "target_edges" / f"{frame}.png",
            recorded_path(target.get("prepared_edges"), root),
        )
    paths[TARGET_CAMERA] = first_existing(candidates) or Path("__missing__")
    return paths


def load_dataset(
    dataset_arg: Path,
    split_arg: str,
    target_source: str,
    limit: int | None,
) -> tuple[Path, Path, dict[str, Any], list[FrameRecord]]:
    root, dataset_path = resolve_dataset_json(dataset_arg)
    try:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RigMatchError(f"无法解析 {dataset_path}：{exc}") from exc
    roles = dataset.get("camera_roles", {})
    if tuple(roles.get("reference_cameras", ())) != REFERENCE_CAMERAS:
        raise RigMatchError("dataset reference camera roles do not match CLI roles")
    if roles.get("target_camera") != TARGET_CAMERA:
        raise RigMatchError("dataset target camera role does not match --target-camera")
    selected = read_split(root, split_arg)
    groups = list(dataset.get("groups", []))
    records: list[FrameRecord] = []
    for group in groups:
        frame = str(group.get("frame", ""))
        if not frame or (selected is not None and frame not in selected):
            continue
        scene_id = str(group.get("scene_id", ""))
        variant = str(group.get("variant", ""))
        records.append(
            FrameRecord(
                frame=frame,
                scene_id=scene_id,
                variant=variant,
                group=group,
                paths=resolve_frame_paths(root, group, target_source),
            )
        )
    if limit is not None:
        records = records[:limit]
    if not records:
        raise RigMatchError(f"split={split_arg!r} 没有可用采集组")
    return root, dataset_path, dataset, records


def image_size_wh(path: Path) -> tuple[int, int]:
    image = imread_checked(path, cv2.IMREAD_GRAYSCALE)
    return int(image.shape[1]), int(image.shape[0])


def gradient_uint8(gray: np.ndarray) -> np.ndarray:
    values = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    scale = float(np.percentile(magnitude, 99.0)) if magnitude.size else 0.0
    if not math.isfinite(scale) or scale <= 1e-8:
        return np.zeros_like(gray)
    return np.clip(np.round(magnitude * (255.0 / scale)), 0, 255).astype(np.uint8)


def roma_representation(
    path: Path,
    is_target: bool,
    mode: str,
) -> np.ndarray:
    color = imread_checked(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    if mode == "rgb-gray":
        if is_target:
            return np.repeat(gray[..., None], 3, axis=2)
        return np.ascontiguousarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    if mode == "gray":
        return np.repeat(gray[..., None], 3, axis=2)
    if mode == "structure":
        grad = gradient_uint8(gray)
        edges = cv2.Canny(gray, 40, 120)
        return np.ascontiguousarray(np.stack((gray, grad, edges), axis=2))
    raise RigMatchError(f"未知RoMa输入表示：{mode}")


def flatten_sample_score(value: Any, count: int) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return None
    if array.shape[0] == count:
        return array.reshape(count, -1).mean(axis=1)
    if array.size == count:
        return array.reshape(count)
    return None


class RoMaSparseMatcher:
    def __init__(
        self,
        setting: str,
        samples: int,
        allow_cpu: bool,
        torch_compile: bool,
    ) -> None:
        try:
            import torch
            from romav2 import RoMaV2
            from romav2.device import device
        except Exception as exc:
            raise RigMatchError(
                "无法导入官方RoMa v2。不要安装同名旧PyPI包；请安装 "
                "Parskatt/RoMaV2 源码。\n"
                f"原始错误：{exc}"
            ) from exc
        if device.type == "cpu" and not allow_cpu:
            raise RigMatchError(
                "RoMa v2未检测到CUDA；CPU很慢。确认后添加 --allow-cpu"
            )
        self.torch = torch
        self.device = device
        self.samples = samples
        self.setting = setting
        torch.set_float32_matmul_precision("highest")
        try:
            self.model = RoMaV2(RoMaV2.Cfg(compile=torch_compile))
            self.model.apply_setting(setting)
        except Exception as exc:
            raise RigMatchError(f"RoMa v2初始化失败：{exc}") from exc
        device_label = str(device)
        if device.type == "cuda":
            device_label = f"cuda: {torch.cuda.get_device_name(device)}"
        print(
            f"RoMa v2设备：{device_label}；setting={setting}；samples={samples}；"
            f"torch.compile={'开' if torch_compile else '关'}"
        )

    def match(
        self,
        image0_rgb: np.ndarray,
        image1_rgb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        h0, w0 = image0_rgb.shape[:2]
        h1, w1 = image1_rgb.shape[:2]
        try:
            with self.torch.inference_mode():
                predictions = self.model.match(
                    np.ascontiguousarray(image0_rgb),
                    np.ascontiguousarray(image1_rgb),
                )
                matches, overlaps, precision_ab, precision_ba = self.model.sample(
                    predictions, self.samples
                )
                points0, points1 = self.model.to_pixel_coordinates(
                    matches, h0, w0, h1, w1
                )
            points0_np = points0.detach().float().cpu().numpy().reshape(-1, 2)
            points1_np = points1.detach().float().cpu().numpy().reshape(-1, 2)
            count = len(points0_np)
            score_parts = [
                part
                for part in (
                    flatten_sample_score(overlaps, count),
                    flatten_sample_score(precision_ab, count),
                    flatten_sample_score(precision_ba, count),
                )
                if part is not None
            ]
            if score_parts:
                scores = np.mean(np.stack(score_parts, axis=0), axis=0)
            else:
                scores = np.ones(count, dtype=np.float32)
            metadata = {
                "matcher": "RoMaV2",
                "setting": self.setting,
                "samples_requested": self.samples,
                "device": str(self.device),
            }
            del predictions, matches, overlaps, precision_ab, precision_ba
            return points0_np, points1_np, scores.astype(np.float32), metadata
        except RuntimeError as exc:
            message = str(exc)
            if "out of memory" in message.casefold():
                raise RigMatchError(
                    "RoMa v2显存不足；使用 --roma-setting fast，并关闭其他占显存程序"
                ) from exc
            if "triton" in message.casefold():
                raise RigMatchError(
                    "Windows下不要开启 --torch-compile；CUDA eager推理不需要Triton"
                ) from exc
            raise RigMatchError(f"RoMa v2推理失败：{exc}") from exc

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def valid_point_mask(
    points0: np.ndarray,
    points1: np.ndarray,
    size0_wh: tuple[int, int],
    size1_wh: tuple[int, int],
) -> np.ndarray:
    p0 = np.asarray(points0, dtype=np.float64)
    p1 = np.asarray(points1, dtype=np.float64)
    w0, h0 = size0_wh
    w1, h1 = size1_wh
    return (
        np.isfinite(p0).all(axis=1)
        & np.isfinite(p1).all(axis=1)
        & (p0[:, 0] >= -0.5)
        & (p0[:, 0] <= w0 - 0.5)
        & (p0[:, 1] >= -0.5)
        & (p0[:, 1] <= h0 - 0.5)
        & (p1[:, 0] >= -0.5)
        & (p1[:, 0] <= w1 - 0.5)
        & (p1[:, 1] >= -0.5)
        & (p1[:, 1] <= h1 - 0.5)
    )


def balanced_subsample(
    points0: np.ndarray,
    points1: np.ndarray,
    scores: np.ndarray,
    size0_wh: tuple[int, int],
    size1_wh: tuple[int, int],
    max_count: int,
    grid_cols: int,
    grid_rows: int,
    max_per_cell: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    points0 = np.asarray(points0, dtype=np.float32).reshape(-1, 2)
    points1 = np.asarray(points1, dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not (len(points0) == len(points1) == len(scores)):
        raise RigMatchError("匹配点和分数长度不一致")
    raw_count = len(points0)
    mask = valid_point_mask(points0, points1, size0_wh, size1_wh)
    indices = np.flatnonzero(mask)
    if not len(indices):
        return points0[:0], points1[:0], scores[:0], {
            "raw_match_count": raw_count,
            "valid_match_count": 0,
            "selected_match_count": 0,
        }
    safe_scores = np.nan_to_num(scores[indices], nan=-np.inf)
    order = indices[np.argsort(-safe_scores, kind="stable")]
    counts0 = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    counts1 = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    selected: list[int] = []
    selected_set: set[int] = set()

    def cell(point: np.ndarray, size_wh: tuple[int, int]) -> tuple[int, int]:
        width, height = size_wh
        col = min(grid_cols - 1, max(0, int(point[0] * grid_cols / width)))
        row = min(grid_rows - 1, max(0, int(point[1] * grid_rows / height)))
        return row, col

    for index in order:
        cell0 = cell(points0[index], size0_wh)
        cell1 = cell(points1[index], size1_wh)
        if counts0[cell0] >= max_per_cell or counts1[cell1] >= max_per_cell:
            continue
        selected.append(int(index))
        selected_set.add(int(index))
        counts0[cell0] += 1
        counts1[cell1] += 1
        if len(selected) >= max_count:
            break
    # Sparse-overlap/telephoto pairs can occupy only a few cells. Keep strong
    # residual matches as a fallback instead of starving robust estimation.
    minimum_target = min(max_count, min(len(order), 100))
    if len(selected) < minimum_target:
        for index in order:
            if int(index) in selected_set:
                continue
            selected.append(int(index))
            selected_set.add(int(index))
            if len(selected) >= minimum_target:
                break
    chosen = np.asarray(selected[:max_count], dtype=np.int64)
    return points0[chosen], points1[chosen], scores[chosen], {
        "raw_match_count": raw_count,
        "valid_match_count": int(len(indices)),
        "selected_match_count": int(len(chosen)),
        "occupied_cells0": int(np.count_nonzero(counts0)),
        "occupied_cells1": int(np.count_nonzero(counts1)),
    }


def save_match(path: Path, record: MatchRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points0=np.asarray(record.points0, dtype=np.float32),
        points1=np.asarray(record.points1, dtype=np.float32),
        scores=np.asarray(record.scores, dtype=np.float32),
        size0_wh=np.asarray(record.size0_wh, dtype=np.int32),
        size1_wh=np.asarray(record.size1_wh, dtype=np.int32),
        metadata_json=np.asarray(
            json.dumps(json_safe(record.metadata), ensure_ascii=False)
        ),
    )


def load_match(path: Path) -> MatchRecord:
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata: dict[str, Any] = {}
            if "metadata_json" in archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
            return MatchRecord(
                points0=np.asarray(archive["points0"], dtype=np.float32).reshape(-1, 2),
                points1=np.asarray(archive["points1"], dtype=np.float32).reshape(-1, 2),
                scores=np.asarray(archive["scores"], dtype=np.float32).reshape(-1),
                size0_wh=tuple(int(x) for x in archive["size0_wh"].reshape(2)),
                size1_wh=tuple(int(x) for x in archive["size1_wh"].reshape(2)),
                metadata=metadata,
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RigMatchError(f"无法读取匹配缓存 {path}：{exc}") from exc


def expected_match_metadata(
    frame: FrameRecord,
    pair: PairSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "frame": frame.frame,
        "scene_id": frame.scene_id,
        "variant": frame.variant,
        "camera0": pair.camera0,
        "camera1": pair.camera1,
        "image0": str(frame.paths[pair.camera0]),
        "image1": str(frame.paths[pair.camera1]),
        "target_source": args.target_source if pair.is_cross_modal else None,
        "cross_representation": args.cross_representation if pair.is_cross_modal else None,
        "matcher_backend": "roma_v2_all_pairs",
        "reference_representation": (
            "rgb" if not pair.is_cross_modal else args.cross_representation
        ),
        "matching_scope": "full_image",
    }


def cache_compatible(record: MatchRecord, expected: dict[str, Any]) -> bool:
    keys = (
        "frame",
        "scene_id",
        "variant",
        "camera0",
        "camera1",
        "target_source",
        "cross_representation",
        "matcher_backend",
        "reference_representation",
        "matching_scope",
    )
    return all(record.metadata.get(key) == expected.get(key) for key in keys)


def match_cache_path(output_root: Path, pair: PairSpec, frame: str) -> Path:
    return output_root / "matches" / pair.name / f"{frame}.npz"


def run_roma_matching(
    frames: list[FrameRecord],
    pairs: list[PairSpec],
    output_root: Path,
    args: argparse.Namespace,
    failures: list[dict[str, Any]],
) -> None:
    pending: list[tuple[FrameRecord, PairSpec]] = []
    for frame in frames:
        for pair in pairs:
            path = match_cache_path(output_root, pair, frame.frame)
            if args.overwrite_matches or not path.is_file():
                pending.append((frame, pair))
                continue
            existing = load_match(path)
            expected = expected_match_metadata(frame, pair, args)
            if not cache_compatible(existing, expected):
                raise RigMatchError(
                    f"缓存配置与当前数据不一致：{path}\n"
                    "请使用新输出目录或添加 --overwrite-matches"
                )
    if not pending:
        print("所有相机对的RoMa v2缓存齐全，跳过模型加载")
        return
    matcher = RoMaSparseMatcher(
        args.roma_setting,
        args.roma_samples,
        args.allow_cpu,
        args.torch_compile,
    )
    try:
        for frame_index, frame in enumerate(frames, 1):
            print(f"[RoMa v2 {frame_index}/{len(frames)}] {frame.frame}")
            representations: dict[tuple[str, str], np.ndarray] = {}
            for pair in pairs:
                path = match_cache_path(output_root, pair, frame.frame)
                expected = expected_match_metadata(frame, pair, args)
                if path.is_file() and not args.overwrite_matches:
                    existing = load_match(path)
                    if not cache_compatible(existing, expected):
                        raise RigMatchError(
                            f"缓存配置与当前命令不一致：{path}\n"
                            "请使用新输出目录或添加 --overwrite-matches"
                        )
                    continue
                try:
                    image0 = frame.paths[pair.camera0]
                    image1 = frame.paths[pair.camera1]
                    if not image0.is_file() or not image1.is_file():
                        raise RigMatchError(f"缺少图像：{image0} 或 {image1}")
                    representation_mode = (
                        args.cross_representation if pair.is_cross_modal else "rgb-gray"
                    )
                    for camera, image_path in (
                        (pair.camera0, image0),
                        (pair.camera1, image1),
                    ):
                        key = (camera, representation_mode)
                        if key not in representations:
                            representations[key] = roma_representation(
                                image_path,
                                camera == TARGET_CAMERA,
                                representation_mode,
                            )
                    key0 = (pair.camera0, representation_mode)
                    key1 = (pair.camera1, representation_mode)
                    points0, points1, scores, model_meta = matcher.match(
                        representations[key0], representations[key1]
                    )
                    size0 = image_size_wh(image0)
                    size1 = image_size_wh(image1)
                    points0, points1, scores, select_meta = balanced_subsample(
                        points0,
                        points1,
                        scores,
                        size0,
                        size1,
                        args.max_matches_per_group,
                        args.grid_cols,
                        args.grid_rows,
                        args.grid_max_per_cell,
                    )
                    metadata = {**expected, **model_meta, **select_meta}
                    save_match(
                        path,
                        MatchRecord(points0, points1, scores, size0, size1, metadata),
                    )
                    print(f"  {pair.label}: {len(points0)} 点 -> {path.name}")
                except Exception as exc:
                    failure = {
                        "stage": "RoMaV2",
                        "frame": frame.frame,
                        "pair": pair.label,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    print(f"  {pair.label}: 失败：{failure['error']}", file=sys.stderr)
                    if args.strict:
                        raise
            del representations
            gc.collect()
            if matcher.device.type == "cuda":
                matcher.torch.cuda.empty_cache()
    finally:
        matcher.close()


def scale_for_common(size_wh: tuple[int, int], common_diagonal: float) -> float:
    width, height = size_wh
    diagonal = math.hypot(width, height)
    if diagonal <= 0:
        raise RigMatchError(f"无效图像尺寸：{size_wh}")
    return common_diagonal / diagonal


def to_common_points(
    points: np.ndarray,
    size_wh: tuple[int, int],
    common_diagonal: float,
) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) * scale_for_common(
        size_wh, common_diagonal
    )


def normalize_F(F: np.ndarray) -> np.ndarray:
    matrix = np.asarray(F, dtype=np.float64).reshape(3, 3)
    norm = float(np.linalg.norm(matrix))
    if not math.isfinite(norm) or norm <= 1e-15:
        raise RigMatchError("F退化为零矩阵")
    matrix = matrix / norm
    pivot = matrix.flat[int(np.argmax(np.abs(matrix)))]
    if pivot < 0:
        matrix = -matrix
    return matrix


def fit_fundamental(
    points0_common: np.ndarray,
    points1_common: np.ndarray,
    threshold: float,
    confidence: float,
    max_iters: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    count = len(points0_common)
    if count < 8:
        return None, np.zeros(count, dtype=bool)
    try:
        F, mask = cv2.findFundamentalMat(
            np.asarray(points0_common, dtype=np.float64),
            np.asarray(points1_common, dtype=np.float64),
            cv2.USAC_MAGSAC,
            float(threshold),
            float(confidence),
            int(max_iters),
        )
    except TypeError:
        F, mask = cv2.findFundamentalMat(
            np.asarray(points0_common, dtype=np.float64),
            np.asarray(points1_common, dtype=np.float64),
            cv2.USAC_MAGSAC,
            float(threshold),
            float(confidence),
        )
    if F is None or np.asarray(F).size < 9:
        return None, np.zeros(count, dtype=bool)
    F = np.asarray(F, dtype=np.float64).reshape(-1, 3)[:3]
    try:
        F = normalize_F(F)
    except RigMatchError:
        return None, np.zeros(count, dtype=bool)
    if mask is None:
        inliers = np.ones(count, dtype=bool)
    else:
        inliers = np.asarray(mask).reshape(-1).astype(bool)
        if len(inliers) != count:
            inliers = np.zeros(count, dtype=bool)
    return F, inliers


def symmetric_epipolar_error(
    F: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
) -> np.ndarray:
    p0 = np.column_stack((np.asarray(points0, dtype=np.float64), np.ones(len(points0))))
    p1 = np.column_stack((np.asarray(points1, dtype=np.float64), np.ones(len(points1))))
    lines1 = (F @ p0.T).T
    lines0 = (F.T @ p1.T).T
    residual = np.abs(np.sum(p1 * lines1, axis=1))
    denom1 = np.hypot(lines1[:, 0], lines1[:, 1])
    denom0 = np.hypot(lines0[:, 0], lines0[:, 1])
    distance1 = residual / np.maximum(denom1, 1e-12)
    distance0 = residual / np.maximum(denom0, 1e-12)
    return np.sqrt(0.5 * (distance0 * distance0 + distance1 * distance1))


def homography_dominance_metrics(
    points0_common: np.ndarray,
    points1_common: np.ndarray,
    fundamental_inliers: np.ndarray,
    threshold: float,
    confidence: float,
    max_iters: int,
) -> dict[str, Any]:
    """Measure how much of the epipolar-consistent set is explained by one H.

    A high value does not prove that the scene is planar: pure rotation, a
    tiny baseline and overwhelmingly distant structure can produce the same
    symptom.  Conversely, a sizeable non-H population is useful evidence that
    the current scene contributes parallax/depth diversity for focal recovery.
    """
    mask_f = np.asarray(fundamental_inliers, dtype=bool).reshape(-1)
    p0 = np.asarray(points0_common, dtype=np.float64)[mask_f]
    p1 = np.asarray(points1_common, dtype=np.float64)[mask_f]
    total = int(len(p0))
    empty = {
        "homography_inlier_count": 0,
        "fundamental_inlier_count": total,
        "homography_dominance_ratio": None,
        "non_homography_count": total,
        "depth_diversity_hint": "INSUFFICIENT",
    }
    if total < 8:
        return empty
    try:
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        H, mask_h = cv2.findHomography(
            p0,
            p1,
            method=method,
            ransacReprojThreshold=float(threshold),
            maxIters=int(max_iters),
            confidence=float(confidence),
        )
    except (TypeError, cv2.error):
        H, mask_h = cv2.findHomography(
            p0,
            p1,
            cv2.RANSAC,
            float(threshold),
        )
    if H is None or mask_h is None:
        return empty
    inlier_count = int(np.count_nonzero(np.asarray(mask_h).reshape(-1)))
    ratio = inlier_count / float(total)
    if ratio >= 0.85:
        hint = "PLANAR_OR_LOW_PARALLAX"
    elif ratio <= 0.65:
        hint = "DEPTH_RICH"
    else:
        hint = "MIXED"
    return {
        "homography_inlier_count": inlier_count,
        "fundamental_inlier_count": total,
        "homography_dominance_ratio": float(ratio),
        "non_homography_count": total - inlier_count,
        "depth_diversity_hint": hint,
    }


def quantile_metrics(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"p25": None, "p50": None, "p75": None, "p90": None, "p95": None, "rms": None}
    quantiles = np.quantile(array, [0.25, 0.5, 0.75, 0.9, 0.95])
    return {
        "p25": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
    }


def grid_coverage(
    points: np.ndarray,
    size_wh: tuple[int, int],
    mask: np.ndarray,
    cols: int = 12,
    rows: int = 9,
) -> float:
    selected = np.asarray(points, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if not len(selected):
        return 0.0
    width, height = size_wh
    columns = np.clip((selected[:, 0] * cols / width).astype(int), 0, cols - 1)
    row_values = np.clip((selected[:, 1] * rows / height).astype(int), 0, rows - 1)
    occupied = len(set(zip(row_values.tolist(), columns.tolist())))
    return occupied / float(cols * rows)


def build_group_geometry(
    frame: FrameRecord,
    pair: PairSpec,
    match_path: Path,
    common_diagonal: float,
    threshold: float,
    homography_threshold: float,
    confidence: float,
    max_iters: int,
) -> GroupGeometry:
    matches = load_match(match_path)
    if (
        matches.metadata.get("matching_scope") != "full_image"
        or matches.metadata.get("matcher_backend") != "roma_v2_all_pairs"
    ):
        raise RigMatchError(
            f"不是兼容的RoMa全相机对缓存：{match_path}；"
            "请换用新输出目录重新匹配"
        )
    for key, expected in (
        ("frame", frame.frame),
        ("camera0", pair.camera0),
        ("camera1", pair.camera1),
    ):
        actual = matches.metadata.get(key)
        if actual is not None and actual != expected:
            raise RigMatchError(
                f"匹配缓存字段不一致：{match_path} 中 {key}={actual!r}，"
                f"当前应为 {expected!r}"
            )
    q0 = to_common_points(matches.points0, matches.size0_wh, common_diagonal)
    q1 = to_common_points(matches.points1, matches.size1_wh, common_diagonal)
    F, inlier_mask = fit_fundamental(q0, q1, threshold, confidence, max_iters)
    if F is None:
        errors = np.full(len(q0), np.inf, dtype=np.float64)
        inlier_mask[:] = False
    else:
        errors = symmetric_epipolar_error(F, q0, q1)
        # MAGSAC's returned mask is authoritative, but also reject any numerical
        # tail beyond twice the requested threshold.
        inlier_mask &= errors <= (2.0 * threshold)
    inlier_errors = errors[inlier_mask]
    depth_metrics = homography_dominance_metrics(
        q0,
        q1,
        inlier_mask,
        homography_threshold,
        confidence,
        max_iters,
    )
    metrics = {
        "frame": frame.frame,
        "scene_id": frame.scene_id,
        "variant": frame.variant,
        "pair": pair.label,
        "matcher": matches.metadata.get("matcher"),
        "selected_match_count": int(len(matches.points0)),
        "group_inlier_count": int(np.count_nonzero(inlier_mask)),
        "group_inlier_ratio": float(np.mean(inlier_mask)) if len(inlier_mask) else 0.0,
        "group_error_common_px": quantile_metrics(inlier_errors),
        "coverage0": grid_coverage(matches.points0, matches.size0_wh, inlier_mask),
        "coverage1": grid_coverage(matches.points1, matches.size1_wh, inlier_mask),
        **depth_metrics,
        "size0_wh": list(matches.size0_wh),
        "size1_wh": list(matches.size1_wh),
        "F_group_common": F,
    }
    return GroupGeometry(
        frame=frame,
        pair=pair,
        match_path=match_path,
        matches=matches,
        F_common=F,
        group_inlier_mask=inlier_mask,
        group_error=errors,
        metrics=metrics,
    )


def deterministic_roles(
    frames: list[str], validation_fraction: float, seed: int
) -> dict[str, str]:
    unique = sorted(set(frames))
    if validation_fraction <= 0.0 or len(unique) < 5:
        return {frame: "train" for frame in unique}
    validation_count = max(1, int(round(len(unique) * validation_fraction)))
    validation_count = min(validation_count, len(unique) - 3)
    ranked = sorted(
        unique,
        key=lambda frame: hashlib.sha256(f"{seed}:{frame}".encode()).digest(),
    )
    validation = set(ranked[:validation_count])
    return {frame: ("validation" if frame in validation else "train") for frame in unique}


def modal_size(geometries: list[GroupGeometry], which: int) -> tuple[int, int] | None:
    values = [
        geometry.matches.size0_wh if which == 0 else geometry.matches.size1_wh
        for geometry in geometries
    ]
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def common_to_native_F(
    F_common: np.ndarray,
    size0_wh: tuple[int, int],
    size1_wh: tuple[int, int],
    common_diagonal: float,
) -> np.ndarray:
    scale0 = scale_for_common(size0_wh, common_diagonal)
    scale1 = scale_for_common(size1_wh, common_diagonal)
    T0 = np.diag((scale0, scale0, 1.0))
    T1 = np.diag((scale1, scale1, 1.0))
    return normalize_F(T1.T @ F_common @ T0)


def evaluate_shared(
    geometry: GroupGeometry,
    F_common: np.ndarray,
    threshold: float,
    common_diagonal: float,
) -> dict[str, Any]:
    q0 = to_common_points(
        geometry.matches.points0, geometry.matches.size0_wh, common_diagonal
    )
    q1 = to_common_points(
        geometry.matches.points1, geometry.matches.size1_wh, common_diagonal
    )
    errors = symmetric_epipolar_error(F_common, q0, q1)
    base = geometry.group_inlier_mask
    selected = errors[base]
    return {
        "errors_all": errors,
        "error_common_px": quantile_metrics(selected),
        "inlier_ratio_on_group_inliers": float(np.mean(selected <= threshold)) if len(selected) else 0.0,
        "evaluated_count": int(len(selected)),
    }


def aggregate_shared_errors(
    geometries: list[GroupGeometry],
    F_common: np.ndarray,
    role: str | None,
    roles: dict[str, str],
    common_diagonal: float,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for geometry in geometries:
        if role is not None and roles.get(geometry.frame.frame) != role:
            continue
        evaluation = evaluate_shared(geometry, F_common, math.inf, common_diagonal)
        errors = evaluation["errors_all"][geometry.group_inlier_mask]
        if len(errors):
            values.append(errors)
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def classify_pair(
    validation_metrics: dict[str, float | None],
    train_metrics: dict[str, float | None],
    median_coverage: float,
    median_group_p50: float | None,
) -> tuple[str, str]:
    evaluation = validation_metrics if validation_metrics.get("p50") is not None else train_metrics
    p50 = evaluation.get("p50")
    p95 = evaluation.get("p95")
    if p50 is None or p95 is None:
        return "FAIL", "有效匹配不足，无法检验共享F"
    if p50 <= 0.8 and p95 <= 2.5 and median_coverage >= 0.20:
        return "PASS", "跨场景共享F稳定，可进入固定相机联合标定"
    if p50 <= 1.5 and p95 <= 4.0 and median_coverage >= 0.10:
        return "MARGINAL", "共享F基本成立，但覆盖或尾部误差仍需改善"
    if median_group_p50 is not None and median_group_p50 <= 1.0:
        return (
            "FAIL",
            "单组F很好但共享F较差：优先检查镜头归类、数码裁切/OIS、分辨率或相机是否移动",
        )
    return "FAIL", "单组F也偏差较大：优先改善跨模态匹配和纹理覆盖"


def draw_match_diagnostic(
    geometry: GroupGeometry,
    F_common: np.ndarray,
    threshold: float,
    common_diagonal: float,
    output_path: Path,
    role: str,
    max_lines: int,
) -> None:
    image0 = imread_checked(geometry.frame.paths[geometry.pair.camera0], cv2.IMREAD_COLOR)
    image1 = imread_checked(geometry.frame.paths[geometry.pair.camera1], cv2.IMREAD_COLOR)
    target_h = 500
    scale0 = min(1.0, target_h / image0.shape[0])
    scale1 = min(1.0, target_h / image1.shape[0])
    shown0 = cv2.resize(
        image0, None, fx=scale0, fy=scale0, interpolation=cv2.INTER_AREA
    )
    shown1 = cv2.resize(
        image1, None, fx=scale1, fy=scale1, interpolation=cv2.INTER_AREA
    )
    height = max(shown0.shape[0], shown1.shape[0])
    header = 72
    canvas = np.zeros((height + header, shown0.shape[1] + shown1.shape[1], 3), dtype=np.uint8)
    canvas[header : header + shown0.shape[0], : shown0.shape[1]] = shown0
    canvas[header : header + shown1.shape[0], shown0.shape[1] :] = shown1

    evaluation = evaluate_shared(geometry, F_common, threshold, common_diagonal)
    errors = evaluation["errors_all"]
    indices = np.flatnonzero(geometry.group_inlier_mask)
    if len(indices) > max_lines:
        order = np.argsort(geometry.matches.scores[indices])[::-1]
        # Keep strong matches but sample across the ranked list to avoid a
        # visualization dominated by one textured object.
        positions = np.linspace(0, len(order) - 1, max_lines).astype(int)
        indices = indices[order[positions]]
    x_offset = shown0.shape[1]
    for index in indices:
        p0 = geometry.matches.points0[index] * scale0
        p1 = geometry.matches.points1[index] * scale1
        a = (int(round(p0[0])), int(round(p0[1] + header)))
        b = (int(round(p1[0] + x_offset)), int(round(p1[1] + header)))
        color = (50, 220, 50) if errors[index] <= threshold else (40, 40, 240)
        cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, a, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, b, 2, color, -1, cv2.LINE_AA)
    q = quantile_metrics(errors[geometry.group_inlier_mask])
    title = (
        f"{geometry.frame.frame}  {geometry.pair.label}  role={role}  "
        f"shared-F p50={q['p50']:.3f}px p95={q['p95']:.3f}px"
        if q["p50"] is not None
        else f"{geometry.frame.frame} {geometry.pair.label}: no valid matches"
    )
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "green: shared-F inlier   red: per-frame match inconsistent with shared F",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    imwrite_checked(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def choose_diagnostic_frames(
    geometries: list[GroupGeometry], roles: dict[str, str], maximum: int
) -> list[GroupGeometry]:
    if maximum <= 0 or len(geometries) <= maximum:
        return geometries
    validation = [g for g in geometries if roles.get(g.frame.frame) == "validation"]
    train = [g for g in geometries if roles.get(g.frame.frame) != "validation"]
    chosen = validation[: max(1, maximum // 3)]
    remaining = maximum - len(chosen)
    if remaining > 0 and train:
        positions = np.linspace(0, len(train) - 1, remaining).astype(int)
        chosen.extend(train[index] for index in positions)
    unique: list[GroupGeometry] = []
    seen: set[int] = set()
    for item in chosen:
        if id(item) in seen:
            continue
        seen.add(id(item))
        unique.append(item)
    return unique[:maximum]


def diagnose_pair(
    pair: PairSpec,
    frames: list[FrameRecord],
    output_root: Path,
    args: argparse.Namespace,
    failures: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    geometries: list[GroupGeometry] = []
    for frame in frames:
        path = match_cache_path(output_root, pair, frame.frame)
        if not path.is_file():
            failures.append(
                {
                    "stage": "diagnose",
                    "frame": frame.frame,
                    "pair": pair.label,
                    "error": f"匹配缓存不存在：{path}",
                }
            )
            continue
        try:
            geometry = build_group_geometry(
                frame,
                pair,
                path,
                args.common_diagonal,
                args.ransac_threshold,
                args.homography_threshold,
                args.ransac_confidence,
                args.ransac_max_iters,
            )
            geometries.append(geometry)
        except Exception as exc:
            failure = {
                "stage": "diagnose",
                "frame": frame.frame,
                "pair": pair.label,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            if args.strict:
                raise

    usable = [
        geometry
        for geometry in geometries
        if geometry.F_common is not None
        and int(np.count_nonzero(geometry.group_inlier_mask)) >= args.min_group_inliers
    ]
    roles = deterministic_roles(
        [geometry.frame.frame for geometry in usable],
        args.validation_fraction,
        args.seed,
    )
    train = [geometry for geometry in usable if roles[geometry.frame.frame] == "train"]
    if len(train) < 2:
        report = {
            "pair": pair.label,
            "status": "FAIL",
            "conclusion": "至少需要2个有足够内点的训练场景",
            "available_group_count": len(geometries),
            "usable_group_count": len(usable),
            "groups": [geometry.metrics for geometry in geometries],
        }
        return report, [geometry.metrics for geometry in geometries]

    def pooled(items: list[GroupGeometry]) -> tuple[np.ndarray, np.ndarray]:
        p0_values: list[np.ndarray] = []
        p1_values: list[np.ndarray] = []
        for geometry in items:
            mask = geometry.group_inlier_mask
            p0_values.append(
                to_common_points(
                    geometry.matches.points0[mask],
                    geometry.matches.size0_wh,
                    args.common_diagonal,
                )
            )
            p1_values.append(
                to_common_points(
                    geometry.matches.points1[mask],
                    geometry.matches.size1_wh,
                    args.common_diagonal,
                )
            )
        return np.concatenate(p0_values), np.concatenate(p1_values)

    train0, train1 = pooled(train)
    all0, all1 = pooled(usable)
    F_train, train_ransac_mask = fit_fundamental(
        train0,
        train1,
        args.ransac_threshold,
        args.ransac_confidence,
        args.ransac_max_iters,
    )
    F_all, all_ransac_mask = fit_fundamental(
        all0,
        all1,
        args.ransac_threshold,
        args.ransac_confidence,
        args.ransac_max_iters,
    )
    if F_train is None or F_all is None:
        report = {
            "pair": pair.label,
            "status": "FAIL",
            "conclusion": "汇总匹配仍无法估计共享F",
            "available_group_count": len(geometries),
            "usable_group_count": len(usable),
            "groups": [geometry.metrics for geometry in geometries],
        }
        return report, [geometry.metrics for geometry in geometries]

    train_errors = aggregate_shared_errors(
        usable, F_train, "train", roles, args.common_diagonal
    )
    validation_errors = aggregate_shared_errors(
        usable, F_train, "validation", roles, args.common_diagonal
    )
    all_errors = aggregate_shared_errors(
        usable, F_all, None, roles, args.common_diagonal
    )
    train_metrics = quantile_metrics(train_errors)
    validation_metrics = quantile_metrics(validation_errors)
    all_metrics = quantile_metrics(all_errors)
    coverages = [
        min(geometry.metrics["coverage0"], geometry.metrics["coverage1"])
        for geometry in usable
    ]
    group_p50 = [
        geometry.metrics["group_error_common_px"]["p50"]
        for geometry in usable
        if geometry.metrics["group_error_common_px"]["p50"] is not None
    ]
    median_coverage = float(np.median(coverages)) if coverages else 0.0
    median_group_p50 = float(np.median(group_p50)) if group_p50 else None
    homography_ratios = [
        float(value)
        for geometry in usable
        if (value := geometry.metrics.get("homography_dominance_ratio")) is not None
    ]
    median_homography_dominance = (
        float(np.median(homography_ratios)) if homography_ratios else None
    )
    depth_rich_group_count = sum(
        geometry.metrics.get("depth_diversity_hint") == "DEPTH_RICH"
        for geometry in usable
    )
    planar_or_low_parallax_group_count = sum(
        geometry.metrics.get("depth_diversity_hint") == "PLANAR_OR_LOW_PARALLAX"
        for geometry in usable
    )
    status, conclusion = classify_pair(
        validation_metrics,
        train_metrics,
        median_coverage,
        median_group_p50,
    )
    canonical0 = modal_size(usable, 0)
    canonical1 = modal_size(usable, 1)
    native_F = (
        common_to_native_F(
            F_all, canonical0, canonical1, args.common_diagonal
        )
        if canonical0 is not None and canonical1 is not None
        else None
    )

    rows: list[dict[str, Any]] = []
    usable_ids = {id(geometry) for geometry in usable}
    for geometry in geometries:
        metrics = dict(geometry.metrics)
        role = roles.get(geometry.frame.frame, "excluded")
        metrics["role"] = role
        if id(geometry) in usable_ids:
            train_eval = evaluate_shared(
                geometry, F_train, args.ransac_threshold, args.common_diagonal
            )
            all_eval = evaluate_shared(
                geometry, F_all, args.ransac_threshold, args.common_diagonal
            )
            metrics["shared_train_error_common_px"] = train_eval["error_common_px"]
            metrics["shared_train_inlier_ratio"] = train_eval[
                "inlier_ratio_on_group_inliers"
            ]
            metrics["shared_all_error_common_px"] = all_eval["error_common_px"]
            metrics["shared_all_inlier_ratio"] = all_eval[
                "inlier_ratio_on_group_inliers"
            ]
        else:
            metrics["shared_train_error_common_px"] = quantile_metrics(np.empty(0))
            metrics["shared_train_inlier_ratio"] = 0.0
            metrics["shared_all_error_common_px"] = quantile_metrics(np.empty(0))
            metrics["shared_all_inlier_ratio"] = 0.0
        rows.append(metrics)

    report = {
        "pair": pair.label,
        "status": status,
        "conclusion": conclusion,
        "coordinate_definition": (
            f"each image isotropically scaled so diagonal={args.common_diagonal:g}; "
            "the common coordinate is independent of native camera resolution"
        ),
        "ransac_threshold_common_px": args.ransac_threshold,
        "available_group_count": len(geometries),
        "usable_group_count": len(usable),
        "train_group_count": sum(role == "train" for role in roles.values()),
        "validation_group_count": sum(role == "validation" for role in roles.values()),
        "roles": roles,
        "median_min_grid_coverage": median_coverage,
        "median_per_group_error_p50": median_group_p50,
        "homography_diagnostic_note": (
            "H dominance is measured only among per-scene F inliers; high values "
            "mean planar, low-parallax or mostly distant support, not a proof of planarity"
        ),
        "median_homography_dominance_ratio": median_homography_dominance,
        "depth_rich_group_count": int(depth_rich_group_count),
        "planar_or_low_parallax_group_count": int(
            planar_or_low_parallax_group_count
        ),
        "shared_train_error_common_px": train_metrics,
        "shared_validation_error_common_px": validation_metrics,
        "shared_all_error_common_px": all_metrics,
        "F_train_common": F_train,
        "F_all_common": F_all,
        "shared_train_ransac_inliers": int(np.count_nonzero(train_ransac_mask)),
        "shared_train_ransac_total": int(len(train_ransac_mask)),
        "shared_all_ransac_inliers": int(np.count_nonzero(all_ransac_mask)),
        "shared_all_ransac_total": int(len(all_ransac_mask)),
        "canonical_size0_wh": canonical0,
        "canonical_size1_wh": canonical1,
        "F_all_native_for_canonical_sizes": native_F,
        "groups": rows,
    }

    if not args.no_diagnostics:
        selected = choose_diagnostic_frames(
            usable, roles, args.diagnostic_max_per_pair
        )
        for geometry in selected:
            try:
                draw_match_diagnostic(
                    geometry,
                    F_train,
                    args.ransac_threshold,
                    args.common_diagonal,
                    output_root
                    / "diagnostics"
                    / pair.name
                    / f"{geometry.frame.frame}.jpg",
                    roles[geometry.frame.frame],
                    args.diagnostic_max_lines,
                )
            except Exception as exc:
                failures.append(
                    {
                        "stage": "diagnostic_image",
                        "frame": geometry.frame.frame,
                        "pair": pair.label,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return report, rows


def write_group_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair",
        "frame",
        "scene_id",
        "variant",
        "role",
        "matcher",
        "selected_match_count",
        "group_inlier_count",
        "group_inlier_ratio",
        "group_error_p50_common_px",
        "group_error_p95_common_px",
        "coverage0",
        "coverage1",
        "homography_inlier_count",
        "homography_dominance_ratio",
        "non_homography_count",
        "depth_diversity_hint",
        "shared_train_p50_common_px",
        "shared_train_p95_common_px",
        "shared_train_inlier_ratio",
        "shared_all_p50_common_px",
        "shared_all_p95_common_px",
        "shared_all_inlier_ratio",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            group_error = row.get("group_error_common_px", {})
            train_error = row.get("shared_train_error_common_px", {})
            all_error = row.get("shared_all_error_common_px", {})
            writer.writerow(
                {
                    "pair": row.get("pair"),
                    "frame": row.get("frame"),
                    "scene_id": row.get("scene_id"),
                    "variant": row.get("variant"),
                    "role": row.get("role"),
                    "matcher": row.get("matcher"),
                    "selected_match_count": row.get("selected_match_count"),
                    "group_inlier_count": row.get("group_inlier_count"),
                    "group_inlier_ratio": row.get("group_inlier_ratio"),
                    "group_error_p50_common_px": group_error.get("p50"),
                    "group_error_p95_common_px": group_error.get("p95"),
                    "coverage0": row.get("coverage0"),
                    "coverage1": row.get("coverage1"),
                    "homography_inlier_count": row.get("homography_inlier_count"),
                    "homography_dominance_ratio": row.get(
                        "homography_dominance_ratio"
                    ),
                    "non_homography_count": row.get("non_homography_count"),
                    "depth_diversity_hint": row.get("depth_diversity_hint"),
                    "shared_train_p50_common_px": train_error.get("p50"),
                    "shared_train_p95_common_px": train_error.get("p95"),
                    "shared_train_inlier_ratio": row.get("shared_train_inlier_ratio"),
                    "shared_all_p50_common_px": all_error.get("p50"),
                    "shared_all_p95_common_px": all_error.get("p95"),
                    "shared_all_inlier_ratio": row.get("shared_all_inlier_ratio"),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "在完整图像上提取固定相机匹配点，用留出场景检验共享F，"
            "并诊断匹配是否具有多深度约束"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"multialign matching {PROGRAM_VERSION}"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="数据整理输出目录，或其 metadata/dataset.json",
    )
    parser.add_argument("--reference-cameras", nargs="+", required=True)
    parser.add_argument("--target-camera", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出目录；默认 <dataset>/fixed_rig_geometry_roma_all",
    )
    parser.add_argument(
        "--split",
        default="scene",
        help="splits下的名称、txt路径或all",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="相机对列表",
    )
    parser.add_argument("--limit", type=int, default=None, help="仅处理前N组，便于试跑")
    parser.add_argument(
        "--target-source",
        choices=("composite", "gray", "edges"),
        default="composite",
        help="target representation used by RoMa",
    )
    parser.add_argument(
        "--cross-representation",
        choices=("gray", "structure", "rgb-gray"),
        default="gray",
        help="送入RoMa的跨模态三通道表示",
    )
    parser.add_argument("--roma-setting", default="fast")
    parser.add_argument("--roma-samples", type=int, default=8000)
    parser.add_argument("--allow-cpu", action="store_true", help="允许RoMa在CPU运行")
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="开启RoMa torch.compile；原生Windows通常不要开启",
    )
    parser.add_argument("--max-matches-per-group", type=int, default=2400)
    parser.add_argument("--grid-cols", type=int, default=24)
    parser.add_argument("--grid-rows", type=int, default=18)
    parser.add_argument("--grid-max-per-cell", type=int, default=6)
    parser.add_argument(
        "--common-diagonal",
        type=float,
        default=1000.0,
        help="F估计的分辨率无关公共坐标对角线",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=1.5,
        help="共享坐标中的MAGSAC阈值",
    )
    parser.add_argument(
        "--homography-threshold",
        type=float,
        default=3.0,
        help="公共坐标中单应性MAGSAC阈值；仅用于多深度诊断",
    )
    parser.add_argument("--ransac-confidence", type=float, default=0.999)
    parser.add_argument("--ransac-max-iters", type=int, default=20000)
    parser.add_argument("--min-group-inliers", type=int, default=30)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--diagnostic-max-per-pair", type=int, default=12)
    parser.add_argument("--diagnostic-max-lines", type=int, default=180)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="不加载匹配模型，仅从已有matches/*.npz重算报告",
    )
    parser.add_argument("--overwrite-matches", action="store_true")
    parser.add_argument("--strict", action="store_true", help="任一组失败时立即退出")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise RigMatchError("--limit必须大于0")
    if args.roma_samples < 100:
        raise RigMatchError("--roma-samples至少100")
    if args.max_matches_per_group < 8:
        raise RigMatchError("--max-matches-per-group至少8")
    if min(args.grid_cols, args.grid_rows, args.grid_max_per_cell) <= 0:
        raise RigMatchError("grid参数必须为正数")
    if (
        args.common_diagonal <= 0
        or args.ransac_threshold <= 0
        or args.homography_threshold <= 0
    ):
        raise RigMatchError("公共对角线、F阈值和H阈值必须为正数")
    if not 0.0 <= args.validation_fraction < 0.5:
        raise RigMatchError("--validation-fraction必须在[0,0.5)内")
    if not 0.0 < args.ransac_confidence < 1.0:
        raise RigMatchError("--ransac-confidence必须在(0,1)内")
    if len({pair.name for pair in args.pairs}) != len(args.pairs):
        raise RigMatchError("--pairs中存在重复相机对")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        configure_rig(args.reference_cameras, args.target_camera)
        pair_values = args.pairs or list(DEFAULT_PAIRS)
        args.pairs = [parse_pair(value) for value in pair_values]
        validate_args(args)
        dataset_root, dataset_path, dataset, frames = load_dataset(
            args.dataset,
            args.split,
            args.target_source,
            args.limit,
        )
        output_root = (
            args.output.expanduser().resolve()
            if args.output is not None
            else dataset_root / "fixed_rig_geometry_roma_all"
        )
        output_root.mkdir(parents=True, exist_ok=True)
        failures: list[dict[str, Any]] = []
        print(f"数据：{dataset_path}")
        print(f"采集组：{len(frames)}；split={args.split}")
        print(f"输出：{output_root}")
        print("相机对：" + ", ".join(pair.label for pair in args.pairs))
        print("匹配范围：完整图像；后端：所有相机对统一RoMa v2")

        if not args.diagnose_only:
            run_roma_matching(frames, args.pairs, output_root, args, failures)

        reports: dict[str, Any] = {}
        csv_rows: list[dict[str, Any]] = []
        print("\n开始固定F诊断……")
        for pair in args.pairs:
            report, rows = diagnose_pair(
                pair, frames, output_root, args, failures
            )
            reports[pair.name] = report
            csv_rows.extend(rows)
            write_json(output_root / "reports" / f"{pair.name}.json", report)
            metrics = report.get("shared_validation_error_common_px", {})
            if metrics.get("p50") is None:
                metrics = report.get("shared_train_error_common_px", {})
            coverage = report.get("median_min_grid_coverage")
            h_ratio = report.get("median_homography_dominance_ratio")
            coverage_text = "--" if coverage is None else f"{100.0 * coverage:.1f}%"
            h_text = "--" if h_ratio is None else f"{h_ratio:.3f}"
            print(
                f"[{pair.label}] {report.get('status')}: "
                f"p50={metrics.get('p50')} p95={metrics.get('p95')} "
                f"coverage={coverage_text} H50={h_text} | "
                f"{report.get('conclusion')}"
            )

        summary = {
            "schema_version": 1,
            "task": "fixed-rig shared fundamental matrix diagnostic",
            "dataset": str(dataset_path),
            "dataset_group_count": dataset.get("group_count"),
            "selected_group_count": len(frames),
            "split": args.split,
            "target_source": args.target_source,
            "cross_representation": args.cross_representation,
            "matcher_backend": "roma_v2_all_pairs",
            "matching_scope": "full_image",
            "common_diagonal": args.common_diagonal,
            "ransac_threshold_common_px": args.ransac_threshold,
            "homography_threshold_common_px": args.homography_threshold,
            "pairs": reports,
            "failures": failures,
        }
        write_json(output_root / "reports" / "summary.json", summary)
        write_json(output_root / "reports" / "failures.json", failures)
        write_group_csv(output_root / "reports" / "per_group.csv", csv_rows)
        print(f"\n总报告：{output_root / 'reports' / 'summary.json'}")
        print(f"逐组表：{output_root / 'reports' / 'per_group.csv'}")
        print(f"诊断图：{output_root / 'diagnostics'}")
        if failures:
            print(
                f"有 {len(failures)} 项失败，详见 "
                f"{output_root / 'reports' / 'failures.json'}",
                file=sys.stderr,
            )
        successful = sum(report.get("status") in {"PASS", "MARGINAL"} for report in reports.values())
        return 0 if successful > 0 else 2
    except KeyboardInterrupt:
        print("\n用户中断；已生成的匹配NPZ可在下次自动续跑。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        if getattr(args, "strict", False):
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
