#!/usr/bin/env python3
"""Jointly calibrate a fixed multi-camera reference/target rig.

The program consumes full-image sparse-match NPZ files produced by
:mod:`multialign.stages.matching`. Every pair must come from the RoMa-all cache
protocol. Known reference intrinsics are fixed by default; low-order distortion
is refined under strong priors. The unknown target focal length (and optionally
its principal point) is refined together with one globally consistent pose graph.

Natural-scene epipolar geometry has no absolute translation scale.  Camera
centres are therefore expressed in the configured anchor-camera coordinate
system. The anchor-to-scale-reference baseline is normalized to one unless
``--scale-baseline-mm`` is provided. A soft prior keeps anchor-to-camera
baseline magnitudes in a similar range; a hard gate rejects implausible ratios.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import traceback
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import cv2
    import numpy as np
    from scipy.optimize import least_squares
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少依赖。请执行：python -m pip install numpy scipy opencv-python"
    ) from exc


REFERENCE_CAMERAS = ("reference_a", "reference_b")
TARGET_CAMERA = "target"
ANCHOR_CAMERA = REFERENCE_CAMERAS[0]
SCALE_REFERENCE_CAMERA = REFERENCE_CAMERAS[1]
STRICT_REFERENCE_CAMERAS: frozenset[str] = frozenset()
CAMERAS = (*REFERENCE_CAMERAS, TARGET_CAMERA)
CAMERA_INDEX = {name: index for index, name in enumerate(CAMERAS)}
NON_ANCHOR = tuple(name for name in CAMERAS if name != ANCHOR_CAMERA)
PROGRAM_VERSION = "3.2-prior-safe-robust-loss"
DISTORTION_NAMES = ("k1", "k2", "p1", "p2", "k3")


class CalibrationError(RuntimeError):
    pass


def configure_rig(
    reference_cameras: Sequence[str],
    target_camera: str,
    anchor_camera: str,
    scale_reference_camera: str,
    strict_reference_cameras: Sequence[str] = (),
) -> None:
    global REFERENCE_CAMERAS, TARGET_CAMERA, ANCHOR_CAMERA
    global SCALE_REFERENCE_CAMERA, STRICT_REFERENCE_CAMERAS
    global CAMERAS, CAMERA_INDEX, NON_ANCHOR
    references = tuple(reference_cameras)
    if len(references) < 2 or len(set(references)) != len(references):
        raise CalibrationError("--reference-cameras requires at least two unique names")
    if target_camera in references:
        raise CalibrationError("--target-camera must not be a reference camera")
    if anchor_camera not in references:
        raise CalibrationError("--anchor-camera must be a reference camera")
    if scale_reference_camera not in references or scale_reference_camera == anchor_camera:
        raise CalibrationError(
            "--scale-reference-camera must be a non-anchor reference camera"
        )
    strict = frozenset(strict_reference_cameras)
    unknown_strict = strict.difference(references)
    if unknown_strict:
        raise CalibrationError(
            "--strict-reference-cameras contains unknown names: "
            + ", ".join(sorted(unknown_strict))
        )
    REFERENCE_CAMERAS = references
    TARGET_CAMERA = target_camera
    ANCHOR_CAMERA = anchor_camera
    SCALE_REFERENCE_CAMERA = scale_reference_camera
    STRICT_REFERENCE_CAMERAS = strict
    CAMERAS = (*references, target_camera)
    CAMERA_INDEX = {name: index for index, name in enumerate(CAMERAS)}
    NON_ANCHOR = tuple(name for name in CAMERAS if name != anchor_camera)


@dataclass
class RawMatch:
    path: Path
    frame: str
    pair: tuple[str, str]
    points0: np.ndarray
    points1: np.ndarray
    scores: np.ndarray
    size0_wh: tuple[int, int]
    size1_wh: tuple[int, int]


@dataclass
class CameraSeed:
    name: str
    size_wh: tuple[int, int]
    source_size_wh: tuple[int, int]
    K0: np.ndarray
    dist0: np.ndarray


@dataclass
class MatchGroup:
    frame: str
    pair: tuple[str, str]
    points0: np.ndarray
    points1: np.ndarray
    scores: np.ndarray
    role: str
    raw_count: int
    inlier_count: int
    coverage0: float
    coverage1: float
    homography_ratio: float | None
    pre_shared_count: int = 0
    shared_inlier_count: int = 0
    shared_inlier_ratio: float = 0.0
    shared_error_p50: float | None = None
    shared_error_p95: float | None = None


@dataclass
class RigModel:
    rotations: dict[str, np.ndarray]
    centers: dict[str, np.ndarray]
    intrinsics: dict[str, np.ndarray]
    distortion: dict[str, np.ndarray]


@dataclass
class ParameterLayout:
    rotation: dict[str, slice]
    center: dict[str, slice]
    reference_log_scale: dict[str, int]
    target_log_scale: int | None
    target_pp: slice | None
    distortion: dict[str, dict[int, int]]
    size: int


@dataclass
class StageResult:
    name: str
    model: RigModel
    optimizer: dict[str, Any]
    metrics: dict[str, Any]
    physical: dict[str, Any]
    bound_hits: list[str]
    gate_passed: bool
    gate_reasons: list[str]


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


def canonical_pair(camera0: str, camera1: str) -> tuple[str, str]:
    if camera0 == camera1 or camera0 not in CAMERA_INDEX or camera1 not in CAMERA_INDEX:
        raise CalibrationError(f"无效相机对：{camera0}-{camera1}")
    if CAMERA_INDEX[camera0] < CAMERA_INDEX[camera1]:
        return camera0, camera1
    return camera1, camera0


def parse_pair(text: str) -> tuple[str, str]:
    normalized = text.strip().replace("__", "-")
    parts = normalized.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"无效相机对：{text}")
    try:
        return canonical_pair(parts[0], parts[1])
    except CalibrationError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def pair_name(pair: tuple[str, str]) -> str:
    return f"{pair[0]}__{pair[1]}"


def pair_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]}-{pair[1]}"


def resolve_matches_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if (root / "matches").is_dir():
        return root / "matches"
    if root.is_dir():
        return root
    raise CalibrationError(f"匹配目录不存在：{root}")


def resolve_quality_report(
    geometry_root: Path,
    matches_root: Path,
    requested: Path | None,
    ignore: bool,
) -> Path | None:
    if ignore:
        return None
    if requested is not None:
        path = requested.expanduser().resolve()
        if not path.is_file():
            raise CalibrationError(f"指定的逐组质量表不存在：{path}")
        return path
    candidates = (
        geometry_root.expanduser().resolve() / "reports" / "per_group.csv",
        matches_root.parent / "reports" / "per_group.csv",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def csv_boolean(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def load_quality_report(
    path: Path | None,
) -> dict[tuple[str, str], tuple[bool, str]]:
    """Read explicit group eligibility decisions from matcher diagnostics.

    RoMa-all matcher v5.0 wrote useful per-group metrics but did not yet write
    the explicit ``shared_fit_eligible`` decision column introduced by the
    later full-frame matcher.  Such a table is not an error: return no inherited
    decisions and let this calibrator run its own per-group MAGSAC plus shared-F
    filtering.  ``pair`` and ``frame`` remain mandatory when an eligibility
    column is present.
    """
    if path is None:
        return {}
    result: dict[tuple[str, str], tuple[bool, str]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            required_identity = {"pair", "frame"}
            missing = required_identity.difference(fields)
            if missing:
                raise CalibrationError(
                    f"逐组质量表缺少列 {sorted(missing)}：{path}"
                )
            if "shared_fit_eligible" not in fields:
                print(
                    "兼容提示：逐组质量表来自旧版全RoMa匹配器，缺少"
                    "shared_fit_eligible；不继承旧表预筛选，改由本标定器重新执行"
                    "逐组MAGSAC和共享F二次清洗。"
                )
                return {}
            for row in reader:
                label = str(row.get("pair") or "").strip().replace("__", "-")
                frame = str(row.get("frame") or "").strip()
                eligible = csv_boolean(row.get("shared_fit_eligible"))
                if not label or not frame or eligible is None:
                    continue
                reason = str(row.get("shared_fit_exclusion_reasons") or "").strip()
                result[(label, frame)] = (eligible, reason)
    except OSError as exc:
        raise CalibrationError(f"无法读取逐组质量表 {path}：{exc}") from exc
    return result


def directory_pair(path: Path) -> tuple[str, str] | None:
    text = path.name.replace("__", "-")
    parts = text.split("-")
    if len(parts) != 2 or any(part not in CAMERA_INDEX for part in parts):
        return None
    if parts[0] == parts[1]:
        return None
    return canonical_pair(parts[0], parts[1])


def discover_pair_directories(root: Path) -> dict[tuple[str, str], Path]:
    found: dict[tuple[str, str], Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        pair = directory_pair(path)
        if pair is None:
            continue
        if pair in found:
            raise CalibrationError(
                f"同一相机对存在多个目录：{found[pair]} 和 {path}"
            )
        found[pair] = path
    if not found:
        raise CalibrationError(f"没有找到 matches/<camera0>__<camera1>：{root}")
    return found


def read_npz(path: Path, expected_pair: tuple[str, str]) -> RawMatch:
    try:
        with np.load(path, allow_pickle=False) as archive:
            p0 = np.asarray(archive["points0"], dtype=np.float64).reshape(-1, 2)
            p1 = np.asarray(archive["points1"], dtype=np.float64).reshape(-1, 2)
            scores = np.asarray(archive["scores"], dtype=np.float64).reshape(-1)
            size0 = tuple(int(x) for x in archive["size0_wh"].reshape(2))
            size1 = tuple(int(x) for x in archive["size1_wh"].reshape(2))
            metadata: dict[str, Any] = {}
            if "metadata_json" in archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"无法读取 {path}：{exc}") from exc
    if not (len(p0) == len(p1) == len(scores)):
        raise CalibrationError(f"点和分数长度不一致：{path}")
    if metadata.get("matching_scope") != "full_image":
        raise CalibrationError(
            f"不是全图匹配缓存：{path}；请用 roma-all v5 新目录重新生成"
        )
    if metadata.get("matcher_backend") != "roma_v2_all_pairs":
        raise CalibrationError(
            f"不是RoMa全相机对缓存：{path}；请用 roma-all v5 新目录重新生成"
        )
    camera0 = str(metadata.get("camera0", ""))
    camera1 = str(metadata.get("camera1", ""))
    if (camera0, camera1) == expected_pair:
        reverse = False
    elif (camera1, camera0) == expected_pair:
        reverse = True
    else:
        folder_parts = path.parent.name.replace("__", "-").split("-")
        reverse = tuple(folder_parts) == expected_pair[::-1]
    if reverse:
        p0, p1 = p1, p0
        size0, size1 = size1, size0
    frame = str(metadata.get("frame") or path.stem)
    return RawMatch(path, frame, expected_pair, p0, p1, scores, size0, size1)


def load_raw_matches(
    pair_dirs: dict[tuple[str, str], Path],
    selected_pairs: list[tuple[str, str]],
    limit: int | None,
) -> list[RawMatch]:
    records: list[RawMatch] = []
    for pair in selected_pairs:
        directory = pair_dirs.get(pair)
        if directory is None:
            raise CalibrationError(f"缺少匹配目录：{pair_name(pair)}")
        paths = sorted(directory.glob("*.npz"))
        if limit is not None:
            paths = paths[:limit]
        if not paths:
            raise CalibrationError(f"匹配目录为空：{directory}")
        for path in paths:
            records.append(read_npz(path, pair))
    return records


def modal_sizes(records: list[RawMatch]) -> dict[str, tuple[int, int]]:
    values: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for record in records:
        values[record.pair[0]].append(record.size0_wh)
        values[record.pair[1]].append(record.size1_wh)
    result: dict[str, tuple[int, int]] = {}
    for camera in CAMERAS:
        if not values[camera]:
            raise CalibrationError(f"匹配数据没有相机 {camera}")
        result[camera] = Counter(values[camera]).most_common(1)[0][0]
    return result


def load_camera_seeds(
    calibration_path: Path,
    sizes: dict[str, tuple[int, int]],
) -> tuple[dict[str, CameraSeed], dict[str, Any]]:
    try:
        calibration = json.loads(
            calibration_path.read_text(encoding="utf-8-sig")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"无法读取初始校准：{exc}") from exc
    cameras = calibration.get("cameras")
    if not isinstance(cameras, dict):
        raise CalibrationError("初始校准中没有 cameras 字典")
    result: dict[str, CameraSeed] = {}
    for name in CAMERAS:
        item = cameras.get(name)
        if not isinstance(item, dict):
            raise CalibrationError(f"初始校准缺少 {name}")
        key = "K" if "K" in item else "intrinsic"
        if key not in item:
            raise CalibrationError(f"{name} 缺少 K/intrinsic")
        K = np.asarray(item[key], dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise CalibrationError(f"{name} 的K无效")
        source_size = tuple(int(x) for x in item.get("image_size", sizes[name]))
        target_size = sizes[name]
        if len(source_size) != 2 or min(source_size) <= 0:
            raise CalibrationError(f"{name} 的image_size无效：{source_size}")
        source_aspect = source_size[0] / source_size[1]
        target_aspect = target_size[0] / target_size[1]
        if not math.isclose(source_aspect, target_aspect, rel_tol=0.01):
            raise CalibrationError(
                f"{name} 长宽比不一致：{source_size} -> {target_size}"
            )
        K = np.diag(
            [target_size[0] / source_size[0], target_size[1] / source_size[1], 1.0]
        ) @ K
        dist = np.asarray(item.get("dist", np.zeros(5)), dtype=np.float64).reshape(-1)
        if len(dist) < 5:
            dist = np.pad(dist, (0, 5 - len(dist)))
        dist = dist[:5]
        if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.isfinite(dist).all():
            raise CalibrationError(f"{name} 的焦距或畸变无效")
        result[name] = CameraSeed(name, target_size, source_size, K, dist)
    return result, calibration


def override_target_seed(
    seeds: dict[str, CameraSeed],
    calibration_path: Path | None,
) -> dict[str, Any] | None:
    """Override only the target K/dist seed from another calibration file."""
    if calibration_path is None:
        return None
    path = calibration_path.expanduser().resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"无法读取目标相机初值校准 {path}：{exc}") from exc
    cameras = data.get("cameras")
    item = cameras.get(TARGET_CAMERA) if isinstance(cameras, dict) else None
    if not isinstance(item, dict):
        raise CalibrationError(
            f"target seed calibration has no cameras.{TARGET_CAMERA}: {path}"
        )
    key = "K" if "K" in item else "intrinsic"
    if key not in item:
        raise CalibrationError(f"目标相机初值校准缺少K/intrinsic：{path}")
    K = np.asarray(item[key], dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise CalibrationError(f"目标相机初值校准中的K无效：{path}")
    target = seeds[TARGET_CAMERA].size_wh
    source = tuple(int(x) for x in item.get("image_size", target))
    if len(source) != 2 or min(source) <= 0:
        raise CalibrationError(f"目标相机初值image_size无效：{source}")
    if not math.isclose(source[0] / source[1], target[0] / target[1], rel_tol=0.01):
        raise CalibrationError(f"目标相机初值长宽比不一致：{source} -> {target}")
    K = np.diag([target[0] / source[0], target[1] / source[1], 1.0]) @ K
    dist = np.asarray(item.get("dist", np.zeros(5)), dtype=np.float64).reshape(-1)
    if len(dist) < 5:
        dist = np.pad(dist, (0, 5 - len(dist)))
    dist = dist[:5]
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.isfinite(dist).all():
        raise CalibrationError(f"目标相机初值焦距或畸变无效：{path}")
    previous = seeds[TARGET_CAMERA]
    seeds[TARGET_CAMERA] = CameraSeed(TARGET_CAMERA, target, source, K, dist)
    return {
        "path": path,
        "source_size_wh": source,
        "target_size_wh": target,
        "previous_K": previous.K0,
        "override_K": K,
        "previous_dist": previous.dist0,
        "override_dist": dist,
    }


def rescale_points(
    points: np.ndarray,
    source: tuple[int, int],
    target: tuple[int, int],
    label: str,
) -> np.ndarray:
    if source == target:
        return np.asarray(points, dtype=np.float64)
    if not math.isclose(
        source[0] / source[1], target[0] / target[1], rel_tol=0.01
    ):
        raise CalibrationError(f"{label} 长宽比改变：{source} -> {target}")
    return np.asarray(points, dtype=np.float64) * np.array(
        [target[0] / source[0], target[1] / source[1]], dtype=np.float64
    )


def undistort_pixels(points: np.ndarray, seed: CameraSeed) -> np.ndarray:
    if np.max(np.abs(seed.dist0)) <= 1e-12:
        return np.asarray(points, dtype=np.float64)
    return cv2.undistortPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
        seed.K0,
        seed.dist0,
        P=seed.K0,
    ).reshape(-1, 2)


def common_scale(size_wh: tuple[int, int], diagonal: float) -> float:
    return diagonal / math.hypot(size_wh[0], size_wh[1])


def to_common(points: np.ndarray, size_wh: tuple[int, int], diagonal: float) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) * common_scale(size_wh, diagonal)


def fit_fundamental(
    p0: np.ndarray,
    p1: np.ndarray,
    threshold: float,
    confidence: float,
    max_iters: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    count = len(p0)
    if count < 8:
        return None, np.zeros(count, dtype=bool)
    method = int(getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC))
    try:
        F, mask = cv2.findFundamentalMat(
            np.asarray(p0, dtype=np.float64),
            np.asarray(p1, dtype=np.float64),
            method,
            float(threshold),
            float(confidence),
            int(max_iters),
        )
    except TypeError:
        F, mask = cv2.findFundamentalMat(
            np.asarray(p0, dtype=np.float64),
            np.asarray(p1, dtype=np.float64),
            method,
            float(threshold),
            float(confidence),
        )
    if F is None or np.asarray(F).size < 9:
        return None, np.zeros(count, dtype=bool)
    F = np.asarray(F, dtype=np.float64).reshape(-1, 3)[:3]
    norm = float(np.linalg.norm(F))
    if not math.isfinite(norm) or norm <= 1e-15:
        return None, np.zeros(count, dtype=bool)
    F /= norm
    inliers = (
        np.asarray(mask).reshape(-1).astype(bool)
        if mask is not None and len(np.asarray(mask).reshape(-1)) == count
        else np.ones(count, dtype=bool)
    )
    return F, inliers


def signed_sampson(F: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    x0 = np.column_stack((p0, np.ones(len(p0))))
    x1 = np.column_stack((p1, np.ones(len(p1))))
    line1 = (F @ x0.T).T
    line0 = (F.T @ x1.T).T
    numerator = np.sum(x1 * line1, axis=1)
    denominator = np.sqrt(
        line1[:, 0] ** 2
        + line1[:, 1] ** 2
        + line0[:, 0] ** 2
        + line0[:, 1] ** 2
        + 1e-18
    )
    return numerator / denominator


def grid_coverage(points: np.ndarray, size: tuple[int, int], cols: int = 12, rows: int = 9) -> float:
    if not len(points):
        return 0.0
    columns = np.clip((points[:, 0] * cols / size[0]).astype(int), 0, cols - 1)
    row_values = np.clip((points[:, 1] * rows / size[1]).astype(int), 0, rows - 1)
    return len(set(zip(row_values.tolist(), columns.tolist()))) / float(cols * rows)


def homography_ratio(p0: np.ndarray, p1: np.ndarray, threshold: float) -> float | None:
    if len(p0) < 8:
        return None
    method = int(getattr(cv2, "USAC_MAGSAC", cv2.RANSAC))
    try:
        H, mask = cv2.findHomography(
            p0, p1, method=method, ransacReprojThreshold=float(threshold),
            maxIters=20000, confidence=0.999
        )
    except (TypeError, cv2.error):
        H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, float(threshold))
    if H is None or mask is None:
        return None
    return float(np.mean(np.asarray(mask).reshape(-1).astype(bool)))


def balanced_indices(
    p0: np.ndarray,
    p1: np.ndarray,
    scores: np.ndarray,
    size0: tuple[int, int],
    size1: tuple[int, int],
    maximum: int,
    cols: int = 12,
    rows: int = 9,
) -> np.ndarray:
    if len(p0) <= maximum:
        return np.arange(len(p0), dtype=np.int64)
    safe_scores = np.nan_to_num(scores, nan=-np.inf)
    order = np.argsort(-safe_scores, kind="stable")
    per_cell = max(2, int(math.ceil(maximum / float(cols * rows))))
    counts0 = np.zeros((rows, cols), dtype=np.int32)
    counts1 = np.zeros((rows, cols), dtype=np.int32)
    selected: list[int] = []

    def cell(point: np.ndarray, size: tuple[int, int]) -> tuple[int, int]:
        col = int(np.clip(point[0] * cols / size[0], 0, cols - 1))
        row = int(np.clip(point[1] * rows / size[1], 0, rows - 1))
        return row, col

    for index in order:
        c0 = cell(p0[index], size0)
        c1 = cell(p1[index], size1)
        if counts0[c0] >= per_cell or counts1[c1] >= per_cell:
            continue
        selected.append(int(index))
        counts0[c0] += 1
        counts1[c1] += 1
        if len(selected) >= maximum:
            break
    if len(selected) < maximum:
        used = set(selected)
        for index in order:
            if int(index) not in used:
                selected.append(int(index))
            if len(selected) >= maximum:
                break
    return np.asarray(selected, dtype=np.int64)


def deterministic_roles(frames: Iterable[str], fraction: float, seed: int) -> dict[str, str]:
    unique = sorted(set(frames))
    if fraction <= 0.0 or len(unique) < 5:
        return {frame: "train" for frame in unique}
    count = max(1, int(round(len(unique) * fraction)))
    count = min(count, len(unique) - 3)
    ranked = sorted(
        unique,
        key=lambda frame: hashlib.sha256(f"{seed}:{frame}".encode()).digest(),
    )
    validation = set(ranked[:count])
    return {frame: ("validation" if frame in validation else "train") for frame in unique}


def prepare_groups(
    records: list[RawMatch],
    seeds: dict[str, CameraSeed],
    sizes: dict[str, tuple[int, int]],
    args: argparse.Namespace,
    eligibility: dict[tuple[str, str], tuple[bool, str]] | None = None,
) -> tuple[list[MatchGroup], list[dict[str, Any]]]:
    provisional: list[MatchGroup] = []
    excluded: list[dict[str, Any]] = []
    eligibility = eligibility or {}
    for record in records:
        c0, c1 = record.pair
        raw_count = len(record.points0)
        quality = eligibility.get((pair_label(record.pair), record.frame))
        if quality is not None and not quality[0]:
            excluded.append({
                "frame": record.frame,
                "pair": pair_label(record.pair),
                "raw_count": raw_count,
                "inlier_count": None,
                "reason": (
                    "matcher diagnostic exclusion: "
                    + (quality[1] or "shared_fit_eligible=False")
                ),
            })
            continue
        raw0 = rescale_points(record.points0, record.size0_wh, sizes[c0], str(record.path))
        raw1 = rescale_points(record.points1, record.size1_wh, sizes[c1], str(record.path))
        finite = np.isfinite(raw0).all(axis=1) & np.isfinite(raw1).all(axis=1)
        finite &= (
            (raw0[:, 0] >= -0.5) & (raw0[:, 0] <= sizes[c0][0] - 0.5)
            & (raw0[:, 1] >= -0.5) & (raw0[:, 1] <= sizes[c0][1] - 0.5)
            & (raw1[:, 0] >= -0.5) & (raw1[:, 0] <= sizes[c1][0] - 0.5)
            & (raw1[:, 1] >= -0.5) & (raw1[:, 1] <= sizes[c1][1] - 0.5)
        )
        raw0, raw1, scores = raw0[finite], raw1[finite], record.scores[finite]
        clean0 = undistort_pixels(raw0, seeds[c0])
        clean1 = undistort_pixels(raw1, seeds[c1])
        q0 = to_common(clean0, sizes[c0], args.common_diagonal)
        q1 = to_common(clean1, sizes[c1], args.common_diagonal)
        F, inliers = fit_fundamental(
            q0, q1, args.group_ransac_threshold,
            args.ransac_confidence, args.ransac_max_iters
        )
        if F is not None:
            errors = np.abs(signed_sampson(F, q0, q1))
            inliers &= errors <= 2.0 * args.group_ransac_threshold
        if np.count_nonzero(inliers) < args.min_group_inliers:
            excluded.append({
                "frame": record.frame,
                "pair": pair_label(record.pair),
                "raw_count": raw_count,
                "inlier_count": int(np.count_nonzero(inliers)),
                "reason": "per-group MAGSAC inliers too few",
            })
            continue
        raw0, raw1, scores = raw0[inliers], raw1[inliers], scores[inliers]
        clean0, clean1 = clean0[inliers], clean1[inliers]
        selected = balanced_indices(
            raw0, raw1, scores, sizes[c0], sizes[c1], args.max_points_per_group
        )
        raw0, raw1, scores = raw0[selected], raw1[selected], scores[selected]
        clean0, clean1 = clean0[selected], clean1[selected]
        q0 = to_common(clean0, sizes[c0], args.common_diagonal)
        q1 = to_common(clean1, sizes[c1], args.common_diagonal)
        provisional.append(MatchGroup(
            record.frame, record.pair, raw0, raw1, scores, "train", raw_count,
            int(np.count_nonzero(inliers)),
            grid_coverage(raw0, sizes[c0]), grid_coverage(raw1, sizes[c1]),
            homography_ratio(q0, q1, args.homography_threshold),
        ))
    roles = deterministic_roles(
        (group.frame for group in provisional), args.validation_fraction, args.seed
    )
    for group in provisional:
        group.role = roles[group.frame]
    return provisional, excluded


def pair_contains_strict_reference(pair: tuple[str, str]) -> bool:
    return any(camera in STRICT_REFERENCE_CAMERAS for camera in pair)


def shared_geometry_filter(
    groups: list[MatchGroup],
    seeds: dict[str, CameraSeed],
    args: argparse.Namespace,
) -> tuple[list[MatchGroup], list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove points that disagree with a fixed cross-scene F for each pair.

    Per-frame MAGSAC can accept a self-consistent but wrong solution on a
    texture-poor telephoto crop.  A real fixed rig has one F per camera pair,
    so a second cross-scene pass is a strong and appropriate outlier test.
    """
    current = list(groups)
    excluded: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for iteration in range(args.shared_filter_iterations):
        next_groups: list[MatchGroup] = []
        iteration_report: dict[str, Any] = {
            "iteration": iteration + 1,
            "pairs": {},
        }
        pairs = sorted(
            set(group.pair for group in current),
            key=lambda pair: (CAMERA_INDEX[pair[0]], CAMERA_INDEX[pair[1]]),
        )
        for pair in pairs:
            pair_groups = [group for group in current if group.pair == pair]
            fitting = [group for group in pair_groups if group.role == "train"]
            if len(fitting) < 2:
                fitting = pair_groups
            c0, c1 = pair
            q0 = np.concatenate([
                to_common(
                    undistort_pixels(group.points0, seeds[c0]),
                    seeds[c0].size_wh,
                    args.common_diagonal,
                )
                for group in fitting
            ])
            q1 = np.concatenate([
                to_common(
                    undistort_pixels(group.points1, seeds[c1]),
                    seeds[c1].size_wh,
                    args.common_diagonal,
                )
                for group in fitting
            ])
            is_long = pair_contains_strict_reference(pair)
            threshold = (
                args.strict_shared_point_threshold
                if is_long
                else args.shared_point_threshold
            )
            minimum_ratio = (
                args.strict_minimum_shared_inlier_ratio
                if is_long
                else args.minimum_shared_inlier_ratio
            )
            F, fit_inliers = fit_fundamental(
                q0,
                q1,
                threshold,
                args.ransac_confidence,
                args.ransac_max_iters,
            )
            if F is None or np.count_nonzero(fit_inliers) < args.min_group_inliers:
                raise CalibrationError(
                    f"共享F二次清洗失败：{pair_label(pair)}；"
                    "请检查该相机对是否存在足够真实匹配"
                )
            before_points = sum(len(group.points0) for group in pair_groups)
            kept_points = 0
            excluded_groups = 0
            for group in pair_groups:
                corrected0 = undistort_pixels(group.points0, seeds[c0])
                corrected1 = undistort_pixels(group.points1, seeds[c1])
                group_q0 = to_common(
                    corrected0, seeds[c0].size_wh, args.common_diagonal
                )
                group_q1 = to_common(
                    corrected1, seeds[c1].size_wh, args.common_diagonal
                )
                errors = np.abs(signed_sampson(F, group_q0, group_q1))
                mask = np.isfinite(errors) & (errors <= threshold)
                count = int(np.count_nonzero(mask))
                if group.pre_shared_count <= 0:
                    group.pre_shared_count = len(group.points0)
                cumulative_ratio = count / max(group.pre_shared_count, 1)
                group.shared_inlier_count = count
                group.shared_inlier_ratio = cumulative_ratio
                group.shared_error_p50 = (
                    float(np.quantile(errors[np.isfinite(errors)], 0.50))
                    if np.any(np.isfinite(errors))
                    else None
                )
                group.shared_error_p95 = (
                    float(np.quantile(errors[np.isfinite(errors)], 0.95))
                    if np.any(np.isfinite(errors))
                    else None
                )
                if count < args.min_group_inliers or cumulative_ratio < minimum_ratio:
                    excluded_groups += 1
                    excluded.append({
                        "frame": group.frame,
                        "pair": pair_label(pair),
                        "raw_count": group.raw_count,
                        "inlier_count": group.inlier_count,
                        "shared_inlier_count": count,
                        "shared_inlier_ratio": cumulative_ratio,
                        "shared_error_p50": group.shared_error_p50,
                        "shared_error_p95": group.shared_error_p95,
                        "reason": (
                            f"fixed-F second-pass rejection at iteration {iteration + 1}: "
                            f"count={count}, ratio={cumulative_ratio:.3f}, "
                            f"required count>={args.min_group_inliers}, "
                            f"ratio>={minimum_ratio:.3f}"
                        ),
                    })
                    continue
                group.points0 = group.points0[mask]
                group.points1 = group.points1[mask]
                group.scores = group.scores[mask]
                group.coverage0 = grid_coverage(group.points0, seeds[c0].size_wh)
                group.coverage1 = grid_coverage(group.points1, seeds[c1].size_wh)
                filtered_q0 = group_q0[mask]
                filtered_q1 = group_q1[mask]
                group.homography_ratio = homography_ratio(
                    filtered_q0, filtered_q1, args.homography_threshold
                )
                kept_points += count
                next_groups.append(group)
            iteration_report["pairs"][pair_label(pair)] = {
                "strict_reference_mode": is_long,
                "point_threshold_common_px": threshold,
                "minimum_group_inlier_ratio": minimum_ratio,
                "groups_before": len(pair_groups),
                "groups_after": len(pair_groups) - excluded_groups,
                "groups_excluded": excluded_groups,
                "points_before": before_points,
                "points_after": kept_points,
                "points_removed": before_points - kept_points,
                "fit_inlier_count": int(np.count_nonzero(fit_inliers)),
            }
        diagnostics.append(iteration_report)
        current = next_groups
    return current, excluded, diagnostics


def graph_diagnostics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    adjacency: dict[str, set[str]] = {camera: set() for camera in CAMERAS}
    for c0, c1 in pairs:
        adjacency[c0].add(c1)
        adjacency[c1].add(c0)
    seen = {ANCHOR_CAMERA}
    queue = deque([ANCHOR_CAMERA])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    connected = len(seen) == len(CAMERAS)
    cycle_rank = len(pairs) - len(CAMERAS) + 1 if connected else 0
    return {
        "connected": connected,
        "reachable_from_anchor": sorted(seen, key=CAMERA_INDEX.get),
        "edge_count": len(pairs),
        "cycle_rank": int(cycle_rank),
        "pairs": [pair_label(pair) for pair in pairs],
    }


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def ensure_rotation(matrix: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rotation_angle_deg(matrix: np.ndarray) -> float:
    """Return the unsigned angle of a rotation matrix in degrees."""
    R = ensure_rotation(matrix)
    cosine = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def rodrigues_to_matrix(vector: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(vector, dtype=np.float64).reshape(3, 1))[0]


def matrix_to_rodrigues(matrix: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(ensure_rotation(matrix))[0].reshape(3)


def project_essential(E: np.ndarray) -> np.ndarray:
    U, singular, Vt = np.linalg.svd(E)
    if np.linalg.det(U @ Vt) < 0:
        Vt[-1] *= -1
    value = 0.5 * (singular[0] + singular[1])
    return U @ np.diag([value, value, 0.0]) @ Vt


def shared_F_for_pair(
    pair: tuple[str, str],
    groups: list[MatchGroup],
    seeds: dict[str, CameraSeed],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [group for group in groups if group.pair == pair and group.role == "train"]
    if len(selected) < 2:
        selected = [group for group in groups if group.pair == pair]
    if not selected:
        raise CalibrationError(f"没有可用于初始化的 {pair_label(pair)}")
    q0 = np.concatenate([
        to_common(
            undistort_pixels(group.points0, seeds[pair[0]]),
            seeds[pair[0]].size_wh,
            args.common_diagonal,
        )
        for group in selected
    ])
    q1 = np.concatenate([
        to_common(
            undistort_pixels(group.points1, seeds[pair[1]]),
            seeds[pair[1]].size_wh,
            args.common_diagonal,
        )
        for group in selected
    ])
    F, inliers = fit_fundamental(
        q0, q1, args.shared_f_threshold,
        args.ransac_confidence, args.ransac_max_iters
    )
    if F is None or np.count_nonzero(inliers) < args.min_group_inliers:
        raise CalibrationError(f"无法估计共享F：{pair_label(pair)}")
    return F, q0[inliers], q1[inliers]


def initial_poses_from_anchor_pairs(
    groups: list[MatchGroup],
    seeds: dict[str, CameraSeed],
    available_pairs: list[tuple[str, str]],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    rotations = {ANCHOR_CAMERA: np.eye(3)}
    # An essential matrix determines translation only up to sign.  Choosing
    # each sign independently with recoverPose is fragile for a narrow-FOV,
    # distant, nearly planar pair. Keep the sign-free directions first, then
    # select all signs jointly with every available
    # pairwise edges and the multi-camera cycles.
    unsigned_centers: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {}
    for camera in NON_ANCHOR:
        pair = canonical_pair(ANCHOR_CAMERA, camera)
        if pair not in available_pairs:
            raise CalibrationError(
                f"initialization requires direct {ANCHOR_CAMERA}-{camera} matches"
            )
        F, q0, q1 = shared_F_for_pair(pair, groups, seeds, args)
        K0 = np.diag([
            common_scale(seeds[ANCHOR_CAMERA].size_wh, args.common_diagonal),
            common_scale(seeds[ANCHOR_CAMERA].size_wh, args.common_diagonal), 1.0
        ]) @ seeds[ANCHOR_CAMERA].K0
        K1 = np.diag([
            common_scale(seeds[camera].size_wh, args.common_diagonal),
            common_scale(seeds[camera].size_wh, args.common_diagonal), 1.0
        ]) @ seeds[camera].K0
        E = project_essential(K1.T @ F @ K0)
        n0 = cv2.undistortPoints(q0.reshape(-1, 1, 2), K0, None).reshape(-1, 2)
        n1 = cv2.undistortPoints(q1.reshape(-1, 1, 2), K1, None).reshape(-1, 2)

        recover_count = 0
        recover_R: np.ndarray | None = None
        recover_error: str | None = None
        try:
            count, recovered_R, _, _ = cv2.recoverPose(E, n0, n1, np.eye(3))
            recover_count = int(count)
            recover_R = ensure_rotation(recovered_R)
        except cv2.error as exc:
            recover_error = str(exc).splitlines()[0]

        # decomposeEssentialMat always exposes both rotation branches even
        # when the positive-depth vote is inconclusive. A rigid camera/sensor
        # rig has roughly parallel optical axes, so the smaller-angle branch
        # is the physically meaningful initialization.  It is only an initial
        # value: all rotations are subsequently refined by bundle adjustment.
        try:
            R_a, R_b, t_axis = cv2.decomposeEssentialMat(E)
        except cv2.error as exc:
            raise CalibrationError(
                f"essential matrix decomposition failed: {ANCHOR_CAMERA}-{camera}: "
                f"{str(exc).splitlines()[0]}"
            ) from exc
        rotation_candidates = [ensure_rotation(R_a), ensure_rotation(R_b)]
        R = min(rotation_candidates, key=rotation_angle_deg)
        t_axis = np.asarray(t_axis, dtype=np.float64).reshape(3)
        C = -R.T @ t_axis
        norm = float(np.linalg.norm(C))
        if not math.isfinite(norm) or norm <= 1e-9:
            raise CalibrationError(
                f"essential translation is degenerate: {ANCHOR_CAMERA}-{camera}"
            )
        C /= norm
        rotations[camera] = R
        unsigned_centers[camera] = C
        reliable = recover_count >= args.min_group_inliers
        if reliable:
            message = (
                f"  {ANCHOR_CAMERA}-{camera}: recoverPose positive depth={recover_count}/"
                f"{len(q0)}；使用小旋转本质矩阵分支初始化"
            )
        else:
            message = (
                f"  {ANCHOR_CAMERA}-{camera}: recoverPose positive depth={recover_count}/"
                f"{len(q0)} < {args.min_group_inliers}；"
                "回退到小旋转本质矩阵分支"
            )
        print(message)
        details[camera] = {
            "shared_F_common": F,
            "recover_pose_inliers": recover_count,
            "recover_pose_reliable": reliable,
            "recover_pose_error": recover_error,
            "input_inliers": int(len(q0)),
            "rotation_candidate_angles_deg": [
                rotation_angle_deg(candidate) for candidate in rotation_candidates
            ],
            "recover_pose_rotation_angle_deg": (
                rotation_angle_deg(recover_R) if recover_R is not None else None
            ),
            "chosen_rotation_angle_deg": rotation_angle_deg(R),
            "unsigned_center_direction": C,
            "initial_R_anchor_to_camera": R,
        }

    # Evaluate all 2^N translation-sign assignments. The direct anchor-X edge
    # alone cannot distinguish the sign in a planar/low-parallax scene, while
    # the remaining cross-camera edges and cycles can. This is more stable than
    # accepting independent recoverPose cheirality votes.
    best_score = math.inf
    best_signs: dict[str, int] | None = None
    best_centers: dict[str, np.ndarray] | None = None
    candidate_scores: list[dict[str, Any]] = []
    for sign_values in product((-1, 1), repeat=len(NON_ANCHOR)):
        signs = dict(zip(NON_ANCHOR, sign_values))
        candidate_centers = {ANCHOR_CAMERA: np.zeros(3)}
        candidate_centers.update({
            camera: float(signs[camera]) * unsigned_centers[camera]
            for camera in NON_ANCHOR
        })
        candidate_model = RigModel(
            {name: value.copy() for name, value in rotations.items()},
            {name: value.copy() for name, value in candidate_centers.items()},
            {name: seed.K0.copy() for name, seed in seeds.items()},
            {name: seed.dist0.copy() for name, seed in seeds.items()},
        )
        metrics = evaluate_model(candidate_model, groups, seeds, args)
        score = metrics["score"]["train"]
        if score is None:
            score = metrics["score"]["all"]
        numeric_score = float(score) if score is not None else math.inf
        candidate_scores.append({"signs": signs, "train_score": numeric_score})
        if math.isfinite(numeric_score) and numeric_score < best_score:
            best_score = numeric_score
            best_signs = signs
            best_centers = candidate_centers

    if best_centers is None or best_signs is None:
        raise CalibrationError("多相机平移符号联合选择失败")
    centers = best_centers
    for camera in NON_ANCHOR:
        details[camera]["selected_translation_sign"] = best_signs[camera]
        details[camera]["initial_center_direction"] = centers[camera]
    candidate_scores.sort(key=lambda item: item["train_score"])
    details["global_translation_sign_selection"] = {
        "method": f"exhaustive_2^{len(NON_ANCHOR)}_all_pair_cycle_score",
        "candidate_count": len(candidate_scores),
        "selected_signs": best_signs,
        "selected_train_score": best_score,
        "top_candidates": candidate_scores[:4],
    }
    print(
        "  多相机联合平移符号："
        + ", ".join(f"{camera}={best_signs[camera]:+d}" for camera in NON_ANCHOR)
        + f"；训练得分={best_score:.4f}"
    )
    return rotations, centers, details


def distortion_mode_indices(mode: str) -> tuple[int, ...]:
    return {
        "fixed": (),
        "radial1": (0,),
        "radial2": (0, 1),
        "standard": (0, 1, 2, 3, 4),
    }[mode]


def distortion_observability(
    groups: list[MatchGroup],
    seeds: dict[str, CameraSeed],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    """Measure how much of each lens radius is actually constrained.

    Cross-camera matching only observes the common field of view.  In
    particular, the outer part of a wide-FOV image may have no partner in the
    other cameras. Higher-order radial coefficients must not be
    inferred by extrapolating from that central overlap.
    """
    points: dict[str, list[np.ndarray]] = {camera: [] for camera in CAMERAS}
    for group in groups:
        points[group.pair[0]].append(group.points0)
        points[group.pair[1]].append(group.points1)
    result: dict[str, dict[str, Any]] = {}
    requested_modes = {
        camera: (
            args.reference_distortion if camera in REFERENCE_CAMERAS
            else args.target_distortion
        )
        for camera in CAMERAS
    }
    for camera in CAMERAS:
        merged = np.concatenate(points[camera]) if points[camera] else np.empty((0, 2))
        seed = seeds[camera]
        width, height = seed.size_wh
        corners = np.asarray(
            [[0.0, 0.0], [width - 1.0, 0.0],
             [0.0, height - 1.0], [width - 1.0, height - 1.0]],
            dtype=np.float64,
        )

        def radii(values: np.ndarray) -> np.ndarray:
            return np.hypot(
                (values[:, 0] - seed.K0[0, 2]) / seed.K0[0, 0],
                (values[:, 1] - seed.K0[1, 2]) / seed.K0[1, 1],
            )

        corner_radius = max(float(np.max(radii(corners))), 1e-12)
        normalized = radii(merged) / corner_radius if len(merged) else np.empty(0)
        p95 = float(np.quantile(normalized, 0.95)) if len(normalized) else 0.0
        p99 = float(np.quantile(normalized, 0.99)) if len(normalized) else 0.0
        maximum = float(np.max(normalized)) if len(normalized) else 0.0
        requested = requested_modes[camera]
        if requested != "auto":
            resolved = requested
            reason = "explicitly requested"
        elif p99 >= args.radial2_min_radius:
            resolved = "radial2"
            reason = "p99 radius supports k1+k2"
        elif p99 >= args.radial1_min_radius:
            resolved = "radial1"
            reason = "p99 radius supports k1 only"
        else:
            resolved = "fixed"
            reason = "common field of view is too central for radial self-calibration"
        result[camera] = {
            "requested_mode": requested,
            "resolved_mode": resolved,
            "reason": reason,
            "selected_point_count": int(len(merged)),
            "observed_radius_p95_fraction": p95,
            "observed_radius_p99_fraction": p99,
            "observed_radius_max_fraction": maximum,
            "radial1_min_fraction": args.radial1_min_radius,
            "radial2_min_fraction": args.radial2_min_radius,
            "validity": "multicamera_overlap_only",
        }
    return result


def build_layout(
    optimize_pose: bool,
    optimize_reference: bool,
    optimize_target: bool,
    target_pp: bool,
    optimize_distortion: bool,
    args: argparse.Namespace,
) -> ParameterLayout:
    cursor = 0
    rotation: dict[str, slice] = {}
    center: dict[str, slice] = {}
    if optimize_pose:
        for camera in NON_ANCHOR:
            rotation[camera] = slice(cursor, cursor + 3)
            cursor += 3
            center[camera] = slice(cursor, cursor + 3)
            cursor += 3
    reference_log_scale: dict[str, int] = {}
    if optimize_reference:
        for camera in REFERENCE_CAMERAS:
            reference_log_scale[camera] = cursor
            cursor += 1
    target_log_scale = None
    target_pp_slice = None
    if optimize_target:
        target_log_scale = cursor
        cursor += 1
        if target_pp:
            target_pp_slice = slice(cursor, cursor + 2)
            cursor += 2
    distortion: dict[str, dict[int, int]] = {}
    if optimize_distortion:
        for camera in CAMERAS:
            mode = args.resolved_distortion_modes[camera]
            mapping: dict[int, int] = {}
            for coefficient in distortion_mode_indices(mode):
                mapping[coefficient] = cursor
                cursor += 1
            if mapping:
                distortion[camera] = mapping
    return ParameterLayout(
        rotation, center, reference_log_scale, target_log_scale,
        target_pp_slice, distortion, cursor
    )


def pack_model(
    model: RigModel,
    initial_rotations: dict[str, np.ndarray],
    seeds: dict[str, CameraSeed],
    layout: ParameterLayout,
) -> np.ndarray:
    x = np.zeros(layout.size, dtype=np.float64)
    for camera in NON_ANCHOR:
        if camera not in layout.rotation:
            continue
        delta = model.rotations[camera] @ initial_rotations[camera].T
        x[layout.rotation[camera]] = matrix_to_rodrigues(delta)
        x[layout.center[camera]] = model.centers[camera]
    for camera, index in layout.reference_log_scale.items():
        scale = math.sqrt(
            model.intrinsics[camera][0, 0] * model.intrinsics[camera][1, 1]
            / (seeds[camera].K0[0, 0] * seeds[camera].K0[1, 1])
        )
        x[index] = math.log(scale)
    if layout.target_log_scale is not None:
        camera = TARGET_CAMERA
        scale = math.sqrt(
            model.intrinsics[camera][0, 0] * model.intrinsics[camera][1, 1]
            / (seeds[camera].K0[0, 0] * seeds[camera].K0[1, 1])
        )
        x[layout.target_log_scale] = math.log(scale)
    if layout.target_pp is not None:
        seed = seeds[TARGET_CAMERA]
        x[layout.target_pp] = [
            (model.intrinsics[TARGET_CAMERA][0, 2] - seed.K0[0, 2]) / seed.size_wh[0],
            (model.intrinsics[TARGET_CAMERA][1, 2] - seed.K0[1, 2]) / seed.size_wh[1],
        ]
    for camera, mapping in layout.distortion.items():
        for coefficient, index in mapping.items():
            x[index] = (
                model.distortion[camera][coefficient]
                - seeds[camera].dist0[coefficient]
            )
    return x


def unpack_model(
    x: np.ndarray,
    initial_rotations: dict[str, np.ndarray],
    seeds: dict[str, CameraSeed],
    layout: ParameterLayout,
    base_model: RigModel,
) -> RigModel:
    rotations = {
        name: value.copy() for name, value in base_model.rotations.items()
    }
    centers = {
        name: value.copy() for name, value in base_model.centers.items()
    }
    for camera in NON_ANCHOR:
        if camera not in layout.rotation:
            continue
        rotations[camera] = ensure_rotation(
            rodrigues_to_matrix(x[layout.rotation[camera]]) @ initial_rotations[camera]
        )
        centers[camera] = np.asarray(x[layout.center[camera]], dtype=np.float64)
    intrinsics = {
        name: value.copy() for name, value in base_model.intrinsics.items()
    }
    for camera, index in layout.reference_log_scale.items():
        scale = math.exp(float(x[index]))
        intrinsics[camera][0, 0] *= scale
        intrinsics[camera][1, 1] *= scale
    if layout.target_log_scale is not None:
        scale = math.exp(float(x[layout.target_log_scale]))
        intrinsics[TARGET_CAMERA][0, 0] *= scale
        intrinsics[TARGET_CAMERA][1, 1] *= scale
    if layout.target_pp is not None:
        dx, dy = x[layout.target_pp]
        intrinsics[TARGET_CAMERA][0, 2] += dx * seeds[TARGET_CAMERA].size_wh[0]
        intrinsics[TARGET_CAMERA][1, 2] += dy * seeds[TARGET_CAMERA].size_wh[1]
    distortion = {
        name: value.copy() for name, value in base_model.distortion.items()
    }
    for camera, mapping in layout.distortion.items():
        for coefficient, index in mapping.items():
            distortion[camera][coefficient] += float(x[index])
    return RigModel(rotations, centers, intrinsics, distortion)


def parameter_bounds(
    layout: ParameterLayout,
    seeds: dict[str, CameraSeed],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    lower = np.full(layout.size, -np.inf, dtype=np.float64)
    upper = np.full(layout.size, np.inf, dtype=np.float64)
    names = [f"x[{index}]" for index in range(layout.size)]
    rotation_bound = math.radians(args.rotation_bound_deg)
    for camera in NON_ANCHOR:
        if camera not in layout.rotation:
            continue
        lower[layout.rotation[camera]] = -rotation_bound
        upper[layout.rotation[camera]] = rotation_bound
        lower[layout.center[camera]] = -args.center_bound
        upper[layout.center[camera]] = args.center_bound
        for axis, label in enumerate("xyz"):
            names[layout.rotation[camera].start + axis] = f"{camera}.rotation_{label}"
            names[layout.center[camera].start + axis] = f"{camera}.center_{label}"
    for camera, index in layout.reference_log_scale.items():
        lower[index] = math.log(args.reference_focal_min)
        upper[index] = math.log(args.reference_focal_max)
        names[index] = f"{camera}.focal_scale"
    if layout.target_log_scale is not None:
        index = layout.target_log_scale
        lower[index] = math.log(args.target_focal_min)
        upper[index] = math.log(args.target_focal_max)
        names[index] = f"{TARGET_CAMERA}.focal_scale"
    if layout.target_pp is not None:
        lower[layout.target_pp] = -args.target_pp_bound_fraction
        upper[layout.target_pp] = args.target_pp_bound_fraction
        names[layout.target_pp.start] = f"{TARGET_CAMERA}.cx_fraction"
        names[layout.target_pp.start + 1] = f"{TARGET_CAMERA}.cy_fraction"
    coefficient_bounds = (
        args.k1_bound,
        args.k2_bound,
        args.tangential_bound,
        args.tangential_bound,
        args.k3_bound,
    )
    for camera, mapping in layout.distortion.items():
        for coefficient, index in mapping.items():
            bound = coefficient_bounds[coefficient]
            seed_value = float(seeds[camera].dist0[coefficient])
            lower[index] = -bound - seed_value
            upper[index] = bound - seed_value
            names[index] = f"{camera}.{DISTORTION_NAMES[coefficient]}"
    return lower, upper, names


def model_F_common(
    model: RigModel,
    pair: tuple[str, str],
    seeds: dict[str, CameraSeed],
    diagonal: float,
) -> np.ndarray:
    c0, c1 = pair
    R0, R1 = model.rotations[c0], model.rotations[c1]
    C0, C1 = model.centers[c0], model.centers[c1]
    relative_R = R1 @ R0.T
    relative_t = R1 @ (C0 - C1)
    translation_norm = float(np.linalg.norm(relative_t))
    if translation_norm <= 1e-8:
        return np.full((3, 3), np.nan, dtype=np.float64)
    # F is invariant to translation scale.  Normalizing improves conditioning
    # and leaves baseline magnitudes to the multi-edge centre geometry/prior.
    relative_t = relative_t / translation_norm
    E = skew(relative_t) @ relative_R
    K0 = np.diag([
        common_scale(seeds[c0].size_wh, diagonal),
        common_scale(seeds[c0].size_wh, diagonal), 1.0
    ]) @ model.intrinsics[c0]
    K1 = np.diag([
        common_scale(seeds[c1].size_wh, diagonal),
        common_scale(seeds[c1].size_wh, diagonal), 1.0
    ]) @ model.intrinsics[c1]
    F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)
    norm = float(np.linalg.norm(F))
    if not math.isfinite(norm) or norm <= 1e-15:
        return np.full((3, 3), np.nan, dtype=np.float64)
    return F / norm


def group_error(
    model: RigModel,
    group: MatchGroup,
    seeds: dict[str, CameraSeed],
    diagonal: float,
) -> np.ndarray:
    F = model_F_common(model, group.pair, seeds, diagonal)
    if not np.isfinite(F).all():
        return np.full(len(group.points0), 100.0, dtype=np.float64)
    c0, c1 = group.pair
    corrected0 = cv2.undistortPoints(
        group.points0.reshape(-1, 1, 2),
        model.intrinsics[c0],
        model.distortion[c0],
        P=model.intrinsics[c0],
    ).reshape(-1, 2)
    corrected1 = cv2.undistortPoints(
        group.points1.reshape(-1, 1, 2),
        model.intrinsics[c1],
        model.distortion[c1],
        P=model.intrinsics[c1],
    ).reshape(-1, 2)
    q0 = to_common(corrected0, seeds[c0].size_wh, diagonal)
    q1 = to_common(corrected1, seeds[c1].size_wh, diagonal)
    return signed_sampson(F, q0, q1)


def robustify_data_residuals(
    residuals: np.ndarray,
    loss: str,
    f_scale: float,
) -> np.ndarray:
    """Encode a robust data cost as residuals for a linear least-squares call.

    SciPy applies ``loss=`` to every residual.  Our residual vector also
    contains physical/K/distortion priors, which must remain quadratic: a
    Cauchy loss would otherwise saturate a violated baseline prior and let a
    weak long-focus edge drift arbitrarily far.  For data residual r, return
    r_eff satisfying r_eff**2 = C**2 * rho((r/C)**2).  The optimizer can then
    use a linear loss globally while only image-match errors are robustified.
    """
    values = np.asarray(residuals, dtype=np.float64)
    if loss == "linear":
        return values
    z = np.square(values / f_scale)
    if loss == "soft_l1":
        rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
    elif loss == "huber":
        rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
    elif loss == "cauchy":
        rho = np.log1p(z)
    else:  # Kept defensive even though argparse restricts the choices.
        raise CalibrationError(f"不支持的鲁棒损失：{loss}")
    magnitude = f_scale * np.sqrt(np.maximum(rho, 0.0))
    return np.copysign(magnitude, values)


def make_residual_function(
    groups: list[MatchGroup],
    initial_rotations: dict[str, np.ndarray],
    initial_centers: dict[str, np.ndarray],
    seeds: dict[str, CameraSeed],
    layout: ParameterLayout,
    base_model: RigModel,
    args: argparse.Namespace,
):
    train = [group for group in groups if group.role == "train"]
    pair_group_counts = Counter(group.pair for group in train)
    mean_pair_groups = float(np.mean(list(pair_group_counts.values())))
    initial_directions = {
        camera: initial_centers[camera] / np.linalg.norm(initial_centers[camera])
        for camera in NON_ANCHOR
    }
    prior_scale = math.sqrt(args.prior_equivalent_points)

    def residual(x: np.ndarray) -> np.ndarray:
        model = unpack_model(x, initial_rotations, seeds, layout, base_model)
        values: list[np.ndarray] = []
        for group in train:
            error = group_error(model, group, seeds, args.common_diagonal)
            group_weight = math.sqrt(args.max_points_per_group / max(len(error), 1))
            pair_weight = math.sqrt(mean_pair_groups / pair_group_counts[group.pair])
            weighted_error = error * group_weight * pair_weight
            values.append(
                robustify_data_residuals(
                    weighted_error, args.loss, args.loss_scale
                )
            )
        priors: list[float] = []
        rotation_sigma = math.radians(args.rotation_prior_deg)
        direction_sigma = math.radians(args.direction_prior_deg)
        for camera in NON_ANCHOR:
            if camera not in layout.rotation:
                continue
            priors.extend(
                (x[layout.rotation[camera]] / rotation_sigma * prior_scale).tolist()
            )
            center = model.centers[camera]
            norm = max(float(np.linalg.norm(center)), 1e-9)
            unit = center / norm
            priors.extend(
                ((unit - initial_directions[camera]) / direction_sigma * prior_scale).tolist()
            )
            priors.append(
                math.log(norm) / args.baseline_log_sigma * prior_scale
            )
        if layout.center:
            scale_norm = max(
                float(np.linalg.norm(model.centers[SCALE_REFERENCE_CAMERA])), 1e-9
            )
            # Epipolar geometry has one unavoidable global translation-scale
            # gauge.  Keep this numerical gauge even when all physical priors
            # are disabled with --prior-equivalent-points 0.
            priors.append(
                math.log(scale_norm)
                / args.gauge_log_sigma
                * max(prior_scale, 1.0)
            )
        for camera, index in layout.reference_log_scale.items():
            priors.append(
                x[index] / args.reference_focal_prior_sigma * prior_scale
            )
        if layout.target_log_scale is not None and args.target_focal_prior_sigma > 0:
            priors.append(
                x[layout.target_log_scale]
                / args.target_focal_prior_sigma
                * prior_scale
            )
        if layout.target_pp is not None:
            priors.extend(
                (
                    x[layout.target_pp]
                    / args.target_pp_prior_fraction
                    * prior_scale
                ).tolist()
            )
        coefficient_sigma = (
            args.k1_prior_sigma,
            args.k2_prior_sigma,
            args.tangential_prior_sigma,
            args.tangential_prior_sigma,
            args.k3_prior_sigma,
        )
        for camera, mapping in layout.distortion.items():
            for coefficient, index in mapping.items():
                priors.append(
                    x[index] / coefficient_sigma[coefficient] * prior_scale
                )
        values.append(np.asarray(priors, dtype=np.float64))
        return np.concatenate(values)

    return residual


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = np.abs(array[np.isfinite(array)])
    if not len(array):
        return {"p50": None, "p90": None, "p95": None, "rms": None, "n": 0}
    q = np.quantile(array, [0.5, 0.9, 0.95])
    return {
        "p50": float(q[0]), "p90": float(q[1]), "p95": float(q[2]),
        "rms": float(np.sqrt(np.mean(array * array))), "n": int(len(array))
    }


def evaluate_model(
    model: RigModel,
    groups: list[MatchGroup],
    seeds: dict[str, CameraSeed],
    args: argparse.Namespace,
) -> dict[str, Any]:
    pair_metrics: dict[str, Any] = {}
    role_scores: dict[str, float | None] = {}
    for role in ("train", "validation", "all"):
        score_values: list[float] = []
        for pair in sorted(set(group.pair for group in groups), key=lambda p: (CAMERA_INDEX[p[0]], CAMERA_INDEX[p[1]])):
            selected = [
                group for group in groups
                if group.pair == pair and (role == "all" or group.role == role)
            ]
            errors = np.concatenate([
                group_error(model, group, seeds, args.common_diagonal)
                for group in selected
            ]) if selected else np.empty(0)
            metrics = quantiles(errors)
            pair_metrics.setdefault(pair_label(pair), {})[role] = metrics
            if metrics["p50"] is not None and metrics["p95"] is not None:
                score_values.append(float(metrics["p50"] + 0.25 * metrics["p95"]))
        role_scores[role] = float(np.mean(score_values)) if score_values else None
    return {"score": role_scores, "pairs": pair_metrics}


def physical_metrics(model: RigModel) -> dict[str, Any]:
    baselines = {
        camera: float(np.linalg.norm(model.centers[camera])) for camera in NON_ANCHOR
    }
    positive = [value for value in baselines.values() if value > 1e-9]
    ratio = max(positive) / min(positive) if positive else math.inf
    pairwise = {
        f"{a}-{b}": float(np.linalg.norm(model.centers[a] - model.centers[b]))
        for a, b in combinations(CAMERAS, 2)
    }
    return {
        "anchor_to_camera_baselines": baselines,
        "anchor_baseline_max_min_ratio": float(ratio),
        "all_pairwise_center_distances": pairwise,
    }


def intrinsic_changes(model: RigModel, seeds: dict[str, CameraSeed]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for camera in CAMERAS:
        K, K0 = model.intrinsics[camera], seeds[camera].K0
        result[camera] = {
            "fx_scale": float(K[0, 0] / K0[0, 0]),
            "fy_scale": float(K[1, 1] / K0[1, 1]),
            "cx_delta_px": float(K[0, 2] - K0[0, 2]),
            "cy_delta_px": float(K[1, 2] - K0[1, 2]),
            "dist": model.distortion[camera],
            "dist_delta": model.distortion[camera] - seeds[camera].dist0,
        }
    return result


def distortion_shift_metrics(model: RigModel, seeds: dict[str, CameraSeed]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for camera in CAMERAS:
        width, height = seeds[camera].size_wh
        xs = np.linspace(0.0, width - 1.0, 9)
        ys = np.linspace(0.0, height - 1.0, 7)
        grid = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)
        corrected = cv2.undistortPoints(
            grid.reshape(-1, 1, 2),
            model.intrinsics[camera],
            model.distortion[camera],
            P=model.intrinsics[camera],
        ).reshape(-1, 2)
        seed_corrected = cv2.undistortPoints(
            grid.reshape(-1, 1, 2),
            seeds[camera].K0,
            seeds[camera].dist0,
            P=seeds[camera].K0,
        ).reshape(-1, 2)
        shifts = np.linalg.norm(corrected - grid, axis=1)
        changes = np.linalg.norm(corrected - seed_corrected, axis=1)
        result[camera] = {
            "absolute_correction_max_px": float(np.max(shifts)),
            "absolute_correction_p95_px": float(np.quantile(shifts, 0.95)),
            "change_from_seed_max_px": float(np.max(changes)),
            "change_from_seed_p95_px": float(np.quantile(changes, 0.95)),
            "change_from_seed_median_px": float(np.median(changes)),
        }
    return result


def stage_gate(
    name: str,
    model: RigModel,
    metrics: dict[str, Any],
    physical: dict[str, Any],
    bound_hits: list[str],
    reference_score: float,
    graph: dict[str, Any],
    seeds: dict[str, CameraSeed],
    optimize_distortion: bool,
    args: argparse.Namespace,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if physical["anchor_baseline_max_min_ratio"] > args.maximum_baseline_ratio:
        reasons.append(
            f"anchor baseline max/min={physical['anchor_baseline_max_min_ratio']:.2f}, "
            f"超过{args.maximum_baseline_ratio:.2f}"
        )
    if bound_hits:
        reasons.append("参数触及边界：" + "、".join(bound_hits))
    distortion_shifts = distortion_shift_metrics(model, seeds)
    for camera, values in distortion_shifts.items():
        limit = (
            args.maximum_reference_distortion_shift
            if camera in REFERENCE_CAMERAS
            else args.maximum_target_distortion_shift
        )
        if values["change_from_seed_max_px"] > limit:
            reasons.append(
                f"{camera}畸变相对初值最大改变量="
                f"{values['change_from_seed_max_px']:.1f}px，"
                f"超过{limit:.1f}px"
            )
    score = metrics["score"]["validation"]
    if score is None:
        score = metrics["score"]["train"]
    if score is None or score > reference_score * (1.0 + args.maximum_validation_worsening):
        reasons.append("留出综合误差相对初始化恶化过多")
    if (
        optimize_distortion
        and score is not None
        and score > reference_score
        * (1.0 - args.minimum_distortion_validation_improvement)
    ):
        reasons.append(
            "畸变自由度未在留出场景产生足够提升："
            f"要求至少{100.0 * args.minimum_distortion_validation_improvement:.1f}%"
        )
    for label, pair_metrics in metrics["pairs"].items():
        values = pair_metrics["validation"]
        if values["p95"] is None:
            values = pair_metrics["train"]
        if values["p95"] is None or values["p95"] > args.maximum_pair_p95:
            reasons.append(f"{label} p95超过{args.maximum_pair_p95:.2f}px")
    if graph["cycle_rank"] < args.minimum_cycle_rank:
        reasons.append(
            f"匹配图独立环数仅{graph['cycle_rank']}，小于{args.minimum_cycle_rank}"
        )
    return not reasons, reasons


def optimize_stage(
    name: str,
    start_model: RigModel,
    groups: list[MatchGroup],
    initial_rotations: dict[str, np.ndarray],
    initial_centers: dict[str, np.ndarray],
    seeds: dict[str, CameraSeed],
    graph: dict[str, Any],
    reference_score: float,
    optimize_pose: bool,
    optimize_reference: bool,
    optimize_target: bool,
    optimize_distortion: bool,
    args: argparse.Namespace,
) -> StageResult:
    reference_tight = optimize_reference and args.reference_intrinsics == "tight"
    target_pp = optimize_target and args.target_model == "focal-pp"
    layout = build_layout(
        optimize_pose, reference_tight, optimize_target, target_pp,
        optimize_distortion, args
    )
    x0 = pack_model(start_model, initial_rotations, seeds, layout)
    lower, upper, names = parameter_bounds(layout, seeds, args)
    residual = make_residual_function(
        groups, initial_rotations, initial_centers, seeds, layout, start_model, args
    )
    print(f"\n优化阶段：{name}；参数={layout.size}；训练组={sum(g.role == 'train' for g in groups)}")
    result = least_squares(
        residual, x0, bounds=(lower, upper), method="trf", jac="2-point",
        # Match residuals are robustified inside make_residual_function;
        # priors deliberately remain quadratic and must not be passed through
        # SciPy's global robust loss.
        loss="linear", f_scale=1.0, x_scale="jac",
        max_nfev=args.max_nfev, verbose=1
    )
    model = unpack_model(result.x, initial_rotations, seeds, layout, start_model)
    metrics = evaluate_model(model, groups, seeds, args)
    physical = physical_metrics(model)
    span = upper - lower
    finite = np.isfinite(span) & (span > 0)
    near_lower = finite & ((result.x - lower) / np.where(finite, span, 1.0) < 0.005)
    near_upper = finite & ((upper - result.x) / np.where(finite, span, 1.0) < 0.005)
    bound_hits = [names[index] for index in np.flatnonzero(near_lower | near_upper)]
    passed, reasons = stage_gate(
        name, model, metrics, physical, bound_hits, reference_score, graph,
        seeds, optimize_distortion, args
    )
    natural_gate_passed = passed
    natural_gate_reasons = list(reasons)
    if args.force_accept:
        passed = True
        reasons = [
            "实验性强制接受：忽略留出误差、参数边界和物理安全门"
        ] + [f"已忽略：{reason}" for reason in natural_gate_reasons]
    singular_values = np.linalg.svd(np.asarray(result.jac), compute_uv=False)
    positive_singular = singular_values[singular_values > 1e-12]
    jacobian_condition = (
        float(positive_singular[0] / positive_singular[-1])
        if len(positive_singular) else math.inf
    )
    return StageResult(
        name, model,
        {
            "success": bool(result.success), "status": int(result.status),
            "message": str(result.message), "nfev": int(result.nfev),
            "cost": float(result.cost), "optimality": float(result.optimality),
            "jacobian_condition_number": jacobian_condition,
            "parameter_count": int(layout.size),
            "optimize_pose": optimize_pose,
            "optimize_reference_K": reference_tight,
            "optimize_target_K": optimize_target,
            "optimize_distortion": optimize_distortion,
            "data_loss": args.loss,
            "data_loss_scale": args.loss_scale,
            "prior_loss": "linear_quadratic",
            "force_accept": args.force_accept,
            "natural_gate_passed": natural_gate_passed,
            "natural_gate_reasons": natural_gate_reasons,
        },
        metrics, physical, bound_hits, passed, reasons
    )


def initial_model(
    rotations: dict[str, np.ndarray],
    centers: dict[str, np.ndarray],
    seeds: dict[str, CameraSeed],
) -> RigModel:
    return RigModel(
        {name: value.copy() for name, value in rotations.items()},
        {name: value.copy() for name, value in centers.items()},
        {name: seed.K0.copy() for name, seed in seeds.items()},
        {name: seed.dist0.copy() for name, seed in seeds.items()},
    )


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    print(title)
    for label, values in metrics["pairs"].items():
        metric = values["validation"]
        if metric["p50"] is None:
            metric = values["train"]
        print(
            f"  {label:24s} p50={metric['p50']:.3f} "
            f"p95={metric['p95']:.3f} RMS={metric['rms']:.3f} n={metric['n']}"
        )
    score = metrics["score"]["validation"]
    if score is None:
        score = metrics["score"]["train"]
    print(f"  综合得分={score:.4f}")


def write_group_csv(path: Path, groups: list[MatchGroup]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "frame", "pair", "role", "raw_count", "inlier_count",
            "pre_shared_count", "shared_inlier_count", "shared_inlier_ratio",
            "shared_error_p50", "shared_error_p95", "selected_count",
            "coverage0", "coverage1", "homography_ratio"
        ))
        writer.writeheader()
        for group in sorted(groups, key=lambda g: (g.frame, g.pair)):
            writer.writerow({
                "frame": group.frame, "pair": pair_label(group.pair),
                "role": group.role, "raw_count": group.raw_count,
                "inlier_count": group.inlier_count,
                "pre_shared_count": group.pre_shared_count,
                "shared_inlier_count": group.shared_inlier_count,
                "shared_inlier_ratio": group.shared_inlier_ratio,
                "shared_error_p50": group.shared_error_p50,
                "shared_error_p95": group.shared_error_p95,
                "selected_count": len(group.points0),
                "coverage0": group.coverage0, "coverage1": group.coverage1,
                "homography_ratio": group.homography_ratio,
            })


def scaled_output_model(model: RigModel, scale_baseline_mm: float | None) -> tuple[RigModel, str, float]:
    scale_norm = float(np.linalg.norm(model.centers[SCALE_REFERENCE_CAMERA]))
    if scale_norm <= 1e-12:
        raise CalibrationError(
            f"scale baseline {ANCHOR_CAMERA}-{SCALE_REFERENCE_CAMERA} is degenerate"
        )
    if scale_baseline_mm is None:
        scale = 1.0 / scale_norm
        unit = f"{ANCHOR_CAMERA}_to_{SCALE_REFERENCE_CAMERA}_baseline_normalized_to_1"
    else:
        scale = scale_baseline_mm / scale_norm
        unit = "mm"
    return RigModel(
        model.rotations,
        {name: center * scale for name, center in model.centers.items()},
        model.intrinsics,
        model.distortion,
    ), unit, scale


def calibration_output(
    model: RigModel,
    seeds: dict[str, CameraSeed],
    selected_stage: str,
    accepted: bool,
    unit: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cameras: dict[str, Any] = {}
    poses: dict[str, Any] = {
        ANCHOR_CAMERA: {
            "R": np.eye(3), "T": np.zeros(3),
            "camera_center_in_anchor": np.zeros(3)
        }
    }
    centers: dict[str, Any] = {ANCHOR_CAMERA: np.zeros(3)}
    for camera in CAMERAS:
        cameras[camera] = {
            "model": "standard",
            "image_size": list(seeds[camera].size_wh),
            "K": model.intrinsics[camera],
            "dist": model.distortion[camera].reshape(1, -1),
            "intrinsic_policy": (
                "fixed_known_reference" if camera in REFERENCE_CAMERAS and args.reference_intrinsics == "fixed"
                else "tight_known_reference_prior" if camera in REFERENCE_CAMERAS
                else args.target_model
            ),
            "distortion_policy": (
                args.resolved_distortion_modes[camera]
            ),
            "distortion_validity": "multicamera_overlap_only",
        }
    for camera in NON_ANCHOR:
        R = model.rotations[camera]
        C = model.centers[camera]
        T = -R @ C
        poses[f"{ANCHOR_CAMERA}_to_{camera}"] = {
            "R": R, "T": T, "camera_center_in_anchor": C,
            "baseline_from_anchor": float(np.linalg.norm(C)),
            "convention": "X_camera = R @ X_anchor + T; C_anchor = -R.T @ T",
        }
        centers[camera] = C
    return {
        "schema_version": 1,
        "task": "fixed multi-camera joint epipolar calibration",
        "accepted_for_use": accepted,
        "forced_output": args.force_accept,
        "selected_stage": selected_stage,
        "translation_unit": unit,
        "absolute_scale_observable": args.scale_baseline_mm is not None,
        "reference_cameras": list(REFERENCE_CAMERAS),
        "target_camera": TARGET_CAMERA,
        "anchor_camera": ANCHOR_CAMERA,
        "scale_reference_camera": SCALE_REFERENCE_CAMERA,
        "coordinate_convention": (
            f"{ANCHOR_CAMERA} is world; X_camera = R_anchor_to_camera @ X_anchor + T_anchor_to_camera"
        ),
        "cameras": cameras,
        "camera_poses": poses,
        "camera_centers_in_anchor": centers,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Joint natural-scene calibration for 2+ known-intrinsics references "
            "and one unknown-intrinsics target"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=PROGRAM_VERSION)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--initial-calibration", type=Path, required=True)
    parser.add_argument("--reference-cameras", nargs="+", required=True)
    parser.add_argument("--target-camera", required=True)
    parser.add_argument("--anchor-camera", required=True)
    parser.add_argument("--scale-reference-camera", required=True)
    parser.add_argument("--strict-reference-cameras", nargs="*", default=[])
    parser.add_argument(
        "--target-seed-calibration",
        type=Path,
        default=None,
        help="optional calibration file that overrides only the target K/dist seed",
    )
    parser.add_argument("--output", type=Path, default=Path("multialign_joint_calibration"))
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=None,
        help="匹配诊断生成的reports/per_group.csv；默认自动寻找",
    )
    parser.add_argument(
        "--ignore-quality-report",
        action="store_true",
        help="忽略匹配诊断中的逐组排除结果（仅用于排错）",
    )
    parser.add_argument("--pairs", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--reference-intrinsics", choices=("fixed", "tight"), default="fixed",
        help="fixed locks known K; tight allows a small shared focal scale change"
    )
    parser.add_argument(
        "--target-model", choices=("fixed", "focal", "focal-pp"),
        default="focal-pp"
    )
    parser.add_argument(
        "--reference-distortion",
        choices=("auto", "fixed", "radial1", "radial2", "standard"),
        default="auto",
        help="参考相机畸变优化自由度；auto按实际重叠半径逐相机降阶"
    )
    parser.add_argument(
        "--target-distortion",
        choices=("auto", "fixed", "radial1", "radial2", "standard"),
        default="auto",
        help="目标相机畸变优化自由度；auto按匹配覆盖选择fixed/radial1/radial2"
    )
    parser.add_argument(
        "--radial1-min-radius", type=float, default=0.45,
        help="auto模式放开k1所需的观测半径p99/全图角点半径"
    )
    parser.add_argument(
        "--radial2-min-radius", type=float, default=0.75,
        help="auto模式放开k2所需的观测半径p99/全图角点半径"
    )
    parser.add_argument("--reference-focal-min", type=float, default=0.94)
    parser.add_argument("--reference-focal-max", type=float, default=1.06)
    parser.add_argument("--reference-focal-prior-sigma", type=float, default=0.02)
    parser.add_argument("--target-focal-min", type=float, default=0.55)
    parser.add_argument("--target-focal-max", type=float, default=1.60)
    parser.add_argument("--target-focal-prior-sigma", type=float, default=0.45)
    parser.add_argument("--target-pp-bound-fraction", type=float, default=0.08)
    parser.add_argument("--target-pp-prior-fraction", type=float, default=0.03)
    parser.add_argument("--k1-bound", type=float, default=0.50)
    parser.add_argument("--k2-bound", type=float, default=1.00)
    parser.add_argument("--tangential-bound", type=float, default=0.05)
    parser.add_argument("--k3-bound", type=float, default=1.00)
    parser.add_argument("--k1-prior-sigma", type=float, default=0.08)
    parser.add_argument("--k2-prior-sigma", type=float, default=0.20)
    parser.add_argument("--tangential-prior-sigma", type=float, default=0.01)
    parser.add_argument("--k3-prior-sigma", type=float, default=0.30)
    parser.add_argument(
        "--maximum-reference-distortion-shift", type=float, default=120.0,
        help="相对输入初值，参考相机畸变映射在全图采样点允许的最大变化（像素）"
    )
    parser.add_argument(
        "--maximum-target-distortion-shift", type=float, default=60.0,
        help="相对输入初值，目标相机畸变映射允许的最大变化（像素）"
    )
    parser.add_argument("--scale-baseline-mm", type=float, default=None)
    parser.add_argument("--baseline-log-sigma", type=float, default=0.35)
    parser.add_argument("--gauge-log-sigma", type=float, default=0.03)
    parser.add_argument("--maximum-baseline-ratio", type=float, default=2.5)
    parser.add_argument("--rotation-prior-deg", type=float, default=12.0)
    parser.add_argument("--direction-prior-deg", type=float, default=35.0)
    parser.add_argument("--rotation-bound-deg", type=float, default=45.0)
    parser.add_argument("--center-bound", type=float, default=3.0)
    parser.add_argument("--prior-equivalent-points", type=float, default=100.0)
    parser.add_argument("--common-diagonal", type=float, default=1000.0)
    parser.add_argument("--group-ransac-threshold", type=float, default=2.5)
    parser.add_argument("--shared-f-threshold", type=float, default=2.5)
    parser.add_argument(
        "--shared-point-threshold",
        type=float,
        default=2.25,
        help="普通相机对跨场景共享F的逐点二次清洗阈值（公共像素）",
    )
    parser.add_argument(
        "--strict-shared-point-threshold",
        type=float,
        default=1.50,
        help="stricter shared-F threshold for --strict-reference-cameras",
    )
    parser.add_argument(
        "--minimum-shared-inlier-ratio",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--strict-minimum-shared-inlier-ratio",
        type=float,
        default=0.20,
        help="strict reference groups must retain at least this ratio and min-group-inliers",
    )
    parser.add_argument("--shared-filter-iterations", type=int, default=2)
    parser.add_argument("--homography-threshold", type=float, default=3.0)
    parser.add_argument("--ransac-confidence", type=float, default=0.999)
    parser.add_argument("--ransac-max-iters", type=int, default=50000)
    parser.add_argument("--min-group-inliers", type=int, default=30)
    parser.add_argument("--max-points-per-group", type=int, default=160)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--minimum-cycle-rank", type=int, default=1)
    parser.add_argument("--maximum-pair-p95", type=float, default=6.0)
    parser.add_argument("--maximum-validation-worsening", type=float, default=0.05)
    parser.add_argument(
        "--minimum-distortion-validation-improvement", type=float, default=0.01,
        help="放开畸变后，留出综合得分相对上一阶段至少要改善的比例"
    )
    parser.add_argument(
        "--loss",
        choices=("linear", "soft_l1", "huber", "cauchy"),
        default="cauchy",
        help="robust loss for joint optimization; cauchy is safest for outlier-prone pairs",
    )
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument(
        "--force-accept",
        action="store_true",
        help=(
            "实验诊断：忽略质量门并强制采用最后阶段；输出仍标记为不可直接使用"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    try:
        configure_rig(
            args.reference_cameras,
            args.target_camera,
            args.anchor_camera,
            args.scale_reference_camera,
            args.strict_reference_cameras,
        )
        if args.pairs is not None:
            args.pairs = [parse_pair(value) for value in args.pairs]
    except (CalibrationError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit必须大于0")
    if not 0.0 <= args.validation_fraction < 0.5:
        parser.error("--validation-fraction必须在[0,0.5)内")
    if not 0.0 <= args.minimum_distortion_validation_improvement < 0.5:
        parser.error("--minimum-distortion-validation-improvement必须在[0,0.5)内")
    if not (0 < args.reference_focal_min < args.reference_focal_max):
        parser.error("参考相机焦距边界无效")
    if not (0 < args.target_focal_min < args.target_focal_max):
        parser.error("目标相机焦距边界无效")
    if args.scale_baseline_mm is not None and args.scale_baseline_mm <= 0:
        parser.error("--scale-baseline-mm must be positive")
    if not 0.0 < args.target_pp_bound_fraction < 0.25:
        parser.error("--target-pp-bound-fraction must be in (0, 0.25)")
    if args.shared_filter_iterations < 1 or args.shared_filter_iterations > 5:
        parser.error("--shared-filter-iterations必须在1到5之间")
    for name in (
        "minimum_shared_inlier_ratio",
        "strict_minimum_shared_inlier_ratio",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')}必须在[0,1]内")
    if args.prior_equivalent_points < 0:
        parser.error("--prior-equivalent-points不能为负")
    if not (0.0 < args.radial1_min_radius < args.radial2_min_radius <= 1.0):
        parser.error("必须满足 0 < --radial1-min-radius < --radial2-min-radius <= 1")
    positive = (
        "k1_bound", "k2_bound", "tangential_bound", "k3_bound",
        "k1_prior_sigma", "k2_prior_sigma", "tangential_prior_sigma",
        "k3_prior_sigma", "maximum_reference_distortion_shift",
        "maximum_target_distortion_shift",
        "shared_point_threshold", "strict_shared_point_threshold",
        "center_bound", "rotation_bound_deg", "maximum_baseline_ratio",
        "loss_scale",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')}必须大于0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        matches_root = resolve_matches_root(args.geometry_root)
        quality_report_path = resolve_quality_report(
            args.geometry_root,
            matches_root,
            args.quality_report,
            args.ignore_quality_report,
        )
        quality_decisions = load_quality_report(quality_report_path)
        output = args.output.expanduser().resolve()
        if output.exists() and any(output.iterdir()) and not args.overwrite:
            raise CalibrationError(f"输出目录非空：{output}；添加 --overwrite")
        output.mkdir(parents=True, exist_ok=True)
        pair_dirs = discover_pair_directories(matches_root)
        selected_pairs = (
            sorted(pair_dirs, key=lambda p: (CAMERA_INDEX[p[0]], CAMERA_INDEX[p[1]]))
            if not args.pairs else list(dict.fromkeys(args.pairs))
        )
        graph = graph_diagnostics(selected_pairs)
        if not graph["connected"]:
            raise CalibrationError(
                "camera-pair graph is disconnected; reachable from anchor: "
                + ", ".join(graph["reachable_from_anchor"])
            )
        print("读取全图匹配：" + ", ".join(pair_label(pair) for pair in selected_pairs))
        raw = load_raw_matches(pair_dirs, selected_pairs, args.limit)
        sizes = modal_sizes(raw)
        seeds, initial_calibration = load_camera_seeds(
            args.initial_calibration.expanduser().resolve(), sizes
        )
        target_override = override_target_seed(
            seeds, args.target_seed_calibration
        )
        if quality_report_path is not None and quality_decisions:
            print(
                f"继承逐组质量门：{quality_report_path}"
                f"（读取{len(quality_decisions)}条判定）"
            )
        elif quality_report_path is not None:
            print(
                f"旧版逐组质量表仅作诊断参考：{quality_report_path}；"
                "本轮使用标定器内置双层清洗。"
            )
        elif not args.ignore_quality_report:
            print("提示：未找到reports/per_group.csv，只使用本程序的双层MAGSAC清洗。")
        if target_override is not None:
            K = seeds[TARGET_CAMERA].K0
            print(
                f"Target K seed override: {target_override['path']}; "
                f"fx={K[0,0]:.3f}, fy={K[1,1]:.3f}, "
                f"cx={K[0,2]:.3f}, cy={K[1,2]:.3f}"
            )
        print("匹配尺寸：" + ", ".join(f"{name}={sizes[name]}" for name in CAMERAS))
        for name in CAMERAS:
            K = seeds[name].K0
            if name in REFERENCE_CAMERAS:
                policy = "固定" if args.reference_intrinsics == "fixed" else "可优化"
            else:
                policy = "固定" if args.target_model == "fixed" else "可优化"
            distortion_policy = (
                args.reference_distortion if name in REFERENCE_CAMERAS
                else args.target_distortion
            )
            print(
                f"  {name:10s} fx={K[0,0]:.3f} fy={K[1,1]:.3f} "
                f"cx={K[0,2]:.3f} cy={K[1,2]:.3f} "
                f"[K={policy}, dist={distortion_policy}]"
            )
        print("逐组MAGSAC清洗与全图均衡采样……")
        groups, excluded = prepare_groups(
            raw, seeds, sizes, args, quality_decisions
        )
        print("Cross-scene shared-F filtering...")
        groups, shared_excluded, shared_filter_report = shared_geometry_filter(
            groups, seeds, args
        )
        excluded.extend(shared_excluded)
        for iteration in shared_filter_report:
            print(f"  迭代{iteration['iteration']}：")
            for label, values in iteration["pairs"].items():
                suffix = " [strict reference]" if values["strict_reference_mode"] else ""
                print(
                    f"    {label:24s} 点 {values['points_before']} -> "
                    f"{values['points_after']}；组 {values['groups_before']} -> "
                    f"{values['groups_after']}{suffix}"
                )
        distortion_observation = distortion_observability(groups, seeds, args)
        args.resolved_distortion_modes = {
            camera: values["resolved_mode"]
            for camera, values in distortion_observation.items()
        }
        coefficient_bounds = (
            args.k1_bound, args.k2_bound, args.tangential_bound,
            args.tangential_bound, args.k3_bound,
        )
        for name, seed in seeds.items():
            mode = args.resolved_distortion_modes[name]
            for coefficient in distortion_mode_indices(mode):
                value = float(seed.dist0[coefficient])
                bound = coefficient_bounds[coefficient]
                if abs(value) >= bound:
                    raise CalibrationError(
                        f"{name}.{DISTORTION_NAMES[coefficient]}初值={value:.6g}超出"
                        f"绝对边界±{bound:.6g}；请先检查畸变模型或调大对应--*-bound"
                    )
        print("畸变可观测性（只对多相机重叠区域负责）：")
        for camera in CAMERAS:
            item = distortion_observation[camera]
            print(
                f"  {camera:10s} radius_p99={item['observed_radius_p99_fraction']:.3f}；"
                f"{item['requested_mode']} -> {item['resolved_mode']}"
            )
        pair_counts = Counter(group.pair for group in groups)
        print(
            f"可用组={len(groups)}；训练={sum(g.role == 'train' for g in groups)}；"
            f"留出={sum(g.role == 'validation' for g in groups)}；排除={len(excluded)}"
        )
        missing = [pair_label(pair) for pair in selected_pairs if pair_counts[pair] < 2]
        if missing:
            raise CalibrationError("以下相机对可用组少于2：" + "、".join(missing))
        print(
            f"匹配图：边={graph['edge_count']}，独立环={graph['cycle_rank']}；"
            "环用于约束相对基线长度"
        )
        rotations, centers, init_details = initial_poses_from_anchor_pairs(
            groups, seeds, selected_pairs, args
        )
        seed_model = initial_model(rotations, centers, seeds)
        seed_metrics = evaluate_model(seed_model, groups, seeds, args)
        seed_physical = physical_metrics(seed_model)
        print_metrics(
            f"\nIndependent {ANCHOR_CAMERA}-X initialization:", seed_metrics
        )
        reference_score = seed_metrics["score"]["validation"]
        if reference_score is None:
            reference_score = seed_metrics["score"]["train"]
        if reference_score is None:
            raise CalibrationError("无法计算初始化得分")

        def score_of(metrics: dict[str, Any]) -> float:
            value = metrics["score"]["validation"]
            if value is None:
                value = metrics["score"]["train"]
            return float(value)

        def show_stage(stage: StageResult, title: str) -> None:
            print_metrics(title, stage.metrics)
            changes = intrinsic_changes(stage.model, seeds)
            print(
                f"  {TARGET_CAMERA} focal scale={changes[TARGET_CAMERA]['fx_scale']:.4f}; "
                f"principal-point delta=({changes[TARGET_CAMERA]['cx_delta_px']:.2f}, "
                f"{changes[TARGET_CAMERA]['cy_delta_px']:.2f})px; "
                f"baseline ratio={stage.physical['anchor_baseline_max_min_ratio']:.3f}; "
                f"Jacobian条件数={stage.optimizer['jacobian_condition_number']:.3e}"
            )
            if stage.optimizer["optimize_distortion"]:
                shifts = distortion_shift_metrics(stage.model, seeds)
                for camera in CAMERAS:
                    dist = stage.model.distortion[camera]
                    print(
                        f"  {camera:10s} k1={dist[0]:+.6f} k2={dist[1]:+.6f}；"
                        f"畸变映射改变量max="
                        f"{shifts[camera]['change_from_seed_max_px']:.2f}px"
                    )
            if args.force_accept:
                natural = stage.optimizer.get("natural_gate_passed", False)
                print(
                    "  质量门=强制通过；自然质量门="
                    + ("通过" if natural else "拒绝")
                )
            else:
                print(f"  质量门={'通过' if stage.gate_passed else '拒绝'}")
            for reason in stage.gate_reasons:
                print("  - " + reason)

        stages: list[StageResult] = []
        pose_stage = optimize_stage(
            "01_pose_fixed_K_dist", seed_model, groups, rotations, centers,
            seeds, graph, reference_score,
            True, False, False, False, args
        )
        stages.append(pose_stage)
        show_stage(pose_stage, "阶段1：固定K和畸变，只优化外参：")
        current = (
            pose_stage.model
            if pose_stage.optimizer.get("success", False)
            else seed_model
        )
        current_metrics = (
            pose_stage.metrics
            if pose_stage.optimizer.get("success", False)
            else seed_metrics
        )

        optimize_reference = args.reference_intrinsics == "tight"
        optimize_target = args.target_model != "fixed"
        optimize_distortion = any(
            mode != "fixed" for mode in args.resolved_distortion_modes.values()
        )
        if optimize_reference or optimize_target:
            intrinsic_stage = optimize_stage(
                "02_pose_and_K_fixed_dist", current, groups, rotations, centers,
                seeds, graph, score_of(current_metrics),
                True, optimize_reference, optimize_target, False, args
            )
            stages.append(intrinsic_stage)
            show_stage(intrinsic_stage, "阶段2：固定畸变，优化外参与允许的K：")
            if intrinsic_stage.gate_passed:
                current = intrinsic_stage.model
                current_metrics = intrinsic_stage.metrics

        if optimize_distortion:
            distortion_stage = optimize_stage(
                "03_distortion_only", current, groups, rotations, centers,
                seeds, graph, score_of(current_metrics),
                False, False, False, True, args
            )
            stages.append(distortion_stage)
            show_stage(distortion_stage, "阶段3：固定K和外参，只优化畸变：")
            if distortion_stage.gate_passed:
                current = distortion_stage.model
                current_metrics = distortion_stage.metrics

            geometry_stage = optimize_stage(
                "04_pose_and_K_after_distortion", current, groups, rotations, centers,
                seeds, graph, score_of(current_metrics),
                True, optimize_reference, optimize_target, False, args
            )
            stages.append(geometry_stage)
            show_stage(geometry_stage, "阶段4：固定畸变，回代优化外参与K：")
            if geometry_stage.gate_passed:
                current = geometry_stage.model
                current_metrics = geometry_stage.metrics

            final_distortion_stage = optimize_stage(
                "05_distortion_after_geometry", current, groups, rotations, centers,
                seeds, graph, score_of(current_metrics),
                False, False, False, True, args
            )
            stages.append(final_distortion_stage)
            show_stage(final_distortion_stage, "阶段5：最终固定几何复核畸变：")

        passed = [stage for stage in stages if stage.gate_passed]
        def selection_score(stage: StageResult) -> float:
            value = stage.metrics["score"]["validation"]
            return float(value if value is not None else stage.metrics["score"]["train"])
        if args.force_accept:
            selected = stages[-1]
            accepted = False
        elif passed:
            selected = min(passed, key=selection_score)
            accepted = True
        else:
            selected = min(stages, key=selection_score)
            accepted = False
        final_model, unit, translation_scale = scaled_output_model(
            selected.model, args.scale_baseline_mm
        )
        calibration = calibration_output(
            final_model, seeds, selected.name, accepted, unit, args
        )
        report = {
            "schema_version": 1,
            "program_version": PROGRAM_VERSION,
            "inputs": {
                "geometry_root": args.geometry_root.resolve(),
                "matches_root": matches_root,
                "initial_calibration": args.initial_calibration.resolve(),
                "target_seed_calibration": (
                    args.target_seed_calibration.resolve()
                    if args.target_seed_calibration is not None else None
                ),
                "quality_report": quality_report_path,
                "pairs": [pair_label(pair) for pair in selected_pairs],
            },
            "policies": {
                "reference_intrinsics": args.reference_intrinsics,
                "target_model": args.target_model,
                "reference_distortion": args.reference_distortion,
                "target_distortion": args.target_distortion,
                "resolved_distortion_modes": args.resolved_distortion_modes,
                "distortion_model": "OpenCV standard [k1,k2,p1,p2,k3]",
                "distortion_objective": (
                    "raw matched pixels are undistorted with each candidate model "
                    "inside every residual evaluation"
                ),
                "baseline": (
                    "soft equal-range prior on anchor-to-camera norms; hard max/min gate"
                ),
                "translation_scale": unit,
                "force_accept": args.force_accept,
            },
            "graph": graph,
            "camera_sizes_wh": sizes,
            "distortion_observability": distortion_observation,
            "initialization": init_details,
            "target_seed_override": target_override,
            "shared_geometry_filter": shared_filter_report,
            "seed_metrics": seed_metrics,
            "seed_physical": seed_physical,
            "stages": {
                stage.name: {
                    "optimizer": stage.optimizer,
                    "metrics": stage.metrics,
                    "physical": stage.physical,
                    "intrinsic_changes": intrinsic_changes(stage.model, seeds),
                    "distortion_shift": distortion_shift_metrics(stage.model, seeds),
                    "bound_hits": stage.bound_hits,
                    "gate_passed": stage.gate_passed,
                    "gate_reasons": stage.gate_reasons,
                }
                for stage in stages
            },
            "selection": {
                "selected_stage": selected.name,
                "accepted_for_use": accepted,
                "forced_output": args.force_accept,
                "translation_output_scale_factor": translation_scale,
                "final_metrics": selected.metrics,
                "final_physical_unscaled": selected.physical,
                "final_intrinsic_changes": intrinsic_changes(selected.model, seeds),
                "final_distortion_shift": distortion_shift_metrics(selected.model, seeds),
            },
            "excluded_groups": excluded,
        }
        calibration_path = output / "rig_calibration.json"
        report_path = output / "optimization_report.json"
        write_json(calibration_path, calibration)
        write_json(report_path, report)
        write_group_csv(output / "groups.csv", groups)
        print("\n最终结果")
        print(f"采用阶段：{selected.name}")
        print(
            "可直接使用："
            + ("是" if accepted else "否（强制诊断输出）" if args.force_accept else "否（仅作诊断/初值）")
        )
        print(f"多相机校准：{calibration_path}")
        print(f"优化报告：{report_path}")
        print(f"逐组表：{output / 'groups.csv'}")
        if args.force_accept:
            print(
                "严重警告：本次使用--force-accept，最终阶段已强制写出；"
                "请依据自然质量门原因和独立场景复核。",
                file=sys.stderr,
            )
        elif not accepted:
            print("警告：没有阶段通过全部几何与物理质量门。", file=sys.stderr)
        return 0 if accepted or args.force_accept else 2
    except KeyboardInterrupt:
        print("用户中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        if args.strict:
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
