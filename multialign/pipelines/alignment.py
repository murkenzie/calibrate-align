#!/usr/bin/env python3
"""Batch-align 2+ calibrated reference cameras to one target camera.

The camera roles are supplied explicitly, so camera names and camera count are
not part of the algorithm.  Each frame is processed in two isolated subprocess
stages: calibrated RoMa/multi-view geometry, followed by metric anchoring of
Depth Anything V2 and target-coordinate rendering.

Outputs are intentionally split into three levels: ``results`` contains the
files most users consume, ``diagnostics`` contains quality-control artifacts,
and the batch root contains machine-readable summaries plus an HTML gallery.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote

import cv2
import numpy as np


PROGRAM_VERSION = "2.1"
REFERENCE_CAMERAS = ("reference_a", "reference_b")
TARGET_CAMERA = "target"
ANCHOR_CAMERA = REFERENCE_CAMERAS[0]
CAMERA_NAMES = (*REFERENCE_CAMERAS, TARGET_CAMERA)
DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


class PipelineError(RuntimeError):
    pass


def configure_rig(
    reference_cameras: Sequence[str], target_camera: str, anchor_camera: str
) -> None:
    global REFERENCE_CAMERAS, TARGET_CAMERA, ANCHOR_CAMERA, CAMERA_NAMES
    references = tuple(reference_cameras)
    if len(references) < 2 or len(set(references)) != len(references):
        raise PipelineError("--reference-cameras requires at least two unique names")
    if target_camera in references:
        raise PipelineError("--target-camera must not also be a reference camera")
    if anchor_camera not in references:
        raise PipelineError("--anchor-camera must be one of --reference-cameras")
    REFERENCE_CAMERAS = references
    TARGET_CAMERA = target_camera
    ANCHOR_CAMERA = anchor_camera
    CAMERA_NAMES = (*references, target_camera)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"无法读取JSON：{path} ({exc})") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"JSON顶层必须是对象：{path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, path)


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    if not path.is_file():
        raise PipelineError(f"缺少图像：{path}")
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise PipelineError(f"OpenCV无法解码：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    params: list[int] = []
    if suffix in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 96]
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise PipelineError(f"OpenCV无法编码：{path}")
    encoded.tofile(str(path))


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise PipelineError(f"不支持的图像形状：{image.shape}")


def resize_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    image = ensure_bgr(image)
    interpolation = cv2.INTER_AREA if image.shape[1] > width else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def labelled_panel(image: np.ndarray, label: str, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    body = resize_panel(image, width, height)
    header_height = max(32, int(round(height * 0.075)))
    panel = cv2.copyMakeBorder(
        body, header_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(22, 22, 22)
    )
    font_scale = max(0.55, min(1.0, width / 700.0))
    cv2.putText(
        panel,
        label,
        (12, int(header_height * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (245, 245, 245),
        max(1, int(round(font_scale * 2))),
        cv2.LINE_AA,
    )
    return panel


def montage_row(items: Iterable[tuple[str, np.ndarray]], size: tuple[int, int]) -> np.ndarray:
    panels = [labelled_panel(image, label, size) for label, image in items]
    if not panels:
        raise PipelineError("无法生成空蒙太奇")
    return cv2.hconcat(panels)


def montage_grid(
    items: Iterable[tuple[str, np.ndarray]],
    size: tuple[int, int],
    maximum_columns: int = 4,
) -> np.ndarray:
    values = list(items)
    if not values:
        raise PipelineError("无法生成空蒙太奇")
    columns = min(maximum_columns, max(1, int(np.ceil(np.sqrt(len(values))))))
    rows: list[np.ndarray] = []
    for start in range(0, len(values), columns):
        chunk = values[start : start + columns]
        while len(chunk) < columns:
            chunk.append(("", placeholder(size, "")))
        rows.append(montage_row(chunk, size))
    return cv2.vconcat(rows)


def placeholder(size: tuple[int, int], text: str = "missing") -> np.ndarray:
    width, height = size
    image = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.putText(
        image,
        text,
        (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 80, 230),
        2,
        cv2.LINE_AA,
    )
    return image


def optional_image(path: Path, size: tuple[int, int]) -> np.ndarray:
    return read_image(path) if path.is_file() else placeholder(size)


def make_frame_visuals(
    diagnostics_dir: Path,
    results_dir: Path,
    reference_cameras: Sequence[str],
    target_camera: str,
    panel_width: int,
) -> dict[str, list[str]]:
    target = read_image(diagnostics_dir / "target_reference.png")
    target_size = (target.shape[1], target.shape[0])
    aligned = {
        name: read_image(diagnostics_dir / f"{name}_aligned.png")
        for name in reference_cameras
    }
    overlays = {
        name: read_image(diagnostics_dir / f"{name}_overlay_50.jpg")
        for name in reference_cameras
    }

    preview_height = max(1, int(round(target_size[1] * panel_width / target_size[0])))
    preview_size = (panel_width, preview_height)
    aligned_grid = montage_grid(
        [(target_camera, target)]
        + [(f"{name} aligned", aligned[name]) for name in reference_cameras],
        preview_size,
    )
    overlay_grid = montage_grid(
        [(target_camera, target)]
        + [(f"{name} 50:50", overlays[name]) for name in reference_cameras],
        preview_size,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    write_image(results_dir / "aligned_grid.png", aligned_grid)
    write_image(results_dir / "overlay_grid.png", overlay_grid)
    write_image(results_dir / "overview.png", cv2.vconcat((aligned_grid, overlay_grid)))

    depth_items = (
        ("model raw depth", diagnostics_dir / "depth_final_model_raw.png"),
        ("render depth", diagnostics_dir / "depth_render_projection.png"),
        ("surface guide", diagnostics_dir / "depth_render_surface_guide.png"),
        ("strict mask", diagnostics_dir / "depth_strict_mask.png"),
        ("geometry anchors", diagnostics_dir / "geometry_anchor_mask.png"),
    )
    debug_depth = montage_grid(
        [(label, optional_image(path, target_size)) for label, path in depth_items],
        preview_size,
    )
    write_image(diagnostics_dir / "debug_depth_grid.png", debug_depth)

    render_kinds = (
        ("surface copy", "aligned_surface_copy.png"),
        ("final", "aligned.png"),
        ("unresolved", "occlusion_unresolved_mask.png"),
        ("structure refined", "texture_structure_refined_mask.png"),
    )
    render_width = max(240, min(panel_width, 360))
    render_height = max(1, int(round(target_size[1] * render_width / target_size[0])))
    debug_items = [
        (
            f"{name} {kind}",
            optional_image(diagnostics_dir / f"{name}_{suffix}", target_size),
        )
        for kind, suffix in render_kinds
        for name in reference_cameras
    ]
    debug_render = montage_grid(
        debug_items,
        (render_width, render_height),
        maximum_columns=min(4, len(reference_cameras)),
    )
    write_image(diagnostics_dir / "debug_render_grid.png", debug_render)
    return {
        "primary": [
            "results/aligned_grid.png",
            "results/overlay_grid.png",
            "results/overview.png",
        ],
        "diagnostics": [
            "diagnostics/debug_depth_grid.png",
            "diagnostics/debug_render_grid.png",
        ],
    }


def promote_primary_results(
    diagnostics_dir: Path,
    results_dir: Path,
    reference_cameras: Sequence[str],
) -> list[str]:
    mapping = {
        "target_reference.png": "target_reference.png",
        "depth_final.png": "target_depth.png",
        "depth_confidence.png": "target_depth_confidence.png",
        "depth_strict_mask.png": "target_depth_strict_mask.png",
        "depth_prior_alignment_maps.npz": "alignment_maps.npz",
    }
    for name in reference_cameras:
        mapping.update(
            {
                f"{name}_aligned.png": f"{name}_aligned.png",
                f"{name}_valid_mask.png": f"{name}_valid_mask.png",
                f"{name}_overlay_50.jpg": f"{name}_overlay_50.jpg",
            }
        )
    missing = [source for source in mapping if not (diagnostics_dir / source).is_file()]
    if missing:
        raise PipelineError("无法整理主结果，缺少：" + ", ".join(missing))
    results_dir.mkdir(parents=True, exist_ok=True)
    for source, destination in mapping.items():
        os.replace(diagnostics_dir / source, results_dir / destination)
    return [f"results/{name}" for name in mapping.values()]


def parse_extensions(values: list[str]) -> set[str]:
    extensions: set[str] = set()
    for value in values:
        suffix = value.lower().strip()
        if not suffix.startswith("."):
            suffix = "." + suffix
        extensions.add(suffix)
    if not extensions:
        raise PipelineError("--extensions不能为空")
    return extensions


def resolve_dataset_root(dataset_root: Path) -> tuple[Path, Path]:
    root = dataset_root.resolve()
    if all((root / name).is_dir() for name in CAMERA_NAMES):
        # Accept either ``rig_prepared_all`` or ``rig_prepared_all/images``.
        # In the latter case retain the parent so split and metadata files are
        # still found automatically.
        parent = root.parent
        if root.name.lower() == "images" and (
            (parent / "splits").is_dir() or (parent / "metadata").is_dir()
        ):
            return parent, root
        return root, root
    images = root / "images"
    if all((images / name).is_dir() for name in CAMERA_NAMES):
        return root, images
    raise PipelineError(
        f"{root}不是有效图像根目录；需要这些相机子目录：{', '.join(CAMERA_NAMES)}"
    )


def load_metadata(dataset_root: Path) -> dict[str, dict[str, str]]:
    path = dataset_root / "metadata" / "dataset.json"
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except PipelineError:
        return {}
    output: dict[str, dict[str, str]] = {}
    for item in value.get("groups", []):
        if not isinstance(item, dict) or "frame" not in item:
            continue
        output[Path(str(item["frame"])).stem] = {
            "scene_id": str(item.get("scene_id", "")),
            "variant": str(item.get("variant", "")),
        }
    return output


def split_stems(dataset_root: Path, split: str) -> set[str] | None:
    if split == "all":
        return None
    path = dataset_root / "splits" / f"{split}.txt"
    if not path.is_file():
        raise PipelineError(f"指定--split {split}，但找不到：{path}")
    stems = {
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not stems:
        raise PipelineError(f"split文件为空：{path}")
    return stems


def discover_frames(
    dataset_root: Path,
    image_root: Path,
    target_camera: str,
    frame_glob: str,
    extensions: set[str],
    split: str,
) -> tuple[list[str], dict[str, list[str]]]:
    target_dir = image_root / target_camera
    allowed_stems = split_stems(dataset_root, split)
    target_frames = sorted(
        path.name
        for path in target_dir.glob(frame_glob)
        if path.is_file()
        and path.suffix.lower() in extensions
        and (allowed_stems is None or path.stem in allowed_stems)
    )
    if not target_frames:
        raise PipelineError(
            f"{target_dir}中没有符合glob={frame_glob!r}, split={split!r}的图像"
        )
    complete: list[str] = []
    missing: dict[str, list[str]] = {}
    for frame in target_frames:
        absent = [name for name in CAMERA_NAMES if not (image_root / name / frame).is_file()]
        if absent:
            missing[frame] = absent
        else:
            complete.append(frame)
    stems: dict[str, list[str]] = {}
    for frame in complete:
        stems.setdefault(Path(frame).stem.lower(), []).append(frame)
    collisions = [items for items in stems.values() if len(items) > 1]
    if collisions:
        raise PipelineError(
            "不同扩展名产生相同输出目录名；请用--extensions限制一种格式："
            + repr(collisions[:5])
        )
    return complete, missing


def run_and_tee(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        return int(process.wait())


def valid_json_and_files(report: Path, files: Iterable[Path]) -> bool:
    if not report.is_file() or any(not path.is_file() for path in files):
        return False
    try:
        read_json(report)
    except PipelineError:
        return False
    return True


def geometry_complete(
    directory: Path, expected_pose_refinement: str | None = None
) -> bool:
    if (directory / ".geometry_incomplete").exists():
        return False
    report_path = directory / "alignment_report.json"
    if not valid_json_and_files(
        report_path,
        (directory / "depth_alignment_maps.npz",),
    ):
        return False
    if expected_pose_refinement is None:
        return True
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    settings = report.get("settings", {})
    return isinstance(settings, dict) and settings.get("pose_refinement", "off") == (
        expected_pose_refinement
    )


def frame_complete(
    directory: Path,
    reference_cameras: Sequence[str],
    expected_render_mode: str | None = None,
) -> bool:
    if (directory / ".pipeline_incomplete").exists():
        return False
    report = directory / "frame_report.json"
    results = directory / "results"
    diagnostics = directory / "diagnostics"
    required = [
        diagnostics / "depth_prior_report.json",
        results / "alignment_maps.npz",
        results / "target_reference.png",
        results / "target_depth.png",
        results / "aligned_grid.png",
        results / "overlay_grid.png",
        results / "overview.png",
    ]
    for name in reference_cameras:
        required.extend(
            (
                results / f"{name}_aligned.png",
                results / f"{name}_valid_mask.png",
                results / f"{name}_overlay_50.jpg",
            )
        )
    if not valid_json_and_files(report, required):
        return False
    try:
        value = read_json(report)
    except PipelineError:
        return False
    if value.get("status") != "success":
        return False
    if expected_render_mode is None:
        return True
    # Reports produced before render_mode existed used completed renders.
    return value.get("render_mode", "complete") == expected_render_mode


def geometry_command(args: argparse.Namespace, frame: str, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(args.geometry_script.resolve()),
        "--reference-cameras", *args.reference_cameras,
        "--target-camera", args.target_camera,
        "--anchor-camera", args.anchor_camera,
        "--calibration", str(args.calibration.resolve()),
        "--image-root", str(args.image_root),
        "--frame", frame,
        "--output-dir", str(output),
        "--roma-setting", args.roma_setting,
        "--representation", args.representation,
        "--pose-refinement", args.pose_refinement,
        "--pose-refine-ransac-threshold", str(args.pose_refine_ransac_threshold),
        "--pose-refine-ransac-max-iters", str(args.pose_refine_ransac_max_iters),
        "--pose-refine-max-samples", str(args.pose_refine_max_samples),
        "--pose-refine-homography-threshold", str(args.pose_refine_homography_threshold),
        "--pose-refine-max-homography-dominance", str(args.pose_refine_max_homography_dominance),
        "--pose-refine-max-rotation-deg", str(args.pose_refine_max_rotation_deg),
        "--pose-refine-max-translation-deg", str(args.pose_refine_max_translation_deg),
        "--pose-refine-min-inliers", str(args.pose_refine_min_inliers),
        "--pose-refine-min-inlier-ratio", str(args.pose_refine_min_inlier_ratio),
        "--pose-refine-min-improvement", str(args.pose_refine_min_improvement),
        "--pose-refine-strength", str(args.pose_refine_strength),
        "--fb-threshold", str(args.geometry_fb_threshold),
        "--epipolar-threshold", str(args.geometry_epipolar_threshold),
        "--reprojection-threshold", str(args.geometry_reprojection_threshold),
        "--minimum-matches", str(args.geometry_minimum_matches),
        "--minimum-views", str(args.geometry_minimum_views),
        "--depth-consistency", str(args.geometry_depth_consistency),
        "--fill-radius", "0",
        "--completion-mode", "off",
        "--occlusion-tolerance", str(args.occlusion_tolerance),
        "--overlay-alpha", "0.5",
        "--preview-scale", "1",
        "--overwrite",
    ]
    if args.allow_cpu:
        command.append("--allow-cpu")
    if args.allow_unaccepted_calibration:
        command.append("--allow-unaccepted-calibration")
    return command


def prior_command(
    args: argparse.Namespace,
    frame: str,
    geometry_npz: Path,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(args.prior_script.resolve()),
        "--reference-cameras", *args.reference_cameras,
        "--target-camera", args.target_camera,
        "--anchor-camera", args.anchor_camera,
        "--calibration", str(args.calibration.resolve()),
        "--geometry-npz", str(geometry_npz),
        "--depth-anything-root", str(args.depth_anything_root.resolve()),
        "--checkpoint", str(args.checkpoint.resolve()),
        "--image-root", str(args.image_root),
        "--frame", frame,
        "--output-dir", str(output),
        "--encoder", args.encoder,
        "--model-input-size", str(args.model_input_size),
        "--projection-max-side", str(args.projection_max_side),
        "--device", args.device,
        "--precision", args.precision,
        "--geometry-mask", "reliable",
        "--prior-cameras", *args.prior_cameras,
        "--reference-depth-camera", args.reference_depth_camera,
        "--final-depth-source", args.final_depth_source,
        "--render-mode", args.render_mode,
        "--segmentation-backend", args.segmentation_backend,
        "--render-unresolved-max-component-area", str(args.render_unresolved_max_component_area),
        "--render-fill-texture-method", args.render_fill_texture_method,
        "--render-inpaint-radius", str(args.render_inpaint_radius),
        "--overlay-alpha", "0.5",
        "--overwrite",
    ]
    if args.allow_cpu:
        command.append("--allow-cpu")
    if args.segmentation_backend != "off":
        command.extend(("--segmentation-checkpoint", str(args.segmentation_checkpoint.resolve())))
        if args.segmentation_root is not None:
            command.extend(("--segmentation-root", str(args.segmentation_root.resolve())))
        if args.segmentation_model_type is not None:
            command.extend(("--segmentation-model-type", args.segmentation_model_type))
    return command


def compact_geometry(directory: Path, reference_cameras: Sequence[str]) -> None:
    keep = {
        "alignment_report.json",
        "depth_alignment_maps.npz",
        "target_reference.jpg",
        "depth_reliable_mask.png",
        "depth_target_color.png",
        "depth_confidence.png",
        "depth_support_count.png",
    }
    for name in reference_cameras:
        keep.update(
            {
                f"{name}_coarse.jpg",
                f"{name}_coarse_mask.png",
                f"{name}_geometry_valid.png",
                f"{name}_candidate_depth.png",
            }
        )
    for path in directory.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()


def compact_prior(
    directory: Path,
    prior_cameras: Sequence[str],
    reference_cameras: Sequence[str],
) -> None:
    keep = {
        "depth_prior_report.json",
        "depth_final_model_raw.png",
        "depth_final_refined_candidate.png",
        "depth_render_projection.png",
        "depth_render_surface_guide.png",
        "depth_complete_mask.png",
        "depth_support_count.png",
        "geometry_anchor_mask.png",
        "geometry_agreement_mask.png",
        "geometry_conflict_mask.png",
        "model_conflict_mask.png",
        "reference_surface_override_mask.png",
        "debug_depth_grid.png",
        "debug_render_grid.png",
    }
    for name in prior_cameras:
        keep.update(
            {
                f"{name}_model_raw.png",
                f"{name}_model_depth.png",
                f"{name}_prior_target.png",
                f"{name}_prior_confidence.png",
                f"{name}_prior_boundary.png",
            }
        )
    for name in reference_cameras:
        keep.update(
            {
                f"{name}_aligned_surface_copy.png",
                f"{name}_aligned_strict.jpg",
                f"{name}_edge_overlay.png",
                f"{name}_strict_mask.png",
                f"{name}_zbuffer_visible_mask.png",
                f"{name}_occlusion_ambiguous_mask.png",
                f"{name}_occlusion_surface_filled_mask.png",
                f"{name}_occlusion_relaxed_eligible_mask.png",
                f"{name}_occlusion_relaxed_filled_mask.png",
                f"{name}_occlusion_unresolved_mask.png",
                f"{name}_display_filled_mask.png",
                f"{name}_texture_structure_refined_mask.png",
                f"{name}_texture_inpaint_solver_mask.png",
                f"{name}_edge_nearest_mask.png",
            }
        )
    for path in directory.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()


def rendering_metrics(
    report_path: Path, anchor_camera: str
) -> tuple[float | None, float | None]:
    try:
        report = read_json(report_path)
        anchor = report.get("rendering", {}).get(anchor_camera, {})
        if "primary_valid_ratio" in anchor:
            coverage = 100.0 * float(anchor["primary_valid_ratio"])
        else:
            coverage = 100.0 * sum(
                float(anchor.get(key, 0.0))
                for key in (
                    "zbuffer_visible_ratio",
                    "occlusion_surface_filled_ratio",
                    "occlusion_relaxed_filled_ratio",
                    "display_filled_ratio",
                )
            )
        coverage = min(max(coverage, 0.0), 100.0)
        unresolved = float(anchor["occlusion_unresolved_ratio"]) * 100.0
        return coverage, unresolved
    except (PipelineError, KeyError, TypeError, ValueError):
        return None, None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "frame", "scene_id", "variant", "status", "geometry_status", "prior_status",
        "seconds", "anchor_coverage_pct", "anchor_unresolved_pct", "output_dir",
        "geometry_log", "prior_log", "error",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def relative_link(root: Path, target: Path) -> str:
    return quote(target.relative_to(root).as_posix(), safe="/._-")


def write_gallery(
    output_root: Path,
    rows: list[dict[str, Any]],
    camera_roles: dict[str, Any],
) -> None:
    references = [str(value) for value in camera_roles.get("reference_cameras", [])]
    target_camera = str(camera_roles.get("target_camera", "target"))
    anchor_camera = str(camera_roles.get("anchor_camera", references[0] if references else ""))
    cards: list[str] = []
    for row in rows:
        frame = html.escape(str(row.get("frame", "")))
        status = html.escape(str(row.get("status", "")))
        output_text = str(row.get("output_dir", ""))
        output = Path(output_text) if output_text else None
        image_html = ""
        links_html = ""
        if output is not None and (output / "results" / "overview.png").is_file():
            overview = relative_link(output_root, output / "results" / "overview.png")
            image_html = f'<a href="{overview}"><img loading="lazy" src="{overview}" alt="{frame}"></a>'
            links = []
            for label, name in (
                ("aligned grid", "results/aligned_grid.png"),
                ("overlay grid", "results/overlay_grid.png"),
                ("alignment maps", "results/alignment_maps.npz"),
                ("depth diagnostics", "diagnostics/debug_depth_grid.png"),
                ("render diagnostics", "diagnostics/debug_render_grid.png"),
                ("depth report", "diagnostics/depth_prior_report.json"),
                ("frame report", "frame_report.json"),
            ):
                target = output / name
                if target.is_file():
                    links.append(
                        f'<a href="{relative_link(output_root, target)}">'
                        f'{html.escape(label)}</a>'
                    )
            links_html = " · ".join(links)
        error = html.escape(str(row.get("error", "")))
        meta = " / ".join(
            item for item in (str(row.get("scene_id", "")), str(row.get("variant", ""))) if item
        )
        cards.append(
            f'<section class="card {status}"><h2>{frame}</h2>'
            f'<p><b>{status}</b> {html.escape(meta)}</p>{image_html}'
            f'<p class="links">{links_html}</p><pre>{error}</pre></section>'
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>多相机批量对齐结果</title>
<style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;background:#111;color:#eee;margin:20px}}
h1{{font-size:24px}} .card{{background:#1d1d1d;border:1px solid #444;border-radius:10px;padding:14px;margin:16px 0}}
.card.failed{{border-color:#b44}} .card.success,.card.skipped_complete{{border-color:#385}}
img{{width:100%;height:auto;background:#000}} a{{color:#8cc8ff}} pre{{white-space:pre-wrap;color:#f99}}
.links{{line-height:1.8}}
</style></head><body><h1>多参考相机 → {html.escape(target_camera)} 批量对齐</h1>
<p>参考相机：{html.escape(', '.join(references))}；质量汇总锚点：{html.escape(anchor_camera)}。</p>
<p>每张总览的上半部分是目标图和无损对齐结果预览，下半部分是50:50叠加。</p>
{''.join(cards)}</body></html>"""
    write_text_atomic(output_root / "index.html", document)


def save_summaries(
    output_root: Path,
    rows: list[dict[str, Any]],
    common: dict[str, Any],
) -> None:
    write_csv(output_root / "batch_summary.csv", rows)
    summary = dict(common)
    summary["updated_at"] = utc_now()
    summary["succeeded_or_skipped"] = sum(
        row.get("status") in ("success", "skipped_complete") for row in rows
    )
    summary["failed"] = sum(row.get("status") == "failed" for row in rows)
    summary["frames"] = rows
    write_json_atomic(output_root / "batch_summary.json", summary)
    write_gallery(output_root, rows, dict(common.get("camera_roles", {})))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从多相机同名图片批量完成几何、单目深度、带可见性mask的对齐和结果总览"
    )
    parser.add_argument("--version", action="version", version=PROGRAM_VERSION)
    parser.add_argument("--reference-cameras", nargs="+", required=True)
    parser.add_argument("--target-camera", required=True)
    parser.add_argument("--anchor-camera", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--depth-anything-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--geometry-script", type=Path,
        default=Path(__file__).resolve().parents[1] / "stages" / "geometry.py",
    )
    parser.add_argument(
        "--prior-script", type=Path,
        default=Path(__file__).resolve().parents[1] / "stages" / "depth_prior.py",
    )
    parser.add_argument(
        "--split",
        default="all",
        help="all, or the stem of a text file under splits/",
    )
    parser.add_argument("--frame-glob", default="*")
    parser.add_argument("--extensions", nargs="+", default=list(DEFAULT_EXTENSIONS))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true", help="重跑深度先验和最终渲染，但仍复用几何")
    parser.add_argument("--rerun-geometry", action="store_true", help="连昂贵的RoMa几何也重新计算")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", choices=("compact", "full"), default="compact")
    parser.add_argument("--panel-width", type=int, default=400)

    parser.add_argument(
        "--roma-setting",
        choices=("precise", "mega1500", "scannet1500", "wxbs", "satast", "base", "fast", "turbo"),
        default="fast",
    )
    parser.add_argument("--representation", choices=("gray", "rgb", "structure"), default="gray")
    parser.add_argument("--pose-refinement", choices=("off", "essential"), default="off")
    parser.add_argument("--pose-refine-ransac-threshold", type=float, default=1.5)
    parser.add_argument("--pose-refine-ransac-max-iters", type=int, default=10000)
    parser.add_argument("--pose-refine-max-samples", type=int, default=5000)
    parser.add_argument("--pose-refine-homography-threshold", type=float, default=2.0)
    parser.add_argument("--pose-refine-max-homography-dominance", type=float, default=0.95)
    parser.add_argument("--pose-refine-max-rotation-deg", type=float, default=3.0)
    parser.add_argument("--pose-refine-max-translation-deg", type=float, default=8.0)
    parser.add_argument("--pose-refine-min-inliers", type=int, default=80)
    parser.add_argument("--pose-refine-min-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--pose-refine-min-improvement", type=float, default=0.05)
    parser.add_argument("--pose-refine-strength", type=float, default=1.0)
    parser.add_argument("--geometry-fb-threshold", type=float, default=2.0)
    parser.add_argument("--geometry-epipolar-threshold", type=float, default=2.0)
    parser.add_argument("--geometry-reprojection-threshold", type=float, default=2.0)
    parser.add_argument("--geometry-minimum-matches", type=int, default=200)
    parser.add_argument("--geometry-minimum-views", type=int)
    parser.add_argument("--geometry-depth-consistency", type=float, default=0.20)
    parser.add_argument("--occlusion-tolerance", type=float, default=0.01)

    parser.add_argument("--encoder", choices=("vits", "vitb", "vitl", "vitg"), default="vits")
    parser.add_argument("--model-input-size", type=int, default=518)
    parser.add_argument("--projection-max-side", type=int, default=1600)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp16", "fp32"), default="auto")
    parser.add_argument(
        "--prior-cameras",
        nargs="+",
        help="references used for dense-depth priors; defaults to the first two",
    )
    parser.add_argument(
        "--reference-depth-camera",
        help="learned surface to preserve; defaults to the anchor camera",
    )
    parser.add_argument("--final-depth-source", choices=("model-raw", "refined"), default="model-raw")
    parser.add_argument(
        "--render-mode",
        choices=("strict", "complete"),
        default="strict",
        help="strict输出未填补的Z-buffer可见像素和精确mask；complete显式启用显示补全",
    )
    parser.add_argument("--render-unresolved-max-component-area", type=int, default=512)
    parser.add_argument(
        "--render-fill-texture-method",
        choices=("navier-stokes", "telea", "copy"), default="navier-stokes",
    )
    parser.add_argument("--render-inpaint-radius", type=float, default=3.0)
    parser.add_argument("--segmentation-backend", choices=("off", "mobilesam", "sam"), default="off")
    parser.add_argument("--segmentation-root", type=Path)
    parser.add_argument("--segmentation-checkpoint", type=Path)
    parser.add_argument("--segmentation-model-type")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--allow-unaccepted-calibration", action="store_true")
    args = parser.parse_args(argv)
    configure_rig(args.reference_cameras, args.target_camera, args.anchor_camera)
    if args.prior_cameras is None:
        args.prior_cameras = list(REFERENCE_CAMERAS[:2])
    if args.reference_depth_camera is None:
        args.reference_depth_camera = args.anchor_camera
    if args.geometry_minimum_views is None:
        args.geometry_minimum_views = min(2, len(REFERENCE_CAMERAS))
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.start < 0 or (args.limit is not None and args.limit <= 0):
        raise PipelineError("--start不能为负，--limit必须大于0")
    if args.panel_width < 160:
        raise PipelineError("--panel-width至少为160")
    if args.model_input_size <= 0 or args.projection_max_side <= 0:
        raise PipelineError("模型和投影尺寸必须大于0")
    if args.pose_refine_min_inliers < 8:
        raise PipelineError("--pose-refine-min-inliers必须至少为8")
    if args.pose_refine_max_samples < 8 or args.pose_refine_ransac_max_iters < 1:
        raise PipelineError("单帧外参修正样本数必须至少8，RANSAC迭代必须大于0")
    if args.pose_refine_min_inliers > args.pose_refine_max_samples:
        raise PipelineError("单帧外参修正的最少内点数不能超过最大样本数")
    if args.pose_refine_ransac_threshold <= 0 or args.pose_refine_homography_threshold <= 0:
        raise PipelineError("单帧外参修正的RANSAC阈值必须大于0")
    if not 0.0 < args.pose_refine_max_homography_dominance <= 1.5:
        raise PipelineError("--pose-refine-max-homography-dominance必须在(0,1.5]内")
    if not 0.0 <= args.pose_refine_min_inlier_ratio <= 1.0:
        raise PipelineError("--pose-refine-min-inlier-ratio必须在[0,1]内")
    if not 0.0 <= args.pose_refine_min_improvement < 1.0:
        raise PipelineError("--pose-refine-min-improvement必须在[0,1)内")
    if not 0.0 < args.pose_refine_strength <= 1.0:
        raise PipelineError("--pose-refine-strength必须在(0,1]内")
    if (
        args.pose_refine_max_rotation_deg <= 0
        or args.pose_refine_max_translation_deg <= 0
    ):
        raise PipelineError("单帧外参修正的最大漂移角必须大于0")
    if args.render_unresolved_max_component_area < 0 or args.render_inpaint_radius < 0:
        raise PipelineError("渲染组件面积和修复半径不能为负")
    unknown_priors = set(args.prior_cameras).difference(REFERENCE_CAMERAS)
    if unknown_priors:
        raise PipelineError(
            "--prior-cameras包含未知参考相机：" + ", ".join(sorted(unknown_priors))
        )
    if len(set(args.prior_cameras)) != len(args.prior_cameras):
        raise PipelineError("--prior-cameras不能重复")
    if args.reference_depth_camera not in args.prior_cameras:
        raise PipelineError("--reference-depth-camera必须包含在--prior-cameras中")
    if not 1 <= args.geometry_minimum_views <= len(REFERENCE_CAMERAS):
        raise PipelineError(
            f"--geometry-minimum-views必须在1到{len(REFERENCE_CAMERAS)}之间"
        )
    if args.segmentation_backend != "off" and args.segmentation_checkpoint is None:
        raise PipelineError("启用分割时必须提供--segmentation-checkpoint")
    for path, label in (
        (args.calibration, "校准JSON"),
        (args.geometry_script, "几何脚本"),
        (args.prior_script, "深度先验脚本"),
        (args.checkpoint, "Depth Anything权重"),
    ):
        if not path.resolve().is_file():
            raise PipelineError(f"找不到{label}：{path.resolve()}")
    if not args.depth_anything_root.resolve().is_dir():
        raise PipelineError(f"找不到Depth Anything仓库：{args.depth_anything_root.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    dataset_root, image_root = resolve_dataset_root(args.dataset_root)
    args.image_root = image_root
    output_root = args.output_root.resolve()
    frames_root = output_root / "frames"
    geometry_root = output_root / "geometry"
    logs_root = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)
    geometry_root.mkdir(parents=True, exist_ok=True)

    frames, missing = discover_frames(
        dataset_root,
        image_root,
        args.target_camera,
        args.frame_glob,
        parse_extensions(args.extensions),
        args.split,
    )
    selected = frames[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise PipelineError("筛选后没有待处理共同帧")
    metadata = load_metadata(dataset_root)
    started_at = utc_now()
    rows: list[dict[str, Any]] = []
    common_summary = {
        "schema": "multialign_end_to_end_batch_v1",
        "program_version": PROGRAM_VERSION,
        "started_at": started_at,
        "dataset_root": str(dataset_root),
        "image_root": str(image_root),
        "calibration": str(args.calibration.resolve()),
        "camera_roles": {
            "reference_cameras": list(args.reference_cameras),
            "target_camera": args.target_camera,
            "anchor_camera": args.anchor_camera,
            "depth_prior_cameras": list(args.prior_cameras),
            "reference_depth_camera": args.reference_depth_camera,
        },
        "output_contract": {
            "primary_results": "frames/<frame>/results/",
            "quality_diagnostics": "frames/<frame>/diagnostics/",
            "frame_report": "frames/<frame>/frame_report.json",
            "geometry_cache": "geometry/<frame>/",
            "logs": "logs/",
        },
        "split": args.split,
        "discovered_complete_frames": len(frames),
        "selected_frames": len(selected),
        "missing_camera_frames": missing,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "image_root"
        },
    }
    print(
        f"共同帧={len(frames)}，缺图帧={len(missing)}，本次={len(selected)}；"
        f"split={args.split}，debug={args.debug}"
    )
    print("几何缓存默认复用；每个GPU阶段独立进程、失败后继续下一帧。")

    if args.dry_run:
        for index, frame in enumerate(selected, 1):
            stem = Path(frame).stem
            print(f"[{index}/{len(selected)}] {frame} -> {frames_root / stem}")
        common_summary["dry_run"] = True
        save_summaries(output_root, rows, common_summary)
        return 0

    for index, frame in enumerate(selected, 1):
        stem = Path(frame).stem
        frame_dir = frames_root / stem
        results_dir = frame_dir / "results"
        diagnostics_dir = frame_dir / "diagnostics"
        geometry_dir = geometry_root / stem
        geometry_log = logs_root / f"{stem}_geometry.log"
        prior_log = logs_root / f"{stem}_prior.log"
        meta = metadata.get(stem, {})
        row: dict[str, Any] = {
            "frame": frame,
            "scene_id": meta.get("scene_id", ""),
            "variant": meta.get("variant", ""),
            "status": "pending",
            "geometry_status": "pending",
            "prior_status": "pending",
            "seconds": 0.0,
            "anchor_coverage_pct": None,
            "anchor_unresolved_pct": None,
            "output_dir": str(frame_dir),
            "geometry_log": str(geometry_log),
            "prior_log": str(prior_log),
            "error": "",
        }
        t0 = time.monotonic()
        try:
            if (
                frame_complete(
                    frame_dir, args.reference_cameras, args.render_mode
                )
                and not args.overwrite
            ):
                row["status"] = "skipped_complete"
                row["geometry_status"] = "reused"
                row["prior_status"] = "reused"
                coverage, unresolved = rendering_metrics(
                    diagnostics_dir / "depth_prior_report.json", args.anchor_camera
                )
                row["anchor_coverage_pct"] = coverage
                row["anchor_unresolved_pct"] = unresolved
                print(f"[{index}/{len(selected)}] {frame}: 已完成，跳过")
            else:
                geometry_dir.mkdir(parents=True, exist_ok=True)
                if (
                    geometry_complete(geometry_dir, args.pose_refinement)
                    and not args.rerun_geometry
                ):
                    row["geometry_status"] = "reused"
                    print(f"\n[{index}/{len(selected)}] {frame}: 复用RoMa几何缓存")
                else:
                    print(f"\n[{index}/{len(selected)}] {frame}: 阶段1/2 RoMa固定几何")
                    geometry_marker = geometry_dir / ".geometry_incomplete"
                    geometry_marker.touch()
                    exit_code = run_and_tee(
                        geometry_command(args, frame, geometry_dir), geometry_log
                    )
                    if exit_code != 0:
                        raise PipelineError(f"几何阶段失败，exit_code={exit_code}")
                    geometry_marker.unlink(missing_ok=True)
                    if not geometry_complete(geometry_dir, args.pose_refinement):
                        raise PipelineError("几何阶段返回成功，但报告或NPZ不完整")
                    row["geometry_status"] = "success"
                    if args.debug == "compact":
                        compact_geometry(geometry_dir, args.reference_cameras)

                frame_dir.mkdir(parents=True, exist_ok=True)
                diagnostics_dir.mkdir(parents=True, exist_ok=True)
                pipeline_marker = frame_dir / ".pipeline_incomplete"
                pipeline_marker.touch()
                (frame_dir / "frame_report.json").unlink(missing_ok=True)
                print(
                    f"[{index}/{len(selected)}] {frame}: 阶段2/2 单目深度锚定与"
                    f"{len(args.reference_cameras)}相机重投影"
                )
                exit_code = run_and_tee(
                    prior_command(
                        args,
                        frame,
                        geometry_dir / "depth_alignment_maps.npz",
                        diagnostics_dir,
                    ),
                    prior_log,
                )
                prior_report = diagnostics_dir / "depth_prior_report.json"
                required_prior = [prior_report]
                required_prior.extend(
                    diagnostics_dir / f"{name}_aligned.png"
                    for name in args.reference_cameras
                )
                required_prior.extend(
                    diagnostics_dir / f"{name}_overlay_50.jpg"
                    for name in args.reference_cameras
                )
                required_prior.extend(
                    diagnostics_dir / f"{name}_valid_mask.png"
                    for name in args.reference_cameras
                )
                if exit_code != 0 or not all(path.is_file() for path in required_prior):
                    raise PipelineError(f"深度先验阶段失败或输出不完整，exit_code={exit_code}")
                row["prior_status"] = "success"

                visuals = make_frame_visuals(
                    diagnostics_dir,
                    results_dir,
                    args.reference_cameras,
                    args.target_camera,
                    args.panel_width,
                )
                primary_files = promote_primary_results(
                    diagnostics_dir, results_dir, args.reference_cameras
                )
                visuals["primary"].extend(primary_files)
                visuals["primary"] = sorted(set(visuals["primary"]))
                coverage, unresolved = rendering_metrics(
                    prior_report, args.anchor_camera
                )
                row["anchor_coverage_pct"] = coverage
                row["anchor_unresolved_pct"] = unresolved
                frame_report = {
                    "schema": "multialign_end_to_end_frame_v2",
                    "status": "success",
                    "render_mode": args.render_mode,
                    "frame": frame,
                    "scene_id": row["scene_id"],
                    "variant": row["variant"],
                    "camera_roles": {
                        "reference_cameras": list(args.reference_cameras),
                        "target_camera": args.target_camera,
                        "anchor_camera": args.anchor_camera,
                        "depth_prior_cameras": list(args.prior_cameras),
                        "reference_depth_camera": args.reference_depth_camera,
                    },
                    "outputs": {
                        "primary_directory": "results",
                        "diagnostics_directory": "diagnostics",
                        "alignment_maps": "results/alignment_maps.npz",
                        "geometry_cache": str(
                            geometry_dir / "depth_alignment_maps.npz"
                        ),
                        "depth_prior_report": "diagnostics/depth_prior_report.json",
                    },
                    "anchor_coverage_pct": coverage,
                    "anchor_unresolved_pct": unresolved,
                    "visuals": visuals,
                    "methods": {
                        "geometry": (
                            "RoMa correspondences + global calibration"
                            + (
                                " + guarded per-frame essential-pose refinement"
                                if args.pose_refinement == "essential"
                                else ""
                            )
                            + " + epipolar/triangulation gates"
                        ),
                        "depth": "Depth Anything V2 relative depth anchored by reliable calibrated geometry",
                        "render": (
                            "model-raw reference lock + one-pass reference-to-target "
                            "reprojection + z-buffer + strict masked output without fill"
                            if args.render_mode == "strict"
                            else (
                                "model-raw reference lock + one-pass reference-to-target "
                                "reprojection + z-buffer + explicitly enabled bounded completion"
                            )
                        ),
                        "overlay": "aligned reference and target at 0.5/0.5 inside valid render support",
                    },
                    "completed_at": utc_now(),
                }
                write_json_atomic(frame_dir / "frame_report.json", frame_report)
                if args.debug == "compact":
                    compact_prior(
                        diagnostics_dir,
                        args.prior_cameras,
                        args.reference_cameras,
                    )
                pipeline_marker.unlink(missing_ok=True)
                if not frame_complete(
                    frame_dir, args.reference_cameras, args.render_mode
                ):
                    raise PipelineError("最终质量检查失败：必要文件缺失")
                row["status"] = "success"
                print(
                    f"[{index}/{len(selected)}] {frame}: 完成；"
                    f"{args.anchor_camera}覆盖="
                    f"{coverage if coverage is not None else float('nan'):.2f}%，"
                    f"未解决={unresolved if unresolved is not None else float('nan'):.2f}%"
                )
        except (PipelineError, OSError, cv2.error, ValueError) as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            print(f"[{index}/{len(selected)}] {frame}: 失败：{exc}")
        row["seconds"] = round(time.monotonic() - t0, 3)
        rows.append(row)
        save_summaries(output_root, rows, common_summary)
        if row["status"] == "failed" and args.stop_on_error:
            break

    failed = sum(row["status"] == "failed" for row in rows)
    succeeded = sum(row["status"] in ("success", "skipped_complete") for row in rows)
    common_summary["finished_at"] = utc_now()
    save_summaries(output_root, rows, common_summary)
    print(f"\n批处理完成：成功/已完成={succeeded}，失败={failed}")
    print(f"汇总表：{output_root / 'batch_summary.csv'}")
    print(f"可视化索引：{output_root / 'index.html'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        print(f"错误：{exc}")
        raise SystemExit(2)
