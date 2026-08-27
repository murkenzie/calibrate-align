#!/usr/bin/env python3
"""Run the calibration half of the unified multi-camera project workflow.

Expected input contains one directory per configured reference camera and one
directory for the target camera. Files belonging to one capture are associated
by filename stem. The
runner creates a tiny metadata adapter containing absolute paths; it never
copies or resizes the source images. It then runs every camera pair through
the dedicated all-RoMa matcher and finally invokes the robust multi-camera
joint optimizer.

The matching stage is resumable.  Existing compatible NPZ files are reused
unless ``--overwrite-matches`` is supplied.  Calibration output is kept unless
``--overwrite-calibration`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence


PROGRAM_VERSION = "1.0"
REFERENCE_CAMERAS = ("reference_a", "reference_b")
TARGET_CAMERA = "target"
ANCHOR_CAMERA = REFERENCE_CAMERAS[0]
SCALE_REFERENCE_CAMERA = REFERENCE_CAMERAS[1]
CAMERAS = (*REFERENCE_CAMERAS, TARGET_CAMERA)
ALL_PAIRS = tuple(f"{a}-{b}" for a, b in combinations(CAMERAS, 2))
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


class PipelineError(RuntimeError):
    pass


def configure_rig(
    reference_cameras: Sequence[str],
    target_camera: str,
    anchor_camera: str,
    scale_reference_camera: str,
) -> None:
    global REFERENCE_CAMERAS, TARGET_CAMERA, ANCHOR_CAMERA
    global SCALE_REFERENCE_CAMERA, CAMERAS, ALL_PAIRS
    references = tuple(reference_cameras)
    if len(references) < 2 or len(set(references)) != len(references):
        raise PipelineError("--reference-cameras requires at least two unique names")
    if target_camera in references:
        raise PipelineError("--target-camera must not be a reference camera")
    if anchor_camera not in references:
        raise PipelineError("--anchor-camera must be a reference camera")
    if scale_reference_camera not in references or scale_reference_camera == anchor_camera:
        raise PipelineError(
            "--scale-reference-camera must be a non-anchor reference camera"
        )
    REFERENCE_CAMERAS = references
    TARGET_CAMERA = target_camera
    ANCHOR_CAMERA = anchor_camera
    SCALE_REFERENCE_CAMERA = scale_reference_camera
    CAMERAS = (*references, target_camera)
    ALL_PAIRS = tuple(f"{a}-{b}" for a, b in combinations(CAMERAS, 2))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(token) if token.isdigit() else token.casefold()
        for token in re.split(r"(\d+)", value)
    )


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"无法读取JSON：{path} ({exc})") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"JSON顶层必须是对象：{path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def parse_extensions(values: Sequence[str]) -> set[str]:
    output: set[str] = set()
    for raw in values:
        suffix = raw.strip().casefold()
        if not suffix.startswith("."):
            suffix = "." + suffix
        output.add(suffix)
    if not output:
        raise PipelineError("--extensions不能为空")
    return output


def resolve_image_root(data_root: Path) -> Path:
    root = data_root.expanduser().resolve()
    if all((root / camera).is_dir() for camera in CAMERAS):
        return root
    images = root / "images"
    if all((images / camera).is_dir() for camera in CAMERAS):
        return images
    expected = ", ".join(str(root / camera) for camera in CAMERAS)
    raise PipelineError(
        "The input root does not contain every configured camera directory:\n  "
        + "\n  ".join(expected.split(", "))
    )


def index_camera_images(
    directory: Path,
    frame_glob: str,
    extensions: set[str],
) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in directory.glob(frame_glob):
        if not path.is_file() or path.suffix.casefold() not in extensions:
            continue
        key = path.stem.casefold()
        previous = indexed.get(key)
        if previous is not None:
            raise PipelineError(
                f"{directory}中同一帧stem出现多个文件：{previous.name}, {path.name}；"
                "请限制--extensions或整理重名文件"
            )
        indexed[key] = path.resolve()
    if not indexed:
        raise PipelineError(
            f"{directory}中没有符合 --frame-glob {frame_glob!r} 的输入图"
        )
    return indexed


def read_stem_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PipelineError(f"帧列表不存在：{resolved}")
    values = {
        Path(line.strip()).stem.casefold()
        for line in resolved.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not values:
        raise PipelineError(f"帧列表为空：{resolved}")
    return values


def discover_groups(
    image_root: Path,
    frame_glob: str,
    extensions: set[str],
    include_file: Path | None,
    exclude_file: Path | None,
    skip_incomplete: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_camera = {
        camera: index_camera_images(image_root / camera, frame_glob, extensions)
        for camera in CAMERAS
    }
    all_stems = set().union(*(set(index) for index in by_camera.values()))
    common = set.intersection(*(set(index) for index in by_camera.values()))
    incomplete: dict[str, list[str]] = {}
    for stem in sorted(all_stems, key=natural_key):
        absent = [camera for camera in CAMERAS if stem not in by_camera[camera]]
        if absent:
            incomplete[stem] = absent
    if incomplete and not skip_incomplete:
        examples = [
            f"{stem}: 缺少 {','.join(cameras)}"
            for stem, cameras in list(incomplete.items())[:20]
        ]
        more = "" if len(incomplete) <= 20 else f"\n  ...另有{len(incomplete)-20}组"
        raise PipelineError(
            "Camera frame stems are incomplete; stopping to prevent mis-grouping:\n  "
            + "\n  ".join(examples)
            + more
            + "\n确认允许跳过后添加 --skip-incomplete"
        )

    include = read_stem_file(include_file)
    exclude = read_stem_file(exclude_file) or set()
    selected = [
        stem
        for stem in common
        if (include is None or stem in include) and stem not in exclude
    ]
    selected.sort(key=natural_key)
    if not selected:
        raise PipelineError("筛选后没有完整的多相机采集组")

    groups: list[dict[str, Any]] = []
    for stem in selected:
        canonical_frame = by_camera[TARGET_CAMERA][stem].stem
        references = {
            camera: {"prepared_jpg": str(by_camera[camera][stem])}
            for camera in REFERENCE_CAMERAS
        }
        groups.append(
            {
                "frame": canonical_frame,
                "scene_id": canonical_frame,
                "variant": "scene",
                "references": references,
                "target": {
                    "camera": TARGET_CAMERA,
                    "prepared_composite": str(by_camera[TARGET_CAMERA][stem])
                },
            }
        )
    inventory = {
        "image_counts": {camera: len(paths) for camera, paths in by_camera.items()},
        "complete_group_count": len(common),
        "selected_group_count": len(groups),
        "incomplete_groups": incomplete,
        "included_by_file": sorted(include, key=natural_key) if include else None,
        "excluded_by_file": sorted(exclude, key=natural_key),
    }
    return groups, inventory


def build_dataset_adapter(
    adapter_root: Path,
    image_root: Path,
    groups: list[dict[str, Any]],
    inventory: dict[str, Any],
) -> Path:
    dataset_path = adapter_root / "metadata" / "dataset.json"
    frames = [str(group["frame"]) for group in groups]
    dataset = {
        "schema_version": 1,
        "adapter_schema": "direct_data_multialign_v1",
        "generated_utc": utc_now(),
        "source_image_root": str(image_root),
        "group_count": len(groups),
        "camera_names": list(CAMERAS),
        "camera_roles": {
            "reference_cameras": list(REFERENCE_CAMERAS),
            "target_camera": TARGET_CAMERA,
            "anchor_camera": ANCHOR_CAMERA,
            "scale_reference_camera": SCALE_REFERENCE_CAMERA,
        },
        "inventory": inventory,
        "groups": groups,
    }
    write_json_atomic(dataset_path, dataset)
    split_text = "\n".join(frames) + "\n"
    write_text_atomic(adapter_root / "splits" / "scene.txt", split_text)
    write_text_atomic(adapter_root / "splits" / "all.txt", split_text)
    return dataset_path


def validate_all_roma_matcher(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PipelineError(f"找不到全RoMa匹配程序：{resolved}")
    try:
        source = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise PipelineError(f"无法读取全RoMa匹配程序：{resolved} ({exc})") from exc
    required = ("roma_v2_all_pairs", "def run_roma_matching(")
    missing = [marker for marker in required if marker not in source]
    if missing:
        raise PipelineError(
            f"{resolved.name} is not the required all-pairs RoMa stage; missing {missing}. "
            "请使用随本包提供的 multialign/stages/matching.py"
        )
    return resolved


def validate_calibrator(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PipelineError(f"找不到多相机联合标定程序：{resolved}")
    source = resolved.read_text(encoding="utf-8")
    required = ("--shared-point-threshold", "--force-accept", "configure_rig")
    missing = [marker for marker in required if marker not in source]
    if missing:
        raise PipelineError(
            f"{resolved.name}不是本流程需要的鲁棒多相机版本，缺少标记：{missing}"
        )
    return resolved


def validate_initial_calibration(
    path: Path,
    required_cameras: Sequence[str] = CAMERAS,
) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PipelineError(f"初始内参文件不存在：{resolved}")
    value = read_json(resolved)
    cameras = value.get("cameras")
    if not isinstance(cameras, dict):
        raise PipelineError(f"初始标定缺少cameras对象：{resolved}")
    missing = [camera for camera in required_cameras if camera not in cameras]
    if missing:
        raise PipelineError(f"初始标定缺少相机：{', '.join(missing)}")
    return resolved


def run_and_tee(command: list[str], log_path: Path, dry_run: bool) -> int:
    command_text = subprocess.list2cmdline(command)
    print("\n> " + command_text, flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + command_text + "\n\n")
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


def match_counts(geometry_root: Path) -> dict[str, int]:
    return {
        pair: len(list((geometry_root / "matches" / pair.replace("-", "__")).glob("*.npz")))
        for pair in ALL_PAIRS
    }


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except PipelineError:
        return None


def matching_command(
    args: argparse.Namespace,
    matcher: Path,
    adapter_root: Path,
    geometry_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(matcher),
        "--dataset", str(adapter_root),
        "--output", str(geometry_root),
        "--split", "scene",
        "--reference-cameras", *REFERENCE_CAMERAS,
        "--target-camera", TARGET_CAMERA,
        "--pairs", *ALL_PAIRS,
        "--target-source", "composite",
        "--cross-representation", args.cross_representation,
        "--roma-setting", args.roma_setting,
        "--roma-samples", str(args.roma_samples),
        "--max-matches-per-group", str(args.max_matches_per_group),
        "--grid-cols", str(args.grid_cols),
        "--grid-rows", str(args.grid_rows),
        "--grid-max-per-cell", str(args.grid_max_per_cell),
        "--common-diagonal", str(args.common_diagonal),
        "--ransac-threshold", str(args.ransac_threshold),
        "--homography-threshold", str(args.homography_threshold),
        "--min-group-inliers", str(args.match_min_group_inliers),
        "--validation-fraction", str(args.validation_fraction),
        "--seed", str(args.seed),
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.allow_cpu:
        command.append("--allow-cpu")
    if args.torch_compile:
        command.append("--torch-compile")
    if args.no_diagnostics:
        command.append("--no-diagnostics")
    if args.diagnose_only:
        command.append("--diagnose-only")
    if args.overwrite_matches:
        command.append("--overwrite-matches")
    if args.strict:
        command.append("--strict")
    return command


def calibration_command(
    args: argparse.Namespace,
    calibrator: Path,
    geometry_root: Path,
    calibration_root: Path,
    initial_calibration: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(calibrator),
        "--geometry-root", str(geometry_root),
        "--initial-calibration", str(initial_calibration),
        "--output", str(calibration_root),
        "--reference-cameras", *REFERENCE_CAMERAS,
        "--target-camera", TARGET_CAMERA,
        "--anchor-camera", ANCHOR_CAMERA,
        "--scale-reference-camera", SCALE_REFERENCE_CAMERA,
        "--pairs", *ALL_PAIRS,
        "--reference-intrinsics", "fixed",
        "--target-model", args.target_model,
        "--reference-distortion", args.reference_distortion,
        "--target-distortion", args.target_distortion,
        "--target-focal-min", str(args.target_focal_min),
        "--target-focal-max", str(args.target_focal_max),
        "--target-pp-bound-fraction", str(args.target_pp_bound_fraction),
        "--target-pp-prior-fraction", str(args.target_pp_prior_fraction),
        "--baseline-log-sigma", str(args.baseline_log_sigma),
        "--gauge-log-sigma", str(args.gauge_log_sigma),
        "--maximum-baseline-ratio", str(args.maximum_baseline_ratio),
        "--rotation-prior-deg", str(args.rotation_prior_deg),
        "--direction-prior-deg", str(args.direction_prior_deg),
        "--rotation-bound-deg", str(args.rotation_bound_deg),
        "--center-bound", str(args.center_bound),
        "--prior-equivalent-points", str(args.prior_equivalent_points),
        "--common-diagonal", str(args.common_diagonal),
        "--shared-point-threshold", str(args.shared_point_threshold),
        "--strict-shared-point-threshold", str(args.strict_shared_point_threshold),
        "--minimum-shared-inlier-ratio", str(args.minimum_shared_inlier_ratio),
        "--strict-minimum-shared-inlier-ratio", str(args.strict_minimum_shared_inlier_ratio),
        "--shared-filter-iterations", str(args.shared_filter_iterations),
        "--min-group-inliers", str(args.calibration_min_group_inliers),
        "--max-points-per-group", str(args.max_points_per_group),
        "--validation-fraction", str(args.validation_fraction),
        "--loss", args.loss,
        "--loss-scale", str(args.loss_scale),
        "--max-nfev", str(args.max_nfev),
        "--seed", str(args.seed),
    ]
    if args.strict_reference_cameras:
        command.extend(("--strict-reference-cameras", *args.strict_reference_cameras))
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.target_seed_calibration is not None:
        command.extend(
            (
                "--target-seed-calibration",
                str(args.target_seed_calibration.expanduser().resolve()),
            )
        )
    if args.scale_baseline_mm is not None:
        command.extend(("--scale-baseline-mm", str(args.scale_baseline_mm)))
    if args.ignore_quality_report:
        command.append("--ignore-quality-report")
    if args.force_accept:
        command.append("--force-accept")
    if args.overwrite_calibration:
        command.append("--overwrite")
    if args.strict:
        command.append("--strict")
    return command


def build_parser() -> argparse.ArgumentParser:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "直接从多相机目录运行全相机对RoMa匹配、共享F诊断和联合标定"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=PROGRAM_VERSION)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--initial-calibration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("multialign_all_roma_result"))
    parser.add_argument("--reference-cameras", nargs="+", required=True)
    parser.add_argument("--target-camera", required=True)
    parser.add_argument("--anchor-camera", required=True)
    parser.add_argument("--scale-reference-camera", required=True)
    parser.add_argument(
        "--matcher-script",
        type=Path,
        default=package_root / "stages" / "matching.py",
    )
    parser.add_argument(
        "--calibrator-script",
        type=Path,
        default=package_root / "stages" / "optimization.py",
    )
    parser.add_argument("--target-seed-calibration", type=Path, default=None)
    parser.add_argument(
        "--step", choices=("prepare", "match", "calibrate", "all"), default="all"
    )
    parser.add_argument("--frame-glob", default="*")
    parser.add_argument("--extensions", nargs="+", default=list(IMAGE_EXTENSIONS))
    parser.add_argument("--include-file", type=Path, default=None)
    parser.add_argument("--exclude-file", type=Path, default=None)
    parser.add_argument("--skip-incomplete", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="试跑前N组；正式标定不要设置")

    match = parser.add_argument_group("全RoMa匹配")
    match.add_argument("--roma-setting", default="fast")
    match.add_argument("--roma-samples", type=int, default=8000)
    match.add_argument(
        "--cross-representation",
        choices=("gray", "structure", "rgb-gray"),
        default="gray",
    )
    match.add_argument("--max-matches-per-group", type=int, default=2400)
    match.add_argument("--grid-cols", type=int, default=24)
    match.add_argument("--grid-rows", type=int, default=18)
    match.add_argument("--grid-max-per-cell", type=int, default=6)
    match.add_argument("--ransac-threshold", type=float, default=1.5)
    match.add_argument("--homography-threshold", type=float, default=3.0)
    match.add_argument("--match-min-group-inliers", type=int, default=30)
    match.add_argument("--overwrite-matches", action="store_true")
    match.add_argument("--diagnose-only", action="store_true")
    match.add_argument("--no-diagnostics", action="store_true")
    match.add_argument("--allow-cpu", action="store_true")
    match.add_argument("--torch-compile", action="store_true")

    calibration = parser.add_argument_group("多相机联合标定")
    calibration.add_argument(
        "--target-model", choices=("fixed", "focal", "focal-pp"), default="focal-pp"
    )
    calibration.add_argument(
        "--reference-distortion",
        choices=("auto", "fixed", "radial1", "radial2", "standard"),
        default="fixed",
    )
    calibration.add_argument(
        "--target-distortion",
        choices=("auto", "fixed", "radial1", "radial2", "standard"),
        default="fixed",
    )
    calibration.add_argument("--target-focal-min", type=float, default=0.55)
    calibration.add_argument("--target-focal-max", type=float, default=1.60)
    calibration.add_argument("--target-pp-bound-fraction", type=float, default=0.08)
    calibration.add_argument("--target-pp-prior-fraction", type=float, default=0.03)
    calibration.add_argument("--scale-baseline-mm", type=float, default=None)
    calibration.add_argument("--baseline-log-sigma", type=float, default=0.25)
    calibration.add_argument("--gauge-log-sigma", type=float, default=0.03)
    calibration.add_argument(
        "--maximum-baseline-ratio",
        type=float,
        default=2.5,
        help="largest allowed ratio between non-zero rig baselines",
    )
    calibration.add_argument("--rotation-prior-deg", type=float, default=12.0)
    calibration.add_argument("--direction-prior-deg", type=float, default=35.0)
    calibration.add_argument("--rotation-bound-deg", type=float, default=20.0)
    calibration.add_argument("--center-bound", type=float, default=3.0)
    calibration.add_argument("--prior-equivalent-points", type=float, default=400.0)
    calibration.add_argument("--shared-point-threshold", type=float, default=2.25)
    calibration.add_argument("--strict-reference-cameras", nargs="*", default=[])
    calibration.add_argument("--strict-shared-point-threshold", type=float, default=1.50)
    calibration.add_argument("--minimum-shared-inlier-ratio", type=float, default=0.25)
    calibration.add_argument("--strict-minimum-shared-inlier-ratio", type=float, default=0.20)
    calibration.add_argument("--shared-filter-iterations", type=int, default=2)
    calibration.add_argument("--calibration-min-group-inliers", type=int, default=30)
    calibration.add_argument("--max-points-per-group", type=int, default=160)
    calibration.add_argument("--loss", choices=("linear", "soft_l1", "huber", "cauchy"), default="cauchy")
    calibration.add_argument("--loss-scale", type=float, default=1.0)
    calibration.add_argument("--max-nfev", type=int, default=200)
    calibration.add_argument("--ignore-quality-report", action="store_true")
    calibration.add_argument("--force-accept", action="store_true")
    calibration.add_argument("--overwrite-calibration", action="store_true")

    parser.add_argument("--common-diagonal", type=float, default=1000.0)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise PipelineError("--limit必须大于0")
    if args.roma_samples < 100:
        raise PipelineError("--roma-samples至少100")
    if args.max_matches_per_group < 8 or args.max_points_per_group < 8:
        raise PipelineError("每组匹配点数量至少8")
    if not 0.0 <= args.validation_fraction < 0.5:
        raise PipelineError("--validation-fraction必须在[0,0.5)内")
    if not 0 < args.target_focal_min < args.target_focal_max:
        raise PipelineError("target focal bounds are invalid")
    unknown_strict = [
        name for name in args.strict_reference_cameras
        if name not in REFERENCE_CAMERAS
    ]
    if unknown_strict:
        raise PipelineError(
            "--strict-reference-cameras contains unknown names: "
            + ", ".join(unknown_strict)
        )
    if args.maximum_baseline_ratio <= 1.0:
        raise PipelineError("--maximum-baseline-ratio必须大于1")
    if args.diagnose_only and args.overwrite_matches:
        raise PipelineError("--diagnose-only不能和--overwrite-matches同时使用")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    report: dict[str, Any] = {
        "schema": "multialign_all_roma_pipeline_v1",
        "program_version": PROGRAM_VERSION,
        "started_utc": utc_now(),
        "status": "running",
    }
    output_root = args.output_root.expanduser().resolve()
    report_path = output_root / "pipeline_report.json"
    try:
        configure_rig(
            args.reference_cameras,
            args.target_camera,
            args.anchor_camera,
            args.scale_reference_camera,
        )
        validate_args(args)
        image_root = resolve_image_root(args.data_root)
        matcher = validate_all_roma_matcher(args.matcher_script)
        calibrator = validate_calibrator(args.calibrator_script)
        initial_calibration = validate_initial_calibration(
            args.initial_calibration, required_cameras=CAMERAS
        )
        if args.target_seed_calibration is not None:
            validate_initial_calibration(
                args.target_seed_calibration,
                required_cameras=(TARGET_CAMERA,),
            )

        groups, inventory = discover_groups(
            image_root,
            args.frame_glob,
            parse_extensions(args.extensions),
            args.include_file,
            args.exclude_file,
            args.skip_incomplete,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        adapter_root = output_root / "dataset_adapter"
        dataset_path = build_dataset_adapter(adapter_root, image_root, groups, inventory)
        geometry_root = output_root / "geometry_all_roma"
        calibration_root = output_root / "calibration"
        logs_root = output_root / "logs"

        report.update(
            {
                "inputs": {
                    "data_root": args.data_root.expanduser().resolve(),
                    "image_root": image_root,
                    "initial_calibration": initial_calibration,
                    "target_seed_calibration": (
                        args.target_seed_calibration.expanduser().resolve()
                        if args.target_seed_calibration is not None
                        else None
                    ),
                    "matcher_script": matcher,
                    "calibrator_script": calibrator,
                },
                "dataset_adapter": dataset_path,
                "inventory": inventory,
                "pairs": list(ALL_PAIRS),
                "camera_roles": {
                    "reference_cameras": list(REFERENCE_CAMERAS),
                    "target_camera": TARGET_CAMERA,
                    "anchor_camera": ANCHOR_CAMERA,
                    "scale_reference_camera": SCALE_REFERENCE_CAMERA,
                },
                "matching_backend": f"RoMaV2 for all {len(ALL_PAIRS)} pairs",
                "matching_scope": "full image, no gray-board ROI",
                "outputs": {
                    "geometry_root": geometry_root,
                    "calibration_root": calibration_root,
                },
            }
        )
        write_json_atomic(report_path, report)
        print(f"输入：{image_root}")
        print(f"完整多相机组：{len(groups)}")
        print(
            f"All {len(ALL_PAIRS)} camera pairs use RoMa v2 on complete images."
        )

        if args.step == "prepare":
            report["status"] = "prepared"
            report["finished_utc"] = utc_now()
            report["seconds"] = time.time() - started
            write_json_atomic(report_path, report)
            print(f"数据清单：{dataset_path}")
            return 0

        if args.step in ("match", "all"):
            command = matching_command(args, matcher, adapter_root, geometry_root)
            report["matching_command"] = command
            write_json_atomic(report_path, report)
            code = run_and_tee(command, logs_root / "01_all_roma_matching.log", args.dry_run)
            report["matching_exit_code"] = code
            report["match_cache_counts"] = match_counts(geometry_root)
            report["matching_summary"] = load_optional_json(
                geometry_root / "reports" / "summary.json"
            )
            write_json_atomic(report_path, report)
            if code != 0:
                raise PipelineError(
                    f"全RoMa匹配/诊断返回{code}；查看 {logs_root / '01_all_roma_matching.log'}"
                )
        else:
            report["match_cache_counts"] = match_counts(geometry_root)

        expected = min(len(groups), args.limit) if args.limit is not None else len(groups)
        counts = report["match_cache_counts"]
        missing_pairs = [pair for pair, count in counts.items() if count < min(expected, 2)]
        if missing_pairs and not args.dry_run:
            raise PipelineError(
                "以下相机对没有至少2组缓存，不能联合标定：" + ", ".join(missing_pairs)
            )

        if args.step == "match":
            report["status"] = "matched"
            report["finished_utc"] = utc_now()
            report["seconds"] = time.time() - started
            write_json_atomic(report_path, report)
            print(f"匹配输出：{geometry_root}")
            return 0

        calibration_report_path = calibration_root / "optimization_report.json"
        calibration_path = calibration_root / "rig_calibration.json"
        if (
            args.step in ("calibrate", "all")
            and calibration_report_path.is_file()
            and calibration_path.is_file()
            and not args.overwrite_calibration
            and not args.dry_run
        ):
            print("联合标定输出已存在，跳过；重算请添加 --overwrite-calibration")
            calibration_code = 0
        else:
            command = calibration_command(
                args, calibrator, geometry_root, calibration_root, initial_calibration
            )
            report["calibration_command"] = command
            write_json_atomic(report_path, report)
            calibration_code = run_and_tee(
                command, logs_root / "02_rig_calibration.log", args.dry_run
            )
        report["calibration_exit_code"] = calibration_code
        calibration_report = load_optional_json(calibration_report_path)
        report["calibration_selection"] = (
            calibration_report.get("selection") if calibration_report else None
        )
        if args.dry_run:
            report["status"] = "dry_run"
            return_code = 0
        elif calibration_code == 0:
            selection = report["calibration_selection"] or {}
            if selection.get("accepted_for_use"):
                report["status"] = "success"
            elif selection.get("forced_output"):
                report["status"] = "forced_diagnostic_output"
            else:
                report["status"] = "completed_unverified"
            return_code = 0
        else:
            report["status"] = "quality_gate_rejected"
            return_code = calibration_code
        report["finished_utc"] = utc_now()
        report["seconds"] = time.time() - started
        write_json_atomic(report_path, report)
        print(f"\n最终标定：{calibration_path}")
        print(f"优化报告：{calibration_report_path}")
        print(f"总流程报告：{report_path}")
        return int(return_code)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["finished_utc"] = utc_now()
        report["seconds"] = time.time() - started
        if output_root.exists():
            write_json_atomic(report_path, report)
        print("\n用户中断；已有RoMa NPZ下次会自动续跑。", file=sys.stderr)
        return 130
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["finished_utc"] = utc_now()
        report["seconds"] = time.time() - started
        if output_root.exists():
            write_json_atomic(report_path, report)
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
