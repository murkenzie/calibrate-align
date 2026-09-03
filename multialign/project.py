"""Project roles, paths, templates, and readiness checks."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_REFERENCE_CAMERAS = ("reference_a", "reference_b")
DEFAULT_TARGET_CAMERA = "target"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CONFIG_NAME = "multialign.project.json"
CAMERA_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ProjectError(RuntimeError):
    pass


def validate_rig_roles(
    reference_cameras: Sequence[str],
    target_camera: str,
    anchor_camera: str,
    scale_reference_camera: str,
) -> tuple[str, ...]:
    references = tuple(str(name).strip() for name in reference_cameras)
    if len(references) < 2:
        raise ProjectError("reference_cameras must contain at least two cameras")
    if len(set(references)) != len(references):
        raise ProjectError("reference_cameras cannot contain duplicates")
    all_names = (*references, str(target_camera).strip())
    invalid = [name for name in all_names if not CAMERA_NAME_PATTERN.fullmatch(name)]
    if invalid:
        raise ProjectError(
            "camera names must match [A-Za-z0-9][A-Za-z0-9_.-]*: "
            + ", ".join(repr(name) for name in invalid)
        )
    if target_camera in references:
        raise ProjectError("target_camera must not appear in reference_cameras")
    if anchor_camera not in references:
        raise ProjectError("anchor_camera must be one of reference_cameras")
    if scale_reference_camera not in references:
        raise ProjectError("scale_reference_camera must be one of reference_cameras")
    if scale_reference_camera == anchor_camera:
        raise ProjectError("scale_reference_camera must differ from anchor_camera")
    return references


def default_config(
    reference_cameras: Sequence[str] = DEFAULT_REFERENCE_CAMERAS,
    target_camera: str = DEFAULT_TARGET_CAMERA,
    anchor_camera: str | None = None,
    scale_reference_camera: str | None = None,
) -> dict[str, Any]:
    references = tuple(reference_cameras)
    anchor = anchor_camera or references[0]
    scale_reference = scale_reference_camera or references[1]
    validate_rig_roles(references, target_camera, anchor, scale_reference)
    return {
        "schema_version": 1,
        "rig": {
            "reference_cameras": list(references),
            "target_camera": target_camera,
            "anchor_camera": anchor,
            "scale_reference_camera": scale_reference,
        },
        "paths": {
            "calibration_images": "inputs/calibration/images",
            "scene_images": "inputs/scenes/images",
            "initial_calibration": "inputs/calibration/initial_calibration.json",
            "depth_anything_root": "external/Depth-Anything-V2",
            "depth_checkpoint": "models/depth_anything_v2_vits.pth",
            "calibration_run": "runs/calibration",
            "alignment_run": "runs/alignment",
        },
        "runtime": {
            "roma_setting": "fast",
            "cross_representation": "gray",
            "depth_prior_cameras": list(references[:2]),
            "reference_depth_camera": anchor,
            "final_depth_source": "model-raw",
            "encoder": "vits",
            "device": "auto",
            "precision": "auto",
            "allow_cpu": False,
        },
        "calibration": {
            "reference_intrinsics": "auto",
            "target_model": "auto",
            "reference_distortion": "fixed",
            "target_distortion": "fixed",
            "rig_motion_model": "fixed",
        },
        "alignment": {
            "render_mode": "strict",
            "pose_refinement": "off",
            "pose_refine_ransac_threshold": 1.5,
            "pose_refine_ransac_max_iters": 10000,
            "pose_refine_max_samples": 5000,
            "pose_refine_homography_threshold": 2.0,
            "pose_refine_max_homography_dominance": 0.95,
            "pose_refine_max_rotation_deg": 3.0,
            "pose_refine_max_translation_deg": 8.0,
            "pose_refine_min_inliers": 80,
            "pose_refine_min_inlier_ratio": 0.25,
            "pose_refine_min_improvement": 0.05,
            "pose_refine_strength": 1.0,
        },
    }


def initial_calibration_template(
    reference_cameras: Sequence[str],
    target_camera: str,
    all_intrinsics_unknown: bool = False,
) -> dict[str, Any]:
    cameras: dict[str, Any] = {}
    for index, name in enumerate(reference_cameras):
        if all_intrinsics_unknown:
            cameras[name] = {
                "intrinsics_known": False,
                "image_size": "auto",
                "K": "auto",
                "focal_ratio": 1.0,
                "dist": [0.0] * 5,
                "note": (
                    "Replace focal_ratio with focal_length_35mm when EXIF is "
                    "available. K and image_size are inferred from matched images."
                ),
            }
            continue
        focal = 900.0 + 150.0 * index
        cameras[name] = {
            "intrinsics_known": True,
            "image_size": [1000, 750],
            "K": [
                [focal, 0.0, 500.0],
                [0.0, focal, 375.0],
                [0.0, 0.0, 1.0],
            ],
            "dist": [0.0] * 5,
        }
    if all_intrinsics_unknown:
        cameras[target_camera] = {
            "intrinsics_known": False,
            "image_size": "auto",
            "K": "auto",
            "focal_ratio": 1.0,
            "dist": [0.0] * 5,
            "note": "K and image_size are inferred; focal_ratio is only a weak seed.",
        }
    else:
        cameras[target_camera] = {
            "intrinsics_known": False,
            "image_size": [1000, 750],
            "K": [
                [1000.0, 0.0, 500.0],
                [0.0, 1000.0, 375.0],
                [0.0, 0.0, 1.0],
            ],
            "dist": [0.0] * 5,
            "note": "K is an optimization seed, not a known calibration.",
        }
    return {
        "schema_version": 1,
        "example_only": True,
        "warning": (
            "All K values are weak seeds; add focal_length_35mm when available."
            if all_intrinsics_unknown
            else (
                "Replace every reference camera image_size/K/dist value with its real "
                "intrinsics. Replace the target image_size and provide a reasonable K seed."
            )
        ),
        "cameras": cameras,
    }


PROJECT_GITIGNORE = """# Private inputs, models, and generated outputs
inputs/
models/
external/
runs/
*.log
*.npz
*.npy
*.pkl
*.pickle
*.pth
*.pt
*.ckpt
"""


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config_path: Path
    calibration_images: Path
    scene_images: Path
    initial_calibration: Path
    depth_anything_root: Path
    depth_checkpoint: Path
    calibration_run: Path
    alignment_run: Path
    config: dict[str, Any]
    reference_cameras: tuple[str, ...]
    target_camera: str
    anchor_camera: str
    scale_reference_camera: str

    @property
    def camera_names(self) -> tuple[str, ...]:
        return (*self.reference_cameras, self.target_camera)

    @property
    def final_calibration(self) -> Path:
        return self.calibration_run / "calibration" / "rig_calibration.json"


@dataclass(frozen=True)
class ImageInventory:
    counts: dict[str, int]
    common_stems: tuple[str, ...]
    duplicate_stems: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class ProjectSnapshot:
    paths: ProjectPaths
    calibration_images: ImageInventory
    scene_images: ImageInventory
    initial_exists: bool
    initial_is_template: bool
    initial_error: str | None
    final_calibration_exists: bool
    final_calibration_accepted: bool | None
    final_calibration_error: str | None
    depth_anything_exists: bool
    depth_checkpoint_exists: bool

    @property
    def calibration_ready(self) -> bool:
        return (
            self.initial_exists
            and not self.initial_is_template
            and self.initial_error is None
            and bool(self.calibration_images.common_stems)
            and not self.calibration_images.duplicate_stems
        )

    @property
    def alignment_ready(self) -> bool:
        return (
            self.final_calibration_exists
            and self.final_calibration_accepted is True
            and self.final_calibration_error is None
            and bool(self.scene_images.common_stems)
            and not self.scene_images.duplicate_stems
            and self.depth_anything_exists
            and self.depth_checkpoint_exists
        )


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectError(f"cannot read {label}: {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise ProjectError(f"{label} must contain a JSON object: {path}")
    return value


def _resolve(root: Path, value: Any, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ProjectError(f"paths.{key} must be a non-empty path string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_project(project: Path) -> ProjectPaths:
    root = project.expanduser().resolve()
    config_path = root / CONFIG_NAME
    if not config_path.is_file():
        raise ProjectError(f"cannot find {CONFIG_NAME}: {config_path}; run multialign init")
    config = _read_json(config_path, "project configuration")
    if config.get("schema_version") != 1:
        raise ProjectError(f"unsupported project schema: {config.get('schema_version')!r}")
    rig = config.get("rig")
    if not isinstance(rig, dict):
        raise ProjectError("project configuration is missing the rig object")
    references = rig.get("reference_cameras")
    if not isinstance(references, list):
        raise ProjectError("rig.reference_cameras must be an array")
    target = str(rig.get("target_camera", ""))
    anchor = str(rig.get("anchor_camera", ""))
    scale_reference = str(rig.get("scale_reference_camera", ""))
    normalized_references = validate_rig_roles(
        references, target, anchor, scale_reference
    )
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ProjectError("project configuration is missing the paths object")
    required_paths = tuple(default_config()["paths"])
    missing = [key for key in required_paths if key not in paths]
    if missing:
        raise ProjectError("project configuration is missing paths: " + ", ".join(missing))
    return ProjectPaths(
        root=root,
        config_path=config_path,
        calibration_images=_resolve(root, paths["calibration_images"], "calibration_images"),
        scene_images=_resolve(root, paths["scene_images"], "scene_images"),
        initial_calibration=_resolve(root, paths["initial_calibration"], "initial_calibration"),
        depth_anything_root=_resolve(root, paths["depth_anything_root"], "depth_anything_root"),
        depth_checkpoint=_resolve(root, paths["depth_checkpoint"], "depth_checkpoint"),
        calibration_run=_resolve(root, paths["calibration_run"], "calibration_run"),
        alignment_run=_resolve(root, paths["alignment_run"], "alignment_run"),
        config=config,
        reference_cameras=normalized_references,
        target_camera=target,
        anchor_camera=anchor,
        scale_reference_camera=scale_reference,
    )


def init_project(
    project: Path,
    reference_cameras: Sequence[str] = DEFAULT_REFERENCE_CAMERAS,
    target_camera: str = DEFAULT_TARGET_CAMERA,
    anchor_camera: str | None = None,
    scale_reference_camera: str | None = None,
    all_intrinsics_unknown: bool = False,
    allow_pose_drift: bool = False,
) -> tuple[ProjectPaths, tuple[Path, ...]]:
    root = project.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / CONFIG_NAME
    created: list[Path] = []
    if not config_path.exists():
        config = default_config(
            reference_cameras,
            target_camera,
            anchor_camera,
            scale_reference_camera,
        )
        if all_intrinsics_unknown:
            config["calibration"]["reference_intrinsics"] = "weak"
            config["calibration"]["target_model"] = "focal-pp"
            config["calibration"]["reference_distortion"] = "auto"
            config["calibration"]["target_distortion"] = "auto"
        if allow_pose_drift:
            config["calibration"]["rig_motion_model"] = "small-drift"
            config["alignment"]["pose_refinement"] = "essential"
        _write_json_atomic(config_path, config)
        created.append(config_path)
    paths = load_project(root)
    for image_root in (paths.calibration_images, paths.scene_images):
        for camera in paths.camera_names:
            (image_root / camera).mkdir(parents=True, exist_ok=True)
    paths.depth_anything_root.parent.mkdir(parents=True, exist_ok=True)
    paths.depth_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.calibration_run.parent.mkdir(parents=True, exist_ok=True)
    if not paths.initial_calibration.exists():
        _write_json_atomic(
            paths.initial_calibration,
            initial_calibration_template(
                paths.reference_cameras,
                paths.target_camera,
                all_intrinsics_unknown=all_intrinsics_unknown,
            ),
        )
        created.append(paths.initial_calibration)
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(PROJECT_GITIGNORE, encoding="utf-8", newline="\n")
        created.append(gitignore)
    return paths, tuple(created)


def _index_images(root: Path, camera_names: Sequence[str]) -> ImageInventory:
    indexed: dict[str, dict[str, list[Path]]] = {}
    counts: dict[str, int] = {}
    for camera in camera_names:
        by_stem: dict[str, list[Path]] = {}
        directory = root / camera
        if directory.is_dir():
            for path in directory.iterdir():
                if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS:
                    by_stem.setdefault(path.stem.casefold(), []).append(path)
        indexed[camera] = by_stem
        counts[camera] = sum(len(items) for items in by_stem.values())
    common = set.intersection(*(set(indexed[name]) for name in camera_names))
    duplicates: dict[str, tuple[str, ...]] = {}
    for camera in camera_names:
        for stem, paths in indexed[camera].items():
            if len(paths) > 1:
                duplicates[f"{camera}/{stem}"] = tuple(sorted(path.name for path in paths))
    return ImageInventory(counts, tuple(sorted(common)), duplicates)


def _inspect_initial(path: Path, camera_names: Sequence[str]) -> tuple[bool, bool, str | None]:
    if not path.is_file():
        return False, False, None
    try:
        value = _read_json(path, "initial calibration")
        cameras = value.get("cameras")
        if not isinstance(cameras, dict) or any(name not in cameras for name in camera_names):
            return True, bool(value.get("example_only")), "incomplete cameras object"
        return True, bool(value.get("example_only")), None
    except ProjectError as exc:
        return True, False, str(exc)


def _inspect_final(path: Path) -> tuple[bool, bool | None, str | None]:
    if not path.is_file():
        return False, None, None
    try:
        value = _read_json(path, "final calibration")
        return True, value.get("accepted_for_use"), None
    except ProjectError as exc:
        return True, None, str(exc)


def inspect_project(project: Path | ProjectPaths) -> ProjectSnapshot:
    paths = project if isinstance(project, ProjectPaths) else load_project(project)
    initial_exists, initial_is_template, initial_error = _inspect_initial(
        paths.initial_calibration, paths.camera_names
    )
    final_exists, final_accepted, final_error = _inspect_final(paths.final_calibration)
    return ProjectSnapshot(
        paths=paths,
        calibration_images=_index_images(paths.calibration_images, paths.camera_names),
        scene_images=_index_images(paths.scene_images, paths.camera_names),
        initial_exists=initial_exists,
        initial_is_template=initial_is_template,
        initial_error=initial_error,
        final_calibration_exists=final_exists,
        final_calibration_accepted=final_accepted,
        final_calibration_error=final_error,
        depth_anything_exists=paths.depth_anything_root.is_dir(),
        depth_checkpoint_exists=paths.depth_checkpoint.is_file(),
    )


def project_layout(paths: ProjectPaths | None = None) -> str:
    references = (
        ",".join(paths.reference_cameras) if paths else "reference_a,reference_b,..."
    )
    target = paths.target_camera if paths else "target"
    return f"""PROJECT/
├── {CONFIG_NAME}
├── .gitignore
├── inputs/
│   ├── calibration/
│   │   ├── initial_calibration.json
│   │   └── images/{{{references},{target}}}/FRAME.*
│   └── scenes/
│       └── images/{{{references},{target}}}/FRAME.*
├── models/depth_anything_v2_vits.pth
├── external/Depth-Anything-V2/
└── runs/
    ├── calibration/
    │   ├── dataset_adapter/
    │   ├── geometry_all_roma/
    │   └── calibration/rig_calibration.json
    └── alignment/
        ├── geometry/FRAME/
        ├── frames/FRAME/
        │   ├── results/
        │   ├── diagnostics/
        │   └── frame_report.json
        ├── logs/
        ├── batch_summary.csv
        ├── batch_summary.json
        └── index.html"""


def describe_inventory(
    inventory: ImageInventory, camera_names: Sequence[str]
) -> str:
    counts = ", ".join(f"{name}={inventory.counts[name]}" for name in camera_names)
    return f"common frames={len(inventory.common_stems)}; {counts}"


def managed_option_present(values: Iterable[str], options: set[str]) -> str | None:
    for value in values:
        for option in options:
            if value == option or value.startswith(option + "="):
                return option
    return None
