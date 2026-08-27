"""Unified command-line interface for generic fixed-rig alignment."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .project import (
    DEFAULT_REFERENCE_CAMERAS,
    DEFAULT_TARGET_CAMERA,
    ProjectError,
    ProjectPaths,
    describe_inventory,
    init_project,
    inspect_project,
    load_project,
    managed_option_present,
    project_layout,
)


CALIBRATION_MANAGED = {
    "--data-root",
    "--initial-calibration",
    "--output-root",
    "--reference-cameras",
    "--target-camera",
    "--anchor-camera",
    "--scale-reference-camera",
}
ALIGNMENT_MANAGED = {
    "--dataset-root",
    "--calibration",
    "--output-root",
    "--depth-anything-root",
    "--checkpoint",
    "--reference-cameras",
    "--target-camera",
    "--anchor-camera",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multialign",
        description=(
            "Calibrate a fixed rig with 2+ known-intrinsics reference cameras and "
            "align them into one unknown-intrinsics target camera"
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create a safe project layout and templates")
    init.add_argument("project", nargs="?", type=Path, default=Path("."))
    init.add_argument(
        "--references",
        nargs="+",
        default=list(DEFAULT_REFERENCE_CAMERAS),
        metavar="CAMERA",
        help="two or more cameras with known intrinsics",
    )
    init.add_argument("--target", default=DEFAULT_TARGET_CAMERA)
    init.add_argument("--anchor", default=None)
    init.add_argument("--scale-reference", default=None)

    layout = commands.add_parser("layout", help="show the stage-to-stage file contract")
    layout.add_argument("project", nargs="?", type=Path)

    status = commands.add_parser("status", help="inspect inputs and recommend the next step")
    status.add_argument("project", nargs="?", type=Path, default=Path("."))

    calibrate = commands.add_parser(
        "calibrate",
        help="prepare data, match all camera pairs, and jointly calibrate the rig",
    )
    calibrate.add_argument("project", nargs="?", type=Path, default=Path("."))

    align = commands.add_parser(
        "align",
        help="load the accepted calibration and batch-align references to the target",
    )
    align.add_argument("project", nargs="?", type=Path, default=Path("."))

    run = commands.add_parser("run", help="calibrate when needed, then align")
    run.add_argument("project", nargs="?", type=Path, default=Path("."))
    run.add_argument("--force-recalibrate", action="store_true")
    run.add_argument("--overwrite-alignment", action="store_true")
    run.add_argument("--allow-cpu", action="store_true")
    run.add_argument("--limit", type=int)
    run.add_argument("--dry-run", action="store_true")
    return parser


def _runtime(paths: ProjectPaths) -> dict:
    value = paths.config.get("runtime", {})
    if not isinstance(value, dict):
        raise ProjectError("runtime must be a JSON object")
    return value


def _role_args(paths: ProjectPaths, *, include_scale: bool) -> list[str]:
    output = [
        "--reference-cameras",
        *paths.reference_cameras,
        "--target-camera",
        paths.target_camera,
        "--anchor-camera",
        paths.anchor_camera,
    ]
    if include_scale:
        output.extend(("--scale-reference-camera", paths.scale_reference_camera))
    return output


def _calibration_args(paths: ProjectPaths, extra: Sequence[str]) -> list[str]:
    blocked = managed_option_present(extra, CALIBRATION_MANAGED)
    if blocked:
        raise ProjectError(f"{blocked} is managed by multialign.project.json")
    runtime = _runtime(paths)
    output = [
        "--data-root",
        str(paths.calibration_images),
        "--initial-calibration",
        str(paths.initial_calibration),
        "--output-root",
        str(paths.calibration_run),
        *_role_args(paths, include_scale=True),
        "--roma-setting",
        str(runtime.get("roma_setting", "fast")),
        "--cross-representation",
        str(runtime.get("cross_representation", "gray")),
    ]
    if bool(runtime.get("allow_cpu", False)):
        output.append("--allow-cpu")
    output.extend(extra)
    return output


def _alignment_args(paths: ProjectPaths, extra: Sequence[str]) -> list[str]:
    blocked = managed_option_present(extra, ALIGNMENT_MANAGED)
    if blocked:
        raise ProjectError(f"{blocked} is managed by multialign.project.json")
    runtime = _runtime(paths)
    prior_cameras = runtime.get(
        "depth_prior_cameras", list(paths.reference_cameras[:2])
    )
    if not isinstance(prior_cameras, list) or not prior_cameras:
        raise ProjectError("runtime.depth_prior_cameras must be a non-empty array")
    unknown = [name for name in prior_cameras if name not in paths.reference_cameras]
    if unknown:
        raise ProjectError(
            "runtime.depth_prior_cameras contains non-reference cameras: "
            + ", ".join(unknown)
        )
    reference_depth = str(runtime.get("reference_depth_camera", paths.anchor_camera))
    if reference_depth not in prior_cameras:
        raise ProjectError(
            "runtime.reference_depth_camera must appear in runtime.depth_prior_cameras"
        )
    output = [
        "--dataset-root",
        str(paths.scene_images),
        "--calibration",
        str(paths.final_calibration),
        "--output-root",
        str(paths.alignment_run),
        "--depth-anything-root",
        str(paths.depth_anything_root),
        "--checkpoint",
        str(paths.depth_checkpoint),
        *_role_args(paths, include_scale=False),
        "--roma-setting",
        str(runtime.get("roma_setting", "fast")),
        "--encoder",
        str(runtime.get("encoder", "vits")),
        "--device",
        str(runtime.get("device", "auto")),
        "--precision",
        str(runtime.get("precision", "auto")),
        "--prior-cameras",
        *(str(value) for value in prior_cameras),
        "--reference-depth-camera",
        reference_depth,
        "--final-depth-source",
        str(runtime.get("final_depth_source", "model-raw")),
    ]
    if bool(runtime.get("allow_cpu", False)):
        output.append("--allow-cpu")
    output.extend(extra)
    return output


def _require_calibration_inputs(paths: ProjectPaths) -> None:
    snapshot = inspect_project(paths)
    if snapshot.calibration_ready:
        return
    reasons: list[str] = []
    if not snapshot.initial_exists:
        reasons.append("initial calibration is missing")
    elif snapshot.initial_is_template:
        reasons.append("initial calibration still contains the example_only marker")
    elif snapshot.initial_error:
        reasons.append(snapshot.initial_error)
    if not snapshot.calibration_images.common_stems:
        reasons.append("there are no complete multi-camera calibration frames")
    if snapshot.calibration_images.duplicate_stems:
        reasons.append("duplicate stems exist within a camera directory")
    raise ProjectError("calibration is not ready: " + "; ".join(reasons))


def _require_alignment_inputs(
    paths: ProjectPaths, allow_unaccepted: bool = False
) -> None:
    snapshot = inspect_project(paths)
    reasons: list[str] = []
    if not snapshot.final_calibration_exists:
        reasons.append("accepted rig calibration is missing")
    elif snapshot.final_calibration_error:
        reasons.append(snapshot.final_calibration_error)
    elif snapshot.final_calibration_accepted is not True and not allow_unaccepted:
        reasons.append("calibration did not pass accepted_for_use")
    if not snapshot.scene_images.common_stems:
        reasons.append("there are no complete multi-camera scene frames")
    if snapshot.scene_images.duplicate_stems:
        reasons.append("duplicate stems exist within a scene camera directory")
    if not snapshot.depth_anything_exists:
        reasons.append("Depth Anything V2 source directory is missing")
    if not snapshot.depth_checkpoint_exists:
        reasons.append("Depth Anything V2 checkpoint is missing")
    if reasons:
        raise ProjectError("alignment is not ready: " + "; ".join(reasons))


def _print_status(paths: ProjectPaths) -> int:
    snapshot = inspect_project(paths)
    print(f"Project: {paths.root}")
    print(
        "Rig: references="
        + ",".join(paths.reference_cameras)
        + f"; target={paths.target_camera}; anchor={paths.anchor_camera}"
    )
    print(
        "Calibration images: "
        + describe_inventory(snapshot.calibration_images, paths.camera_names)
    )
    print(
        "Scene images: "
        + describe_inventory(snapshot.scene_images, paths.camera_names)
    )
    print(f"Initial calibration: {paths.initial_calibration}")
    print(f"Final calibration: {paths.final_calibration}")
    if snapshot.alignment_ready:
        print(f"Next: multialign align {shlex.quote(str(paths.root))}")
        return 0
    if snapshot.calibration_ready:
        print(f"Next: multialign calibrate {shlex.quote(str(paths.root))}")
        return 0
    print("Next: complete the missing inputs and run multialign status again")
    return 1


def _run_calibration(paths: ProjectPaths, extra: Sequence[str]) -> int:
    _require_calibration_inputs(paths)
    from .pipelines.calibration import main as calibration_main

    return int(calibration_main(_calibration_args(paths, extra)))


def _run_alignment(paths: ProjectPaths, extra: Sequence[str]) -> int:
    allow_unaccepted = "--allow-unaccepted-calibration" in extra
    _require_alignment_inputs(paths, allow_unaccepted=allow_unaccepted)
    from .pipelines.alignment import main as alignment_main

    return int(alignment_main(_alignment_args(paths, extra)))


def _run_all(paths: ProjectPaths, args: argparse.Namespace) -> int:
    snapshot = inspect_project(paths)
    if args.force_recalibrate or not (
        snapshot.final_calibration_exists
        and snapshot.final_calibration_accepted is True
    ):
        calibration_extra: list[str] = []
        if args.allow_cpu:
            calibration_extra.append("--allow-cpu")
        if args.limit is not None:
            calibration_extra.extend(("--limit", str(args.limit)))
        if args.dry_run:
            calibration_extra.append("--dry-run")
        if args.force_recalibrate:
            calibration_extra.append("--overwrite-calibration")
        result = _run_calibration(paths, calibration_extra)
        if result != 0 or args.dry_run:
            return result
    alignment_extra: list[str] = []
    if args.allow_cpu:
        alignment_extra.append("--allow-cpu")
    if args.limit is not None:
        alignment_extra.extend(("--limit", str(args.limit)))
    if args.dry_run:
        alignment_extra.append("--dry-run")
    if args.overwrite_alignment:
        alignment_extra.append("--overwrite")
    return _run_alignment(paths, alignment_extra)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    try:
        if args.command == "init":
            if extra:
                parser.error("unrecognized arguments: " + " ".join(extra))
            paths, created = init_project(
                args.project,
                args.references,
                args.target,
                args.anchor,
                args.scale_reference,
            )
            print(f"Project initialized: {paths.root}")
            for path in created:
                print(f"  created {path}")
            print(project_layout(paths))
            return 0
        if args.command == "layout":
            if extra:
                parser.error("unrecognized arguments: " + " ".join(extra))
            paths = load_project(args.project) if args.project is not None else None
            print(project_layout(paths))
            return 0
        paths = load_project(args.project)
        if args.command == "status":
            if extra:
                parser.error("unrecognized arguments: " + " ".join(extra))
            return _print_status(paths)
        if args.command == "calibrate":
            return _run_calibration(paths, extra)
        if args.command == "align":
            return _run_alignment(paths, extra)
        if args.command == "run":
            if extra:
                parser.error("unrecognized arguments: " + " ".join(extra))
            return _run_all(paths, args)
        raise ProjectError(f"unknown command: {args.command}")
    except ProjectError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
