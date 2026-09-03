# Project and output contract

The directory contract prevents one stage from accidentally consuming files
from another rig, dataset, or run. High-level commands resolve these paths from
`multialign.project.json`; managed path arguments cannot be overridden on the
same command line.

## Configuration

```json
{
  "schema_version": 1,
  "rig": {
    "reference_cameras": ["camera_left", "camera_right"],
    "target_camera": "camera_target",
    "anchor_camera": "camera_left",
    "scale_reference_camera": "camera_right"
  },
  "calibration": {
    "reference_intrinsics": "auto",
    "target_model": "auto",
    "reference_distortion": "fixed",
    "target_distortion": "fixed",
    "rig_motion_model": "fixed"
  },
  "alignment": {
    "render_mode": "strict",
    "pose_refinement": "off"
  }
}
```

At least two unique reference names are required. The target name must be
different. Anchor and scale-reference cameras must both be references and must
be different from each other.

## Inputs

```text
inputs/
├── calibration/
│   ├── initial_calibration.json
│   └── images/CAMERA/FRAME.*
└── scenes/images/CAMERA/FRAME.*
```

Files sharing one stem are treated as a synchronized capture. Missing cameras
or duplicate stems stop the default workflow instead of silently pairing the
wrong images.

The initial calibration format is:

```json
{
  "schema_version": 1,
  "example_only": false,
  "cameras": {
    "camera_left": {
      "intrinsics_known": true,
      "image_size": [1920, 1080],
      "K": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
      "dist": [0, 0, 0, 0, 0]
    },
    "camera_target": {
      "intrinsics_known": false,
      "image_size": [1280, 1024],
      "K": [[900, 0, 640], [0, 900, 512], [0, 0, 1]],
      "dist": [0, 0, 0, 0, 0]
    }
  }
}
```

Measured `K` values use numeric `image_size` and matrices as above. An unknown
camera may instead use the following weak-seed form:

```json
{
  "intrinsics_known": false,
  "image_size": "auto",
  "K": "auto",
  "focal_length_35mm": 23.0,
  "principal_point_fraction": [0.5, 0.5],
  "dist": [0, 0, 0, 0, 0]
}
```

`focal_ratio` may replace `focal_length_35mm`; it multiplies the larger image
dimension. Both forms are optimization seeds, not claims that the intrinsics
are measured.

## Stage handoffs

| Producer | Authoritative output | Consumer |
|---|---|---|
| Dataset adapter | `runs/calibration/dataset_adapter/metadata/dataset.json` | All-pair matcher |
| All-pair matcher | `geometry_all_roma/matches/**/*.npz` and `reports/summary.json` | Joint optimizer |
| Joint optimizer | `calibration/rig_calibration.json` | Scene geometry |
| Scene geometry | `alignment/geometry/FRAME/depth_alignment_maps.npz` | Depth-prior stage |
| Depth-prior stage | `alignment/frames/FRAME/results/alignment_maps.npz` | Downstream processing |
| Batch runner | `batch_summary.csv`, `batch_summary.json`, `index.html` | Review and automation |

The accepted calibration is always:

```text
runs/calibration/calibration/rig_calibration.json
```

## Calibration run

```text
runs/calibration/
├── pipeline_report.json
├── dataset_adapter/
├── geometry_all_roma/
│   ├── matches/CAMERA0__CAMERA1/FRAME.npz
│   ├── diagnostics/
│   └── reports/
├── calibration/
│   ├── rig_calibration.json
│   ├── optimization_report.json
│   └── groups.csv
└── logs/
```

The generic pose convention is:

```text
X_camera = R_anchor_to_camera @ X_anchor + T_anchor_to_camera
C_anchor = -R.T @ T
```

Without a measured baseline, translation uses a normalized gauge. With
`--scale-baseline-mm`, the translation unit is millimetres.

When per-frame pose refinement is accepted, the geometry NPZ also stores one
`reference_from_target__CAMERA` 4x4 transform per reference. These transforms
retain the nominal baseline scale and override only that frame's relative pose
in depth anchoring and rendering.

## Alignment run

```text
runs/alignment/
├── geometry/FRAME/
│   ├── depth_alignment_maps.npz
│   └── alignment_report.json
├── frames/FRAME/
│   ├── results/
│   │   ├── target_reference.png
│   │   ├── target_depth.png
│   │   ├── target_depth_confidence.png
│   │   ├── target_depth_strict_mask.png
│   │   ├── alignment_maps.npz
│   │   ├── CAMERA_aligned.png
│   │   ├── CAMERA_valid_mask.png
│   │   ├── CAMERA_overlay_50.jpg
│   │   ├── aligned_grid.png
│   │   ├── overlay_grid.png
│   │   └── overview.png
│   ├── diagnostics/
│   │   ├── depth_prior_report.json
│   │   ├── debug_depth_grid.png
│   │   ├── debug_render_grid.png
│   │   └── additional quality-control artifacts
│   └── frame_report.json
├── logs/
├── batch_summary.csv
├── batch_summary.json
└── index.html
```

### Which files to consume

| Need | File |
|---|---|
| Lossless aligned image | `results/CAMERA_aligned.png` |
| Exact valid visible mask (255 valid, 0 invalid) | `results/CAMERA_valid_mask.png` |
| Human alignment check | `results/CAMERA_overlay_50.jpg` |
| Dynamic contact sheet | `results/overview.png` |
| Dense target depth preview | `results/target_depth.png` |
| Numeric maps and masks | `results/alignment_maps.npz` |
| Per-frame status and paths | `frame_report.json` |
| Full depth/render statistics | `diagnostics/depth_prior_report.json` |
| Batch status | `batch_summary.csv` or `batch_summary.json` |

The overview is a preview. It automatically changes row/column count as
reference cameras are added. In the default strict mode, each lossless aligned
PNG is zero outside its exact `CAMERA_valid_mask.png`; occlusions are not filled.

## Resume behavior

- Compatible match caches are reused.
- Per-frame geometry is reused unless `--rerun-geometry` is supplied.
- Complete final frames are skipped unless `--overwrite` is supplied.
- Incomplete markers prevent partial outputs from being treated as successful.
- Geometry and depth inference run in separate subprocesses so GPU memory is
  released between stages and frames.
