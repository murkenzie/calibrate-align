# Algorithm pipeline

This document describes the algorithm independently of camera names and sensor
modality.

## 1. Discover synchronized captures

The calibration runner indexes every configured camera directory by file stem.
It creates a small metadata adapter containing the selected paths and explicit
camera roles. Images are not copied, resized, or rewritten.

The default behavior stops on incomplete captures or duplicate stems. This is a
data-integrity gate: silently shifting one camera by one frame invalidates the
fixed-rig geometry.

## 2. Match every camera pair

For `R` references and one target, the number of unordered pairs is:

\[
\binom{R+1}{2} = \frac{R(R+1)}{2}.
\]

Each pair uses the same RoMa v2 sparse-matching backend over the full images.
Reference/reference pairs use RGB. Pairs containing the target may use gray,
RGB-gray, or a three-channel structure representation to accommodate a
cross-modal target.

The stage then:

1. balances samples over both image grids;
2. fits a robust fundamental matrix independently per capture;
3. fits one shared fundamental matrix for the fixed rig;
4. evaluates it on held-out captures;
5. reports homography dominance to expose planar or low-parallax data.

Only sparse correspondences are saved. RoMa flow is not a final image warp.

## 3. Jointly calibrate the rig

The optimizer creates a shared pose graph across all cameras:

- the anchor reference is the identity pose;
- known reference intrinsics are fixed by default;
- target focal length and principal point are optimized within bounded priors;
- distortion freedom is selected from actual observed image radius;
- rotations, camera centres, and allowed intrinsics are refined with a robust
  loss;
- train/validation residuals, pose-graph connectivity, baseline ratios,
  parameter bounds, and degeneracy metrics form the acceptance gate.

Translation signs are selected jointly over all pairwise and cycle evidence.
The search therefore scales with the actual number of non-anchor cameras rather
than assuming a specific rig size.

The optimizer writes `rig_calibration.json`. Alignment accepts it only when
`accepted_for_use` is true unless a diagnostic override is explicitly requested.

## 4. Estimate strict target-view geometry

For each scene frame:

1. intrinsics and fixed poses generate coarse reference-to-target hypotheses;
2. RoMa proposes reference/target correspondences;
3. forward/backward, epipolar, reprojection, and triangulation-angle gates remove
   inconsistent points;
4. each reference independently produces a target-grid depth candidate;
5. multi-view consistency and confidence fuse the candidates;
6. Z-buffer visibility produces strict reference-to-target maps.

Pixels without reliable geometric support remain invalid. Optional geometric
completion is saved separately from strict measurement support.

## 5. Anchor dense depth priors

Depth Anything V2 produces relative depth for selected references. By default,
the first two references are used for this expensive stage; this is configurable
with `runtime.depth_prior_cameras`. All references still contribute to
calibration/geometry and all are rendered to the target.

For each selected reference, the stage robustly fits an affine transform in
inverse-depth space against reliable triangulated anchors. The network output is
never treated as metric depth without this scale-and-shift fit.

The learned priors are fused with geometry-based conflict checks. One configured
reference surface is preserved wherever its learned projection is valid, while
other priors fill genuine gaps. Optional target-edge and segmentation guidance
may refine object boundaries.

## 6. Render once into the target grid

Each original reference image is sampled once through the final calibrated
target-to-reference map. Rendering applies:

- target-depth projection;
- Z-buffer rejection;
- edge-aware interpolation;
- same-surface fill for narrow occlusion gaps;
- a bounded relaxed pass for small non-border components;
- structure-continuation inpainting only inside geometrically authorized masks.

Strict renders and masks remain separate from visually complete renders. Large
disocclusions and field-of-view loss stay unresolved instead of being invented.

## 7. Package outputs

The batch runner moves stable consumer-facing files into `results/`, retains
quality-control artifacts in `diagnostics/`, writes a per-frame JSON report, and
updates batch CSV/JSON summaries plus a responsive HTML gallery.

The packaging layer does not change the algorithm or resample the lossless
aligned PNG files. Only preview grids are resized and dynamically tiled.

## Assumptions and failure modes

- Camera poses are rigid during both calibration and use.
- Captures are synchronized closely enough for the scene to be effectively
  static.
- Calibration data spans multiple depths and useful common image area.
- Cross-modal views retain enough common structure for matching.
- A monocular depth prior improves density but does not replace calibrated
  geometry or make absolute scale observable.
- Highly repetitive texture, severe occlusion, rolling-shutter motion, planar
  data, and weak baseline can all reduce reliability.
