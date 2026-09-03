from __future__ import annotations

import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from multialign.stages import depth_prior, geometry, optimization


class WeakCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        optimization.configure_rig(
            ("main", "wide"), "spectral", "main", "wide"
        )

    def test_auto_K_uses_35mm_equivalent_and_resolves_weak_policy(self) -> None:
        sizes = {
            "main": (4096, 3072),
            "wide": (4080, 3072),
            "spectral": (800, 600),
        }
        payload = {
            "cameras": {
                name: {
                    "intrinsics_known": False,
                    "image_size": "auto",
                    "K": "auto",
                    "focal_length_35mm": f35,
                    "dist": [0.0] * 5,
                }
                for name, f35 in (("main", 23.0), ("wide", 14.0), ("spectral", 35.0))
            }
        }
        del payload["cameras"]["wide"]["intrinsics_known"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "initial.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            seeds, _ = optimization.load_camera_seeds(path, sizes)

        expected = math.hypot(*sizes["main"]) * 23.0 / math.hypot(36.0, 24.0)
        self.assertAlmostEqual(seeds["main"].K0[0, 0], expected, places=6)
        self.assertEqual(seeds["main"].seed_source, "focal_length_35mm")
        self.assertFalse(seeds["main"].intrinsics_known)
        self.assertFalse(seeds["wide"].intrinsics_known)

        args = argparse.Namespace(reference_intrinsics="auto", target_model="auto")
        report = optimization.resolve_intrinsic_policies(args, seeds)
        self.assertEqual(args.reference_intrinsics, "weak")
        self.assertEqual(args.target_model, "focal-pp")
        self.assertEqual(report["resolved_reference_intrinsics"], "weak")

    def test_auto_K_cannot_be_marked_as_measured(self) -> None:
        sizes = {name: (640, 480) for name in ("main", "wide", "spectral")}
        payload = {
            "cameras": {
                name: {
                    "intrinsics_known": name == "main",
                    "K": "auto",
                    "focal_ratio": 1.0,
                    "dist": [0.0] * 5,
                }
                for name in sizes
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "initial.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(optimization.CalibrationError):
                optimization.load_camera_seeds(path, sizes)

    def test_pack_unpack_is_a_roundtrip_across_optimization_stages(self) -> None:
        sizes = {name: (1000, 800) for name in ("main", "wide", "spectral")}
        seeds = {
            name: optimization.CameraSeed(
                name=name,
                size_wh=sizes[name],
                source_size_wh=sizes[name],
                K0=np.array(
                    [[focal, 0.0, 500.0], [0.0, focal, 400.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                ),
                dist0=np.zeros(5, dtype=np.float64),
            )
            for name, focal in (("main", 1000.0), ("wide", 800.0), ("spectral", 600.0))
        }
        intrinsics = {name: seed.K0.copy() for name, seed in seeds.items()}
        for name, scale, shift in (
            ("main", 1.10, (20.0, -12.0)),
            ("wide", 0.90, (-16.0, 8.0)),
            ("spectral", 1.20, (10.0, 14.0)),
        ):
            intrinsics[name][0, 0] *= scale
            intrinsics[name][1, 1] *= scale
            intrinsics[name][0, 2] += shift[0]
            intrinsics[name][1, 2] += shift[1]
        distortion = {
            name: np.array([value, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            for name, value in (("main", 0.04), ("wide", -0.03), ("spectral", 0.02))
        }
        model = optimization.RigModel(
            rotations={name: np.eye(3) for name in sizes},
            centers={name: np.zeros(3) for name in sizes},
            intrinsics=intrinsics,
            distortion=distortion,
        )
        args = argparse.Namespace(
            resolved_distortion_modes={name: "radial1" for name in sizes}
        )
        layout = optimization.build_layout(
            False, True, True, True, True, True, args
        )
        packed = optimization.pack_model(
            model,
            {name: np.eye(3) for name in sizes},
            seeds,
            layout,
        )
        unpacked = optimization.unpack_model(
            packed,
            {name: np.eye(3) for name in sizes},
            seeds,
            layout,
            model,
        )
        for name in sizes:
            np.testing.assert_allclose(unpacked.intrinsics[name], model.intrinsics[name])
            np.testing.assert_allclose(unpacked.distortion[name], model.distortion[name])

    def test_selection_cannot_accept_a_stage_that_skipped_required_K(self) -> None:
        model = optimization.RigModel({}, {}, {}, {})

        def stage(
            name: str, score: float, intrinsic_policy_satisfied: bool
        ) -> optimization.StageResult:
            return optimization.StageResult(
                name=name,
                model=model,
                optimizer={},
                metrics={"score": {"validation": score, "train": score}},
                physical={},
                bound_hits=[],
                gate_passed=True,
                gate_reasons=[],
                intrinsic_policy_satisfied=intrinsic_policy_satisfied,
            )

        fixed_seed = stage("fixed_seed", 0.1, False)
        weak_K = stage("weak_K", 0.3, True)
        selected, accepted, eligible = optimization.select_calibration_stage(
            [fixed_seed, weak_K], False
        )
        self.assertIs(selected, weak_K)
        self.assertTrue(accepted)
        self.assertEqual(eligible, [weak_K])

        selected, accepted, eligible = optimization.select_calibration_stage(
            [fixed_seed], False
        )
        self.assertIs(selected, fixed_seed)
        self.assertFalse(accepted)
        self.assertEqual(eligible, [])


class FramePoseRefinementTests(unittest.TestCase):
    @staticmethod
    def _args(**updates: float | int | str) -> argparse.Namespace:
        values: dict[str, float | int | str] = {
            "pose_refinement": "essential",
            "pose_refine_max_samples": 1200,
            "pose_refine_ransac_threshold": 1.5,
            "pose_refine_ransac_max_iters": 10000,
            "pose_refine_homography_threshold": 2.0,
            "pose_refine_max_homography_dominance": 0.95,
            "pose_refine_min_inliers": 80,
            "pose_refine_min_inlier_ratio": 0.50,
            "pose_refine_max_rotation_deg": 3.0,
            "pose_refine_max_translation_deg": 8.0,
            "pose_refine_min_improvement": 0.05,
            "pose_refine_strength": 1.0,
        }
        values.update(updates)
        return argparse.Namespace(**values)

    @staticmethod
    def _synthetic_correspondences(planar: bool = False) -> tuple[np.ndarray, ...]:
        rng = np.random.default_rng(2026)
        depth = (
            np.full(800, 6.0, dtype=np.float64)
            if planar
            else rng.uniform(4.0, 10.0, 800)
        )
        points = np.column_stack(
            (
                rng.uniform(-1.8, 1.8, 800),
                rng.uniform(-1.2, 1.2, 800),
                depth,
            )
        )
        prior = np.eye(4, dtype=np.float64)
        prior[:3, 3] = [0.30, 0.0, 0.0]
        delta_rotation, _ = cv2.Rodrigues(
            np.radians(np.array([0.25, -0.85, 0.35], dtype=np.float64))
        )
        actual = np.eye(4, dtype=np.float64)
        actual[:3, :3] = delta_rotation
        direction_angle = math.radians(3.0)
        actual[:3, 3] = 0.30 * np.array(
            [math.cos(direction_angle), math.sin(direction_angle), 0.0]
        )
        target = points[:, :2] / points[:, 2:3]
        moved = (actual[:3, :3] @ points.T).T + actual[:3, 3]
        reference = moved[:, :2] / moved[:, 2:3]
        focal = 900.0
        noise = rng.normal(0.0, 0.12 / focal, size=target.shape)
        target = target + noise
        reference = reference + rng.normal(0.0, 0.12 / focal, size=reference.shape)
        target_pixels = target * focal + np.array([640.0, 480.0])
        quality = np.ones(len(points), dtype=np.float64)
        return target, reference, target_pixels, quality, prior, actual, np.array([focal])

    def test_small_drift_is_accepted_and_baseline_is_preserved(self) -> None:
        target, reference, pixels, quality, prior, actual, focal = (
            self._synthetic_correspondences()
        )
        refined, report = geometry.refine_reference_from_target_pose(
            target,
            reference,
            pixels,
            quality,
            prior,
            (1280, 960),
            float(focal[0]),
            self._args(),
        )
        self.assertTrue(report["accepted"], report)
        self.assertAlmostEqual(
            np.linalg.norm(refined[:3, 3]), np.linalg.norm(prior[:3, 3]), places=10
        )
        self.assertLess(
            geometry.rotation_angle_deg(refined[:3, :3] @ actual[:3, :3].T),
            0.25,
        )

    def test_drift_outside_gate_falls_back_to_calibration(self) -> None:
        target, reference, pixels, quality, prior, _actual, focal = (
            self._synthetic_correspondences()
        )
        refined, report = geometry.refine_reference_from_target_pose(
            target,
            reference,
            pixels,
            quality,
            prior,
            (1280, 960),
            float(focal[0]),
            self._args(pose_refine_max_rotation_deg=0.20),
        )
        self.assertFalse(report["accepted"])
        np.testing.assert_allclose(refined, prior)

    def test_planar_matches_do_not_override_pose(self) -> None:
        target, reference, pixels, quality, prior, _actual, focal = (
            self._synthetic_correspondences(planar=True)
        )
        refined, report = geometry.refine_reference_from_target_pose(
            target,
            reference,
            pixels,
            quality,
            prior,
            (1280, 960),
            float(focal[0]),
            self._args(),
        )
        self.assertFalse(report["accepted"], report)
        np.testing.assert_allclose(refined, prior)


class PoseHandoffTests(unittest.TestCase):
    def test_geometry_npz_carries_frame_pose_into_depth_stage(self) -> None:
        depth_prior.configure_rig(("main", "wide"), "spectral", "main")
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [0.2, -0.01, 0.03]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "geometry.npz"
            shape = (6, 8)
            np.savez_compressed(
                path,
                target_depth_z=np.ones(shape, dtype=np.float32),
                target_depth_reliable_mask=np.ones(shape, dtype=np.uint8),
                target_depth_confidence=np.ones(shape, dtype=np.float32),
                target_depth_support_count=np.full(shape, 2, dtype=np.uint8),
                reference_from_target__main=pose,
            )
            _depth, _mask, _confidence, _support, overrides, report = (
                depth_prior.load_geometry_npz(path, (8, 6), "reliable", 0)
            )
        np.testing.assert_allclose(overrides["main"], pose)
        self.assertEqual(report["pose_override_cameras"], ["main"])


class StrictRenderTests(unittest.TestCase):
    def test_default_render_mode_is_strict(self) -> None:
        args = depth_prior.parse_args(
            [
                "--reference-cameras", "main", "wide",
                "--target-camera", "spectral",
                "--anchor-camera", "main",
                "--calibration", "calibration.json",
                "--geometry-npz", "geometry.npz",
                "--depth-anything-root", "Depth-Anything-V2",
                "--checkpoint", "depth.pth",
                "--output-dir", "output",
            ]
        )
        self.assertEqual(args.render_mode, "strict")

    def test_zero_completion_keeps_zbuffer_rejection_masked(self) -> None:
        identity = np.eye(4, dtype=np.float64)
        K = np.eye(3, dtype=np.float64)
        camera = depth_prior.CameraModel(
            "reference", K, np.zeros(5), (2, 1), identity
        )
        target = depth_prior.CameraModel(
            "target", K, np.zeros(5), (2, 1), identity
        )
        reference = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        # Both target pixels project onto reference pixel zero.  The farther
        # sample must be rejected and remain black when every fill path is off.
        rays = np.array([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
        depth = np.array([[1.0, 2.0]], dtype=np.float32)
        valid = np.ones((1, 2), dtype=bool)
        rendered = depth_prior.render_reference(
            reference,
            camera,
            target,
            rays,
            depth,
            valid,
            depth,
            "nearest",
            0.04,
            0,
            0.0,
            0,
            0.06,
            0.25,
            0,
            0,
            0.05,
            "adaptive",
            "copy",
            0.0,
        )
        np.testing.assert_array_equal(rendered.sampleable, [[True, True]])
        np.testing.assert_array_equal(rendered.zbuffer_visible, [[True, False]])
        np.testing.assert_array_equal(rendered.visual_mask, rendered.zbuffer_visible)
        self.assertFalse(np.any(rendered.occlusion_filled))
        self.assertFalse(np.any(rendered.occlusion_relaxed_filled))
        self.assertFalse(np.any(rendered.display_filled))
        self.assertFalse(np.any(rendered.texture_structure_refined))
        np.testing.assert_array_equal(rendered.aligned_complete[0, 1], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
