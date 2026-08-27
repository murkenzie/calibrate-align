from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from multialign import cli
from multialign.project import init_project


CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(CV2_AVAILABLE, "OpenCV is not installed")
class AlignmentOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        from multialign.pipelines import alignment

        self.alignment = alignment

    def test_dynamic_montage_reflows(self) -> None:
        import numpy as np

        items = [
            (f"camera_{index}", np.zeros((40, 60, 3), dtype=np.uint8))
            for index in range(6)
        ]
        montage = self.alignment.montage_grid(items, (60, 40))
        self.assertGreater(montage.shape[0], 40)
        self.assertGreater(montage.shape[1], 60)

    def test_frame_completion_uses_results_and_diagnostics(self) -> None:
        references = ("left", "right", "top")
        with tempfile.TemporaryDirectory() as temporary:
            frame = Path(temporary) / "frame"
            results = frame / "results"
            diagnostics = frame / "diagnostics"
            results.mkdir(parents=True)
            diagnostics.mkdir(parents=True)
            required = (
                diagnostics / "depth_prior_report.json",
                results / "alignment_maps.npz",
                results / "target_reference.png",
                results / "target_depth.png",
                results / "aligned_grid.png",
                results / "overlay_grid.png",
                results / "overview.png",
            )
            for path in required:
                path.write_bytes(b"placeholder")
            (frame / "frame_report.json").write_text(
                json.dumps({"status": "success"}), encoding="utf-8"
            )
            for camera in references:
                for suffix in ("aligned.png", "valid_mask.png", "overlay_50.jpg"):
                    (results / f"{camera}_{suffix}").write_bytes(b"placeholder")
            self.assertTrue(self.alignment.frame_complete(frame, references))

    def test_high_level_alignment_dry_run_uses_dynamic_roles(self) -> None:
        references = ("left", "right", "top")
        with tempfile.TemporaryDirectory() as temporary:
            paths, _ = init_project(
                Path(temporary) / "project",
                references,
                "target",
                "left",
                "right",
            )
            paths.final_calibration.parent.mkdir(parents=True, exist_ok=True)
            paths.final_calibration.write_text(
                json.dumps({"accepted_for_use": True}), encoding="utf-8"
            )
            for camera in (*references, "target"):
                (paths.scene_images / camera / "frame_001.jpg").write_bytes(
                    b"placeholder"
                )
            paths.depth_anything_root.mkdir(parents=True, exist_ok=True)
            paths.depth_checkpoint.write_bytes(b"placeholder")
            result = cli.main(["align", str(paths.root), "--dry-run"])
            self.assertEqual(result, 0)
            summary = json.loads(
                (paths.alignment_run / "batch_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["camera_roles"]["reference_cameras"], list(references)
            )


if __name__ == "__main__":
    unittest.main()
