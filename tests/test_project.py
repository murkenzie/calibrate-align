from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multialign.project import (
    ProjectError,
    default_config,
    init_project,
    inspect_project,
    project_layout,
    validate_rig_roles,
)


class ProjectTests(unittest.TestCase):
    def test_default_roles_are_generic(self) -> None:
        config = default_config()
        self.assertEqual(
            config["rig"]["reference_cameras"],
            ["reference_a", "reference_b"],
        )
        self.assertEqual(config["rig"]["target_camera"], "target")
        self.assertNotIn("camera_names", config)

    def test_init_supports_three_references(self) -> None:
        references = ("left", "right", "top")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            paths, _ = init_project(
                root,
                references,
                "target",
                "left",
                "right",
            )
            for image_root in (paths.calibration_images, paths.scene_images):
                for camera in (*references, "target"):
                    self.assertTrue((image_root / camera).is_dir())

            initial = json.loads(paths.initial_calibration.read_text(encoding="utf-8"))
            self.assertTrue(initial["example_only"])
            for camera in references:
                self.assertTrue(initial["cameras"][camera]["intrinsics_known"])
            self.assertFalse(initial["cameras"]["target"]["intrinsics_known"])

            for image_root in (paths.calibration_images, paths.scene_images):
                for camera in (*references, "target"):
                    (image_root / camera / "frame_001.jpg").write_bytes(b"placeholder")
            snapshot = inspect_project(paths)
            self.assertEqual(snapshot.calibration_images.common_stems, ("frame_001",))
            self.assertEqual(snapshot.scene_images.common_stems, ("frame_001",))

            layout = project_layout(paths)
            self.assertIn("results/", layout)
            self.assertIn("diagnostics/", layout)

    def test_role_validation_rejects_invalid_rigs(self) -> None:
        with self.assertRaises(ProjectError):
            validate_rig_roles(("only_one",), "target", "only_one", "only_one")
        with self.assertRaises(ProjectError):
            validate_rig_roles(("left", "right"), "left", "left", "right")
        with self.assertRaises(ProjectError):
            validate_rig_roles(("left", "right"), "target", "left", "left")


if __name__ == "__main__":
    unittest.main()
