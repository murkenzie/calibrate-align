from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from multialign import cli
from multialign.pipelines import calibration
from multialign.project import init_project


class CalibrationPipelineTests(unittest.TestCase):
    def test_pair_count_scales_with_rig_size(self) -> None:
        for count in (2, 3, 5):
            references = tuple(f"ref_{index}" for index in range(count))
            calibration.configure_rig(
                references,
                "target",
                references[0],
                references[1],
            )
            expected = math.comb(count + 1, 2)
            self.assertEqual(len(calibration.ALL_PAIRS), expected)
            self.assertEqual(
                calibration.CAMERAS,
                (*references, "target"),
            )

    def test_adapter_records_dynamic_camera_roles(self) -> None:
        references = ("left", "right", "upper")
        calibration.configure_rig(references, "target", "left", "right")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            for camera in (*references, "target"):
                directory = image_root / camera
                directory.mkdir(parents=True)
                (directory / "frame_001.jpg").write_bytes(b"placeholder")

            groups, inventory = calibration.discover_groups(
                image_root,
                "*",
                {".jpg"},
                None,
                None,
                False,
            )
            self.assertEqual(len(groups), 1)
            self.assertEqual(set(groups[0]["references"]), set(references))
            self.assertEqual(groups[0]["target"]["camera"], "target")

            adapter = root / "adapter"
            dataset_path = calibration.build_dataset_adapter(
                adapter, image_root, groups, inventory
            )
            dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
            self.assertEqual(
                dataset["camera_roles"]["reference_cameras"],
                list(references),
            )
            self.assertEqual(dataset["camera_roles"]["anchor_camera"], "left")
            self.assertEqual(dataset["camera_roles"]["target_camera"], "target")

    def test_high_level_calibration_dry_run_connects_all_stages(self) -> None:
        references = ("left", "right", "upper")
        with tempfile.TemporaryDirectory() as temporary:
            paths, _ = init_project(
                Path(temporary) / "project",
                references,
                "target",
                "left",
                "right",
            )
            initial = json.loads(paths.initial_calibration.read_text(encoding="utf-8"))
            initial["example_only"] = False
            paths.initial_calibration.write_text(
                json.dumps(initial), encoding="utf-8"
            )
            for camera in (*references, "target"):
                (paths.calibration_images / camera / "frame_001.jpg").write_bytes(
                    b"placeholder"
                )
            result = cli.main(["calibrate", str(paths.root), "--dry-run"])
            self.assertEqual(result, 0)

    def test_unknown_intrinsics_profile_reaches_calibrator(self) -> None:
        references = ("main", "wide")
        with tempfile.TemporaryDirectory() as temporary:
            paths, _ = init_project(
                Path(temporary) / "project",
                references,
                "spectral",
                "main",
                "wide",
                all_intrinsics_unknown=True,
                allow_pose_drift=True,
            )
            initial = json.loads(paths.initial_calibration.read_text(encoding="utf-8"))
            initial["example_only"] = False
            paths.initial_calibration.write_text(
                json.dumps(initial), encoding="utf-8"
            )
            for camera in (*references, "spectral"):
                (paths.calibration_images / camera / "frame_001.jpg").write_bytes(
                    b"placeholder"
                )
            result = cli.main(["calibrate", str(paths.root), "--dry-run"])
            self.assertEqual(result, 0)
            report = json.loads(
                (paths.calibration_run / "pipeline_report.json").read_text(
                    encoding="utf-8"
                )
            )
            command = report["calibration_command"]
            self.assertEqual(
                command[command.index("--reference-intrinsics") + 1], "weak"
            )
            self.assertEqual(
                command[command.index("--rig-motion-model") + 1], "small-drift"
            )


if __name__ == "__main__":
    unittest.main()
