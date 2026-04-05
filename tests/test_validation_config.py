from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.config import TrainConfig


class ValidationConfigTests(unittest.TestCase):
    def test_zero_steps_and_epoch_validation_are_allowed_for_auto_sizing(self) -> None:
        config = TrainConfig.from_dict(
            {
                "train_datasets": ["/tmp/train.binpack"],
                "validation_datasets": ["/tmp/valid.binpack"],
                "steps_per_epoch": 0,
                "validation_every": 0,
                "validation_steps": 0,
            }
        )
        self.assertEqual(config.steps_per_epoch, 0)
        self.assertEqual(config.validation_every, 0)
        self.assertEqual(config.validation_steps, 0)

    def test_score_clip_and_score_scale_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "score_clip": -1.0,
                }
            )
        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "score_scale": 0.0,
                }
            )

    def test_console_mode_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "console_mode": "json",
                }
            )

    def test_dataset_directories_and_globs_expand_to_binpack_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_dir = root / "train"
            nested_dir = train_dir / "nested"
            valid_dir = root / "valid"
            train_dir.mkdir()
            nested_dir.mkdir()
            valid_dir.mkdir()
            (train_dir / "a.binpack").write_bytes(b"")
            (nested_dir / "b.binpack").write_bytes(b"")
            (valid_dir / "c.binpack").write_bytes(b"")
            (valid_dir / "notes.txt").write_text("ignore", encoding="utf-8")

            config = TrainConfig.from_dict(
                {
                    "train_datasets": ["train"],
                    "validation_datasets": ["valid/*.binpack"],
                },
                base_dir=root,
            )

            self.assertEqual(
                config.train_datasets,
                [
                    str((train_dir / "a.binpack").resolve()),
                    str((nested_dir / "b.binpack").resolve()),
                ],
            )
            self.assertEqual(
                config.validation_datasets,
                [str((valid_dir / "c.binpack").resolve())],
            )


if __name__ == "__main__":
    unittest.main()
