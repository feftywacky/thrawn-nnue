from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.config import TrainConfig


class ValidationConfigTests(unittest.TestCase):
    def test_validation_steps_required_when_validation_datasets_present(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "validation_datasets": ["/tmp/valid.binpack"],
                    "validation_steps": 0,
                }
            )

    def test_validation_every_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "validation_every": 0,
                }
            )

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


if __name__ == "__main__":
    unittest.main()
