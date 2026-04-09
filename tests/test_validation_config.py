from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.config import TrainConfig


class ValidationConfigTests(unittest.TestCase):
    def test_position_budget_fields_are_allowed_for_auto_interval_validation(self) -> None:
        config = TrainConfig.from_dict(
            {
                "train_datasets": ["/tmp/train.binpack"],
                "validation_datasets": ["/tmp/valid.binpack"],
                "total_train_positions": 10_000,
                "superbatch_positions": 2_000,
                "validation_interval_positions": 0,
                "validation_positions": 0,
            }
        )
        self.assertEqual(config.total_train_positions, 10_000)
        self.assertEqual(config.superbatch_positions, 2_000)
        self.assertEqual(config.validation_interval_positions, 0)
        self.assertEqual(config.validation_positions, 0)
        self.assertEqual(config.feature_set, "a768")

    def test_score_clip_and_score_scale_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "total_train_positions": 10_000,
                    "superbatch_positions": 1_000,
                    "score_clip": -1.0,
                }
            )
        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "total_train_positions": 10_000,
                    "superbatch_positions": 1_000,
                    "score_scale": 0.0,
                }
            )

    def test_prefetch_batches_accepts_zero_and_rejects_negative_values(self) -> None:
        config = TrainConfig.from_dict(
            {
                "train_datasets": ["/tmp/train.binpack"],
                "total_train_positions": 10_000,
                "superbatch_positions": 1_000,
                "prefetch_batches": 0,
            }
        )
        self.assertEqual(config.prefetch_batches, 0)

        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "total_train_positions": 10_000,
                    "superbatch_positions": 1_000,
                    "prefetch_batches": -1,
                }
            )

    def test_console_mode_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "total_train_positions": 10_000,
                    "superbatch_positions": 1_000,
                    "console_mode": "json",
                }
            )

    def test_a768_dual_alias_is_normalized_and_output_buckets_must_be_positive(self) -> None:
        config = TrainConfig.from_dict(
            {
                "train_datasets": ["/tmp/train.binpack"],
                "total_train_positions": 10_000,
                "superbatch_positions": 1_000,
                "feature_set": "a768_dual",
            }
        )
        self.assertEqual(config.feature_set, "a768")

        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "total_train_positions": 10_000,
                    "superbatch_positions": 1_000,
                    "output_buckets": 0,
                }
            )

    def test_position_budget_fields_are_required(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig.from_dict({"train_datasets": ["/tmp/train.binpack"]})

    def test_overlap_between_train_and_validation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/shared.binpack"],
                    "validation_datasets": ["/tmp/shared.binpack"],
                    "total_train_positions": 10_000,
                    "superbatch_positions": 1_000,
                }
            )

    def test_legacy_epoch_fields_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "total_train_positions": 10_000,
                    "superbatch_positions": 1_000,
                    "steps_per_epoch": 100,
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
                    "total_train_positions": 10_000,
                    "superbatch_positions": 1_000,
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

    def test_dual_head_training_fields_and_legacy_eval_lambda_alias_are_supported(self) -> None:
        config = TrainConfig.from_dict(
            {
                "train_datasets": ["/tmp/train.binpack"],
                "total_train_positions": 10_000,
                "superbatch_positions": 1_000,
                "head_type": "dual_value_wdl",
                "optimizer": "ranger",
                "teacher_lambda_start": 1.0,
                "teacher_lambda_end": 0.75,
                "warmup_positions": 500,
                "filter_min_ply": 16,
                "filter_max_abs_score_cp": 1200.0,
                "filter_skip_bestmove_captures": True,
                "filter_wld_skip": True,
            }
        )
        self.assertEqual(config.head_type, "dual_value_wdl")
        self.assertEqual(config.optimizer, "ranger")
        self.assertEqual(config.filter_min_ply, 16)
        self.assertTrue(config.filter_skip_bestmove_captures)
        self.assertTrue(config.filter_wld_skip)

        legacy = TrainConfig.from_dict(
            {
                "train_datasets": ["/tmp/train.binpack"],
                "total_train_positions": 10_000,
                "superbatch_positions": 1_000,
                "eval_lambda": 0.6,
            }
        )
        self.assertEqual(legacy.teacher_lambda_start, 0.6)
        self.assertEqual(legacy.teacher_lambda_end, 0.6)


if __name__ == "__main__":
    unittest.main()
