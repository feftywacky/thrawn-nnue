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
                "epoch_positions": 2_000,
                "validation_interval_positions": 0,
                "validation_positions": 0,
            }
        )
        self.assertEqual(config.total_train_positions, 10_000)
        self.assertEqual(config.epoch_positions, 2_000)
        self.assertEqual(config.validation_interval_positions, 0)
        self.assertEqual(config.validation_positions, 0)
        self.assertEqual(config.feature_set, "halfkp")

    def test_score_clip_and_cp_loss_beta_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "total_train_positions": 10_000,
                    "epoch_positions": 1_000,
                    "score_clip": -1.0,
                }
            )
        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "total_train_positions": 10_000,
                    "epoch_positions": 1_000,
                    "cp_loss_beta": 0.0,
                }
            )

    def test_prefetch_batches_accepts_zero_and_rejects_negative_values(self) -> None:
        config = TrainConfig.from_dict(
            {
                "train_datasets": ["/tmp/train.binpack"],
                "total_train_positions": 10_000,
                "epoch_positions": 1_000,
                "prefetch_batches": 0,
            }
        )
        self.assertEqual(config.prefetch_batches, 0)

        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "total_train_positions": 10_000,
                    "epoch_positions": 1_000,
                    "prefetch_batches": -1,
                }
            )

    def test_console_mode_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "total_train_positions": 10_000,
                    "epoch_positions": 1_000,
                    "console_mode": "json",
                }
            )

    def test_feature_set_must_be_halfkp_and_legacy_keys_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "total_train_positions": 10_000,
                    "epoch_positions": 1_000,
                    "feature_set": "a768",
                }
            )
        with self.assertRaisesRegex(ValueError, "Unknown config keys"):
            TrainConfig.from_dict(
                {
                    "train_datasets": ["/tmp/train.binpack"],
                    "total_train_positions": 10_000,
                    "epoch_positions": 1_000,
                    "score_scale": 1.0,
                    "output_buckets": 8,
                    "eval_lambda": 0.7,
                }
            )

    def test_removed_scheduler_and_superbatch_keys_are_rejected(self) -> None:
        for key, value in (
            ("superbatch_positions", 1_000),
            ("lr_schedule", "exponential"),
            ("lr_drop_fractions", [0.8, 0.95]),
            ("lr_drop_factor", 0.1),
            ("lr_gamma", 0.992),
        ):
            with self.assertRaisesRegex(ValueError, "Unknown config keys"):
                TrainConfig.from_dict(
                    {
                        "train_datasets": ["/tmp/train.binpack"],
                        "total_train_positions": 10_000,
                        "epoch_positions": 1_000,
                        key: value,
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
                    "epoch_positions": 1_000,
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
                    "epoch_positions": 1_000,
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

    def test_default_reference_config_loads_with_halfkp_settings(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "v1.toml"
        config = TrainConfig.from_toml(config_path)

        self.assertEqual(config.run_name, "v1")
        self.assertEqual(config.batch_size, 32_768)
        self.assertEqual(config.total_train_positions, 5_000_000_000)
        self.assertEqual(config.epoch_positions, 100_000_000)
        self.assertEqual(config.ft_size, 256)
        self.assertEqual(config.l1_size, 32)
        self.assertEqual(config.l2_size, 32)
        self.assertEqual(config.num_features, 40_960)
        self.assertEqual(config.max_active_features, 30)
        self.assertEqual(config.score_clip, 4000.0)
        self.assertEqual(config.wdl_lambda, 0.5)


if __name__ == "__main__":
    unittest.main()
