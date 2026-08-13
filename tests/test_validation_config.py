from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.config import TrainConfig


class ValidationConfigTests(unittest.TestCase):
    def test_stockfish_epoch_budget_is_primary(self) -> None:
        config = TrainConfig.from_dict(
            {
                "datasets": ["/tmp/train.binpack"],
                "batch_size": 1024,
                "max_epochs": 10,
                "epoch_size": 1_048_500,
                "start_lambda": 1.0,
                "end_lambda": 0.75,
            }
        )

        self.assertEqual(config.max_epochs, 10)
        self.assertEqual(config.epoch_size, 1_048_500)
        self.assertEqual(config.total_positions, 10_485_000)
        self.assertEqual(config.num_batches_per_epoch, 1024)
        self.assertEqual(config.start_lambda, 1.0)
        self.assertEqual(config.end_lambda, 0.75)

    def test_lambda_key_maps_to_python_field(self) -> None:
        config = TrainConfig.from_dict(
            {
                "datasets": ["/tmp/train.binpack"],
                "max_epochs": 1,
                "epoch_size": 1,
                "lambda": 0.5,
            }
        )

        self.assertEqual(config.lambda_, 0.5)
        self.assertEqual(config.start_lambda, 0.5)
        self.assertEqual(config.end_lambda, 0.5)

    def test_stockfish_style_fields_are_validated(self) -> None:
        valid = {
            "datasets": ["/tmp/train.binpack"],
            "max_epochs": 10,
            "epoch_size": 1_000,
        }
        for key, value in (
            ("max_epochs", 0),
            ("epoch_size", 0),
            ("validation_size", -1),
            ("check_val_every_n_epoch", 0),
            ("network_save_period", -1),
            ("batch_size", 0),
            ("num_workers", 0),
            ("data_loader_queue_size", -1),
            ("random_fen_skipping", -1),
            ("early_fen_skipping", -2),
            ("simple_eval_skipping", -2),
            ("lr", 0.0),
            ("gamma", 0.0),
            ("lambda", 1.1),
            ("start_lambda", -0.1),
            ("in_scaling", 0.0),
            ("out_scaling", 0.0),
            ("pow_exp", 0.0),
            ("qp_asymmetry", -1.0),
            ("nnue2score", 0.0),
        ):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    TrainConfig.from_dict({**valid, key: value})

    def test_start_and_end_lambda_must_be_specified_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "Either both or none"):
            TrainConfig.from_dict(
                {
                    "datasets": ["/tmp/train.binpack"],
                    "max_epochs": 10,
                    "epoch_size": 1_000,
                    "start_lambda": 1.0,
                }
            )

    def test_unknown_config_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown config keys"):
            TrainConfig.from_dict(
                {
                    "datasets": ["/tmp/train.binpack"],
                    "max_epochs": 10,
                    "epoch_size": 1_000,
                    "unknown_budget_knob": 10_000,
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
                    "datasets": ["train"],
                    "validation_datasets": ["valid/*.binpack"],
                    "max_epochs": 10,
                    "epoch_size": 1_000,
                },
                base_dir=root,
            )

            self.assertEqual(
                config.datasets,
                [
                    str((train_dir / "a.binpack").resolve()),
                    str((nested_dir / "b.binpack").resolve()),
                ],
            )
            self.assertEqual(
                config.validation_datasets,
                [str((valid_dir / "c.binpack").resolve())],
            )

    def test_default_config_uses_halfka_v2_hm_stockfish_tail_shape(self) -> None:
        config = TrainConfig(datasets=["/tmp/train.binpack"], max_epochs=1, epoch_size=1)
        config.validate()

        self.assertEqual(config.features, "HalfKAv2_hm")
        self.assertEqual(config.ft_size, 1024)
        self.assertEqual(config.hidden_size, 31)
        self.assertEqual(config.forward_size, 1)
        self.assertEqual(config.fc1_output_size, 32)
        self.assertEqual(config.num_features, 22_528)
        self.assertEqual(config.max_active_features, 32)
        self.assertTrue(config.filtered)
        self.assertTrue(config.wld_filtered)
        self.assertEqual(config.random_fen_skipping, 0)
        self.assertEqual(config.early_fen_skipping, -1)
        self.assertEqual(config.soft_early_fen_skipping, 20)
        self.assertEqual(config.simple_eval_skipping, -1)
        self.assertEqual(config.pc_y0, 0.0)
        self.assertEqual(config.pc_y1, 0.4)
        self.assertEqual(config.pc_y2, 1.0)
        self.assertEqual(config.pc_y3, 1.0)
        self.assertEqual(config.pc_y4, 0.75)

    def test_legacy_fixed_knobs_are_accepted_only_at_default_values(self) -> None:
        config = TrainConfig.from_dict(
            {
                "datasets": ["/tmp/train.binpack"],
                "max_epochs": 1,
                "epoch_size": 1,
                "output_perspective": "stm",
                "optimizer_name": "adamw",
                "weight_decay": 0.0,
                "param_index": 0,
            }
        )

        self.assertEqual(config.lr, 8.75e-4)

        with self.assertRaisesRegex(ValueError, "weight_decay is no longer configurable"):
            TrainConfig.from_dict(
                {
                    "datasets": ["/tmp/train.binpack"],
                    "max_epochs": 1,
                    "epoch_size": 1,
                    "weight_decay": 0.1,
                }
            )

    def test_halfka_v2_hm_shape_is_derived_from_feature_name(self) -> None:
        config = TrainConfig.from_dict(
            {
                "datasets": ["/tmp/train.binpack"],
                "max_epochs": 1,
                "epoch_size": 1,
                "features": "HalfKAv2_hm",
            }
        )

        self.assertEqual(config.features, "HalfKAv2_hm")
        self.assertEqual(config.num_features, 22_528)
        self.assertEqual(config.max_active_features, 32)

    def test_production_configs_load_with_new_architecture_defaults(self) -> None:
        # configs/v1-v3.toml are the repo's real, currently-used training
        # recipes. They don't set hidden_size/forward_size/fc1_output_size
        # themselves, so this is the guarantee that actually matters
        # post-refactor: the config files still load end-to-end and pick up
        # the current architecture's defaults, rather than silently keeping
        # stale values or failing to parse. (v4-v6.toml were deleted in a prior commit;
        # the tests that referenced them asserted specific finetune lr/gamma
        # schedules for those particular files, which no longer exist and
        # aren't reconstructable against v1-v3 without inventing new,
        # unrelated assertions -- so those tests were deleted rather than
        # repointed.)
        root = Path(__file__).resolve().parents[1] / "configs"
        for name in ("v1.toml", "v2.toml", "v3.toml"):
            with self.subTest(config=name):
                config = TrainConfig.from_toml(root / name)
                self.assertTrue(config.datasets, f"{name} should resolve at least one dataset path")
                self.assertEqual(config.features, "HalfKAv2_hm")
                self.assertEqual(config.num_features, 22_528)
                self.assertEqual(config.max_active_features, 32)
                self.assertEqual(config.ft_size, 1024)
                self.assertEqual(config.hidden_size, 31)
                self.assertEqual(config.forward_size, 1)
                self.assertEqual(config.fc1_output_size, 32)


if __name__ == "__main__":
    unittest.main()
