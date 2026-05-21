from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ModuleNotFoundError:
    torch = None

from thrawn_nnue.native import NativeBatch, write_fixture_binpack
from thrawn_nnue.config import TrainConfig
from thrawn_nnue.training import (
    _advance_scheduler_for_epoch_boundaries,
    _PreparedBatchSource,
    _clip_model_weights,
    _create_scheduler,
    _create_state,
    _model_output_to_score,
    _run_validation,
    _scalar_head_loss,
    train_from_config,
)


def _make_native_batch(values: list[float]) -> NativeBatch:
    size = len(values)
    return NativeBatch(
        white_indices=np.zeros((size, 30), dtype=np.int32),
        black_indices=np.zeros((size, 30), dtype=np.int32),
        stm=np.ones((size,), dtype=np.float32),
        score=np.asarray(values, dtype=np.float32),
        result_wdl=np.full((size,), 0.5, dtype=np.float32),
    )


class _DummyStream:
    def __init__(self, batches: list[NativeBatch], *, fail_on_call: int | None = None) -> None:
        self._batches = list(batches)
        self._fail_on_call = fail_on_call
        self.requests: list[int] = []
        self.calls = 0

    def next_batch(self, batch_size: int) -> NativeBatch | None:
        self.calls += 1
        self.requests.append(batch_size)
        if self._fail_on_call is not None and self.calls == self._fail_on_call:
            raise RuntimeError("synthetic producer failure")
        if not self._batches:
            return None
        batch = self._batches.pop(0)
        if batch.stm.shape[0] > batch_size:
            raise AssertionError("requested batch size smaller than fixture batch")
        return batch


@unittest.skipUnless(torch is not None, "PyTorch is required for validation training tests")
class ValidationTrainingTests(unittest.TestCase):
    @staticmethod
    def _exponential_epoch_lr(base_lr: float, gamma: float, epoch_step: int) -> float:
        return base_lr * (gamma ** epoch_step)

    def test_scalar_head_loss_reports_score_and_wdl_components(self) -> None:
        prediction = torch.tensor([[150.0], [-50.0]], dtype=torch.float32)
        target = torch.tensor([[100.0], [-100.0]], dtype=torch.float32)
        result = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

        losses = _scalar_head_loss(
            prediction,
            target,
            result,
            lambda_weight=0.9,
            in_offset=270.0,
            out_offset=270.0,
            in_scaling=4000.0,
            out_scaling=4000.0,
            pow_exp=2.5,
            qp_asymmetry=0.0,
            w1=0.0,
            w2=0.5,
            torch=torch,
        )

        self.assertIn("wdl_loss", losses)
        self.assertIn("teacher_wdl_loss", losses)
        self.assertIn("result_wdl_loss", losses)
        self.assertGreater(float(losses["loss"].item()), 0.0)

    def test_model_output_to_score_applies_nnue2score(self) -> None:
        raw = torch.tensor([[1.25], [-0.5]], dtype=torch.float32)

        scaled = _model_output_to_score(raw, 600.0)

        self.assertTrue(torch.equal(scaled, torch.tensor([[750.0], [-300.0]], dtype=torch.float32)))

    def test_clip_model_weights_respects_dense_export_scale(self) -> None:
        class _Layer:
            def __init__(self, weights):
                self.weight = torch.tensor(weights, dtype=torch.float32)

        class _Model:
            def __init__(self) -> None:
                self.fc0 = _Layer([[5.0, -5.0]])
                self.fc1 = _Layer([[4.0, -4.0]])
                self.fc2 = _Layer([[3.0, -3.0]])

        model = _Model()
        config = TrainConfig.from_dict(
            {
                "datasets": ["/tmp/train.binpack"],
                "max_epochs": 10,
                "epoch_size": 1_000,
                "export_dense_scale": 64.0,
            }
        )

        _clip_model_weights(model, config)

        expected_limit = torch.tensor((127.0 - 0.5) / 64.0, dtype=torch.float32)
        self.assertLessEqual(float(model.fc0.weight.abs().max()), float(expected_limit))
        self.assertLessEqual(float(model.fc1.weight.abs().max()), float(expected_limit))
        self.assertLessEqual(float(model.fc2.weight.abs().max()), float(expected_limit))

    def test_create_scheduler_supports_stockfish_exponential_decay(self) -> None:
        base_lr = 0.000875
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))],
            lr=base_lr,
        )
        config = TrainConfig.from_dict(
            {
                "datasets": ["/tmp/train.binpack"],
                "max_epochs": 10,
                "epoch_size": 1_000,
                "lr": base_lr,
                "gamma": 0.992,
            }
        )

        scheduler = _create_scheduler(config, optimizer, torch)
        self.assertEqual(scheduler.__class__.__name__, "ExponentialLR")
        self.assertAlmostEqual(scheduler.gamma, 0.992)

    def test_epoch_scheduler_does_not_step_within_epoch(self) -> None:
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))],
            lr=0.000875,
        )
        config = TrainConfig.from_dict(
            {
                "datasets": ["/tmp/train.binpack"],
                "max_epochs": 10,
                "epoch_size": 1_000,
                "lr": 0.000875,
                "gamma": 0.992,
            }
        )
        scheduler = _create_scheduler(config, optimizer, torch)
        state = type("State", (), {"scheduler": scheduler, "config": config})()
        optimizer.step()

        _advance_scheduler_for_epoch_boundaries(
            state,
            positions_before_step=100,
            positions_after_step=999,
        )

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.000875)

    def test_epoch_scheduler_steps_once_on_single_boundary(self) -> None:
        base_lr = 0.000875
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))],
            lr=base_lr,
        )
        config = TrainConfig.from_dict(
            {
                "datasets": ["/tmp/train.binpack"],
                "max_epochs": 10,
                "epoch_size": 1_000,
                "lr": base_lr,
                "gamma": 0.992,
            }
        )
        scheduler = _create_scheduler(config, optimizer, torch)
        state = type("State", (), {"scheduler": scheduler, "config": config})()
        optimizer.step()

        _advance_scheduler_for_epoch_boundaries(
            state,
            positions_before_step=900,
            positions_after_step=1_100,
        )

        expected = self._exponential_epoch_lr(base_lr, 0.992, epoch_step=1)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected)

    def test_epoch_scheduler_steps_multiple_times_when_batch_crosses_multiple_epochs(self) -> None:
        base_lr = 0.000875
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))],
            lr=base_lr,
        )
        config = TrainConfig.from_dict(
            {
                "datasets": ["/tmp/train.binpack"],
                "max_epochs": 10,
                "epoch_size": 1_000,
                "lr": base_lr,
                "gamma": 0.992,
            }
        )
        scheduler = _create_scheduler(config, optimizer, torch)
        state = type("State", (), {"scheduler": scheduler, "config": config})()
        optimizer.step()

        _advance_scheduler_for_epoch_boundaries(
            state,
            positions_before_step=900,
            positions_after_step=3_100,
        )

        expected = self._exponential_epoch_lr(base_lr, 0.992, epoch_step=3)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected)

    def test_prefetched_batches_match_synchronous_order(self) -> None:
        expected_batches = [_make_native_batch([1.0, 2.0]), _make_native_batch([3.0]), _make_native_batch([4.0, 5.0])]

        sync_stream = _DummyStream([_make_native_batch([1.0, 2.0]), _make_native_batch([3.0]), _make_native_batch([4.0, 5.0])])
        with _PreparedBatchSource(
            sync_stream,
            batch_size=2,
            total_positions=None,
            queue_size=0,
            pin_memory=False,
            torch=torch,
        ) as sync_source:
            sync_scores = [batch.tensors["score"].squeeze(1).tolist() for batch in sync_source]

        prefetched_stream = _DummyStream(expected_batches)
        with _PreparedBatchSource(
            prefetched_stream,
            batch_size=2,
            total_positions=None,
            queue_size=2,
            pin_memory=False,
            torch=torch,
        ) as prefetched_source:
            prefetched_scores = [batch.tensors["score"].squeeze(1).tolist() for batch in prefetched_source]

        self.assertEqual(sync_scores, prefetched_scores)
        self.assertEqual(prefetched_stream.requests, [2, 2, 2, 2])

    def test_prefetch_source_propagates_producer_exceptions(self) -> None:
        stream = _DummyStream([_make_native_batch([1.0, 2.0])], fail_on_call=2)
        with _PreparedBatchSource(
            stream,
            batch_size=2,
            total_positions=None,
            queue_size=2,
            pin_memory=False,
            torch=torch,
        ) as source:
            first = next(source)
            self.assertEqual(first.batch_positions, 2)
            with self.assertRaisesRegex(RuntimeError, "synthetic producer failure"):
                next(source)

    def test_run_validation_reports_cp_metrics_and_material_sanity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            train_path = tmp / "train.binpack"
            valid_path = tmp / "valid.binpack"
            write_fixture_binpack(train_path)
            write_fixture_binpack(valid_path)

            config = TrainConfig.from_dict(
                {
                    "datasets": [str(train_path)],
                    "validation_datasets": [str(valid_path)],
                    "validation_size": 2,
                    "max_epochs": 10,
                    "epoch_size": 1_000,
                    "default_root_dir": str(tmp / "run"),
                    "accelerator": "cpu",
                    "filtered": False,
                    "wld_filtered": False,
                }
            )
            state = _create_state(config)
            model_before = {k: v.detach().clone() for k, v in state.model.state_dict().items()}

            metrics = _run_validation(state)

            self.assertEqual(metrics["event"], "validation")
            self.assertEqual(metrics["validation_batches"], 1)
            self.assertEqual(metrics["validation_positions"], 2)
            self.assertIn("validation_wdl_loss", metrics)
            self.assertIn("cp_mae", metrics)
            self.assertIn("cp_rmse", metrics)
            self.assertIn("cp_corr", metrics)
            self.assertIn("score_mae", metrics)
            self.assertIn("score_rmse", metrics)
            self.assertIn("score_corr", metrics)
            self.assertIn("material_sanity", metrics)
            self.assertIn("material_ordering_ok", metrics)
            for key, before in model_before.items():
                self.assertTrue(torch.equal(before, state.model.state_dict()[key]))

    def test_train_from_config_logs_cp_and_epoch_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            train_path = tmp / "train.binpack"
            valid_path = tmp / "valid.binpack"
            write_fixture_binpack(train_path)
            write_fixture_binpack(valid_path)

            config = TrainConfig.from_dict(
                {
                    "run_name": "tiny",
                    "datasets": [str(train_path)],
                    "validation_datasets": [str(valid_path)],
                    "max_epochs": 2,
                    "epoch_size": 2,
                    "validation_size": 2,
                    "batch_size": 2,
                    "network_save_period": 1,
                    "default_root_dir": str(tmp / "run"),
                    "accelerator": "cpu",
                    "amp": False,
                    "filtered": False,
                    "wld_filtered": False,
                }
            )

            checkpoint_path = train_from_config(config, console_mode="text")

            self.assertTrue(checkpoint_path.exists())
            metrics_path = Path(config.default_root_dir) / "metrics.jsonl"
            records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
            train_records = [record for record in records if record["event"] == "train"]
            validation_records = [record for record in records if record["event"] == "validation"]
            self.assertTrue(all("wdl_loss" in record for record in train_records))
            self.assertEqual([record["epoch_index"] for record in train_records], [1, 2])
            self.assertTrue(all("validation_wdl_loss" in record for record in validation_records))


if __name__ == "__main__":
    unittest.main()
