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
    _normalize_teacher_scores,
    _run_validation,
    _scalar_head_loss,
    train_from_config,
)


def _make_native_batch(values: list[float]) -> NativeBatch:
    size = len(values)
    return NativeBatch(
        white_indices=np.zeros((size, 30), dtype=np.int32),
        black_indices=np.zeros((size, 30), dtype=np.int32),
        white_counts=np.full((size,), 15, dtype=np.int32),
        black_counts=np.full((size,), 15, dtype=np.int32),
        stm=np.ones((size,), dtype=np.float32),
        score_cp=np.asarray(values, dtype=np.float32),
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
    def test_score_normalization_only_clips(self) -> None:
        values = torch.tensor([[-5000.0], [2000.0], [9000.0]], dtype=torch.float32)
        config = TrainConfig.from_dict(
            {
                "train_datasets": ["/tmp/train.binpack"],
                "total_train_positions": 10_000,
                "epoch_positions": 1_000,
                "score_clip": 4000.0,
            }
        )
        normalized = _normalize_teacher_scores(values, config, torch)
        expected = torch.tensor([[-4000.0], [2000.0], [4000.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(normalized, expected))

    def test_scalar_head_loss_reports_cp_and_wdl_components(self) -> None:
        prediction = torch.tensor([[150.0], [-50.0]], dtype=torch.float32)
        target = torch.tensor([[100.0], [-100.0]], dtype=torch.float32)
        result = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

        losses = _scalar_head_loss(
            prediction,
            target,
            result,
            wdl_scale=410.0,
            cp_loss_beta=128.0,
            wdl_lambda=0.1,
            output_regularization=0.0,
            torch=torch,
        )

        self.assertIn("cp_loss", losses)
        self.assertIn("wdl_loss", losses)
        self.assertGreater(float(losses["loss"].item()), 0.0)

    def test_clip_model_weights_respects_dense_export_scale(self) -> None:
        class _Layer:
            def __init__(self, weights):
                self.weight = torch.tensor(weights, dtype=torch.float32)

        class _Model:
            def __init__(self) -> None:
                self.l1 = _Layer([[5.0, -5.0]])
                self.l2 = _Layer([[4.0, -4.0]])
                self.output = _Layer([[3.0, -3.0]])

        model = _Model()
        config = TrainConfig.from_dict(
            {
                "train_datasets": ["/tmp/train.binpack"],
                "total_train_positions": 10_000,
                "epoch_positions": 1_000,
                "export_dense_scale": 64.0,
            }
        )

        _clip_model_weights(model, config)

        expected_limit = torch.tensor((127.0 - 0.5) / 64.0, dtype=torch.float32)
        self.assertLessEqual(float(model.l1.weight.abs().max()), float(expected_limit))
        self.assertLessEqual(float(model.l2.weight.abs().max()), float(expected_limit))
        self.assertLessEqual(float(model.output.weight.abs().max()), float(expected_limit))

    def test_create_scheduler_supports_exponential_decay(self) -> None:
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))],
            lr=0.000875,
        )
        config = TrainConfig.from_dict(
            {
                "train_datasets": ["/tmp/train.binpack"],
                "total_train_positions": 10_000,
                "epoch_positions": 1_000,
                "lr_gamma": 0.992,
            }
        )

        scheduler = _create_scheduler(config, optimizer, torch)
        self.assertEqual(scheduler.__class__.__name__, "ExponentialLR")

    def test_epoch_scheduler_does_not_step_within_epoch(self) -> None:
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))],
            lr=0.000875,
        )
        config = TrainConfig.from_dict(
            {
                "train_datasets": ["/tmp/train.binpack"],
                "total_train_positions": 10_000,
                "epoch_positions": 1_000,
                "lr_gamma": 0.992,
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
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))],
            lr=0.000875,
        )
        config = TrainConfig.from_dict(
            {
                "train_datasets": ["/tmp/train.binpack"],
                "total_train_positions": 10_000,
                "epoch_positions": 1_000,
                "lr_gamma": 0.992,
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

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.000875 * 0.992)

    def test_epoch_scheduler_steps_multiple_times_when_batch_crosses_multiple_epochs(self) -> None:
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))],
            lr=0.000875,
        )
        config = TrainConfig.from_dict(
            {
                "train_datasets": ["/tmp/train.binpack"],
                "total_train_positions": 10_000,
                "epoch_positions": 1_000,
                "lr_gamma": 0.992,
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

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.000875 * (0.992**3))

    def test_prefetched_batches_match_synchronous_order(self) -> None:
        expected_batches = [_make_native_batch([1.0, 2.0]), _make_native_batch([3.0]), _make_native_batch([4.0, 5.0])]

        sync_stream = _DummyStream([_make_native_batch([1.0, 2.0]), _make_native_batch([3.0]), _make_native_batch([4.0, 5.0])])
        with _PreparedBatchSource(
            sync_stream,
            batch_size=2,
            total_positions=None,
            prefetch_batches=0,
            torch=torch,
        ) as sync_source:
            sync_scores = [batch.tensors["score_cp"].squeeze(1).tolist() for batch in sync_source]

        prefetched_stream = _DummyStream(expected_batches)
        with _PreparedBatchSource(
            prefetched_stream,
            batch_size=2,
            total_positions=None,
            prefetch_batches=2,
            torch=torch,
        ) as prefetched_source:
            prefetched_scores = [batch.tensors["score_cp"].squeeze(1).tolist() for batch in prefetched_source]

        self.assertEqual(sync_scores, prefetched_scores)
        self.assertEqual(prefetched_stream.requests, [2, 2, 2, 2])

    def test_prefetch_source_propagates_producer_exceptions(self) -> None:
        stream = _DummyStream([_make_native_batch([1.0, 2.0])], fail_on_call=2)
        with _PreparedBatchSource(
            stream,
            batch_size=2,
            total_positions=None,
            prefetch_batches=2,
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
                    "train_datasets": [str(train_path)],
                    "validation_datasets": [str(valid_path)],
                    "validation_positions": 2,
                    "total_train_positions": 10_000,
                    "epoch_positions": 1_000,
                    "output_dir": str(tmp / "run"),
                    "device": "cpu",
                }
            )
            state = _create_state(config)
            model_before = {k: v.detach().clone() for k, v in state.model.state_dict().items()}

            metrics = _run_validation(state)

            self.assertEqual(metrics["event"], "validation")
            self.assertEqual(metrics["validation_batches"], 1)
            self.assertEqual(metrics["validation_positions"], 2)
            self.assertIn("validation_cp_loss", metrics)
            self.assertIn("validation_wdl_loss", metrics)
            self.assertIn("cp_mae", metrics)
            self.assertIn("cp_rmse", metrics)
            self.assertIn("cp_corr", metrics)
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
                    "train_datasets": [str(train_path)],
                    "validation_datasets": [str(valid_path)],
                    "total_train_positions": 4,
                    "epoch_positions": 2,
                    "validation_interval_positions": 2,
                    "validation_positions": 2,
                    "batch_size": 2,
                    "checkpoint_every": 1,
                    "log_every": 1,
                    "output_dir": str(tmp / "run"),
                    "device": "cpu",
                    "amp": False,
                }
            )

            checkpoint_path = train_from_config(config, console_mode="text")

            self.assertTrue(checkpoint_path.exists())
            metrics_path = Path(config.output_dir) / "metrics.jsonl"
            records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
            train_records = [record for record in records if record["event"] == "train"]
            validation_records = [record for record in records if record["event"] == "validation"]
            self.assertTrue(all("cp_loss" in record for record in train_records))
            self.assertTrue(all("wdl_loss" in record for record in train_records))
            self.assertEqual([record["epoch_index"] for record in train_records], [1, 2])
            self.assertTrue(all("validation_cp_loss" in record for record in validation_records))
            self.assertTrue(all("validation_wdl_loss" in record for record in validation_records))


if __name__ == "__main__":
    unittest.main()
