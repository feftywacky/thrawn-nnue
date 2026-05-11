from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ModuleNotFoundError:
    torch = None

from thrawn_nnue.checkpoint import load_checkpoint, save_checkpoint

if torch is not None:
    from thrawn_nnue.model import HalfKPNNUE


class CheckpointLoadBehaviorTests(unittest.TestCase):
    def test_load_checkpoint_disables_weights_only_mode(self) -> None:
        fake_torch = Mock()
        fake_torch.load.return_value = {"ok": True}

        with patch("thrawn_nnue.checkpoint._require_torch", return_value=fake_torch):
            payload = load_checkpoint("/tmp/checkpoint.pt", map_location="cpu")

        self.assertEqual(payload, {"ok": True})
        fake_torch.load.assert_called_once_with(
            Path("/tmp/checkpoint.pt"),
            map_location="cpu",
            weights_only=False,
        )


@unittest.skipUnless(torch is not None, "PyTorch is required for checkpoint metadata tests")
class CheckpointMetadataTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_resume_metadata(self) -> None:
        model = HalfKPNNUE()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.992)
        scaler = torch.cuda.amp.GradScaler(enabled=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config={"run_name": "test"},
                global_step=3017,
                positions_seen=12_345_678,
                epoch_index=12,
                best_validation_loss=0.25,
                best_validation_positions=9_999,
                best_checkpoint_metric_name="validation_score_mae",
                best_checkpoint_metric_value=123.0,
                best_checkpoint_positions=11_111,
                best_checkpoint_sort_key=(0.0, 123.0, 0.01, -0.9, 0.25),
            )
            payload = load_checkpoint(checkpoint_path)
            self.assertEqual(payload["global_step"], 3017)
            self.assertEqual(payload["positions_seen"], 12_345_678)
            self.assertEqual(payload["epoch_index"], 12)
            self.assertIn("rng_state", payload)
            self.assertEqual(payload["best_validation_loss"], 0.25)
            self.assertEqual(payload["best_validation_positions"], 9_999)
            self.assertEqual(payload["best_checkpoint_metric_name"], "validation_score_mae")
            self.assertEqual(payload["best_checkpoint_metric_value"], 123.0)
            self.assertEqual(payload["best_checkpoint_positions"], 11_111)
            self.assertEqual(payload["best_checkpoint_sort_key"], [0.0, 123.0, 0.01, -0.9, 0.25])


if __name__ == "__main__":
    unittest.main()
