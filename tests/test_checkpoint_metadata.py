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
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.1)
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
                superbatch_index=12,
            )
            payload = load_checkpoint(checkpoint_path)
            self.assertEqual(payload["global_step"], 3017)
            self.assertEqual(payload["positions_seen"], 12_345_678)
            self.assertEqual(payload["superbatch_index"], 12)
            self.assertIn("rng_state", payload)


if __name__ == "__main__":
    unittest.main()
