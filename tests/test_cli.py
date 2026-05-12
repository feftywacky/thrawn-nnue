from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.cli import main


class CliTests(unittest.TestCase):
    def test_calibrate_scale_command_is_removed(self) -> None:
        argv = sys.argv
        try:
            sys.argv = ["thrawn-nnue", "calibrate-scale"]
            with self.assertRaises(SystemExit) as ctx:
                main()
        finally:
            sys.argv = argv
        self.assertEqual(ctx.exception.code, 2)

    def test_test_command_dispatches_to_unittest_runner(self) -> None:
        argv = sys.argv
        try:
            sys.argv = ["thrawn-nnue", "test", "--pattern", "test_cli.py", "--verbosity", "1", "--failfast"]
            with patch("thrawn_nnue.cli._run_test_suite", return_value=0) as runner:
                with self.assertRaises(SystemExit) as ctx:
                    main()
        finally:
            sys.argv = argv

        self.assertEqual(ctx.exception.code, 0)
        runner.assert_called_once_with(pattern="test_cli.py", verbosity=1, failfast=True)

    def test_fine_tune_command_uses_checkpoint_as_initial_weights(self) -> None:
        argv = sys.argv
        config = object()
        try:
            sys.argv = [
                "thrawn-nnue",
                "fine-tune",
                "--config",
                "configs/v5.toml",
                "--checkpoint",
                "configs/runs/v4/checkpoints/best.pt",
                "--console-mode",
                "text",
            ]
            with (
                patch("thrawn_nnue.config.load_config", return_value=config) as load_config,
                patch("thrawn_nnue.training.train_from_config", return_value=Path("out.pt")) as train_from_config,
                patch("builtins.print") as print_mock,
            ):
                main()
        finally:
            sys.argv = argv

        load_config.assert_called_once_with("configs/v5.toml")
        train_from_config.assert_called_once_with(
            config,
            console_mode="text",
            init_checkpoint="configs/runs/v4/checkpoints/best.pt",
        )
        print_mock.assert_called_once_with("out.pt")


if __name__ == "__main__":
    unittest.main()
