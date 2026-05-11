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


if __name__ == "__main__":
    unittest.main()
