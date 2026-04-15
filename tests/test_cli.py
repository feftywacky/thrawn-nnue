from __future__ import annotations

import sys
from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()
