from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.board import BoardState, flip_vertical, square_to_index
from thrawn_nnue.features import (
    MAX_ACTIVE_FEATURES,
    extract_dual_perspective,
    feature_index,
    orient_square,
    padded_feature_indices,
)


class FeatureTests(unittest.TestCase):
    def test_black_perspective_flips_vertically_and_swaps_colors(self) -> None:
        e7 = square_to_index("e7")
        self.assertEqual(orient_square(e7, "black"), flip_vertical(e7))
        self.assertEqual(feature_index(square_to_index("e2"), "P", "white"), 12)
        self.assertEqual(feature_index(square_to_index("e7"), "p", "black"), 12)

    def test_padded_dual_perspective_has_fixed_width(self) -> None:
        board = BoardState.from_fen("8/8/8/8/8/8/4p3/4K2k w - - 0 1")
        features = extract_dual_perspective(board)
        self.assertEqual(len(features.white), MAX_ACTIVE_FEATURES)
        self.assertEqual(len(features.black), MAX_ACTIVE_FEATURES)
        self.assertEqual(features.stm, 1.0)
        self.assertIn(-1, features.white)
        self.assertIn(-1, features.black)

    def test_padding_preserves_real_feature_prefix(self) -> None:
        board = BoardState.from_fen("8/8/8/8/8/8/4p3/4K2k b - - 0 1")
        indices = padded_feature_indices(board, "white")
        real = [i for i in indices if i >= 0]
        self.assertEqual(indices[: len(real)], real)
        self.assertTrue(all(i == -1 for i in indices[len(real) :]))


if __name__ == "__main__":
    unittest.main()
