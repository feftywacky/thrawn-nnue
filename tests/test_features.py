from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.board import BoardState, flip_vertical, square_to_index
from thrawn_nnue.features import (
    HALFKA_V2_HM_MAX_ACTIVE_FEATURES,
    HALFKA_V2_HM_NUM_FEATURES,
    MAX_ACTIVE_FEATURES,
    NUM_FEATURES,
    active_feature_indices,
    factor_feature_index,
    feature_index,
    king_square,
    orient_square,
)


class FeatureTests(unittest.TestCase):
    def test_black_perspective_flips_vertically_and_swaps_colors(self) -> None:
        e7 = square_to_index("e7")
        self.assertEqual(orient_square(e7, "black"), flip_vertical(e7))
        self.assertEqual(factor_feature_index(square_to_index("e2"), "P", "white"), 12)
        self.assertEqual(factor_feature_index(square_to_index("e7"), "p", "black"), 12)
        self.assertEqual(
            feature_index(square_to_index("e1"), square_to_index("e2"), "P", "white"),
            feature_index(square_to_index("e8"), square_to_index("e7"), "p", "black"),
        )

    def test_active_features_exclude_kings_and_fit_halfkp_width(self) -> None:
        board = BoardState.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")
        white = active_feature_indices(board, "white")
        black = active_feature_indices(board, "black")
        self.assertEqual(len(white), MAX_ACTIVE_FEATURES)
        self.assertEqual(len(black), MAX_ACTIVE_FEATURES)
        self.assertTrue(all(0 <= value < NUM_FEATURES for value in white))
        self.assertTrue(all(0 <= value < NUM_FEATURES for value in black))

    def test_halfka_v2_hm_features_include_kings_and_fit_stockfish_width(self) -> None:
        board = BoardState.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")
        white = active_feature_indices(board, "white", features="HalfKAv2_hm^")
        black = active_feature_indices(board, "black", features="HalfKAv2_hm^")

        self.assertEqual(len(white), HALFKA_V2_HM_MAX_ACTIVE_FEATURES)
        self.assertEqual(len(black), HALFKA_V2_HM_MAX_ACTIVE_FEATURES)
        self.assertTrue(all(0 <= value < HALFKA_V2_HM_NUM_FEATURES for value in white))
        self.assertTrue(all(0 <= value < HALFKA_V2_HM_NUM_FEATURES for value in black))

    def test_king_square_uses_perspective_king(self) -> None:
        board = BoardState.from_fen("8/8/8/8/8/8/4p3/4K2k b - - 0 1")
        self.assertEqual(king_square(board, "white"), square_to_index("e1"))
        self.assertEqual(king_square(board, "black"), square_to_index("h1"))


if __name__ == "__main__":
    unittest.main()
