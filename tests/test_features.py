from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.board import BoardState, square_to_index
from thrawn_nnue.features import (
    FEATURES,
    HALFKA_V2_HM_MAX_ACTIVE_FEATURES,
    HALFKA_V2_HM_NUM_FEATURES,
    halfka_v2_hm_feature_index,
    halfka_v2_hm_piece_square_offset,
    king_square,
    active_feature_indices,
)


class FeatureTests(unittest.TestCase):
    def test_halfka_v2_hm_features_include_kings_and_fit_stockfish_width(self) -> None:
        board = BoardState.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")
        white = active_feature_indices(board, "white", features=FEATURES)
        black = active_feature_indices(board, "black", features=FEATURES)

        self.assertEqual(len(white), HALFKA_V2_HM_MAX_ACTIVE_FEATURES)
        self.assertEqual(len(black), HALFKA_V2_HM_MAX_ACTIVE_FEATURES)
        self.assertTrue(all(0 <= value < HALFKA_V2_HM_NUM_FEATURES for value in white))
        self.assertTrue(all(0 <= value < HALFKA_V2_HM_NUM_FEATURES for value in black))
        self.assertEqual(HALFKA_V2_HM_NUM_FEATURES, 22_528)

    def test_king_piece_plane_is_shared_between_colors(self) -> None:
        self.assertEqual(halfka_v2_hm_piece_square_offset("K", "white"), 10 * 64)
        self.assertEqual(halfka_v2_hm_piece_square_offset("k", "white"), 10 * 64)
        self.assertEqual(halfka_v2_hm_piece_square_offset("K", "black"), 10 * 64)
        self.assertEqual(halfka_v2_hm_piece_square_offset("k", "black"), 10 * 64)

    def test_feature_index_matches_stockfish_mirrored_king_bucket_layout(self) -> None:
        a1 = square_to_index("a1")
        h1 = square_to_index("h1")
        e1 = square_to_index("e1")
        e2 = square_to_index("e2")
        e8 = square_to_index("e8")
        e7 = square_to_index("e7")

        self.assertEqual(
            halfka_v2_hm_feature_index(a1, e2, "P", "white"),
            (e2 ^ 7) + 28 * 704,
        )
        self.assertEqual(
            halfka_v2_hm_feature_index(h1, e2, "P", "white"),
            e2 + 28 * 704,
        )
        self.assertEqual(
            halfka_v2_hm_feature_index(e1, e2, "P", "white"),
            e2 + 31 * 704,
        )
        self.assertEqual(
            halfka_v2_hm_feature_index(e8, e7, "p", "black"),
            (e7 ^ 56) + 31 * 704,
        )

    def test_king_square_uses_perspective_king(self) -> None:
        board = BoardState.from_fen("8/8/8/8/8/8/4p3/4K2k b - - 0 1")
        self.assertEqual(king_square(board, "white"), square_to_index("e1"))
        self.assertEqual(king_square(board, "black"), square_to_index("h1"))


if __name__ == "__main__":
    unittest.main()
