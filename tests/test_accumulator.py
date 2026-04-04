from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.accumulator import apply_updates, feature_deltas, refresh_accumulator
from thrawn_nnue.board import BoardState
from thrawn_nnue.features import NUM_FEATURES, active_feature_indices


TEST_CASES = [
    ("quiet", "8/8/8/8/8/8/4P3/4K2k w - - 0 1", "e2e4"),
    ("capture", "8/8/8/3p4/4P3/8/8/4K2k w - - 0 1", "e4d5"),
    ("promotion", "6k1/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7e8q"),
    ("castling", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
    ("en_passant", "7k/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
    ("king_move", "7k/8/8/8/8/8/8/4K3 w - - 0 1", "e1e2"),
]


class AccumulatorTests(unittest.TestCase):
    def test_incremental_update_matches_refresh(self) -> None:
        bias = [0.05 * i for i in range(8)]
        weights = [
            [((feature_index % 11) - 5) * 0.03125 + 0.01 * dim for dim in range(8)]
            for feature_index in range(NUM_FEATURES)
        ]

        for name, fen, move in TEST_CASES:
            with self.subTest(case=name):
                before = BoardState.from_fen(fen)
                after = before.apply_uci(move)

                for perspective in ("white", "black"):
                    prev = refresh_accumulator(
                        active_feature_indices(before, perspective),
                        weights,
                        bias,
                    )
                    removed, added = feature_deltas(before, after, perspective)
                    updated = apply_updates(prev, removed, added, weights)
                    refreshed = refresh_accumulator(
                        active_feature_indices(after, perspective),
                        weights,
                        bias,
                    )
                    self.assertEqual(len(updated), len(refreshed))
                    for lhs, rhs in zip(updated, refreshed):
                        self.assertAlmostEqual(lhs, rhs, places=7)


if __name__ == "__main__":
    unittest.main()
