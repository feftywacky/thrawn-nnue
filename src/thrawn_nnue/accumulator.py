from __future__ import annotations

from typing import Iterable

from .board import BoardState
from .features import active_feature_indices, king_square


def refresh_accumulator(
    active_indices: Iterable[int],
    weights: list[list[float]],
    bias: list[float],
) -> list[float]:
    acc = list(bias)
    for feature_index in active_indices:
        row = weights[feature_index]
        for i, value in enumerate(row):
            acc[i] += value
    return acc


def apply_updates(
    previous: list[float],
    removed_indices: Iterable[int],
    added_indices: Iterable[int],
    weights: list[list[float]],
) -> list[float]:
    acc = list(previous)
    for feature_index in removed_indices:
        row = weights[feature_index]
        for i, value in enumerate(row):
            acc[i] -= value
    for feature_index in added_indices:
        row = weights[feature_index]
        for i, value in enumerate(row):
            acc[i] += value
    return acc


def requires_refresh(before: BoardState, after: BoardState, perspective: str) -> bool:
    return king_square(before, perspective) != king_square(after, perspective)


def feature_deltas(before: BoardState, after: BoardState, perspective: str) -> tuple[list[int], list[int]]:
    before_features = set(active_feature_indices(before, perspective))
    after_features = set(active_feature_indices(after, perspective))
    removed = sorted(before_features - after_features)
    added = sorted(after_features - before_features)
    return removed, added
