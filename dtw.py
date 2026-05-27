from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DTWResult:
    total_cost: float
    normalized_cost: float
    path: list[tuple[int, int]]
    local_cost_matrix: np.ndarray
    global_cost_matrix: np.ndarray


def pairwise_distance_matrix(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    a = np.asarray(seq_a, dtype=np.float64)
    b = np.asarray(seq_b, dtype=np.float64)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("DTW expects 2D feature sequences shaped [frames, features]")

    if metric == "manhattan":
        return np.sum(np.abs(a[:, np.newaxis, :] - b[np.newaxis, :, :]), axis=2)

    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def dynamic_time_warping(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    *,
    metric: str = "euclidean",
    window_ratio: float | None = 0.2,
) -> DTWResult:
    local_cost = pairwise_distance_matrix(seq_a, seq_b, metric=metric)
    n_rows, n_cols = local_cost.shape

    band = None
    if window_ratio is not None:
        band = max(abs(n_rows - n_cols), int(np.ceil(max(n_rows, n_cols) * window_ratio)))

    global_cost = np.full((n_rows + 1, n_cols + 1), np.inf, dtype=np.float64)
    global_cost[0, 0] = 0.0

    for row in range(1, n_rows + 1):
        if band is None:
            col_start = 1
            col_stop = n_cols + 1
        else:
            col_start = max(1, row - band)
            col_stop = min(n_cols + 1, row + band + 1)

        for col in range(col_start, col_stop):
            prev_cost = min(
                global_cost[row - 1, col],
                global_cost[row, col - 1],
                global_cost[row - 1, col - 1],
            )
            global_cost[row, col] = local_cost[row - 1, col - 1] + prev_cost

    path: list[tuple[int, int]] = []
    row = n_rows
    col = n_cols
    while row > 0 and col > 0:
        path.append((row - 1, col - 1))
        candidates = [
            (global_cost[row - 1, col - 1], row - 1, col - 1),
            (global_cost[row - 1, col], row - 1, col),
            (global_cost[row, col - 1], row, col - 1),
        ]
        _, row, col = min(candidates, key=lambda item: item[0])

    while row > 0:
        row -= 1
        path.append((row, 0))
    while col > 0:
        col -= 1
        path.append((0, col))

    path.reverse()
    total_cost = float(global_cost[n_rows, n_cols])
    normalized_cost = total_cost / max(1, len(path))
    return DTWResult(
        total_cost=total_cost,
        normalized_cost=normalized_cost,
        path=path,
        local_cost_matrix=local_cost,
        global_cost_matrix=global_cost[1:, 1:],
    )
