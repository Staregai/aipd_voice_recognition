from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def export_confusion_matrix_csv(path: str | Path, labels: list[str], matrix: np.ndarray) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expected/predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def export_text(path: str | Path, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")
