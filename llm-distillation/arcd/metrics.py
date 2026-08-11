"""
arcd/metrics.py
================

Agrégation, affichage et sauvegarde des métriques.

Le MetricTracker peut maintenant :

    - accumuler les métriques batch par batch
    - afficher les moyennes d'une époque
    - sauvegarder automatiquement les résultats dans un CSV

Compatible avec :
    - Baseline
    - Hinton KD
    - ARCD
"""

from collections import defaultdict
from pathlib import Path
import csv


class MetricTracker:

    def __init__(self):
        self._sums = defaultdict(float)
        self._count = 0

    # ---------------------------------------------------------
    # Accumulation
    # ---------------------------------------------------------

    def update(self, metrics: dict, batch_size: int = 1):

        for key, value in metrics.items():
            self._sums[key] += float(value) * batch_size

        self._count += batch_size

    # ---------------------------------------------------------
    # Moyennes
    # ---------------------------------------------------------

    def average(self):

        if self._count == 0:
            return {}

        return {
            key: value / self._count
            for key, value in self._sums.items()
        }

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def reset(self):

        self._sums = defaultdict(float)
        self._count = 0

    # ---------------------------------------------------------
    # Affichage
    # ---------------------------------------------------------

    def log(self, epoch: int, accuracy: float | None = None):

        avg = self.average()

        order = [
            "loss",
            "L_KD",
            "L_CE",
            "C",
            "T",
            "S",
            "lambda"
        ]

        parts = [f"Epoch {epoch}"]

        for key in order:

            if key in avg:
                parts.append(f"{key}={avg[key]:.4f}")

        if accuracy is not None:
            parts.append(f"accuracy={accuracy:.4f}")

        print(" | ".join(parts))

    # ---------------------------------------------------------
    # Sauvegarde CSV
    # ---------------------------------------------------------

    def save_csv(
        self,
        filepath: str,
        epoch: int,
        accuracy: float | None = None,
    ):

        avg = self.average()

        row = {"epoch": epoch}

        row.update(avg)

        if accuracy is not None:
            row["accuracy"] = accuracy

        filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)

        file_exists = filepath.exists()

        with open(filepath, "a", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=row.keys()
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)