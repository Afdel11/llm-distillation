"""
arcd/metrics.py
================
Agrégation et affichage des métriques pendant l'entraînement.
Séparé du reste pour que le format d'affichage puisse évoluer
(CSV, TensorBoard, W&B...) sans toucher à losses.py.
"""

from collections import defaultdict


class MetricTracker:
    """
    Accumule les dicts de métriques renvoyés par ARCDLoss (ou tout autre
    criterion suivant la même convention) sur une epoch, puis affiche
    la moyenne.

    Usage:
        tracker = MetricTracker()
        for batch in loader:
            loss, metrics = criterion(...)
            tracker.update(metrics)
        tracker.log(epoch=4)
        tracker.reset()
    """

    def __init__(self):
        self._sums = defaultdict(float)
        self._count = 0

    def update(self, metrics: dict, batch_size: int = 1):
        for key, value in metrics.items():
            self._sums[key] += value * batch_size
        self._count += batch_size

    def average(self) -> dict:
        if self._count == 0:
            return {}
        return {key: total / self._count for key, total in self._sums.items()}

    def reset(self):
        self._sums = defaultdict(float)
        self._count = 0

    def log(self, epoch: int, accuracy: float = None):
        avg = self.average()
        parts = [f"Epoch {epoch}"]
        order = ["loss", "L_KD", "L_CE", "C", "T", "S", "lambda"]
        for key in order:
            if key in avg:
                parts.append(f"{key}={avg[key]:.4f}")
        if accuracy is not None:
            parts.append(f"accuracy={accuracy:.4f}")
        print(" | ".join(parts))
