from arcd.metrics import MetricTracker


def test_weighted_average_over_batches():
    tracker = MetricTracker()
    tracker.update({"loss": 2.0, "C": 0.9, "T": 0.8, "S": 0.1, "lambda": 0.7}, batch_size=4)
    tracker.update({"loss": 1.0, "C": 0.5, "T": 0.4, "S": 0.9, "lambda": 0.1}, batch_size=4)
    avg = tracker.average()
    assert abs(avg["loss"] - 1.5) < 1e-6


def test_log_does_not_raise(capsys):
    tracker = MetricTracker()
    tracker.update({"loss": 2.0, "C": 0.9, "T": 0.8, "S": 0.1, "lambda": 0.7})
    tracker.log(epoch=1, accuracy=0.734)
    captured = capsys.readouterr()
    assert "Epoch 1" in captured.out


def test_reset_empties_tracker():
    tracker = MetricTracker()
    tracker.update({"loss": 1.0}, batch_size=2)
    tracker.reset()
    assert tracker.average() == {}
