from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import analyze_results  # noqa: E402
import platesv2_baseline  # noqa: E402
from tests.conftest import TRAIN_PER_CLASS, make_training_args  # noqa: E402


def test_predictions_come_from_trained_models(
    plates_zip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = make_training_args()
    epochs_run = []
    original_train_one_epoch = platesv2_baseline.train_one_epoch

    def counting_train_one_epoch(**kwargs: object) -> float:
        epochs_run.append(1)
        return original_train_one_epoch(**kwargs)

    monkeypatch.setattr(platesv2_baseline, "train_one_epoch", counting_train_one_epoch)

    samples, oof_probs, labels = analyze_results.train_and_get_predictions(
        plates_zip, args
    )

    assert len(epochs_run) == args.folds * args.epochs
    assert len(samples) == len(labels) == 2 * TRAIN_PER_CLASS
    assert oof_probs.shape == (2 * TRAIN_PER_CLASS, platesv2_baseline.NUM_CLASSES)
    np.testing.assert_allclose(oof_probs.sum(axis=1), 1.0, rtol=1e-5)
