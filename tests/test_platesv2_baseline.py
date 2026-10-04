from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from platesv2_baseline import (
    LABEL_TO_INT,
    NUM_CLASSES,
    PlateDataset,
    SquarePad,
    build_model,
    build_transforms,
    load_samples,
    predict_proba,
    run_cross_validation,
    write_submission,
)
from tests.conftest import TEST_IMAGES, TRAIN_PER_CLASS, make_training_args

IMAGE_SIZE = 64


def test_square_pad_makes_square_and_keeps_content_centered() -> None:
    image = Image.new("RGB", (40, 20), (200, 10, 10))

    padded = SquarePad()(image)

    assert padded.size == (40, 40)
    assert padded.getpixel((20, 20)) == (200, 10, 10)


def test_load_samples_splits_train_and_test(plates_zip: Path) -> None:
    train_samples, test_samples = load_samples(plates_zip)

    assert len(train_samples) == 2 * TRAIN_PER_CLASS
    assert len(test_samples) == TEST_IMAGES
    assert {sample.label for sample in train_samples} == set(LABEL_TO_INT.values())
    assert all(sample.label is None for sample in test_samples)


def test_dataset_returns_normalized_tensor_label_and_id(plates_zip: Path) -> None:
    train_samples, test_samples = load_samples(plates_zip)
    train_transform, eval_transform = build_transforms(IMAGE_SIZE)

    tensor, label, image_id = PlateDataset(
        train_samples, np.array([0]), train_transform
    )[0]
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert label == train_samples[0].label
    assert image_id == train_samples[0].image_id

    _, test_label, _ = PlateDataset(test_samples, np.array([0]), eval_transform)[0]
    assert test_label == -1


def test_predict_proba_returns_probabilities(plates_zip: Path) -> None:
    _, test_samples = load_samples(plates_zip)
    device = torch.device("cpu")
    model = build_model(device=device, weights_mode="none")

    probabilities = predict_proba(
        model=model,
        samples=test_samples,
        indices=np.arange(len(test_samples)),
        transform=build_transforms(IMAGE_SIZE)[1],
        device=device,
        batch_size=2,
        workers=0,
    )

    assert probabilities.shape == (TEST_IMAGES, NUM_CLASSES)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)


def test_build_model_trains_only_head_and_last_block() -> None:
    model = build_model(device=torch.device("cpu"), weights_mode="none")

    trainable = {
        name for name, param in model.named_parameters() if param.requires_grad
    }

    assert trainable
    assert all(
        name.startswith(("classifier.", f"features.{len(model.features) - 1}."))
        for name in trainable
    )
    assert model.classifier[3].out_features == NUM_CLASSES


def test_write_submission_maps_predictions_to_labels(
    plates_zip: Path, tmp_path: Path
) -> None:
    _, test_samples = load_samples(plates_zip)
    sample_submission = tmp_path / "sample_submission.csv"
    ids = [sample.image_id for sample in test_samples]
    pd.DataFrame({"id": ids, "label": "dirty"}).to_csv(sample_submission, index=False)
    probabilities = np.tile([0.1, 0.9], (len(test_samples), 1))
    probabilities[0] = [0.8, 0.2]
    output = tmp_path / "out" / "submission.csv"

    write_submission(sample_submission, output, test_samples, probabilities)

    result = pd.read_csv(output, dtype={"id": str})
    assert result["id"].tolist() == ids
    assert result["label"].tolist() == ["dirty"] + ["cleaned"] * (TEST_IMAGES - 1)


def test_write_submission_fails_on_missing_predictions(
    plates_zip: Path, tmp_path: Path
) -> None:
    _, test_samples = load_samples(plates_zip)
    sample_submission = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": ["9999"], "label": "dirty"}).to_csv(
        sample_submission, index=False
    )
    probabilities = np.tile([0.1, 0.9], (len(test_samples), 1))

    with pytest.raises(ValueError, match="Missing predictions"):
        write_submission(
            sample_submission, tmp_path / "submission.csv", test_samples, probabilities
        )


def test_cross_validation_returns_oof_and_test_probabilities(plates_zip: Path) -> None:
    train_samples, test_samples = load_samples(plates_zip)

    oof_prob, test_prob = run_cross_validation(
        train_samples, test_samples, make_training_args(), torch.device("cpu")
    )

    assert oof_prob.shape == (len(train_samples), NUM_CLASSES)
    assert test_prob.shape == (len(test_samples), NUM_CLASSES)
    np.testing.assert_allclose(oof_prob.sum(axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(test_prob.sum(axis=1), 1.0, rtol=1e-5)
