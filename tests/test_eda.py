from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from eda import (  # noqa: E402
    analyze_class_distribution,
    analyze_image_sizes,
    format_stats,
    load_train_data,
)
from tests.conftest import TRAIN_PER_CLASS  # noqa: E402


def test_load_train_data_reads_only_train_images(plates_zip: Path) -> None:
    images, labels, sizes = load_train_data(plates_zip)

    assert len(images) == len(labels) == len(sizes) == 2 * TRAIN_PER_CLASS
    assert set(labels) == {"cleaned", "dirty"}
    assert set(sizes) == {(96, 64)}


def test_format_stats() -> None:
    assert format_stats([1, 3], precision=0) == "min=1, max=3, mean=2.0, std=1.0"
    assert (
        format_stats([0.5, 1.5], precision=2)
        == "min=0.50, max=1.50, mean=1.00, std=0.50"
    )


def test_plots_are_saved_to_output_dir(plates_zip: Path, tmp_path: Path) -> None:
    _, labels, sizes = load_train_data(plates_zip)

    analyze_class_distribution(labels, tmp_path)
    analyze_image_sizes(sizes, labels, tmp_path)

    assert (tmp_path / "class_distribution.png").exists()
    assert (tmp_path / "image_sizes.png").exists()
