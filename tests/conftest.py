import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TRAIN_PER_CLASS = 10
TEST_IMAGES = 4


def make_jpeg(rng: np.random.Generator, width: int, height: int) -> bytes:
    pixels = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def plates_zip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Синтетический архив с той же структурой, что и plates.zip с Kaggle."""
    rng = np.random.default_rng(0)
    zip_path = tmp_path_factory.mktemp("data") / "plates.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for label in ("cleaned", "dirty"):
            for index in range(TRAIN_PER_CLASS):
                archive.writestr(
                    f"plates/train/{label}/{index:04d}.jpg", make_jpeg(rng, 96, 64)
                )
        for index in range(TEST_IMAGES):
            archive.writestr(f"plates/test/{index:04d}.jpg", make_jpeg(rng, 64, 96))
        archive.writestr("plates/train/.DS_Store", b"")
    return zip_path
