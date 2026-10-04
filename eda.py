"""
Разведочный анализ данных (EDA) для датасета Cleaned vs Dirty V2.

Скрипт строит графики по тренировочной выборке:
- Распределение классов
- Размеры изображений
- Цветовые характеристики
- Примеры изображений из каждого класса
"""

import argparse
import io
import zipfile
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

# Настройка стиля графиков
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# Константы
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ZIP_PATH = BASE_DIR.parent / "platesv2" / "plates.zip"
DEFAULT_OUTPUT_DIR = BASE_DIR / "_inspect"
TRAIN_PREFIX = "plates/train/"
CHANNELS = ("R", "G", "B")
SAMPLES_PER_CLASS = 10
SAMPLE_GRID_COLS = 5

ImageSize = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA for Kaggle Cleaned vs Dirty V2 train images."
    )
    parser.add_argument(
        "--zip-path", type=Path, default=DEFAULT_ZIP_PATH, help="Path to plates.zip"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved plots",
    )
    return parser.parse_args()


def load_train_data(
    zip_path: Path,
) -> tuple[list[Image.Image], list[str], list[ImageSize]]:
    """Загрузка тренировочных данных из архива."""
    images = []
    labels = []
    sizes = []

    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name.startswith(TRAIN_PREFIX) and name.endswith(".jpg")
        )

        for name in names:
            label_name = Path(name).parent.name
            image_bytes = archive.read(name)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            images.append(image)
            labels.append(label_name)
            sizes.append(image.size)

    return images, labels, sizes


def save_figure(save_path: Path) -> None:
    """Сохранение текущего графика в файл и закрытие фигуры."""
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Сохранено: {save_path}")
    plt.close()


def format_stats(values: Sequence[float], precision: int) -> str:
    """Строка с min/max/mean/std для набора значений."""
    low, high = min(values), max(values)
    if precision:
        low_high = f"min={low:.{precision}f}, max={high:.{precision}f}"
    else:
        low_high = f"min={low}, max={high}"
    mean_precision = max(precision, 1)
    return (
        f"{low_high}, mean={np.mean(values):.{mean_precision}f}, "
        f"std={np.std(values):.{mean_precision}f}"
    )


def analyze_class_distribution(labels: Sequence[str], output_dir: Path) -> None:
    """Анализ распределения классов."""
    unique, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    bars = ax.bar(
        unique, counts, color=["#e74c3c", "#2ecc71"], alpha=0.7, edgecolor="black"
    )
    ax.set_xlabel("Класс", fontsize=12)
    ax.set_ylabel("Количество изображений", fontsize=12)
    ax.set_title(
        "Распределение классов в тренировочной выборке", fontsize=14, fontweight="bold"
    )

    # Добавляем значения на столбцы
    for bar, count in zip(bars, counts, strict=True):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    save_figure(output_dir / "class_distribution.png")


def plot_histogram_with_mean(
    ax: plt.Axes,
    values: Sequence[float],
    color: str,
    xlabel: str,
    title: str,
    mean_format: str,
) -> None:
    """Гистограмма с вертикальной линией среднего значения."""
    mean_value = float(np.mean(values))
    ax.hist(values, bins=15, color=color, edgecolor="black", alpha=0.7)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Количество", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axvline(
        mean_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Среднее: {mean_value:{mean_format}}",
    )
    ax.legend()


def analyze_image_sizes(
    sizes: Sequence[ImageSize], labels: Sequence[str], output_dir: Path
) -> None:
    """Анализ размеров изображений."""
    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    aspects = [w / h for w, h in sizes]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_histogram_with_mean(
        axes[0, 0],
        widths,
        color="skyblue",
        xlabel="Ширина (px)",
        title="Распределение ширины изображений",
        mean_format=".0f",
    )
    plot_histogram_with_mean(
        axes[0, 1],
        heights,
        color="lightcoral",
        xlabel="Высота (px)",
        title="Распределение высоты изображений",
        mean_format=".0f",
    )
    plot_histogram_with_mean(
        axes[1, 0],
        aspects,
        color="lightgreen",
        xlabel="Соотношение сторон (ширина/высота)",
        title="Распределение соотношения сторон",
        mean_format=".2f",
    )

    # Scatter: ширина vs высота
    colors = ["red" if label == "dirty" else "green" for label in labels]
    axes[1, 1].scatter(widths, heights, c=colors, alpha=0.6, edgecolors="black", s=100)
    axes[1, 1].set_xlabel("Ширина (px)", fontsize=11)
    axes[1, 1].set_ylabel("Высота (px)", fontsize=11)
    axes[1, 1].set_title("Ширина vs Высота", fontsize=12, fontweight="bold")
    axes[1, 1].legend(["dirty", "cleaned"], loc="best")

    save_figure(output_dir / "image_sizes.png")

    # Статистика
    print("\nСтатистика размеров изображений:")
    print(f"  Ширина:  {format_stats(widths, precision=0)}")
    print(f"  Высота:  {format_stats(heights, precision=0)}")
    print(f"  Aspect:  {format_stats(aspects, precision=2)}")


def analyze_color_stats(
    images: Sequence[Image.Image], labels: Sequence[str], output_dir: Path
) -> None:
    """Анализ цветовых характеристик изображений."""
    stats_by_class: dict[str, dict[str, list[float]]] = {
        "cleaned": {channel: [] for channel in CHANNELS},
        "dirty": {channel: [] for channel in CHANNELS},
    }

    for img, label in zip(images, labels, strict=True):
        arr = np.array(img)
        for channel_index, channel in enumerate(CHANNELS):
            stats_by_class[label][channel].append(arr[:, :, channel_index].mean())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, channel in enumerate(CHANNELS):
        cleaned_vals = stats_by_class["cleaned"][channel]
        dirty_vals = stats_by_class["dirty"][channel]

        axes[idx].hist(
            cleaned_vals,
            bins=10,
            alpha=0.6,
            label="Cleaned",
            color="lightgreen",
            edgecolor="black",
        )
        axes[idx].hist(
            dirty_vals,
            bins=10,
            alpha=0.6,
            label="Dirty",
            color="lightcoral",
            edgecolor="black",
        )
        axes[idx].set_xlabel(f"Среднее значение канала {channel}", fontsize=11)
        axes[idx].set_ylabel("Количество", fontsize=11)
        axes[idx].set_title(
            f"Распределение канала {channel}", fontsize=12, fontweight="bold"
        )
        axes[idx].legend()

    save_figure(output_dir / "color_stats.png")


def visualize_samples(
    images: Sequence[Image.Image], labels: Sequence[str], output_dir: Path
) -> None:
    """Визуализация примеров изображений из каждого класса."""
    rows_per_class = SAMPLES_PER_CLASS // SAMPLE_GRID_COLS

    fig, axes = plt.subplots(2 * rows_per_class, SAMPLE_GRID_COLS, figsize=(15, 12))
    fig.suptitle("Примеры изображений из датасета", fontsize=16, fontweight="bold")

    # Первые ряды - cleaned, последние - dirty
    for class_index, class_name in enumerate(("cleaned", "dirty")):
        class_images = [
            img
            for img, label in zip(images, labels, strict=True)
            if label == class_name
        ]
        for i in range(SAMPLES_PER_CLASS):
            row = class_index * rows_per_class + i // SAMPLE_GRID_COLS
            col = i % SAMPLE_GRID_COLS
            axes[row, col].imshow(class_images[i])
            axes[row, col].axis("off")
            axes[row, col].set_title(f"{class_name.capitalize()} #{i}", fontsize=10)

    save_figure(output_dir / "sample_images.png")


def main() -> None:
    """Основная функция для запуска EDA."""
    args = parse_args()

    print("=" * 60)
    print("Разведочный анализ данных (EDA)")
    print("=" * 60)

    if not args.zip_path.exists():
        raise FileNotFoundError(f"Missing archive: {args.zip_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nЗагрузка данных...")
    images, labels, sizes = load_train_data(args.zip_path)
    print(f"Загружено {len(images)} изображений")

    print("\nАнализ распределения классов...")
    analyze_class_distribution(labels, args.output_dir)

    print("\nАнализ размеров изображений...")
    analyze_image_sizes(sizes, labels, args.output_dir)

    print("\nАнализ цветовых характеристик...")
    analyze_color_stats(images, labels, args.output_dir)

    print("\nВизуализация примеров изображений...")
    visualize_samples(images, labels, args.output_dir)

    print("\n" + "=" * 60)
    print(f"EDA завершен! Все графики сохранены в директории {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
