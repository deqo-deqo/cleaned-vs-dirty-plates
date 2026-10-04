import io
import zipfile
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
TRAIN_PREFIX = "plates/train/"
LABEL_TO_INT = {"dirty": 0, "cleaned": 1}


def load_train_data(zip_path: Path):
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


def analyze_class_distribution(labels):
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
    for bar, count in zip(bars, counts):
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

    plt.tight_layout()
    plt.savefig("_inspect/class_distribution.png", dpi=150, bbox_inches="tight")
    print("Сохранено: _inspect/class_distribution.png")
    plt.close()


def analyze_image_sizes(sizes, labels):
    """Анализ размеров изображений."""
    widths = [s[0] for s in sizes]
    heights = [s[1] for s in sizes]
    aspects = [w / h for w, h in sizes]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Распределение ширины
    axes[0, 0].hist(widths, bins=15, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Ширина (px)", fontsize=11)
    axes[0, 0].set_ylabel("Количество", fontsize=11)
    axes[0, 0].set_title(
        "Распределение ширины изображений", fontsize=12, fontweight="bold"
    )
    axes[0, 0].axvline(
        np.mean(widths),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Среднее: {np.mean(widths):.0f}",
    )
    axes[0, 0].legend()

    # Распределение высоты
    axes[0, 1].hist(heights, bins=15, color="lightcoral", edgecolor="black", alpha=0.7)
    axes[0, 1].set_xlabel("Высота (px)", fontsize=11)
    axes[0, 1].set_ylabel("Количество", fontsize=11)
    axes[0, 1].set_title(
        "Распределение высоты изображений", fontsize=12, fontweight="bold"
    )
    axes[0, 1].axvline(
        np.mean(heights),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Среднее: {np.mean(heights):.0f}",
    )
    axes[0, 1].legend()

    # Соотношение сторон
    axes[1, 0].hist(aspects, bins=15, color="lightgreen", edgecolor="black", alpha=0.7)
    axes[1, 0].set_xlabel("Соотношение сторон (ширина/высота)", fontsize=11)
    axes[1, 0].set_ylabel("Количество", fontsize=11)
    axes[1, 0].set_title(
        "Распределение соотношения сторон", fontsize=12, fontweight="bold"
    )
    axes[1, 0].axvline(
        np.mean(aspects),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Среднее: {np.mean(aspects):.2f}",
    )
    axes[1, 0].legend()

    # Scatter: ширина vs высота
    colors = ["red" if label == "dirty" else "green" for label in labels]
    axes[1, 1].scatter(widths, heights, c=colors, alpha=0.6, edgecolors="black", s=100)
    axes[1, 1].set_xlabel("Ширина (px)", fontsize=11)
    axes[1, 1].set_ylabel("Высота (px)", fontsize=11)
    axes[1, 1].set_title("Ширина vs Высота", fontsize=12, fontweight="bold")
    axes[1, 1].legend(["dirty", "cleaned"], loc="best")

    plt.tight_layout()
    plt.savefig("_inspect/image_sizes.png", dpi=150, bbox_inches="tight")
    print("Сохранено: _inspect/image_sizes.png")
    plt.close()

    # Статистика
    print(f"\nСтатистика размеров изображений:")
    print(
        f"  Ширина:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}, std={np.std(widths):.1f}"
    )
    print(
        f"  Высота:  min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}, std={np.std(heights):.1f}"
    )
    print(
        f"  Aspect:  min={min(aspects):.2f}, max={max(aspects):.2f}, mean={np.mean(aspects):.2f}, std={np.std(aspects):.2f}"
    )


def analyze_color_stats(images, labels):
    """Анализ цветовых характеристик изображений."""
    stats_by_class = {
        "cleaned": {"R": [], "G": [], "B": []},
        "dirty": {"R": [], "G": [], "B": []},
    }

    for img, label in zip(images, labels):
        arr = np.array(img)
        stats_by_class[label]["R"].append(arr[:, :, 0].mean())
        stats_by_class[label]["G"].append(arr[:, :, 1].mean())
        stats_by_class[label]["B"].append(arr[:, :, 2].mean())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    channels = ["R", "G", "B"]
    colors_plot = ["red", "green", "blue"]

    for idx, (channel, color) in enumerate(zip(channels, colors_plot)):
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

    plt.tight_layout()
    plt.savefig("_inspect/color_stats.png", dpi=150, bbox_inches="tight")
    print("Сохранено: _inspect/color_stats.png")
    plt.close()


def visualize_samples(images, labels):
    """Визуализация примеров изображений из каждого класса."""
    cleaned_imgs = [img for img, label in zip(images, labels) if label == "cleaned"]
    dirty_imgs = [img for img, label in zip(images, labels) if label == "dirty"]

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle("Примеры изображений из датасета", fontsize=16, fontweight="bold")

    # Первые 2 ряда - cleaned
    for i in range(10):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(cleaned_imgs[i])
        axes[row, col].axis("off")
        axes[row, col].set_title(f"Cleaned #{i}", fontsize=10)

    # Последние 2 ряда - dirty
    for i in range(10):
        row = 2 + i // 5
        col = i % 5
        axes[row, col].imshow(dirty_imgs[i])
        axes[row, col].axis("off")
        axes[row, col].set_title(f"Dirty #{i}", fontsize=10)

    plt.tight_layout()
    plt.savefig("_inspect/sample_images.png", dpi=150, bbox_inches="tight")
    print("Сохранено: _inspect/sample_images.png")
    plt.close()


def main():
    """Основная функция для запуска EDA."""
    print("=" * 60)
    print("Разведочный анализ данных (EDA)")
    print("=" * 60)

    zip_path = Path("../platesv2/plates.zip")
    if not zip_path.exists():
        print(f"Ошибка: файл {zip_path} не найден!")
        return

    print("\nЗагрузка данных...")
    images, labels, sizes = load_train_data(zip_path)
    print(f"Загружено {len(images)} изображений")

    print("\nАнализ распределения классов...")
    analyze_class_distribution(labels)

    print("\nАнализ размеров изображений...")
    analyze_image_sizes(sizes, labels)

    print("\nАнализ цветовых характеристик...")
    analyze_color_stats(images, labels)

    print("\nВизуализация примеров изображений...")
    visualize_samples(images, labels)

    print("\n" + "=" * 60)
    print("EDA завершен! Все графики сохранены в директории _inspect/")
    print("=" * 60)


if __name__ == "__main__":
    main()
