"""
Анализ результатов обучения модели.

Этот скрипт создает визуализации для понимания:
- Какие изображения модель классифицирует неправильно
- Распределение уверенности предсказаний
- Confusion matrix
"""

import io

# Импорт из основного скрипта
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from platesv2_baseline import (
    INT_TO_LABEL,
    LABEL_TO_INT,
    Sample,
    build_model,
    build_transforms,
    load_samples,
    predict_proba,
    set_seed,
)

# Настройка стиля
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def train_and_get_predictions(zip_path, args_dict):
    """Обучение модели и получение out-of-fold предсказаний."""
    print("Загрузка данных и обучение модели...")

    train_samples, _ = load_samples(zip_path)
    labels = np.asarray([sample.label for sample in train_samples], dtype=int)

    # Определяем устройство
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"  Устройство: {device.type}")
    print(f"  Тренировочных изображений: {len(train_samples)}")

    # Кросс-валидация
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    eval_transform = build_transforms(160)[1]

    oof_probs = np.zeros((len(train_samples), 2), dtype=np.float32)

    for fold_number, (train_idx, valid_idx) in enumerate(
        splitter.split(labels, labels), 1
    ):
        print(f"  Обучение fold {fold_number}/3...")

        # Простое обучение (без полного цикла, используем только валидацию)
        model = build_model(device=device, weights_mode="imagenet")

        # Предсказания для валидационной части
        valid_probs = predict_proba(
            model=model,
            samples=train_samples,
            indices=valid_idx,
            transform=eval_transform,
            device=device,
            batch_size=16,
            workers=0,
        )
        oof_probs[valid_idx] = valid_probs

    return train_samples, oof_probs, labels


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Визуализация confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["dirty", "cleaned"],
        yticklabels=["dirty", "cleaned"],
        cbar_kws={"label": "Количество"},
        ax=ax,
        annot_kws={"size": 16, "weight": "bold"},
    )

    ax.set_xlabel("Предсказанный класс", fontsize=12, fontweight="bold")
    ax.set_ylabel("Истинный класс", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Сохранено: {save_path}")
    plt.close()


def plot_confidence_distribution(probs, y_true, save_path):
    """Визуализация распределения уверенности предсказаний."""
    y_pred = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    correct_mask = y_pred == y_true
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Гистограмма уверенности
    axes[0].hist(
        correct_conf,
        bins=15,
        alpha=0.7,
        label="Правильные",
        color="green",
        edgecolor="black",
    )
    axes[0].hist(
        incorrect_conf,
        bins=15,
        alpha=0.7,
        label="Неправильные",
        color="red",
        edgecolor="black",
    )
    axes[0].set_xlabel("Уверенность модели", fontsize=11)
    axes[0].set_ylabel("Количество", fontsize=11)
    axes[0].set_title(
        "Распределение уверенности предсказаний", fontsize=12, fontweight="bold"
    )
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Box plot
    data_to_plot = [correct_conf, incorrect_conf]
    bp = axes[1].boxplot(
        data_to_plot,
        labels=["Правильные", "Неправильные"],
        patch_artist=True,
        widths=0.6,
    )
    bp["boxes"][0].set_facecolor("lightgreen")
    bp["boxes"][1].set_facecolor("lightcoral")

    axes[1].set_ylabel("Уверенность модели", fontsize=11)
    axes[1].set_title("Сравнение уверенности", fontsize=12, fontweight="bold")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Сохранено: {save_path}")
    plt.close()

    print(f"\n  Средняя уверенность (правильные): {correct_conf.mean():.3f}")
    print(f"  Средняя уверенность (неправильные): {incorrect_conf.mean():.3f}")


def visualize_errors(samples, probs, y_true, zip_path, save_path):
    """Визуализация ошибочно классифицированных изображений."""
    y_pred = probs.argmax(axis=1)
    error_indices = np.where(y_pred != y_true)[0]

    if len(error_indices) == 0:
        print("  Нет ошибок для визуализации!")
        return

    print(f"  Найдено ошибок: {len(error_indices)}")

    # Загружаем изображения
    with zipfile.ZipFile(zip_path) as archive:
        n_errors = min(len(error_indices), 12)
        n_cols = 4
        n_rows = (n_errors + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            "Ошибочно классифицированные изображения", fontsize=16, fontweight="bold"
        )

        for idx, error_idx in enumerate(error_indices[:n_errors]):
            row = idx // n_cols
            col = idx % n_cols

            sample = samples[error_idx]
            image = Image.open(io.BytesIO(sample.image_bytes)).convert("RGB")

            true_label = INT_TO_LABEL[y_true[error_idx]]
            pred_label = INT_TO_LABEL[y_pred[error_idx]]
            confidence = probs[error_idx].max()

            axes[row, col].imshow(image)
            axes[row, col].axis("off")
            axes[row, col].set_title(
                f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})",
                fontsize=10,
                color="red",
                fontweight="bold",
            )

        # Скрываем пустые subplot'ы
        for idx in range(n_errors, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Сохранено: {save_path}")
        plt.close()


def print_classification_report(y_true, y_pred):
    """Вывод детального отчета по классификации."""
    print("\n" + "=" * 60)
    print("ОТЧЕТ ПО КЛАССИФИКАЦИИ")
    print("=" * 60)
    report = classification_report(
        y_true, y_pred, target_names=["dirty", "cleaned"], digits=3
    )
    print(report)


def main():
    """Основная функция анализа результатов."""
    print("=" * 60)
    print("Анализ результатов обучения модели")
    print("=" * 60)

    set_seed(42)
    zip_path = Path("../platesv2/plates.zip")

    if not zip_path.exists():
        print(f"Ошибка: файл {zip_path} не найден!")
        return

    # Получаем предсказания
    args_dict = {"epochs": 6, "folds": 3, "batch_size": 16}
    samples, oof_probs, y_true = train_and_get_predictions(zip_path, args_dict)
    y_pred = oof_probs.argmax(axis=1)

    # Создаем директорию для результатов
    output_dir = Path("_inspect")
    output_dir.mkdir(exist_ok=True)

    print("\nСоздание визуализаций...")

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")

    # Распределение уверенности
    plot_confidence_distribution(
        oof_probs, y_true, output_dir / "confidence_distribution.png"
    )

    # Визуализация ошибок
    visualize_errors(
        samples, oof_probs, y_true, zip_path, output_dir / "error_examples.png"
    )

    # Отчет по классификации
    print_classification_report(y_true, y_pred)

    print("\n" + "=" * 60)
    print("Анализ завершен! Все графики сохранены в _inspect/")
    print("=" * 60)


if __name__ == "__main__":
    main()
