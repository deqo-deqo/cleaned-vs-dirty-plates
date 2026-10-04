"""
Обучение baseline-модели для Kaggle Cleaned vs Dirty V2.

Transfer learning на MobileNetV3-Small с кросс-валидацией, early stopping
и Test Time Augmentation. Результат - файл submission.csv.
"""

from __future__ import annotations

import argparse
import copy
import io
import random
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, ImageStat
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

TRAIN_PREFIX = "plates/train/"
TEST_PREFIX = "plates/test/"
LABEL_TO_INT = {"dirty": 0, "cleaned": 1}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = len(LABEL_TO_INT)

ImageOp = Callable[[Image.Image], Image.Image]


@dataclass(frozen=True)
class Sample:
    """Изображение из архива: id, сырые байты и метка (None для теста)."""

    image_id: str
    image_bytes: bytes
    label: int | None = None


class SquarePad:
    """Преобразование изображения в квадратную форму с padding.

    Добавляет отступы (padding) до квадрата, заполняя их средним цветом изображения.
    Это сохраняет пропорции объектов и избегает искажений при resize.
    """

    def __call__(self, image: Image.Image) -> Image.Image:
        fill = tuple(int(v) for v in ImageStat.Stat(image).mean)
        side = max(image.size)
        canvas = Image.new("RGB", (side, side), fill)
        offset = ((side - image.width) // 2, (side - image.height) // 2)
        canvas.paste(image, offset)
        return canvas


class PlateDataset(Dataset):
    """Датасет поверх списка Sample с выбором подмножества по индексам."""

    def __init__(
        self,
        samples: list[Sample],
        indices: np.ndarray,
        transform: transforms.Compose,
        image_op: ImageOp | None = None,
    ) -> None:
        self.samples = samples
        self.indices = [int(index) for index in indices]
        self.transform = transform
        self.image_op = image_op

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[self.indices[index]]
        image = Image.open(io.BytesIO(sample.image_bytes)).convert("RGB")
        if self.image_op is not None:
            image = self.image_op(image)
        tensor = self.transform(image)
        label = -1 if sample.label is None else sample.label
        return tensor, label, sample.image_id


def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Гиперпараметры обучения, общие для обучения и анализа результатов."""
    parser.add_argument("--folds", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for train and eval"
    )
    parser.add_argument(
        "--image-size", type=int, default=160, help="Square image size for the model"
    )
    parser.add_argument(
        "--lr-head", type=float, default=3e-4, help="Learning rate for the classifier"
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=5e-5,
        help="Learning rate for the unfrozen backbone layers",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="AdamW weight decay"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience in epochs",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing for cross entropy",
    )
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Use ImageNet pretrained weights or start from scratch",
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="DataLoader workers; keep 0 on macOS"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "platesv2"

    parser = argparse.ArgumentParser(
        description="PyTorch baseline for Kaggle Cleaned vs Dirty V2."
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=data_dir / "plates.zip",
        help="Path to plates.zip",
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=data_dir / "sample_submission.csv",
        help="Path to sample_submission.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "submission.csv",
        help="Where to write Kaggle submission",
    )
    add_training_args(parser)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Фиксация seed для воспроизводимости результатов."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    """Выбор устройства: CUDA, затем MPS, иначе CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_samples(zip_path: Path) -> tuple[list[Sample], list[Sample]]:
    """Чтение train и test изображений из архива plates.zip."""
    train_samples: list[Sample] = []
    test_samples: list[Sample] = []

    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name.endswith(".jpg")
            and (name.startswith(TRAIN_PREFIX) or name.startswith(TEST_PREFIX))
        )

        for name in names:
            image_id = Path(name).stem
            image_bytes = archive.read(name)
            if name.startswith(TRAIN_PREFIX):
                label_name = Path(name).parent.name
                train_samples.append(
                    Sample(
                        image_id=image_id,
                        image_bytes=image_bytes,
                        label=LABEL_TO_INT[label_name],
                    )
                )
            else:
                test_samples.append(Sample(image_id=image_id, image_bytes=image_bytes))

    return train_samples, test_samples


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Создание трансформаций для обучения и валидации.

    Train transform включает агрессивную аугментацию для борьбы с переобучением
    на малом датасете (40 изображений):
    - RandomResizedCrop: случайный crop с масштабированием
    - Flips: горизонтальные и вертикальные отражения
    - Rotation: повороты до 22 градусов
    - ColorJitter: изменение яркости, контраста, насыщенности

    Eval transform использует детерминированную обработку:
    - Resize с небольшим увеличением (1.12x)
    - CenterCrop до целевого размера
    """
    train_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.72, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(
                degrees=22,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.ColorJitter(
                brightness=0.12,
                contrast=0.12,
                saturation=0.12,
                hue=0.04,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize(
                int(image_size * 1.12),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_model(device: torch.device, weights_mode: str) -> nn.Module:
    """Создание модели MobileNetV3-Small с transfer learning.

    Архитектура:
    - Backbone: MobileNetV3-Small (предобучен на ImageNet)
    - Замораживаем все слои кроме последнего блока features[-1]
    - Заменяем classifier на бинарную классификацию (2 класса)

    Это позволяет использовать предобученные признаки и дообучить
    только верхние слои на малом датасете (40 изображений).
    """
    # Загрузка предобученных весов
    if weights_mode == "none":
        weights = None
    else:
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        except Exception:
            weights = None

    try:
        model = models.mobilenet_v3_small(weights=weights)
    except Exception:
        print("warning=failed_to_load_imagenet_weights fallback=random_init")
        model = models.mobilenet_v3_small(weights=None)

    # Замена последнего слоя для бинарной классификации
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)

    # Заморозка всех параметров
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Размораживаем только последний блок backbone и classifier
    for parameter in model.features[-1].parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return model.to(device)


def build_optimizer(
    model: nn.Module, args: argparse.Namespace
) -> torch.optim.Optimizer:
    """AdamW с раздельными learning rate для backbone и classifier."""
    head_params = list(model.classifier.parameters())
    backbone_params = list(model.features[-1].parameters())
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Одна эпоха обучения; возвращает средний loss по датасету."""
    model.train()
    running_loss = 0.0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * images.size(0)

    return running_loss / max(len(loader.dataset), 1)


def predict_proba(
    model: nn.Module,
    samples: list[Sample],
    indices: np.ndarray,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    """Предсказание вероятностей с Test Time Augmentation (TTA).

    TTA применяет 3 варианта аугментации (оригинал, зеркало, переворот)
    и усредняет предсказания для повышения стабильности результатов.
    """
    tta_ops = [
        None,  # Оригинальное изображение
        ImageOps.mirror,  # Горизонтальное отражение
        ImageOps.flip,  # Вертикальное отражение
    ]
    probabilities = []

    model.eval()
    with torch.no_grad():
        for image_op in tta_ops:
            dataset = PlateDataset(
                samples=samples,
                indices=indices,
                transform=transform,
                image_op=image_op,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
            )

            fold_probs = []
            for images, _, _ in loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                fold_probs.append(probs)

            probabilities.append(np.vstack(fold_probs))

    # Усреднение предсказаний по всем вариантам TTA
    return np.mean(np.stack(probabilities, axis=0), axis=0)


def train_fold(
    train_samples: list[Sample],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    fold_number: int,
) -> tuple[nn.Module, float]:
    """Обучение модели на одном фолде с early stopping."""
    train_transform, eval_transform = build_transforms(args.image_size)
    train_dataset = PlateDataset(train_samples, train_idx, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    model = build_model(device=device, weights_mode=args.weights)
    optimizer = build_optimizer(model, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Сохраняем лучшее состояние модели для early stopping
    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        valid_probs = predict_proba(
            model=model,
            samples=train_samples,
            indices=valid_idx,
            transform=eval_transform,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        valid_targets = np.asarray(
            [train_samples[i].label for i in valid_idx], dtype=int
        )
        valid_pred = valid_probs.argmax(axis=1)
        valid_score = accuracy_score(valid_targets, valid_pred)

        print(
            f"fold={fold_number} epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} valid_acc={valid_score:.4f}"
        )

        if valid_score > best_score:
            best_score = valid_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        # Early stopping: останавливаемся если нет улучшения
        if epoch - best_epoch >= args.patience:
            break

    # Загружаем лучшие веса
    model.load_state_dict(best_state)
    return model, best_score


def run_cross_validation(
    train_samples: list[Sample],
    test_samples: list[Sample],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Кросс-валидация с сохранением out-of-fold предсказаний.

    Использует StratifiedKFold для сохранения баланса классов в каждом фолде.
    Возвращает out-of-fold предсказания для train и усредненные предсказания для test.
    """
    labels = np.asarray([sample.label for sample in train_samples], dtype=int)
    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    eval_transform = build_transforms(args.image_size)[1]

    oof_prob = np.zeros((len(train_samples), NUM_CLASSES), dtype=np.float32)
    test_prob = np.zeros((len(test_samples), NUM_CLASSES), dtype=np.float32)
    fold_scores = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        splitter.split(labels, labels), 1  # Используем labels напрямую
    ):
        model, fold_score = train_fold(
            train_samples=train_samples,
            train_idx=train_idx,
            valid_idx=valid_idx,
            args=args,
            device=device,
            fold_number=fold_number,
        )
        fold_scores.append(fold_score)

        # Out-of-fold предсказания для валидационной части
        valid_prob = predict_proba(
            model=model,
            samples=train_samples,
            indices=valid_idx,
            transform=eval_transform,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        oof_prob[valid_idx] = valid_prob

        # Предсказания для тестовой выборки
        test_indices = np.arange(len(test_samples))
        fold_test_prob = predict_proba(
            model=model,
            samples=test_samples,
            indices=test_indices,
            transform=eval_transform,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        test_prob += fold_test_prob / args.folds

        print(f"fold={fold_number} best_valid_acc={fold_score:.4f}")

    # Итоговые метрики по всем фолдам
    oof_pred = oof_prob.argmax(axis=1)
    overall_acc = accuracy_score(labels, oof_pred)
    fold_scores_array = np.asarray(fold_scores, dtype=np.float32)
    print(
        f"cv_acc={overall_acc:.4f} "
        f"fold_mean={fold_scores_array.mean():.4f} "
        f"fold_std={fold_scores_array.std():.4f}"
    )

    return oof_prob, test_prob


def write_submission(
    sample_submission_path: Path,
    output_path: Path,
    test_samples: list[Sample],
    probabilities: np.ndarray,
) -> None:
    """Запись предсказаний в формате Kaggle submission."""
    sample_submission = pd.read_csv(sample_submission_path, dtype={"id": str})
    mapping = {
        sample.image_id.zfill(4): INT_TO_LABEL[int(prob.argmax())]
        for sample, prob in zip(test_samples, probabilities, strict=True)
    }
    sample_submission["id"] = sample_submission["id"].astype(str).str.zfill(4)
    sample_submission["label"] = sample_submission["id"].map(mapping)

    if sample_submission["label"].isna().any():
        missing = sample_submission.loc[
            sample_submission["label"].isna(), "id"
        ].tolist()[:10]
        raise ValueError(f"Missing predictions for ids: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_submission.to_csv(output_path, index=False)
    print(f"saved_submission={output_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.zip_path.exists():
        raise FileNotFoundError(f"Missing archive: {args.zip_path}")
    if not args.sample_submission.exists():
        raise FileNotFoundError(f"Missing sample submission: {args.sample_submission}")

    device = resolve_device()
    train_samples, test_samples = load_samples(args.zip_path)
    print(
        f"device={device.type} train_images={len(train_samples)} "
        f"test_images={len(test_samples)}"
    )

    _, test_prob = run_cross_validation(
        train_samples=train_samples,
        test_samples=test_samples,
        args=args,
        device=device,
    )
    write_submission(
        sample_submission_path=args.sample_submission,
        output_path=args.output,
        test_samples=test_samples,
        probabilities=test_prob,
    )


if __name__ == "__main__":
    main()
