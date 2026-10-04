# Классификация изображений тарелок: Cleaned vs Dirty V2

Проект машинного обучения для бинарной классификации изображений тарелок (чистые vs грязные) с использованием PyTorch и transfer learning.

## Описание проекта

Проект решает задачу бинарной классификации изображений тарелок на два класса:
- **Cleaned** - чистые тарелки после мытья
- **Dirty** - грязные тарелки с остатками еды

**Особенности:**
- Малый датасет (40 тренировочных изображений)
- Transfer learning с MobileNetV3-Small
- Агрессивная аугментация данных
- Test Time Augmentation (TTA)
- Кросс-валидация с 3 фолдами
- Полный анализ данных и результатов

**Результаты:** Cross-Validation Accuracy = 65%

## Структура проекта

```
глубокие нейронные сети/
├── platesv2_baseline.py      # Основной скрипт обучения модели
├── eda.py                     # Разведочный анализ данных (EDA)
├── analyze_results.py         # Анализ результатов и ошибок
├── tests/                     # Тесты на синтетических данных (pytest)
├── pyproject.toml             # Зависимости и настройки инструментов (Poetry)
├── poetry.lock                # Зафиксированные версии зависимостей
├── poetry.toml                # Виртуальное окружение создаётся в .venv проекта
├── .pre-commit-config.yaml    # Хуки pre-commit (black, isort, flake8)
├── .flake8                    # Настройки линтера flake8
├── requirements.txt           # Зависимости для установки через pip
├── requirements-colab.txt     # Зависимости для Google Colab
├── README.md                  # Этот файл
├── ОТЧЕТ.md                   # Подробный отчет о проекте
├── submission.csv             # Результаты классификации
└── _inspect/                  # Визуализации и графики
    ├── class_distribution.png
    ├── image_sizes.png
    ├── color_stats.png
    ├── sample_images.png
    ├── confusion_matrix.png
    ├── confidence_distribution.png
    └── error_examples.png
```

## Быстрый старт

### Установка зависимостей

Зависимости управляются через [Poetry](https://python-poetry.org/). Виртуальное
окружение создаётся в папке `.venv` внутри проекта (она добавлена в `.gitignore`),
точные версии пакетов зафиксированы в `poetry.lock`.

```bash
poetry install
source .venv/bin/activate   # либо запускать команды через `poetry run ...`
```

Альтернатива без Poetry:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Данные соревнования (`plates.zip` и `sample_submission.csv`) по умолчанию ожидаются
в соседней папке `../platesv2/`. Другой путь можно передать через `--zip-path`.

### Запуск разведочного анализа данных (EDA)

```bash
python eda.py
```

Создает визуализации:
- Распределение классов
- Анализ размеров изображений
- Цветовые характеристики
- Примеры изображений из каждого класса

### Обучение модели

```bash
# Быстрый тест (2 эпохи, 2 фолда)
python platesv2_baseline.py --epochs 2 --folds 2

# Полное обучение (10 эпох, 3 фолда)
python platesv2_baseline.py --epochs 10 --folds 3 --batch-size 16
```

### Анализ результатов

```bash
python analyze_results.py
```

Создает визуализации:
- Confusion matrix
- Распределение уверенности предсказаний
- Примеры ошибочных предсказаний
- Детальный отчет по классификации

## Качество кода

В проекте настроены форматтеры и линтеры, которые запускаются автоматически
перед каждым коммитом через [pre-commit](https://pre-commit.com/):

- **black** - автоформатирование кода
- **isort** - сортировка импортов
- **flake8** - проверка стиля (PEP8), ошибок и сложности кода
- базовые хуки: пробелы в конце строк, перевод строки в конце файла,
  проверка YAML/TOML, защита от больших файлов

```bash
# Один раз после клонирования: подключить хуки к git
poetry run pre-commit install

# Запустить все проверки вручную
poetry run pre-commit run --all-files

# Запустить инструменты по отдельности
poetry run black .
poetry run isort .
poetry run flake8 .

# Тесты (используют синтетические изображения, датасет не нужен)
poetry run pytest
```

## Параметры командной строки

### platesv2_baseline.py

```bash
python platesv2_baseline.py [OPTIONS]

Опции:
  --zip-path PATH              Путь к plates.zip (default: ../platesv2/plates.zip)
  --sample-submission PATH     Путь к sample_submission.csv
  --output PATH                Путь для сохранения submission.csv
  --folds INT                  Количество фолдов для CV (default: 3)
  --epochs INT                 Количество эпох обучения (default: 6)
  --batch-size INT             Размер батча (default: 16)
  --image-size INT             Размер изображения (default: 160)
  --lr-head FLOAT              Learning rate для classifier (default: 3e-4)
  --lr-backbone FLOAT          Learning rate для backbone (default: 5e-5)
  --weight-decay FLOAT         AdamW weight decay (default: 1e-4)
  --patience INT               Early stopping patience (default: 3)
  --label-smoothing FLOAT      Label smoothing (default: 0.05)
  --weights {imagenet,none}    Использовать ImageNet веса (default: imagenet)
  --workers INT                DataLoader workers (default: 0)
  --seed INT                   Random seed (default: 42)
```

### eda.py и analyze_results.py

```bash
python eda.py [--zip-path PATH] [--output-dir PATH]
python analyze_results.py [--zip-path PATH] [--output-dir PATH] [параметры обучения]

Опции:
  --zip-path PATH              Путь к plates.zip (default: ../platesv2/plates.zip)
  --output-dir PATH            Папка для графиков (default: _inspect/)
```

`analyze_results.py` обучает модели той же кросс-валидацией, что и
`platesv2_baseline.py`, и принимает те же параметры обучения (`--epochs`,
`--folds`, `--seed` и т.д.), поэтому анализ относится к тем же моделям,
которые формируют `submission.csv`.

### Примеры использования

```bash
# Обучение без предобученных весов
python platesv2_baseline.py --weights none

# Увеличение количества эпох и фолдов
python platesv2_baseline.py --epochs 15 --folds 5

# Изменение размера изображения
python platesv2_baseline.py --image-size 224 --batch-size 8
```

## Архитектура модели

**Backbone:** MobileNetV3-Small (предобучен на ImageNet)

**Стратегия transfer learning:**
1. Загружаем предобученные веса ImageNet
2. Замораживаем все слои backbone
3. Размораживаем только последний блок `features[-1]`
4. Заменяем classifier на бинарную классификацию (2 класса)
5. Обучаем с раздельными learning rates:
   - Backbone: 5e-5 (медленное дообучение)
   - Classifier: 3e-4 (быстрое обучение)

**Обоснование выбора:**
- Компактная модель (~2.5M параметров) подходит для малых датасетов
- Предобученные веса позволяют использовать признаки из ImageNet
- Эффективная архитектура для быстрого инференса

## Предобработка и аугментация

### Аугментация для обучения

```python
- SquarePad()                          # Приведение к квадрату
- RandomResizedCrop(160, scale=(0.72, 1.0))  # Случайный crop
- RandomHorizontalFlip()               # Горизонтальное отражение
- RandomVerticalFlip()                 # Вертикальное отражение
- RandomRotation(degrees=22)           # Повороты
- ColorJitter(brightness=0.12, ...)    # Цветовые изменения
- Normalize(ImageNet mean/std)         # Нормализация
```

### Test Time Augmentation (TTA)

На этапе инференса применяется TTA с 3 вариантами:
1. Оригинальное изображение
2. Горизонтальное отражение
3. Вертикальное отражение

Финальное предсказание = усреднение по всем вариантам

## Результаты

### Метрики по фолдам

| Fold | Validation Accuracy |
|------|---------------------|
| 1    | 64.29%             |
| 2    | 69.23%             |
| 3    | 61.54%             |
| **Mean** | **65.02%**     |
| **Std**  | **3.18%**      |

### Детальный отчет

```
              precision    recall  f1-score   support

       dirty      0.800     0.400     0.533        20
     cleaned      0.600     0.900     0.720        20

    accuracy                          0.650        40
```

## Визуализации

Все визуализации сохраняются в директории `_inspect/`:

**EDA (eda.py):**
- `class_distribution.png` - распределение классов
- `image_sizes.png` - анализ размеров изображений
- `color_stats.png` - цветовые характеристики
- `sample_images.png` - примеры изображений

**Анализ результатов (analyze_results.py):**
- `confusion_matrix.png` - матрица ошибок
- `confidence_distribution.png` - распределение уверенности
- `error_examples.png` - примеры ошибочных предсказаний

## Google Colab

### Рекомендуемая структура в Google Drive

```text
MyDrive/
  kaggle-plates/
    глубокие нейронные сети/
      platesv2_baseline.py
      eda.py
      analyze_results.py
      requirements-colab.txt
      README.md
      ОТЧЕТ.md
    platesv2/
      plates.zip
      sample_submission.csv
```

### Запуск в Colab

```python
# Подключение Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Установка зависимостей (без torch/torchvision - уже есть в Colab)
!python -m pip install -q -r requirements-colab.txt

# Запуск EDA
!python eda.py

# Обучение модели
!python platesv2_baseline.py --epochs 10 --folds 3 --batch-size 32

# Анализ результатов
!python analyze_results.py
```

**Примечание:** Colab уже имеет GPU-версии torch и torchvision, поэтому `requirements-colab.txt` не включает их.

## Зависимости

Основной источник - `pyproject.toml` и `poetry.lock`. Инструменты разработки
(black, isort, flake8, pre-commit, pytest) вынесены в группу `dev`.

### requirements.txt (локальная установка через pip)

```
numpy==2.4.4
pandas==3.0.2
Pillow==12.2.0
scikit-learn==1.8.0
torch==2.11.0
torchvision==0.26.0
matplotlib==3.10.8
seaborn==0.13.2
```

### requirements-colab.txt (Google Colab)

```
numpy
pandas
Pillow
scikit-learn
matplotlib
seaborn
```

## Рекомендации по улучшению

### Высокий приоритет
1. **Увеличение датасета** - собрать минимум 200-500 изображений на класс
2. **Использование внешних датасетов** - предобучение на смежных задачах

### Средний приоритет
3. **Более мощные архитектуры** - EfficientNet-B0/B1, ResNet-50
4. **Ансамблирование** - комбинация нескольких моделей
5. **Подбор гиперпараметров** - grid search или Bayesian optimization

### Низкий приоритет
6. **Визуализация признаков** - Grad-CAM для интерпретации
7. **Анализ сложных примеров** - идентификация пограничных случаев

Подробные рекомендации см. в разделе 9 файла [ОТЧЕТ.md](ОТЧЕТ.md)

## Документация

- **README.md** - инструкции по запуску и использованию
- **ОТЧЕТ.md** - подробный отчет о проекте с анализом и выводами

---

**Дата создания:** 23 апреля 2025  
**Версия:** 1.0
