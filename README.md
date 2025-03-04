# Image Processing and Augmentation Pipeline (M2)


Проект для предобработки и аугментации изображений с последующим формированием сбалансированного датасета.


## Особенности

- 📦 Распаковка исходного архива с изображениями

- 🖼️ Предобработка изображений (ресайз до 224x224, конвертация в RGB)

- 🔄 Аугментация данных:

  - Случайный поворот (±15°)

  - Добавление случайных черных прямоугольников

  - Визуализация результатов аугментаций

  - Защита от повторной обработки

- ⚖️ Балансировка классов в датасете

- 📊 Автоматическая генерация отчета

- 📁 Создание итогового ZIP-архива с датасетом


## Требования

- Python 3.8+

- Библиотеки: `Pillow`, `numpy`, `matplotlib`


Установите зависимости:

```bash

pip install -r requirements.txt

Структура проекта

.

├── data/                   # Исходные данные

│   └── Gauge.zip           # Исходный архив с изображениями

├── src/                    # Исходный код

│   ├── main.py             # Основной скрипт обработки

│   └── ...                 # Дополнительные модули (при наличии)

├── temp_extract/           # Временная директория распаковки

├── images/                 # Распакованные изображения

│   ├── train/              # Обучающая выборка

│   ├── test/               # Тестовая выборка

│   └── val/                # Валидационная выборка

└── preprocessed_images/    # Обработанные данные

    └── ...                 # Структура аналогична images/
```

## Использование
Основной сценарий

    Поместите архив Gauge.zip в корневую директорию проекта
    Запустите основной скрипт:

bash

python main.py

Этапы обработки:

    Распаковка архива → temp_extract/
    Структурирование данных → images/
    Предобработка (ресайз + конвертация) → preprocessed_images/
    Аугментация (по 2 новых изображения на класс)
    Балансировка до целевых размеров:
        Train: 23,000
        Test: 1,000
        Val: 2,000
    Создание архива → preprocessed_images.zip
    Генерация отчета → dataset_report.txt

## Демонстрация аугментаций

Для визуализации примеров аугментаций выполните:

python

# В коде:

demo_augmentations()

Пример вывода: Augmentation Demo
Выходные данные

    preprocessed_images.zip - финальный датасет
    dataset_report.txt - отчет со статистикой

## Пример отчета:

Структура итогового датасета:

========================================

TRAIN: 23000 изображений (10 классов)

TEST: 1000 изображений (10 классов)

VAL: 2000 изображений (10 классов)

## Особенности реализации

    🔒 Защита от повторной аугментации
    🚀 Оптимизированная обработка больших объемов данных
    📝 Подробное логирование процесса
    🧹 Автоматическая очистка временных файлов
# Image Processing and YOLO Model Training Pipeline (M3)


Проект для обработки изображений и обучения детекционных моделей YOLO.


## 🌟 Новые возможности

- 🧠 Поддержка трех архитектур YOLOv8: `yolov8n`, `yolov8s`, `yolov8m`

- ⚙️ Гибкая настройка гиперпараметров обучения

- 📊 Расчет метрик качества: Accuracy, ROC-AUC, F1-Score

- ⏱️ Замер времени обучения (общее и на эпоху)

- 📁 Автоматическая генерация YOLO-аннотаций


## 🛠️ Требования

```bash

pip install ultralytics numpy pillow matplotlib
```

🗂️ Структура проекта 

.

├── data/

│   └── preprocessed_images.zip  # Обработанные изображения

├── runs/

│   └── detect/                   # Результаты обучения YOLO

├── src/

│   ├── train.py                 # Класс YOLOTrainer

│   └── ...                      # Другие скрипты

└── dataset/                     # Автогенерируемая структура YOLO

    ├── images/

    │   ├── train/               # Обучающие изображения

    │   └── val/                 # Валидационные изображения

    └── labels/                  # Сгенерированные аннотации

## 🚀 Обучение модели
Основной сценарий

python

from train import YOLOTrainer


trainer = YOLOTrainer()

results = trainer.train(

    model_type='yolov8n',

    epochs=50,

    batch=32,

    imgsz=640,

    lr0=0.01,

    weight_decay=0.0005

)


## Метод train() возвращает список с:

    Обученная модель (YOLO object)
    Словарь метрик:

    python

{

    'accuracy': 0.92,

    'roc_auc': 0.88,

    'f1_score': 0.90

    }

    Время на эпоху (секунды)
    Общее время обучения (минуты)

## 📈 Пример вывода

bash

# Результаты обучения:

1. Обученная модель: YOLO

2. Метрики:

   - accuracy: 0.9234

   - roc_auc: 0.8812

   - f1_score: 0.9015

3. Время на эпоху: 12.34 сек

4. Общее время обучения: 10.25 мин

## 🧩 Особенности реализации

    Автогенерация аннотаций: Преобразует структуру папок в YOLO-формат
    Динамическая конфигурация: Автоматическое создание data.yaml
    Валидация данных: Проверка целостности датасета перед обучением
    Оптимизация памяти: Очистка временных файлов после выполнения

## 📌 Важные замечания

    Архив preprocessed_images.zip должен находиться в корне проекта
    Для обучения доступны только предопределенные модели YOLO
    Все гиперпараметры можно переопределять через **kwargs
    Результаты обучения сохраняются в runs/detect/train_*/

Автор: Nikskss

Контакты: [v.denisenko2307@gmail.com]
