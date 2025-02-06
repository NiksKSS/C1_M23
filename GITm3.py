import os
import yaml
import time
import zipfile
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import shutil


    def __init__(self):
        # Словарь для сопоставления типов моделей с их весами
        self.model_map = {
            'yolov8n': 'yolov8n.pt',
            'yolov8s': 'yolov8s.pt',
            'yolov8m': 'yolov8m.pt'
        }
        self.metrics = {}  # Словарь для хранения метрик после обучения
        self.training_time = 0  # Время обучения


    def _prepare_dataset(self, zip_path: str):
        extract_dir = Path("dataset")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref: #Распаковка zip-архива с изображениями
                zip_ref.extractall(extract_dir)# Директория для распаковки данных
        for split in ['train', 'val']:
                    split_path = extract_dir / split
                    if split_path.exists():
                        self._generate_annotations(split_path) #Генерация аннотаций для обучающей и валидационной выборок
        return self._create_data_yaml(extract_dir)


    def _generate_annotations(self, split_path: Path):
        for class_dir in split_path.iterdir():
            if class_dir.is_dir() and '-' in class_dir.name:
                try:
                    # Извлечение ID класса из имени директории
                    class_id = int(class_dir.name.split('-')[0])
                    label_path = split_path.parent / 'labels' / class_dir.name
                    label_path.mkdir(parents=True, exist_ok=True)  # Создание директории для аннотаций

                    # Генерация аннотаций для каждого изображения
                    for img in class_dir.glob('*.*'):
                        if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            with open(label_path / f"{img.stem}.txt", 'w') as f:
                                f.write(f"{class_id} 0.5 0.5 1.0 1.0")  # Запись аннотации
                except ValueError:
                    continue  # Игнорирование ошибок преобразования


    def _create_data_yaml(self, dataset_path: Path):
        classes = set()  # Множество для хранения уникальных классов
        for d in (dataset_path / 'train').iterdir():
            if d.is_dir() and '-' in d.name:
                try:
                    classes.add(int(d.name.split('-')[0]))  # Извлечение ID класса
                except ValueError:
                    continue
        config = {
            'path': str(dataset_path),
            'train': 'train',
            'val': 'val',
            'names': {i: f"class_{i}" for i in sorted(classes)}  # Имена классов
        }
        config = {
            'path': str(dataset_path),
            'train': 'train',
            'val': 'val',
            'names': {i: f"class_{i}" for i in sorted(classes)}  # Имена классов
        }

 def train(self, model_type: str, **kwargs) -> list:
        """
        Основной метод для обучения модели
        """
        # Указываем путь к вашему архиву
        data_zip = "preprocessed_images.zip"  # <-- Важное изменение здесь

        # Проверка модели
        if model_type not in self.model_map:
            raise ValueError(f"Доступные модели: {list(self.model_map.keys())}")

        # Подготовка данных
        data_yaml = self._prepare_dataset(data_zip)

        # Инициализация модели
        model = YOLO(self.model_map[model_type])

        # Настройка параметров обучения
        params = {
            'data': str(data_yaml),
            'epochs': 100,
            'imgsz': 640,
            'batch': 16,
            'name': f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True
        }
        params.update(kwargs)

        # Обучение с замером времени
        start_time = time.time()
        model.train(**params)
        self.training_time = time.time() - start_time

        # Расчет метрик
        results = model.val()
        self.metrics = {
            'accuracy': results.results_dict['metrics/accuracy'],
            'roc_auc': results.results_dict['metrics/roc_auc'],
            'f1_score': results.results_dict['metrics/f1']
        }

        return [
            model,
            self.metrics,
            self.training_time / params['epochs'],
            self.training_time / 60
        ]

# Пример использования
if __name__ == "__main__":
    trainer = YOLOTrainer()

    result = trainer.train(
        model_type='yolov8n',
        epochs=50,
        batch=32,
        imgsz=640,
        lr0=0.01,
        weight_decay=0.0005
    )

    print("\nРезультаты обучения:")
    print(f"1. Обученная модель: {type(result[0]).__name__}")
    print(f"2. Метрики:")
    for k, v in result[1].items():
        print(f"   - {k}: {v:.4f}")
    print(f"3. Время на эпоху: {result[2]:.2f} сек")
    print(f"4. Общее время обучения: {result[3]:.2f} мин")