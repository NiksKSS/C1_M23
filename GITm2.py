import os
import zipfile
import shutil
import random
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from collections import defaultdict

ARCHIVE_NAME = 'Gauge.zip'
TEMP_DIR = 'temp_extract'
ORIGINAL_DIR = 'images'
PROCESSED_DIR = 'preprocessed_images'
TARGET_SIZES = {
    'train': 23000,
    'test': 1000,
    'val': 2000
}


def preprocess_image(src_path, dst_path):
    try:
        with Image.open(src_path) as img:
            img_resized = img.resize((224, 224)).convert('RGB')

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            img_resized.save(dst_path)

            return True
    except Exception as e:
        print(f"Ошибка обработки {src_path}: {str(e)}")
        return False


def augment_image(image_path, show_results=False):
    base, ext = os.path.splitext(image_path)
    tilted_path = f"{base}_tilted{ext}"
    corrupted_path = f"{base}_corrupted{ext}"

    if os.path.exists(tilted_path) and os.path.exists(corrupted_path):
        if not show_results:
            print(f"Аугментации для {os.path.basename(image_path)} уже существуют")
        return

    try:
        with Image.open(image_path) as img:
            original = img.copy()

            # Поворот
            angle = random.uniform(-15, 15)
            rotated = img.rotate(angle, expand=False)

            # Добавление прямоугольника
            draw = ImageDraw.Draw(img)
            w, h = img.size
            rect_w = random.randint(20, 50)
            rect_h = random.randint(20, 50)
            x = random.randint(0, w - rect_w)
            y = random.randint(0, h - rect_h)
            draw.rectangle([x, y, x + rect_w, y + rect_h], fill='black')
            corrupted = img

            if show_results:
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(original)
                ax[0].set_title('Original')
                ax[1].imshow(rotated)
                ax[1].set_title(f'tilted {angle:.1f}°')
                ax[2].imshow(corrupted)
                ax[2].set_title('corrupted')
                plt.tight_layout()
                plt.show()
                return

            rotated.save(tilted_path)
            corrupted.save(corrupted_path)

            if not show_results:
                print(f"Созданы аугментации для {os.path.basename(image_path)}")

    except Exception as e:
        print(f"Ошибка аугментации: {str(e)}")


def process_images():
    processed = 0
    for root, _, files in os.walk(ORIGINAL_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src = os.path.join(root, file)
                rel_path = os.path.relpath(root, ORIGINAL_DIR)
                dst = os.path.join(PROCESSED_DIR, rel_path, file)

                if preprocess_image(src, dst):
                    processed += 1
    return processed


def balance_classes():
    train_dir = os.path.join(PROCESSED_DIR, 'train')

    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)

        if os.path.isdir(class_dir):
            images = [f for f in os.listdir(class_dir)
                      if not f.endswith(('_tilted.jpg', '_corrupted.jpg'))]

            if images:
                selected = os.path.join(class_dir, images[0])
                augment_image(selected)


def analyze_dataset():
    sizes = defaultdict(int)
    structure = {}

    for split in ['train', 'test', 'val']:
        split_dir = os.path.join(PROCESSED_DIR, split)
        if not os.path.exists(split_dir):
            continue

        class_counts = {}
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                count = len([f for f in os.listdir(class_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                class_counts[class_name] = count
                sizes[split] += count

        structure[split] = {
            'classes': len(class_counts),
            'total': sizes[split],
            'per_class': class_counts
        }

    return sizes, structure


def balance_to_target(split, target_size):
    split_dir = os.path.join(PROCESSED_DIR, split)

    current_files = []
    for root, _, files in os.walk(split_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                current_files.append(os.path.join(root, file))

    need_add = target_size - len(current_files)
    if need_add <= 0:
        return

    for i in range(need_add):
        src = current_files[i % len(current_files)]
        base, ext = os.path.splitext(src)
        dst = f"{base}_copy_{i}{ext}"
        shutil.copy2(src, dst)


def create_zip_archive():
    print("\nСоздание архива...")
    shutil.make_archive('preprocessed_images', 'zip', PROCESSED_DIR)
    print("Архив preprocessed_images.zip создан")


def generate_report(structure):
    report = [
        "Структура итогового датасета:",
        "========================================",
        f"TRAIN: {structure['train']['total']} изображений "
        f"({structure['train']['classes']} классов)",
        f"TEST: {structure['test']['total']} изображений "
        f"({structure['test']['classes']} классов)",
        f"VAL: {structure['val']['total']} изображений "
        f"({structure['val']['classes']} классов)"
    ]

    with open('dataset_report.txt', 'w') as f:
        f.write('\n'.join(report))
    print("\nОтчет сохранен в dataset_report.txt")


def demo_augmentations():
    samples = []
    train_dir = os.path.join(PROCESSED_DIR, 'train')

    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    for class_name in random.sample(classes, 2):
        class_dir = os.path.join(train_dir, class_name)
        images = [f for f in os.listdir(class_dir)
                  if not f.endswith(('_tilted.jpg', '_corrupted.jpg'))]
        if images:
            samples.append(os.path.join(class_dir, images[0]))

    for sample in samples:
        augment_image(sample, show_results=True)


def main():
    try:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        shutil.rmtree(ORIGINAL_DIR, ignore_errors=True)
        shutil.rmtree(PROCESSED_DIR, ignore_errors=True)

        with zipfile.ZipFile(ARCHIVE_NAME, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)
            print(f"Архив {ARCHIVE_NAME} распакован")

        data_root = TEMP_DIR
        items = os.listdir(TEMP_DIR)
        if len(items) == 1 and os.path.isdir(os.path.join(TEMP_DIR, items[0])):
            data_root = os.path.join(TEMP_DIR, items[0])

        os.makedirs(ORIGINAL_DIR, exist_ok=True)
        for split in ['train', 'test', 'val']:
            src = os.path.join(data_root, split)
            dst = os.path.join(ORIGINAL_DIR, split)
            if os.path.exists(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)

        processed = process_images()
        balance_classes()

        for split, target in TARGET_SIZES.items():
            balance_to_target(split, target)

        sizes, structure = analyze_dataset()
        create_zip_archive()
        generate_report(structure)

        print("\nИтоговые размеры датасета:")
        print(f"Обучающая выборка: {sizes['train']}")
        print(f"Тестовая выборка: {sizes['test']}")
        print(f"Валидационная выборка: {sizes['val']}")

        demo_augmentations()

    except Exception as e:
        print(f"Ошибка: {str(e)}")
    finally:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
