import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle
import random
import os
from PIL import Image, ImageDraw, ImageFont

# Сопоставление имен файлов с аннотациями
def map_images_to_annotations(annotations_data):
    # Создание словаря для сопоставления имени файла с аннотациями
    image_to_annotations = {}
    for image in annotations_data['images']:
        # Фильтрация аннотаций по ID изображения
        image_annotations = [ann for ann in annotations_data['annotations'] if ann['image_id'] == image['id']]
        # Добавление в словарь, используя имя файла как ключ
        image_to_annotations[image['file_name']] = image_annotations
    return image_to_annotations


# Функция для отображения подписей
def add_subplot_caption(axs, caption, pos):
    axs[pos].text(0.5, -0.2, caption, va='center', ha='center', fontsize=12, fontname="Times New Roman", transform=axs[pos].transAxes)


# Функция для чтения аннотаций
def read_annotations(annotations_path):
    with open(annotations_path) as f:
        return json.load(f)

# Функция для генерации случайного цвета
def get_random_color():
    return [random.randint(0, 255) for _ in range(3)]

def generate_annotated_images(original_images, annotations_data, category_info, output_folder):
    annotated_images_paths = []
    for img_path in original_images:
        img_name = os.path.basename(img_path)
        image = cv2.imread(img_path)
        annotations = annotations_data.get(img_name, [])

        # Отрисовка аннотаций на изображении
        for ann in annotations:
            bbox = ann['bbox']
            category_id = ann['category_id']
            category_name = category_info[category_id]
            color = get_random_color()
            
            # Рисуем bounding box
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), color, 2)
            
            # Добавляем текст
            text_x, text_y = int(bbox[0]), int(bbox[1] - 10)
            cv2.putText(image, category_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Сохраняем аннотированное изображение
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, image)
        annotated_images_paths.append(output_path)
    
    return annotated_images_paths


# Функция для отрисовки аннотаций на изображении
def draw_annotations(image, annotations, category_info):
    colors = {category_id: get_random_color() for category_id in category_info.keys()}
    
    for ann in annotations:
        bbox = ann['bbox']
        category_id = ann['category_id']
        category_name = category_info[category_id]
        color = colors[category_id]
        
        # Рисуем bounding box
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), color, 2)
        
        # Добавляем тень для текста (для улучшения читаемости)
        # text_x, text_y = int(bbox[0]), int(bbox[1] - 10)
        # cv2.putText(image, category_name, (text_x+1, text_y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 4)
        
        # Добавляем текст
        # cv2.putText(image, category_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image


from matplotlib import gridspec

# Функция для создания макета изображений с использованием GridSpec
def create_image_collage(original_images, annotated_images, results_paths, category_info, output_path):
    spacing = 10  # Отступ между изображениями и между рядами
    caption_height = 70  # Высота пространства под подписи
    captions = ['Исходное изображение', 'Аннотация', "Используемая архитектура", "TDD", "Adaptive Teacher"]  # Список подписей

    # Вычисляем общую ширину коллажа
    with Image.open(original_images[0]) as img:
        column_width = img.width
    total_width = column_width * (2 + len(results_paths))

    # Вычисляем высоту для одного ряда изображений
    max_height = max(Image.open(img).height for img in original_images)

    # Вычисляем общую высоту коллажа
    collage_height = (max_height + spacing) * len(original_images) + caption_height

    # Создаем фон для коллажа
    collage = Image.new('RGB', (total_width, collage_height), color="white")

    # Загружаем шрифт для подписей
    try:
        font = ImageFont.truetype("times.ttf", size=63)  # Пробуем загрузить Times New Roman
    except IOError:
        font = ImageFont.load_default()  # Используем шрифт по умолчанию, если Times New Roman не доступен

    draw = ImageDraw.Draw(collage)

    # Вставляем изображения в коллаж
    y_offset = 0
    for i, orig_path in enumerate(original_images):
        x_offset = 0
        paths = [orig_path, annotated_images[i]] + [rp[i] for rp in results_paths]
        for path in paths:
            with Image.open(path) as img:
                collage.paste(img, (x_offset, y_offset))
                x_offset += img.width + spacing
        y_offset += max_height + spacing

    # Добавляем подписи под изображениями
    y_offset = collage_height - caption_height
    for i, caption in enumerate(captions):
        text_bbox = draw.textbbox((0, 0), caption, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (column_width + spacing) * i + (column_width - text_width) / 2 - spacing // 2
        text_y = y_offset
        draw.text((text_x, text_y), caption, fill="black", font=font)

    # Сохраняем коллаж
    collage.save(output_path)

    # Масштабирование коллажа, если необходимо
    max_width_for_word = 1700  # Максимальная ширина изображения, хорошо отображаемая в Word
    aspect_ratio = collage_height / total_width
    desired_height = int(max_width_for_word * aspect_ratio)  # Высчитываем высоту для сохранения пропорций

    # Изменяем размер изображения
    collage_resized = collage.resize((max_width_for_word, desired_height), Image.Resampling.LANCZOS)
    resized_output_path = os.path.splitext(output_path)[0] + "_resized.jpg"
    collage_resized.save(resized_output_path)
    # collage.save(output_path)

    # cv2.imwrite('C:/Users/ivanin.em.MAIN/Desktop/2pcnet/datasets/combined_image.jpg', cv2.cvtColor(image, cv2.COLOR_RGBA2BGR))

def get_category_info(annotations_data):
    category_info = {category['id']: category['name'] for category in annotations_data['categories']}
    return category_info

# Список путей к исходным изображениям
original_images = [
    "C:/Users/ivanin.em.MAIN/Desktop/2pcnet/datasets/bdd100k/val/b1cd1e94-26dd524f.jpg",
    "C:/Users/ivanin.em.MAIN/Desktop/2pcnet/datasets/bdd100k/val/b2d502aa-ef17ffbd.jpg",
    "C:/Users/ivanin.em.MAIN/Desktop/2pcnet/datasets/bdd100k/val/b2e2f4ed-6ba045d0.jpg"
]

# Путь к аннотациям
annotations_path = "C:/Users/ivanin.em.MAIN/Desktop/2pcnet/datasets/bdd100k/val_night.json"

# Списки путей к результатам
result_images_1 = [
    "C:/Users/ivanin.em.MAIN/Desktop/2pcnet/datasets/result_image1.jpg",
    "C:/Users/ivanin.em.MAIN/Desktop/2pcnet/datasets/result_image2.jpg",
    "C:/Users/ivanin.em.MAIN/Desktop/2pcnet/datasets/result_image3.jpg"
]
result_images_2 = [
    "C:/Users/ivanin.em.MAIN/Desktop/TDD/datasets/result_image1.jpg",
    "C:/Users/ivanin.em.MAIN/Desktop/TDD/datasets/result_image2.jpg",
    "C:/Users/ivanin.em.MAIN/Desktop/TDD/datasets/result_image3.jpg"
]
result_images_3 = [
    "C:/Users/ivanin.em.MAIN/Desktop/adaptive_teacher/datasets/result_image1.jpg",
    "C:/Users/ivanin.em.MAIN/Desktop/adaptive_teacher/datasets/result_image2.jpg",
    "C:/Users/ivanin.em.MAIN/Desktop/adaptive_teacher/datasets/result_image3.jpg"
]

# Чтение аннотаций и сопоставление категорий
annotations_data = read_annotations(annotations_path)
category_info = get_category_info(annotations_data)

# Сопоставление имен файлов с аннотациями
image_to_annotations = map_images_to_annotations(annotations_data)

output_folder = "C:/Users/ivanin.em.MAIN/Desktop/2pcnet/datasets/"
annotated_images = generate_annotated_images(original_images, image_to_annotations, category_info, output_folder)

# Создание макета изображений
# create_image_collage(original_images, image_to_annotations, [result_images_1, result_images_2, result_images_3], category_info, "C:/Users/ivanin.em.MAIN/Desktop/2pcnet/datasets/combined_image.jpg")
create_image_collage(original_images, annotated_images, [result_images_1, result_images_2, result_images_3], category_info, "C:/Users/ivanin.em.MAIN/Desktop/2pcnet/datasets/combined_image.jpg")
