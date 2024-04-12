from PIL import Image
import os

# Зададим количество изображений в одной строке
images_per_row = 3

# Имена изображений для склейки
image_names = [
    "c65715c7-47338a24.jpg", "c86264ca-bf5ac7a5.jpg", "bf80a27d-38c70fee.jpg",
    "c57ec379-3d606dec.jpg", "b48dd7b4-9cc0d29a.jpg", "b89886bb-f019c7eb.jpg",
    "c768f8a1-4c7b2504.jpg", "c4742900-c0e2297d.jpg", "ca3bf9b0-55d6ae41.jpg",
    "b47cbe8c-411ffa20.jpg", "c560e52d-71c73e43.jpg", "b4b99a3e-21dfc344.jpg"
]

# Путь к папке с изображениями
images_path = r"C:/Users/ivanin.em/Desktop/2pcnet/datasets/bdd100k/val"

# Загрузка изображений
images = [Image.open(os.path.join(images_path, name)) for name in image_names]

# Вычисляем максимальные размеры для создания отступов
max_width = max(im.width for im in images)
max_height = max(im.height for im in images)
padding = 10

# Количество строк и столбцов для сетки
num_rows = len(images) // images_per_row + int(len(images) % images_per_row > 0)

# Создание нового изображения с заданными размерами
total_width = images_per_row * (max_width + padding) - padding
total_height = num_rows * (max_height + padding) - padding
grid_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))

# Размещаем изображения на сетке
for index, im in enumerate(images):
    row_num = index // images_per_row
    col_num = index % images_per_row
    x_offset = col_num * (max_width + padding)
    y_offset = row_num * (max_height + padding)
    grid_image.paste(im, (x_offset, y_offset))

grid_image.save('C:/Users/ivanin.em/Desktop/2pcnet/bad.jpg')
