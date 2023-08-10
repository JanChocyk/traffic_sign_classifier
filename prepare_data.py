import os
import random
import shutil

# DANE TRENINGOWE
data_dir = os.path.join(os.getcwd(), 'dataset\\DATA')
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
split_ratio = 0.8  # Stosunek podziału

# Tworzenie katalogów train i valid, jeśli nie istnieją
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Iteracja przez katalogi klas
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        train_class_dir = os.path.join(train_dir, class_name)
        valid_class_dir = os.path.join(valid_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(valid_class_dir, exist_ok=True)

        # Pobranie listy plików obrazów w klasie
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(image_files)

        # Obliczenie liczby zdjęć do treningu i walidacji
        num_train = int(len(image_files) * split_ratio)
        train_images = image_files[:num_train]
        valid_images = image_files[num_train:]

        # Przeniesienie zdjęć do odpowiednich katalogów
        for img in train_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy(src, dst)

        for img in valid_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(valid_class_dir, img)
            shutil.copy(src, dst)

# DANE TESTOWE
test_dir = os.path.join(os.getcwd(), 'dataset\\TEST')
output_dir = os.path.join(os.getcwd(), 'dataset\\DATA\\test')

# Utwórz katalogi dla poszczególnych klas
for class_num in range(58):
    class_dir = os.path.join(output_dir, str(class_num))
    os.makedirs(class_dir, exist_ok=True)

# Przenieś zdjęcia do odpowiednich katalogów
for filename in os.listdir(test_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        class_num = int(filename[:3])
        class_dir = os.path.join(output_dir, str(class_num))
        src_path = os.path.join(test_dir, filename)
        dst_path = os.path.join(class_dir, filename)
        shutil.copy(src_path, dst_path)