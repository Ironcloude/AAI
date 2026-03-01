import os

data_dir = "./data/task_2/Fruit_And_Vegetable_Diseases_Dataset"

for folder in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        num_images = len(os.listdir(folder_path))
        print(f"{folder} : {num_images} images")