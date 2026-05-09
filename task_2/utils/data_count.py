from pathlib import Path
import os 

data_dir = Path(__file__).resolve().parent.parent / "data" / "golden_dataset" 

total_samples = 0

for folder in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        num_images = len(os.listdir(folder_path))
        print(f"{folder} : {num_images} images")
        total_samples += num_images

print("-" * 30)
print(f"Total samples: {total_samples}")
