import os
import shutil
from datasets import load_dataset
from PIL import Image
from ultralytics import YOLO

import dogs

from datasets import load_dataset

ds = load_dataset("amaye15/stanford-dogs")
print(ds)
# Split dataset
train_data, test_data = ds['train'], ds['test']

# Create directories
base_dir = os.path.join(os.getcwd(), 'dataset')
splits = ['train', 'test']

map = dogs.dogs
label_to_name = dogs = {k: v.replace(' ', '_') for k, v in map.items()}

# Create directories
labels = [label.replace(' ', '_') for label in train_data.features['label'].names]

for split in splits:
    for label in labels:
        os.makedirs(os.path.join(base_dir, split, label), exist_ok=True)

# Function to save images
def save_images(data, sub_name):
    for i, item in enumerate(data):
        image = item['pixel_values']
        label_str = str(item['label'])
        label_value = label_to_name[label_str]
        image_dir = os.path.join(base_dir, sub_name, label_value)
        os.makedirs(image_dir, exist_ok=True)  # Ensure directory exists
        index = 0
        if(sub_name=='train'):
            index = 10000+i
        else:
            index = 1000+i

        image_path = os.path.join(image_dir, f"{index}-{label_value}.jpg")
        image.save(image_path)

# Save images to corresponding directories
save_images(train_data, 'train')
save_images(test_data, 'test')

print("Dataset conversion completed.")