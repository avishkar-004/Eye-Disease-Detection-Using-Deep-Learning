import os
import shutil
import random


def split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """Split dataset into train, validation, and test sets."""
    test_ratio = 1 - train_ratio - val_ratio

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        n_train = int(len(images) * train_ratio)
        n_val = int(len(images) * val_ratio)

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split_name, split_images in splits.items():
            split_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(split_dir, exist_ok=True)

            for img_name in split_images:
                src = os.path.join(class_path, img_name)
                dst = os.path.join(split_dir, img_name)
                shutil.copy2(src, dst)

            print(f"{class_name}/{split_name}: {len(split_images)} images")


if __name__ == "__main__":
    source = os.path.join(os.path.dirname(__file__), "..", "dataset")
    output = os.path.join(os.path.dirname(__file__), "..", "dataset_split")
    split_dataset(source, output)
    print("Dataset split complete.")
