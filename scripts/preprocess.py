import os
import cv2
import numpy as np
from PIL import Image


def load_and_preprocess_images(data_dir, target_size=(256, 256)):
    """Load images from dataset directory and preprocess them."""
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"Loading {class_name} images...")
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, target_size)
                img = img / 255.0  # Normalize
                images.append(img)
                labels.append(label_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    return np.array(images), np.array(labels), class_names


def augment_image(image):
    """Apply basic data augmentation to an image."""
    augmented = []

    # Original
    augmented.append(image)

    # Horizontal flip
    augmented.append(cv2.flip(image, 1))

    # Rotation
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    for angle in [15, -15]:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        augmented.append(rotated)

    # Brightness adjustment
    bright = np.clip(image * 1.2, 0, 1)
    dark = np.clip(image * 0.8, 0, 1)
    augmented.append(bright)
    augmented.append(dark)

    return augmented


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    images, labels, class_names = load_and_preprocess_images(data_dir)
    print(f"\nLoaded {len(images)} images across {len(class_names)} classes")
    print(f"Classes: {class_names}")
    print(f"Image shape: {images[0].shape}")
    print(f"Label distribution: {np.bincount(labels)}")
