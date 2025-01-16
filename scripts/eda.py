import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import Counter


def analyze_dataset(data_dir):
    """Perform exploratory data analysis on the eye disease dataset."""
    class_counts = {}
    image_sizes = []

    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[class_name] = len(images)

        # Sample a few images for size analysis
        for img_name in images[:50]:
            try:
                img = Image.open(os.path.join(class_path, img_name))
                image_sizes.append(img.size)
            except Exception:
                pass

    return class_counts, image_sizes


def plot_class_distribution(class_counts, save_path=None):
    """Plot the distribution of classes in the dataset."""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values(),
                   color=['#2196F3', '#FF5722', '#4CAF50', '#FFC107'])
    plt.xlabel('Disease Class')
    plt.ylabel('Number of Images')
    plt.title('Eye Disease Dataset - Class Distribution')

    for bar, count in zip(bars, class_counts.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    class_counts, sizes = analyze_dataset(data_dir)

    print("Class Distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} images")

    total = sum(class_counts.values())
    print(f"\nTotal images: {total}")

    plot_class_distribution(class_counts,
                            os.path.join(results_dir, "class_distribution.png"))
