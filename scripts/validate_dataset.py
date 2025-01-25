import os
from PIL import Image


def validate_images(data_dir):
    """Validate all images in the dataset directory."""
    corrupted = []
    valid_count = 0

    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path)
                img.verify()
                valid_count += 1
            except Exception as e:
                corrupted.append((img_path, str(e)))

    print(f"Valid images: {valid_count}")
    print(f"Corrupted images: {len(corrupted)}")

    if corrupted:
        print("\nCorrupted files:")
        for path, error in corrupted:
            print(f"  {path}: {error}")

    return valid_count, corrupted


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    validate_images(data_dir)
