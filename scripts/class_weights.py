import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def calculate_class_weights(data_dir):
    """Calculate class weights for imbalanced dataset handling."""
    class_counts = {}

    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        count = len([f for f in os.listdir(class_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[class_name] = count

    labels = []
    for idx, (name, count) in enumerate(sorted(class_counts.items())):
        labels.extend([idx] * count)

    labels = np.array(labels)
    classes = np.unique(labels)

    weights = compute_class_weight('balanced', classes=classes, y=labels)
    weight_dict = {i: w for i, w in enumerate(weights)}

    print("Class weights:")
    for idx, (name, count) in enumerate(sorted(class_counts.items())):
        print(f"  {name}: count={count}, weight={weight_dict[idx]:.4f}")

    return weight_dict


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    weights = calculate_class_weights(data_dir)
    print(f"\nWeight dict for training: {weights}")
