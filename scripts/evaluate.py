import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.augmentation import get_train_val_generators


def evaluate_model(model_path, data_dir):
    """Evaluate the trained model on validation data."""
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    _, val_gen = get_train_val_generators(data_dir, batch_size=32)

    print("Evaluating model...")
    loss, accuracy = model.evaluate(val_gen, verbose=1)
    print(f"\nValidation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Get predictions
    val_gen.reset()
    predictions = model.predict(val_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes

    # Class names
    class_names = list(val_gen.class_indices.keys())

    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes,
                                target_names=class_names))

    return true_classes, predicted_classes, class_names


def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Eye Disease Detection')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    model_path = os.path.join(os.path.dirname(__file__), "..",
                               "Flask", "eye_disease_model.h5")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    true_labels, pred_labels, class_names = evaluate_model(model_path, data_dir)
    plot_confusion_matrix(
        true_labels, pred_labels, class_names,
        os.path.join(results_dir, "confusion_matrix.png")
    )


def plot_normalized_confusion_matrix(true_labels, pred_labels,
                                      class_names, save_path):
    """Generate normalized confusion matrix."""
    import seaborn as sns
    cm = confusion_matrix(true_labels, pred_labels, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
