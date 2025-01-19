import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def plot_training_history(history, save_path=None):
    """Plot training and validation accuracy/loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def save_predictions_sample(model, test_images, test_labels,
                             class_names, save_path=None):
    """Save a grid of sample predictions."""
    n_samples = min(16, len(test_images))
    indices = np.random.choice(len(test_images), n_samples, replace=False)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, idx in enumerate(indices):
        ax = axes[i // 4][i % 4]
        img = test_images[idx]
        true_label = class_names[test_labels[idx]]
        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
        pred_label = class_names[np.argmax(pred)]

        ax.imshow(img)
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label}",
                     color=color, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
