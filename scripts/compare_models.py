import os
import sys
import json
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_model import build_eye_disease_model, compile_model
from models.transfer_model import build_transfer_model
from scripts.augmentation import get_train_val_generators


def compare_models(data_dir, results_dir, epochs=20):
    """Compare custom CNN vs transfer learning models."""
    train_gen, val_gen = get_train_val_generators(data_dir, batch_size=32)

    models_config = {
        'Custom CNN': build_eye_disease_model(),
        'VGG16 Transfer': build_transfer_model('vgg16'),
    }

    results = {}

    for name, model in models_config.items():
        print(f"\nTraining {name}...")
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            train_gen, epochs=epochs,
            validation_data=val_gen, verbose=1
        )

        results[name] = {
            'train_acc': history.history['accuracy'],
            'val_acc': history.history['val_accuracy'],
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
        }

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, data in results.items():
        ax1.plot(data['val_acc'], label=name)
        ax2.plot(data['val_loss'], label=name)

    ax1.set_title('Validation Accuracy Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Validation Loss Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "model_comparison.png"), dpi=150)
    plt.close()

    print("\nModel comparison saved.")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    compare_models(data_dir, results_dir)
