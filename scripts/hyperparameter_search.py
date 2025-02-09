import os
import sys
import itertools
import json
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_model import build_eye_disease_model
from scripts.augmentation import get_train_val_generators


def grid_search(data_dir, results_path):
    """Perform grid search over hyperparameters."""
    param_grid = {
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [16, 32, 64],
        'dropout_rate': [0.3, 0.5]
    }

    results = []

    for lr, bs, dr in itertools.product(
        param_grid['learning_rate'],
        param_grid['batch_size'],
        param_grid['dropout_rate']
    ):
        print(f"\n{'='*50}")
        print(f"Testing: lr={lr}, batch_size={bs}, dropout={dr}")
        print(f"{'='*50}")

        train_gen, val_gen = get_train_val_generators(
            data_dir, batch_size=bs
        )

        model = build_eye_disease_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            train_gen,
            epochs=10,
            validation_data=val_gen,
            verbose=0
        )

        best_val_acc = max(history.history['val_accuracy'])
        result = {
            'learning_rate': lr,
            'batch_size': bs,
            'dropout_rate': dr,
            'best_val_accuracy': float(best_val_acc)
        }
        results.append(result)
        print(f"Best val accuracy: {best_val_acc:.4f}")

        tf.keras.backend.clear_session()

    # Save results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda x: x['best_val_accuracy'])
    print(f"\nBest configuration: {best}")
    return results


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    results_path = os.path.join(os.path.dirname(__file__), "..",
                                 "results", "hyperparameter_results.json")
    grid_search(data_dir, results_path)
