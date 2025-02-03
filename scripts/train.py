import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_model import build_eye_disease_model, compile_model
from scripts.augmentation import get_train_val_generators
from scripts.utils import plot_training_history


def train_model(data_dir, model_save_path, epochs=50, batch_size=32):
    """Train the eye disease detection model."""
    print("Loading dataset...")
    train_gen, val_gen = get_train_val_generators(
        data_dir, target_size=(256, 256), batch_size=batch_size
    )

    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Classes: {train_gen.class_indices}")

    # Build and compile model
    model = build_eye_disease_model()
    model = compile_model(model)
    model.summary()

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train
    history = train_gen.__class__
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Save training curves
    results_dir = os.path.join(os.path.dirname(data_dir), "results")
    os.makedirs(results_dir, exist_ok=True)
    plot_training_history(
        history,
        os.path.join(results_dir, "training_history.png")
    )

    print(f"\nModel saved to {model_save_path}")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

    return model, history


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    model_path = os.path.join(os.path.dirname(__file__), "..",
                               "Flask", "eye_disease_model.h5")
    train_model(data_dir, model_path)
