import os
import sys
import tensorflow as tf


def print_model_details(model_path):
    """Print detailed model architecture and parameter counts."""
    model = tf.keras.models.load_model(model_path)

    print("=" * 60)
    print("Eye Disease Detection Model Summary")
    print("=" * 60)
    model.summary()

    total_params = model.count_params()
    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable = total_params - trainable

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {non_trainable:,}")
    print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), "..",
                               "Flask", "eye_disease_model.h5")
    print_model_details(model_path)
