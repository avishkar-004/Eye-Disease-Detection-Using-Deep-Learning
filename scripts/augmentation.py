import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generator(rotation_range=20, width_shift=0.2,
                          height_shift=0.2, horizontal_flip=True,
                          zoom_range=0.2, fill_mode='nearest'):
    """Create a Keras ImageDataGenerator with specified augmentation params."""
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        horizontal_flip=horizontal_flip,
        zoom_range=zoom_range,
        fill_mode=fill_mode,
        rescale=1.0 / 255.0
    )
    return datagen


def get_train_val_generators(data_dir, target_size=(256, 256),
                              batch_size=32, validation_split=0.2):
    """Create train and validation generators from directory."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=validation_split
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator


if __name__ == "__main__":
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    train_gen, val_gen = get_train_val_generators(data_dir)
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Class indices: {train_gen.class_indices}")
