import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from PIL import Image


class EyeDiseaseDataset(Sequence):
    """Custom data loader for eye disease dataset."""

    def __init__(self, data_dir, batch_size=32, target_size=(256, 256),
                 augment=False, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle

        self.image_paths = []
        self.labels = []
        self.class_names = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        for idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(idx)

        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]

        batch_images = []
        batch_labels = []

        for i in batch_indices:
            img = Image.open(self.image_paths[i]).resize(self.target_size)
            img_array = np.array(img) / 255.0
            batch_images.append(img_array)
            batch_labels.append(self.labels[i])

        images = np.array(batch_images)
        labels = tf.keras.utils.to_categorical(
            batch_labels, num_classes=len(self.class_names)
        )
        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
