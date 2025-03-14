import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset path
dataset_path = os.path.abspath("dataset")  # Ensure correct dataset path

# Image settings
img_size = (256, 256)
batch_size = 32
seed = 42  

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# Load training data with augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  
    shuffle=True,
    seed=seed,
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=seed,
    subset="validation"
)

# Class Names
class_names = list(train_data.class_indices.keys())
print("Class Names:", class_names)

# Function to create CNN model
def create_cnn_model(num_classes, input_shape=(256, 256, 3)):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        
        # Data Augmentation
        data_augmentation,
        
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Flatten & Fully Connected Layers
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Create and compile model
cnn_model = create_cnn_model(num_classes=len(class_names))

# Train model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

cnn_model.fit(train_data, validation_data=val_data, epochs=25, callbacks=[early_stopping])

# Save the model
model_path = "/"
cnn_model.save(model_path)
print(f"Model saved at {model_path}")
