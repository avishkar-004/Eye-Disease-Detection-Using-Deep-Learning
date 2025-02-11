import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50


def build_transfer_model(base_model_name='vgg16',
                          input_shape=(256, 256, 3),
                          num_classes=4,
                          freeze_base=True):
    """Build a transfer learning model using pretrained backbone."""
    if base_model_name == 'vgg16':
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == 'resnet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")

    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


if __name__ == "__main__":
    model = build_transfer_model('vgg16')
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
