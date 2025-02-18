import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CLASS_NAMES, IMG_SIZE


def load_model(model_path):
    """Load the trained model."""
    return tf.keras.models.load_model(model_path)


def predict_single_image(model, image_path):
    """Predict disease class for a single image."""
    img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]

    display_names = ["Normal", "Diabetic Retinopathy", "Cataract", "Glaucoma"]

    return {
        'class': display_names[predicted_idx],
        'confidence': float(confidence),
        'probabilities': {
            name: float(prob)
            for name, prob in zip(display_names, predictions[0])
        }
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    model_path = os.path.join(os.path.dirname(__file__), "..",
                               "Flask", "eye_disease_model.h5")
    model = load_model(model_path)

    result = predict_single_image(model, sys.argv[1])
    print(f"Predicted: {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("All probabilities:")
    for cls, prob in result['probabilities'].items():
        print(f"  {cls}: {prob:.4f}")
