import os
import sys
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def batch_predict(model_path, image_dir, output_path):
    """Run predictions on all images in a directory."""
    model = tf.keras.models.load_model(model_path)
    class_names = ["Normal", "Diabetic Retinopathy", "Cataract", "Glaucoma"]

    results = []
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    for img_name in tqdm(image_files, desc="Predicting"):
        img_path = os.path.join(image_dir, img_name)
        try:
            img = Image.open(img_path).resize((256, 256))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(predictions[0])

            results.append({
                'image': img_name,
                'predicted_class': class_names[predicted_idx],
                'confidence': float(predictions[0][predicted_idx])
            })
        except Exception as e:
            results.append({
                'image': img_name,
                'error': str(e)
            })

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nPredictions saved to {output_path}")
    print(f"Total: {len(results)}, Successful: {sum(1 for r in results if 'error' not in r)}")


if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), "..",
                               "Flask", "eye_disease_model.h5")
    image_dir = sys.argv[1] if len(sys.argv) > 1 else "test_images"
    output_path = "results/batch_predictions.json"
    batch_predict(model_path, image_dir, output_path)
