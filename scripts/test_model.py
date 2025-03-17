import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_loading():
    model_path = os.path.join(os.path.dirname(__file__), "..", "Flask", "eye_disease_model.h5")
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    model = tf.keras.models.load_model(model_path)
    assert model is not None
    print("PASS: Model loads successfully")
    return model

def test_model_input_shape(model):
    test_input = np.random.rand(1, 256, 256, 3).astype(np.float32)
    output = model.predict(test_input, verbose=0)
    assert output.shape == (1, 4)
    print("PASS: Model accepts correct input shape")

def test_model_output_range(model):
    test_input = np.random.rand(1, 256, 256, 3).astype(np.float32)
    output = model.predict(test_input, verbose=0)
    assert np.all(output >= 0) and np.all(output <= 1)
    assert abs(np.sum(output) - 1.0) < 1e-5
    print("PASS: Model output is valid probability distribution")

if __name__ == "__main__":
    model = test_model_loading()
    test_model_input_shape(model)
    test_model_output_range(model)
    print("\nAll tests passed!")
