import os
import sys
import tempfile
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Flask"))

def create_test_image(path):
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    img.save(path)
    return path

def test_home_page():
    from app import app
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    print("PASS: Home page renders")

if __name__ == "__main__":
    test_home_page()
    print("\nAll Flask tests passed!")
