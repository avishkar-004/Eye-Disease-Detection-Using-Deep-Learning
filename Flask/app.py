import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Define the absolute path for uploads inside the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current project directory
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')  # Ensure it's inside 'Eye_Disease_Detection'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create directory if it does not exist

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load trained model
MODEL_PATH = os.path.join(BASE_DIR, "eye_disease_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction function
def predict_image(image_path):
    img = Image.open(image_path).resize((256, 256))  # Resize to match model input
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img)
    class_index = np.argmax(predictions)  # Get predicted class index
    class_names = ["Normal", "Diabetic Retinopathy", "Cataract", "Glaucoma"]  # Adjust according to your dataset
    return class_names[class_index]  # Return predicted class name

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file part")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", message="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)  # Save the file to the correct path

            # Predict the image
            prediction = predict_image(filepath)

            return render_template("index.html", filename=filename, prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
