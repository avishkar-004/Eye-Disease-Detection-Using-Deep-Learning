import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Define the absolute path for uploads inside the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load trained model
MODEL_PATH = os.path.join(BASE_DIR, "eye_disease_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    img = Image.open(image_path).resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    class_names = ["Normal", "Diabetic Retinopathy", "Cataract", "Glaucoma"]
    return class_names[class_index]


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
            file.save(filepath)
            prediction = predict_image(filepath)
            return render_template("index.html", filename=filename, prediction=prediction)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)


@app.route("/about")
def about():
    return render_template("about.html")


@app.errorhandler(404)
def page_not_found(e):
    return render_template("index.html", message="Page not found"), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template("index.html", message="Internal server error"), 500
