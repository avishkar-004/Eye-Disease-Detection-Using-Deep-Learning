"""Configuration parameters for the eye disease detection model."""

# Dataset parameters
IMG_SIZE = 256
NUM_CLASSES = 4
CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
DISPLAY_NAMES = ["Normal", "Diabetic Retinopathy", "Cataract", "Glaucoma"]

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10
LR_REDUCE_PATIENCE = 5

# Data augmentation
ROTATION_RANGE = 20
WIDTH_SHIFT = 0.2
HEIGHT_SHIFT = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True

# Paths
DATASET_DIR = "dataset"
MODEL_SAVE_PATH = "Flask/eye_disease_model.h5"
RESULTS_DIR = "results"
UPLOADS_DIR = "Flask/static/uploads"

# Flask configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
