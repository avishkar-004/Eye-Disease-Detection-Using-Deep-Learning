# Eye Disease Detection Using Deep Learning

CNN-based classification system for detecting eye diseases from retinal images using TensorFlow/Keras with a Flask web application for real-time predictions.

## Overview

This project implements a Convolutional Neural Network (CNN) to classify eye diseases into four categories:
- **Normal** - Healthy eye with no detected disease
- **Diabetic Retinopathy** - Damage to retinal blood vessels due to diabetes
- **Cataract** - Clouding of the eye lens
- **Glaucoma** - Optic nerve damage from increased eye pressure

## Dataset

The dataset contains retinal fundus images organized into four classes:
- **Cataract** - ~1000 images
- **Diabetic Retinopathy** - ~1100 images
- **Glaucoma** - ~1000 images
- **Normal** - ~1070 images

Images are preprocessed to 256x256 pixels and normalized to [0, 1] range.

## Setup and Installation

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/avishkar-004/Eye-Disease-Detection-Using-Deep-Learning.git
cd Eye-Disease-Detection-Using-Deep-Learning
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Flask App

```bash
cd Flask
python app.py
```

The app will be available at `http://localhost:5000`.

### Training the Model

```bash
python scripts/train.py
```

### Running Tests

```bash
python scripts/test_model.py
python scripts/test_flask.py
```

### Docker Deployment

```bash
docker-compose up -d
```

## Model Architecture

The custom CNN consists of:
- 3 convolutional blocks with BatchNormalization and Dropout
- Each block: Conv2D -> BatchNorm -> Conv2D -> MaxPool -> Dropout
- Fully connected layers: 512 -> 256 -> 4 (softmax)
- Training uses Adam optimizer with learning rate scheduling

## Results

- Validation Accuracy: ~92%
- Uses early stopping and learning rate reduction on plateau

## Technologies

- Python 3.10, TensorFlow 2.15, Flask 3.0, OpenCV, scikit-learn

## Author

Avishkar Pawar
