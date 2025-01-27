# Eye Disease Detection Using Deep Learning

CNN-based classification system for detecting eye diseases from retinal images using TensorFlow/Keras with a Flask web application for real-time predictions.

## Overview

This project implements a Convolutional Neural Network (CNN) to classify eye diseases into four categories:
- Normal
- Diabetic Retinopathy
- Cataract
- Glaucoma

## Project Structure

```
Eye-Disease-Detection-Using-Deep-Learning/
├── dataset/                    # Training dataset organized by disease class
│   ├── cataract/
│   ├── diabetic_retinopathy/
│   ├── glaucoma/
│   └── normal/
├── Flask/                      # Web application
│   ├── app.py                  # Flask server
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   ├── styles.css
│   │   └── uploads/
│   └── eye_disease_model.h5    # Trained model weights
├── notebooks/                  # Jupyter notebooks for training
├── scripts/                    # Utility scripts
├── results/                    # Training results and plots
├── requirements.txt
└── README.md
```


## Dataset

The dataset contains retinal fundus images organized into four classes:
- **Cataract** - ~1000 images
- **Diabetic Retinopathy** - ~1100 images
- **Glaucoma** - ~1000 images
- **Normal** - ~1070 images

Images are preprocessed to 256x256 pixels and normalized to [0, 1] range.
