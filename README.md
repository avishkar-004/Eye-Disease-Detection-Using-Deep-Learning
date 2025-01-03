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

