# Plant Disease Detection System

A Machine Learning based web application that detects plant diseases from leaf images using Deep Learning (CNN) and provides predictions with confidence.



## Features

- Upload leaf images for disease detection
- Predicts plant disease with confidence score
- Displays uploaded image preview
- Provides basic disease information
- Simple and user-friendly UI
- Fast prediction using pretrained model



## Technologies Used

- Python
- TensorFlow / Keras
- Flask (Backend)
- HTML & CSS (Frontend)
- NumPy, Pillow




## Dataset

This project uses a subset of the PlantVillage dataset containing labeled images of plant leaves with different diseases.



## Model Details

- Model Type: Convolutional Neural Network (CNN)
- Technique: Transfer Learning
- Base Model: MobileNetV2
- Input Size: 224 x 224 images
- Output: Disease classification with probability



##  How It Works

1. User uploads a leaf image
2. Image is resized and normalized
3. Model processes the image
4. Prediction is generated using softmax probabilities
5. Result is displayed with confidence and disease info

---

## 🖥️ Project Structure
