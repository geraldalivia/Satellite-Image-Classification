# 🛰️ Satellite Image Classification Using CNN (Convolutional Neural Network)

This project showcases a deep learning pipeline using CNNs to classify satellite images into four categories: cloudy, desert, green_area, and water.

## 📁 Data Source
Satellite Imgaes from [Satellite Image Classification](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification). 
Split Data Image for build model from [Satellite_dataset_split]([https://drive.google.com/file/d/1--d07m7J4CniV6pfScx_S6sde5XJu05J/view?usp=drive_link](https://drive.google.com/drive/folders/1hlZ0AoyLmD-GM3TND6EdLArtIrK-J-q6?usp=sharing)).
Detail of dataset satellite
- Total Images: 5631 gambar
- Classes: 4 kelas (cloudy, desert, green_area, water)
- Images per Class: 1,500 gambar 
- Image Format: PNG/JPG
Dataset Split:
- Training: 70% (1,050 per kelas)
- Validation: 15% (225 per kelas)
- Testing: 15% (225 per kelas)

## 🚀 Highlights
- Data loading from **Kaggle** and saved dataset to **Google Drive**
- Preprocessing with ImageDataGenerator
- Exploratory Data Analysis (EDA) using matplotlib, seaborn, and class distribution plots
- Built and trained a custom CNN model for satellite image classification
-  Model optimization using:
  - EarlyStopping – prevent overfitting
  - ReduceLROnPlateau – dynamic learning rate adjustment
  - ModelCheckpoint - by callbacks
- Evaluated on a separate test set for reliable performance metrics
- Visualized training & validation accuracy/loss over epochs
- Saved models in multiple formats:
  - ✅ SavedModel (default TensorFlow format)
  - ✅ TensorFlow Lite (for edge/mobile deployment)
  - ✅ TensorFlow.js (for browser-based inference)
  - ✅ model.keras (for inference needed)

## 🧠 Technologies Used
- Python – Core programming language for the entire pipeline.
- Google Colab – Cloud-based development environment with access to Google Drive for dataset management.
- TensorFlow & Keras – Deep learning framework for building, training, and evaluating Convolutional Neural Networks (CNNs).
- TensorFlow.js
- TensorFlow Lite
- Matplotlib & Seaborn – For data visualization and exploratory data analysis (EDA).
- NumPy & Pandas For efficient numerical operations and dataset handling.
- ImageDataGenerator – For image preprocessing and real-time data augmentation.
- EarlyStopping & ReduceLROnPlateau – For optimized training via callbacks.
- Pathlib & OS Modules - For file and directory management during data preprocessing.

## 🏗️ Model Features:
- Total Parameters: ~13M parameters
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Data Augmentation: Rotation, zoom, flip, shift
- Regularization: Dropout layers
- Callbacks: Early stopping, learning rate reduction

## 📈 Performance
Test Accuracy: 93.74%
Test Loss: 0.1575
Training Time: 20 epochs
Model Size: ~50-250 MB

## 📦 Model Export
There are three model that i saved into format, include:
- `saved_model/` – savedmodel
- `tflite` – TensorFlow Lite
- `tfjs/` – TensorFlow.js
