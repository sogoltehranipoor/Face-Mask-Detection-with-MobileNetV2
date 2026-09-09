# Face-Mask-Detection-with-MobileNetV2
A real-time Face Mask Detection system built using Deep Learning, Transfer Learning, and OpenCV.

The system detects faces from a webcam and classifies each detected face into one of two classes:

* `With Mask`
* `Without Mask`

## Project Overview

This project uses a pretrained **MobileNetV2** CNN as the feature extractor. The model is first trained using Feature Extraction and then improved through Fine-Tuning.

The trained model is finally used for real-time inference through a webcam.

## Methodology

### 1. Dataset

The dataset contains two classes:

```text
face_mask/
├── with_mask/
└── without_mask/
```

Images are loaded using TensorFlow's dataset pipeline rather than loading the entire dataset into memory.

### 2. Preprocessing

The images are resized to:

```text
224 × 224 × 3
```

MobileNetV2 preprocessing is applied before feeding images into the CNN.

Data augmentation is also used, including:

* Random horizontal flipping
* Random rotation
* Random zoom

### 3. Feature Extraction

A pretrained MobileNetV2 model with ImageNet weights is used:

```python
MobileNetV2(
    weights="imagenet",
    include_top=False
)
```

The pretrained CNN layers are initially frozen, and a new classification head is added.

### 4. Classification Head

The classification head consists of:

```text
GlobalAveragePooling2D
        ↓
Dense(128, ReLU)
        ↓
Dropout(0.5)
        ↓
Dense(1, Sigmoid)
```

### 5. Fine-Tuning

After Feature Extraction, the last 30 layers of MobileNetV2 are unfrozen.

A smaller learning rate is used during Fine-Tuning to allow the pretrained features to adapt to the Face Mask dataset without significantly destroying the previously learned features.

### 6. Real-Time Detection

OpenCV is used for webcam inference.

The pipeline is:

```text
Webcam
   ↓
Face Detection
   ↓
Crop Face
   ↓
Resize to 224×224
   ↓
MobileNetV2 Preprocessing
   ↓
CNN Prediction
   ↓
With Mask / Without Mask
```

Press **`q`** to close the webcam.

##  Technologies

* Python
* TensorFlow
* Keras
* MobileNetV2
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn

##  Project Structure

```text
Face-Mask-Detection/
│
├── face_mask_training.ipynb
├── face_mask_webcam.ipynb
├── face_mask.h5
└── README.md
```

##  Skills Demonstrated

* Convolutional Neural Networks (CNN)
* Transfer Learning
* Feature Extraction
* Fine-Tuning
* Image Classification
* Data Augmentation
* Image Preprocessing
* Face Detection
* Real-Time Computer Vision
* Webcam-based Inference

##  Future Improvements

Possible improvements include:

* Improving detection accuracy
* Using a more advanced face detector
* Optimizing real-time inference speed
* Supporting multiple faces simultaneously
* Deploying the model as a web application
