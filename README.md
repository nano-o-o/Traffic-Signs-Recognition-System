# 🚦 Traffic Signs Recognition System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)

## 📌 Project Overview

This project is a **Traffic Sign Recognition System** that uses **Computer Vision and Deep Learning** to detect and classify traffic signs in real time using a webcam.

The system captures video frames, processes the images to detect possible traffic signs, and uses a trained **Convolutional Neural Network (CNN)** model to classify the detected sign.

The recognized traffic sign and its confidence score are then displayed on the screen.

This project demonstrates the application of **image processing, machine learning, and real-time detection systems**.

---

# 🧠 Features

* Real-time traffic sign detection using webcam
* Image preprocessing using OpenCV
* CNN-based classification model
* Bounding box detection around traffic signs
* Confidence score display
* Top-3 prediction probabilities
* Prediction smoothing using buffer

---

# 🧰 Requirements

The following technologies and libraries are required:

* **Python 3.x**
* **TensorFlow / Keras**
* **OpenCV**
* **NumPy**

---

# ⚙️ Installation

Follow these steps to set up the project environment.

## 1. Install Python

Make sure **Python 3.x** is installed.

Check your Python version:

```
python --version
```

or

```
python3 --version
```

---

## 2. Clone the Repository

```
git clone https://github.com/your-username/traffic-sign-recognition.git
```

```
cd traffic-sign-recognition
```

---

## 3. Install Required Libraries

Install the required Python libraries using pip:

```
pip install tensorflow
pip install opencv-python
pip install numpy
```

Or install everything at once:

```
pip install tensorflow opencv-python numpy
```

---

## 4. Verify Installation

Run Python:

```
python
```

Test importing the libraries:

```
import cv2
import numpy
import tensorflow
```

If no errors appear, the installation was successful.

---

# ▶️ Usage

Run the traffic sign detection system:

```
python detection.py
```

The webcam will start and begin detecting traffic signs in real time.

Press **Q** on the keyboard to close the camera window.

---


# 🧠 Supported Traffic Signs

The model currently recognizes:

* construction_vehicle_ahead
* crossroads
* no_entry
* no_parking
* roundabout
* stop
* t_junction
* workers_ahead

---

# 👨‍💻 Contributors

**Wasin Srirachtrkul**
DGE | 2420110286
Email: [wwasinnano@gmail.com](mailto:wwasinnano@gmail.com) / [2420110286@tni.ac.th](mailto:2420110286@tni.ac.th)

**Prince Yeoj Cavan Maranga**
DGE | 2420110013
Email: [princeyeojmaranga@gmail.com](mailto:princeyeojmaranga@gmail.com) / [2420110013@tni.ac.th](mailto:2420110013@tni.ac.th)

---

# 📚 Technologies Used

* Computer Vision
* Deep Learning
* Convolutional Neural Networks (CNN)
* Real-time image processing
