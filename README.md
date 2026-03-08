# Traffic Signs Recognition System

## Project Overview

This project is a Traffic Sign Recognition System that uses Convolutional Neural Network (CNN) to detect and classify traffic signs in real time using a webcam.

The system captures video frames, processes the images to detect possible traffic signs, and uses a trained Convolutional Neural Network (CNN) model to classify the detected sign. The recognized traffic sign and its confidence score are then displayed on the screen.

## Features

* Real-time traffic sign detection using webcam
* Image preprocessing using OpenCV
* CNN-based classification model
* Bounding box detection around traffic signs
* Confidence score display
* Top-3 prediction probabilities
* Prediction smoothing using buffer

## Requirements

The following libraries are required:

* **Python 3.x**
* **TensorFlow**
* **OpenCV**
* **NumPy**

## Installation

Follow these steps to set up the project environment.

### 1. Install Python

Make sure **Python 3.x** is installed.

Check your Python version:
```
python --version
```

### 2. Clone the Repository

```
git clone https://github.com/your-username/traffic-sign-recognition.git
```

### 3. Install Required Libraries

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

## Usage

Run the traffic sign detection system:

```
main.py
```

The webcam will start and begin detecting the supported traffic signs in real time.

Press **Q** on the keyboard to close the camera window.

## Supported Traffic Signs

The model currently recognizes 8 traffic signs:

* No Parking
* No Entry
* Stop
* T-junction
* Crossroads
* Roundabout
* Workers Ahead
* Construction Vehicle Ahead

## Contributors

**Wasin Srirachtrkul** | DGE | 2420110286 <br>
Email: [wwasinnano@gmail.com](mailto:wwasinnano@gmail.com) / [2420110286@tni.ac.th](mailto:2420110286@tni.ac.th)

**Prince Yeoj Cavan Maranga** | DGE | 2420110013 <br>
Email: [princeyeojmaranga@gmail.com](mailto:princeyeojmaranga@gmail.com) / [2420110013@tni.ac.th](mailto:2420110013@tni.ac.th)
