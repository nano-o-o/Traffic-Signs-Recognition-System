"""
cv2 (OpenCV) – Used for image processing and webcam access.
numpy – Used for numerical operations and handling image arrays.
load_model – Loads the trained deep learning model.
deque – Stores recent predictions to stabilize results.
Counter – Helps determine the most common prediction.
"""
# These libraries allow the program to capture images, process them, and classify traffic signs using AI.
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, Counter


"""
This loads a pre-trained neural network model.
The file traffic_sign_model.h5 contains the trained weights.
The model was trained earlier to recognize traffic signs.
""""
#The model will analyze images and predict which traffic sign is detected.
model = load_model("traffic_sign_model.h5")

# This list maps numeric model outputs to readable labels.
# The AI outputs numbers, so we convert them to human-readable traffic sign names.
class_names = [
    "construction_vehicle_ahead",
    "crossroads",
    "no_entry",
    "no_parking",
    "roundabout",
    "stop",
    "t_junction",
    "workers_ahead"
]

"""
Sometimes AI predictions flicker or change quickly.
Using a buffer allows the program to choose the most common prediction.
This makes detection more stable.
"""
# A buffer stores the last 8 predictions.
prediction_buffer = deque(maxlen=8)

# Capture real-time video for traffic sign detection.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

"""
These variables store previous results.
This ensures results remain visible even if detection momentarily disappears.
"""
frame_count = 0

last_label = "No sign detected"
last_confidence = 0

last_top3 = None
last_top3_probs = None

while True:

"""
cap.read() captures a frame from the webcam.
ret checks if capture was successful
"""
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1


"""
Reduces computation
Makes edge detection easier
"""
# Color image → grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# This smooths the image and removes noise.
    blur = cv2.GaussianBlur(gray,(5,5),0)

# Edges are important because traffic signs have distinct shapes.
    edges = cv2.Canny(blur,50,150)

# Clearer object shapes.
    kernel = np.ones((5,5),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations=1)
    edges = cv2.erode(edges,kernel,iterations=1)

"""
Contours are object boundaries.
The program searches for possible traffic sign shapes.
"""
    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_crop = None
    box = None

    if len(contours) > 0:

# Among all detected shapes, the program selects the largest one.
        largest = max(contours, key=cv2.contourArea)

"""
Very small objects are ignored.
Prevents detecting noise or random objects.
"""
        if cv2.contourArea(largest) > 1500:

            x,y,w,h = cv2.boundingRect(largest)

"""
Traffic signs are usually square or circular.
This check removes long rectangles or unrelated objects.
"""
            aspect_ratio = w / float(h)

            if 0.6 < aspect_ratio < 1.4:

"""
Extracts only the region containing the possible sign.
This cropped image will be sent to the AI model.
"""
                detected_crop = frame[y:y+h, x:x+w]
                box = (x,y,w,h)

                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

    if detected_crop is not None:
# Removes noise before classification.
        img = cv2.GaussianBlur(detected_crop,(5,5),0)

# CLAHE improves image contrast.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        l = clahe.apply(l)

        lab = cv2.merge((l,a,b))
        img = cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

"""
The model expects 64×64 images.
So every image must be resized.
"""
        img = cv2.resize(img,(64,64))

"""
Pixel values become 0–1 instead of 0–255.
This improves neural network performance.
"""
        img = img/255.0

# Deep learning models expect batch input.
        img = np.expand_dims(img,axis=0)

        if frame_count % 2 == 0:
            
#The neural network analyzes the image and returns probabilities for each class.
            predictions = model.predict(img,verbose=0)

"""
argmax → best class
max → confidence level
"""
            class_id = np.argmax(predictions)
            confidence = np.max(predictions)

            sorted_probs = np.sort(predictions[0])
            margin = sorted_probs[-1] - sorted_probs[-2]

"""
Prediction must be strong and clearly better than others.
This prevents wrong classifications.
"""
            if confidence > 0.75 and margin > 0.15:
                prediction_buffer.append(class_id)
                last_confidence = confidence

                last_top3 = np.argsort(predictions[0])[-3:][::-1]
                last_top3_probs = predictions[0]

            if confidence < 0.4:
                
# Adds prediction to buffer for smoothing results.                
                prediction_buffer.clear()

    if len(prediction_buffer) > 0:
        most_common = Counter(prediction_buffer).most_common(1)[0][0]
        last_label = class_names[most_common]

    text = f"{last_label} ({last_confidence:.2f})"

# Shows the detected traffic sign name and confidence on the screen.
    if box is not None:
        x,y,w,h = box
        cv2.putText(frame,
                    text,
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

""" Displays the top 3 possible predictions with probabilities. """
    if last_top3 is not None:

        y_offset = 130
        for i in last_top3:

            label_text = f"{class_names[i]}: {last_top3_probs[i]:.2f}"

            cv2.putText(frame,
                        label_text,
                        (20,y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,255,255),
                        2)

            y_offset += 25
            
"""Shows the real-time detection window."""
    cv2.imshow("Traffic Sign Detection",frame)

# Press Q to exit the program.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"""
Stops webcam
Closes all windows
"""
cap.release()
cv2.destroyAllWindows()
