import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, Counter

model = load_model("traffic_sign_model.h5")

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

prediction_buffer = deque(maxlen=8)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

frame_count = 0

last_label = "No sign detected"
last_confidence = 0

last_top3 = None
last_top3_probs = None

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blur,50,150)

    kernel = np.ones((5,5),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations=1)
    edges = cv2.erode(edges,kernel,iterations=1)

    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_crop = None
    box = None

    if len(contours) > 0:

        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 1500:

            x,y,w,h = cv2.boundingRect(largest)

            aspect_ratio = w / float(h)

            if 0.6 < aspect_ratio < 1.4:

                detected_crop = frame[y:y+h, x:x+w]
                box = (x,y,w,h)

                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

    if detected_crop is not None:

        img = cv2.GaussianBlur(detected_crop,(5,5),0)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        l = clahe.apply(l)

        lab = cv2.merge((l,a,b))
        img = cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(64,64))

        img = img/255.0
        img = np.expand_dims(img,axis=0)

        if frame_count % 2 == 0:

            predictions = model.predict(img,verbose=0)

            class_id = np.argmax(predictions)
            confidence = np.max(predictions)

            sorted_probs = np.sort(predictions[0])
            margin = sorted_probs[-1] - sorted_probs[-2]

            if confidence > 0.75 and margin > 0.15:
                prediction_buffer.append(class_id)
                last_confidence = confidence

                last_top3 = np.argsort(predictions[0])[-3:][::-1]
                last_top3_probs = predictions[0]

            if confidence < 0.4:
                prediction_buffer.clear()

    if len(prediction_buffer) > 0:
        most_common = Counter(prediction_buffer).most_common(1)[0][0]
        last_label = class_names[most_common]

    text = f"{last_label} ({last_confidence:.2f})"

    if box is not None:
        x,y,w,h = box
        cv2.putText(frame,
                    text,
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

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

    cv2.imshow("Traffic Sign Detection",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()