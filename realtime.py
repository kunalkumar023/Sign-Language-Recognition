import os
# from sklearn.base import accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score


model = load_model('smnist.h5')

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

while True:
    _, frame = cap.read()

    # Convert the frame to RGB for hands processing
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = frame.shape[1]
            y_min = frame.shape[0]

            for lm in handLMs.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

            # Draw a rectangle around the hand
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Crop the region of interest (hand sign)
            roi = frame[y_min:y_max, x_min:x_max]

            # Resize the ROI for model input
            roi = cv2.resize(roi, (28, 28))
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Prepare the data for prediction
            pixeldata = roi_gray / 255.0
            pixeldata = pixeldata.reshape(-1, 28, 28, 1)

            # Make prediction
            prediction = model.predict(pixeldata)
            predicted_class = np.argmax(prediction)

            # Display the predicted class
            cv2.putText(frame, f"{letterpred[predicted_class]}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)



    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if k == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
