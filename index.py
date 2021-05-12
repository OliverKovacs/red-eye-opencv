# Oliver Kovacs 2021

import cv2
import numpy as np
import time

camera = 0
cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

face_cascade = cv2.CascadeClassifier("face.xml")
eye_cascade = cv2.CascadeClassifier("eye.xml")
eye = cv2.imread("eye.png")

speed = 0.002

def now():
    return round(time.time() * 1000)

def color():
    return (
        int(127.5 * (np.sin(now() * speed * np.pi) + 1)),
        int(127.5 * (np.sin(now() * speed * np.pi + (2/3) * np.pi) + 1)),
        int(127.5 * (np.sin(now() * speed * np.pi + (4/3) * np.pi) + 1))
    )

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color(), 2)

    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for (ex, ey, ew, eh) in eyes:
        eye_sq = cv2.addWeighted(frame[ey: ey + eh, ex: ex + ew], 0.5 , cv2.resize(eye, (ew, eh)), 1, 1)
        frame[ey: ey + eh, ex: ex + ew] = eye_sq

    cv2.imshow("red-eye-opencv", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()