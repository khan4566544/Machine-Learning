from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch  # Windows-only TTS (perfect!)

# ==================== WINDOWS TTS FUNCTION ====================
def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# ==================== WINDOWS CAMERA FIX ====================
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # This fixes 99% of Windows camera issues
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ==================== LOAD CASCADE (Reliable path) ====================
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
facedetect = cv2.CascadeClassifier(cascade_path)
if facedetect.empty():
    raise IOError("Cannot load haarcascade_frontalface_alt.xml")

# ==================== LOAD TRAINED DATA ====================
try:
    with open('data/names.pkl', 'rb') as f:
        LABELS = pickle.load(f)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
except FileNotFoundError:
    print("Error: data/names.pkl or data/faces_data.pkl not found!")
    print("Run the face collection script first.")
    exit()

print('Shape of Faces matrix --> ', FACES.shape)

# ==================== TRAIN KNN ====================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# ==================== LOAD BACKGROUND IMAGE ====================
if not os.path.exists("background.png"):
    print("Warning: background.png not found! Using black background.")
    imgBackground = np.zeros((720, 1280, 3), dtype=np.uint8)
else:
    imgBackground = cv2.imread("background.png")
    if imgBackground is None:
        imgBackground = np.zeros((720, 1280, 3), dtype=np.uint8)

# ==================== CREATE FOLDERS ====================
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

# ==================== ATTENDANCE TRACKING (Avoid duplicates) ====================
COL_NAMES = ['NAME', 'TIME']
already_attended = set()  # Track who attended today

# ==================== MAIN LOOP ====================
print("\n=== Face Recognition Attendance System ===")
print("Press 'o' when face is detected to mark attendance")
print("Press 'q' to quit\n")

while True:
    ret, frame = video.read()
    if not ret:
        print("Camera error!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        prediction = knn.predict(resized_img)
        confidence = knn.predict_proba(resized_img).max()

        name = str(prediction[0])
        ts = time.time()
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

        # Draw bounding boxes and name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -1)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"{int(confidence*100)}%", (x + w - 100, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)

        # Attendance logic on 'o' key
        attendance = [name, timestamp]

    # Paste camera feed into background
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Face Recognition Attendance System - Press O to Mark | Q to Quit", imgBackground)

    key = cv2.waitKey(1) & 0xFF

    # Mark attendance when 'o' is pressed and face is detected
    if key == ord('o') and len(faces) > 0:
        if name not in already_attended:
            speak(f"Attendance taken for {name}")
            csv_path = f"Attendance/Attendance_{date}.csv"

            if os.path.isfile(csv_path):
                with open(csv_path, "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance)
            else:
                with open(csv_path, "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COL_NAMES)
                    writer.writerow(attendance)

            already_attended.add(name)
            print(f"Attendance Marked: {name} at {timestamp}")
        else:
            speak(f"Attendance already taken for {name} today")
            print(f"Already marked: {name}")

    # Quit
    if key == ord('q'):
        speak("System shutting down")
        break

# ==================== CLEANUP ====================
video.release()
cv2.destroyAllWindows()
print("Attendance system closed. Goodbye!")