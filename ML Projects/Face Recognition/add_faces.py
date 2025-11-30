import cv2
import pickle
import numpy as np
import os

# ==================== WINDOWS FIXES ====================
# 1. Use CAP_DSHOW to avoid camera lag/freezing on Windows
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Optional: Set resolution and buffer size for smoother performance on Windows
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ==================== LOAD CASCADE ====================
# Use OpenCV's built-in path - works everywhere, no need to worry about relative paths
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
facedetect = cv2.CascadeClassifier(cascade_path)

if facedetect.empty():
    raise IOError("Failed to load haarcascade_frontalface_default.xml - check OpenCV installation")

# ==================== CREATE DATA FOLDER IF NOT EXISTS ====================
if not os.path.exists('data'):
    os.makedirs('data')

# ==================== INPUT NAME ====================
name = input("Enter Your Name: ").strip()
if not name:
    name = "Unknown"

faces_data = []
i = 0

print("\n=== Look at the camera and move your head slightly ===")
print("Collecting 100 face samples... Press 'q' to quit early.\n")

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Cannot read frame from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop and resize face
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))

        # Collect every 10th frame to avoid duplicates (up to 100 samples)
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)
            print(f"Collected: {len(faces_data)}/100")

        i += 1

        # Draw rectangle and counter
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        count_text = f"Samples: {len(faces_data)}/100"
        cv2.putText(frame, count_text, (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collecting Face Data - Press 'q' to quit", frame)

    # Press 'q' or collect 100 samples
    if cv2.waitKey(1) == ord('q') or len(faces_data) >= 100:
        break

# ==================== CLEANUP ====================
video.release()
cv2.destroyAllWindows()

# If we didn't collect enough, pad with the last image (optional)
while len(faces_data) < 100:
    faces_data.append(faces_data[-1] if faces_data else np.zeros((50, 50, 3), dtype=np.uint8))

# Convert to numpy array and flatten
faces_data = np.asarray(faces_data, dtype=np.uint8)
faces_data = faces_data.reshape(100, -1)  # 100 samples × (50×50×3)

# ==================== SAVE NAMES ====================
names_path = 'data/names.pkl'
if not os.path.exists(names_path):
    names_list = [name] * 100
else:
    with open(names_path, 'rb') as f:
        names_list = pickle.load(f)
    names_list += [name] * 100

with open(names_path, 'wb') as f:
    pickle.dump(names_list, f)


faces_path = 'data/faces_data.pkl'
if not os.path.exists(faces_path):
    all_faces = faces_data
else:
    with open(faces_path, 'rb') as f:
        all_faces = pickle.load(f)
    all_faces = np.append(all_faces, faces_data, axis=0)

with open(faces_path, 'wb') as f:
    pickle.dump(all_faces, f)

print(f"\n=== SUCCESS! ===")
print(f"Face data for '{name}' has been saved successfully!")
print(f"Total people in database: {len(all_faces) // 100}")