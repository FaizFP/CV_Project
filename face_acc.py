import cv2
import face_recognition
from ultralytics import YOLO
import numpy as np
import os
import dlib
from datetime import datetime
import csv

# Load dlib's pre-trained models
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Landmark model
detector = dlib.get_frontal_face_detector()

# Path ke folder database wajah
database_path = "C:\\Users\\faiz\\Documents\\tgstgs\\SEMESTER 3\\Sistem Cerdas\\absensi"

# Load model YOLOv8
model = YOLO("yolov8n-face.pt") 
model.info()

# Fungsi untuk memuat database wajah
def load_face_database(path):
    face_encodings = []
    face_names = []
    for file in os.listdir(path):
        if file.endswith(('.jpg', '.png')):
            image_path = os.path.join(path, file)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                face_encodings.append(encoding[0])
                face_names.append(os.path.splitext(file)[0])
    return face_encodings, face_names

# Load wajah dari database
known_face_encodings, known_face_names = load_face_database(database_path)

# Fungsi untuk mencocokkan wajah
def match_face(face_encoding, known_encodings, known_names):
    matches = face_recognition.compare_faces(known_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
    if matches:
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            return known_names[best_match_index]
    return "Unknown"

# Fungsi untuk menghitung Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])  # Vertical distance
    B = np.linalg.norm(eye[2] - eye[4])  # Vertical distance
    C = np.linalg.norm(eye[0] - eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold EAR untuk deteksi kedipan
EAR_THRESHOLD = 0.25
blink_detected = 1  # Indikator kedipan terdeteksi

# CSV file untuk log
csv_file = "attendance_log.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Time", "Status"])

# Mulai kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi wajah dengan YOLOv8
    results = model(frame, imgsz=640, conf=0.5)
    for result in results:  # Iterasi melalui hasil YOLO
        for box in result.boxes.xyxy:  # Bounding box
            x1, y1, x2, y2 = map(int, box[:4])

            # Crop wajah
            face_frame = frame[y1:y2, x1:x2]

            # Proses hanya jika wajah terdeteksi
            if face_frame.size > 0:
                rgb_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_frame)

                if face_encodings:
                    face_encoding = face_encodings[0]

                    # Cocokkan dengan database
                    name = match_face(face_encoding, known_face_encodings, known_face_names)

                    # Landmark detection
                    gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
                    dets = detector(gray, 0)

                    for det in dets:
                        shape = predictor(gray, det)
                        landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

                        # Mata kiri dan kanan
                        left_eye = landmarks[36:42]
                        right_eye = landmarks[42:48]

                        # Hitung EAR
                        left_ear = calculate_ear(left_eye)
                        right_ear = calculate_ear(right_eye)
                        avg_ear = (left_ear + right_ear) / 2.0

                        # Deteksi kedipan
                        if avg_ear < EAR_THRESHOLD and not blink_detected:
                            blink_detected = True  # Tandai kedipan terdeteksi
                            # Ketika kedipan terdeteksi, tampilkan "IN" di video
                            cv2.putText(frame, "IN", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            # Ketika kedipan terdeteksi, catat ke CSV
                            with open(csv_file, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "IN"])
                            print(f"Detected: {name} - IN")

                        if avg_ear >= EAR_THRESHOLD:
                            blink_detected = False  # Reset kedipan jika mata terbuka

                    # Tampilkan nama pada bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("Face Recognition with Blink Detection", frame)

    # Keluar jika menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
