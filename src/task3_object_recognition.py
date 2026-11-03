
import cv2
import os
from ultralytics import YOLO

# =========================================================
# SETUP
# =========================================================

OUTPUT_DIR = 'Outputs/task3_detection'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# PART A: HAAR CASCADE - FACE DETECTION
# =========================================================

face_img = cv2.imread('Data/face_photo.jpeg')

if face_img is None:
    raise FileNotFoundError(
        "Face photograph not found: Data/face_photo.jpeg"
    )

gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

# Improve image contrast
gray_face = cv2.equalizeHist(gray_face)

# Load Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(
    'Data/haarcascade_frontalface_default.xml'
)

if face_cascade.empty():
    raise FileNotFoundError(
        "Could not load Haar Cascade XML file."
    )

# Detect faces with stricter settings
faces = face_cascade.detectMultiScale(
    gray_face,
    scaleFactor=1.1,
    minNeighbors=6,
    minSize=(80, 80)
)

# Draw detected face regions
img_haar = face_img.copy()

for (x, y, w, h) in faces:

    cv2.rectangle(
        img_haar,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

    cv2.putText(
        img_haar,
        'Face',
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        2
    )

cv2.imwrite(
    f'{OUTPUT_DIR}/haar_face_detection.jpg',
    img_haar
)

print("\n========== HAAR CASCADE ==========")
print("Face regions detected:", len(faces))

if len(faces) == 0:
    print("No face detected. Try a clearer front-facing photograph.")

# =========================================================
# PART B: YOLO - KTM MOTORCYCLE IMAGE DETECTION
# =========================================================

print("\nLoading YOLO model...")

model = YOLO('yolov8n.pt')

bike_img = cv2.imread('Data/Original_Photo.jpg')

if bike_img is None:
    raise FileNotFoundError(
        "KTM photograph not found: Data/Original_Photo.jpg"
    )

results_bike = model(bike_img, conf=0.25)

bike_detections = results_bike[0].boxes

img_yolo_bike = results_bike[0].plot()

cv2.imwrite(
    f'{OUTPUT_DIR}/yolo_motorcycle_result.jpg',
    img_yolo_bike
)

print("\n========== YOLO - KTM PHOTOGRAPH ==========")

if len(bike_detections) == 0:
    print("No objects detected.")

else:
    for box in bike_detections:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        print(
            f"Detected: {class_name} "
            f"(Confidence: {confidence:.2f})"
        )

# =========================================================
# PART C: YOLO - VIDEO FRAME OBJECT DETECTION
# =========================================================

video_path = 'Data/Original_video.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(
        "Could not open video: " + video_path
    )

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames == 0:
    cap.release()
    raise ValueError("Video contains no readable frames.")

print("\nTotal video frames:", total_frames)

# Sample approximately 30 frames
sample_interval = max(1, total_frames // 30)

best_frame = None
best_results = None
best_count = -1
motorcycle_found = False

for frame_number in range(0, total_frames, sample_interval):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()

    if not ret:
        continue

    results = model(frame, conf=0.25)
    detections = results[0].boxes

    if len(detections) > best_count:
        best_count = len(detections)
        best_frame = frame.copy()
        best_results = results

    # COCO class ID 3 = motorcycle
    for box in detections:

        class_id = int(box.cls[0])

        if class_id == 3:
            best_frame = frame.copy()
            best_results = results
            motorcycle_found = True
            break

    if motorcycle_found:
        break

cap.release()

if best_frame is None:
    raise ValueError("Could not extract a readable video frame.")

# Save selected video frame
cv2.imwrite(
    f'{OUTPUT_DIR}/video_frame_for_yolo.jpg',
    best_frame
)

# Save annotated video frame
video_detections = best_results[0].boxes

img_yolo_video = best_results[0].plot()

cv2.imwrite(
    f'{OUTPUT_DIR}/yolo_video_result.jpg',
    img_yolo_video
)

print("\n========== YOLO - VIDEO FRAME ==========")

if len(video_detections) == 0:
    print("No objects detected.")

else:
    for box in video_detections:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        print(
            f"Detected: {class_name} "
            f"(Confidence: {confidence:.2f})"
        )

if motorcycle_found:
    print("Motorcycle detected in sampled video frame.")
else:
    print("Motorcycle not detected in sampled frames.")

# =========================================================
# PART D: COMPARISON SUMMARY
# =========================================================

print("\n========== TASK 3 COMPARISON SUMMARY ==========")

print(
    f"{'Test Case':<25}"
    f"{'Method':<18}"
    f"{'Detections'}"
)

print(
    f"{'Face photograph':<25}"
    f"{'Haar Cascade':<18}"
    f"{len(faces)} face region(s)"
)

print(
    f"{'KTM photograph':<25}"
    f"{'YOLO':<18}"
    f"{len(bike_detections)} object(s)"
)

print(
    f"{'Video frame':<25}"
    f"{'YOLO':<18}"
    f"{len(video_detections)} object(s)"
)

print("\nTask 3 completed successfully!")
print(f"Check {OUTPUT_DIR} folder.")
