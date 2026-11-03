
import cv2
import numpy as np
import os

# =========================================================
# SETUP
# =========================================================

VIDEO_PATH = 'Data/Original_video.mp4'
OUTPUT_DIR = 'Outputs/task4_motion'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 1. EXTRACT TWO CONSECUTIVE FRAMES
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(
        "Could not open video: " + VIDEO_PATH
    )

ret1, frame1 = cap.read()
ret2, frame2 = cap.read()

if not ret1 or not ret2:
    cap.release()
    raise ValueError("Could not extract two consecutive frames.")

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# =========================================================
# 2. FRAME DIFFERENCING
# =========================================================

diff = cv2.absdiff(gray1, gray2)

_, diff_thresh = cv2.threshold(
    diff,
    25,
    255,
    cv2.THRESH_BINARY
)

cv2.imwrite(
    f'{OUTPUT_DIR}/frame_differencing.jpg',
    diff_thresh
)

print("[Frame Differencing] Completed.")


# =========================================================
# 3. GMM BACKGROUND SUBTRACTION
# =========================================================

cap2 = cv2.VideoCapture(VIDEO_PATH)

if not cap2.isOpened():
    raise FileNotFoundError("Could not reopen video for GMM.")

# Gaussian Mixture-based background subtraction
backSub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

frame_count = 0
fgmask = None

while True:

    ret, frame = cap2.read()

    if not ret:
        break

    fgmask = backSub.apply(frame)
    frame_count += 1

    # Save foreground mask at frame 30
    if frame_count == 30:

        cv2.imwrite(
            f'{OUTPUT_DIR}/gmm_foreground_mask.jpg',
            fgmask
        )

cap2.release()

if fgmask is None:
    raise ValueError("GMM could not process any video frames.")

print("[GMM Background Subtraction] Completed.")
print("Total frames processed:", frame_count)


# =========================================================
# 4. LUCAS-KANADE OPTICAL FLOW
# =========================================================

cap3 = cv2.VideoCapture(VIDEO_PATH)

if not cap3.isOpened():
    raise FileNotFoundError("Could not reopen video for optical flow.")

ret, old_frame = cap3.read()

if not ret:
    cap3.release()
    raise ValueError("Could not read the first optical flow frame.")

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Parameters for feature detection
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

# Detect good features to track
p0 = cv2.goodFeaturesToTrack(
    old_gray,
    mask=None,
    **feature_params
)

if p0 is None:
    cap3.release()
    raise ValueError(
        "No suitable tracking features found in the first frame."
    )

# Create mask for drawing motion vectors
mask = np.zeros_like(old_frame)

ret, frame = cap3.read()

if not ret:
    cap3.release()
    raise ValueError("Could not read the second optical flow frame.")

frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Calculate optical flow
p1, st, err = cv2.calcOpticalFlowPyrLK(
    old_gray,
    frame_gray,
    p0,
    None,
    **lk_params
)

if p1 is None or st is None:
    cap3.release()
    raise ValueError("Optical flow calculation failed.")

good_new = p1[st.flatten() == 1]
good_old = p0[st.flatten() == 1]

# Draw motion vectors
for new, old in zip(good_new, good_old):

    a, b = new.ravel()
    c, d = old.ravel()

    mask = cv2.line(
        mask,
        (int(a), int(b)),
        (int(c), int(d)),
        (0, 255, 0),
        2
    )

    frame = cv2.circle(
        frame,
        (int(a), int(b)),
        5,
        (0, 0, 255),
        -1
    )

# Combine frame and motion vectors
output_flow = cv2.add(frame, mask)

cv2.imwrite(
    f'{OUTPUT_DIR}/lucas_kanade_optical_flow.jpg',
    output_flow
)

cap3.release()
cap.release()

print("[Lucas-Kanade Optical Flow] Completed.")
print("Tracked feature points:", len(good_new))


# =========================================================
# 5. FINAL SUMMARY
# =========================================================

print("\n========== TASK 4 SUMMARY ==========")

print("1. Frame Differencing: Completed")
print("2. GMM Background Subtraction: Completed")
print("3. Lucas-Kanade Optical Flow: Completed")

print("\nTask 4 completed successfully!")
print("Check Outputs/task4_motion folder.")
