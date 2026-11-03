
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import os

# ---------- Create Output Folder ----------
os.makedirs('Outputs/task2_features', exist_ok=True)

# ---------- Load Motorcycle Images ----------
img1 = cv2.imread('Data/Original_Photo.jpg')
img2 = cv2.imread('Data/Original_Photo_02.jpg')

if img1 is None or img2 is None:
    raise FileNotFoundError(
        "Images not found. Check both files inside the Data folder."
    )

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ---------- 1. HOG Feature Extraction ----------
hog_features, hog_image = hog(
    gray1,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    visualize=True
)

hog_image_rescaled = exposure.rescale_intensity(
    hog_image, in_range=(0, 10)
)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Motorcycle Photo')
ax[0].axis('off')

ax[1].imshow(hog_image_rescaled, cmap='gray')
ax[1].set_title('HOG Features')
ax[1].axis('off')

plt.tight_layout()
plt.savefig('Outputs/task2_features/hog_comparison.jpg', dpi=150)
plt.close()

print("HOG descriptor length:", len(hog_features))

# ---------- 2. SIFT Keypoints and Descriptors ----------
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

if des1 is None or des2 is None:
    raise ValueError("SIFT could not detect enough features. Try clearer images.")

print("Keypoints in Image 1:", len(kp1))
print("Keypoints in Image 2:", len(kp2))

img1_kp = cv2.drawKeypoints(
    img1, kp1, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

img2_kp = cv2.drawKeypoints(
    img2, kp2, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imwrite('Outputs/task2_features/sift_keypoints_image1.jpg', img1_kp)
cv2.imwrite('Outputs/task2_features/sift_keypoints_image2.jpg', img2_kp)

# ---------- 3. Original Feature Matching ----------
bf = cv2.BFMatcher(cv2.NORM_L2)

def get_good_matches(descriptor1, descriptor2):
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    return good

good_matches = get_good_matches(des1, des2)

match_img = cv2.drawMatches(
    img1, kp1, img2, kp2, good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    'Outputs/task2_features/sift_matches_original.jpg',
    match_img
)

print("\n--- Original Pair Matching ---")
print("Good matches:", len(good_matches))

# ---------- 4. Matching After 45 Degree Rotation ----------
rows, cols = img2.shape[:2]

M = cv2.getRotationMatrix2D(
    (cols / 2, rows / 2), 45, 1
)

img2_rotated = cv2.warpAffine(img2, M, (cols, rows))
gray2_rot = cv2.cvtColor(img2_rotated, cv2.COLOR_BGR2GRAY)

kp2_rot, des2_rot = sift.detectAndCompute(gray2_rot, None)

if des2_rot is None:
    raise ValueError("No SIFT features detected after rotation.")

good_matches_rot = get_good_matches(des1, des2_rot)

match_img_rot = cv2.drawMatches(
    img1, kp1, img2_rotated, kp2_rot,
    good_matches_rot, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    'Outputs/task2_features/sift_matches_rotated.jpg',
    match_img_rot
)

print("\n--- After 45 Degree Rotation ---")
print("Good matches:", len(good_matches_rot))

# ---------- 5. Matching After 50% Resizing ----------
img2_resized = cv2.resize(img2, None, fx=0.5, fy=0.5)
gray2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

kp2_res, des2_res = sift.detectAndCompute(gray2_resized, None)

if des2_res is None:
    raise ValueError("No SIFT features detected after resizing.")

good_matches_res = get_good_matches(des1, des2_res)

match_img_res = cv2.drawMatches(
    img1, kp1, img2_resized, kp2_res,
    good_matches_res, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    'Outputs/task2_features/sift_matches_resized.jpg',
    match_img_res
)

print("\n--- After 50% Resizing ---")
print("Good matches:", len(good_matches_res))

# ---------- 6. Summary Table ----------
print("\n========== SUMMARY TABLE ==========")
print(f"{'Condition':<25}{'Good Matches':<15}")
print(f"{'Original Pair':<25}{len(good_matches):<15}")
print(f"{'Rotated 45 Degrees':<25}{len(good_matches_rot):<15}")
print(f"{'Resized 50%':<25}{len(good_matches_res):<15}")

print("\nTask 2 completed successfully!")
print("Check Outputs/task2_features folder.")
