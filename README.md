# CCA 3 -- Computer Vision Assignment

This repository contains the implementation of the CCA 3 practical
assignment for the Computer Vision course. It covers image restoration,
image segmentation, feature extraction, object recognition, and motion
estimation using Python and computer vision libraries.

## Student Details

-   **Name:** Panchajanya Datta
-   **PRN:** 1272250524
-   **Course:** MCA
-   **Programming Language:** Python

## Objective

To implement and study computer vision techniques using photographs and
video footage, including image denoising, segmentation, feature
matching, object detection, and motion estimation.

## Technologies Used

  -----------------------------------------------------------------------
  Library / Tool                      Purpose
  ----------------------------------- -----------------------------------
  OpenCV (`cv2`)                      Image processing, filtering,
                                      morphology, SIFT, Haar Cascade,
                                      video processing, and optical flow

  NumPy                               Numerical and array operations

  Matplotlib                          Image visualization

  scikit-image                        HOG feature extraction and
                                      PSNR/SSIM metrics

  Ultralytics YOLO                    Deep-learning-based object
                                      detection
  -----------------------------------------------------------------------

## Project Structure

``` text
CCA3_Assignment/
│
├── Data/
│   ├── Original_Photo.jpg
│   ├── Original_Photo_02.jpg
│   ├── face_photo.jpeg
│   ├── haarcascade_frontalface_default.xml
│   └── Original_video.mp4
│
├── Outputs/
│   ├── task1_restoration/
│   ├── task2_features/
│   ├── task3_detection/
│   └── task4_motion/
│
├── src/
│   ├── task1_restoration_segmentation.py
│   ├── task2_feature_matching.py
│   ├── task3_object_recognition.py
│   └── task4_motion_estimation.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

> **Note:** The folder and file names in this README match the paths
> used by the updated scripts. Keep the capitalization and extensions
> consistent, especially `Data/` and `Outputs/`.

## Tasks Implemented

### Task 1: Image Restoration and Segmentation

Input: `Data/Original_Photo.jpg`

-   Added Gaussian noise and salt-and-pepper noise.
-   Applied Gaussian filtering to reduce Gaussian noise.
-   Applied median filtering to reduce salt-and-pepper noise.
-   Evaluated image quality using Peak Signal-to-Noise Ratio (PSNR) and
    Structural Similarity Index (SSIM).
-   Applied Otsu thresholding to create a binary image.
-   Performed erosion, dilation, opening, and closing.
-   Applied Canny edge detection.
-   Performed watershed segmentation.

Outputs are saved in `Outputs/task1_restoration/`.

### Task 2: Feature Extraction and Feature Matching

Inputs: - `Data/Original_Photo.jpg` - `Data/Original_Photo_02.jpg`

-   Extracted and visualized Histogram of Oriented Gradients (HOG)
    features.
-   Detected SIFT keypoints and descriptors in both images.
-   Matched SIFT features using a Brute-Force Matcher and Lowe's ratio
    test.
-   Repeated feature matching after rotating the second image by 45
    degrees.
-   Repeated feature matching after resizing the second image to 50%.
-   Printed a comparison of the number of good matches for the original,
    rotated, and resized image pairs.

Outputs are saved in `Outputs/task2_features/`.

### Task 3: Object Recognition

Inputs: - `Data/face_photo.jpeg` - `Data/Original_Photo.jpg` -
`Data/Original_video.mp4` - `Data/haarcascade_frontalface_default.xml`

-   Used a pretrained Haar Cascade classifier for face detection in the
    face photograph.
-   Used the pretrained YOLOv8n model to detect objects in the KTM
    motorcycle photograph.
-   Extracted and sampled video frames for YOLO object detection.
-   Printed detected class names and confidence scores.
-   Compared the detection results from Haar Cascade and YOLO.

The face photograph is processed separately from the video. The YOLO
model is expected to identify supported object classes, such as
motorcycles, when they are sufficiently visible.

Outputs are saved in `Outputs/task3_detection/`.

### Task 4: Motion Estimation

Input: `Data/Original_video.mp4`

-   Extracted consecutive video frames.
-   Applied frame differencing to identify changes between consecutive
    frames.
-   Applied MOG2 Gaussian Mixture Model (GMM) background subtraction to
    obtain a foreground mask.
-   Applied Lucas--Kanade optical flow to track image features between
    frames and visualize motion vectors.

Outputs are saved in `Outputs/task4_motion/`.

## Setup and Installation

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CCA3_Assignment
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the URL of your GitHub
repository.

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows Command Prompt:**

``` bash
.venv\Scripts\activate
```

**Windows PowerShell:**

``` powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the repository does not contain a complete `requirements.txt`, the
scripts use these main packages:

``` bash
pip install opencv-contrib-python numpy matplotlib scikit-image ultralytics
```

The `opencv-contrib-python` package includes the SIFT functionality used
by the project. Avoid installing `opencv-python` and
`opencv-contrib-python` together in the same environment because both
provide the `cv2` module.

## Running the Tasks

Run commands from the project root directory, with the virtual
environment activated.

``` bash
python src/task1_restoration_segmentation.py
python src/task2_feature_matching.py
python src/task3_object_recognition.py
python src/task4_motion_estimation.py
```

The first execution of Task 3 may download the pretrained `yolov8n.pt`
model if it is not already available. An internet connection may be
required for this initial download.

## Input Data Requirements

Before running the scripts, make sure the `Data/` folder contains the
required files with the exact names:

  ---------------------------------------------------------------------------
  File                                    Used for
  --------------------------------------- -----------------------------------
  `Original_Photo.jpg`                    Task 1, Task 2, and Task 3

  `Original_Photo_02.jpg`                 Task 2

  `face_photo.jpeg`                       Haar Cascade face detection in Task
                                          3

  `haarcascade_frontalface_default.xml`   Haar Cascade classifier

  `Original_video.mp4`                    YOLO video-frame detection in Task
                                          3 and motion estimation in Task 4
  ---------------------------------------------------------------------------

Use your own photographs and self-recorded video where required by the
assignment. Preserve the original input files. If your video is large,
consider sharing it through an appropriate cloud-drive link rather than
committing a large file to GitHub, subject to your assignment submission
requirements.

## Output Directories

Each script creates or writes its results to the corresponding
directory:

-   `Outputs/task1_restoration/` -- noisy images, restored images,
    binary and morphological results, Canny edges, and watershed result.
-   `Outputs/task2_features/` -- HOG visualization, SIFT keypoint
    images, and feature-matching comparisons.
-   `Outputs/task3_detection/` -- Haar Cascade and YOLO detection
    images, plus the selected video frame.
-   `Outputs/task4_motion/` -- frame-differencing result, GMM foreground
    mask, and Lucas--Kanade optical-flow visualization.

## Notes

-   Run each command from the repository root so that the relative
    `Data/` and `Outputs/` paths resolve correctly.
-   Detection results depend on image quality, object visibility,
    lighting, camera angle, and the pretrained model's supported
    classes.
-   A detected region is not necessarily a correct detection; inspect
    the saved annotated images and report false positives or missed
    detections where applicable.
-   The number of detected objects and feature matches can vary with the
    input data and model execution.

## Author

**Panchajanya Datta**\
**PRN:** 1272250524\
**Course:** MCA
