# EE471 Week 9 In-Class Exercise: MediaPipe Computer Vision

This repository contains the solutions for the EE471 Week 9 in-class exercise. The project utilizes Google's **MediaPipe** library to perform various computer vision tasks, including a 6-task vision demo, pose estimation to detect raised arms, and face landmark detection to determine face direction.

Tasks 2 and 3 are containerized using **Cog** for reproducible machine learning environments.

---

## 🛠️ Installation

1. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install mediapipe opencv-python-headless numpy Pillow
```

2. (Optional) Install Cog for Docker containerization:
```bash
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog
```

---

## 🚀 Task 1: MediaPipe Vision Tasks (Bonus)

We implemented all **6 MediaPipe Vision Tasks** as a bonus. The script automatically downloads the necessary `.tflite` and `.task` models and runs them on sample images.

**To run the demo:**
```bash
python task1/task1_demo.py
```

**Tasks Performed & Outputs Saved in `task1/`:**
1. **Object Detection:** Detects objects and draws bounding boxes (`1_object_detection.jpg`).
2. **Image Classification:** Classifies the image and writes top 3 categories (`2_image_classification.jpg`).
3. **Image Segmentation:** Separates the subject from the background (`3_image_segmentation.jpg`).
4. **Hand Landmark Detection:** Detects and draws hand landmarks (`4_hand_landmark.jpg`).
5. **Face Landmark Detection:** Detects face mesh landmarks (`5_face_landmark.jpg`).
6. **Pose Landmark Detection:** Detects full-body pose landmarks (`6_pose_landmark.jpg`).

---

## 🦾 Task 2: Raised Arm Detection (Pose Estimation)

This task uses MediaPipe Pose Landmarker to determine which arm is raised ("left", "right", "both", or "None"). The logic uses anatomical left/right mappings (person's perspective) by comparing wrist and shoulder y-coordinates and cross-checking the x-coordinates for accuracy.

**To test the script locally:**
```bash
cd task2
python predict_arm.py
```

**Expected Output:**
- `pose-1.jpg` -> `right`
- `pose-2.jpg` -> `right`
- `pose-3.jpg` -> `both`

**To run with Cog:**
```bash
cd task2
cog predict -i image=@../pose-1.jpg
```

---

## 👤 Task 3: Face Direction Detection

This task uses MediaPipe Face Landmarker (Face Mesh) to determine the direction the face is looking ("left", "right", or "straight"). The logic calculates the 2D distances along the X-axis between the tip of the nose and the left/right edges of the face (cheeks/tragus).

**To test the script locally:**
```bash
cd task3
python predict_face.py
```

**Expected Output:**
- `face-1.png` -> `left`
- `face-2.png` -> `right`
- `face-3.png` -> `straight`

**To run with Cog:**
```bash
cd task3
cog predict -i image=@../face-1.png
```

---

## 📹 Video Recordings

*(Don't forget to record and upload the videos for each task as requested by the assignment!)*
