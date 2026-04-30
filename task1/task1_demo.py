import urllib.request
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe model URLs
MODELS = {
    'object_detector': ('efficientdet_lite0.tflite', 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite'),
    'image_classifier': ('efficientnet_lite0.tflite', 'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite'),
    'image_segmenter': ('deeplabv3.tflite', 'https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite'),
    'hand_landmarker': ('hand_landmarker.task', 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'),
    'face_landmarker': ('face_landmarker.task', 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'),
    'pose_landmarker': ('pose_landmarker.task', 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task')
}

os.makedirs('models', exist_ok=True)
os.makedirs('task1', exist_ok=True)

def download_models():
    print("Modeller indiriliyor...")
    for model_key, (filename, url) in MODELS.items():
        filepath = os.path.join('models', filename)
        if not os.path.exists(filepath):
            print(f"Indiriliyor: {filename}...")
            urllib.request.urlretrieve(url, filepath)
    print("Tum modeller hazir.\n")

# --- VISUALIZATION HELPERS ---
def draw_simple_landmarks(rgb_image, landmarks_list):
    annotated_image = np.copy(rgb_image)
    h, w, _ = annotated_image.shape
    for landmarks in landmarks_list:
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)
    return annotated_image

def draw_objects_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, (0, 255, 0), 3)
        
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (bbox.origin_x, bbox.origin_y - 10)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    return annotated_image

# --- 6 TASKS DEMO ---
def run_object_detection(image_path):
    print("1. Object Detection çalıştırılıyor...")
    base_options = python.BaseOptions(model_asset_path='models/efficientdet_lite0.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.3)
    with vision.ObjectDetector.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        annotated_image = draw_objects_on_image(image.numpy_view(), result)
        cv2.imwrite('task1/1_object_detection.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print("-> task1/1_object_detection.jpg kaydedildi.\n")

def run_image_classification(image_path):
    print("2. Image Classification çalıştırılıyor...")
    base_options = python.BaseOptions(model_asset_path='models/efficientnet_lite0.tflite')
    options = vision.ImageClassifierOptions(base_options=base_options, max_results=3)
    with vision.ImageClassifier.create_from_options(options) as classifier:
        image = mp.Image.create_from_file(image_path)
        result = classifier.classify(image)
        annotated_image = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)
        y = 50
        for category in result.classifications[0].categories:
            text = f"{category.category_name}: {category.score:.2f}"
            cv2.putText(annotated_image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            y += 50
        cv2.imwrite('task1/2_image_classification.jpg', annotated_image)
        print("-> task1/2_image_classification.jpg kaydedildi.\n")

def run_image_segmentation(image_path):
    print("3. Image Segmentation çalıştırılıyor...")
    base_options = python.BaseOptions(model_asset_path='models/deeplabv3.tflite')
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        image = mp.Image.create_from_file(image_path)
        result = segmenter.segment(image)
        category_mask = result.category_mask
        image_data = image.numpy_view()
        image_rgb = image_data[:, :, :3] # Force 3 channels
        mask = np.squeeze(category_mask.numpy_view()) # Force 2D
        
        bg_color = (192, 192, 192)
        condition = np.stack((mask,) * 3, axis=-1) > 0.1
        output_image = np.where(condition, image_rgb, bg_color)
        cv2.imwrite('task1/3_image_segmentation.jpg', cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
        print("-> task1/3_image_segmentation.jpg kaydedildi.\n")

def run_hand_landmark(image_path):
    print("4. Hand Landmark Detection çalıştırılıyor...")
    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    with vision.HandLandmarker.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        annotated_image = draw_simple_landmarks(image.numpy_view(), result.hand_landmarks)
        cv2.imwrite('task1/4_hand_landmark.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print("-> task1/4_hand_landmark.jpg kaydedildi.\n")

def run_face_landmark(image_path):
    print("5. Face Landmark Detection çalıştırılıyor...")
    base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    with vision.FaceLandmarker.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        annotated_image = draw_simple_landmarks(image.numpy_view(), result.face_landmarks)
        cv2.imwrite('task1/5_face_landmark.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print("-> task1/5_face_landmark.jpg kaydedildi.\n")

def run_pose_landmark(image_path):
    print("6. Pose Landmark Detection çalıştırılıyor...")
    base_options = python.BaseOptions(model_asset_path='models/pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    with vision.PoseLandmarker.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        annotated_image = draw_simple_landmarks(image.numpy_view(), result.pose_landmarks)
        cv2.imwrite('task1/6_pose_landmark.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print("-> task1/6_pose_landmark.jpg kaydedildi.\n")

if __name__ == '__main__':
    download_models()
    pose_img = 'pose-1.jpg'
    face_img = 'face-1.png'
    
    print("--- TASK 1: 6 Vision Görevi Başlıyor ---\n")
    run_object_detection(pose_img)
    run_image_classification(pose_img)
    run_image_segmentation(pose_img)
    run_hand_landmark(pose_img)
    run_face_landmark(face_img)
    run_pose_landmark(pose_img)
    print("--- TASK 1 TAMAMLANDI ---")
