import urllib.request
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
    for model_key, (filename, url) in MODELS.items():
        filepath = os.path.join('models', filename)
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(url, filepath)

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
        text = f"{category.category_name} ({category.score:.2f})"
        cv2.putText(annotated_image, text, (bbox.origin_x, bbox.origin_y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    return annotated_image

def run_object_detection(image_path, suffix="1"):
    options = vision.ObjectDetectorOptions(base_options=python.BaseOptions(model_asset_path='models/efficientdet_lite0.tflite'), score_threshold=0.3)
    with vision.ObjectDetector.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        annotated_image = draw_objects_on_image(image.numpy_view(), result)
        cv2.imwrite(f'task1/1_object_detection_{suffix}.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

def run_image_classification(image_path, suffix="1"):
    options = vision.ImageClassifierOptions(base_options=python.BaseOptions(model_asset_path='models/efficientnet_lite0.tflite'), max_results=3)
    with vision.ImageClassifier.create_from_options(options) as classifier:
        image = mp.Image.create_from_file(image_path)
        result = classifier.classify(image)
        annotated_image = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)
        y = 50
        for category in result.classifications[0].categories:
            cv2.putText(annotated_image, f"{category.category_name}: {category.score:.2f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            y += 50
        cv2.imwrite(f'task1/2_image_classification_{suffix}.jpg', annotated_image)

def run_image_segmentation(image_path, suffix="1"):
    options = vision.ImageSegmenterOptions(base_options=python.BaseOptions(model_asset_path='models/deeplabv3.tflite'), output_category_mask=True)
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        image = mp.Image.create_from_file(image_path)
        result = segmenter.segment(image)
        mask = np.squeeze(result.category_mask.numpy_view())
        condition = np.stack((mask,) * 3, axis=-1) > 0.1
        output_image = np.where(condition, image.numpy_view()[:, :, :3], (192, 192, 192))
        cv2.imwrite(f'task1/3_image_segmentation_{suffix}.jpg', cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_RGB2BGR))

def run_hand_landmark(image_path, suffix="1"):
    options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='models/hand_landmarker.task'), num_hands=2)
    with vision.HandLandmarker.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        annotated_image = draw_simple_landmarks(image.numpy_view(), result.hand_landmarks)
        cv2.imwrite(f'task1/4_hand_landmark_{suffix}.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

def run_face_landmark(image_path, suffix="1"):
    options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='models/face_landmarker.task'), num_faces=1)
    with vision.FaceLandmarker.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        annotated_image = draw_simple_landmarks(image.numpy_view(), result.face_landmarks)
        cv2.imwrite(f'task1/5_face_landmark_{suffix}.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

def run_pose_landmark(image_path, suffix="1"):
    options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='models/pose_landmarker.task'))
    with vision.PoseLandmarker.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        annotated_image = draw_simple_landmarks(image.numpy_view(), result.pose_landmarks)
        cv2.imwrite(f'task1/6_pose_landmark_{suffix}.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    download_models()
    pose_images = ['pose-1.jpg', 'pose-2.jpg', 'pose-3.jpg']
    face_images = ['face-1.png', 'face-2.png', 'face-3.png']
    
    print("--- TASK 1: Tüm görseller işleniyor ---")
    for i, img in enumerate(pose_images):
        run_object_detection(img, suffix=f"pose{i+1}")
        run_image_classification(img, suffix=f"pose{i+1}")
        run_image_segmentation(img, suffix=f"pose{i+1}")
        run_hand_landmark(img, suffix=f"pose{i+1}")
        run_pose_landmark(img, suffix=f"pose{i+1}")
        
    for i, img in enumerate(face_images):
        run_face_landmark(img, suffix=f"face{i+1}")
        
    print("--- TASK 1 TAMAMLANDI ---")
