import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def detect_raised_arm(image_path, model_path='../models/pose_landmarker.task'):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    
    with vision.PoseLandmarker.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        
        if not result.pose_landmarks:
            return "None"
            
        landmarks = result.pose_landmarks[0]
        
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        
        is_left_up = left_wrist.y < left_shoulder.y
        is_right_up = right_wrist.y < right_shoulder.y
        
        if is_left_up and is_right_up:
            return "both"
        elif is_left_up and not is_right_up:
            return "left"
        elif is_right_up and not is_left_up:
            return "right"
        else:
            return "None"

if __name__ == "__main__":
    import cv2
    import os
    
    def process_and_save(img_path):
        result = detect_raised_arm(img_path)
        print(f"{img_path} -> {result}")
        if result != "None":
            # Gorseli oku, uzerine sonucu yaz ve kaydet
            image = cv2.imread(img_path)
            if image is not None:
                # Yaziyi daha kucuk ve sade yapalim ki ekrana sigsin
                cv2.putText(image, result.upper(), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                filename = os.path.basename(img_path)
                cv2.imwrite(f"output_{filename}", image)

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        process_and_save(img_path)
    else:
        images = ['../pose-1.jpg', '../pose-2.jpg', '../pose-3.jpg']
        for img in images:
            process_and_save(img)
