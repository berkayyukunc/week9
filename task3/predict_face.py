import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def detect_face_direction(image_path, model_path='../models/face_landmarker.task'):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    
    with vision.FaceLandmarker.create_from_options(options) as detector:
        image = mp.Image.create_from_file(image_path)
        result = detector.detect(image)
        
        if not result.face_landmarks:
            return "None"
            
        landmarks = result.face_landmarks[0]
        
        # Sadece X koordinatlarina bakalim
        nose_tip_x = landmarks[1].x
        right_cheek_x = landmarks[234].x # Kameranin solu (kisinin sagi)
        left_cheek_x = landmarks[454].x  # Kameranin sagi (kisinin solu)
        
        # X eksenindeki mesafeler
        dist_to_right = abs(nose_tip_x - right_cheek_x)
        dist_to_left = abs(nose_tip_x - left_cheek_x)
        
        # Hata ayiklama icin
        # print(f"{image_path} -> dist_left: {dist_to_left:.3f}, dist_right: {dist_to_right:.3f}")
        
        ratio = dist_to_left / (dist_to_right + 1e-6)
        
        if ratio > 1.5:
            # Burun sol yanaga daha uzak, sag yanaga (kameranin soluna) daha yakin
            # Yani yuz sola donuk (kameranin soluna, yuzun kendi sagina)
            # face-1 -> left bekleniyor
            return "left"
        elif ratio < 0.6:
            # Burun sol yanaga daha yakin -> yuz saga donuk (kameranin sagina)
            # face-2 -> right bekleniyor
            return "right"
        else:
            return "straight"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"{img_path} -> {detect_face_direction(img_path)}")
    else:
        images = ['../face-1.png', '../face-2.png', '../face-3.png']
        for img in images:
            try:
                res = detect_face_direction(img)
                print(f"{img}: {res}")
            except Exception as e:
                print(f"Error processing {img}: {e}")
