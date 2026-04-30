from cog import BasePredictor, Input, Path
import urllib.request
import os
import predict_face

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model_path = "face_landmarker.task"
        if not os.path.exists(self.model_path):
            print("Indiriliyor: face_landmarker.task...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
            
    def predict(
        self,
        image: Path = Input(description="Input image for face direction estimation"),
    ) -> str:
        """Run a single prediction on the model"""
        result = predict_face.detect_face_direction(str(image), model_path=self.model_path)
        return result
