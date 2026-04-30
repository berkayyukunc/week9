from cog import BasePredictor, Input, Path
import urllib.request
import os
import predict_arm

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model_path = "pose_landmarker.task"
        if not os.path.exists(self.model_path):
            print("Indiriliyor: pose_landmarker.task...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
            urllib.request.urlretrieve(url, self.model_path)
            
    def predict(
        self,
        image: Path = Input(description="Input image for pose estimation"),
    ) -> str:
        """Run a single prediction on the model"""
        result = predict_arm.detect_raised_arm(str(image), model_path=self.model_path)
        return result
