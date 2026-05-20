import cv2
import torch 
import numpy as np 
from app.emotion_model import SimpleEmotionModel 



class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 
            'haarcascade_frontalface_default.xml'
            
        )

        self.model = SimpleEmotionModel()

        self.model.load_state_dict(
            torch.load(
                "models/emotion_model/emotion_model.pth",
                map_location=torch.device("cpu")
                
            )
        )
        self.model.eval()


        self.emotions = [
            "Angry", "Disgust", "Fear","Contempt"
            "Happiness", "Sadness", "Surprise", "Neutral"
        ]

    def preprocess(self, face):
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.astype(np.float32) / 255.0
    
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=0)

        return torch.tensor(face)



    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )
        
        results = []
    
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            input_tensor = self.preprocess(face)
            
            with torch.no_grad():
                output = self.model(input_tensor)

                probabilities = torch.softmax(output, dim=1)

                confidence, prediction = torch.max(
                    probabilities,
                    dim=1
                )

                prediction = prediction.item()

                confidence = confidence.item()



            emotion = self.emotions[prediction] 


            results.append({
                "box": (x, y, w, h),
                "emotion": emotion,
                "confidence": round(confidence * 100, 2) # placeholder
            
            })
            
        return results
        
    