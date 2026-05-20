import cv2
from app.camera import CameraStream 
from app.emotion_detector import EmotionDetector 

camera = CameraStream()
emotion_detector = EmotionDetector()

while True:
    frame = camera.get_frame()
    
    if frame is None:
        break 
    
    results = emotion_detector.detect_emotion(frame)
    
    for result in results:
        x, y, w, h = result["box"]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        
        cv2.putText(
            frame,
            result["emotion"],
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
    cv2.imshow("Emotion Detection (Base)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
camera.release()
