import cv2
from app.camera import CameraStream 
from app.posture_detector import PostureDetector 

camera = CameraStream()
posture_detector = PostureDetector()

while True:
    frame = camera.get_frame()
    
    if frame is None:
        break 
    
    frame, posture = posture_detector.detect_posture(frame)
    
    cv2.putText(
        frame,
        f"Posture: {posture}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0),
        2
        
    )
    
    cv2.imshow("Posture Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
    
camera.release()