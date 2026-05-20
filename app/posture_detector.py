import cv2 
import mediapipe as mp 
import math 

class PostureDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose 
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils 
        
    def calculate_angle(self, p1, p2):
        return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    
    def detect_posture(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_frame)
        
        posture_status = "Unknown" 
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark 
            
            # Get key points 
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Convert top pixel coordinates 
            h, w, _ = frame.shape 
            l_shoulder = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            
            r_shoulder = (int(right_shoulder.x * w), int(right_shoulder.y * h))
            
            # Draw skeleton
            self.mp_draw.draw_landmarks(
                frame,
                result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Calculate shoulder angle 
            angle = abs(self.calculate_angle(l_shoulder, r_shoulder))
            
            # Simple posture logic 
            if angle < 10:
                posture_status = "Straight" 
            else:
                posture_status = "Slouched"
        
        return frame, posture_status
          