import sys
import cv2 

from app.camera import CameraStream
from app.emotion_detector import EmotionDetector 
from app.posture_detector import PostureDetector 
from app.fusion_engine import FusionEngine 
from app.state_tracker import StateTracker 
from app.logger import BehaviorLogger
from app.emotion_tracker import EmotionTracker 

print("Starting application...")


# Initialize Components 

camera = CameraStream() 
print("Camera initialized")

print("Loading emotion detector...")
emotion_detector = EmotionDetector()
print("Emotion detector loaded")

print("Loading posture detector...")
posture_detector = PostureDetector()
print("Posture detector loaded")

fusion_engine = FusionEngine()
tracker = StateTracker()
logger = BehaviorLogger()
emotion_tracker = EmotionTracker()

print("Entering main loop...")

import time
time.sleep(2)

prev_time = 0
frame_count = 0

while True:

    try:

        # Get Frame

        frame = camera.get_frame()



        if frame is None:
            print("Skipping bad frame")
            continue

        
        # Emotion detection 

        emotion_results = []

        try:
            emotion_results = emotion_detector.detect_emotion(
                frame
            )

        except Exception as e:

            print("Emotion detection error:", e)


        # Posture detection 

        posture = "Unknown"

        try:
            frame, posture = posture_detector.detect_posture(
                frame
            ) 

        except Exception as e:

            print("Posture detection error:", e)



        # Draw emotion

        stable_emotion = "Unknown" 

        for result in emotion_results:

            x, y, w, h = result["box"]

            
            stable_emotion = emotion_tracker.smooth_emotion(
                result["emotion"]
            )
            label = f"{result['emotion']} ({result['confidence']}%)"

            # Face Box

            cv2.rectangle(
                frame, 
                (x, y), 
                (x+w, y+h), 
                (255, 0, 0), 
                2)
            

            # Emotion Text 

            cv2.putText(
               frame,
               label,
               (x, y - 10),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.7,
               (255, 0, 0),
                2
            )
        
        # Fusion 

        try:
            raw_state = fusion_engine.analyze(
                emotion_results, 
                posture
            )
            stable_state = tracker.update(
                raw_state
            ) 

        except: 

            stable_state = "Unknown"



        # Logging

        try:
            if len(emotion_results) > 0:
                logger.log(
                    stable_state,
                    emotion_results[0]["confidence"],
                    posture,
                    stable_emotion
                )
        except Exception as e:

            print("Logging error:", e)

        # Draw posture + final state 
        cv2.putText(
            frame,
            f"Posture: {posture}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # State Text


        cv2.putText(
            frame,
            f"State: {stable_state}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        
        # Show Frame


        cv2.imshow("Human Behavior Analysis", frame)

        # Exit Key 

        key = cv2.waitKey(1)

        if key == ord('q'):

            print("Exiting application...")
            break 


    except Exception as e:

        print("Main loop error:", e)