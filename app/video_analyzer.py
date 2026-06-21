import cv2 

from app.attention_score import AttentionScore 
from app.behavior_analyzer import BehaviorAnalyzer

class VideoAnalyzer:

    def __init__(
        self,
        emotion_detector,
        posture_detector
    ): 

        self.emotion_detector = emotion_detector
        self.posture_detector = posture_detector 

        self.attention_engine = AttentionScore()
        self.behavior_engine = BehaviorAnalyzer()

    def analyze_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        emotions = []
        attentions = []
        postures = []
        confidences = []
        timestamps = []



        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break 

            frame_count += 1

            # Analyze every 10th frame 
            if frame_count % 10 != 0:
                continue

            emotion_results = (
                self.emotion_detector
                .detect_emotion(frame)

            )

            _, posture = (
                self.posture_detector
                .detect_posture(frame)

            )

            if len(emotion_results) > 0:

                emotion = emotion_results[0]["emotion"]

                attention = (
                    self.attention_engine
                    .calculate(
                        emotion,
                        posture
                    )
                )

                emotions.append(emotion)
                attentions.append(attention)
                postures.append(posture)

                confidence = emotion_results[0]["confidence"]

                confidences.append(confidence)

                timestamps.append(frame_count)

        cap.release()

        return {
            "emotions": emotions,
            "attentions": attentions,
            "postures": postures,
            "confidences": confidences,
            "timestamps": timestamps
        }