from collections import deque
from statistics import mode

class EmotionTracker:

    def __init__(self, buffer_size=15):

        self.emotion_history = deque(
            maxlen=buffer_size
        )

    def smooth_emotion(self, emotion):

        self.emotion_history.append(emotion)

        try:
            stable_emotion = mode(
                self.emotion_history
            )

        except:
            stable_emotion = emotion


        return stable_emotion