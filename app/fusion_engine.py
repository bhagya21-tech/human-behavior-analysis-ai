class FusionEngine:  
    def __init__(self):
        pass 

    def analyze(self, emotion_result, posture_status):
        emotion = "unknown"

        if emotion_result and len(emotion_result) > 0:
            emotion = emotion_result[0]["emotion"]

        # Basic fusion rules
        if emotion in ["sad", "fear", "angry"] and posture_status == "Slouched":
            state = "Low Confidence"

        elif emotion == "happy" and posture_status == "Straight":
            state = "Confident" 

        elif posture_status == "Slouched":
            state = "Disengaged"
        
        elif posture_status == "Straight":
            state = "Attentive"

        else:
            state = "Neutral"
        
        return state
        