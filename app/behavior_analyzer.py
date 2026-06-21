class BehaviorAnalyzer:

    def analyze(self, emotion, posture, attention):
        
        if attention >= 80:
            return "Highly Engaged"

        elif attention >= 60:
            return "Engaged"

        elif attention >= 40:
            return "Moderately Focused"

        else: 
            return "Disengaged"