class AttentionScore:

    def calculate(self, emotion, posture):

        score = 50

        if posture == "Straight":
            score += 30

        if emotion in ["Neutral", "Happy"]:
            score += 20

        if emotion in ["Sad", "Angry"]:
            score -= 10

        return max(0, min(100, score))