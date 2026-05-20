import csv 
import os
from datetime import datetime 

class BehaviorLogger:

    def __init__(self, file_name="data/behavior_log.csv"):
        self.file_name = file_name 

        # Create file with headers if not exists
        if not os.path.exists(self.file_name):
            with open(self.file_name, "w", newline="") as file:
                writer = csv.writer(file)

                writer.writerow([
                    "timestamp",
                    "emotion",
                    "confidence",
                    "posture",
                    "state"
                ])



    def log(self,
            state, 
            confidence,
            posture,
            emotion):

        with open(self.file_name, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now(),
                emotion,
                confidence,
                state, 
                posture
            ])

            