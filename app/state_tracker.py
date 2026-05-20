from collections import deque
from statistics import mode

class StateTracker:
    def __init__(self, max_len=20):
        self.history = deque(maxlen=max_len)

    def update(self, state):
        self.history.append(state)

        try:
            return mode(self.history)

        except:
            return state 
            