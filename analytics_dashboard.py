import pandas as pd 
import matplotlib.pyplot as plt 

# Load CSV 
df = pd.read_csv("data/behavior_log.csv")

# ----------------------
# Emotion Frequency
# ----------------------

emotion_count = df["emotion"].value_counts()

plt.figure(figsize=(8, 5))

emotion_counts.plot(kind="bar")

plt.title("Emotion Frequency")

plt.xlabel("Emotion")
plt.ylabel("Count")

plt.tight_layout()

plt.show()

# -----------------------
# Posture Statistics 
# -----------------------
posture_counts = df["posture"].value_counts()

plt.figure(figsize=(6, 5))

posture_counts.plot(kind="pie", autopct="%1.1f%%")

plt.title("Posture Distribution")

plt.ylabel("")

plt.tight_layout()

plt.show()

# -------------------------
# Behavior State Analysis 
# -------------------------
state_counts = df["state"].value_counts()

plt.figure(figsize=(8, 5))

state_counts.plot(kind="bar")

plt.title("Behavior States") 

plt.xlabel("State")
plt.ylabel("Frequency")

plt.tight_layout()

plt.show()