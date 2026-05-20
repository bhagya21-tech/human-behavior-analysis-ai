# Human Behavior Analysis AI

A real-time multimodal AI system that combines facial emotion recognition, posture analysis, temporal smoothing, and behavioral fusion to understand human behavioral states using computer vision and deep learning.

---

# Project Overview

This project analyzes:

* Facial emotions
* Human posture
* Behavioral state
* Temporal emotion consistency
* Real-time posture trends
* Session analytics

The system combines multiple computer vision modules into a unified behavioral intelligence pipeline.

---

# Features

## Real-Time Emotion Detection

* CNN-based emotion recognition
* FER2013 dataset training
* Confidence score prediction
* Temporal emotion smoothing

## Posture Detection

* Full-body landmark detection using MediaPipe
* Straight vs Slouched posture analysis
* Real-time skeletal tracking

## Behavioral Fusion Engine

Combines:

* Emotion
* Posture
* Temporal consistency

Predicts:

* Attentive
* Disengaged
* Confident
* Low Confidence
* Neutral behavioral states

## Temporal Stabilization

* Reduces prediction flickering
* Rolling buffer smoothing
* Stable real-time inference

## Behavior Analytics Dashboard

* Emotion frequency graphs
* Posture statistics
* Behavioral state analytics
* Session-based insights

---

# Tech Stack

## Computer Vision

* OpenCV
* MediaPipe

## Deep Learning

* PyTorch
* TorchVision

## Data & Analytics

* Pandas
* Matplotlib
* NumPy

---

# Project Architecture

```text
Webcam Input
      ↓
Face Detection + Pose Detection
      ↓
Emotion CNN + Posture Analysis
      ↓
Temporal Smoothing
      ↓
Fusion Engine
      ↓
Behavior Prediction
      ↓
CSV Logging
      ↓
Analytics Dashboard
```

---

# Folder Structure

```text
emotion_posture_fusion/
│
├── app/
│   ├── main.py
│   ├── camera.py
│   ├── emotion_detector.py
│   ├── emotion_model.py
│   ├── posture_detector.py
│   ├── fusion_engine.py
│   ├── emotion_tracker.py
│   ├── state_tracker.py
│   ├── logger.py
│
├── data/
│   ├── fer2013/
│   ├── behavior_log.csv
│
├── models/
│   ├── emotion_model/
│       ├── emotion_model.pth
│
├── analytics_dashboard.py
├── train_emotion_model.py
├── requirements.txt
├── README.md
```

---

# Dataset

## FER2013 Dataset

Used for training facial emotion recognition model.

Classes:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

Dataset Source:
[https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

---

# Installation

## Clone Repository

```bash
git clone https://github.com/bhagya21-tech/human-behavior-analysis-ai.git
cd human-behavior-analysis-ai
```

## Create Virtual Environment

```bash
python -m venv venv310
```

## Activate Environment

### Windows

```bash
venv310\Scripts\activate
```

### Linux/Mac

```bash
source venv310/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Train Emotion Model

```bash
python train_emotion_model.py
```

---

# Run Main Application

```bash
python -m app.main
```

---

# Run Analytics Dashboard

```bash
python analytics_dashboard.py
```

---

# Example Outputs

## Real-Time Detection

* Emotion prediction
* Confidence score
* Skeleton tracking
* Behavioral state analysis

## Analytics Dashboard

* Emotion distribution graphs
* Posture pie charts
* Behavioral state statistics

---

# Key AI Concepts Implemented

* Convolutional Neural Networks (CNN)
* Emotion Recognition
* Pose Estimation
* Temporal Smoothing
* Real-Time Inference
* Human Behavior Analysis
* Multimodal AI Fusion
* Computer Vision Pipelines
* Behavioral Analytics

---

# Future Improvements

* Real-time Streamlit dashboard
* Attention tracking
* Fatigue detection
* Interview performance analysis
* Head pose estimation
* Eye gaze tracking
* FastAPI deployment
* Docker containerization
* Cloud deployment
* Transformer-based emotion recognition

---

# Resume Description

Built a multimodal human behavior analysis system combining CNN-based facial emotion recognition, posture estimation, temporal smoothing, and behavioral fusion to infer real-time human behavioral states using computer vision and deep learning.

---

# Skills Demonstrated

* AI/ML Engineering
* Computer Vision
* Deep Learning
* PyTorch
* Real-Time Video Processing
* Model Training & Evaluation
* Behavioral Analytics
* Data Visualization
* Software Engineering
* Debugging & Environment Management

---

# Author

Bhagyashri Raut

---

# License

This project is licensed under the MIT License.
