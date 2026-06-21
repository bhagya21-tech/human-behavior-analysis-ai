import streamlit as st 
import cv2 
import numpy as np 
import pandas as pd
import plotly.express as px

from app.emotion_detector import EmotionDetector 
from app.posture_detector import PostureDetector 
from app.attention_score import AttentionScore 
from app.behavior_analyzer import BehaviorAnalyzer 
from app.video_analyzer import VideoAnalyzer
from collections import Counter

st.set_page_config(
    page_title="AI Ineterview Analyzer",
    layout="wide"
)

st.title("🧠 AI Interview & Human Behavior Analyzer")

emotion_detector = EmotionDetector()
posture_detector = PostureDetector()

attention_engine = AttentionScore()
behavior_engine = BehaviorAnalyzer()



video_file = st.file_uploader(
    "Upload Interview Video",
    type=["mp4", "avi", "mov"]
)


if video_file:

    with open(
        "uploads/temp_video.mp4",
        "wb"
    ) as f:

        f.write(
            video_file.read()

        )

    st.info(
        "Analyzing video..."
    )

    video_engine = VideoAnalyzer(
        emotion_detector,
        posture_detector
    )

    results = video_engine.analyze_video(
        "uploads/temp_video.mp4"
    )

    if len(results["emotions"]) > 0:
        dominant_emotion = (
            Counter(
                results["emotions"]

            ).most_common(1)[0][0]
        )

        avg_attention = (
            sum(results["attentions"])
            /
            len(results["attentions"])
        )

        dominant_posture = (
            Counter(
                results["postures"]
            
            ).most_common(1)[0][0]
        )

        st.success(
            "Video Analysis Complete"
        )

        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Dominant Emotion",
                dominant_emotion
            )

        with col2:
            st.metric(
                "Average Attention",
                f"{avg_attention:.1f}%"
            )

        with col3:
            st.metric(
                "Dominant Posture",
                dominant_posture
            )

        timeline_df = pd.DataFrame({

            "Frame": results["timestamps"],

            "Attention": results["attentions"],

            "Confidence": results["confidences"]

        })

        fig1 = px.line(
            timeline_df,

            x="Frame",

            y="Attention",

            title="Attention Timeline"

        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        fig2 = px.line(
            timeline_df,

            x="Frame",

            y="Confidence",

            title="Confidence Timeline"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        emotion_df = pd.DataFrame({

            "Emotion":
            results["emotions"]
        })

        fig3 = px.histogram(

            emotion_df,

            x="Emotion",

            title="Emotion Distribution"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )


        

        

st.divider()

uploaded_file = st.file_uploader(
    "Upload Interview Image",
    type=["jpg", "jpeg", "png"]

)

if uploaded_file:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR 
    )


    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Uploaded Image"
    )

   

    emotion_results = emotion_detector.detect_emotion(
        image

    )

    _, posture = posture_detector.detect_posture(
        image.copy()
    )

    if len(emotion_results) > 0:

        emotion = emotion_results[0]["emotion"]

        confidence = emotion_results[0]["confidence"]

    else:
        emotion = "Unknown"

        confidence = 0

    attention_score = attention_engine.calculate(
        emotion,
        posture
    )
     
    behavior = behavior_engine.analyze(
        emotion,
        posture,
        attention_score
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Emotion",
            emotion
        )

        st.metric(
            "Confidence",
            f"{confidence}%"
        )

    with col2:

        st.metric(
            "Posture",
            posture
        )

        st.metric(
            "Attention Score",
            f"{attention_score}%"
        )

    st.success(
        f"Behavior Assessment: {behavior}"
    )
    