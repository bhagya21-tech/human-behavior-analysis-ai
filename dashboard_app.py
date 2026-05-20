import streamlit as st
import pandas as pd 
import plotly.express as px
import time
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Human Behavior AI Dashboard",
    layout="wide"
)

st.title("🧠 Human Behavior AI Dashboard")

st_autorefresh(interval=2000, key="dashboard_refresh")

# Auto - refresh
try:
    df = pd.read_csv("data/behavior_log.csv")

    # ===========================
    # Metrics
    # ===========================
except Exception as e:

    st.error(f"CSV Error: {e}")

    st.stop()

# Empty CSV Check


if len(df) == 0:

    st.warning("No data found yet.")

    st.stop()

# Latest Metrics 

latest = df.iloc[-1]



col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Current Emotion",
    latest["emotion"]
)

col2.metric(
    "Confidence",
    f"{latest['confidence']}%"
)

col3.metric(
    "Posture",
    latest["posture"]
)

col4.metric(
    "Behavior State",
    latest["state"]
)

st.divider()

# =====================
# Emotion Frequency
# =====================

emotion_counts = (
    df["emotion"]
    .value_counts()
    .reset_index()
)

emotion_counts.columns = [
    "Emotion", 
    "Count"
]

fig1 = px.bar(
    emotion_counts,
    x="Emotion",
    y="Count",
    title="Emotion frequency"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# =============================
# Posture Distribution
# =============================
posture_counts = (
    df["posture"]
    .value_counts()
    .reset_index()
)

posture_counts.columns = [
    "Posture",
    "Count"
]

fig2 = px.pie(
    posture_counts,
    names="Posture",
    values="Count",
    title="Posture Distribution"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ====================================
# Behavior Timeline
# ====================================

timeline_df = df.tail(30)

fig3 = px.line(
    timeline_df,
    y="confidence",
    title="Confidence Timeline"
)

st.plotly_chart(
    fig3,
    use_container_width=True,
)

# ==================================
# Raw Data 
# ==================================
st.subheader("Live Session Data")

st.dataframe(
    df.tail(20),
    use_container_width=True
)



    
