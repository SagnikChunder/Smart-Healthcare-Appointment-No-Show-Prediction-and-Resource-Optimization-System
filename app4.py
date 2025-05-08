import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("model.pkl")

# App Title
st.title("Smart Healthcare: No-Show Appointment Predictor")

# Sidebar Inputs
st.sidebar.header("Patient Details")

age = st.sidebar.slider("Age", 0, 100, 30)
waiting_days = st.sidebar.slider("Waiting Days", 0, 100, 10)
alcoholism = st.sidebar.selectbox("Alcoholism", [0, 1])
sms_received = st.sidebar.selectbox("SMS Received", [0, 1])
sch_hour = st.sidebar.slider("Scheduled Hour", 0, 23, 10)

# Construct input DataFrame (must match model training columns)
input_data = pd.DataFrame({
    "Age": [age],
    "WaitingDays": [waiting_days],
    "Alcoholism": [alcoholism],
    "SMS_received": [sms_received],
    "ScheduledHour": [sch_hour]
})

# Show input
st.subheader("Patient Info:")
st.write(input_data)

# Make prediction
if st.button("Predict No-Show"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"❌ Likely to NO-SHOW (Probability: {prediction_proba:.2f})")
    else:
        st.success(f"✅ Likely to ATTEND (Probability: {prediction_proba:.2f})")

    # Plot probability
    fig, ax = plt.subplots()
    ax.bar(["Show", "No-Show"], model.predict_proba(input_data)[0], color=["green", "red"])
    ax.set_title("Prediction Probability")
    st.pyplot(fig)
