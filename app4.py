import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Load model
model = joblib.load("xgb_model.pkl")

# Load background data for explainer (use a small sample of training data)
@st.cache_data
def load_background_data():
    return pd.read_csv("background_sample.csv")  # A small sample of training data

background_data = load_background_data()

st.title("🩺 Smart Healthcare: Appointment No-Show Prediction")
st.markdown("This app predicts whether a patient will miss their appointment, with explainability using LIME.")

# Input fields
gender = st.selectbox("Gender", ["F", "M"])
age = st.slider("Age", 0, 115, 30)
scholarship = st.selectbox("Scholarship", [0, 1])
hypertension = st.selectbox("Hypertension", [0, 1])
diabetes = st.selectbox("Diabetes", [0, 1])
alcoholism = st.selectbox("Alcoholism", [0, 1])
sms_received = st.selectbox("SMS Received", [0, 1])
waiting_days = st.slider("Days between Scheduling and Appointment", -30, 60, 0)

# Prepare input
input_data = pd.DataFrame([{
    "Gender": 1 if gender == "F" else 0,
    "Age": age,
    "Scholarship": scholarship,
    "Hypertension": hypertension,
    "Diabetes": diabetes,
    "Alcoholism": alcoholism,
    "SMS_received": sms_received,
    "waiting_days": waiting_days
}])

# Predict and explain
if st.button("Predict"):
    proba = model.predict_proba(input_data)[0][1]
    st.success(f"🔮 Probability of No-Show: {proba:.2%}")
    st.progress(proba)

    st.subheader("LIME Explanation for this Prediction")

    explainer = LimeTabularExplainer(
        training_data=np.array(background_data),
        feature_names=background_data.columns,
        class_names=['Show', 'No-Show'],
        mode='classification'
    )

    explanation = explainer.explain_instance(
        data_row=input_data.iloc[0].values,
        predict_fn=model.predict_proba
    )

    fig = explanation.as_pyplot_figure()
    st.pyplot(fig)
