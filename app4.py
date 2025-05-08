import streamlit as st
import pandas as pd
import numpy as np
pip install joblib
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("model.pkl")

# Title
st.title("🧠 Smart Healthcare: No-Show Appointment Predictor")

st.markdown("Predict whether a patient will show up for a medical appointment based on the input features.")

st.subheader("📋 Input Patient Information")

# Input form
age = st.slider("Age", 0, 100, 35)
gender = st.selectbox("Gender", ["Female", "Male"])
scholarship = st.selectbox("Scholarship", ["No", "Yes"])
hipertension = st.selectbox("Hipertension", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
alcoholism = st.selectbox("Alcoholism", ["No", "Yes"])
handcap = st.slider("Handcap Level", 0, 4, 0)
sms_received = st.selectbox("SMS Received", ["No", "Yes"])
weekday = st.selectbox("Appointment Weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
scheduled_hour = st.slider("Scheduled Hour (24hr)", 0, 23, 14)
waiting_days = st.slider("Waiting Days", 0, 60, 5)

# Encode categorical variables
gender = 1 if gender == "Male" else 0
scholarship = 1 if scholarship == "Yes" else 0
hipertension = 1 if hipertension == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
alcoholism = 1 if alcoholism == "Yes" else 0
sms_received = 1 if sms_received == "Yes" else 0
weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
               "Friday": 4, "Saturday": 5, "Sunday": 6}
appointment_weekday = weekday_map[weekday]

# Create input dataframe
input_data = {
    'Age': age,
    'Scholarship': scholarship,
    'Hipertension': hipertension,
    'Diabetes': diabetes,
    'SMS_received': sms_received,
    'Alcoholism': alcoholism,
    'AppointmentWeekDay': appointment_weekday,
    'Gender': gender,
    'Handcap': handcap,
    'ScheduledHour': scheduled_hour,
    'WaitingDays': waiting_days
}
feature_list = list(input_data.keys())
X = pd.DataFrame([input_data])

# Predict
if st.button("Predict No-Show"):
    prediction = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    st.subheader("📢 Prediction Result")
    st.write(f"### {'❌ No-Show' if prediction == 1 else '✅ Show'}")
    st.write(f"**Probability of No-Show:** `{prob:.2f}`")

    # Probability bar chart
    fig, ax = plt.subplots()
    ax.bar(["Show", "No-Show"], [1 - prob, prob], color=["green", "red"])
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probability")
    st.pyplot(fig)

    # Display input summary
    st.subheader("📊 Input Summary")
    st.dataframe(X.T.rename(columns={0: "Value"}))
