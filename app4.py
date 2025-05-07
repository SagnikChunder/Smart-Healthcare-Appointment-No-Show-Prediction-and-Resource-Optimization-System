import streamlit as st
import pandas as pd
import numpy as np
import os
os.system("pip install shap")
import shap
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

# Title and description
st.set_page_config(page_title="Healthcare No-Show Prediction", layout="wide")
st.title("Smart Healthcare: Appointment No-Show Prediction")
st.markdown("Predict patient no-shows and optimize resources using ML and SHAP explanations.")

# Load model and explainer
@st.cache_resource
def load_model_and_explainer():
    model = joblib.load("xgb_model.pkl")
    explainer = shap.Explainer(model)
    return model, explainer

try:
    model, explainer = load_model_and_explainer()
except Exception as e:
    st.error(f"Failed to load model or explainer: {e}")
    st.stop()

# Input sidebar
st.sidebar.header("Patient Appointment Features")

def user_input_features():
    age = st.sidebar.slider('Age', 0, 100, 30)
    hypertension = st.sidebar.selectbox('Hypertension', [0, 1])
    diabetes = st.sidebar.selectbox('Diabetes', [0, 1])
    alcohol = st.sidebar.selectbox('Alcoholism', [0, 1])
    sms_received = st.sidebar.selectbox('SMS Received', [0, 1])
    scholarship = st.sidebar.selectbox('Scholarship', [0, 1])
    gender = st.sidebar.selectbox('Gender', ['F', 'M'])
    handcap = st.sidebar.slider('Handicap Level', 0, 4, 0)
    waiting_days = st.sidebar.slider('Waiting Days', 0, 100, 10)

    # Encode gender
    gender_F = 1 if gender == 'F' else 0
    gender_M = 1 if gender == 'M' else 0

    data = {
        'Age': age,
        'Hypertension': hypertension,
        'Diabetes': diabetes,
        'Alcoholism': alcohol,
        'SMS_received': sms_received,
        'Scholarship': scholarship,
        'Handcap': handcap,
        'Waiting_Days': waiting_days,
        'Gender_F': gender_F,
        'Gender_M': gender_M,
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Predict button
if st.button("Predict No-Show"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Results")
    st.write(f"🔍 **No-show Probability:** `{probability:.2%}`")
    st.write(f"📌 **Prediction:** {'No-Show' if prediction == 1 else 'Show-Up'}")

    # SHAP explanation
    st.subheader("Explainability with SHAP")
    try:
        shap_values = explainer(input_df)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.warning(f"SHAP explanation error: {e}")

# Optional: Data upload & batch prediction
st.markdown("---")
st.subheader("📄 Upload CSV for Batch Predictions")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)
        predictions = model.predict(batch_df)
        probabilities = model.predict_proba(batch_df)[:, 1]
        batch_df["No_Show_Prob"] = probabilities
        batch_df["Prediction"] = ["No-Show" if p == 1 else "Show-Up" for p in predictions]

        st.write("📊 Results:")
        st.dataframe(batch_df.head())

        st.download_button("Download Results", batch_df.to_csv(index=False), "batch_predictions.csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")