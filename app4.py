import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib

# Load trained model (replace with your actual path)
model = joblib.load('xgb_model.pkl')

st.set_page_config(page_title="Smart Healthcare Dashboard", layout="wide")
st.title("🧠 Smart Healthcare: No-Show Prediction & Resource Optimization")

# Upload patient data
st.sidebar.header("Upload Appointment Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Patient Appointments")
    st.dataframe(df)

    # Predict no-show probability
    X = df.drop(columns=['No-show'], errors='ignore')  # drop label if present
    probs = model.predict_proba(X)[:, 1]
    df['No-show Probability'] = probs

    st.subheader("No-Show Predictions")
    st.dataframe(df[['No-show Probability']])

    # Show SHAP Explanation for first row
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    st.subheader("SHAP Explanation (First Record)")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches='tight')

    # Overbooking threshold slider
    st.sidebar.header("Overbooking Simulation")
    threshold = st.sidebar.slider("No-show probability threshold", 0.5, 0.95, 0.7)

    df['Overbook'] = df['No-show Probability'] > threshold
    overbooked_df = df[df['Overbook']]

    st.subheader("Overbooked Appointments")
    st.dataframe(overbooked_df)

    # Cost simulation
    y_true = df.get('No-show', np.random.binomial(1, probs))  # fallback mock
    cost = 0
    for p, y in zip(probs, y_true):
        if p > threshold:
            cost += 0 if y == 1 else -300
        else:
            if y == 1:
                cost += 500

    st.metric("Total Simulated Cost (₹)", f"{cost:,}")

    # Plot: Cost vs Threshold
    thresholds = np.arange(0.5, 0.95, 0.05)
    costs = []
    for t in thresholds:
        c = 0
        for p, y in zip(probs, y_true):
            if p > t:
                c += 0 if y == 1 else -300
            else:
                if y == 1:
                    c += 500
        costs.append(c)

    st.subheader("Cost vs Threshold Optimization")
    fig, ax = plt.subplots()
    ax.plot(thresholds, costs, marker='o')
    ax.set_xlabel("Overbooking Threshold")
    ax.set_ylabel("Simulated Cost (₹)")
    ax.set_title("Optimize Overbooking")
    st.pyplot(fig)
