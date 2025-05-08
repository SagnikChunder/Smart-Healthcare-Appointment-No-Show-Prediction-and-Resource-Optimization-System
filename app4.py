import gradio as gr
import pandas as pd
import joblib

# Load model
model = joblib.load("xgb_model.pkl")

# Prediction function
def predict_show(gender, age, hypertension, diabetes, sms_received, waiting_days):
    data = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Hypertension": [hypertension],
        "Diabetes": [diabetes],
        "SMS_received": [sms_received],
        "WaitingDays": [waiting_days]
    })
    
    prediction = model.predict(data)[0]
    return "Will No-Show 😐" if prediction == 1 else "Will Attend ✅"

# UI Interface
iface = gr.Interface(
    fn=predict_show,
    inputs=[
        gr.Dropdown(["M", "F"], label="Gender"),
        gr.Slider(0, 100, step=1, label="Age"),
        gr.Radio([0, 1], label="Hypertension (0=No, 1=Yes)"),
        gr.Radio([0, 1], label="Diabetes (0=No, 1=Yes)"),
        gr.Radio([0, 1], label="SMS Received (0=No, 1=Yes)"),
        gr.Slider(-10, 60, step=1, label="Waiting Days")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Smart Healthcare: No-Show Prediction",
    description="Enter patient details to predict whether they will attend their appointment.",
)

iface.launch()
