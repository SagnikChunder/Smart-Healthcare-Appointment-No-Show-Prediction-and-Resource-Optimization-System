import gradio as gr
import joblib
import numpy as np

model = joblib.load("model.pkl")

def predict(age, waiting_days, alcoholism, sms_received, sch_hour):
    X = np.array([[age, waiting_days, alcoholism, sms_received, sch_hour]])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    return f"{'No-Show' if prediction == 1 else 'Show'} (Probability: {proba:.2f})"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(0, 100, label="Age"),
        gr.Slider(0, 100, label="Waiting Days"),
        gr.Radio([0, 1], label="Alcoholism"),
        gr.Radio([0, 1], label="SMS Received"),
        gr.Slider(0, 23, label="Scheduled Hour"),
    ],
    outputs="text",
    title="No-Show Appointment Predictor"
)

demo.launch()
