import gradio as gr
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("xgb_model.pkl")

# Maps for encoding
gender_map = {"M": 0, "F": 1}
weekday_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5
}

# Prediction function
def predict_show(Gender, Age, Hipertension, Diabetes, Alcoholism,
                 Scholarship, Handcap, SMS_received,
                 WaitingDays, ScheduledHour, AppointmentWeekDay):
    try:
        gender_encoded = gender_map.get(Gender)
        weekday_encoded = weekday_map.get(AppointmentWeekDay)

        # Input validation
        if gender_encoded is None or weekday_encoded is None:
            return "🚫 Error: Invalid categorical input."

        # Prepare input data with exact feature names and order
        input_data = pd.DataFrame([[
            gender_encoded, Age, Hipertension, Diabetes, Alcoholism,
            Scholarship, Handcap, SMS_received, WaitingDays,
            ScheduledHour, weekday_encoded
        ]], columns=[
            'Gender', 'Age', 'Hipertension', 'Diabetes', 'Alcoholism',
            'Scholarship', 'Handcap', 'SMS_received', 'WaitingDays',
            'ScheduledHour', 'AppointmentWeekDay'
        ])

        # Make prediction
        prediction = model.predict(input_data)[0]
        return "🔍 Prediction Result: Will No-Show 😐" if prediction == 1 else "🔍 Prediction Result: Will Attend ✅"

    except Exception as e:
        return f"⚠️ Error during prediction: {str(e)}"

# Gradio Interface
iface = gr.Interface(
    fn=predict_show,
    inputs=[
        gr.Dropdown(["M", "F"], label="Gender"),
        gr.Slider(0, 100, step=1, label="Age"),
        gr.Radio([0, 1], label="Hipertension"),
        gr.Radio([0, 1], label="Diabetes"),
        gr.Radio([0, 1], label="Alcoholism"),
        gr.Radio([0, 1], label="Scholarship"),
        gr.Radio([0, 1], label="Handcap"),
        gr.Radio([0, 1], label="SMS Received"),
        gr.Slider(-10, 60, step=1, label="Waiting Days"),
        gr.Slider(0, 23, step=1, label="Scheduled Hour"),
        gr.Dropdown(list(weekday_map.keys()), label="Appointment Weekday")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Smart Healthcare: Appointment No-Show Prediction",
    description="Enter patient details to predict whether they will attend their medical appointment."
)

iface.launch()