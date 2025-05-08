import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import joblib
import numpy as np

# Load model
model = joblib.load("xgb_model.pkl")

# Init app
app = dash.Dash(__name__)
app.title = "No-Show Appointment Predictor"

app.layout = html.Div([
    html.H1("Smart Healthcare: No-Show Prediction", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Age"),
        dcc.Slider(id='age', min=0, max=100, step=1, value=35),
        
        html.Label("Waiting Days"),
        dcc.Slider(id='waiting_days', min=0, max=100, step=1, value=10),
        
        html.Label("Alcoholism"),
        dcc.RadioItems(id='alcoholism', options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], value=0),
        
        html.Label("SMS Received"),
        dcc.RadioItems(id='sms_received', options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], value=0),

        html.Label("Scheduled Hour"),
        dcc.Slider(id='scheduled_hour', min=0, max=23, step=1, value=9),
    ], style={'width': '50%', 'margin': 'auto'}),

    html.Br(),

    html.Div(id='prediction-result', style={'textAlign': 'center', 'fontSize': 24, 'fontWeight': 'bold'})
])

@app.callback(
    Output('prediction-result', 'children'),
    [Input('age', 'value'),
     Input('waiting_days', 'value'),
     Input('alcoholism', 'value'),
     Input('sms_received', 'value'),
     Input('scheduled_hour', 'value')]
)
def predict(age, waiting_days, alcoholism, sms_received, scheduled_hour):
    X = np.array([[age, waiting_days, alcoholism, sms_received, scheduled_hour]])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    result = "No-Show" if pred == 1 else "Show"
    return f"Prediction: {result} (Probability of No-Show: {proba:.2f})"

if __name__ == '__main__':
    app.run_server(debug=True)
