# gradio app for Diabetes Prediction

import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1. Load the Model
with open("Diabetes_rf_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Logic Function
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_df = pd.DataFrame([[
        Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    ]],
    columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])
    
    # Predict
    prediction = model.predict(input_df)[0]
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    return f"Predicted Outcome: {result}"

# 3. App Interface
inputs = [
    gr.Number(label="Pregnancies", value=0),
    gr.Number(label="Glucose", value=120),
    gr.Number(label="Blood Pressure", value=70),
    gr.Number(label="Skin Thickness", value=20),
    gr.Number(label="Insulin", value=79),
    gr.Number(label="BMI", value=25.0),
    gr.Number(label="Diabetes Pedigree Function", value=0.5),
    gr.Number(label="Age", value=33)
]

app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs="text",
    title="Diabetes Predictor",
    description="Enter your health parameters to predict if you are Diabetic or Non-Diabetic"
)
app.launch(share=True)