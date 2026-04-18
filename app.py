import gradio as gr
import pandas as pd
import pickle
# Loading the model
with open("Diabetes_rf_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_df = pd.DataFrame([[
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])
    #predict
    prediction = model.predict(input_df)[0]
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    return f"Predicted Outcome: {result}"

inputs = [
    gr.Number(label="Pregnancies"),
    gr.Number(label="Glucose"),
    gr.Number(label="Blood Pressure"),
    gr.Number(label="Skin Thickness"),
    gr.Number(label="Insulin"),
    gr.Number(label="BMI"),
    gr.Number(label="Diabetes Pedigree Function"),
    gr.Number(label="Age")
]
app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs="text",
    title="Diabetes Prediction App",
    description="Enter medical details to predict diabetes"
)
app.launch()
