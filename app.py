# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:17:59 2025

@author: Mohan244643
"""

import streamlit as st
import pandas as pd
import joblib
import json


# Load the trained model, scaler, pca, and encoders
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")


# Load the dictionary of label encoders
encoders = joblib.load("encoders.pkl")


# Load the list of cities from the JSON file
with open('cities.json', 'r') as f:
    cities = json.load(f)
    
# Load the list of cities from the JSON file
with open('profession.json', 'r') as f:
    profession = json.load(f)
  
# Load the list of degrees from the JSON file
with open('degree.json', 'r') as f:
    degree = json.load(f)    


# Streamlit UI
st.title("Depression Prediction Model")

# Input fields for the features
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=25)
city = st.selectbox("City", cities)  # Replace with your actual cities
profession = st.selectbox("Profession", profession)  # Replace with your actual professions
academic_pressure = st.slider("Academic Pressure", min_value=0, max_value=5, step=1)
work_pressure = st.slider("Work Pressure", min_value=0, max_value=5, step=1)
cgpa = st.number_input("CGPA", min_value=0.00, max_value=5.00,step=0.01)
study_satisfaction = st.slider("Study Satisfaction", min_value=0, max_value=5, step=1)
job_satisfaction = st.slider("Job Satisfaction", min_value=0, max_value=5, step=1)
sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours","More than 8 hours","Others"])
dietary_habits = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy","Others"])
degree = st.selectbox("Degree", degree)
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
work_study_hours = st.slider("Work/Study Hours", min_value=0, max_value=12, step=1)
financial_stress = st.slider("Financial Stress", min_value=0, max_value=5, step=1)
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

# Convert categorical values to numeric using the loaded encoders
input_data = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "City": [city],
    "Profession": [profession],
    "Academic Pressure": [academic_pressure],
    "Work Pressure": [work_pressure],
    "CGPA": [cgpa],
    "Study Satisfaction": [study_satisfaction],
    "Job Satisfaction": [job_satisfaction],
    "Sleep Duration": [sleep_duration],
    "Dietary Habits": [dietary_habits],
    "Degree": [degree],
    "Have you ever had suicidal thoughts ?": [suicidal_thoughts],
    "Work/Study Hours": [work_study_hours],
    "Financial Stress": [financial_stress],
    "Family History of Mental Illness": [family_history],
})

# Encode the categorical columns using the corresponding label encoders
for col in encoders:
    input_data[col] = encoders[col].transform(input_data[col])

# Preprocess the input data
scaled_input_data = scaler.transform(input_data)  # Scale the input data
pca_input_data = pca.transform(scaled_input_data)  # Apply PCA transformation

# Make prediction
prediction = model.predict(pca_input_data)

# Display the result
if prediction == 1:
    st.write("Prediction: The individual is likely to have depression.")
else:
    st.write("Prediction: The individual is unlikely to have depression.")
