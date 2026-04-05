import streamlit as st
import joblib
import torch
import numpy as np
import os
import sys

# 🔥 FIX PATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.model import AttritionNN

st.title("💼 Employee Attrition Predictor")
st.markdown("### Enter Employee Details")

# ---------------------------
# LOAD MODELS
# ---------------------------
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

nb_model = joblib.load(os.path.join(MODEL_DIR, "nb_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# Load NN
input_size = joblib.load(os.path.join(MODEL_DIR, "input_size.pkl"))
nn_model = AttritionNN(input_size)
nn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "nn_model.pth")))
nn_model.eval()

# ---------------------------
# INPUTS (19 FEATURES)
# ---------------------------
age = st.number_input("Age", 18, 60, 30)
daily_rate = st.number_input("Daily Rate", 100, 1500, 800)
distance = st.number_input("Distance From Home", 1, 30, 5)
hourly_rate = st.number_input("Hourly Rate", 30, 100, 65)
monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
monthly_rate = st.number_input("Monthly Rate", 2000, 30000, 15000)
num_companies = st.number_input("Num Companies Worked", 0, 10, 2)
percent_salary_hike = st.number_input("Percent Salary Hike", 10, 25, 15)
total_working_years = st.number_input("Total Working Years", 0, 40, 8)
training_times = st.number_input("Training Times Last Year", 0, 10, 3)
years_at_company = st.number_input("Years At Company", 0, 40, 5)
years_in_role = st.number_input("Years In Current Role", 0, 20, 3)
years_since_promo = st.number_input("Years Since Last Promotion", 0, 15, 1)
years_with_manager = st.number_input("Years With Current Manager", 0, 20, 3)

gender = st.selectbox("Gender", ["Male", "Female"])
job_role = st.selectbox("Job Role", list(range(9)))
department = st.selectbox("Department", list(range(3)))
education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
overtime = st.checkbox("OverTime")

gender = 1 if gender == "Male" else 0
overtime = 1 if overtime else 0

# ---------------------------
# FEATURE ARRAY
# ---------------------------
features = np.array([[
    age, daily_rate, distance, hourly_rate, monthly_income, monthly_rate,
    num_companies, percent_salary_hike, total_working_years, training_times,
    years_at_company, years_in_role, years_since_promo, years_with_manager,
    gender, job_role, department, education, overtime
]])

features = scaler.transform(features)

# ---------------------------
# MODEL SELECT
# ---------------------------
model_type = st.selectbox("Select Model", ["Naive Bayes", "Neural Network"])

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("Predict"):
    if model_type == "Naive Bayes":
        pred = nb_model.predict(features)[0]
    else:
        with torch.no_grad():
            pred = nn_model(torch.FloatTensor(features))
            pred = (pred > 0.5).float().item()

    if pred == 1:
        st.error("⚠️ Employee is likely to leave")
    else:
        st.success("✅ Employee is likely to stay")