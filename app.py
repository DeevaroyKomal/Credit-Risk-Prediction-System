import streamlit as st
import pandas as pd
import pickle


# Load trained calibrated model

with open("credit_risk_model.pkl", "rb") as f:
    model = pickle.load(f)


# Streamlit page setup

st.set_page_config(page_title="Credit Risk Prediction", layout="centered")
st.title("💳 Credit Risk Prediction System")
st.write("Enter applicant details below:")


# User Inputs

person_age = st.number_input("Age", 18, 100, 30)
person_income = st.number_input("Annual Income", 10000, 1000000, 50000)
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_emp_length = st.number_input("Employment Length (years)", 0, 40, 5)
loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.number_input("Loan Amount", 500, 50000, 10000)
loan_int_rate = st.number_input("Interest Rate (%)", 1.0, 30.0, 12.0)
loan_percent_income = st.slider("Loan % of Income", 0.0, 1.0, 0.2)
cb_person_default_on_file = st.selectbox("Previous Default", ["Y", "N"])
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", 1, 30, 5)


# Feature Engineering

debt_to_income = loan_amnt / person_income
total_obligation_ratio = loan_percent_income + debt_to_income


# emp_length_group creation function

def create_emp_length_group(df):
    bins = [-1, 1, 3, 5, 10, 40]
    labels = ['<1', '1-3', '3-5', '5-10', '10+']
    df['emp_length_group'] = pd.cut(
        df['person_emp_length'],
        bins=bins,
        labels=labels
    )
    return df


# Create input DataFrame & apply emp_length_group

input_data = {
    "person_age": person_age,
    "person_income": person_income,
    "person_home_ownership": person_home_ownership,
    "person_emp_length": person_emp_length,
    "loan_intent": loan_intent,
    "loan_grade": loan_grade,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_default_on_file": cb_person_default_on_file,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "debt_to_income": debt_to_income,
    "total_obligation_ratio": total_obligation_ratio
}

input_df = pd.DataFrame([input_data])
input_df = create_emp_length_group(input_df)


# Prediction Button

if st.button("Predict Risk"):
    prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error(f"⚠️ High Risk Applicant (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk Applicant (Probability: {prob:.2f})")
